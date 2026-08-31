# Model Architecture — Whole-Board Solder-Joint Classification

Part of [issue #5](../..) (child of the AOI-replacement epic, issue #1). Covers the
architecture-selection portion of that issue's scope; training pipeline and initial
public-dataset pretraining are covered in the rest of this directory.

## Task framing

MVP scope (per the epic) is **whole-board classification**, not per-joint localization:
given one image of an assembled board, predict:

1. **Pass / fail** (binary)
2. **Defect type(s)** present, matching the existing AOI's taxonomy: insufficient solder,
   excess solder / bridging, tombstoning, cold/dull joint, misalignment, missing component.

A board can plausibly exhibit more than one defect type simultaneously (e.g. a missing
component *and* a cold joint elsewhere on the same board), so the defect-type head is
modeled as **multi-label** (independent sigmoid per class), not single-label softmax.
Pass/fail is kept as its own head rather than derived from "any defect head fired" —
during shadow mode we're graded against the *existing AOI's pass/fail verdict*
specifically (agreement/escape/false-reject rate), so that head should be free to learn
directly from that ground truth signal even if it doesn't perfectly correspond to the
defect-type labels available in the public pretraining data.

## Candidate architectures considered

| Candidate | Verdict | Why |
|---|---|---|
| **MobileNetV3 (Small/Large)** | **Selected** | NAS-designed for mobile/edge latency-accuracy tradeoff; hard-swish + squeeze-excite blocks are supported by TensorRT INT8 on Jetson (SE is not a problem outside of edge-TPU-specific toolchains); ships in torchvision with ImageNet-pretrained weights, so transfer learning and ONNX→TensorRT export both work with no extra dependencies; native `torch.ao.quantization` QAT/PTQ support. |
| EfficientNet-Lite0/1 | Rejected (for now) | Built by Google specifically to strip SE blocks and swish for edge-TPU quantization — solves a constraint we don't have (we're targeting Jetson/TensorRT, not edge TPU). No pretrained weights in torchvision; pulling them in means adding `timm` as a dependency for marginal benefit here. Worth revisiting if MobileNetV3 doesn't hit the latency budget after quantization. |
| Full EfficientNet (B0+) / ResNet-50 | Rejected | Meaningfully larger (compute and params) for accuracy gains that don't matter much on a defect-classification task with a handful of classes; heavier INT8 latency on Orin Nano-class compute; no reason to pay the cost here. |
| ViT / hybrid transformer | Rejected | Data-hungry (public pretraining sets here are ~1-2k images each, real line data will be similarly modest), worse quantization story on Jetson today, and per-joint texture/local-defect detection is exactly the inductive bias CNNs are good at and ViTs have to learn from scratch. |
| Custom small CNN from scratch | Rejected | No transfer-learning benefit from ImageNet features (edge/texture detectors transfer fine to solder-joint imagery); reinventing a worse version of MobileNetV3 for no latency win. |

## Selected architecture

**Backbone:** `torchvision.models.mobilenet_v3_large`, initialized from ImageNet-1k
pretrained weights (`MobileNet_V3_Large_Weights.IMAGENET1K_V2`).

- ~5.4M params, ~219M MAdds at 224×224 — comfortably within Jetson Orin Nano-class
  (20 TOPS INT8 on the 8GB module) real-time budget for a per-board classification task
  that only needs to keep up with board arrival rate, not video framerate.
- `mobilenet_v3_small` is wired in as a config toggle (`model.backbone: mobilenet_v3_small`)
  for a ~2.5M-param fallback if the Large variant doesn't hit the latency budget once
  quantized and benchmarked on real hardware — that benchmarking itself is out of scope
  here (no Jetson device or real board images yet); the toggle just avoids a rewrite later.
- Both variants use hard-swish and squeeze-excite, both are QAT-friendly, both quantize
  cleanly to INT8 via `torch.ao.quantization` and export via ONNX → TensorRT.

**Heads:** the backbone's classifier is replaced with two independent linear heads off
the same pooled feature vector:

- `pass_fail_head`: `Linear(in_features, 1)` — trained with `BCEWithLogitsLoss`.
- `defect_type_head`: `Linear(in_features, 6)` — one sigmoid logit per defect class,
  trained with `BCEWithLogitsLoss` (multi-label, not `CrossEntropyLoss`).

Total loss is a weighted sum (`pass_fail_loss_weight` / `defect_type_loss_weight` in
config) since the epic's acceptance criteria are graded on pass/fail agreement/escape/
false-reject rate — that head's loss is weighted higher by default.

**Input:** 224×224 RGB, ImageNet normalization (matches pretrained backbone stats).
Board images will need resizing/tiling from whatever the eventual camera resolution is;
that's the data pipeline's concern (issue #4), not this issue's.

**Quantization path (documented, not yet executed — no edge device to benchmark against):**
1. Train in FP32 with transfer learning (this issue).
2. Post-training static INT8 quantization via `torch.ao.quantization` using a calibration
   pass over held-out data, OR quantization-aware fine-tuning if PTQ accuracy drop is
   unacceptable — both are supported by the MobileNetV3 architecture out of the box.
3. Export to ONNX, compile with TensorRT on the actual Jetson Orin Nano-class device.
4. Benchmark latency against board arrival rate. **Explicitly deferred** — needs real
   hardware, tracked as part of the fine-tuning/deployment step stubbed out in
   `finetune_line_data.py`.

## Why not decide this later / why now

The acceptance criteria on this issue require a deployable checkpoint sized and shaped
for the target device, so backbone choice has to be locked in before pretraining rather
than after — swapping backbones after transfer-learning would throw away the pretraining
run's compute. MobileNetV3 was chosen over the alternatives above specifically because it
needs no dataset beyond what's already available (ImageNet pretrained weights ship with
torchvision) and no extra dependency to start pretraining today.
