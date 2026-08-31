# Whole-Board Solder-Joint Classification — Model Training

Covers issue #5 (child of the AOI-replacement epic, issue #1), scoped to the
portion of that issue that doesn't require real production-line data:

1. Architecture selection — see [ARCHITECTURE.md](ARCHITECTURE.md).
2. Training pipeline scaffold (this directory).
3. Initial transfer-learning pretraining on public datasets (SolDef_AI, PCB-AoI).
4. Baseline checkpoint + eval report on a held-out split of that public data.

Fine-tuning on real line data is explicitly stubbed out — see
[finetune_line_data.py](finetune_line_data.py) — since there's no ESP32 rig,
camera, or real board image yet (that's epic issues #2/#3), and no
AOI-verdict correlation log to fine-tune against (epic issue #4).

## Status / known blocker

**Steps 1-2 are done.** Step 3-4 (actually running pretraining on the public
datasets and producing a real baseline checkpoint) is blocked in this
environment: both datasets are Kaggle-hosted and Kaggle requires
authenticated API access to download, and no Kaggle credentials are
available here (checked env vars, `~/.kaggle/`, and the sibling
`data-pipeline` worktree — none had them). `data/download_public_datasets.py`
fails fast with this explanation rather than hanging on Kaggle's interactive
OAuth flow.

Everything else has been validated end-to-end against synthetic data (see
`scripts/smoke_test.py` and "What's actually been run" below) so that once
credentials are available, running the real pipeline is just:

```bash
python data/download_public_datasets.py --out data/raw
python data/prepare_manifest.py --inspect --raw-dir data/raw   # sanity-check class folder names first
python data/prepare_manifest.py --raw-dir data/raw --out data/manifest.csv
python train.py --config configs/default.yaml
python eval.py --checkpoint checkpoints/best.pt --config configs/default.yaml
```

To unblock: provide Kaggle credentials (`KAGGLE_API_TOKEN` env var, or
`~/.kaggle/kaggle.json`) and re-run the commands above.

## Layout

```
model/
  ARCHITECTURE.md            backbone selection + justification
  labels.py                  shared defect taxonomy (matches existing AOI)
  model.py                   MobileNetV3 backbone + pass/fail + defect-type heads
  data/
    augmentations.py         train/eval transforms
    datasets.py               manifest-driven Dataset
    download_public_datasets.py   Kaggle download (blocked, see above)
    prepare_manifest.py       raw dataset -> unified manifest.csv
  train.py                    two-phase transfer-learning loop
  eval.py                     held-out eval -> markdown + json report
  metrics.py                  agreement / escape / false-reject rate + defect-type metrics
  finetune_line_data.py       STUB -- real-line-data fine-tuning, blocked on epic issue #4
  scripts/smoke_test.py       synthetic end-to-end pipeline validation (not real training)
  configs/default.yaml
```

## What's actually been run

`scripts/smoke_test.py` generates a tiny synthetic image dataset (trivially
separable by color, purely for pipeline plumbing, not a model-quality
signal), then runs the real `train.py` and `eval.py` entrypoints against it
unmodified. This confirms manifest loading, image transforms, the two-headed
model's forward/backward pass, checkpointing, and eval-report generation
(including the agreement/escape/false-reject-rate metrics) all work
end-to-end. It does **not** stand in for the real baseline checkpoint/eval
report deliverable -- that requires the actual public datasets.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
