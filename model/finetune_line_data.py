"""STUB — fine-tuning on real production-line data.

Explicitly out of scope for this run: there is no ESP32 rig, camera, or real
board image yet (that's issues #2/#3 in the epic), and no AOI-verdict
correlation log to use as ground truth (issue #4, the data pipeline). This
file exists to pin down the interface the real implementation will need to
satisfy, so `train.py` / `eval.py` don't need to change shape later — only
this file needs to grow a real body.

TODO (blocked on issue #4 — data pipeline / AOI-verdict correlation logging):
  1. `line_data_manifest_path`: a manifest in the same schema as
     data/datasets.py's WholeBoardDataset (image_path, pass_fail, one column
     per labels.DEFECT_CLASSES, split), except:
       - pass_fail and defect labels come from the EXISTING AOI's verdict at
         inspection time, not human annotation (see epic issue #1: "this
         makes the new model's accuracy ceiling equal to the existing AOI's").
       - images come from the ESP32 rig's captures, correlated to the
         existing AOI's verdict by sequence/timestamp (or barcode/serial if
         issue #4's pre-flight check confirms one exists).
  2. Load the checkpoint produced by train.py on the public-dataset
     pretraining split (checkpoints/best.pt) as the starting point -- this is
     fine-tuning, not training from scratch.
  3. Fine-tune at a lower LR than the pretraining run (typically ~10x lower)
     since the backbone already has useful features; likely few epochs given
     real line data volume will be much smaller than a full pretraining set.
  4. Re-run eval.py-equivalent logic against a held-out split of the LINE
     data (not the public dataset's held-out split) to check the epic's
     actual acceptance bar: >=98% agreement, <=0.5% escape rate, <=3% false
     reject rate, all against the existing AOI's verdicts.
  5. Only after that held-out check passes does the epic move to the live
     shadow-mode run (issue #6).

Do not implement against synthetic/placeholder data -- the whole point of
this step is grading against the real AOI's real verdicts. A placeholder
implementation here would produce a checkpoint that looks fine-tuned but
proves nothing about the actual acceptance criteria.
"""
from __future__ import annotations

from pathlib import Path


def finetune_on_line_data(
    pretrained_checkpoint: Path,
    line_data_manifest: Path,
    output_checkpoint: Path,
) -> None:
    raise NotImplementedError(
        "Fine-tuning on real line data is blocked on the data pipeline "
        "(epic issue #4: ESP32 rig images + AOI-verdict correlation "
        "logging). See module docstring for the interface this needs to "
        "satisfy once that data exists."
    )


if __name__ == "__main__":
    raise SystemExit(
        "finetune_line_data.py is a stub — see the module docstring for "
        "what's blocking implementation."
    )
