"""End-to-end pipeline smoke test using synthetic data.

This does NOT produce the baseline checkpoint/eval report deliverable --
that requires the real public datasets, which this environment can't
download (no Kaggle credentials; see data/download_public_datasets.py).

What this validates instead: that datasets.py, augmentations.py, model.py,
train.py, and eval.py all actually run correctly together end-to-end --
manifest loading, image loading + transforms, the two-headed model
forward/backward pass, checkpointing, and the eval report generation
(including the agreement/escape/false-reject-rate metrics) -- against a
tiny generated dataset, so that once real data is available the exact same
commands (train.py / eval.py) are known to work rather than untested.

Usage:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_DIR))

from labels import DEFECT_CLASSES  # noqa: E402


def make_synthetic_dataset(out_dir: Path, n_images: int = 60, seed: int = 0) -> Path:
    rng = random.Random(seed)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_images):
        is_pass = rng.random() < 0.5
        # Encode "pass" as a mostly-green image and "fail" as mostly-red, so
        # the synthetic task is trivially learnable -- this is purely a
        # pipeline-plumbing check, not a model-quality check.
        color = (40, 180, 40) if is_pass else (200, 40, 40)
        img = Image.new("RGB", (64, 64), color=color)
        path = images_dir / f"board_{i:03d}.png"
        img.save(path)

        defect_flags = {c: 0 for c in DEFECT_CLASSES}
        if not is_pass:
            defect_flags[rng.choice(DEFECT_CLASSES)] = 1

        rows.append(
            {
                "image_path": str(path.relative_to(out_dir)),
                "pass_fail": int(is_pass),
                **defect_flags,
                "split": rng.choices(["train", "val", "test"], weights=[0.7, 0.15, 0.15])[0],
            }
        )

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "pass_fail", *DEFECT_CLASSES, "split"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    smoke_dir = MODEL_DIR / ".smoke_test_tmp"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir()

    manifest_path = make_synthetic_dataset(smoke_dir)
    print(f"Synthetic manifest: {manifest_path}")

    config_path = smoke_dir / "config.yaml"
    config_path.write_text(f"""
model:
  backbone: mobilenet_v3_small
  pretrained: true
  dropout: 0.2

data:
  manifest: {manifest_path}
  image_size: 64
  batch_size: 8
  num_workers: 0
  val_fraction: 0.15
  test_fraction: 0.15
  split_seed: 42

train:
  epochs: 2
  lr: 1.0e-3
  weight_decay: 1.0e-4
  optimizer: adamw
  pass_fail_loss_weight: 1.5
  defect_type_loss_weight: 1.0
  freeze_backbone_epochs: 1
  early_stopping_patience: 10
  checkpoint_dir: {smoke_dir / "checkpoints"}
  seed: 42
  device: cpu

eval:
  pass_fail_threshold: 0.5
  defect_type_threshold: 0.5
  report_dir: {smoke_dir / "reports"}
""")

    print("\n--- train.py ---")
    subprocess.run(
        [sys.executable, "train.py", "--config", str(config_path)],
        cwd=MODEL_DIR, check=True,
    )

    print("\n--- eval.py ---")
    subprocess.run(
        [
            sys.executable, "eval.py",
            "--checkpoint", str(smoke_dir / "checkpoints" / "best.pt"),
            "--config", str(config_path),
        ],
        cwd=MODEL_DIR, check=True,
    )

    report = (smoke_dir / "reports" / "baseline_eval_report.md").read_text()
    print("\n--- eval report ---")
    print(report)
    print(f"\nSmoke test passed. Artifacts left in {smoke_dir} for inspection "
          "(not committed -- see .gitignore).")


if __name__ == "__main__":
    main()
