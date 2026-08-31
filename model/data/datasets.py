"""Unified whole-board dataset, driven by a manifest CSV.

Every raw dataset (SolDef_AI, PCB-AoI, and eventually real line data) gets
normalized into the same manifest schema by a dataset-specific prep script
(see prepare_manifest.py), so the training/eval code never needs to know
which source a given image came from:

    image_path,pass_fail,insufficient_solder,excess_solder_bridging,tombstoning,cold_joint,misalignment,missing_component,split

- image_path: path to the image, relative to the manifest's own directory
- pass_fail: 1 = pass, 0 = fail
- one 0/1 column per class in labels.DEFECT_CLASSES (all 0 if pass_fail=1)
- split: "train" | "val" | "test" (test = held-out split for the eval report)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from labels import DEFECT_CLASSES

MANIFEST_FIELDS = ["image_path", "pass_fail", *DEFECT_CLASSES, "split"]


@dataclass
class BoardRecord:
    image_path: Path
    pass_fail: int
    defect_labels: list[int]


def read_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: {sorted(missing)}"
            )
        return list(reader)


class WholeBoardDataset(Dataset):
    """Loads images + pass/fail + multi-label defect targets from a manifest split."""

    def __init__(self, manifest_path: str | Path, split: str, transform=None):
        manifest_path = Path(manifest_path)
        self.root = manifest_path.parent
        self.transform = transform
        rows = read_manifest(manifest_path)
        self.records: list[BoardRecord] = [
            BoardRecord(
                image_path=self.root / row["image_path"],
                pass_fail=int(row["pass_fail"]),
                defect_labels=[int(row[c]) for c in DEFECT_CLASSES],
            )
            for row in rows
            if row["split"] == split
        ]
        if not self.records:
            raise ValueError(
                f"No rows found for split='{split}' in {manifest_path}. "
                "Did prepare_manifest.py run successfully?"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = {
            "pass_fail": torch.tensor(record.pass_fail, dtype=torch.float32),
            "defect_type": torch.tensor(record.defect_labels, dtype=torch.float32),
        }
        return image, target
