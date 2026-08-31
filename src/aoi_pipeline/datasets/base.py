"""Shared download/inspect/prep pipeline for the public pretraining datasets.

Both SolDef_AI and PCB-AoI are image-classification-style Kaggle datasets
where the defect label is encoded as the parent folder name of each image
(the common Kaggle layout: `<class_name>/<image>.png`). This module
implements that generically; concrete subclasses only supply the dataset's
Kaggle handle and the raw-folder-name -> DefectType mapping.

Actually downloading requires Kaggle credentials (~/.kaggle/kaggle.json or
KAGGLE_USERNAME/KAGGLE_KEY env vars) on the machine that runs this — see
docs/datasets.md. `inspect()` is the step that confirms acceptance
criterion #1 (image count, label format) and should be re-run once real
credentials are available, since the raw-label maps below are seeded from
public documentation, not from actually opening the archives.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from aoi_pipeline.schema.labels import BoardLabel, DefectType, LabelSource, Verdict

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class DatasetSpec:
    name: str
    kaggle_slug: str
    expected_image_count: int
    license: str
    description: str
    raw_label_map: dict[str, DefectType] = field(default_factory=dict)


@dataclass
class DatasetInspectionReport:
    name: str
    image_count: int
    raw_label_values: list[str]
    unmapped_raw_labels: list[str]
    sample_paths: list[str]

    def is_usable(self, spec: DatasetSpec, tolerance: float = 0.1) -> bool:
        """Loose sanity check against the expected image count from the issue
        (SolDef_AI: 1150, PCB-AoI: 1200+) — exact counts can drift by dataset
        version, so this is a tolerance band, not an exact match."""
        if self.image_count == 0:
            return False
        lower = spec.expected_image_count * (1 - tolerance)
        return self.image_count >= lower


def _find_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


class DatasetPreparer:
    """Downloads, inspects, and normalizes one public dataset into our
    labeling schema. Subclasses just supply a `DatasetSpec`."""

    def __init__(self, spec: DatasetSpec, raw_dir: Path, processed_dir: Path) -> None:
        self.spec = spec
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    def raw_path(self) -> Path:
        return self.raw_dir / self.spec.name

    def download(self, force: bool = False) -> Path:
        dest = self.raw_path()
        if dest.exists() and not force:
            logger.info("%s already downloaded at %s (use force=True to re-fetch)", self.spec.name, dest)
            return dest

        try:
            import kagglehub
        except ImportError as e:
            raise RuntimeError(
                "kagglehub is not installed. Run `pip install kagglehub` "
                "(or `pip install -e '.[dev]'` from the repo root)."
            ) from e

        try:
            cache_path = Path(kagglehub.dataset_download(self.spec.kaggle_slug))
        except Exception as e:  # kagglehub raises assorted exception types
            raise RuntimeError(
                f"Failed to download '{self.spec.kaggle_slug}' via kagglehub. "
                "Confirm Kaggle credentials are configured "
                "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars) "
                "and that you've accepted the dataset's terms at "
                f"https://www.kaggle.com/datasets/{self.spec.kaggle_slug}"
            ) from e

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(cache_path, dest)
        return dest

    def inspect(self, raw_path: Path | None = None) -> DatasetInspectionReport:
        raw_path = raw_path or self.raw_path()
        images = _find_images(raw_path)
        raw_labels = sorted({p.parent.name for p in images})
        unmapped = [label for label in raw_labels if label not in self.spec.raw_label_map]
        if unmapped:
            logger.warning(
                "%s: %d raw label(s) not in raw_label_map: %s — extend the map "
                "in aoi_pipeline.datasets before relying on prep() output for these",
                self.spec.name,
                len(unmapped),
                unmapped,
            )
        return DatasetInspectionReport(
            name=self.spec.name,
            image_count=len(images),
            raw_label_values=raw_labels,
            unmapped_raw_labels=unmapped,
            sample_paths=[str(p) for p in images[:5]],
        )

    def prep(self, raw_path: Path | None = None) -> Path:
        """Writes a JSONL manifest (one BoardLabel per image, plus
        image_path) into processed_dir/<name>/manifest.jsonl."""
        raw_path = raw_path or self.raw_path()
        images = _find_images(raw_path)
        out_dir = self.processed_dir / self.spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.jsonl"

        with open(manifest_path, "w") as f:
            for image_path in images:
                label = self.label_for(image_path)
                record = {"image_path": str(image_path), **label.to_dict()}
                f.write(json.dumps(record) + "\n")

        logger.info("%s: wrote %d records to %s", self.spec.name, len(images), manifest_path)
        return manifest_path

    def label_for(self, image_path: Path) -> BoardLabel:
        """Derives a BoardLabel from an image's parent folder name, using
        `spec.raw_label_map`. Unmapped folder names fall back to
        DefectType.OTHER / FAIL rather than raising, so prep() can still run
        end-to-end on a partially-mapped dataset (see inspect() warnings)."""
        raw_label = image_path.parent.name
        defect_type = self.spec.raw_label_map.get(raw_label, DefectType.OTHER)
        verdict = Verdict.PASS if defect_type == DefectType.NONE else Verdict.FAIL
        return BoardLabel(
            board_id=image_path.stem,
            verdict=verdict,
            source=LabelSource.PUBLIC_DATASET,
            defect_types=[] if verdict == Verdict.PASS else [defect_type],
            raw_label=raw_label,
        )
