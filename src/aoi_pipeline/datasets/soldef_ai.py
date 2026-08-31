"""SolDef_AI: An Open Source PCB Dataset for Mask R-CNN Defect Detection in
Soldering Processes of Electronic Components.

https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection

1150 images of SMT solder joints (0603/0805/1206 resistors, 0805
capacitors), each component shot from 3 viewpoints. Two defect families per
the dataset's paper: component placement defects (misalignment) and solder
joint defects (excess/insufficient solder material). `raw_label_map` below
is seeded from the published paper/Kaggle description, not from having
opened the archive — `DatasetPreparer.inspect()` will log any folder names
it doesn't recognize so the map can be corrected against the real download.
"""

from __future__ import annotations

from pathlib import Path

from aoi_pipeline.datasets.base import DatasetPreparer, DatasetSpec
from aoi_pipeline.schema.labels import DefectType

SOLDEF_AI_SPEC = DatasetSpec(
    name="soldef_ai",
    kaggle_slug="mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
    expected_image_count=1150,
    license="see Kaggle dataset page",
    description=(
        "1150 SMT solder-joint images (resistors/capacitors), 3 viewpoints "
        "per component, labeled for placement and solder-joint defects."
    ),
    raw_label_map={
        "good": DefectType.NONE,
        "ok": DefectType.NONE,
        "misalignment": DefectType.MISALIGNMENT,
        "insufficient_solder": DefectType.INSUFFICIENT_SOLDER,
        "excess_solder": DefectType.EXCESS_SOLDER,
    },
)


class SolDefAIPreparer(DatasetPreparer):
    def __init__(self, raw_dir: Path, processed_dir: Path) -> None:
        super().__init__(SOLDEF_AI_SPEC, raw_dir, processed_dir)
