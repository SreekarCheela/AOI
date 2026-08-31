"""The PCB-AoI Public Dataset (KubeEdge-Ianvs / China Telecom, Raisecom).

https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi

~1271 optical-inspection images (600x600) from the solder-paste-inspection
stage, spanning 230+ boards. Two defect categories are documented:
`Bad_qiaojiao` (component lead lifted / not seated — "qiaojiao" = tilted
foot) and `Bad_podu` (solder slope/profile defect). `raw_label_map` below is
seeded from public documentation of the dataset, not from having opened the
archive — `DatasetPreparer.inspect()` will log any folder names it doesn't
recognize (e.g. a "good" class, if the on-disk layout has one) so the map
can be corrected against the real download.
"""

from __future__ import annotations

from pathlib import Path

from aoi_pipeline.datasets.base import DatasetPreparer, DatasetSpec
from aoi_pipeline.schema.labels import DefectType

PCB_AOI_SPEC = DatasetSpec(
    name="pcb_aoi",
    kaggle_slug="kubeedgeianvs/pcb-aoi",
    expected_image_count=1200,
    license="see Kaggle dataset page",
    description=(
        "1271 solder-paste-inspection images across 230+ PCB boards, "
        "labeled Bad_qiaojiao (lifted lead) / Bad_podu (solder slope defect)."
    ),
    raw_label_map={
        "good": DefectType.NONE,
        "Bad_qiaojiao": DefectType.LIFTED_LEAD,
        "Bad_podu": DefectType.INSUFFICIENT_SOLDER,
    },
)


class PcbAoiPreparer(DatasetPreparer):
    def __init__(self, raw_dir: Path, processed_dir: Path) -> None:
        super().__init__(PCB_AOI_SPEC, raw_dir, processed_dir)
