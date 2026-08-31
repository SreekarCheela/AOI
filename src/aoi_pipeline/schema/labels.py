"""Labeling schema: whole-board pass/fail + defect-type label, as code.

This is the shared vocabulary between the public pretraining datasets
(SolDef_AI, PCB-AoI) and the labels derived from the existing AOI's verdicts
once real line data is being collected. Keeping it as one schema means a
model trained on public data and fine-tuned on line data sees a consistent
label space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Whole-board pass/fail call."""

    PASS = "pass"
    FAIL = "fail"


class DefectType(str, Enum):
    """Defect taxonomy covering SMT/solder-joint and bare-board AOI defects.

    Deliberately a superset of what SolDef_AI and PCB-AoI each label, so both
    public datasets and future line data (from the existing AOI's verdicts)
    normalize into the same space. Dataset-specific raw label strings are
    preserved on `BoardLabel.raw_label` for traceability — see
    `aoi_pipeline.datasets` for the per-dataset mapping into this enum.
    """

    NONE = "none"  # used only on PASS boards
    MISALIGNMENT = "misalignment"  # component shifted/rotated off its pad
    TOMBSTONE = "tombstone"  # component lifted/standing on one end
    LIFTED_LEAD = "lifted_lead"  # lead not making contact with the pad
    INSUFFICIENT_SOLDER = "insufficient_solder"
    EXCESS_SOLDER = "excess_solder"
    SOLDER_BRIDGE = "solder_bridge"  # unwanted short between adjacent pads
    COLD_JOINT = "cold_joint"
    MISSING_COMPONENT = "missing_component"
    WRONG_COMPONENT = "wrong_component"
    SHORT_CIRCUIT = "short_circuit"
    OPEN_CIRCUIT = "open_circuit"
    SCRATCH = "scratch"  # bare-board surface defect
    OTHER = "other"


class LabelSource(str, Enum):
    """Where a label came from — needed to weight/filter training data."""

    PUBLIC_DATASET = "public_dataset"
    EXISTING_AOI = "existing_aoi"  # correlated verdict from the AOI being replaced
    HUMAN_REVIEW = "human_review"


@dataclass
class BoardLabel:
    """Label for a single board/image."""

    board_id: str
    verdict: Verdict
    source: LabelSource
    defect_types: list[DefectType] = field(default_factory=list)
    raw_label: str | None = None
    confidence: float | None = None
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.verdict == Verdict.FAIL and not self.defect_types:
            raise ValueError("FAIL verdict requires at least one defect_type")
        if self.verdict == Verdict.PASS and any(
            d != DefectType.NONE for d in self.defect_types
        ):
            raise ValueError("PASS verdict must not carry non-NONE defect_types")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["source"] = self.source.value
        d["defect_types"] = [dt.value for dt in self.defect_types]
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoardLabel:
        return cls(
            board_id=data["board_id"],
            verdict=Verdict(data["verdict"]),
            source=LabelSource(data["source"]),
            defect_types=[DefectType(dt) for dt in data.get("defect_types", [])],
            raw_label=data.get("raw_label"),
            confidence=data.get("confidence"),
            notes=data.get("notes", ""),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
        )
