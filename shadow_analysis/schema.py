"""Core data schema shared by capture logging and labeling.

Recreated here (not imported from another worktree) per issue #6 scope: the
data pipeline (#4 in another worktree) and model training (#5, another
worktree) own the canonical versions. This module must stay
field-compatible with those so real logs can be loaded without translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class DefectType(str, Enum):
    NONE = "NONE"
    MISALIGNMENT = "MISALIGNMENT"
    TOMBSTONE = "TOMBSTONE"
    LIFTED_LEAD = "LIFTED_LEAD"
    INSUFFICIENT_SOLDER = "INSUFFICIENT_SOLDER"
    EXCESS_SOLDER = "EXCESS_SOLDER"
    SOLDER_BRIDGE = "SOLDER_BRIDGE"
    COLD_JOINT = "COLD_JOINT"
    MISSING_COMPONENT = "MISSING_COMPONENT"
    WRONG_COMPONENT = "WRONG_COMPONENT"
    SHORT_CIRCUIT = "SHORT_CIRCUIT"
    OPEN_CIRCUIT = "OPEN_CIRCUIT"
    SCRATCH = "SCRATCH"
    OTHER = "OTHER"


class LabelSource(str, Enum):
    PUBLIC_DATASET = "PUBLIC_DATASET"
    EXISTING_AOI = "EXISTING_AOI"
    HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass
class CaptureRecord:
    sequence_number: int
    timestamp: datetime
    image_path: str
    station_id: str = "unknown"
    board_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BoardLabel:
    board_id: str
    verdict: Verdict
    source: LabelSource
    defect_types: list[DefectType]
    raw_label: str | None = None
    confidence: float | None = None
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LabeledCapture:
    """A capture paired with the verdict assigned to that board.

    The comparison module correlates two streams of these — one from the
    existing AOI (ground truth) and one from the new model (under test).
    """

    capture: CaptureRecord
    label: BoardLabel
