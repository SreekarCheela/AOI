"""Generate realistic-looking synthetic label streams for testing.

No real line data exists yet (shadow mode is blocked on hardware install,
issues #2/#3). These helpers build plausible CaptureRecord/BoardLabel pairs
so the comparison math can be exercised against known-shape scenarios
before any real logs exist.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from shadow_analysis.schema import (
    BoardLabel,
    CaptureRecord,
    DefectType,
    LabeledCapture,
    LabelSource,
    Verdict,
)

DEFAULT_DEFECT_POOL = [
    DefectType.MISALIGNMENT,
    DefectType.TOMBSTONE,
    DefectType.INSUFFICIENT_SOLDER,
    DefectType.SOLDER_BRIDGE,
    DefectType.MISSING_COMPONENT,
]


def _make_pair(
    index: int,
    base_time: datetime,
    verdict: Verdict,
    defect_types: list[DefectType],
    use_board_id: bool,
    station_id: str,
    source: LabelSource,
) -> LabeledCapture:
    timestamp = base_time + timedelta(seconds=index * 5)
    board_id = f"BOARD-{index:06d}" if use_board_id else None
    capture = CaptureRecord(
        sequence_number=index,
        timestamp=timestamp,
        image_path=f"/data/captures/{station_id}/{index:06d}.png",
        station_id=station_id,
        board_id=board_id,
    )
    label = BoardLabel(
        board_id=board_id or f"BOARD-{index:06d}",
        verdict=verdict,
        source=source,
        defect_types=defect_types,
    )
    return LabeledCapture(capture=capture, label=label)


def generate_matched_run(
    n_boards: int,
    *,
    fail_rate: float = 0.05,
    escape_rate: float = 0.0,
    false_reject_rate: float = 0.0,
    use_board_id: bool = True,
    seed: int = 0,
    base_time: datetime | None = None,
) -> tuple[list[LabeledCapture], list[LabeledCapture]]:
    """Build a ground-truth stream and a model stream with controlled error rates.

    fail_rate: fraction of boards the existing AOI (ground truth) fails.
    escape_rate: fraction of ground-truth FAIL boards the model wrongly passes.
    false_reject_rate: fraction of ground-truth PASS boards the model wrongly fails.

    Every board is captured by both streams (fully correlated), so this is
    useful for exercising the comparison math in isolation from correlation
    edge cases.
    """
    rng = random.Random(seed)
    base_time = base_time or datetime(2026, 1, 1, 8, 0, 0)

    ground_truth: list[LabeledCapture] = []
    model_output: list[LabeledCapture] = []

    for i in range(n_boards):
        gt_verdict = Verdict.FAIL if rng.random() < fail_rate else Verdict.PASS
        gt_defects = [rng.choice(DEFAULT_DEFECT_POOL)] if gt_verdict == Verdict.FAIL else [DefectType.NONE]

        if gt_verdict == Verdict.FAIL:
            model_verdict = Verdict.PASS if rng.random() < escape_rate else Verdict.FAIL
        else:
            model_verdict = Verdict.FAIL if rng.random() < false_reject_rate else Verdict.PASS
        model_defects = gt_defects if model_verdict == Verdict.FAIL else [DefectType.NONE]

        gt_lc = _make_pair(
            i, base_time, gt_verdict, gt_defects, use_board_id, "existing-aoi", LabelSource.EXISTING_AOI
        )
        # LabelSource has no MODEL_INFERENCE member (schema is fixed/shared with the
        # labeling pipeline, not owned by this module) — HUMAN_REVIEW is the closest
        # stand-in for "not the existing AOI's own verdict". comparison.py never reads
        # `source`, so this only affects synthetic fixtures, not the comparison math.
        model_lc = _make_pair(
            i, base_time, model_verdict, model_defects, use_board_id, "new-system", LabelSource.HUMAN_REVIEW
        )
        ground_truth.append(gt_lc)
        model_output.append(model_lc)

    return ground_truth, model_output
