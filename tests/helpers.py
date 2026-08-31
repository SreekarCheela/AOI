from datetime import datetime, timedelta

from shadow_analysis.schema import (
    BoardLabel,
    CaptureRecord,
    DefectType,
    LabeledCapture,
    LabelSource,
    Verdict,
)

BASE_TIME = datetime(2026, 1, 1, 8, 0, 0)


def make_labeled_capture(
    seq: int,
    verdict: Verdict,
    *,
    board_id: str | None = None,
    defect_types: list[DefectType] | None = None,
    source: LabelSource = LabelSource.EXISTING_AOI,
    timestamp: datetime | None = None,
    station_id: str = "station-1",
) -> LabeledCapture:
    ts = timestamp if timestamp is not None else BASE_TIME + timedelta(seconds=seq)
    defects = defect_types if defect_types is not None else (
        [DefectType.NONE] if verdict == Verdict.PASS else [DefectType.OTHER]
    )
    capture = CaptureRecord(
        sequence_number=seq,
        timestamp=ts,
        image_path=f"/data/captures/{seq:06d}.png",
        station_id=station_id,
        board_id=board_id,
    )
    label = BoardLabel(
        board_id=board_id or f"seq-{seq}",
        verdict=verdict,
        source=source,
        defect_types=defects,
    )
    return LabeledCapture(capture=capture, label=label)
