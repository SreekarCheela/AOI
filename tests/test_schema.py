from datetime import datetime, timezone

import pytest

from aoi_pipeline.schema.capture import CaptureRecord
from aoi_pipeline.schema.labels import BoardLabel, DefectType, LabelSource, Verdict


def test_board_label_fail_requires_defect_type():
    with pytest.raises(ValueError):
        BoardLabel(
            board_id="b1", verdict=Verdict.FAIL, source=LabelSource.HUMAN_REVIEW, defect_types=[]
        )


def test_board_label_pass_rejects_real_defect_type():
    with pytest.raises(ValueError):
        BoardLabel(
            board_id="b1",
            verdict=Verdict.PASS,
            source=LabelSource.HUMAN_REVIEW,
            defect_types=[DefectType.TOMBSTONE],
        )


def test_board_label_roundtrip():
    label = BoardLabel(
        board_id="b1",
        verdict=Verdict.FAIL,
        source=LabelSource.PUBLIC_DATASET,
        defect_types=[DefectType.INSUFFICIENT_SOLDER],
        raw_label="insufficient",
        confidence=0.9,
    )
    restored = BoardLabel.from_dict(label.to_dict())
    assert restored == label


def test_capture_record_roundtrip():
    record = CaptureRecord(
        sequence_number=42,
        timestamp=datetime.now(timezone.utc),
        image_path="/data/raw/img_0042.png",
        station_id="station-01",
        board_id=None,
        metadata={"trigger": "photoeye"},
    )
    restored = CaptureRecord.from_dict(record.to_dict())
    assert restored == record
