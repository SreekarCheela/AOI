from datetime import datetime, timedelta, timezone

from aoi_pipeline.correlation.matcher import CorrelationMatcher
from aoi_pipeline.correlation.strategies import (
    CompositeMatchStrategy,
    ExistingAoiVerdict,
    IdMatchStrategy,
    SequenceTimestampMatchStrategy,
)
from aoi_pipeline.schema.capture import CaptureRecord
from aoi_pipeline.schema.labels import DefectType, Verdict

BASE = datetime(2026, 8, 31, 12, 0, 0, tzinfo=timezone.utc)


def _capture(seq: int, offset_seconds: float, board_id: str | None = None) -> CaptureRecord:
    return CaptureRecord(
        sequence_number=seq,
        timestamp=BASE + timedelta(seconds=offset_seconds),
        image_path=f"/data/raw/img_{seq}.png",
        board_id=board_id,
    )


def _verdict(
    vid: str, offset_seconds: float, verdict: Verdict = Verdict.PASS, board_id: str | None = None
) -> ExistingAoiVerdict:
    return ExistingAoiVerdict(
        verdict_id=vid,
        timestamp=BASE + timedelta(seconds=offset_seconds),
        verdict=verdict,
        defect_types=[] if verdict == Verdict.PASS else [DefectType.TOMBSTONE],
        board_id=board_id,
    )


def test_sequence_timestamp_matches_within_tolerance():
    captures = [_capture(0, 0.0), _capture(1, 10.0)]
    verdicts = [_verdict("v0", 0.5), _verdict("v1", 10.4)]
    strategy = SequenceTimestampMatchStrategy(tolerance=timedelta(seconds=2))
    matches, unmatched_c, unmatched_v = strategy.match(captures, verdicts)
    assert len(matches) == 2
    assert not unmatched_c and not unmatched_v


def test_sequence_timestamp_leaves_out_of_tolerance_unmatched():
    captures = [_capture(0, 0.0)]
    verdicts = [_verdict("v0", 5.0)]
    strategy = SequenceTimestampMatchStrategy(tolerance=timedelta(seconds=2))
    matches, unmatched_c, unmatched_v = strategy.match(captures, verdicts)
    assert not matches
    assert len(unmatched_c) == 1
    assert len(unmatched_v) == 1


def test_id_match_strategy_matches_on_board_id_regardless_of_time():
    captures = [_capture(0, 0.0, board_id="BOARD-1")]
    verdicts = [_verdict("v0", 999.0, board_id="BOARD-1")]
    strategy = IdMatchStrategy()
    matches, unmatched_c, unmatched_v = strategy.match(captures, verdicts)
    assert len(matches) == 1
    assert matches[0].method == "id"


def test_composite_prefers_id_then_falls_back_to_sequence_timestamp():
    captures = [
        _capture(0, 0.0, board_id="BOARD-1"),
        _capture(1, 10.0, board_id=None),
    ]
    verdicts = [
        _verdict("v0", 999.0, board_id="BOARD-1"),
        _verdict("v1", 10.3, board_id=None),
    ]
    matcher = CorrelationMatcher(strategy=CompositeMatchStrategy())
    report = matcher.run(captures, verdicts)
    assert report.match_rate == 1.0
    methods = {m.method for m in report.matches}
    assert methods == {"id", "sequence_timestamp"}


def test_match_rate_meets_acceptance_threshold_on_mostly_clean_run():
    # 100 captures, 96 with a verdict within tolerance, 4 with no corresponding
    # verdict at all (e.g. existing AOI dropped a frame) -> should clear the
    # >=95% match-rate acceptance bar from issue #4.
    captures = [_capture(i, i * 5.0) for i in range(100)]
    verdicts = [_verdict(f"v{i}", i * 5.0 + 0.2) for i in range(96)]
    matcher = CorrelationMatcher()
    report = matcher.run(captures, verdicts)
    assert report.match_rate >= 0.95
    assert len(report.unmatched_captures) == 4
