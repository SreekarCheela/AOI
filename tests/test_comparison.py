from datetime import timedelta

import pytest

from shadow_analysis.comparison import compare, correlate
from shadow_analysis.schema import DefectType, Verdict
from tests.helpers import BASE_TIME, make_labeled_capture


def test_perfect_agreement():
    gt = [
        make_labeled_capture(0, Verdict.PASS, board_id="B0"),
        make_labeled_capture(1, Verdict.PASS, board_id="B1"),
        make_labeled_capture(2, Verdict.FAIL, board_id="B2", defect_types=[DefectType.TOMBSTONE]),
        make_labeled_capture(3, Verdict.FAIL, board_id="B3", defect_types=[DefectType.SOLDER_BRIDGE]),
    ]
    model = [
        make_labeled_capture(0, Verdict.PASS, board_id="B0"),
        make_labeled_capture(1, Verdict.PASS, board_id="B1"),
        make_labeled_capture(2, Verdict.FAIL, board_id="B2", defect_types=[DefectType.TOMBSTONE]),
        make_labeled_capture(3, Verdict.FAIL, board_id="B3", defect_types=[DefectType.SOLDER_BRIDGE]),
    ]

    result = compare(gt, model)

    assert result.matched_count == 4
    assert result.agree_count == 4
    assert result.escape_count == 0
    assert result.false_reject_count == 0
    assert result.agreement_rate == 1.0
    assert result.escape_rate == 0.0
    assert result.false_reject_rate == 0.0


def test_all_escapes():
    gt = [
        make_labeled_capture(i, Verdict.FAIL, board_id=f"B{i}", defect_types=[DefectType.COLD_JOINT])
        for i in range(5)
    ]
    model = [make_labeled_capture(i, Verdict.PASS, board_id=f"B{i}") for i in range(5)]

    result = compare(gt, model)

    assert result.matched_count == 5
    assert result.agree_count == 0
    assert result.escape_count == 5
    assert result.false_reject_count == 0
    assert result.agreement_rate == 0.0
    assert result.escape_rate == 1.0
    assert result.false_reject_rate == 0.0

    assert len(result.defect_breakdown) == 1
    cold_joint = result.defect_breakdown[0]
    assert cold_joint.defect_type == DefectType.COLD_JOINT
    assert cold_joint.ground_truth_count == 5
    assert cold_joint.caught_count == 0
    assert cold_joint.escaped_count == 5
    assert cold_joint.catch_rate == 0.0


def test_all_false_rejects():
    gt = [make_labeled_capture(i, Verdict.PASS, board_id=f"B{i}") for i in range(5)]
    model = [
        make_labeled_capture(i, Verdict.FAIL, board_id=f"B{i}", defect_types=[DefectType.SCRATCH])
        for i in range(5)
    ]

    result = compare(gt, model)

    assert result.matched_count == 5
    assert result.agree_count == 0
    assert result.escape_count == 0
    assert result.false_reject_count == 5
    assert result.agreement_rate == 0.0
    assert result.escape_rate == 0.0
    assert result.false_reject_rate == 1.0
    # False rejects don't count toward defect breakdown — ground truth was PASS,
    # so there's no ground-truth defect type to attribute them to.
    assert result.defect_breakdown == []


def test_mixed_scenario_exact_counts():
    gt = []
    model = []

    # 4 boards: both PASS (agree)
    for i in range(4):
        gt.append(make_labeled_capture(i, Verdict.PASS, board_id=f"B{i}"))
        model.append(make_labeled_capture(i, Verdict.PASS, board_id=f"B{i}"))

    # 2 boards: both FAIL with MISALIGNMENT (agree, caught)
    for i in range(4, 6):
        gt.append(
            make_labeled_capture(i, Verdict.FAIL, board_id=f"B{i}", defect_types=[DefectType.MISALIGNMENT])
        )
        model.append(
            make_labeled_capture(i, Verdict.FAIL, board_id=f"B{i}", defect_types=[DefectType.MISALIGNMENT])
        )

    # 2 boards: gt FAIL (TOMBSTONE), model PASS (escape)
    for i in range(6, 8):
        gt.append(
            make_labeled_capture(i, Verdict.FAIL, board_id=f"B{i}", defect_types=[DefectType.TOMBSTONE])
        )
        model.append(make_labeled_capture(i, Verdict.PASS, board_id=f"B{i}"))

    # 2 boards: gt PASS, model FAIL (false reject)
    for i in range(8, 10):
        gt.append(make_labeled_capture(i, Verdict.PASS, board_id=f"B{i}"))
        model.append(
            make_labeled_capture(i, Verdict.FAIL, board_id=f"B{i}", defect_types=[DefectType.OTHER])
        )

    result = compare(gt, model)

    assert result.matched_count == 10
    assert result.agree_count == 6
    assert result.escape_count == 2
    assert result.false_reject_count == 2
    assert result.agreement_rate == pytest.approx(0.6)
    assert result.escape_rate == pytest.approx(0.2)
    assert result.false_reject_rate == pytest.approx(0.2)

    breakdown = {s.defect_type: s for s in result.defect_breakdown}
    assert breakdown[DefectType.MISALIGNMENT].ground_truth_count == 2
    assert breakdown[DefectType.MISALIGNMENT].caught_count == 2
    assert breakdown[DefectType.MISALIGNMENT].catch_rate == 1.0
    assert breakdown[DefectType.TOMBSTONE].ground_truth_count == 2
    assert breakdown[DefectType.TOMBSTONE].escaped_count == 2
    assert breakdown[DefectType.TOMBSTONE].catch_rate == 0.0


def test_correlate_by_board_id():
    gt = [make_labeled_capture(0, Verdict.PASS, board_id="B0")]
    # Different sequence numbers/timestamps, same board_id -> should still match.
    model = [
        make_labeled_capture(
            99, Verdict.PASS, board_id="B0", timestamp=BASE_TIME + timedelta(hours=3)
        )
    ]

    result = correlate(gt, model)

    assert len(result.pairs) == 1
    assert result.pairs[0].method == "board_id"
    assert result.unmatched_ground_truth == []
    assert result.unmatched_model == []


def test_correlate_by_sequence_and_timestamp_fallback():
    # No board_id on either side -> falls back to sequence_number + timestamp.
    gt = [make_labeled_capture(5, Verdict.PASS, board_id=None)]
    model = [
        make_labeled_capture(
            5, Verdict.PASS, board_id=None, timestamp=BASE_TIME + timedelta(seconds=5, milliseconds=500)
        )
    ]

    result = correlate(gt, model, timestamp_tolerance=timedelta(seconds=2))

    assert len(result.pairs) == 1
    assert result.pairs[0].method == "sequence_timestamp"


def test_correlate_sequence_fallback_rejects_out_of_tolerance_timestamp():
    gt = [make_labeled_capture(5, Verdict.PASS, board_id=None)]
    model = [
        make_labeled_capture(5, Verdict.PASS, board_id=None, timestamp=BASE_TIME + timedelta(minutes=10))
    ]

    result = correlate(gt, model, timestamp_tolerance=timedelta(seconds=2))

    assert result.pairs == []
    assert len(result.unmatched_ground_truth) == 1
    assert len(result.unmatched_model) == 1


def test_correlate_unmatched_records_on_both_sides():
    gt = [
        make_labeled_capture(0, Verdict.PASS, board_id="B0"),
        make_labeled_capture(1, Verdict.FAIL, board_id="B1", defect_types=[DefectType.OTHER]),
    ]
    model = [
        make_labeled_capture(0, Verdict.PASS, board_id="B0"),
        make_labeled_capture(2, Verdict.PASS, board_id="B2"),
    ]

    result = correlate(gt, model)

    assert len(result.pairs) == 1
    assert [lc.label.board_id for lc in result.unmatched_ground_truth] == ["B1"]
    assert [lc.label.board_id for lc in result.unmatched_model] == ["B2"]


def test_compare_raises_on_no_matches():
    gt = [make_labeled_capture(0, Verdict.PASS, board_id="B0")]
    model = [make_labeled_capture(0, Verdict.PASS, board_id="different")]

    with pytest.raises(ValueError):
        compare(gt, model)
