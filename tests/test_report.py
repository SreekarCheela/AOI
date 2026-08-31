from shadow_analysis.comparison import compare
from shadow_analysis.report import (
    PER_JOINT_LOCALIZATION_FAST_FOLLOW,
    PROCEED_TOWARD_LIVE_GATING_CUTOVER,
    REFINE_AND_RERUN,
    RunMetadata,
    generate_report,
    recommend,
)
from shadow_analysis.schema import DefectType, Verdict
from tests.helpers import make_labeled_capture


def _build(n_pass_agree, n_fail_agree, n_escape, n_false_reject, defect=DefectType.OTHER):
    gt, model = [], []
    seq = 0

    for _ in range(n_pass_agree):
        gt.append(make_labeled_capture(seq, Verdict.PASS, board_id=f"B{seq}"))
        model.append(make_labeled_capture(seq, Verdict.PASS, board_id=f"B{seq}"))
        seq += 1

    for _ in range(n_fail_agree):
        gt.append(make_labeled_capture(seq, Verdict.FAIL, board_id=f"B{seq}", defect_types=[defect]))
        model.append(make_labeled_capture(seq, Verdict.FAIL, board_id=f"B{seq}", defect_types=[defect]))
        seq += 1

    for _ in range(n_escape):
        gt.append(make_labeled_capture(seq, Verdict.FAIL, board_id=f"B{seq}", defect_types=[defect]))
        model.append(make_labeled_capture(seq, Verdict.PASS, board_id=f"B{seq}"))
        seq += 1

    for _ in range(n_false_reject):
        gt.append(make_labeled_capture(seq, Verdict.PASS, board_id=f"B{seq}"))
        model.append(make_labeled_capture(seq, Verdict.FAIL, board_id=f"B{seq}", defect_types=[defect]))
        seq += 1

    return gt, model


def test_recommend_refine_and_rerun_when_criteria_fail():
    # 1000 boards, 20% escape rate -> way over the 0.5% threshold.
    gt, model = _build(n_pass_agree=800, n_fail_agree=160, n_escape=40, n_false_reject=0)
    result = compare(gt, model)

    rec = recommend(result)

    assert rec.action == REFINE_AND_RERUN
    assert "escape rate" in rec.rationale


def test_recommend_cutover_when_all_criteria_clean():
    gt, model = _build(n_pass_agree=990, n_fail_agree=10, n_escape=0, n_false_reject=0)
    result = compare(gt, model)

    rec = recommend(result)

    assert rec.action == PROCEED_TOWARD_LIVE_GATING_CUTOVER


def test_recommend_fast_follow_when_criteria_met_but_defect_type_weak():
    gt, model = [], []
    seq = 0
    # 1980 clean pass/fail agreements to comfortably clear the 98% agreement bar
    # and keep the 6 LIFTED_LEAD escapes below (i.e. well under) the 0.5% escape-rate cap.
    for _ in range(1880):
        gt.append(make_labeled_capture(seq, Verdict.PASS, board_id=f"B{seq}"))
        model.append(make_labeled_capture(seq, Verdict.PASS, board_id=f"B{seq}"))
        seq += 1
    for _ in range(100):
        gt.append(
            make_labeled_capture(seq, Verdict.FAIL, board_id=f"B{seq}", defect_types=[DefectType.SCRATCH])
        )
        model.append(
            make_labeled_capture(seq, Verdict.FAIL, board_id=f"B{seq}", defect_types=[DefectType.SCRATCH])
        )
        seq += 1
    # 10 boards with a defect type the model catches only 40% of the time
    # (well under the 90% weak-spot bar), but small enough not to blow the
    # overall escape-rate/agreement thresholds across ~2000 boards total.
    for _ in range(4):
        gt.append(
            make_labeled_capture(
                seq, Verdict.FAIL, board_id=f"B{seq}", defect_types=[DefectType.LIFTED_LEAD]
            )
        )
        model.append(
            make_labeled_capture(
                seq, Verdict.FAIL, board_id=f"B{seq}", defect_types=[DefectType.LIFTED_LEAD]
            )
        )
        seq += 1
    for _ in range(6):
        gt.append(
            make_labeled_capture(
                seq, Verdict.FAIL, board_id=f"B{seq}", defect_types=[DefectType.LIFTED_LEAD]
            )
        )
        model.append(make_labeled_capture(seq, Verdict.PASS, board_id=f"B{seq}"))
        seq += 1

    result = compare(gt, model)
    # Sanity check we're still within the headline acceptance criteria.
    assert result.agreement_rate >= 0.98
    assert result.escape_rate <= 0.005

    rec = recommend(result)

    assert rec.action == PER_JOINT_LOCALIZATION_FAST_FOLLOW
    assert "LIFTED_LEAD" in rec.rationale


def test_generate_report_contains_key_sections():
    gt, model = _build(n_pass_agree=980, n_fail_agree=20, n_escape=0, n_false_reject=0)
    result = compare(gt, model)

    report = generate_report(result, RunMetadata(run_label="test-run", start_date="2026-01-01", end_date="2026-01-14"))

    assert "test-run" in report
    assert "Acceptance Criteria" in report
    assert "Per-Defect-Type Breakdown" in report
    assert "Recommendation" in report
    assert "Proceed Toward Live Gating Cutover" in report
