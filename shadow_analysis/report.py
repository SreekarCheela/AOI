"""Markdown report generator for shadow-mode comparison results.

Thresholds below are the epic's shadow-mode acceptance criteria (README.md
/ issue #1 / issue #6):
  - >=98% pass/fail agreement
  - <=0.5% escape rate
  - <=3% false-reject rate
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shadow_analysis.comparison import ComparisonResult

AGREEMENT_THRESHOLD = 0.98
ESCAPE_RATE_THRESHOLD = 0.005
FALSE_REJECT_RATE_THRESHOLD = 0.03

# Below this per-defect-type catch rate (with enough ground-truth samples to
# be meaningful), flag the defect type as a weak spot in the recommendation.
WEAK_DEFECT_CATCH_RATE = 0.90
WEAK_DEFECT_MIN_SAMPLES = 5

REFINE_AND_RERUN = "REFINE_AND_RERUN"
PER_JOINT_LOCALIZATION_FAST_FOLLOW = "PER_JOINT_LOCALIZATION_FAST_FOLLOW"
PROCEED_TOWARD_LIVE_GATING_CUTOVER = "PROCEED_TOWARD_LIVE_GATING_CUTOVER"


@dataclass
class RunMetadata:
    """Optional context about the run itself, embedded in the report header.

    Shadow mode can't physically run yet (blocked on hardware install,
    issues #2/#3), so every field defaults to a placeholder. Once real logs
    exist, pass this in with actual values.
    """

    run_label: str = "(unlabeled run)"
    start_date: str | None = None
    end_date: str | None = None
    spot_check_notes: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    action: str
    rationale: str


def recommend(result: ComparisonResult) -> Recommendation:
    meets_agreement = result.agreement_rate >= AGREEMENT_THRESHOLD
    meets_escape = result.escape_rate <= ESCAPE_RATE_THRESHOLD
    meets_false_reject = result.false_reject_rate <= FALSE_REJECT_RATE_THRESHOLD

    if not (meets_agreement and meets_escape and meets_false_reject):
        failed = []
        if not meets_agreement:
            failed.append("agreement rate")
        if not meets_escape:
            failed.append("escape rate")
        if not meets_false_reject:
            failed.append("false-reject rate")
        return Recommendation(
            action=REFINE_AND_RERUN,
            rationale=(
                "Acceptance criteria not met on: " + ", ".join(failed) + ". "
                "Refine the model (or investigate data/labeling issues) and re-run shadow mode."
            ),
        )

    weak_defect_types = [
        d
        for d in result.defect_breakdown
        if d.catch_rate is not None
        and d.catch_rate < WEAK_DEFECT_CATCH_RATE
        and d.ground_truth_count >= WEAK_DEFECT_MIN_SAMPLES
    ]
    if weak_defect_types:
        names = ", ".join(f"{d.defect_type.value} ({d.catch_rate:.0%} catch rate)" for d in weak_defect_types)
        return Recommendation(
            action=PER_JOINT_LOCALIZATION_FAST_FOLLOW,
            rationale=(
                "Overall acceptance criteria are met, but catch rate is below "
                f"{WEAK_DEFECT_CATCH_RATE:.0%} for: {names}. "
                "Per-joint localization may help target these specific defect types "
                "before proceeding toward cutover."
            ),
        )

    return Recommendation(
        action=PROCEED_TOWARD_LIVE_GATING_CUTOVER,
        rationale=(
            "All acceptance criteria are met with no significant per-defect-type weak spots. "
            "The system appears ready for live-gating cutover planning "
            "(actual PLC/line integration is out of scope for this epic)."
        ),
    )


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _criteria_table(result: ComparisonResult) -> str:
    rows = [
        ("Pass/fail agreement", _pct(result.agreement_rate), f">= {_pct(AGREEMENT_THRESHOLD)}", result.agreement_rate >= AGREEMENT_THRESHOLD),
        ("Escape rate", _pct(result.escape_rate), f"<= {_pct(ESCAPE_RATE_THRESHOLD)}", result.escape_rate <= ESCAPE_RATE_THRESHOLD),
        ("False-reject rate", _pct(result.false_reject_rate), f"<= {_pct(FALSE_REJECT_RATE_THRESHOLD)}", result.false_reject_rate <= FALSE_REJECT_RATE_THRESHOLD),
    ]
    lines = [
        "| Metric | Observed | Threshold | Result |",
        "|---|---|---|---|",
    ]
    for name, observed, threshold, met in rows:
        lines.append(f"| {name} | {observed} | {threshold} | {'PASS' if met else 'FAIL'} |")
    return "\n".join(lines)


def _defect_breakdown_table(result: ComparisonResult) -> str:
    if not result.defect_breakdown:
        return "No ground-truth defects observed in this run."

    lines = [
        "| Defect Type | GT Count | Caught | Escaped | Catch Rate |",
        "|---|---|---|---|---|",
    ]
    for stat in result.defect_breakdown:
        catch_rate = _pct(stat.catch_rate) if stat.catch_rate is not None else "n/a"
        lines.append(
            f"| {stat.defect_type.value} | {stat.ground_truth_count} | {stat.caught_count} | "
            f"{stat.escaped_count} | {catch_rate} |"
        )
    return "\n".join(lines)


def generate_report(result: ComparisonResult, metadata: RunMetadata | None = None) -> str:
    metadata = metadata or RunMetadata()
    rec = recommend(result)

    spot_checks = (
        "\n".join(f"- {note}" for note in metadata.spot_check_notes)
        if metadata.spot_check_notes
        else "- (none logged)"
    )

    return f"""# Shadow-Mode Comparison Report — {metadata.run_label}

**Run window:** {metadata.start_date or "n/a"} to {metadata.end_date or "n/a"}

## Correlation Summary

- Ground-truth (existing AOI) records: {result.total_ground_truth}
- Model (new system) records: {result.total_model}
- Matched pairs: {result.matched_count}
- Unmatched ground-truth records: {result.unmatched_ground_truth_count}
- Unmatched model records: {result.unmatched_model_count}

## Acceptance Criteria

{_criteria_table(result)}

Agree: {result.agree_count} / Escapes: {result.escape_count} / False-rejects: {result.false_reject_count} (of {result.matched_count} matched boards)

## Per-Defect-Type Breakdown

Catch rate = fraction of ground-truth boards with this defect type that the model also failed.

{_defect_breakdown_table(result)}

## Weekly Spot-Checks (camera/lighting alignment drift)

{spot_checks}

## Recommendation

**{rec.action.replace('_', ' ').title()}**

{rec.rationale}
"""
