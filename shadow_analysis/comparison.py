"""Correlate and compare two label streams from a shadow-mode run.

Ground truth = existing AOI's verdicts. Under-test = new model's verdicts.
Both streams are lists of LabeledCapture (a CaptureRecord + the BoardLabel
assigned to that board by whichever system produced the stream).

Correlation strategy (mirrors README's "Correlation" plan):
  1. Prefer CaptureRecord.board_id when both sides have it set (checklist
     item 1 confirmed a real barcode/serial exists).
  2. Fall back to sequence_number, sanity-checked against timestamp within
     a tolerance window, for records where a real board_id isn't available.

All three headline rates (agreement, escape, false-reject) are computed as
a fraction of *matched* boards, matching how the shadow-mode acceptance
criteria in README.md / issue #1 are phrased (e.g. "<=0.5% escape rate"
over the whole run, not conditioned on ground-truth verdict).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from shadow_analysis.schema import DefectType, LabeledCapture, Verdict

DEFAULT_TIMESTAMP_TOLERANCE = timedelta(seconds=30)


@dataclass
class MatchedPair:
    ground_truth: LabeledCapture
    model: LabeledCapture
    method: str  # "board_id" or "sequence_timestamp"


@dataclass
class CorrelationResult:
    pairs: list[MatchedPair]
    unmatched_ground_truth: list[LabeledCapture]
    unmatched_model: list[LabeledCapture]


@dataclass
class DefectTypeStats:
    defect_type: DefectType
    ground_truth_count: int
    caught_count: int
    escaped_count: int
    catch_rate: float | None  # None only if ground_truth_count == 0 (excluded from breakdown)


@dataclass
class ComparisonResult:
    total_ground_truth: int
    total_model: int
    matched_count: int
    unmatched_ground_truth_count: int
    unmatched_model_count: int
    agree_count: int
    escape_count: int
    false_reject_count: int
    agreement_rate: float
    escape_rate: float
    false_reject_rate: float
    defect_breakdown: list[DefectTypeStats]


def _real_board_id(lc: LabeledCapture) -> str | None:
    return lc.capture.board_id


def correlate(
    ground_truth: list[LabeledCapture],
    model_output: list[LabeledCapture],
    timestamp_tolerance: timedelta = DEFAULT_TIMESTAMP_TOLERANCE,
) -> CorrelationResult:
    gt_with_id = [lc for lc in ground_truth if _real_board_id(lc) is not None]
    gt_without_id = [lc for lc in ground_truth if _real_board_id(lc) is None]
    mo_with_id = [lc for lc in model_output if _real_board_id(lc) is not None]
    mo_without_id = [lc for lc in model_output if _real_board_id(lc) is None]

    pairs: list[MatchedPair] = []
    unmatched_gt: list[LabeledCapture] = []
    unmatched_model: list[LabeledCapture] = []

    # Pass 1: match by board_id.
    mo_by_id: dict[str, list[LabeledCapture]] = {}
    for lc in mo_with_id:
        mo_by_id.setdefault(lc.capture.board_id, []).append(lc)

    for gt_lc in gt_with_id:
        bucket = mo_by_id.get(gt_lc.capture.board_id)
        if bucket:
            pairs.append(MatchedPair(gt_lc, bucket.pop(0), method="board_id"))
        else:
            unmatched_gt.append(gt_lc)

    for bucket in mo_by_id.values():
        unmatched_model.extend(bucket)

    # Pass 2: fall back to sequence_number, sanity-checked by timestamp.
    mo_by_seq: dict[int, list[LabeledCapture]] = {}
    for lc in mo_without_id:
        mo_by_seq.setdefault(lc.capture.sequence_number, []).append(lc)

    for gt_lc in gt_without_id:
        bucket = mo_by_seq.get(gt_lc.capture.sequence_number, [])
        match_index = None
        for i, candidate in enumerate(bucket):
            delta = candidate.capture.timestamp - gt_lc.capture.timestamp
            if abs(delta) <= timestamp_tolerance:
                match_index = i
                break
        if match_index is not None:
            pairs.append(MatchedPair(gt_lc, bucket.pop(match_index), method="sequence_timestamp"))
        else:
            unmatched_gt.append(gt_lc)

    for bucket in mo_by_seq.values():
        unmatched_model.extend(bucket)

    return CorrelationResult(pairs=pairs, unmatched_ground_truth=unmatched_gt, unmatched_model=unmatched_model)


def _compute_defect_breakdown(pairs: list[MatchedPair]) -> list[DefectTypeStats]:
    counts: dict[DefectType, dict[str, int]] = {}

    for pair in pairs:
        gt_label = pair.ground_truth.label
        if gt_label.verdict != Verdict.FAIL:
            continue
        model_verdict = pair.model.label.verdict
        for defect_type in gt_label.defect_types:
            if defect_type == DefectType.NONE:
                continue
            bucket = counts.setdefault(defect_type, {"gt": 0, "caught": 0, "escaped": 0})
            bucket["gt"] += 1
            if model_verdict == Verdict.FAIL:
                bucket["caught"] += 1
            else:
                bucket["escaped"] += 1

    breakdown = [
        DefectTypeStats(
            defect_type=defect_type,
            ground_truth_count=bucket["gt"],
            caught_count=bucket["caught"],
            escaped_count=bucket["escaped"],
            catch_rate=bucket["caught"] / bucket["gt"] if bucket["gt"] else None,
        )
        for defect_type, bucket in counts.items()
    ]
    breakdown.sort(key=lambda s: s.ground_truth_count, reverse=True)
    return breakdown


def compare(
    ground_truth: list[LabeledCapture],
    model_output: list[LabeledCapture],
    timestamp_tolerance: timedelta = DEFAULT_TIMESTAMP_TOLERANCE,
) -> ComparisonResult:
    correlation = correlate(ground_truth, model_output, timestamp_tolerance)
    pairs = correlation.pairs
    matched = len(pairs)

    if matched == 0:
        raise ValueError("no correlated pairs between ground truth and model output streams")

    agree = sum(1 for p in pairs if p.ground_truth.label.verdict == p.model.label.verdict)
    escapes = sum(
        1
        for p in pairs
        if p.model.label.verdict == Verdict.PASS and p.ground_truth.label.verdict == Verdict.FAIL
    )
    false_rejects = sum(
        1
        for p in pairs
        if p.model.label.verdict == Verdict.FAIL and p.ground_truth.label.verdict == Verdict.PASS
    )

    return ComparisonResult(
        total_ground_truth=len(ground_truth),
        total_model=len(model_output),
        matched_count=matched,
        unmatched_ground_truth_count=len(correlation.unmatched_ground_truth),
        unmatched_model_count=len(correlation.unmatched_model),
        agree_count=agree,
        escape_count=escapes,
        false_reject_count=false_rejects,
        agreement_rate=agree / matched,
        escape_rate=escapes / matched,
        false_reject_rate=false_rejects / matched,
        defect_breakdown=_compute_defect_breakdown(pairs),
    )
