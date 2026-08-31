"""Evaluation metrics for the shadow-mode acceptance criteria (epic issue #1).

Pass/fail is graded against ground truth as: 1 = pass, 0 = fail (defect present).

Definitions (rates expressed as fraction of the *total evaluated boards*, matching
how the epic's acceptance criteria list them alongside pass/fail agreement rate):

- agreement_rate: fraction of boards where model pass/fail verdict == ground truth.
- escape_rate: fraction of *all* boards where the model said pass but ground truth
  was fail (a real defect "escaped" detection). This is the industry-standard AOI
  framing (escapes per boards inspected, not per defective board) and is what the
  epic's <=0.5% bar is calibrated against.
- false_reject_rate: fraction of *all* boards where the model said fail but ground
  truth was pass (a good board incorrectly rejected).

`escape_rate_of_defects` / `false_reject_rate_of_passes` are also provided
(normalized against the actual-fail / actual-pass subpopulations respectively,
i.e. miss rate and false-positive rate) since they're a more standard way to
read model quality independent of the pass/fail class balance in a given
sample -- useful for debugging, even though the epic's acceptance bar is
defined against the total-boards denominator above.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from labels import DEFECT_CLASSES


def pass_fail_confusion(preds: np.ndarray, targets: np.ndarray) -> dict:
    """preds/targets: 1D arrays of 0/1, 1=pass. Returns raw confusion counts."""
    assert preds.shape == targets.shape
    total = len(targets)
    true_pass = targets == 1
    true_fail = targets == 0
    pred_pass = preds == 1
    pred_fail = preds == 0

    return {
        "total": int(total),
        "true_positive_pass": int(np.sum(true_pass & pred_pass)),
        "true_negative_fail": int(np.sum(true_fail & pred_fail)),
        # escape: ground truth fail, model said pass
        "escapes": int(np.sum(true_fail & pred_pass)),
        # false reject: ground truth pass, model said fail
        "false_rejects": int(np.sum(true_pass & pred_fail)),
        "actual_pass": int(np.sum(true_pass)),
        "actual_fail": int(np.sum(true_fail)),
    }


def pass_fail_rates(preds: np.ndarray, targets: np.ndarray) -> dict:
    c = pass_fail_confusion(preds, targets)
    total = max(c["total"], 1)
    actual_fail = max(c["actual_fail"], 1)
    actual_pass = max(c["actual_pass"], 1)

    return {
        "agreement_rate": (c["true_positive_pass"] + c["true_negative_fail"]) / total,
        "escape_rate": c["escapes"] / total,
        "false_reject_rate": c["false_rejects"] / total,
        "escape_rate_of_defects": c["escapes"] / actual_fail,
        "false_reject_rate_of_passes": c["false_rejects"] / actual_pass,
        **c,
    }


def defect_type_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """preds/targets: 2D arrays [N, num_classes] of 0/1 multi-label predictions."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(DEFECT_CLASSES)
    }
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        targets, preds, average="micro", zero_division=0
    )
    return {
        "per_class": per_class,
        "micro": {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f1)},
    }
