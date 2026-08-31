"""Defect taxonomy shared across data prep, training, and eval.

Matches the existing AOI's defect taxonomy (see epic issue #1) so that the
new model's outputs are directly comparable to the ground-truth verdicts it
will eventually be fine-tuned and graded against.
"""

# Multi-label defect classes. A board can have zero (pass), one, or several
# of these simultaneously.
DEFECT_CLASSES = [
    "insufficient_solder",
    "excess_solder_bridging",
    "tombstoning",
    "cold_joint",
    "misalignment",
    "missing_component",
]

NUM_DEFECT_CLASSES = len(DEFECT_CLASSES)

DEFECT_CLASS_TO_IDX = {name: i for i, name in enumerate(DEFECT_CLASSES)}
