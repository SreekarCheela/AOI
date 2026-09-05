"""Whole-pipeline guard.

The trap tests each pin one mechanism. This one pins the thing anybody actually
cares about: a misregistered board carrying one of every defect type is fully
caught, correctly labelled, and no good part is flagged.

It exists so a refactor cannot quietly trade detection for tidiness. If this goes
red and the trap tests stay green, the regression is in how the stages fit
together rather than in any one of them.
"""
from __future__ import annotations

from aoi import pipeline, synth

# One of every injectable defect, spread across package families.
# TANT-A (C1) and D2 are deliberately left good: the README records that
# tombstone/missing discrimination degrades on packages whose body nearly covers
# the lands, so TANT-A is not a fair target for that class and is used here as a
# known-good part instead.
INJECTED = {
    "R1": ("missing", synth.Defect("missing")),
    # pad 0 specifically: this is the case that used to clear the old global
    # 0.62 presence gate and come back as "skew". It stays here as the e2e
    # guard for that fix.
    "R2": ("tombstone", synth.Defect("tombstone", pad=0)),
    "R3": ("offset", synth.Defect("offset", dx=0.25)),
    "R4": ("insufficient", synth.Defect("insufficient", pad=1)),
    "R5": ("skew", synth.Defect("skew", dtheta=20.0)),
    "R6": ("bridge", synth.Defect("bridge")),
    "D1": ("reversed", synth.Defect("reversed")),
}


def _inspect(brd, golden):
    stack = synth.render(
        brd,
        defects={r: d for r, (_, d) in INJECTED.items()},
        opts=synth.RenderOpts(seed=77, place_sigma_um=0.0,
                              board_shift_mm=(0.38, -0.29), board_rot_deg=0.55))
    return stack, pipeline.Inspector(golden).inspect(stack, brd)


def test_every_injected_defect_is_caught(brd, golden):
    _, res = _inspect(brd, golden)
    flagged = {v.refdes for v in res.failed}
    missed = set(INJECTED) - flagged
    assert not missed, f"missed {sorted(missed)}"


def test_no_good_component_is_flagged(brd, golden):
    _, res = _inspect(brd, golden)
    flagged = {v.refdes for v in res.failed}
    false_calls = flagged - set(INJECTED)
    assert not false_calls, (
        f"false calls on {sorted(false_calls)}. False call rate is the number "
        "that decides whether the machine gets used (spec section 8).")


def test_every_defect_is_labelled_with_its_own_kind(brd, golden):
    _, res = _inspect(brd, golden)
    by_refdes = {v.refdes: v for v in res.verdicts}
    wrong = {r: (kind, by_refdes[r].reason)
             for r, (kind, _) in INJECTED.items()
             if by_refdes[r].reason != kind}
    assert not wrong, (
        "defects caught but mislabelled (injected -> reported): "
        + ", ".join(f"{r}: {a} -> {b}" for r, (a, b) in sorted(wrong.items())))


def test_alignment_is_recovered_before_measuring(brd, golden):
    _, res = _inspect(brd, golden)
    assert abs(abs(res.align_shift_mm[0]) - 0.38) < 0.08
    assert abs(abs(res.align_shift_mm[1]) - 0.29) < 0.08
    assert abs(abs(res.align_rot_deg) - 0.55) < 0.2
