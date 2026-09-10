"""What correlation still does for presence, now that body localisation leads.

The decision moved to `test_body.py`: where the body is beats how well it
correlates, because correlation cannot say WHERE. Correlation keeps two jobs
that survived the change, and they are what this file pins.

  * a second opinion before a part is called missing, so a segmentation that
    fails wholesale cannot flag a whole board

A package teach never saw is no longer handled here at all: it is uninspected
rather than gated on any threshold, and test_untaught.py owns that.

The history is worth keeping. Presence was once a global `ncc < 0.62`, which
could not mean the same thing for two packages -- an 0603 matches its own
template at 0.92 and a SOD-123 at 0.99 -- so a tombstoned 0402 scoring 0.599 on
one pad and 0.647 on the other was found or missed depending on which land
survived. Making it package-relative fixed that; measuring the body directly
made the whole gate unnecessary.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aoi import config, pipeline, rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with

CHIPS = ["0402", "0603", "0805"]
CHIP_REFDES = {"0402": "R1", "0603": "R3", "0805": "R5"}


# ------------------------------------------------------- what teach now records
def test_golden_records_the_good_part_ncc_distribution(golden):
    """fit_golden has to measure how well a GOOD part matches its own template,
    or there is nothing to make the threshold relative to."""
    for name, g in golden.packages.items():
        assert 0.5 < g.ncc_mean <= 1.0, \
            f"{name} golden NCC mean {g.ncc_mean:.3f} is not a plausible score"
        assert g.ncc_std >= 0.0
        assert g.ncc_absent < g.ncc_mean, (
            f"{name} absent threshold {g.ncc_absent:.3f} is not below the golden "
            f"mean {g.ncc_mean:.3f}")


def test_the_absent_gate_separates_for_every_package(golden, brd):
    """What this gate has to do, and nothing more.

    It used to assert that the spread of golden NCC across packages exceeded
    0.04, as a proxy for "a global constant would not serve every package".
    Both halves of that have since stopped being true, and it is worth recording
    why rather than quietly relaxing the number.

    The proxy moved: correcting the parallax smear made every package match its
    own template better, so the golden means closed up from a 0.07 spread to
    0.033 while nothing about the gate changed.

    And the claim itself no longer holds. It was the justification for retiring
    a global `ncc < 0.62` FAIL gate, back when correlation had to separate
    "absent" from "walked off its lands" -- a 0.572-to-0.612 window, far too
    tight to hold, and genuinely impossible with one constant. Body localisation
    took that job away: a displaced part still HAS a body, so it never reaches
    here. All that is left is "is there a body at all", and measured across
    every package that window is enormous -- a missing part scores up to 0.554
    where the worst good part scores 0.924.

    So a global constant WOULD work for what remains. Keeping the gate relative
    costs nothing and stays correct if the window ever narrows again; what would
    be wrong is leaving a test that claims a justification the code outgrew.
    What the gate must still do, for every package, is separate.
    """
    good, absent = {}, {}
    for comp in brd.components:
        if golden.packages.get(comp.package) is None:
            continue
        clean = render_with(brd, {})
        gone = render_with(brd, {comp.refdes: synth.Defect(kind="missing")})
        good.setdefault(comp.package, []).append(rules.measure(clean, comp, golden).ncc)
        absent.setdefault(comp.package, []).append(rules.measure(gone, comp, golden).ncc)

    for pkg in sorted(good):
        lo, hi = min(good[pkg]), max(absent[pkg])
        gate = golden.packages[pkg].ncc_absent
        assert hi < gate < lo, (
            f"{pkg}: the absent gate {gate:.3f} does not sit between the best a "
            f"missing part scored ({hi:.3f}) and the worst a good one scored "
            f"({lo:.3f})")
        assert (lo - gate) / lo > 0.10, (
            f"{pkg}: only {(lo - gate) / lo:.1%} margin between the gate and a "
            f"good part -- too tight to survive real boards")


# ------------------------------- the original 0402 bug, kept as a history pin
# (full coverage for every package now lives in test_body.py)
@pytest.mark.parametrize("package", CHIPS)
@pytest.mark.parametrize("pad", [0, 1])
def test_tombstone_is_caught_on_either_pad(brd, golden, package, pad):
    """The bug. Detection must not depend on which land survived.

    Both pads are exercised for every chip package, because the original failure
    was invisible on one pad and present on the other.
    """
    refdes = CHIP_REFDES[package]
    stack = render_with(brd, {refdes: synth.Defect("tombstone", pad=pad)})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    verdict = rule_verdict(m, golden)
    assert verdict.reason == "tombstone", (
        f"{package} tombstone on pad {pad} reported as {verdict.reason!r} "
        f"(ncc={m.ncc:.3f}, golden={golden.packages[package].ncc_mean:.3f})")


@pytest.mark.parametrize("package", CHIPS)
def test_tombstone_detection_does_not_depend_on_which_pad_survives(brd, golden, package):
    """Stated as its own assertion because it is the property that broke: the
    two pads must reach the same verdict, whatever that verdict is."""
    refdes = CHIP_REFDES[package]
    got = []
    for pad in (0, 1):
        stack = render_with(brd, {refdes: synth.Defect("tombstone", pad=pad)})
        m = rules.measure(stack, brd.by_refdes(refdes), golden)
        got.append(rule_verdict(m, golden).reason)
    assert got[0] == got[1], \
        f"{package} tombstone reads {got[0]!r} on pad 0 but {got[1]!r} on pad 1"
