"""Known limitations, pinned as xfail so they are tracked rather than forgotten.

These are not test bugs. Each one is a real gap found by measurement and left
unfixed on purpose, recorded here so that:

  * the behaviour is written down with the numbers that prove it, and
  * the day someone fixes it, the test flips to XPASS and says so out loud.

`strict=False` is deliberate: an XPASS here is good news, not a suite failure.
When one goes green, delete it and move the case into the real suite.
"""
from __future__ import annotations

import pytest

from aoi import rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with


@pytest.mark.xfail(strict=False, reason=(
    "0402 tombstone detection is a coin flip. The presence gate in "
    "fusion.rule_verdict fires at ncc < 0.62, but a tombstoned 0402 lands at "
    "0.599 when pad 1 survives and 0.647 when pad 0 does. The pad-0 case "
    "therefore clears the gate, falls through to the placement tests, and comes "
    "back labelled 'skew' because the square standing body gives the rotation "
    "search a spurious 24 deg peak. 0603 (0.51) and 0805 (0.47) both sit well "
    "clear of the threshold, so this is specific to the smallest chip - which "
    "is also the one most prone to tombstoning in reality."))
@pytest.mark.parametrize("refdes", ["R1", "R2", "R6"])
def test_0402_tombstone_on_pad_0_is_mislabelled_as_skew(brd, golden, refdes):
    stack = render_with(brd, {refdes: synth.Defect("tombstone", pad=0)})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    verdict = rule_verdict(m, golden)
    assert verdict.reason == "tombstone", (
        f"{refdes} ncc={m.ncc:.3f} dtheta={m.dtheta_deg:+.1f} -> {verdict.reason!r}")


@pytest.mark.xfail(strict=False, reason=(
    "TANT-A tombstones are not merely confused with 'missing' as README.md "
    "states, they are mislabelled outright: pad 0 reads ncc=0.764 and comes back "
    "as 'offset', pad 1 reads 0.533 and comes back as 'reversed' because the "
    "polarity margin flips when the body stands on one land. The body nearly "
    "covers the lands on this package, so the fillet window is only ~0.2 mm "
    "wide and the surviving joint is not distinguishable. Both still fail the "
    "board, so no defect escapes - but the operator is sent to the wrong "
    "process fix."))
@pytest.mark.parametrize("pad", [0, 1])
def test_tant_a_tombstone_is_mislabelled(brd, golden, pad):
    stack = render_with(brd, {"C1": synth.Defect("tombstone", pad=pad)})
    m = rules.measure(stack, brd.by_refdes("C1"), golden)
    verdict = rule_verdict(m, golden)
    assert verdict.reason == "tombstone", (
        f"C1 pad {pad} ncc={m.ncc:.3f} -> {verdict.reason!r}")


def test_tant_a_tombstone_still_fails_the_board(brd, golden):
    """The containment guarantee that makes the two xfails above tolerable:
    mislabelled is survivable, undetected is not. Whatever name it comes back
    under, a tombstoned part must never be passed."""
    for pad in (0, 1):
        stack = render_with(brd, {"C1": synth.Defect("tombstone", pad=pad)})
        m = rules.measure(stack, brd.by_refdes("C1"), golden)
        assert rule_verdict(m, golden).failed, \
            f"C1 tombstone on pad {pad} passed inspection"
