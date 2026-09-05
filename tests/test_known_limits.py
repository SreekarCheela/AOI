"""Known limitations, pinned as xfail so they are tracked rather than forgotten.

These are not test bugs. Each one is a real gap found by measurement and left
unfixed on purpose, recorded here so that:

  * the behaviour is written down with the numbers that prove it, and
  * the day someone fixes it, the test flips to XPASS and says so out loud.

`strict=False` is deliberate: an XPASS here is good news, not a suite failure.
When one goes green, delete it and move the case into the real suite.

FIXED AND MOVED OUT
  0402 tombstone depended on which pad survived, because the presence gate was a
  global `ncc < 0.62`. Fixed by making it package-relative; then made moot
  entirely by body localisation. Covered in test_body.py.

  Tombstones on tall polarised packages read 'offset' or 'reversed', and gross
  displacement on 0603/0805 read 'missing'. Both were the same root cause: NCC
  cannot say WHERE the body went, so absent and elsewhere were indistinguishable.
  Fixed by measuring the body's position and extent. Covered in test_body.py.

  A package teach never saw flagged every instance of itself as missing, because
  measure left ncc at its 0.0 default and 0.0 is a real measurement meaning
  "correlates with nothing". Fixed by making unmeasurable fields NaN and adding
  an uninspected verdict state. Covered in test_untaught.py.
"""
from __future__ import annotations

import pytest

from aoi import rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with


@pytest.mark.xfail(strict=False, reason=(
    "An 0402 displaced far enough reads 'bridge', not 'offset'. At 500 um the "
    "body and its solder land squarely over the inter-pad gap, and the specular "
    "reading there is what the bridge test measures. The bridge test runs first "
    "by design, because a bridge wrecks every other measurement on the "
    "component, so it wins before placement is ever considered. Only the "
    "smallest package does this: on 0603 and up the gap is wide enough relative "
    "to the displacement that the reading stays clear. The part is still caught "
    "and still fails the board; the operator is sent to look for a short that is "
    "not there. Fixing it needs the bridge test to know the body is sitting in "
    "the gap, which is now measurable -- body_shift_um is a feature -- but it "
    "changes a rule that currently has no exceptions, so it is left alone."))
@pytest.mark.parametrize("dx", [0.50, 0.80])
def test_grossly_displaced_0402_is_labelled_offset(brd, golden, dx):
    m = rules.measure(render_with(brd, {"R1": synth.Defect("offset", dx=dx)}),
                      brd.by_refdes("R1"), golden)
    assert rule_verdict(m, golden).reason == "offset"


@pytest.mark.xfail(strict=False, reason=(
    "Skew on a tall polarised package is reported as 'offset'. Rotating a long "
    "body moves the correlation peak far enough to trip the 150 um placement "
    "limit, and the offset test runs before the skew test. The part is caught "
    "and is genuinely displaced, so the call is defensible, but a 20 degree "
    "rotation is more usefully described as skew. Body localisation does not "
    "help here: a rotated body is still lying flat and still spans its own "
    "length, which is exactly what the aspect test measures."))
@pytest.mark.parametrize("refdes,package", [("D1", "SOD-123"), ("C1", "TANT-A")])
def test_skew_on_tall_packages_is_labelled_skew(brd, golden, refdes, package):
    m = rules.measure(render_with(brd, {refdes: synth.Defect("skew", dtheta=20.0)}),
                      brd.by_refdes(refdes), golden)
    assert rule_verdict(m, golden).reason == "skew"
