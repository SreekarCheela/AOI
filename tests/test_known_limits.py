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


@pytest.mark.xfail(strict=False, reason=(
    "A substituted part whose body length happens to match the EXPECTED part's "
    "width is reported as 'tombstone', not 'wrong_part'. An 0603 body is "
    "1.60 mm long and a SOD-123 is 1.60 mm wide, so on a SOD-123 footprint the "
    "0603 spans exactly 1.00x the expected width along the pad axis -- which is "
    "the definition of a part standing on one land. The same happens to an 0402 "
    "on an 0805 footprint (0.80x). On body geometry alone the two are not "
    "distinguishable, and the tombstone test runs first on purpose: standing on "
    "one land is far commoner than a wrong reel, so the reading with a specific "
    "explanation claims it. The part is caught and the board still fails; the "
    "operator is sent to look at the joint rather than at the stores. The fix "
    "is to make the tombstone test require what a tombstone actually implies -- "
    "one land wetted and the other bare, which spill_fractions now measures per "
    "land -- rather than body shape alone."))
@pytest.mark.parametrize("refdes,package,fitted", [
    ("D1", "SOD-123", "0603"),
    ("R5", "0805", "0402"),
])
def test_a_small_substitution_is_labelled_wrong_part(brd, golden, refdes, package,
                                                     fitted):
    m = rules.measure(render_with(brd, {refdes: synth.Defect("wrong_part",
                                                             substitute=fitted)}),
                      brd.by_refdes(refdes), golden)
    assert rule_verdict(m, golden).reason == "wrong_part"


@pytest.mark.xfail(strict=False, reason=(
    "A SOD-123 fitted on a TANT-A footprint is reported as 'insufficient', not "
    "'wrong_part'. The two bodies share a width and differ only in length, "
    "3.20 against 2.70 mm, so the segmented area lands at 0.86x golden -- inside "
    "the good population, whose low tail reaches 0.80x when LED output drops 8%. "
    "There is no threshold that separates them, and widening the bound to catch "
    "this would sit under the good side's own spread. It is caught anyway, by "
    "the joint: a shorter body leaves a longer run from land to body edge, the "
    "fillet is stretched over it and reads starved. The board fails and the part "
    "is flagged, under a name that describes a real consequence of the wrong "
    "part being there. Separating them needs the marking classifier the spec "
    "asks for in section 5, or a segmentation whose thresholds track the crop's "
    "own substrate level instead of an absolute one learned at teach."))
def test_a_same_width_substitution_is_labelled_wrong_part(brd, golden):
    m = rules.measure(render_with(brd, {"C1": synth.Defect("wrong_part",
                                                           substitute="SOD-123")}),
                      brd.by_refdes("C1"), golden)
    assert rule_verdict(m, golden).reason == "wrong_part"
