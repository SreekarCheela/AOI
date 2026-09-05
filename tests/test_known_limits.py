"""Known limitations, pinned as xfail so they are tracked rather than forgotten.

These are not test bugs. Each one is a real gap found by measurement and left
unfixed on purpose, recorded here so that:

  * the behaviour is written down with the numbers that prove it, and
  * the day someone fixes it, the test flips to XPASS and says so out loud.

`strict=False` is deliberate: an XPASS here is good news, not a suite failure.
When one goes green, delete it and move the case into the real suite.

FIXED AND MOVED OUT: 0402 tombstone detection used to depend on which pad
survived. That was the global `ncc < 0.62` gate; it is now package-relative and
covered properly in test_presence_gate.py.
"""
from __future__ import annotations

import pytest

from aoi import rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with

# Both tall polarised packages fail the same way and for the same reason, so
# they share one xfail rather than two near-identical ones.
TALL_POLARISED = [("D1", "SOD-123"), ("C1", "TANT-A")]


@pytest.mark.xfail(strict=False, reason=(
    "Tombstones on tall polarised packages are mislabelled. Two separate causes "
    "stack up. (1) The polarity test runs before presence by design - a reversed "
    "part IS present, so it has to be settled first - but a body standing on one "
    "land flips the 180-degree template margin, so pad 1 comes back 'reversed'. "
    "(2) On pad 0 the surviving joint reads only ~0.43 of the golden low-angle "
    "band, below the 0.75 needed to read as stood-up, so it falls through to the "
    "placement tests and comes back 'offset'. Both packages have a body that "
    "nearly covers the lands, leaving a fillet window ~0.2 mm wide, which is "
    "what makes the surviving joint so weak. Fixing this needs a measurement "
    "that distinguishes a standing body from a displaced one, not another "
    "threshold."))
@pytest.mark.parametrize("refdes,package", TALL_POLARISED)
@pytest.mark.parametrize("pad", [0, 1])
def test_tall_polarised_tombstone_is_mislabelled(brd, golden, refdes, package, pad):
    stack = render_with(brd, {refdes: synth.Defect("tombstone", pad=pad)})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    verdict = rule_verdict(m, golden)
    assert verdict.reason == "tombstone", (
        f"{package} pad {pad}: ncc={m.ncc:.3f} -> {verdict.reason!r}")


@pytest.mark.parametrize("refdes,package", TALL_POLARISED)
@pytest.mark.parametrize("pad", [0, 1])
def test_tall_polarised_tombstone_still_fails_the_board(brd, golden, refdes, package, pad):
    """The containment guarantee that makes the xfail above tolerable.
    Mislabelled is survivable, undetected is not. Whatever name it comes back
    under, a tombstoned part must never be passed."""
    stack = render_with(brd, {refdes: synth.Defect("tombstone", pad=pad)})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    assert rule_verdict(m, golden).failed, \
        f"{package} tombstone on pad {pad} passed inspection"


@pytest.mark.xfail(strict=False, reason=(
    "NCC cannot separate 'no body at all' from 'body walked off its lands', "
    "because the two populations overlap ACROSS packages. Sampled over four "
    "components and three seeds per package: a missing part reaches 0.572 of "
    "golden (SOD-123) while a part displaced 500 um falls to 0.505 (0603). No "
    "value of NCC_ABSENT_FRACTION sits between those, so on chips a gross "
    "displacement reads 'missing'. On an 0402 it does not even get that far - "
    "the body ends up over the inter-pad gap and trips the bridge test, which "
    "runs first by design because a bridge wrecks every other measurement. "
    "Separating these needs a measurement that sees the body somewhere else on "
    "the crop, not a better threshold. Every case still fails the board; only "
    "the label is wrong."))
@pytest.mark.parametrize("refdes,package",
                         [("R1", "0402"), ("R3", "0603"), ("R5", "0805")])
def test_gross_displacement_is_labelled_offset(brd, golden, refdes, package):
    stack = render_with(brd, {refdes: synth.Defect("offset", dx=0.50)})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    assert rule_verdict(m, golden).reason == "offset"


@pytest.mark.xfail(strict=False, reason=(
    "Skew on a tall polarised package is reported as 'offset'. Rotating a long "
    "body moves the correlation peak far enough to trip the 150 um placement "
    "limit, and the offset test runs before the skew test. The part is caught "
    "and is genuinely displaced, so the call is defensible, but a 20 degree "
    "rotation is more usefully described as skew."))
@pytest.mark.parametrize("refdes,package", TALL_POLARISED)
def test_skew_on_tall_packages_is_labelled_skew(brd, golden, refdes, package):
    stack = render_with(brd, {refdes: synth.Defect("skew", dtheta=20.0)})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    assert rule_verdict(m, golden).reason == "skew"


@pytest.mark.xfail(strict=False, reason=(
    "A package the golden set never covered flags every instance of it as "
    "'missing'. rules.measure returns early when there is no golden entry, "
    "leaving ncc at its 0.0 default, and the untaught branch of the presence "
    "gate reads that as an absent part. On a real board this is a false-call "
    "storm: 100 percent of a package the teach boards happened not to carry. "
    "Pre-existing, unchanged by the package-relative gate. The fix is to report "
    "these as untestable rather than failed, which needs a verdict state the "
    "rule layer does not currently have."))
def test_untaught_package_does_not_flag_good_parts(brd, golden):
    empty = rules.Golden()
    stack = render_with(brd, {})
    flagged = [c.refdes for c in brd.components
               if rule_verdict(rules.measure(stack, c, empty), empty).failed]
    assert not flagged, f"{len(flagged)} good parts flagged with no golden model: {flagged}"
