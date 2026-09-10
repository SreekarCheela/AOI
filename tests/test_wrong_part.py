"""Wrong part: the right footprint carrying the wrong body.

Spec section 5 asks for body dimensions from the segmented outline plus a
marking classifier. Only the first half is here; OCR needs a character model
per manufacturer and a legible marking, which an 0402 does not have.

This test sits with presence rather than with joint quality, because it decides
whether the golden model may be applied at all. Every test after it -- the
placement search, the polarity margin, the fillet windows -- compares against
statistics for the package the centroid file NAMES. Fit a different part and
all of them are scoring against the wrong reference, so their readings are not
wrong so much as meaningless.

Scored on blob AREA, not on the bounding box. See `test_area_is_used_because_
the_bounding_box_is_not` for the measurement that settled it.
"""
from __future__ import annotations

import numpy as np
import pytest

from aoi import board as B
from aoi import config, packages, pipeline, rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with

# Feeder mix-ups worth catching: one size step within a family, and a swap
# across families that a reel label would not stop.
SUBSTITUTIONS = [
    ("R3", "0603", "0402"),      # a size down
    ("R1", "0402", "0603"),      # a size up
    ("R5", "0805", "0603"),
    ("R3", "0603", "0805"),
    ("C1", "TANT-A", "0805"),
    ("R1", "0402", "0805"),      # two sizes up
    ("D1", "SOD-123", "TANT-A"), # right family, wrong part
    ("R4", "0603", "0402"),      # same substitution, part rotated 180
    ("R2", "0402", "0603"),      # same substitution, part rotated 90
]


@pytest.mark.parametrize("refdes,package,fitted", SUBSTITUTIONS)
def test_a_substituted_part_is_caught(brd, golden, refdes, package, fitted):
    m = rules.measure(render_with(brd, {refdes: synth.Defect("wrong_part",
                                                             substitute=fitted)}),
                      brd.by_refdes(refdes), golden)
    v = rule_verdict(m, golden)
    assert v.failed, (
        f"a {fitted} body on a {package} footprint passed inspection "
        f"(area {m.body_area_ratio:.2f}x golden)")
    assert v.reason == "wrong_part", (
        f"a {fitted} on a {package} footprint reported as {v.reason!r} "
        f"(area {m.body_area_ratio:.2f}x golden, aspect {m.body_aspect:.2f})")


def test_a_good_part_is_never_called_wrong(brd, golden):
    """Sampled the way the machine will meet it, because the low bound is the
    one that has to hold: the segmentation thresholds are absolute diffuse and
    specular levels learned at teach, so dimming the whole image erodes the blob
    edge and shrinks the measured body."""
    flagged = []
    for seed, vig, gain in [(301, 0.06, 1.00), (302, 0.06, 1.00), (303, 0.06, 1.00),
                            (7100, 0.12, 1.00), (7101, 0.06, 0.92),
                            (7102, 0.12, 0.95), (7103, 0.02, 1.06)]:
        stack = synth.render(brd, opts=synth.RenderOpts(
            seed=seed, vignette=vig, gain=gain))
        stack, _, _ = pipeline.align_board(stack, brd)
        for c in brd.components:
            m = rules.measure(stack, c, golden)
            if rule_verdict(m, golden).reason == "wrong_part":
                flagged.append((c.refdes, c.package, round(m.body_area_ratio, 3),
                                f"gain {gain}"))
    assert not flagged, f"{len(flagged)} false 'wrong_part' calls: {flagged[:6]}"


def test_the_good_population_keeps_clear_of_both_bounds(brd, golden):
    """The margin, stated as a number so a regression is visible as one."""
    ratios = []
    for seed, vig, gain in [(301, 0.06, 1.00), (7100, 0.12, 1.00),
                            (7101, 0.06, 0.92), (7103, 0.02, 1.06)]:
        stack = synth.render(brd, opts=synth.RenderOpts(
            seed=seed, vignette=vig, gain=gain))
        for c in brd.components:
            ratios.append(rules.measure(stack, c, golden).body_area_ratio)
    lo, hi = min(ratios), max(ratios)
    assert lo > config.WRONG_PART_AREA_LO, (
        f"a good part measured {lo:.3f}x golden area, at the "
        f"{config.WRONG_PART_AREA_LO} low bound")
    assert hi < config.WRONG_PART_AREA_HI, (
        f"a good part measured {hi:.3f}x golden area, at the "
        f"{config.WRONG_PART_AREA_HI} high bound")
    assert lo / config.WRONG_PART_AREA_LO > 1.05, (
        f"only {lo / config.WRONG_PART_AREA_LO:.2f}x of margin under the low "
        f"bound (worst good {lo:.3f}); LED ageing eats this monotonically")


# ------------------------------------------------- why area and not the bbox
def test_area_is_used_because_the_bounding_box_is_not_robust(brd, golden):
    """The measurement that chose the feature.

    A bounding box is a maximum-extent statistic: one speck of noise that
    survives the morphological open and touches the body moves it, and a body
    that is not axis-aligned inflates it by the rotation alone. Area is a bulk
    statistic and is invariant to rotation. Both effects are measured here
    rather than asserted, because the second one is what would have made a
    bbox-based test flag every skewed part on the board as a wrong part.
    """
    bbox_dev, area_dev = [], []
    for seed in (301, 302, 303, 7100):
        stack = synth.render(brd, opts=synth.RenderOpts(seed=seed))
        for c in brd.components:
            g = golden.packages[c.package]
            region, _ = rules.roi.extract_region(stack, c, c.pkg)
            b = rules.body_blob(region, g.body_spec_max, g.body_dif_min)
            bbox_dev.append(abs(b.width_mm / g.body_width_mm - 1.0))
            area_dev.append(abs(b.area_px / g.body_area_px - 1.0))
    assert max(area_dev) < max(bbox_dev), (
        f"area wanders {max(area_dev):.3f} against the bounding box's "
        f"{max(bbox_dev):.3f}; the reason for preferring area no longer holds")

    # rotation: the bbox of a skewed body grows, its area does not
    grew, held = [], []
    for refdes in ("R3", "R5", "D1"):
        c = brd.by_refdes(refdes)
        g = golden.packages[c.package]
        stack = render_with(brd, {refdes: synth.Defect("skew", dtheta=16.0)})
        region, _ = rules.roi.extract_region(stack, c, c.pkg)
        b = rules.body_blob(region, g.body_spec_max, g.body_dif_min)
        grew.append(b.width_mm / g.body_width_mm)
        held.append(b.area_px / g.body_area_px)
    assert min(grew) > 1.25, (
        f"a 16 degree skew only inflated the bounding box to {min(grew):.2f}x; "
        "if that is no longer true the argument for area is weaker than stated")
    assert max(held) < config.WRONG_PART_AREA_HI, (
        f"a skewed part measured {max(held):.2f}x golden AREA, which would "
        "report it as a wrong part")


@pytest.mark.parametrize("kind,kwargs", [
    ("offset", {"dx": 0.18}),
    ("offset", {"dx": 0.10, "dy": 0.10}),
    ("skew", {"dtheta": 16.0}),
    ("insufficient", {"pad": 0}),
    ("excess", {"pad": 0}),
])
def test_other_defects_are_not_relabelled_wrong_part(brd, golden, kind, kwargs):
    """A wrong-part call sends someone to the stores to check a reel. Every
    other defect on the board has to keep its own name."""
    for refdes in ("R1", "R3", "R5", "D1", "C1"):
        m = rules.measure(render_with(brd, {refdes: synth.Defect(kind, **kwargs)}),
                          brd.by_refdes(refdes), golden)
        assert rule_verdict(m, golden).reason != "wrong_part", (
            f"{refdes} with an injected {kind} was reported as a wrong part "
            f"(area {m.body_area_ratio:.2f}x golden)")


def test_a_tombstone_keeps_its_own_name(brd, golden):
    """Both faults make the body measure smaller than golden, and the tombstone
    test runs first on purpose: standing on one land is by far the more common
    fault, and it has a specific signature -- the body spans its own WIDTH.
    Letting the wrong-part test claim it would send an operator to the stores
    for a reel that is perfectly correct."""
    for refdes in ("R1", "R3", "R5", "C1"):
        m = rules.measure(render_with(brd, {refdes: synth.Defect("tombstone", pad=0)}),
                          brd.by_refdes(refdes), golden)
        v = rule_verdict(m, golden)
        assert v.reason == "tombstone", (
            f"{refdes} tombstone reported as {v.reason!r} "
            f"(area {m.body_area_ratio:.2f}x, aspect {m.body_aspect:.2f})")


def test_a_clipped_body_is_not_called_wrong_part(brd, golden):
    """A blob touching the crop edge has an extent set by the crop, not by the
    part, so its area is not evidence about what was fitted.

    Exercised directly rather than through the renderer, because on this board
    the case never reaches the wrong-part test: a component displaced far enough
    to clip its own crop has also put its solder in the inter-pad gap, and the
    bridge test claims it at step 1. Measured over a displacement sweep of five
    packages, all 32 clipped-and-shrunken readings came back 'bridge'.

    That makes the guard defence in depth rather than a load-bearing gate on
    this geometry -- and it is worth keeping as one, because the interception is
    a coincidence of chip land patterns rather than a rule. Constructing the
    measurement directly is what keeps it honest: the guard is proven, and the
    fact that nothing currently reaches it is stated rather than implied.
    """
    comp = brd.by_refdes("R3")
    m = rules.measure(render_with(brd, {}), comp, golden)
    assert rule_verdict(m, golden).reason == "ok", "starting from a good part"

    # a body measuring well under the low bound, its blob running off the crop
    m.body_area_ratio = config.WRONG_PART_AREA_LO * 0.5
    m.body_clipped = 1.0
    assert rule_verdict(m, golden).reason != "wrong_part", (
        f"a blob clipped by the crop edge, measuring "
        f"{m.body_area_ratio:.2f}x golden area, was reported as a wrong part; "
        "its extent is the crop boundary, not the part")

    # the same reading with the blob wholly inside the crop IS a wrong part,
    # so the guard is what made the difference and not the area
    m.body_clipped = 0.0
    assert rule_verdict(m, golden).reason == "wrong_part", (
        "the unclipped case is not caught either, so the test above proves "
        "nothing about the clipping guard")


def test_gross_displacement_never_reads_as_a_wrong_part(brd, golden):
    """The end-to-end half of the guard above: whatever intercepts a badly
    displaced part, it must not be the wrong-part test. Sending someone to the
    stores to check a reel because a part slid is a wasted trip."""
    wrong = []
    for refdes in ("R1", "R3", "R5", "D1", "C1"):
        for dx in (0.5, 0.9, 1.3, 2.0):
            m = rules.measure(render_with(brd, {refdes: synth.Defect("offset", dx=dx)}),
                              brd.by_refdes(refdes), golden)
            v = rule_verdict(m, golden)
            assert v.failed, f"{refdes} displaced {dx} mm passed inspection"
            if v.reason == "wrong_part":
                wrong.append((refdes, dx, round(m.body_area_ratio, 3),
                              bool(m.body_clipped)))
    assert not wrong, f"displacement reported as a wrong part: {wrong}"


# ----------------------------------------------------------------- multilead
@pytest.fixture(scope="module")
def leaded():
    comps = [
        B.Component("U1", "QFP-32",   x=12.0, y=12.0, rot=0.0),
        B.Component("U2", "SOIC-8",   x=26.0, y=8.0,  rot=0.0),
        B.Component("Q1", "SOT-23-6", x=34.0, y=8.0,  rot=0.0),
    ]
    brd = B.Board("LEADED", 40.0, 26.0, comps, [(2.0, 2.0), (38.0, 2.0), (38.0, 24.0)])
    stacks = [synth.render(brd, opts=synth.RenderOpts(seed=500 + i)) for i in range(5)]
    return brd, rules.fit_golden(stacks, brd)


@pytest.mark.parametrize("refdes,package,fitted", [
    ("U1", "QFP-32", "SOIC-8"),
    ("U2", "SOIC-8", "SOT-23-6"),
    ("Q1", "SOT-23-6", "0603"),
])
def test_substitutions_are_caught_on_leaded_packages(leaded, refdes, package, fitted):
    """The aspect test that guards tombstoning is gated on two terminals, so a
    square QFP body cannot reach it. Nothing gates this one, and a square body
    is exactly the case where an aspect ratio says nothing at all -- which is
    the second reason it is scored on area."""
    brd, golden = leaded
    stack = synth.render(brd, defects={refdes: synth.Defect("wrong_part",
                                                            substitute=fitted)},
                         opts=synth.RenderOpts(seed=11, place_sigma_um=0.0))
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    v = rule_verdict(m, golden)
    assert v.reason == "wrong_part", (
        f"a {fitted} body on a {package} footprint reported as {v.reason!r} "
        f"(area {m.body_area_ratio:.2f}x golden)")


def test_no_false_wrong_part_calls_on_good_leaded_parts(leaded):
    brd, golden = leaded
    flagged = []
    for seed in (911, 912, 913, 914):
        stack = synth.render(brd, opts=synth.RenderOpts(seed=seed))
        stack, _, _ = pipeline.align_board(stack, brd)
        for c in brd.components:
            m = rules.measure(stack, c, golden)
            if rule_verdict(m, golden).reason == "wrong_part":
                flagged.append((c.refdes, round(m.body_area_ratio, 3)))
    assert not flagged, f"false 'wrong_part' calls on good leaded parts: {flagged}"


# ------------------------------------- illumination robustness, added 10 Sep
def _worst_ratio_and_calls(brd, golden, gain):
    st = synth.render(brd, opts=synth.RenderOpts(seed=7, gain=gain))
    ratios, calls = [], 0
    for c in brd.components:
        m = rules.measure(st, c, golden)
        if not np.isfinite(m.body_area_ratio):
            continue
        ratios.append(m.body_area_ratio)
        if not (config.WRONG_PART_AREA_LO <= m.body_area_ratio
                <= config.WRONG_PART_AREA_HI):
            calls += 1
    return min(ratios), calls


@pytest.fixture(scope="module")
def dim_board():
    brd = B.random_board(30.0, 22.0, seed=1)
    golden = rules.fit_golden(
        [synth.render(brd, opts=synth.RenderOpts(seed=100 + i)) for i in range(6)], brd)
    return brd, golden


@pytest.mark.parametrize("gain", [0.75, 0.84, 0.92, 1.00, 1.15, 1.30])
def test_the_body_measures_the_same_under_changed_illumination(dim_board, gain):
    """The segmentation thresholds are absolute levels learned at teach, so
    anything that changes overall brightness moves the image away from them and
    erodes the blob edge. The body then measures smaller than it is and a good
    part reads as a substitution.

    The defence is the same one the fillet test gets: make the measurement care
    about the ratio the picture shows rather than the absolute number it was
    taught. The crop's own bare substrate is the reference, and teach already
    measured it.
    """
    brd, golden = dim_board
    worst, calls = _worst_ratio_and_calls(brd, golden, gain)
    assert calls == 0, (
        f"LED gain {gain}: {calls} good part(s) read as the wrong part, worst "
        f"area ratio {worst:.3f} against a lower bound of "
        f"{config.WRONG_PART_AREA_LO}")
    assert worst > 0.9, (
        f"LED gain {gain}: worst area ratio {worst:.3f} -- the rescaling is "
        f"holding the verdict but the measurement is still drifting")


def test_without_the_rescaling_a_modest_dim_breaks_the_test(dim_board):
    """The failure this defends against, measured rather than asserted. Pinning
    it keeps the rescaling justified: if this ever stops failing, the absolute
    thresholds are sufficient again and the extra machinery can go.

    Measured: an 8% dim produced 2 false calls and a 16% dim produced 24 of a
    54-component board -- effectively every part.
    """
    brd, golden = dim_board
    lo, hi = config.BODY_GAIN_MIN, config.BODY_GAIN_MAX
    config.BODY_GAIN_MIN = config.BODY_GAIN_MAX = 1.0       # pin -> no rescaling
    try:
        _, at_92 = _worst_ratio_and_calls(brd, golden, 0.92)
        _, at_84 = _worst_ratio_and_calls(brd, golden, 0.84)
    finally:
        config.BODY_GAIN_MIN, config.BODY_GAIN_MAX = lo, hi

    assert at_92 > 0, ("an 8% dim no longer breaks the unrescaled test, so the "
                       "rescaling can no longer be justified from measurement")
    assert at_84 > 10, (f"a 16% dim produced only {at_84} false calls unrescaled; "
                        f"this test is not reproducing the failure it describes")


def test_the_rescaling_is_bounded(dim_board):
    """A crop that is not merely dim -- a part over a copper pour, a crop half
    off the image -- must be measured rather than accommodated."""
    assert config.BODY_GAIN_MIN < 1.0 < config.BODY_GAIN_MAX
    assert config.BODY_GAIN_MIN >= 0.25 and config.BODY_GAIN_MAX <= 4.0, (
        "the gain bounds are wide enough to normalise away a genuinely wrong crop")
