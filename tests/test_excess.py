"""Excess solder: the other side of the fillet test.

Not the same defect as a heavy fillet, and not reachable by relaxing the
one-sided Mahalanobis guard. Solder wets outward only as far as flux and land
allow; volume past that piles up, and the surface inverts from a concave ramp
-- shallow at the toe, steep at the termination -- into a convex cap, flat over
its apex and steep only at the rim.

That inversion costs the mid-slope face, so a balled joint's LOW band collapses
exactly as a starved joint's does. Before this test existed every one of them
came back as "insufficient": the opposite fault, and an operator sent to add
paste to a joint that already carries too much.

What separates them is where the solder is. A starved joint leaves its own land
bare and puts nothing outside it. A bulged one spills onto substrate that CAD
says carries no copper.
"""
from __future__ import annotations

import numpy as np
import pytest

from aoi import board as B
from aoi import config, packages, pipeline, roi, rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with

PACKAGES = ["0402", "0603", "0805", "SOD-123", "TANT-A"]
REFDES = {"0402": "R1", "0603": "R3", "0805": "R5",
          "SOD-123": "D1", "TANT-A": "C1"}


# ------------------------------------------------------------------- geometry
@pytest.mark.parametrize("name", sorted(packages.LIBRARY))
def test_the_spill_strip_fits_inside_the_inspection_crop(name):
    """The failure this guards against is silent, which is why it is pinned.

    The strip looks just outside each land, and on every package in the library
    the outermost land IS the package extent -- so the only room it has is the
    crop's own margin. A strip wider than that is clipped by `local_box_px`
    against the crop bounds, and a clipped strip does not raise or look wrong:
    it measures the wrong region, or none, and the excess test quietly stops
    working for that package.

    That is exactly what happened at a first attempt. The window was grown 1.6x
    about the fillet box, which put its outer edge 0.60 mm past the land on a
    QFP-32 against 0.35 mm of crop margin, and the excess signal there read
    x1.01 while every chip package read cleanly.
    """
    pkg = packages.get(name)
    hx, hy = pkg.extent()
    fl, fw = pkg.footprint(0.35)
    for i, (centre, half) in enumerate(rules.spill_boxes(pkg)):
        assert abs(centre[0]) + half[0] <= fl / 2.0 + 1e-9, (
            f"{name} land {i} spill strip reaches {abs(centre[0]) + half[0]:.3f} mm "
            f"along x, past the {fl / 2.0:.3f} mm crop half-width")
        assert abs(centre[1]) + half[1] <= fw / 2.0 + 1e-9, (
            f"{name} land {i} spill strip reaches {abs(centre[1]) + half[1]:.3f} mm "
            f"across, past the {fw / 2.0:.3f} mm crop half-height")


@pytest.mark.parametrize("name", sorted(packages.LIBRARY))
def test_the_spill_strip_sits_outside_the_land_not_on_it(name):
    """It measures solder where there is no copper. Overlapping the land would
    put a large constant under the measurement and bury the signal in it."""
    pkg = packages.get(name)
    for i, (l, (centre, half)) in enumerate(zip(pkg.lands(), rules.spill_boxes(pkg))):
        if abs(l.ax) > abs(l.ay):
            inner = centre[0] + l.ax * half[0]        # edge nearest the body
            edge = l.cx - l.ax * l.hx                 # the land's outer edge
        else:
            inner = centre[1] + l.ay * half[1]
            edge = l.cy - l.ay * l.hy
        assert inner == pytest.approx(edge, abs=1e-9), (
            f"{name} land {i} spill strip starts at {inner:.3f} mm, not at the "
            f"land's outer edge {edge:.3f} mm")


# ----------------------------------------------------------------- separation
@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("pad", [0, 1])
def test_a_balled_joint_is_caught_on_either_pad(brd, golden, package, pad):
    refdes = REFDES[package]
    m = rules.measure(render_with(brd, {refdes: synth.Defect("excess", pad=pad)}),
                      brd.by_refdes(refdes), golden)
    v = rule_verdict(m, golden)
    assert v.reason == "excess", (
        f"{package} pad {pad} balled joint reported as {v.reason!r} "
        f"(spill {m.spill_worst:+.3f}, band distance {m.band_maha:.1f})")


def test_a_balled_joint_would_otherwise_be_called_insufficient(brd, golden):
    """The regression this test exists for, stated as the measurement.

    A cap's flat apex grows at the expense of the mid-slope face, so the low
    band falls -- which satisfies the insufficient test's own condition. The
    one-sided guard on that test was written for a fillet that is uniformly
    heavier and does not catch this shape at all.
    """
    comp = brd.by_refdes("R3")
    g = golden.packages["0603"]
    m = rules.measure(render_with(brd, {"R3": synth.Defect("excess", pad=0)}),
                      comp, golden)

    assert m.bands[0, 0] < g.band_mean[0, 0], (
        "this joint no longer loses its low band, so it is not exercising the "
        f"overlap: {m.bands[0, 0]:.3f} vs golden {g.band_mean[0, 0]:.3f}")
    assert rules.band_distance(m.bands, g)[0] > config.BAND_MAHALANOBIS_FAIL, (
        "the balled joint does not trip the fillet gate, so nothing would have "
        "mislabelled it")
    assert rule_verdict(m, golden).reason == "excess"


def test_a_starved_joint_puts_nothing_outside_its_land(brd, golden):
    """Why the two are separable at all.

    Starvation is a shortfall INSIDE the land. It exposes copper, which is just
    as specular as solder, so measuring brightness or metal area cannot see it
    -- but it adds nothing outside, and that is the whole discriminator.
    """
    for package in PACKAGES:
        refdes = REFDES[package]
        m = rules.measure(render_with(brd, {refdes: synth.Defect("insufficient", pad=0)}),
                          brd.by_refdes(refdes), golden)
        assert m.spill_worst < config.EXCESS_SPILL_FRACTION, (
            f"{package} starved joint reads {m.spill_worst:+.3f} outside its "
            f"land, at the {config.EXCESS_SPILL_FRACTION} excess threshold")
        assert rule_verdict(m, golden).reason == "insufficient"


def test_good_and_balled_joints_are_far_apart(brd, golden):
    """The margin the threshold sits in, sampled the way the machine will meet
    it: placement scatter, vignetting and LED output all moving."""
    good = []
    for seed, vig, gain in [(301, 0.06, 1.00), (302, 0.06, 1.00), (303, 0.06, 1.00),
                            (7100, 0.12, 1.00), (7101, 0.06, 0.92),
                            (7102, 0.12, 0.95), (7103, 0.02, 1.06)]:
        stack = synth.render(brd, opts=synth.RenderOpts(
            seed=seed, vignette=vig, gain=gain))
        stack, _, _ = pipeline.align_board(stack, brd)
        good.extend(rules.measure(stack, c, golden).spill_worst for c in brd.components)

    balled = []
    for package in PACKAGES:
        for pad in (0, 1):
            m = rules.measure(
                render_with(brd, {REFDES[package]: synth.Defect("excess", pad=pad)}),
                brd.by_refdes(REFDES[package]), golden)
            balled.append(m.spill_worst)

    worst_good, weakest = max(good), min(balled)
    assert worst_good < config.EXCESS_SPILL_FRACTION, (
        f"a good joint reads {worst_good:+.3f} outside its land, at the "
        f"{config.EXCESS_SPILL_FRACTION} threshold")
    assert weakest > config.EXCESS_SPILL_FRACTION, (
        f"the weakest balled joint only reaches {weakest:+.3f}")
    assert weakest / max(worst_good, 1e-9) > 3.0, (
        f"only {weakest / worst_good:.1f}x between the worst good joint "
        f"({worst_good:+.3f}) and the weakest balled one ({weakest:+.3f})")


@pytest.mark.parametrize("vignette,gain,label", [
    (0.12, 1.00, "vignette 0.12"),
    (0.06, 0.92, "LED output down 8%"),
    (0.02, 1.06, "LED output up 6%"),
])
def test_illumination_drift_does_not_manufacture_excess_calls(brd, golden, vignette,
                                                              gain, label):
    flagged = []
    for i in range(3):
        stack = synth.render(brd, opts=synth.RenderOpts(
            seed=7200 + i, vignette=vignette, gain=gain))
        stack, _, _ = pipeline.align_board(stack, brd)
        for c in brd.components:
            m = rules.measure(stack, c, golden)
            if rule_verdict(m, golden).reason == "excess":
                flagged.append((c.refdes, c.package, round(m.spill_worst, 3)))
    assert not flagged, f"{len(flagged)} false 'excess' calls under {label}: {flagged[:6]}"


# ------------------------------------------------- what golden has to record
def test_teach_records_the_spill_a_good_board_already_shows(golden):
    """Subtracted rather than assumed to be zero.

    The golden boards come off the same fab with the same etch, so an oversized
    land or a copper pour reaching near the footprint is present in golden too.
    Recording it is what stops board fabrication from reading as excess solder;
    only a change from golden counts.
    """
    for name, g in golden.packages.items():
        n = packages.get(name).n_pads
        assert g.spill_mean.shape == (n,), \
            f"{name} spill_mean is {g.spill_mean.shape}, expected ({n},)"
        assert np.all(g.spill_mean < 0.10), (
            f"{name} shows {g.spill_mean.max():.3f} solder outside its land on a "
            "GOOD board; either the strip overlaps the land or the package "
            "geometry is wrong")


def test_golden_spill_is_subtracted_not_assumed_zero(brd, golden):
    """A board feature outside the land must cancel, not accumulate.

    On synthetic boards golden's spill is ~0 for every package, so on this data
    the subtraction is a no-op and no end-to-end test can distinguish it from
    dropping the term. It is not a no-op on a real board: an oversized land
    etch, a via-in-pad tab or a copper pour reaching near the footprint all put
    standing metal in this strip on EVERY board including the golden ones. Take
    the subtraction out and that becomes a permanent excess call on every
    instance of the package.

    So the mechanism is pinned directly: raise golden's recorded spill and the
    same image must stop reading as excess.
    """
    from dataclasses import replace

    comp = brd.by_refdes("R3")
    stack = render_with(brd, {"R3": synth.Defect("excess", pad=0)})
    raw = rules.measure(stack, comp, golden)
    assert rule_verdict(raw, golden).reason == "excess", "no defect to cancel"

    g = golden.packages[comp.package]
    raw_frac = raw.spill + np.asarray(g.spill_mean, np.float32)[:len(raw.spill)]

    # a golden that already shows exactly this much metal outside its lands
    taught = rules.Golden(packages=dict(golden.packages))
    taught.packages[comp.package] = replace(g, spill_mean=raw_frac)
    m = rules.measure(stack, comp, taught)

    assert m.spill_worst == pytest.approx(0.0, abs=1e-6), (
        f"golden's own spill did not cancel: {m.spill_worst:+.4f} left over")
    assert rule_verdict(m, taught).reason != "excess", (
        "a feature the golden boards show too was still reported as excess "
        "solder; the test is measuring the board, not the joint")


def test_a_segmentation_failure_cannot_flag_excess(brd, golden):
    """The spill strip is thresholded at `body_spec_max`, the same level the
    body segmentation uses as its ceiling. If that level stops separating
    anything, every pixel reads as metal and every land looks flooded.

    The pipeline already knows when that has happened: it is the case where
    segmentation finds no body but correlation overrules it. That disagreement
    is evidence about the threshold, not about the part, and every measurement
    resting on it is void for that component.
    """
    from dataclasses import replace

    comp = brd.by_refdes("R5")
    broken = rules.Golden(packages=dict(golden.packages))
    broken.packages[comp.package] = replace(golden.packages[comp.package],
                                            body_spec_max=0.0)
    m = rules.measure(render_with(brd, {}), comp, broken)

    assert m.spill_worst > config.EXCESS_SPILL_FRACTION, (
        "test is not exercising the guard: a collapsed threshold did not make "
        f"the spill strip read as flooded ({m.spill_worst:+.3f})")
    assert not rule_verdict(m, broken).failed, (
        "a total segmentation failure flagged a good part as excess solder; the "
        "measurement is not gated on the segmentation being trustworthy")
