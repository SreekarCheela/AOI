"""Body localisation: where the body is on the crop, and how big it is.

The presence decision used to be a correlation score. NCC answers one question --
does this look like a good part -- and collapses every way of not being one into
a single low number. It cannot tell a body that is absent from a body that is
somewhere else, and two separate label bugs came from exactly that:

    missing vs displaced    the populations overlap in NCC. A missing part
                            reaches 0.572 of golden while a part displaced
                            500 um falls to 0.505, so no threshold sits between
                            them and gross displacement read as "missing".

    tombstone vs displaced  a tombstoned SOD-123 scores 0.81 and a TANT-A 0.77,
                            both above anything a gate could use without
                            swallowing good parts, so they read as "offset" or,
                            when the standing body flipped the polarity margin,
                            as "reversed".

Segmenting the body and measuring its position and extent answers the question
directly, and both bugs go away. These tests pin that.
"""
from __future__ import annotations

import numpy as np
import pytest

from aoi import config, packages, pipeline, rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with

ALL = ["0402", "0603", "0805", "SOD-123", "TANT-A"]
REFDES = {"0402": "R1", "0603": "R3", "0805": "R5",
          "SOD-123": "D1", "TANT-A": "C1"}


# ----------------------------------------------------------- the measurement
def test_teach_learns_a_segmentation_that_finds_the_real_body(golden):
    """The blob has to be the actual component, not an arbitrary bright region.

    Checked against the package library: the measured extent must match the
    datasheet body within a tolerance the crop margin can explain. If this
    drifts, the segmentation is finding something else and every test below is
    measuring the wrong object.
    """
    for name, g in golden.packages.items():
        pkg = packages.get(name)
        assert abs(g.body_len_mm - pkg.body_l) < 0.25, (
            f"{name} golden blob is {g.body_len_mm:.2f} mm long, datasheet body "
            f"is {pkg.body_l:.2f} mm")
        assert abs(g.body_width_mm - pkg.body_w) < 0.25, (
            f"{name} golden blob is {g.body_width_mm:.2f} mm wide, datasheet "
            f"body is {pkg.body_w:.2f} mm")


def test_segmentation_thresholds_sit_between_the_populations_they_split(golden):
    """The specular threshold must clear a polarity marking (which is part of
    the body and must not split it in two) and stay well under metal."""
    for name, g in golden.packages.items():
        assert 0.12 < g.body_spec_max < 0.40, (
            f"{name} specular threshold {g.body_spec_max:.3f} is outside the gap "
            "between a marking (~0.118) and copper or solder (~0.62)")
        assert 0.20 < g.body_dif_min < 0.30, (
            f"{name} diffuse threshold {g.body_dif_min:.3f} is outside the gap "
            "between substrate (~0.20) and body (~0.26)")


@pytest.mark.parametrize("package", ALL)
def test_a_good_part_measures_its_own_shape(brd, golden, package):
    comp = brd.by_refdes(REFDES[package])
    m = rules.measure(render_with(brd, {}), comp, golden)
    assert m.body_area_ratio > 0.85, \
        f"{package} good part segments to {m.body_area_ratio:.2f} of golden area"
    assert m.body_aspect > config.BODY_STANDING_ASPECT, \
        f"{package} good part reads aspect {m.body_aspect:.2f}, below the standing limit"


# ------------------------------------------------------- missing vs displaced
@pytest.mark.parametrize("package", ALL)
def test_a_missing_part_leaves_no_body_at_all(brd, golden, package):
    refdes = REFDES[package]
    m = rules.measure(render_with(brd, {refdes: synth.Defect("missing")}),
                      brd.by_refdes(refdes), golden)
    assert m.body_area_ratio < config.BODY_PRESENT_AREA
    assert rule_verdict(m, golden).reason == "missing"


@pytest.mark.parametrize("package", ["0603", "0805", "SOD-123", "TANT-A"])
def test_a_grossly_displaced_part_is_not_called_missing(brd, golden, package):
    """The first bug this measurement fixes. The part is still there, so there
    is still a body to find, whatever the correlation score says.

    0402 is excluded: at 500 um its body lands over the inter-pad gap and trips
    the bridge test, which runs first by design. Tracked separately.
    """
    refdes = REFDES[package]
    m = rules.measure(render_with(brd, {refdes: synth.Defect("offset", dx=0.50)}),
                      brd.by_refdes(refdes), golden)
    assert m.body_area_ratio > config.BODY_PRESENT_AREA, \
        f"{package} displaced 500 um segments to {m.body_area_ratio:.2f}, read as absent"
    assert rule_verdict(m, golden).reason == "offset"


# ----------------------------------------------------- tombstone vs displaced
@pytest.mark.parametrize("package", ALL)
@pytest.mark.parametrize("pad", [0, 1])
def test_every_tombstone_is_caught_on_either_pad(brd, golden, package, pad):
    """The second bug. Tall polarised packages used to come back as 'offset' or
    'reversed' because a standing body still correlates well and flips the
    polarity margin. Extent does not care about either."""
    refdes = REFDES[package]
    m = rules.measure(render_with(brd, {refdes: synth.Defect("tombstone", pad=pad)}),
                      brd.by_refdes(refdes), golden)
    assert rule_verdict(m, golden).reason == "tombstone", \
        f"{package} pad {pad} reads aspect {m.body_aspect:.2f}"


def test_a_standing_body_spans_its_own_width(brd, golden):
    """Why the aspect test needs no per-package tuning.

    A part standing on one land presents its width along the pad axis, whatever
    package it is, so the ratio lands at 1.0 for every one of them. Normalising
    against golden LENGTH instead would need a different threshold per package
    and would collide with parts clipped by the crop edge.
    """
    for package in ALL:
        refdes = REFDES[package]
        m = rules.measure(
            render_with(brd, {refdes: synth.Defect("tombstone", pad=0)}),
            brd.by_refdes(refdes), golden)
        assert m.body_aspect <= 1.10, (
            f"{package} tombstone spans {m.body_aspect:.2f}x its own width; the "
            "test only needs one threshold because this is always ~1.0")


def test_tombstones_and_flat_parts_do_not_overlap(brd, golden):
    """The margin the single threshold lives in, measured rather than assumed.

    Includes an 800 um displacement, which is the case that breaks the obvious
    alternative: normalised against golden length it measures 0.758 and collides
    with a tombstone, because the crop edge cut it short rather than the part
    standing up.
    """
    standing, flat = [], []
    for package in ALL:
        refdes = REFDES[package]
        for pad in (0, 1):
            m = rules.measure(
                render_with(brd, {refdes: synth.Defect("tombstone", pad=pad)}),
                brd.by_refdes(refdes), golden)
            standing.append(m.body_aspect)
        for dx in (0.0, 0.25, 0.50, 0.80):
            d = {refdes: synth.Defect("offset", dx=dx)} if dx else {}
            m = rules.measure(render_with(brd, d), brd.by_refdes(refdes), golden)
            flat.append(m.body_aspect)

    hi, lo = max(standing), min(flat)
    assert hi < config.BODY_STANDING_ASPECT < lo, (
        f"tombstones reach {hi:.2f} and flat parts fall to {lo:.2f}, which "
        f"straddles the {config.BODY_STANDING_ASPECT} threshold")
    assert lo / hi > 1.25, \
        f"only {lo / hi:.2f}x between the two populations ({hi:.2f} and {lo:.2f})"


# ------------------------------------------------------------- false calls
def test_segmentation_does_not_flag_good_boards(brd, golden):
    """The whole mechanism runs on every component of every board, so a
    segmentation that occasionally loses the body would be a false-call source
    rather than a fix."""
    flagged = []
    for seed in (6001, 6002, 6003, 6004, 6005):
        stack = synth.render(brd, opts=synth.RenderOpts(seed=seed))
        stack, _, _ = pipeline.align_board(stack, brd)
        for c in brd.components:
            m = rules.measure(stack, c, golden)
            v = rule_verdict(m, golden)
            if v.failed:
                flagged.append((c.refdes, v.reason, round(m.body_aspect, 2),
                                round(m.body_area_ratio, 2)))
    assert not flagged, f"{len(flagged)} false calls: {flagged[:8]}"


def test_a_segmentation_failure_cannot_flag_everything_missing(brd, golden):
    """The guard that makes this safe to deploy.

    Segmentation is the sharper instrument and also the one that can fail
    wholesale, if a real board's body and substrate sit closer in tone than the
    golden boards suggested. On its own, a segmentation that finds nothing would
    call every component missing. Correlation has to agree before that verdict
    is allowed.
    """
    from dataclasses import replace

    comp = brd.by_refdes("R5")
    m = rules.measure(render_with(brd, {}), comp, golden)
    broken = rules.Golden(packages=dict(golden.packages))
    # a threshold no real pixel can satisfy: the segmentation now finds nothing
    broken.packages[comp.package] = replace(golden.packages[comp.package],
                                            body_spec_max=0.0)
    m2 = rules.measure(render_with(brd, {}), comp, broken)

    assert m2.body_area_ratio < config.BODY_PRESENT_AREA, \
        "test did not actually break the segmentation"
    assert not rule_verdict(m2, broken).failed, (
        "a total segmentation failure flagged a good part as missing; the NCC "
        "second opinion is not gating the call")
