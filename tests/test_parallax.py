"""Lens perspective: the correction that makes a telecentric lens unnecessary.

The lens is not telecentric, so it views a part standing above the board from an
angle and the part images further from the optical axis than it really sits.
Section 2 of the spec puts that at 91 um for a 1 mm part at the corner of a tile
-- 7.6 px, against a placement tolerance of 150 um -- and says the pipeline
corrects it from known package height rather than paying ~$1,500 for telecentric
optics.

Two halves have to exist together or neither is worth anything. The renderer has
to produce the displacement, or correcting for it is subtracting an error that
was never introduced; and the correction has to use the same geometry the
renderer did, or it doubles the error instead of removing it. Both are checked
here, and the sign is checked explicitly because getting it backwards is the
failure that still looks like a working pipeline.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aoi import board as B
from aoi import config, packages, pipeline, rules, synth

TILE_HALF_DIAGONAL_MM = math.hypot(
    config.TILE_PX[0] * config.MM_PER_PX / 2.0,
    config.TILE_PX[1] * config.MM_PER_PX / 2.0)


# --------------------------------------------------------------- the geometry
def test_the_model_reproduces_the_figure_the_spec_designed_around():
    """Section 2's focal-length table: a 1 mm part at the corner of a tile, seen
    through a 75 mm lens at 336 mm, is displaced 91 um -- 7.6 px at 12 um.

    That number is why a 75 mm lens was chosen over a 35 mm one (194 um) and why
    the residual is corrected in software rather than bought away.
    """
    shift = config.parallax_shift_mm(TILE_HALF_DIAGONAL_MM, 1.0)
    assert shift * 1000.0 == pytest.approx(91.0, abs=1.5), (
        f"{shift * 1000:.1f} um against the spec's 91 um")
    assert shift / config.MM_PER_PX == pytest.approx(7.6, abs=0.15)


def test_a_part_on_the_optical_axis_is_not_displaced():
    assert config.parallax_shift_mm(0.0, 1.6) == 0.0
    assert config.parallax_local_mm(10.0, 10.0, (10.0, 10.0), 1.6, 0.0) == (0.0, 0.0)


def test_a_flat_feature_is_not_displaced():
    """Lands, fillets and substrate are all at zero height. If this ever returns
    a shift, the correction will start moving things that never moved."""
    assert config.parallax_shift_mm(30.0, 0.0) == 0.0


def test_displacement_grows_with_height_and_with_distance_from_the_axis():
    heights = [config.parallax_shift_mm(20.0, h) for h in (0.35, 1.0, 1.6)]
    assert heights == sorted(heights) and heights[0] < heights[-1]
    radii = [config.parallax_shift_mm(r, 1.0) for r in (5.0, 15.0, 30.0)]
    assert radii == sorted(radii) and radii[0] < radii[-1]


def test_the_local_frame_conversion_preserves_magnitude():
    """Rotating into the component frame may not change how far it moved."""
    for rot in (0.0, 37.0, 90.0, 180.0, 271.0):
        lx, ly = config.parallax_local_mm(40.0, 25.0, (10.0, 10.0), 1.0, rot)
        assert math.hypot(lx, ly) == pytest.approx(
            config.parallax_shift_mm(math.hypot(30.0, 15.0), 1.0), rel=1e-9)


# ------------------------------------------------------------- the renderer
def _one_part_board(pkg: str, x: float, y: float, w=60.0, h=40.0):
    comps = [B.Component("P1", pkg, x=x, y=y, rot=0.0)]
    return B.Board("PAR", w, h, comps, [(2.0, 2.0), (w - 2.0, 2.0), (w - 2.0, h - 2.0)])


def _body_centroid_px(stack, brd):
    """Where the body actually landed, from the diffuse channel."""
    dif = stack[..., config.CHANNELS.index("diffuse")].astype(np.float32)
    comp = brd.components[0]
    g = rules.fit_golden([stack], brd)
    reg, _ = __import__("aoi.roi", fromlist=["x"]).extract_region(stack, comp, comp.pkg)
    blob = rules.body_blob(reg, g.packages[comp.package].body_spec_max,
                           g.packages[comp.package].body_dif_min)
    return blob


def test_the_renderer_displaces_a_tall_body_and_leaves_its_lands_alone():
    """Both halves in one picture. If the lands moved too, the whole crop would
    simply be offset and nothing downstream could tell -- it is the body moving
    relative to its own lands that the correction exists for."""
    axis = (30.0, 20.0)
    centre = _one_part_board("TANT-A", *axis)
    edge = _one_part_board("TANT-A", 55.0, 20.0)          # 25 mm off-axis

    st_c = synth.render(centre, opts=synth.RenderOpts(seed=3, place_sigma_um=0.0))
    st_e = synth.render(edge, opts=synth.RenderOpts(seed=3, place_sigma_um=0.0))

    b_c = _body_centroid_px(st_c, centre)
    b_e = _body_centroid_px(st_e, edge)
    assert b_c.found and b_e.found, "body segmentation failed; test cannot measure"

    expected = config.parallax_shift_mm(25.0, packages.get("TANT-A").body_h)
    moved = b_e.dx_mm - b_c.dx_mm
    assert moved == pytest.approx(expected, abs=0.02), (
        f"body moved {moved * 1000:.0f} um off-axis against a predicted "
        f"{expected * 1000:.0f} um -- the renderer is not modelling the lens")
    assert moved > 0, "the body must appear displaced OUTWARD, away from the axis"


# ------------------------------------------------------------ the correction
def _spread_board(pkg="TANT-A", w=60.0, h=40.0):
    """Four parts of one package at four different distances from the axis.

    This is the arrangement that makes parallax matter, and getting it wrong is
    instructive enough to record: with a SINGLE part, teach builds that
    package's template from that one component at that one radius, so the body
    sits at the same displaced place in the template and in the frame and the
    error cancels whether or not anything corrects it. Both readings come back
    zero and the test proves nothing.

    Real golden statistics are per PACKAGE and averaged over every instance of
    it on the board, which sit at different radii and therefore carry different
    displacements. That average is what smears, and that is what the correction
    has to undo.
    """
    cx, cy = w / 2.0, h / 2.0
    comps = [B.Component(f"C{i}", pkg, x=cx + dx, y=cy, rot=0.0)
             for i, dx in enumerate((0.0, 8.0, 17.0, 25.0))]
    return B.Board("SPREAD", w, h, comps,
                   [(2.0, 2.0), (w - 2.0, 2.0), (w - 2.0, h - 2.0)])


def _worst_offset(brd, axis, stacks, test, correct: bool, monkeypatch=None):
    """Worst placement reading over a defect-free board.

    The uncorrected baseline is produced by neutralising the CORRECTION only.
    The frames are rendered before this runs, so they keep the displacement the
    lens put there; what changes is whether teach and inspect know about it.
    Turning it off by passing no axis would not do: teach has to build its
    template the same way inspect measures, and both read the same function.
    """
    if correct:
        g = rules.fit_golden(stacks, brd, optical_axis_mm=axis)
        return max(math.hypot(m.dx_um, m.dy_um)
                   for m in (rules.measure(test, c, g, optical_axis_mm=axis)
                             for c in brd.components))
    monkeypatch.setattr(config, "parallax_local_mm",
                        lambda *a, **k: (0.0, 0.0))
    g = rules.fit_golden(stacks, brd, optical_axis_mm=axis)
    return max(math.hypot(m.dx_um, m.dy_um)
               for m in (rules.measure(test, c, g, optical_axis_mm=axis)
                         for c in brd.components))


def test_the_correction_removes_the_error_rather_than_doubling_it():
    """The sign check, stated as its own test because it is the failure mode.

    A correction applied the wrong way round leaves a pipeline that runs, tests
    that pass, and a placement reading twice as wrong as doing nothing. Compare
    corrected against uncorrected on the same frames: it has to go down.
    """
    brd = _spread_board()
    axis = (brd.width / 2.0, brd.height / 2.0)
    stacks = [synth.render(brd, opts=synth.RenderOpts(seed=400 + i,
                                                      place_sigma_um=0.0))
              for i in range(5)]
    test = synth.render(brd, opts=synth.RenderOpts(seed=777, place_sigma_um=0.0))

    corrected = _worst_offset(brd, axis, stacks, test, correct=True)
    with pytest.MonkeyPatch.context() as mp:
        uncorrected = _worst_offset(brd, axis, stacks, test, correct=False,
                                    monkeypatch=mp)

    assert uncorrected > 20.0, (
        f"uncorrected worst offset is only {uncorrected:.0f} um, so there is "
        f"barely an error to remove and this test proves nothing -- check that "
        f"the renderer is still modelling the lens")
    assert corrected < uncorrected, (
        f"corrected {corrected:.0f} um is not better than uncorrected "
        f"{uncorrected:.0f} um -- check the sign of the correction")
    assert corrected < 0.5 * uncorrected, (
        f"corrected {corrected:.0f} um against uncorrected {uncorrected:.0f} um "
        f"is less than half; the correction is only partly landing")


def test_placement_reads_the_same_wherever_the_part_sits_in_the_frame():
    """What the correction is FOR. A good part is a good part whether it sits
    under the lens or at the edge of the frame, and the number the machine
    reports has to say so -- otherwise a board's corners fail for being corners.
    """
    brd = _spread_board()
    axis = (brd.width / 2.0, brd.height / 2.0)
    stacks = [synth.render(brd, opts=synth.RenderOpts(seed=300 + i,
                                                      place_sigma_um=0.0))
              for i in range(5)]
    g = rules.fit_golden(stacks, brd, optical_axis_mm=axis)
    test = synth.render(brd, opts=synth.RenderOpts(seed=888, place_sigma_um=0.0))

    readings = {c.refdes: math.hypot(*(lambda m: (m.dx_um, m.dy_um))(
        rules.measure(test, c, g, optical_axis_mm=axis))) for c in brd.components}

    spread = max(readings.values()) - min(readings.values())
    assert spread < 0.25 * config.MAX_OFFSET_UM, (
        f"a good part reads {({k: round(v) for k, v in readings.items()})} "
        f"depending only on where it sits in the frame; that spread is "
        f"{spread:.0f} um against a {config.MAX_OFFSET_UM:.0f} um tolerance")
    for refdes, v in readings.items():
        assert v < 0.5 * config.MAX_OFFSET_UM, (
            f"a good part at {refdes} reads {v:.0f} um, over half the tolerance "
            f"before any real placement error exists")


def test_a_short_part_needs_almost_no_correction():
    """Height is what drives it. An 0402 is 0.35 mm tall and barely moves, which
    is why the effect went unnoticed on a board of chips."""
    tall = config.parallax_shift_mm(25.0, packages.get("TANT-A").body_h)
    short = config.parallax_shift_mm(25.0, packages.get("0402").body_h)
    assert short < 0.25 * tall
    assert short / config.MM_PER_PX < 3.0, "an 0402 should move under 3 px"


# ------------------------------------------------- teach and inspect must agree
def test_teach_records_the_axis_so_inspect_cannot_disagree_with_it():
    """Correcting one side and not the other is worse than correcting neither:
    the fillet windows and the body template are both built relative to where
    the lens puts a part. The axis rides on the golden model so a caller cannot
    accidentally use one at teach and another at inspect."""
    brd = _one_part_board("TANT-A", 55.0, 20.0)
    stacks = [synth.render(brd, opts=synth.RenderOpts(seed=600 + i)) for i in range(3)]
    axis = (30.0, 20.0)
    g = rules.fit_golden(stacks, brd, optical_axis_mm=axis)
    assert g.optical_axis_mm == axis

    default = rules.fit_golden(stacks, brd)
    assert default.optical_axis_mm == (brd.width / 2.0, brd.height / 2.0), (
        "teach must default to the centre of the frame, which is what the "
        "prototype images in one shot")


def test_the_pipeline_passes_an_axis_end_to_end():
    """The wiring, checked where it actually runs."""
    brd = B.random_board(30.0, 22.0, seed=5)
    golden = rules.fit_golden(
        [synth.render(brd, opts=synth.RenderOpts(seed=700 + i)) for i in range(5)], brd)
    stack = synth.render(brd, opts=synth.RenderOpts(seed=9))
    from aoi import models
    res = pipeline.Inspector(golden, models.load_backend("none", None), None).inspect(
        stack, brd)
    assert res.measures, "no measurements produced"
    worst = max(math.hypot(m.dx_um, m.dy_um) for m in res.measures
                if np.isfinite(m.dx_um))
    assert worst < config.MAX_OFFSET_UM, (
        f"a defect-free board has a component reading {worst:.0f} um off "
        f"nominal, over the {config.MAX_OFFSET_UM:.0f} um tolerance")
