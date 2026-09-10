"""The scan raster, and which tile owns which component.

The motion this describes does not exist yet -- the prototype renders a whole
board as one frame. That is exactly why the arithmetic is worth pinning now:
tile ownership is decided from CAD, not from pixels, so it can be got right
before a stage exists, and getting it wrong later is silent. A crop that runs
off its tile is filled by edge replication, and the lands in that strip read as
bare board: a permanent invisible defect on one component.
"""
from __future__ import annotations

import math

import pytest

from aoi import board as B
from aoi import config, packages, tiling


# ------------------------------------------------------------------ the raster
def test_the_raster_matches_the_number_the_spec_designed_around():
    """Section 2: a 150 mm board is covered by 4 x 5 tiles of 49 x 36 mm, four
    channels each -- 20 positions, 80 frames, about 5.8 seconds. If this stops
    agreeing, either the sampling pitch or the sensor changed, and the cycle
    time, the data volume and the BOM all move with it.
    """
    tiles = tiling.plan(150.0, 150.0)
    assert len({t.col for t in tiles}) == 4
    assert len({t.row for t in tiles}) == 5
    assert len(tiles) == 20
    assert len(tiles) * len(config.CHANNELS) == 80


def test_a_tile_is_the_sensor_at_the_specified_sampling():
    w, h = tiling.tile_size_mm()
    assert w == pytest.approx(49.15, abs=0.02)
    assert h == pytest.approx(36.0, abs=0.02)


def test_a_board_that_fits_one_exposure_gets_one_tile():
    """The degenerate case, and the one the prototype actually renders. It has
    to fall out of the same function rather than being a special path."""
    tiles = tiling.plan(40.0, 30.0)
    assert len(tiles) == 1
    assert (tiles[0].cx, tiles[0].cy) == (20.0, 15.0)


def test_tiles_cover_the_whole_board():
    for w, h in ((150.0, 150.0), (100.0, 60.0), (72.0, 54.0), (49.0, 36.0)):
        tiles = tiling.plan(w, h)
        xs = [t.bounds for t in tiles]
        assert min(b[0] for b in xs) <= 0.0 + 1e-9, f"{w}x{h}: left edge uncovered"
        assert min(b[1] for b in xs) <= 0.0 + 1e-9, f"{w}x{h}: top edge uncovered"
        assert max(b[2] for b in xs) >= w - 1e-9, f"{w}x{h}: right edge uncovered"
        assert max(b[3] for b in xs) >= h - 1e-9, f"{w}x{h}: bottom edge uncovered"


def test_neighbouring_tiles_overlap_by_at_least_the_specified_amount():
    """The overlap is what lets a cheap stage be good enough -- +-100 um of
    positioning error, absorbed rather than accumulated. It is not for
    stitching; nothing is ever stitched."""
    tw, th = tiling.tile_size_mm()
    tiles = tiling.plan(150.0, 150.0)
    cols = sorted({t.cx for t in tiles})
    rows = sorted({t.cy for t in tiles})
    for a, b in zip(cols, cols[1:]):
        assert (tw - (b - a)) / tw >= tiling.DEFAULT_OVERLAP - 1e-9
    for a, b in zip(rows, rows[1:]):
        assert (th - (b - a)) / th >= tiling.DEFAULT_OVERLAP - 1e-9


def test_the_unused_margin_is_shared_between_both_edges():
    """Packed from one corner, the whole slack piles up at the far edge and the
    last row of tiles hangs off the board for no reason."""
    tiles = tiling.plan(100.0, 60.0)
    left = min(t.bounds[0] for t in tiles)
    right = max(t.bounds[2] for t in tiles) - 100.0
    assert left == pytest.approx(-right, abs=0.01)


def test_an_impossible_overlap_is_refused():
    with pytest.raises(ValueError):
        tiling.plan(150.0, 150.0, overlap=1.0)
    with pytest.raises(ValueError):
        tiling.plan(150.0, 150.0, overlap=-0.1)


# -------------------------------------------------------------- tile ownership
def test_every_component_gets_a_tile_that_holds_its_whole_crop():
    """Not its centre -- its whole crop. This is the check that has to exist
    before a stage does."""
    brd = B.random_board(140.0, 140.0, seed=3, pitch=6.0)
    tiles = tiling.plan(brd.width, brd.height)
    owner, orphans = tiling.assign(brd, tiles)
    assert not orphans, f"{len(orphans)} components no tile can see: {orphans[:5]}"
    for comp in brd.components:
        box = tiling.component_box(comp)
        assert tiles[owner[comp.refdes]].contains(*box), (
            f"{comp.refdes} assigned a tile that clips its crop")


def test_the_crop_box_holds_at_any_rotation():
    """A crop is extracted de-rotated, so a part at 45 degrees sweeps a square of
    its own diagonal. Sizing the box from the nominal axes would put the corner
    of a QFP outside the box that was checked, which is the whole failure."""
    for rot in (0.0, 45.0, 90.0, 137.0):
        comp = B.Component("U1", "QFP-32", x=50.0, y=50.0, rot=rot)
        x0, y0, x1, y1 = tiling.component_box(comp)
        side = max(packages.get("QFP-32").footprint(0.35))
        assert (x1 - x0) == pytest.approx(side)
        assert (y1 - y0) == pytest.approx(side)


def test_a_part_that_no_tile_can_hold_is_reported_not_mis_assigned():
    """Returned rather than raised: the rest of the board is still inspectable.
    But it must not be silently given a tile that clips it."""
    big = B.Component("U1", "QFP-32", x=1.0, y=1.0)     # crop runs off the board
    brd = B.Board("EDGE", 60.0, 40.0, [big], [])
    owner, orphans = tiling.assign(brd)
    assert orphans == ["U1"]
    assert "U1" not in owner


def test_the_nearest_tile_wins_where_several_would_fit():
    """The overlap makes ties common, and the choice is not arbitrary: the lens
    perspective error grows with distance from the centre of the frame, so the
    nearest tile measures the part most accurately -- and its centre is what
    gets handed to `measure` as the optical axis."""
    brd = B.random_board(140.0, 140.0, seed=4, pitch=7.0)
    tiles = tiling.plan(brd.width, brd.height)
    owner, orphans = tiling.assign(brd, tiles)
    for comp in brd.components:
        if comp.refdes in orphans:
            continue
        box = tiling.component_box(comp)
        fits = [t for t in tiles if t.contains(*box)]
        best = min(fits, key=lambda t: math.hypot(comp.x - t.cx, comp.y - t.cy))
        assert owner[comp.refdes] == best.index


def test_the_optical_axis_is_the_centre_of_the_owning_tile():
    brd = B.random_board(140.0, 140.0, seed=5, pitch=7.0)
    tiles = tiling.plan(brd.width, brd.height)
    owner, _ = tiling.assign(brd, tiles)
    for comp in brd.components[:12]:
        axis = tiling.optical_axis_for(comp, brd, tiles, owner)
        t = tiles[owner[comp.refdes]]
        assert axis == (t.cx, t.cy)


def test_an_unplaceable_part_still_gets_an_axis_rather_than_crashing():
    big = B.Component("U1", "QFP-32", x=1.0, y=1.0)
    brd = B.Board("EDGE", 60.0, 40.0, [big], [])
    assert tiling.optical_axis_for(big, brd) == (30.0, 20.0)


def test_tile_assignment_keeps_a_part_well_inside_its_frame():
    """What ownership buys the measurement. Every part should sit comfortably
    inside its tile, not against the edge -- which also bounds how large the
    lens perspective correction ever has to be."""
    brd = B.random_board(140.0, 140.0, seed=6, pitch=7.0)
    tiles = tiling.plan(brd.width, brd.height)
    owner, orphans = tiling.assign(brd, tiles)
    tw, th = tiling.tile_size_mm()
    worst = 0.0
    for comp in brd.components:
        if comp.refdes in orphans:
            continue
        t = tiles[owner[comp.refdes]]
        worst = max(worst, math.hypot(comp.x - t.cx, comp.y - t.cy))
    assert worst <= math.hypot(tw, th) / 2.0 + 1e-6, (
        f"a component sits {worst:.1f} mm from its tile centre, outside the "
        f"tile's own half-diagonal")


def test_single_frame_ownership_matches_what_the_prototype_renders():
    """The prototype images a whole 40 x 30 board in one frame, so every part
    belongs to that one tile and the optical axis is the board centre -- which
    is exactly what `pipeline.inspect` uses today. The two must agree, or
    turning the stage on would silently move every axis."""
    brd = B.random_board(40.0, 30.0, seed=1)
    owner, orphans = tiling.assign(brd)
    assert not orphans
    assert set(owner.values()) == {0}
    for comp in brd.components[:5]:
        assert tiling.optical_axis_for(comp, brd) == (brd.width / 2.0,
                                                      brd.height / 2.0)
