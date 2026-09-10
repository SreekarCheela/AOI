"""The scan raster: which camera position sees which component.

A 150 mm board cannot be photographed in one frame at 12 um/px -- that needs a
156 MP sensor, which costs more than the rest of the machine. So the head moves
and the board is covered in overlapping tiles, one 4096 x 3000 exposure each,
four channels per position. Section 2 of the spec sets that at 4 x 5 tiles for a
150 mm board, 80 frames, about 5.8 seconds.

None of that motion exists yet, and this module does not pretend otherwise: it
plans the raster and decides tile ownership, which is the part that has to be
right before a stage exists rather than after. The prototype renders a whole
board as one frame, which is the degenerate case of a single tile -- and `plan`
returns exactly that when the board fits, so nothing has to change twice.

Two things here are load-bearing and neither is obvious.

The overlap is not for stitching. Nothing is ever stitched: every tile is
registered against board features and every component is cropped from the tile
it belongs to. The overlap exists so the stage is allowed to be inaccurate --
+-100 um, per section 4 -- without a component ever landing in a seam.

And a component's whole CROP has to fit inside its tile, not just its centre. A
crop that runs off the frame is filled by edge replication, and the lands that
land in the replicated strip read as bare board: an invisible, permanent defect
on that one component. That is the same failure the board layout already guards
against at the image edge, and a tile boundary is an image edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import config
from .board import Board

# Section 2: 10% overlap between neighbouring tiles.
DEFAULT_OVERLAP = 0.10


@dataclass(frozen=True)
class Tile:
    """One camera position, in board mm."""
    index: int
    row: int
    col: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) in board mm."""
        return (self.cx - self.w / 2.0, self.cy - self.h / 2.0,
                self.cx + self.w / 2.0, self.cy + self.h / 2.0)

    def contains(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Whether a whole box lies inside this tile."""
        tx0, ty0, tx1, ty1 = self.bounds
        return tx0 <= x0 and ty0 <= y0 and x1 <= tx1 and y1 <= ty1


def tile_size_mm(mm_per_px: float = config.MM_PER_PX,
                 tile_px: tuple[int, int] = config.TILE_PX) -> tuple[float, float]:
    """What one exposure covers, in mm. 49.15 x 36.00 at the specified sampling."""
    return (tile_px[0] * mm_per_px, tile_px[1] * mm_per_px)


def plan(board_w: float, board_h: float, overlap: float = DEFAULT_OVERLAP,
         mm_per_px: float = config.MM_PER_PX,
         tile_px: tuple[int, int] = config.TILE_PX) -> list[Tile]:
    """Camera positions covering the board, row-major.

    Tiles are spread evenly rather than packed from one corner, so the unused
    margin is shared between both edges instead of piling up at the far one.
    A board that fits in a single exposure gets a single tile centred on it,
    which is what the prototype renders today.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    tw, th = tile_size_mm(mm_per_px, tile_px)

    def counts(span: float, tile: float) -> int:
        if span <= tile:
            return 1
        step = tile * (1.0 - overlap)
        return int(math.ceil((span - tile) / step)) + 1

    nx, ny = counts(board_w, tw), counts(board_h, th)
    xs = [board_w / 2.0] if nx == 1 else [
        tw / 2.0 + i * (board_w - tw) / (nx - 1) for i in range(nx)]
    ys = [board_h / 2.0] if ny == 1 else [
        th / 2.0 + j * (board_h - th) / (ny - 1) for j in range(ny)]

    return [Tile(index=j * nx + i, row=j, col=i, cx=x, cy=y, w=tw, h=th)
            for j, y in enumerate(ys) for i, x in enumerate(xs)]


def component_box(comp, margin: float = 0.35) -> tuple[float, float, float, float]:
    """The board-mm box a component's inspection crop occupies.

    Uses the LARGER footprint axis on both sides, so the box holds at any
    rotation. A crop is extracted de-rotated, and a part at 45 degrees sweeps a
    square of its own diagonal -- assuming otherwise would put the corner of a
    QFP outside the box that was checked.
    """
    half = max(comp.pkg.footprint(margin)) / 2.0
    return (comp.x - half, comp.y - half, comp.x + half, comp.y + half)


def assign(board: Board, tiles: list[Tile] | None = None,
           margin: float = 0.35) -> tuple[dict[str, int], list[str]]:
    """Give every component the tile that can actually see all of it.

    Returns (refdes -> tile index, refdes with no tile that fits).

    Where several tiles hold the whole crop -- which the overlap makes common --
    the one whose centre is nearest wins. That is not arbitrary: the lens
    perspective error grows with distance from the centre of the frame, so the
    nearest tile is the one that measures the part most accurately, and it is
    the tile centre that `rules.measure` is then given as the optical axis.

    The unplaceable list is returned rather than raised. One component that will
    not fit is a program problem for that component, and the rest of the board
    can still be inspected -- but it must be reported, because the alternative
    is a crop silently filled with replicated edge pixels reading as bare board.
    """
    tiles = tiles if tiles is not None else plan(board.width, board.height)
    owner: dict[str, int] = {}
    orphans: list[str] = []
    for comp in board.components:
        box = component_box(comp, margin)
        fits = [t for t in tiles if t.contains(*box)]
        if not fits:
            orphans.append(comp.refdes)
            continue
        owner[comp.refdes] = min(
            fits, key=lambda t: math.hypot(comp.x - t.cx, comp.y - t.cy)).index
    return owner, orphans


def optical_axis_for(comp, board: Board, tiles: list[Tile] | None = None,
                     owner: dict[str, int] | None = None,
                     margin: float = 0.35) -> tuple[float, float]:
    """The optical axis to measure this component against.

    The centre of the tile that images it. Falls back to the centre of the board
    when the component has no tile, which is also the answer for the
    single-frame case the prototype renders.
    """
    tiles = tiles if tiles is not None else plan(board.width, board.height)
    if owner is None:
        owner, _ = assign(board, tiles, margin)
    idx = owner.get(comp.refdes)
    if idx is None:
        return (board.width / 2.0, board.height / 2.0)
    t = tiles[idx]
    return (t.cx, t.cy)


def summary(board: Board, tiles: list[Tile] | None = None) -> str:
    """One line for the operator: raster shape, frame count, scan cost."""
    tiles = tiles if tiles is not None else plan(board.width, board.height)
    nx = len({t.col for t in tiles})
    ny = len({t.row for t in tiles})
    frames = len(tiles) * len(config.CHANNELS)
    tw, th = tile_size_mm()
    return (f"{nx} x {ny} tiles of {tw:.1f} x {th:.1f} mm, "
            f"{len(tiles)} positions, {frames} frames")
