"""Package library: body and land geometry per package, in mm.

In production this is generated from datasheets (spec section 6); here it is
hand-entered for the packages the prototype renders.

Land geometry is described as ROWS rather than as a pair of pads. A chip is one
land on the left and one on the right, which is the degenerate case; a SOIC is
four a side; a QFP is eight on each of four sides. Everything downstream works
from `lands()`, so nothing else in the pipeline knows how many terminals a
package has.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Component-local axes: +x runs along the pad-to-pad axis of a chip, +y runs
# DOWN, matching the image convention board.py establishes. A row name says
# which side of the body the lands sit on.
ROW_AXIS: dict[str, tuple[float, float]] = {
    # row       unit vector pointing from the land IN toward the body, which is
    #           the direction solder climbs
    "left":   (+1.0, 0.0),
    "right":  (-1.0, 0.0),
    "top":    (0.0, +1.0),
    "bottom": (0.0, -1.0),
}


@dataclass(frozen=True)
class Land:
    """One terminal's land, in component-local mm."""
    cx: float
    cy: float
    hx: float           # half extent along x
    hy: float           # half extent along y
    ax: float           # unit vector toward the body: the fillet climbs this way
    ay: float
    row: str
    index: int          # position within its row, 0 first

    @property
    def half_along(self) -> float:
        """Half extent along the fillet axis."""
        return self.hx if abs(self.ax) > abs(self.ay) else self.hy


@dataclass(frozen=True)
class Package:
    name: str
    body_l: float       # length, along the x axis
    body_w: float       # width, across
    body_h: float       # height above board
    pad_l: float        # land length along its own fillet axis
    pad_w: float        # land width across that axis
    pad_gap: float      # inner-edge to inner-edge gap between opposing rows
    polarized: bool = False
    rows: tuple[str, ...] = ("left", "right")
    per_row: int = 1
    pitch: float = 0.0  # centre-to-centre within a row; unused when per_row == 1

    # ---------------------------------------------------------------- geometry
    @property
    def n_pads(self) -> int:
        return len(self.rows) * self.per_row

    @property
    def is_two_terminal(self) -> bool:
        """Chips, diodes, tantalums. Only these can tombstone: a part needs
        exactly two terminals to stand up on one of them."""
        return self.n_pads == 2

    @property
    def pad_pitch(self) -> float:
        """Centre-to-centre distance between opposing rows."""
        return self.pad_gap + self.pad_l

    def lands(self) -> list[Land]:
        """Every land on the package, in component-local mm."""
        out: list[Land] = []
        d = self.pad_pitch / 2.0
        for row in self.rows:
            ax, ay = ROW_AXIS[row]
            along_x = abs(ax) > abs(ay)
            hx = self.pad_l / 2.0 if along_x else self.pad_w / 2.0
            hy = self.pad_w / 2.0 if along_x else self.pad_l / 2.0
            for i in range(self.per_row):
                # spread the row about its own centre
                off = (i - (self.per_row - 1) / 2.0) * self.pitch
                cx = -ax * d + (0.0 if along_x else off)
                cy = -ay * d + (off if along_x else 0.0)
                out.append(Land(cx, cy, hx, hy, ax, ay, row, i))
        return out

    def adjacent_pairs(self) -> list[tuple[Land, Land]]:
        """Neighbouring lands within a row -- where a solder bridge can form.

        A bridge shorts two terminals that sit next to each other. On a chip
        that is the single gap across the body; on a lead row it is every
        consecutive pair, and those gaps are far narrower and far more likely.
        """
        lands = self.lands()
        if self.is_two_terminal:
            return [(lands[0], lands[1])]
        out = []
        for row in self.rows:
            in_row = sorted((l for l in lands if l.row == row), key=lambda l: l.index)
            out.extend(zip(in_row, in_row[1:]))
        return out

    def extent(self) -> tuple[float, float]:
        """Half extents covering the body and every land."""
        hx = self.body_l / 2.0
        hy = self.body_w / 2.0
        for l in self.lands():
            hx = max(hx, abs(l.cx) + l.hx)
            hy = max(hy, abs(l.cy) + l.hy)
        return hx, hy

    @property
    def span(self) -> float:
        """Total land-pattern length along x."""
        return 2.0 * self.extent()[0]

    def fillet_run(self, land: Land) -> float:
        """How far solder climbs on this land: outer land edge to body edge.

        Must be positive. If the body reaches past the land there is no lead
        exposed, the fillet window degenerates to a sliver of bare board, and
        the joint reads as permanently starved -- which is a silent, total loss
        of the fillet test for that package rather than an error.
        """
        if abs(land.ax) > abs(land.ay):
            return abs(land.cx) + land.hx - self.body_l / 2.0
        return abs(land.cy) + land.hy - self.body_w / 2.0

    def validate(self) -> list[str]:
        """Geometry problems that would silently disable a test."""
        bad = []
        for i, l in enumerate(self.lands()):
            run = self.fillet_run(l)
            if run <= 0.15:
                bad.append(f"{self.name} land {i} ({l.row} {l.index}) has fillet "
                           f"run {run:+.2f} mm: the body reaches past the land, so "
                           f"there is nothing for solder to climb")
        if self.per_row > 1:
            span = (self.per_row - 1) * self.pitch
            across = self.body_w if self.rows[0] in ("left", "right") else self.body_l
            if span > across:
                bad.append(f"{self.name} row spans {span:.2f} mm across a "
                           f"{across:.2f} mm body: lands run off the end")
        for a, b in self.adjacent_pairs():
            gap = max(abs(b.cx - a.cx) / 2.0 - min(a.hx, b.hx),
                      abs(b.cy - a.cy) / 2.0 - min(a.hy, b.hy))
            if gap <= 0.0:
                bad.append(f"{self.name} has overlapping lands")
        return bad

    def footprint(self, margin: float = 0.35) -> tuple[float, float]:
        """Bounding box (length, width) to crop for inspection, with margin."""
        hx, hy = self.extent()
        return (2.0 * hx + 2.0 * margin, 2.0 * hy + 2.0 * margin)


LIBRARY: dict[str, Package] = {
    p.name: p for p in (
        # --- two-terminal ---------------------------------------------------
        Package("0402", body_l=1.00, body_w=0.50, body_h=0.35,
                pad_l=0.50, pad_w=0.60, pad_gap=0.50),
        Package("0603", body_l=1.60, body_w=0.80, body_h=0.45,
                pad_l=0.75, pad_w=0.90, pad_gap=0.70),
        Package("0805", body_l=2.00, body_w=1.25, body_h=0.55,
                pad_l=0.90, pad_w=1.30, pad_gap=0.90),
        Package("SOD-123", body_l=2.70, body_w=1.60, body_h=1.00,
                pad_l=0.90, pad_w=1.20, pad_gap=1.40, polarized=True),
        Package("TANT-A", body_l=3.20, body_w=1.60, body_h=1.60,
                pad_l=1.20, pad_w=1.40, pad_gap=1.20, polarized=True),

        # --- multi-lead -----------------------------------------------------
        # Two rows of gull-wing leads on the long sides. body_l is the axis the
        # rows face ACROSS (the 3.9 mm width), body_w the axis they run ALONG
        # (the 4.9 mm length) -- getting those the wrong way round puts the body
        # edge outside the land and leaves no lead for solder to climb.
        # 1.27 mm pitch puts a lead at ~50 px and the gap between two at ~56 px,
        # both comfortably resolved at 12 um/px.
        Package("SOIC-8", body_l=3.90, body_w=4.90, body_h=1.75,
                pad_l=1.55, pad_w=0.60, pad_gap=2.90, polarized=True,
                rows=("left", "right"), per_row=4, pitch=1.27),

        # Three leads a side. The real 3-lead SOT-23 is asymmetric, 2 against 1,
        # which the row model does not express; SOT-23-6 is genuinely symmetric,
        # so it is modelled rather than approximated.
        Package("SOT-23-6", body_l=1.60, body_w=2.90, body_h=1.10,
                pad_l=0.80, pad_w=0.45, pad_gap=0.90, polarized=True,
                rows=("left", "right"), per_row=3, pitch=0.95),

        # Quad. 0.80 mm pitch is the finest the spec calls workable for presence
        # and gross bridging; section 11 is explicit that lead coplanarity and
        # subtle wetting are out of reach at this sampling.
        Package("QFP-32", body_l=7.00, body_w=7.00, body_h=1.40,
                pad_l=1.40, pad_w=0.40, pad_gap=6.20, polarized=True,
                rows=("left", "right", "top", "bottom"), per_row=8, pitch=0.80),
    )
}


def get(name: str) -> Package:
    try:
        return LIBRARY[name]
    except KeyError:
        raise KeyError(f"package {name!r} not in library; known: {sorted(LIBRARY)}") from None
