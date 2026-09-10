"""Board description and centroid (pick-and-place) import.

Coordinates are in mm with the origin at the top-left of the board and +y
running DOWN, matching image convention so nothing has to be flipped mid
pipeline. Real centroid exports are usually y-up; `load_centroid` flips them.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import centroid, packages


@dataclass
class Component:
    refdes: str
    package: str
    x: float                # mm, centre
    y: float                # mm, centre
    rot: float = 0.0        # deg, CCW in the (x-right, y-down) frame
    value: str = ""

    @property
    def pkg(self) -> packages.Package:
        return packages.get(self.package)


@dataclass
class Board:
    name: str
    width: float                                    # mm
    height: float                                   # mm
    components: list[Component] = field(default_factory=list)
    fiducials: list[tuple[float, float]] = field(default_factory=list)

    def by_refdes(self, refdes: str) -> Component:
        for c in self.components:
            if c.refdes == refdes:
                return c
        raise KeyError(refdes)


# ---------------------------------------------------------------- centroid I/O
def load_centroid(path: str | Path, board_width: float, board_height: float,
                  y_up: bool = True, name: str = "board",
                  unit: str | None = None, side: str | None = "top") -> Board:
    """Read a placement file into a Board.

    Parsing lives in `aoi/centroid.py`, which handles the shapes real tools
    actually emit -- different column names, units in the heading or in the
    value, a title block above the header, whitespace or semicolon delimiters,
    a decimal comma, and a side column spelled half a dozen ways. Use
    `centroid.read` directly when you want the report of what was skipped;
    this raises instead, because a Board with two thirds of its components is
    not a partial success.

    Both board dimensions are required. Width used to be hard-coded to 0.0 with
    no way to supply it, which is not a cosmetic default: `synth.render` sizes
    the image from it and centres rotation on `width / 2`, so a board loaded
    from a real centroid file rendered as a zero-width array. Nothing in the
    repository called this function, so no test caught it -- and this is the
    documented entry point for the first real assembly.
    """
    rep = centroid.read(path, unit=unit, side=side)
    if rep.missing:
        raise ValueError(
            f"{path}: no column found for {', '.join(rep.missing)}.\n"
            f"Columns seen: {rep.columns or 'none recognised'}.\n"
            f"Add the heading to aoi/centroid.ALIASES if the file is valid.")

    comps: list[Component] = []
    for pl in rep.placements:
        y, rot = pl.y, pl.rot
        if y_up:                          # convert to y-down image convention
            y = board_height - y
            rot = -rot
        comps.append(Component(pl.refdes, pl.package, pl.x, y,
                               rot % 360.0, pl.value))
    return Board(name, board_width, board_height, comps)


def save_centroid(board: Board, path: str | Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["RefDes", "Package", "X", "Y", "Rotation", "Value"])
        w.writeheader()
        for c in board.components:
            w.writerow({"RefDes": c.refdes, "Package": c.package, "X": round(c.x, 4),
                        "Y": round(c.y, 4), "Rotation": round(c.rot, 2), "Value": c.value})


# ------------------------------------------------------------ layout generator
# Clear space a component must leave around a fiducial, on top of its own
# footprint: the disc radius plus room for the alignment search not to lock onto
# the wrong bright thing.
FIDUCIAL_KEEPOUT = 0.25

MIX_PASSIVE = ["0402", "0402", "0402", "0603", "0603", "0805", "SOD-123", "TANT-A"]

# Passives plus leaded parts. A real assembly is mostly two-terminal by count and
# entirely dependent on the handful of ICs for function, so both have to be in
# the same frame before the pipeline can claim to meet a real board.
MIX_MIXED = MIX_PASSIVE + ["SOIC-8", "SOIC-8", "SOT-23-6", "SOT-23-6", "QFP-32"]

_PREFIX = {"TANT-A": "C", "SOD-123": "D", "SOT-23-6": "Q",
           "SOIC-8": "U", "QFP-32": "U"}


def random_board(width: float = 40.0, height: float = 30.0, seed: int = 0,
                 pitch: float = 4.2, name: str = "DEMO-1",
                 mix: list[str] | None = None) -> Board:
    """Lay parts out on a jittered grid. Stands in for a real CAD import."""
    rng = random.Random(seed)
    mix = mix or MIX_PASSIVE
    # Wide enough for the largest part in the mix to sit in the first slot with
    # its whole crop on the board. A fixed number silently drops big packages
    # from the edge rows instead, which is how a "mixed" board ends up carrying
    # no QFPs at all.
    margin = max(3.0, max(max(packages.get(p).footprint(0.35)) for p in mix) / 2.0)
    fids = [(2.0, 2.0), (width - 2.0, 2.0), (width - 2.0, height - 2.0)]
    comps, n = [], 0
    xs = [x for x in _frange(margin, width - margin, pitch)]
    ys = [y for y in _frange(margin, height - margin, pitch)]
    for gy in ys:
        for gx in xs:
            pkg = rng.choice(mix)
            # Leave a keep-out around each fiducial, sized to the part. Alignment
            # correlates a 0.5 mm disc inside a +-2.5 mm window; a component whose
            # own metal reaches into that window can win the peak and hand back a
            # bogus board transform, which then displaces every crop on the board.
            # A 0402 at 1.6 mm is harmless, a SOIC-8 at 1.1 mm is not, so the
            # clearance scales with the footprint rather than being one number.
            # Use the LARGER footprint axis both times, so the check holds at any
            # rotation. The grid margin is a single number and cannot be: a 0402
            # needs 1.1 mm of room and a QFP-32 needs 4.9 mm.
            half = max(packages.get(pkg).footprint(0.35)) / 2.0
            if any(math.hypot(gx - fx, gy - fy) < half + FIDUCIAL_KEEPOUT
                   for fx, fy in fids):
                continue
            # A crop that runs off the image is filled by BORDER_REPLICATE, and
            # the lands that land in the replicated strip read as bare board --
            # an invisible, permanent defect on that one component.
            if not (half <= gx <= width - half and half <= gy <= height - half):
                continue
            n += 1
            comps.append(Component(
                refdes=f"{_PREFIX.get(pkg, 'R')}{n}",
                package=pkg,
                x=gx + rng.uniform(-0.3, 0.3),
                y=gy + rng.uniform(-0.3, 0.3),
                rot=rng.choice([0.0, 90.0, 180.0, 270.0]) + rng.uniform(-1.5, 1.5),
                value=f"{rng.choice([10, 100, 1, 47, 22])}{rng.choice(['k', 'R', 'n', 'u'])}",
            ))
    return Board(name, width, height, comps, fids)


def mixed_board(seed: int = 0, name: str = "MIXED-1") -> Board:
    """A board carrying leaded parts as well as passives.

    Bigger and coarser than the passive board because a QFP-32 needs a 9.7 mm
    inspection crop; at the passive pitch its neighbours would land inside it.
    """
    return random_board(72.0, 54.0, seed=seed, pitch=11.0, name=name, mix=MIX_MIXED)


def _frange(a: float, b: float, step: float):
    v = a
    while v <= b + 1e-9:
        yield v
        v += step


def rot_matrix(deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    return math.cos(r), math.sin(r)
