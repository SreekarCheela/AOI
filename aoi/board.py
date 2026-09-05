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

from . import packages


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
def load_centroid(path: str | Path, board_height: float, y_up: bool = True,
                  name: str = "board") -> Board:
    """Read a centroid CSV with columns RefDes, Package, X, Y, Rotation."""
    comps: list[Component] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            k = {key.strip().lower(): (val or "").strip() for key, val in row.items()}
            y = float(k["y"])
            rot = float(k.get("rotation", 0) or 0)
            if y_up:                      # convert to y-down image convention
                y = board_height - y
                rot = -rot
            comps.append(Component(k["refdes"], k["package"], float(k["x"]), y,
                                   rot % 360.0, k.get("value", "")))
    return Board(name, 0.0, board_height, comps)


def save_centroid(board: Board, path: str | Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["RefDes", "Package", "X", "Y", "Rotation", "Value"])
        w.writeheader()
        for c in board.components:
            w.writerow({"RefDes": c.refdes, "Package": c.package, "X": round(c.x, 4),
                        "Y": round(c.y, 4), "Rotation": round(c.rot, 2), "Value": c.value})


# ------------------------------------------------------------ layout generator
_MIX = ["0402", "0402", "0402", "0603", "0603", "0805", "SOD-123", "TANT-A"]


def random_board(width: float = 40.0, height: float = 30.0, seed: int = 0,
                 pitch: float = 4.2, name: str = "DEMO-1") -> Board:
    """Lay parts out on a jittered grid. Stands in for a real CAD import."""
    rng = random.Random(seed)
    margin = 3.0
    comps, n = [], 0
    xs = [x for x in _frange(margin, width - margin, pitch)]
    ys = [y for y in _frange(margin, height - margin, pitch)]
    for gy in ys:
        for gx in xs:
            pkg = rng.choice(_MIX)
            n += 1
            comps.append(Component(
                refdes=f"{'C' if pkg in ('TANT-A',) else 'D' if pkg == 'SOD-123' else 'R'}{n}",
                package=pkg,
                x=gx + rng.uniform(-0.3, 0.3),
                y=gy + rng.uniform(-0.3, 0.3),
                rot=rng.choice([0.0, 90.0, 180.0, 270.0]) + rng.uniform(-1.5, 1.5),
                value=f"{rng.choice([10, 100, 1, 47, 22])}{rng.choice(['k', 'R', 'n', 'u'])}",
            ))
    fids = [(2.0, 2.0), (width - 2.0, 2.0), (width - 2.0, height - 2.0)]
    return Board(name, width, height, comps, fids)


def _frange(a: float, b: float, step: float):
    v = a
    while v <= b + 1e-9:
        yield v
        v += step


def rot_matrix(deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    return math.cos(r), math.sin(r)
