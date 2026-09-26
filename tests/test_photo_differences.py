"""The whole-board difference wired into inspect: what it adds, what it leaves alone.

Synthetic where the rule can be shown without photos; the rest on the real
fixture (skipped with a reason when photos/ is absent).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aoi.photo import diff, inspect as I, judge, program, teach
from aoi.photo.model import DIFF_TYPE

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "photos/fixtures/2026-09-25"
needs_fixture = pytest.mark.skipif(not (FIXTURE / "Golden.jpg").exists(),
                                   reason=f"real-photo fixture {FIXTURE} absent (photos/ is gitignored)")


# ------------------------------------------------------------ plain rules
def test_metal_fraction_reads_bare_tin_not_mask_or_bodies():
    img = np.zeros((40, 80, 3), np.uint8)
    img[:, :40] = (200, 200, 200)          # tinned trace: bright, grey
    img[:, 40:] = (40, 120, 30)            # soldermask green
    b = diff.Blob(0, 0, 80, 40, 1.0, 3200)
    assert diff.metal_fraction(img, b) == pytest.approx(0.5)
    body = np.full((40, 40, 3), 30, np.uint8)          # dark part body
    assert diff.metal_fraction(body, diff.Blob(0, 0, 40, 40, 1.0, 1600)) == 0.0


def test_quarter_turn_needs_the_same_body_size():
    g = {"long": 200, "short": 100, "axis": 0.0}
    turned = {"long": 205, "short": 96, "axis": 90.0}
    assert judge._quarter_turn(g, turned)
    # the same axis change read off a body segmented much larger is not a turn
    loose = {"long": 280, "short": 150, "axis": 90.0}
    assert not judge._quarter_turn(g, loose)
    squashed = {"long": 185, "short": 60, "axis": 90.0}
    assert not judge._quarter_turn(g, squashed)


# ------------------------------------------------------------ real photos
@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("programs")
    parts = program.read_parts_csv(FIXTURE / "parts.csv")
    without_c7 = [p for p in parts if p.refdes != "C7"]
    full = teach.build_program("full", [FIXTURE / "Golden.jpg"], parts, px_per_mm=45.2,
                               root=root, single_golden=True).program
    holed = teach.build_program("holed", [FIXTURE / "Golden.jpg"], without_c7, px_per_mm=45.2,
                                root=root, single_golden=True).program
    return {"parts": {p.refdes: p for p in parts}, "full": full, "holed": holed}


def _diffs(res):
    return [pr for pr in res.parts if pr.part.type == DIFF_TYPE]


@needs_fixture
def test_part_nobody_boxed_is_still_reported_where_it_went_missing(fx):
    """C7 is missing on test1. Taught without a C7 box, the part checks cannot
    see it; the whole-board difference must put an UNEXPECTED DIFFERENCE on it."""
    res, _ = I.inspect(fx["holed"], FIXTURE / "test1.jpg")
    c7 = fx["parts"]["C7"]
    cx, cy = c7.centre
    hits = [pr for pr in _diffs(res)
            if pr.part.x - 40 <= cx <= pr.part.x + pr.part.w + 40
            and pr.part.y - 40 <= cy <= pr.part.y + pr.part.h + 40]
    assert hits, [(d.refdes, d.part.x, d.part.y) for d in _diffs(res)]
    assert all(c.state == "UNEXPECTED DIFFERENCE" and c.severity == "REVIEW"
               for pr in hits for c in pr.calls)


@needs_fixture
def test_fully_taught_boards_get_no_unexpected_differences(fx):
    """With every part boxed, the only off-part blobs on the fixture were two
    boards' tinned traces differing (test3); the bare-metal rule drops them."""
    for t in ("test1", "test2", "test3"):
        res, _ = I.inspect(fx["full"], FIXTURE / f"{t}.jpg")
        assert _diffs(res) == [], (t, [(d.refdes, d.part.x, d.part.y) for d in _diffs(res)])
