"""Whole-board difference (aoi/photo/diff.py).

Synthetic tests paint a fake board (soldermask green on a grey desk, silkscreen,
lands, part bodies, random texture so registration and phase correlation have
something to lock onto), then change it the ways a real test photo changes: a
part removed, the exposure, a lamp gradient, glare, a 1-2 px residual shift.

The fixture tests use the 2026-09-25 iPhone shoot of PT-PCB-004-03
(photos/fixtures/2026-09-25, gitignored) and skip when it is absent. The
module's constants were tuned on these same photos, so the fixture numbers are
in-sample. Measured when this was written (Apple M5, 12 MP, ~0.72 s per photo
for diff_blobs after registration):

    board   missing parts covered   false blobs   blobs
    test1   8 / 8                   0             12  (also covers T1 offset)
    test2   1 / 1                   0             1
    test3   3 / 5 (not R11, R14)    11            15

The experiment it was ported from (scratchpad nobox/final.py, box-free A1)
measured 8/8, 1/1, 3/5 with 0, 0, 10 false blobs. The test3 false blobs sit on
specular solder (U1 tab and leads, BD1 lead, C5 land), glare on the tinned top
traces, and the R1 silkscreen; they carry scores no lower than the true blobs,
so a score cut does not remove them. Part boxes (inspect.py) are what filter them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi.photo import diff

FIXTURE = Path("photos/fixtures/2026-09-25")
needs_fixture = pytest.mark.skipif(not (FIXTURE / "Golden.jpg").exists(),
                                   reason=f"real-photo fixture {FIXTURE} absent (photos/ is gitignored)")


# ------------------------------------------------------------------ fake board
W, H = 1100, 800
BOARD = (100, 80, 900, 640)                   # x, y, w, h of the soldermask
REMOVED = (660, 380, 44, 22)                  # the part a test board is missing
SMUDGE = (260, 520, 70, 50)                   # region that varies between goldens


def _parts() -> list[tuple[int, int, int, int]]:
    """A grid of 0805-ish bodies, alternating orientation, one of them REMOVED."""
    out = []
    for i, y in enumerate(range(140, 660, 80)):
        for j, x in enumerate(range(160, 960, 100)):
            out.append((x, y, 44, 22) if (i + j) % 2 == 0 else (x + 10, y - 10, 22, 44))
    assert REMOVED in out
    return out


def paint_board(missing: tuple = (), smudge: int | None = None, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (190, 195, 200), np.uint8)
    x, y, w, h = BOARD
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 130, 35), -1)
    # traces and silkscreen: fixed pattern (seeded separately from the noise)
    prng = np.random.default_rng(1234)
    for _ in range(60):
        p = prng.integers([x + 10, y + 10], [x + w - 10, y + h - 10], size=(2, 2))
        cv2.line(img, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), (70, 160, 50), 4)
    for k in range(25):
        p = prng.integers([x + 20, y + 30], [x + w - 80, y + h - 10])
        cv2.putText(img, f"R{k}", (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (230, 235, 235), 2, cv2.LINE_AA)
    for _ in range(30):
        c = prng.integers([x + 15, y + 15], [x + w - 15, y + h - 15])
        cv2.circle(img, (int(c[0]), int(c[1])), 9, (200, 205, 210), -1)
        cv2.circle(img, (int(c[0]), int(c[1])), 4, (40, 40, 40), -1)
    for (px, py, pw, ph) in _parts():
        # lands (always there), body (unless missing)
        if pw > ph:
            cv2.rectangle(img, (px - 6, py), (px + 10, py + ph), (185, 190, 195), -1)
            cv2.rectangle(img, (px + pw - 10, py), (px + pw + 6, py + ph), (185, 190, 195), -1)
        else:
            cv2.rectangle(img, (px, py - 6), (px + pw, py + 10), (185, 190, 195), -1)
            cv2.rectangle(img, (px, py + ph - 10), (px + pw, py + ph + 6), (185, 190, 195), -1)
        if (px, py, pw, ph) not in missing:
            cv2.rectangle(img, (px + 4, py + 2), (px + pw - 4, py + ph - 2), (30, 30, 32), -1)
            cv2.putText(img, "103", (px + 6, py + ph // 2 + 4), cv2.FONT_HERSHEY_PLAIN, 0.7,
                        (200, 200, 200), 1)
    if smudge is not None:           # flux residue: a dark patch whose place and shape vary per board
        sx, sy, sw, sh = SMUDGE
        srng = np.random.default_rng(100 + smudge)
        c = srng.integers([sx + 20, sy + 15], [sx + sw - 20, sy + sh - 15])
        axes = (int(srng.integers(14, 26)), int(srng.integers(8, 14)))
        cv2.ellipse(img, (int(c[0]), int(c[1])), axes, float(srng.uniform(0, 180)), 0, 360,
                    (40, 80, 120), -1)
    noise = rng.normal(0, 3.0, img.shape)
    img = np.clip(cv2.GaussianBlur(img, (3, 3), 0.8).astype(np.float32) + noise, 0, 255)
    return img.astype(np.uint8)


def shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def relight(img: np.ndarray, gain: float, gradient: float = 0.0,
            glare: tuple[int, int, int, float] | None = None) -> np.ndarray:
    """Global gain, a left-to-right lamp gradient, and an optional glare spot (x, y, r, amp)."""
    f = img.astype(np.float32) * gain
    ramp = np.linspace(1.0 - gradient, 1.0 + gradient, img.shape[1], dtype=np.float32)
    f *= ramp[None, :, None]
    if glare is not None:
        gx, gy, r, amp = glare
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
        f += (amp * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * r * r)))[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def full_cover(img: np.ndarray) -> np.ndarray:
    return np.full(img.shape[:2], 255, np.uint8)


def _hits(blobs, rect, min_frac=0.3) -> list:
    """Blobs covering at least min_frac of rect."""
    x, y, w, h = rect
    out = []
    for b in blobs:
        ix = max(0, min(b.x + b.w, x + w) - max(b.x, x))
        iy = max(0, min(b.y + b.h, y + h) - max(b.y, y))
        if ix * iy >= min_frac * w * h:
            out.append(b)
    return out


@pytest.fixture(scope="module")
def golden():
    return paint_board()


# ------------------------------------------------------------------ synthetic
def test_same_board_new_noise_has_no_blobs(golden):
    test = paint_board(seed=7)
    assert diff.diff_blobs(golden, test, full_cover(test)) == []


def test_brightness_gradient_and_subpixel_shift_are_not_differences(golden):
    test = relight(shift(paint_board(seed=8), 1.6, -1.3), gain=1.25, gradient=0.15)
    assert diff.diff_blobs(golden, test, full_cover(test)) == []


def test_glare_alone_is_not_a_difference(golden):
    test = relight(paint_board(seed=9), gain=0.9, glare=(760, 250, 45, 60.0))
    assert diff.diff_blobs(golden, test, full_cover(test)) == []


def test_missing_part_found_through_shift_brightness_and_glare(golden):
    test = relight(shift(paint_board(missing=(REMOVED,), seed=10), -1.4, 1.8),
                   gain=1.2, gradient=0.1, glare=(300, 250, 45, 60.0))
    blobs = diff.diff_blobs(golden, test, full_cover(test))
    assert len(blobs) == 1, blobs
    assert _hits(blobs, REMOVED), (blobs, REMOVED)
    b = blobs[0]
    assert b.score > 1.0 and b.area_px > 0
    assert diff.overlap(b, (b.x, b.y, b.w, b.h)) == pytest.approx(1.0)
    x, y, w, h = REMOVED
    assert diff.overlap(b, (x - 20, y - 20, w + 40, h + 40)) > 0.8


def test_blob_boxes_are_full_resolution_pixels(golden):
    """A photo larger than the working size is reduced inside; boxes come back scaled up."""
    big_g = cv2.resize(golden, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    test = cv2.resize(paint_board(missing=(REMOVED,), seed=11), None, fx=2, fy=2,
                      interpolation=cv2.INTER_CUBIC)
    assert max(big_g.shape[:2]) > diff.DIFF_WORK_SIDE
    blobs = diff.diff_blobs(big_g, test, full_cover(test))
    x, y, w, h = REMOVED
    assert len(blobs) == 1 and _hits(blobs, (2 * x, 2 * y, 2 * w, 2 * h)), blobs


def test_uncovered_and_off_board_pixels_are_ignored(golden):
    test = paint_board(missing=(REMOVED,), seed=12)
    cover = full_cover(test)
    x, y, w, h = REMOVED
    cover[y - 40:y + h + 40, x - 40:x + w + 40] = 0      # part not photographed
    assert diff.diff_blobs(golden, test, cover) == []
    off = np.zeros(test.shape[:2], np.uint8)
    off[:, :x - 60] = 1                                   # board mask excluding the part
    assert diff.diff_blobs(golden, test, None, board_mask=off) == []
    # a change on the desk, outside the board, is never reported
    desk = test.copy()
    cv2.rectangle(desk, (20, 20), (80, 70), (20, 20, 20), -1)
    assert len(diff.diff_blobs(golden, desk, None)) == 1


def test_min_side_drops_small_blobs(golden):
    test = paint_board(missing=(REMOVED,), seed=13)
    assert len(diff.diff_blobs(golden, test, None, min_side_px=20)) == 1
    assert diff.diff_blobs(golden, test, None, min_side_px=400) == []


def test_spread_map_suppresses_a_region_the_goldens_disagree_on(tmp_path):
    goldens = [paint_board(smudge=k, seed=20 + k) for k in range(3)]
    goldens[1] = shift(goldens[1], 2.5, -1.5)            # registered internally
    test = paint_board(smudge=5, missing=(REMOVED,), seed=30)

    single = diff.diff_blobs(goldens[0], test, None)
    assert _hits(single, SMUDGE, 0.1), "the smudge should differ against one golden"
    assert _hits(single, REMOVED)

    sp = diff.spread_map(goldens)
    assert sp is not None and sp.dtype == np.float32 and sp.ndim == 3 and sp.shape[2] == 2
    np.save(tmp_path / "spread.npy", sp)
    sp = np.load(tmp_path / "spread.npy")
    multi = diff.diff_blobs(goldens[0], test, None, spread=sp)
    assert not _hits(multi, SMUDGE, 0.1), multi
    assert _hits(multi, REMOVED), multi
    assert len(multi) == 1, multi


def test_spread_map_needs_two_goldens(golden):
    assert diff.spread_map([golden]) is None
    assert diff.spread_map([]) is None


def test_overlap():
    b = diff.Blob(10, 10, 10, 10, 1.0, 100)
    assert diff.overlap(b, (0, 0, 100, 100)) == 1.0
    assert diff.overlap(b, (15, 10, 100, 100)) == pytest.approx(0.5)
    assert diff.overlap(b, (50, 50, 5, 5)) == 0.0


def test_shape_mismatch_is_refused(golden):
    with pytest.raises(ValueError):
        diff.diff_blobs(golden, golden[:-10], None)


# ------------------------------------------------------------------ fixture
def _frac(part, b) -> float:
    """Fraction of the part box a blob covers (the experiment's scoring rule, >= 0.2 = covered)."""
    ax, ay, aw, ah = part
    ix = max(0, min(ax + aw, b.x + b.w) - max(ax, b.x))
    iy = max(0, min(ay + ah, b.y + b.h) - max(ay, b.y))
    return ix * iy / float(aw * ah)


@pytest.fixture(scope="module")
def fixture_blobs():
    if not (FIXTURE / "Golden.jpg").exists():
        pytest.skip(f"real-photo fixture {FIXTURE} absent (photos/ is gitignored)")
    from aoi.photo import io, register
    g = io.load_bgr(FIXTURE / "Golden.jpg")
    out = {}
    for n in ("test1", "test2", "test3"):
        t = io.load_bgr(FIXTURE / f"{n}.jpg")
        reg = register.register(io.grey(g), io.grey(t))
        assert reg.ok, reg.note
        warped, cover = register.warp_to_ref(t, reg.H, g.shape)
        out[n] = diff.diff_blobs(g, warped, cover)
    with open(FIXTURE / "parts.csv") as f:
        parts = {r["refdes"]: (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
                 for r in csv.DictReader(f)}
    truth = json.loads((FIXTURE / "truth.json").read_text())["truth"]
    return g, out, parts, truth


# measured (in-sample): missing covered 8/8, 1/1, 3/5; false blobs 0, 0, 11
MIN_MISSING = {"test1": 8, "test2": 1, "test3": 3}
MAX_FALSE = {"test1": 1, "test2": 1, "test3": 12}


@needs_fixture
def test_fixture_missing_parts_covered(fixture_blobs):
    _, blobs, parts, truth = fixture_blobs
    total = 0
    for n, bl in blobs.items():
        missing = [t["refdes"] for t in truth if t["image"] == n and t["defect"] == "missing"]
        found = [r for r in missing if any(_frac(parts[r], b) >= 0.2 for b in bl)]
        assert len(found) >= MIN_MISSING[n], (n, found, missing)
        total += len(found)
    assert total >= 12            # of 14; experiment (box-free) also 12/14


@needs_fixture
def test_fixture_offset_transistor_covered(fixture_blobs):
    _, blobs, parts, _ = fixture_blobs
    assert any(_frac(parts["T1"], b) >= 0.2 for b in blobs["test1"])


@needs_fixture
def test_fixture_false_blobs_bounded(fixture_blobs):
    _, blobs, parts, truth = fixture_blobs
    for n, bl in blobs.items():
        bad = [parts[t["refdes"]] for t in truth if t["image"] == n]
        false = [b for b in bl if not any(_frac(p, b) >= 0.2 for p in bad)]
        assert len(false) <= MAX_FALSE[n], (n, false)


@needs_fixture
def test_fixture_spread_map_is_small_and_saveable(fixture_blobs, tmp_path):
    g = fixture_blobs[0]
    sp = diff.spread_map([g, shift(g, 3.0, -2.0)])
    assert sp.dtype == np.float32 and sp.shape[2] == 2
    assert sp.nbytes < 10e6, sp.nbytes
    np.save(tmp_path / "s.npy", sp)
    assert (tmp_path / "s.npy").stat().st_size < 10e6
    # the same board shifted: nothing to spread over except noise
    assert float(np.percentile(sp[..., 0][sp[..., 0] > 0], 50)) < diff.T_COLOUR
