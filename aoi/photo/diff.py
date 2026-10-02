"""Whole-board difference: where does the test board differ from golden 1?

No part boxes are needed. The test photo arrives already warped into golden-1
pixels (register.warp_to_ref); what is left is the 10-15 px of residual drift a
single homography leaves, a different exposure, glare, and the actual defects.

    1  work on a reduced copy (DIFF_WORK_SIDE), cropped to the board
    2  touch-up alignment: per-tile phase correlation on gradient images,
       median-filtered and interpolated into a smooth displacement field
    3  colour difference: Lab with local illumination divided out (L / local mean,
       a,b minus local mean), minimised over +-SHIFT_R px of shift, so a lamp
       gradient, a white-balance shift or 1-2 px of leftover misalignment is not
       a difference
    4  structure difference: 1 - local NCC of grey, maximised over the same
       shifts; glare changes brightness a lot and pattern little, a missing,
       turned or moved part changes pattern
    5  a pixel differs when BOTH exceed their threshold (and, with >= 2 goldens,
       SPREAD_K times what the goldens differ from each other there), then a
       local-fraction vote, morphology and connected components give blobs

Every constant below was tuned on the 2026-09-25 iPhone fixture of
PT-PCB-004-03 (photos/fixtures/2026-09-25: one golden, three test boards, 12 MP)
at the working resolution of 2016 px long side, and checked on the 2026-09-26
Sony three-golden shoot. The fixture numbers are therefore in-sample. The
experiment that chose them: scratchpad nobox/{a1,a1b,a1c,final}.py.

All coordinates returned are full-resolution golden-1 pixels.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from . import gate

# The test side runs on its own small thread pool (cv2 and numpy release the
# GIL), in row bands and tile wavefronts arranged so every pixel gets the same
# arithmetic in the same order as one pass over the whole crop: the numbers are
# bit-identical to the single-threaded form. Separate from features.pool(),
# which this module's caller may already be running on.
DIFF_WORKERS = max(1, min(4, os.cpu_count() or 1))
DIFF_BAND_ROWS = 64                    # rows per band: small enough that a band's planes stay in cache
_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(DIFF_WORKERS, thread_name_prefix="aoi-diff")
    return _POOL


def _bands(H: int, rows: int | None = None) -> list[tuple[int, int]]:
    n = max(1, round(H / (rows or DIFF_BAND_ROWS)))
    cuts = [round(i * H / n) for i in range(n + 1)]
    return [(cuts[i], cuts[i + 1]) for i in range(n) if cuts[i + 1] > cuts[i]]

# ------------------------------------------------------------------ constants
# Working copy long side. Half of the 4032 px iPhone frame the method was tuned
# on; every pixel-valued constant below is in working pixels at this size. A
# smaller photo is used as is (never upsampled).
DIFF_WORK_SIDE = 2016

# Touch-up alignment (fixture: residual drift 10-15 px full res, 5-7 working px;
# 160 px tiles at 80 px steps held it to under 1 px with parts still rigid).
TILE = 160
TILE_STEP = 80
TILE_MAX_SHIFT = 12            # working px; a larger tile answer is a false peak
TILE_MIN_RESPONSE = 0.08       # phase-correlation peak below this: tile ignored
TILE_MIN_STD = 2.0             # flat tile (no gradient): ignored

# Colour difference. Sigma 25 working px (~1.1 mm at 45 px/mm full res) removes
# lamp gradients without removing a 0603 body. a,b weighted 1.5 so a colour
# change counts about as much as a brightness change.
LAB_SIGMA = 25.0
AB_GAIN = 1.5
SHIFT_R = 2                    # +-2 working px (= +-4 full res) shift tolerance

# Structure difference. Window 9 px and C=400 (grey-level variance units)
# beat window 15 / C=25 on the fixture (a1b/a1c sweeps): the large C keeps flat
# soldermask, whose NCC is noise, from reading as changed.
NCC_WIN = 9
NCC_C = 400.0

# AND thresholds (a1c sweep, fixture): colour > 15, structure > 0.3.
T_COLOUR = 15.0
T_STRUCT = 0.3

# Blob extraction. A pixel survives when > 60% of its 15x15 neighbourhood
# differs; then open 5, close 9, and components under 150 px (about 25 x 25 full
# res px, half a 0603 body) are dropped. Boxes are padded by half the vote window.
VOTE_WIN = 15
VOTE_FRAC = 0.6
OPEN_K = 5
CLOSE_K = 9
MIN_AREA = 150
BLOB_PAD = VOTE_WIN // 2

# The board edge against the desk always differs a little after warping; stay
# 15 working px inside the board and inside the covered area.
BOARD_ERODE = 15
COVER_ERODE = 2 * SHIFT_R + 3
CROP_MARGIN = 40               # working px of context kept around the board crop

# Golden spread. Pairwise golden differences, blurred (sigma 2), and the
# threshold raised to SPREAD_K times that where it is larger (Sony 3-golden
# shoot: removed glare and flux-residue blobs the single golden produced).
SPREAD_K = 2.0
SPREAD_BLUR = 2.0
SPREAD_DOWN = 2                # stored at 1/2 working res: 6 MB for a 12 MP photo


@dataclass
class Blob:
    """A region that differs from golden 1, in full-resolution golden-1 pixels."""
    x: int
    y: int
    w: int
    h: int
    score: float      # mean of min(colour/threshold, structure/threshold) over the blob
    area_px: int      # differing pixels (before padding), full-res pixel units


# ------------------------------------------------------------------ helpers
def _scale(shape: tuple[int, ...]) -> float:
    return min(1.0, DIFF_WORK_SIDE / float(max(shape[:2])))


def _small(img: np.ndarray, s: float, interp=cv2.INTER_AREA) -> np.ndarray:
    if s >= 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def _fit(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A mask of any size resampled to (h, w), as bool."""
    m = np.asarray(mask)
    if m.shape[:2] != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0


def _grad(g: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def _grey(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def touch_up(ref_grey: np.ndarray, img_bgr: np.ndarray, ref_grad: np.ndarray | None = None) -> np.ndarray:
    """`img_bgr` re-aligned onto `ref_grey` by a smooth per-tile translation field.

    Rigid over a tile, so a part that moved inside a 160 px tile of board is not
    dragged along with it; the median filter stops one tile whose content
    changed (a missing part) from bending its neighbours."""
    # A copy of a cached gradient: cv2.phaseCorrelate (OpenCV 5) writes the
    # Hanning window into its inputs, so half-overlapping tiles see the ones
    # before them already windowed. The thresholds were tuned with that
    # behaviour, so it is kept; the cached golden gradient must not collect it.
    A = _grad(ref_grey) if ref_grad is None else ref_grad.copy()
    B = _grad(_grey(img_bgr))
    H, W = A.shape
    if H < TILE or W < TILE:
        return img_bgr
    ys = list(range(0, H - TILE + 1, TILE_STEP))
    xs = list(range(0, W - TILE + 1, TILE_STEP))
    dx = np.zeros((len(ys), len(xs)), np.float32)
    dy = np.zeros_like(dx)
    ok = np.zeros(dx.shape, bool)
    win = cv2.createHanningWindow((TILE, TILE), cv2.CV_32F)
    # The windowing phaseCorrelate leaves in its inputs is exactly input * win
    # (float32), and it is the only thing a tile passes on to the tiles after
    # it. So one cheap pass in row-major order does the flat-tile checks, keeps
    # a copy of what each tile would have been handed, and applies the window
    # itself; the correlations run on the pool on those copies as the pass
    # hands them out, with the same inputs, hence the same answers, as the
    # one-by-one loop.
    def tile(i: int, j: int, a: np.ndarray, b: np.ndarray) -> None:
        (sx, sy), r = cv2.phaseCorrelate(a, b, win)
        if r > TILE_MIN_RESPONSE and abs(sx) < TILE_MAX_SHIFT and abs(sy) < TILE_MAX_SHIFT:
            dx[i, j], dy[i, j], ok[i, j] = sx, sy, True

    pool = _pool()
    jobs = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            a = A[y:y + TILE, x:x + TILE]
            b = B[y:y + TILE, x:x + TILE]
            if a.std() < TILE_MIN_STD or b.std() < TILE_MIN_STD:
                continue
            jobs.append(pool.submit(tile, i, j, a.copy(), b.copy()))
            np.multiply(a, win, out=a)
            np.multiply(b, win, out=b)
    for f in jobs:
        f.result()
    if not ok.any():
        return img_bgr
    for f in (dx, dy):
        f[~ok] = np.median(f[ok])
        if min(f.shape) >= 3:
            f[:] = cv2.medianBlur(f, 3)
    fx = cv2.resize(dx, (W, H), interpolation=cv2.INTER_CUBIC)
    fy = cv2.resize(dy, (W, H), interpolation=cv2.INTER_CUBIC)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(img_bgr, gx + fx, gy + fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# The LAB_SIGMA local mean is taken at 1/WIDE_DOWN resolution: a 201-tap blur
# at working resolution was ~100 ms a plane on an M5 (several times that on a
# Raspberry Pi) and a quarter of a board's CPU. Padded by WIDE_PAD (reflected,
# as GaussianBlur's own border) so the board edge reads as it did. Against the
# full-resolution blur on the 2026-09-28 fixture: at most 0.04 grey levels off
# away from the crop edge, against a colour threshold of 15.
WIDE_DOWN = 2
WIDE_PAD = 80


def _wide_blur(x: np.ndarray) -> np.ndarray:
    """GaussianBlur(x, LAB_SIGMA) by way of a 1/WIDE_DOWN copy: area-averaged
    down, blurred with the sigma that leaves the same total spread, linear up."""
    k, P = WIDE_DOWN, WIDE_PAD
    H, W = x.shape[:2]
    # bottom and right padded a little more, to a multiple of k: an exact k:1 scale
    xp = cv2.copyMakeBorder(x, P, P + (-(H + 2 * P)) % k, P, P + (-(W + 2 * P)) % k, cv2.BORDER_REFLECT_101)
    h, w = xp.shape[:2]
    sm = cv2.resize(xp, (w // k, h // k), interpolation=cv2.INTER_AREA)
    sig = float(np.sqrt(LAB_SIGMA ** 2 - (k * k - 1) / 12.0) / k)
    up = cv2.resize(cv2.GaussianBlur(sm, (0, 0), sig), (w, h), interpolation=cv2.INTER_LINEAR)
    return up[P:P + H, P:P + W]


def _norm_lab_planes(bgr: np.ndarray) -> list[np.ndarray]:
    """_norm_lab as three separate float32 planes, which is how _colour_diff reads them."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    Lm, am, bm = cv2.split(_wide_blur(lab))
    return [100.0 * L / (Lm + 1.0), (a - am) * AB_GAIN, (b - bm) * AB_GAIN]


def _norm_lab(bgr: np.ndarray) -> np.ndarray:
    """Lab with local illumination removed (L over its local mean, a/b minus theirs)."""
    return np.dstack(_norm_lab_planes(bgr))


def _planes(lab: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    return lab if isinstance(lab, list) else [np.ascontiguousarray(lab[..., c]) for c in range(3)]


def _shifts(r: int):
    return [(oy, ox) for oy in range(-r, r + 1) for ox in range(-r, r + 1)]


def _colour_diff(Gl: np.ndarray | list[np.ndarray], Tl: np.ndarray | list[np.ndarray],
                 r: int = SHIFT_R) -> np.ndarray:
    """min over shifts of T within +-r of the Euclidean Lab distance.

    On separate channel planes (either argument may already be a list of the
    three): the same additions in the same order as a sum over the interleaved
    channel axis, so the same numbers, about 3x faster. In row bands on the
    pool; every pixel is on its own, so a band is exactly the full-crop sum."""
    Gp = _planes(Gl)
    H, W = Gp[0].shape
    Pp = [cv2.copyMakeBorder(t, r, r, r, r, cv2.BORDER_REPLICATE) for t in _planes(Tl)]
    best = np.empty((H, W), np.float32)

    def band(rows: tuple[int, int]) -> None:
        y0, y1 = rows
        s = np.empty((y1 - y0, W), np.float32); d = np.empty_like(s)
        out = best[y0:y1]
        for n, (oy, ox) in enumerate(_shifts(r)):
            sl = (slice(r + oy + y0, r + oy + y1), slice(r + ox, r + ox + W))
            np.subtract(Gp[0][y0:y1], Pp[0][sl], out=s); s *= s
            for c in (1, 2):
                np.subtract(Gp[c][y0:y1], Pp[c][sl], out=d); d *= d; s += d
            if n == 0:
                out[:] = s
            else:
                np.minimum(out, s, out=out)
        np.sqrt(out, out=out)

    list(_pool().map(band, _bands(H)))
    return best


def _struct_ref(Gg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Golden side of _struct_diff: grey as float, local mean, local variance."""
    a = Gg.astype(np.float32)
    k = (NCC_WIN, NCC_WIN)
    ma = cv2.blur(a, k)
    return a, ma, cv2.blur(a * a, k) - ma * ma


def _struct_diff(Gg: np.ndarray, Tg: np.ndarray, r: int = SHIFT_R, ref=None) -> np.ndarray:
    """1 - max over shifts (+-r) of local NCC (SSIM-style form) on grey.

    Shifting commutes with the box filter, so the test's local mean and variance
    are computed once and sliced; only the cross term is re-blurred per shift.

    In row bands on the pool: each band blurs its rows plus NCC_WIN // 2 of the
    rows either side, which is every row the box filter reads for the band's own
    rows, so they come out bit-identical to one blur over the whole crop."""
    a, ma, va = _struct_ref(Gg) if ref is None else ref
    b0 = Tg.astype(np.float32)
    H, W = a.shape
    k = (NCC_WIN, NCC_WIN)
    h = NCC_WIN // 2
    P = cv2.copyMakeBorder(b0, r, r, r, r, cv2.BORDER_REPLICATE)
    mP = cv2.blur(P, k)
    vP = cv2.blur(P * P, k) - mP * mP
    best = np.empty((H, W), np.float32)

    def band(rows: tuple[int, int]) -> None:
        # (2 cov + C) / (va + vb + C), cov = blur(a b) - ma mb, evaluated in
        # that order into reused buffers
        y0, y1 = rows
        e0, e1 = max(0, y0 - h), min(H, y1 + h)
        i0, i1 = y0 - e0, y1 - e0
        ab, mab, vab = a[e0:e1], ma[y0:y1], va[y0:y1]
        out = best[y0:y1]
        prod = np.empty((e1 - e0, W), np.float32); blr = np.empty_like(prod)
        c = np.empty((y1 - y0, W), np.float32); t = np.empty_like(c)
        for n, (oy, ox) in enumerate(_shifts(r)):
            sl = (slice(r + oy + y0, r + oy + y1), slice(r + ox, r + ox + W))
            np.multiply(ab, P[r + oy + e0:r + oy + e1, r + ox:r + ox + W], out=prod)
            cv2.blur(prod, k, dst=blr)
            np.multiply(mab, mP[sl], out=t)
            np.subtract(blr[i0:i1], t, out=c)
            np.multiply(c, 2.0, out=c); np.add(c, NCC_C, out=c)
            np.add(vab, vP[sl], out=t); np.add(t, NCC_C, out=t)
            np.divide(c, t, out=c)
            if n == 0:
                out[:] = c
            else:
                np.maximum(out, c, out=out)
        np.subtract(1.0, out, out=out)

    list(_pool().map(band, _bands(H)))
    return best


def _crop_box(valid: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.flatnonzero(valid.any(axis=1)), np.flatnonzero(valid.any(axis=0))
    if len(ys) == 0:
        return None
    H, W = valid.shape
    y0 = max(0, int(ys[0]) - CROP_MARGIN); y1 = min(H, int(ys[-1]) + 1 + CROP_MARGIN)
    x0 = max(0, int(xs[0]) - CROP_MARGIN); x1 = min(W, int(xs[-1]) + 1 + CROP_MARGIN)
    return y0, y1, x0, x1


def _board_valid(golden_s: np.ndarray, board_mask: np.ndarray | None) -> np.ndarray:
    shape = golden_s.shape[:2]
    if board_mask is None:
        bm = gate.board_mask(golden_s, side=max(shape)) > 0
    else:
        bm = _fit(board_mask, shape)
    return cv2.erode(bm.astype(np.uint8), np.ones((BOARD_ERODE, BOARD_ERODE), np.uint8)) > 0


class RefMaps:
    """Everything _pair_maps computes from the golden crop alone. Built once per
    golden and crop box, so a batch of boards does not redo golden 1's half."""

    def __init__(self, g_s: np.ndarray):
        self.grey = _grey(g_s)
        self.grad = _grad(self.grey)
        self.lab = _norm_lab_planes(g_s)
        self.struct = _struct_ref(self.grey)


def _pair_maps(g_s: np.ndarray, t_s: np.ndarray, ref: RefMaps | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(colour, structure) difference of working-res crops, test touched up first.

    The structure half runs on a thread of its own beside the colour half: the
    colour half's large Gaussian blurs are OpenCV's own parallel work and leave
    cores idle between them."""
    ref = ref or RefMaps(g_s)
    t_s = touch_up(ref.grey, t_s, ref.grad)
    got: dict = {}

    def struct() -> None:
        try:
            got["sm"] = _struct_diff(ref.grey, _grey(t_s), ref=ref.struct)
        except BaseException as e:       # re-raised on the caller's thread
            got["err"] = e

    th = threading.Thread(target=struct, name="aoi-diff-struct")
    th.start()
    try:
        cm = _colour_diff(ref.lab, _norm_lab_planes(t_s))
    finally:
        th.join()
    if "err" in got:
        raise got["err"]
    return cm, got["sm"]


# ------------------------------------------------------------------ public API
def spread_map(goldens_bgr: list[np.ndarray]) -> np.ndarray | None:
    """How much the goldens differ from each other, per pixel of golden 1.

    goldens[0] is golden 1; the others are registered onto it here. Returns a
    float32 array (h, w, 2): channel 0 colour spread, channel 1 structure
    spread, at 1/SPREAD_DOWN of the working resolution (about 1008 x 756 x 2 for
    a 12 MP photo, 6 MB). None with fewer than two goldens. Raises ValueError
    when an extra golden does not register onto golden 1."""
    if goldens_bgr is None or len(goldens_bgr) < 2:
        return None
    from . import register
    g0 = goldens_bgr[0]
    s = _scale(g0.shape)
    ims = [_small(g0, s)]
    covers = [np.ones(ims[0].shape[:2], bool)]
    g0_grey = _grey(g0)
    for k, g in enumerate(goldens_bgr[1:], start=2):
        reg = register.register(g0_grey, _grey(g))
        if not reg.ok:
            raise ValueError(f"golden {k} does not register onto golden 1: {reg.note}")
        w, cov = register.warp_to_ref(g, reg.H, g0.shape)
        ims.append(_small(w, s))
        covers.append(_fit(_small(cov, s, cv2.INTER_NEAREST), ims[0].shape[:2]))
    Hs, Ws = ims[0].shape[:2]
    out = np.zeros((Hs, Ws, 2), np.float32)
    box = _crop_box(_board_valid(ims[0], None)) or (0, Hs, 0, Ws)
    y0, y1, x0, x1 = box
    crop = [im[y0:y1, x0:x1] for im in ims]
    cov_c = [c[y0:y1, x0:x1] for c in covers]
    ce = np.ones((COVER_ERODE, COVER_ERODE), np.uint8)
    gc = np.zeros(crop[0].shape[:2], np.float32)
    gs = np.zeros_like(gc)
    for i in range(len(crop)):
        for j in range(len(crop)):
            if i == j:
                continue
            c, st = _pair_maps(crop[i], crop[j])
            both = cv2.erode((cov_c[i] & cov_c[j]).astype(np.uint8), ce) > 0
            np.maximum(gc, np.where(both, c, 0), out=gc)
            np.maximum(gs, np.where(both, st, 0), out=gs)
    out[y0:y1, x0:x1, 0] = cv2.GaussianBlur(gc, (0, 0), SPREAD_BLUR)
    out[y0:y1, x0:x1, 1] = cv2.GaussianBlur(gs, (0, 0), SPREAD_BLUR)
    h2, w2 = max(1, Hs // SPREAD_DOWN), max(1, Ws // SPREAD_DOWN)
    return cv2.resize(out, (w2, h2), interpolation=cv2.INTER_AREA).astype(np.float32)


def diff_blobs(golden_bgr: np.ndarray, aligned_bgr: np.ndarray, cover: np.ndarray | None,
               spread: np.ndarray | None = None, board_mask: np.ndarray | None = None,
               min_side_px: int | None = None, cache: dict | None = None) -> list[Blob]:
    """Regions where the aligned test differs from golden 1, strongest first.

    aligned_bgr: the test already in golden-1 pixels (register.warp_to_ref).
    cover: which of those pixels the test photo covers (any size, nonzero = yes),
      or None for all. Uncovered and off-board pixels are never reported.
    spread: spread_map() of the goldens, or None (single golden).
    board_mask: 1 on the board at any resolution; gate.board_mask(golden) if None.
    min_side_px: if given (full-res px, e.g. the smallest part's short side), a
      blob whose longer side is under half of it is dropped.
    cache: a dict kept with golden 1 across a batch; golden-only work is stored
      in it (it must only ever be passed with the same golden and board mask)."""
    if golden_bgr.shape[:2] != aligned_bgr.shape[:2]:
        raise ValueError(f"aligned test {aligned_bgr.shape[:2]} is not in golden pixels "
                         f"{golden_bgr.shape[:2]}")
    cache = {} if cache is None else cache
    s = _scale(golden_bgr.shape)
    if "g_s" not in cache:
        cache["g_s"] = _small(golden_bgr, s)
        cache["board_valid"] = _board_valid(cache["g_s"], board_mask)
    g_s = cache["g_s"]
    t_s = _small(aligned_bgr, s)
    shape = g_s.shape[:2]
    valid = cache["board_valid"].copy()
    if cover is not None:
        valid &= cv2.erode(_fit(cover, shape).astype(np.uint8),
                           np.ones((COVER_ERODE, COVER_ERODE), np.uint8)) > 0
    box = _crop_box(valid)
    if box is None:
        return []
    y0, y1, x0, x1 = box
    g_c, t_c, v_c = g_s[y0:y1, x0:x1], t_s[y0:y1, x0:x1], valid[y0:y1, x0:x1]

    key = ("ref", box)
    if key not in cache:
        cache[key] = RefMaps(g_c)
    cm, sm = _pair_maps(g_c, t_c, cache[key])
    tc = T_COLOUR
    ts = T_STRUCT
    if spread is not None:
        # golden-only as well; kept with the spread it came from, so another
        # spread passed with the same cache is never given this one's thresholds
        held = cache.get(("thr", box))
        if held is None or held[0] is not spread:
            sp = np.asarray(spread, np.float32)
            if sp.ndim != 3 or sp.shape[2] != 2:
                raise ValueError(f"spread map must be (h, w, 2), got {sp.shape}")
            sp = cv2.resize(sp, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
            held = cache[("thr", box)] = (spread, np.maximum(T_COLOUR, SPREAD_K * sp[..., 0]),
                                          np.maximum(T_STRUCT, SPREAD_K * sp[..., 1]))
        tc, ts = held[1], held[2]
    rc = cm / tc
    rs = sm / ts
    hit = ((rc > 1.0) & (rs > 1.0) & v_c).astype(np.float32)

    frac = cv2.blur(hit, (VOTE_WIN, VOTE_WIN))
    m = ((frac > VOTE_FRAC) & v_c).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((OPEN_K, OPEN_K), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((CLOSE_K, CLOSE_K), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    strength = np.minimum(rc, rs)
    Hf, Wf = golden_bgr.shape[:2]
    inv = 1.0 / s
    min_long = 0.5 * min_side_px if min_side_px else 0.0
    out: list[Blob] = []
    for j in range(1, n):
        x, y, w, h, a = (int(v) for v in st[j])
        if a < MIN_AREA:
            continue
        if max(w, h) * inv < min_long:
            continue
        sub = lab[y:y + h, x:x + w] == j
        score = float(strength[y:y + h, x:x + w][sub].mean())
        bx0 = (x0 + x - BLOB_PAD) * inv
        by0 = (y0 + y - BLOB_PAD) * inv
        bx1 = (x0 + x + w + BLOB_PAD) * inv
        by1 = (y0 + y + h + BLOB_PAD) * inv
        X0, Y0 = max(0, int(np.floor(bx0))), max(0, int(np.floor(by0)))
        X1, Y1 = min(Wf, int(np.ceil(bx1))), min(Hf, int(np.ceil(by1)))
        out.append(Blob(X0, Y0, X1 - X0, Y1 - Y0, score, int(round(a * inv * inv))))
    out.sort(key=lambda b: -b.score)
    return out


def metal_fraction(bgr: np.ndarray, blob: Blob) -> float:
    """Fraction of the blob that is bare tinned metal in `bgr`: bright and grey
    (Lab L > 140, chroma < 22), which soldermask, silkscreen and part bodies are not."""
    crop = bgr[max(0, blob.y):blob.y + blob.h, max(0, blob.x):blob.x + blob.w]
    if crop.size == 0:
        return 0.0
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0
    return float(((L > 140) & (np.hypot(a, b) < 22)).mean())


def overlap(blob: Blob, rect: tuple[int, int, int, int]) -> float:
    """Fraction of the blob's box area that lies inside rect (x, y, w, h)."""
    x, y, w, h = rect
    ix = max(0, min(blob.x + blob.w, x + w) - max(blob.x, x))
    iy = max(0, min(blob.y + blob.h, y + h) - max(blob.y, y))
    area = blob.w * blob.h
    return ix * iy / float(area) if area > 0 else 0.0
