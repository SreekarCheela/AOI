"""Per-part measurements: one golden window, one test window, a dict of numbers.

Everything here was found on the 2026-09-25 iPhone photos of PT-PCB-004-03
(REPORT.md in out/photo_run_20260925). The lessons that shaped it:

* The global homography leaves 10-15 px of drift, so every part is re-aligned on
  the ring of board around it, body masked out (`local_shift`).
* Two photos are never exposed alike, so the test window's colour levels are
  mapped onto the golden window's, using the ring only (`normalise`).
* Missing parts do not reliably look like "not a body": these footprints are
  mask-defined, the gap between bare lands is dark laminate, and dull solder
  lands are brown. No single feature separated them. Four together did:
  median colour change, loss of the body's long edges across the gap, the
  soldermask/bare-metal fraction of the body core, and whether soldermask runs
  continuously through the land gap (`missing_features`).
* Placement is measured on a body map (1 on body, 0 on soldermask and on metal),
  so the fixed lands cannot argue for zero offset (`place`).
* Skew is measured against the silkscreen outline, the design reference, mod 90
  (`silk_skew`); a quarter turn needs the golden's long axis (`body_rect`).
* Polarity from a 0/180 body correlation was fooled by lighting; the cathode band
  and bevel are stripes dark across the whole width, so they are read from an
  80th-percentile profile, which ignores marking strokes (`stripe_end`).

All coordinates are golden-1 pixels. Images are padded once by PAD so every
window is in bounds.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from .. import config
from .io import PhotoError
from .model import Part

PAD = 400
# Parts are measured on a small thread pool: OpenCV and most numpy work release
# the GIL, and on the 2026-09-25 fixture 4 workers gave 3.2x over one, with
# bit-identical numbers; more workers gained nothing (the rest holds the GIL).
WORKERS = max(1, min(int(os.environ.get("AOI_PHOTO_WORKERS", "0")) or 6, os.cpu_count() or 1))
_POOL: ThreadPoolExecutor | None = None


def pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(WORKERS, thread_name_prefix="aoi-measure")
    return _POOL


def measure_all(ref: "Frame", img: "Frame", parts: list[Part], **kw) -> dict[str, dict]:
    """measure() for every part, in parallel, keyed by refdes in `parts` order."""
    return dict(zip([p.refdes for p in parts], pool().map(lambda p: measure(ref, img, p, **kw), parts)))
SEARCH = 24                     # local re-registration search, px
# The window margin round a part is 0.9x its long side, capped so the window plus
# the local search stays inside the PAD border added to every frame.
MAX_MARGIN = PAD - SEARCH - 8
MIN_RING_COVER = 0.5            # share of the ring that must be real photo, not border
W_LAB = np.array([0.5, 1.0, 1.0], np.float32)   # L down-weighted: exposure moves it most


# ---------------------------------------------------------------- frames
@dataclass
class Frame:
    """One photo in golden-1 pixels, padded, with the derived planes every part needs."""
    bgr: np.ndarray        # uint8, padded
    grey: np.ndarray       # float32, padded
    cover: np.ndarray      # uint8, 255 where the photo has real pixels

    @classmethod
    def of(cls, bgr: np.ndarray, cover: np.ndarray | None = None) -> "Frame":
        if cover is None:
            cover = np.full(bgr.shape[:2], 255, np.uint8)
        b = cv2.copyMakeBorder(bgr, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        c = cv2.copyMakeBorder(cover, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
        return cls(b, cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32), c)


class Geo:
    """Window geometry for one part, in padded pixels."""

    def __init__(self, p: Part):
        self.p = p
        self.m = min(int(max(35, 0.9 * max(p.w, p.h))), MAX_MARGIN)
        if int(0.6 * min(p.w, p.h)) + 4 >= self.m or max(p.w, p.h) > 2 * PAD + 400:
            raise PhotoError(f"{p.refdes}: box {p.w}x{p.h} px is too large to measure "
                             f"(sides up to about {int((MAX_MARGIN - 4) / 0.6)} px); "
                             "shoot from further away or box a smaller part")
        self.x0, self.y0 = p.x + PAD - self.m, p.y + PAD - self.m
        self.W, self.H = p.w + 2 * self.m, p.h + 2 * self.m
        self.bx = self.by = self.m                     # body box origin in the window
        e = int(0.3 * min(p.w, p.h)) + 6
        self.ring = np.ones((self.H, self.W), np.float32)
        self.ring[max(0, self.by - e):self.by + p.h + e, max(0, self.bx - e):self.bx + p.w + e] = 0
        lx, ly = (0.25, 0.2) if p.horizontal else (0.2, 0.25)
        self.core = (slice(self.by + int(ly * p.h), self.by + p.h - int(ly * p.h)),
                     slice(self.bx + int(lx * p.w), self.bx + p.w - int(lx * p.w)))
        self.body = (slice(self.by, self.by + p.h), slice(self.bx, self.bx + p.w))
        # the body and the board just round it: what place() and the gap
        # contrast read, so it has to be real photo, never replicated border
        c = min(self.m, int(0.35 * min(p.w, p.h)) + 4)
        self.near = (slice(self.by - c, self.by + p.h + c), slice(self.bx - c, self.bx + p.w + c))

    def cut(self, img: np.ndarray, dx: float = 0.0, dy: float = 0.0, extra: int = 0) -> np.ndarray:
        if dx == 0 and dy == 0:
            return img[self.y0 - extra:self.y0 + self.H + extra, self.x0 - extra:self.x0 + self.W + extra]
        M = np.float32([[1, 0, -(self.x0 - extra + dx)], [0, 1, -(self.y0 - extra + dy)]])
        return cv2.warpAffine(img, M, (self.W + 2 * extra, self.H + 2 * extra),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------- utilities
def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def subpixel(res: np.ndarray, loc: tuple[int, int]) -> tuple[float, float]:
    """Parabolic refinement of a correlation peak (maximum)."""
    x, y = loc
    fx = fy = 0.0
    if 0 < x < res.shape[1] - 1:
        l, c, r = res[y, x - 1], res[y, x], res[y, x + 1]; d = l - 2 * c + r
        fx = 0.5 * (l - r) / d if abs(d) > 1e-9 else 0.0
    if 0 < y < res.shape[0] - 1:
        u, c, b = res[y - 1, x], res[y, x], res[y + 1, x]; d = u - 2 * c + b
        fy = 0.5 * (u - b) / d if abs(d) > 1e-9 else 0.0
    return x + float(np.clip(fx, -1, 1)), y + float(np.clip(fy, -1, 1))


def lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)


def wrap(d: float, period: float) -> float:
    return (d + period / 2.0) % period - period / 2.0


def mean4(ang_deg: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean direction mod 90 and its coherence in [0, 1]."""
    if len(ang_deg) == 0 or w.sum() <= 0:
        return float("nan"), 0.0
    z = (w * np.exp(4j * np.deg2rad(ang_deg))).sum() / w.sum()
    return float(np.rad2deg(np.angle(z)) / 4.0), float(abs(z))


def _grad(g: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


# ------------------------------------------------------------ alignment
def local_shift(geo: Geo, ref: Frame, img: Frame, ring: np.ndarray | None = None) -> tuple[float, float, float]:
    """Residual shift of the board round the part (body masked out), and match quality."""
    tpl = _grad(geo.cut(ref.grey))
    big = _grad(geo.cut(img.grey, extra=SEARCH))
    res = cv2.matchTemplate(big, tpl, cv2.TM_CCOEFF_NORMED, mask=geo.ring if ring is None else ring)
    res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, v, _, loc = cv2.minMaxLoc(res)
    sx, sy = subpixel(res, loc)
    return sx - SEARCH, sy - SEARCH, float(v)


def normalise(t_win: np.ndarray, g_win: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Map the test window's per-channel levels onto the golden's, from the ring
    only: a missing or moved part must not drag the mapping toward itself."""
    m = ring > 0
    out = np.empty(t_win.shape, np.float32)
    for c in range(3):
        tq = np.percentile(t_win[..., c][m], (10, 50, 90))
        gq = np.percentile(g_win[..., c][m], (10, 50, 90))
        k, b = np.linalg.lstsq(np.vstack([tq, np.ones(3)]).T, gq, rcond=None)[0]
        out[..., c] = t_win[..., c] * k + b
    return out


# ---------------------------------------------------------- body classes
class Classes:
    """What a bare footprint shows: green soldermask or bright bare metal.

    A placed part covers its footprint centre with something that is neither.
    Thresholds come from this part's own golden ring."""

    def __init__(self, geo: Geo, g_lab: np.ndarray, ring: np.ndarray | None = None):
        r = g_lab[(geo.ring if ring is None else ring) > 0].reshape(-1, 3)
        sm = r[:, 1][r[:, 1] < 124]
        self.a_sm = float(np.median(sm)) if len(sm) > 50 else 112.0
        self.a_thr = 0.5 * (self.a_sm + 128.0)
        chroma = np.hypot(r[:, 1] - 128, r[:, 2] - 128)
        metal = r[(chroma < 22) & (r[:, 0] > np.percentile(r[:, 0], 85))]
        self.L_pad = float(np.median(metal[:, 0])) if len(metal) > 20 else 210.0
        self.L_body = float(np.median(g_lab[geo.core][..., 0]))
        self.L_thr = (max(150.0, 0.5 * (self.L_body + self.L_pad))
                      if self.L_body < self.L_pad - 30 else self.L_pad - 15.0)

    def green(self, l: np.ndarray) -> np.ndarray:
        return l[..., 1] < self.a_thr

    def metal(self, l: np.ndarray) -> np.ndarray:
        return (np.hypot(l[..., 1] - 128, l[..., 2] - 128) < 22) & (l[..., 0] > self.L_thr)

    def body_map(self, l: np.ndarray) -> np.ndarray:
        return (~self.green(l) & ~self.metal(l)).astype(np.float32)


def colour_class(l_core: np.ndarray) -> str:
    """Coarse body colour: 'dark' (resistor, IC, diode), 'tan' (ceramic cap), 'other'."""
    L, a, b = (float(np.median(l_core[..., i])) for i in range(3))
    chroma = np.hypot(a - 128, b - 128)
    if chroma >= 12 and b > 132:
        return "tan"
    if L < 120 and chroma < 12:
        return "dark"
    return "other"


# ------------------------------------------------------- missing features
def _orient(g: np.ndarray, geo: Geo) -> float:
    """Share of gradient energy across the long axis in the middle of the body.
    A body's long edges run through the gap; bare lands put their inner edges there."""
    p = geo.p
    b = cv2.GaussianBlur(g, (0, 0), 1.2)
    gx, gy = cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)
    ex = int(0.25 * min(p.w, p.h))
    if p.horizontal:
        sl = (slice(geo.by - ex, geo.by + p.h + ex), slice(geo.bx + int(.3 * p.w), geo.bx + p.w - int(.3 * p.w)))
        across, along = gy[sl] ** 2, gx[sl] ** 2
    else:
        sl = (slice(geo.by + int(.3 * p.h), geo.by + p.h - int(.3 * p.h)), slice(geo.bx - ex, geo.bx + p.w + ex))
        across, along = gx[sl] ** 2, gy[sl] ** 2
    return float(across.sum() / (across.sum() + along.sum() + 1e-6))


_W2 = W_LAB ** 2
DE_EXHAUSTIVE = 49              # shift grids up to 7x7 are searched exhaustively
DE_KEEP = 3                     # coarse candidates refined at full resolution


def _dE_at(g: np.ndarray, t: np.ndarray, geo: Geo, ox: int, oy: int, step: int = 1) -> float:
    return _dE_many(g, t, geo, [(ox, oy)], step)[0]


def _dE_many(g: np.ndarray, t: np.ndarray, geo: Geo, shifts: list[tuple[int, int]], step: int = 1) -> list[float]:
    """_dE_at for several shifts in one numpy pass: the same numbers, without a
    Python round trip (and the GIL) per shift."""
    p = geo.p
    if not shifts:
        return []
    gw = g[geo.by:geo.by + p.h:step, geo.bx:geo.bx + p.w:step]
    tw = np.stack([t[geo.by + oy:geo.by + oy + p.h:step, geo.bx + ox:geo.bx + ox + p.w:step]
                   for ox, oy in shifts])
    d = tw - gw
    e = np.sqrt((d * d * _W2).sum(-1)).reshape(len(shifts), -1)
    return [float(v) for v in np.median(e, axis=1)]


def _dE_med(gl: np.ndarray, tl: np.ndarray, geo: Geo) -> float:
    """Median colour change over the body box, at the best small integer shift.

    Coarse to fine: every second shift on every second pixel first, then the
    best few candidates' full 3x3 neighbourhoods at full resolution, climbing
    until no neighbour is better. Every integer shift is within one step of a
    coarse one, and the blurred colour change is smooth in the shift, so this
    lands on the exhaustive minimum (checked on the 2026-09-25 fixture) at about
    a tenth of the cost. Small grids are still searched exhaustively."""
    p = geo.p
    g = cv2.GaussianBlur(gl, (0, 0), 1.5); t = cv2.GaussianBlur(tl, (0, 0), 1.5)
    s = max(2, int(0.08 * max(p.w, p.h)))
    if (2 * s + 1) ** 2 <= DE_EXHAUSTIVE:
        return min(_dE_many(g, t, geo, [(ox, oy) for oy in range(-s, s + 1) for ox in range(-s, s + 1)]))
    grid = sorted(set(range(-s, s + 1, 2)) | {s})
    cs = [(ox, oy) for oy in grid for ox in grid]
    coarse = sorted((v, ox, oy) for v, (ox, oy) in zip(_dE_many(g, t, geo, cs, 2), cs))
    seen: dict[tuple[int, int], float] = {}

    def around(ox: int, oy: int) -> None:
        new = [(ox + dx, oy + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if abs(ox + dx) <= s and abs(oy + dy) <= s and (ox + dx, oy + dy) not in seen]
        seen.update(zip(new, _dE_many(g, t, geo, new)))

    for _, ox, oy in coarse[:DE_KEEP]:
        around(ox, oy)
    while True:                              # climb from the best exact shift
        (bx, by), best = min(seen.items(), key=lambda kv: kv[1])
        around(bx, by)
        if min(seen.values()) >= best:
            return best


def _gap_contrast(l: np.ndarray, geo: Geo) -> float:
    """Smallest colour step between a strip across the body and the board just
    outside its long sides, over strip positions 35-65% along the length. Near
    zero when soldermask runs straight through a bare footprint's land gap."""
    p = geo.p
    L, S = (p.w, p.h) if p.horizontal else (p.h, p.w)
    sw = max(2, int(0.10 * L)); o0, o1 = int(0.15 * S) + 2, int(0.6 * S) + 4
    best = np.inf
    for c in np.linspace(0.35, 0.65, 7):
        a0 = int(c * L - sw / 2); a1 = a0 + sw
        if p.horizontal:
            ins = l[geo.by + int(.2 * S):geo.by + S - int(.2 * S), geo.bx + a0:geo.bx + a1]
            outs = (l[geo.by - o1:geo.by - o0, geo.bx + a0:geo.bx + a1],
                    l[geo.by + S + o0:geo.by + S + o1, geo.bx + a0:geo.bx + a1])
        else:
            ins = l[geo.by + a0:geo.by + a1, geo.bx + int(.2 * S):geo.bx + S - int(.2 * S)]
            outs = (l[geo.by + a0:geo.by + a1, geo.bx - o1:geo.bx - o0],
                    l[geo.by + a0:geo.by + a1, geo.bx + S + o0:geo.bx + S + o1])
        mi = np.median(ins.reshape(-1, 3), 0)
        for o in outs:
            best = min(best, float(np.sqrt((((mi - np.median(o.reshape(-1, 3), 0)) * W_LAB) ** 2).sum())))
    return best


# --------------------------------------------------------------- placement
FLAT_PTP = 0.05                 # a body map with less range than this has no edge to match
PLACE_NEAR = 6                  # full-resolution translation search round the coarse guess, px
PLACE_KEEP = 3                  # coarse candidates refined at full resolution
PLACE_MIN_HALF = 12             # templates under 2x this on a side are searched at full resolution only


def place(geo: Geo, Bg: np.ndarray, Bt: np.ndarray, rng_deg: float = 20.0) -> tuple[float, float, float]:
    """Shift and turn carrying the golden body map onto the test one.

    NaN for all three when either map is flat round the part (all body or no
    body at all): there is no edge to match, and the minimum of a flat
    difference is wherever the search happened to start, not a placement."""
    p = geo.p
    sig = max(1.0, 0.06 * min(p.w, p.h))
    g = cv2.GaussianBlur(Bg, (0, 0), sig); t = cv2.GaussianBlur(Bt, (0, 0), sig)
    e = int(0.35 * min(p.w, p.h)) + 4
    y0, y1, x0, x1 = geo.by - e, geo.by + p.h + e, geo.bx - e, geo.bx + p.w + e
    cx, cy = geo.bx + p.w / 2.0, geo.by + p.h / 2.0
    s = int(0.4 * max(p.w, p.h)) + 4
    ty0, ty1 = max(0, y0 - s), min(t.shape[0], y1 + s)
    tx0, tx1 = max(0, x0 - s), min(t.shape[1], x1 + s)
    tw = t[ty0:ty1, tx0:tx1]
    if float(np.ptp(g[y0:y1, x0:x1])) < FLAT_PTP or float(np.ptp(tw)) < FLAT_PTP:
        return float("nan"), float("nan"), float("nan")

    def tpl_at(src: np.ndarray, a: float, k: float) -> np.ndarray:
        # only the template's own rectangle is rotated, not the whole window;
        # k = 0.5 on the half-resolution maps
        R = cv2.getRotationMatrix2D((cx * k, cy * k), a, 1.0)
        R[0, 2] -= x0 * k; R[1, 2] -= y0 * k
        return cv2.warpAffine(src, R, (int((x1 - x0) * k), int((y1 - y0) * k)), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

    seen: dict[float, tuple] = {}

    def at(a: float):
        """Full search at angle a: (value, response map, location)."""
        a = round(float(a), 6)
        if a not in seen:
            tpl = tpl_at(g, a, 1.0)
            res = cv2.matchTemplate(tw, tpl, cv2.TM_SQDIFF) / tpl.size
            v, _, loc, _ = cv2.minMaxLoc(res)
            seen[a] = (float(v), res, loc)
        return seen[a]

    coarse_angles = np.arange(-rng_deg, rng_deg + 1e-6, 2.0)
    th, tw_ = y1 - y0, x1 - x0
    n_loc = (tw.shape[0] - th + 1) * (tw.shape[1] - tw_ + 1)
    if min(th, tw_) < 2 * PLACE_MIN_HALF or n_loc <= 4 * (2 * PLACE_NEAR + 1) ** 2:
        best = min(coarse_angles, key=lambda a: at(a)[0])
        best = float(min(np.arange(best - 2.0, best + 2.0 + 1e-6, 0.5), key=lambda a: at(a)[0]))
        _, res, loc = at(best)
        px, py = subpixel(-res, loc)
        return float(px + tx0 - x0), float(py + ty0 - y0), best

    # Coarse to fine (an image pyramid, the usual way to make rotation +
    # translation matching fast): the 2 deg steps on half-resolution maps, then
    # the 0.5 deg steps at full resolution, translation searched only
    # +-PLACE_NEAR px round where the coarse step put the body. A minimum on the
    # edge of that window means the guess was off: that angle is searched in full.
    g2 = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    t2 = cv2.resize(tw, (tw.shape[1] // 2, tw.shape[0] // 2), interpolation=cv2.INTER_AREA)
    coarse = []
    for a in coarse_angles:
        tpl = tpl_at(g2, a, 0.5)
        v, _, loc, _ = cv2.minMaxLoc(cv2.matchTemplate(t2, tpl, cv2.TM_SQDIFF) / tpl.size)
        coarse.append((v, float(a), loc))
    ry, rx = tw.shape[0] - th, tw.shape[1] - tw_          # largest valid location

    def near(a: float, c_loc: tuple[int, int]):
        """(value, response map, location in tw, clear of the window edge?) searched round the guess."""
        gx, gy = min(max(2 * c_loc[0], 0), rx), min(max(2 * c_loc[1], 0), ry)
        lx0, lx1 = max(0, gx - PLACE_NEAR), min(rx, gx + PLACE_NEAR)
        ly0, ly1 = max(0, gy - PLACE_NEAR), min(ry, gy + PLACE_NEAR)
        tpl = tpl_at(g, a, 1.0)
        res = cv2.matchTemplate(tw[ly0:ly1 + th, lx0:lx1 + tw_], tpl, cv2.TM_SQDIFF) / tpl.size
        v, _, (x, y), _ = cv2.minMaxLoc(res)
        inside = ((0 < x < res.shape[1] - 1 or lx0 + x in (0, rx))
                  and (0 < y < res.shape[0] - 1 or ly0 + y in (0, ry)))
        return float(v), res, (x, y), (lx0, ly0), inside

    # the PLACE_KEEP best coarse angles each refined: a body map with two near
    # equal minima must not be settled by half-resolution rounding
    fine: dict[float, tuple] = {}
    for _, c_best, c_loc in sorted(coarse)[:PLACE_KEEP]:
        for a in np.arange(c_best - 2.0, c_best + 2.0 + 1e-6, 0.5):
            r = near(float(a), c_loc)
            k = round(float(a), 6)
            if k not in fine or r[0] < fine[k][0]:
                fine[k] = r
    best = min(fine, key=lambda a: fine[a][0])
    _, res, loc, (lx0, ly0), inside = fine[best]
    if not inside:
        best = float(min(fine, key=lambda a: at(a)[0]))
        _, res, loc = at(best)
        px, py = subpixel(-res, loc)
        return float(px + tx0 - x0), float(py + ty0 - y0), best
    px, py = subpixel(-res, loc)
    return float(px + lx0 + tx0 - x0), float(py + ly0 + ty0 - y0), best


def body_rect(Bm: np.ndarray, geo: Geo) -> dict | None:
    """Largest body blob overlapping the expected box: size along the part's own
    pad axis, width, area, long-axis orientation, and how much of each end of the
    taught box it covers."""
    p = geo.p
    k = max(3, int(0.12 * min(p.w, p.h)) | 1)
    mk = cv2.morphologyEx((Bm > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, st, _ = cv2.connectedComponentsWithStats(mk)
    exp = np.zeros_like(mk); exp[geo.body] = 1
    best, bo = 0, 0
    for i in range(1, n):
        o = int(((labels == i) & (exp > 0)).sum())
        if o > bo:
            best, bo = i, o
    if best == 0 or st[best, cv2.CC_STAT_AREA] < 0.12 * p.w * p.h:
        return None
    blob = labels == best
    ys, xs = np.nonzero(blob)
    (cx, cy), (w, h), a = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    if w < h:
        a += 90.0; w, h = h, w
    # extent along the taught pad axis, inside the taught box's band
    inband = blob[geo.body]
    along = inband.any(axis=0) if p.horizontal else inband.any(axis=1)
    third = max(1, len(along) // 3)
    return dict(cx=float(cx), cy=float(cy), long=float(w), short=float(h), axis=float(a % 180.0),
                area=float(st[best, cv2.CC_STAT_AREA]), extent=float(along.sum()),
                end_a=float(inband[:, :third].mean() if p.horizontal else inband[:third].mean()),
                end_b=float(inband[:, -third:].mean() if p.horizontal else inband[-third:].mean()))


# -------------------------------------------------------------- silkscreen
def silk_mask(l: np.ndarray, geo: Geo, cls: Classes) -> np.ndarray:
    """Thin white lines: bright and neutral, minus blobs (lands, through-holes),
    minus the body box (marking text is not silkscreen)."""
    white = ((np.hypot(l[..., 1] - 128, l[..., 2] - 128) < 20)
             & (l[..., 0] > max(150.0, cls.L_pad - 45))).astype(np.uint8)
    blobs = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((19, 19), np.uint8))
    thin = cv2.subtract(white, cv2.dilate(blobs, np.ones((7, 7), np.uint8)))
    thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    p = geo.p; e = int(0.15 * min(p.w, p.h)) + 3
    thin[geo.by - e:geo.by + p.h + e, geo.bx - e:geo.bx + p.w + e] = 0
    return thin


def silk_angle(thin: np.ndarray, geo: Geo) -> tuple[float, float]:
    """Direction (mod 90) of the long straight silkscreen segments round the part, and total length."""
    p = geo.p
    edges = cv2.Canny(thin * 255, 50, 150)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=25,
                           minLineLength=max(30, int(0.6 * min(p.w, p.h))), maxLineGap=6)
    if segs is None or len(segs) == 0:
        return float("nan"), 0.0
    s = segs.reshape(-1, 4).astype(np.float32)
    ang = np.rad2deg(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0]))
    ln = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    a, _ = mean4(ang, ln)
    ok = np.abs([wrap(v - a, 90.0) for v in ang]) < 10
    a, _ = mean4(ang[ok], ln[ok])
    return a, float(ln[ok].sum())


def body_angle(Bm: np.ndarray, geo: Geo) -> tuple[float, float]:
    """Direction (mod 90) of the body's edges, and their coherence."""
    p = geo.p; e = int(0.3 * min(p.w, p.h)) + 4
    b = cv2.GaussianBlur(Bm, (0, 0), 1.5)
    gx, gy = cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)
    sl = (slice(geo.by - e, geo.by + p.h + e), slice(geo.bx - e, geo.bx + p.w + e))
    gx, gy = gx[sl].ravel(), gy[sl].ravel(); m = gx * gx + gy * gy
    keep = m > 0.05 * max(float(m.max()), 1e-9)
    return mean4(np.rad2deg(np.arctan2(gy[keep], gx[keep])), m[keep])


def outline_axis(thin: np.ndarray, geo: Geo) -> float | None:
    """Long-axis direction (mod 180) of the part's own silkscreen rectangle, or None
    when the outline is shared with neighbours, open, or absent."""
    p = geo.p
    cnts, _ = cv2.findContours(cv2.dilate(thin, np.ones((5, 5), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = geo.bx + p.w / 2.0, geo.by + p.h / 2.0
    for c in cnts:
        (rx, ry), (w, h), a = cv2.minAreaRect(c)
        if cv2.pointPolygonTest(cv2.boxPoints(((rx, ry), (w, h), a)), (cx, cy), False) < 0:
            continue
        L, S = max(w, h), min(w, h)
        body_L, body_S = max(p.w, p.h), min(p.w, p.h)
        # its own rectangle: encloses the body, not much larger than it, and not square
        if not (0.8 * body_S <= S <= 3.0 * body_L and L <= 3.0 * body_L and L / max(S, 1) >= 1.2):
            continue
        return float((a + (90.0 if w < h else 0.0)) % 180.0)
    return None


# ----------------------------------------------------------------- polarity
def patch_in_golden_frame(win: np.ndarray, geo: Geo, dx: float, dy: float, dth: float, inset: float = 0.08,
                          inset_long: float | None = None) -> np.ndarray:
    """Test body resampled into the golden body box, undoing the measured offset and turn."""
    p = geo.p
    # place() found that the golden body turned by +dth (cv2 convention) and
    # shifted by (dx, dy) lies on the test body; that same forward map, used as
    # the inverse map of the warp, samples the test body back into the golden box.
    R = cv2.getRotationMatrix2D((geo.bx + p.w / 2.0, geo.by + p.h / 2.0), dth, 1.0)
    R[0, 2] += dx; R[1, 2] += dy
    full = cv2.warpAffine(win, R, (geo.W, geo.H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)
    il = inset if inset_long is None else inset_long
    ix, iy = (il, inset) if p.horizontal else (inset, il)
    return full[geo.by + int(iy * p.h):geo.by + p.h - int(iy * p.h),
                geo.bx + int(ix * p.w):geo.bx + p.w - int(ix * p.w)]


def highpass(patch: np.ndarray, p: Part) -> np.ndarray:
    a = patch.astype(np.float32)
    a = a - cv2.GaussianBlur(a, (0, 0), max(2.0, 0.25 * min(p.w, p.h)))
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    h, w = a.shape
    m = 0.12
    return a[int(m * h):h - int(m * h), int(m * w):w - int(m * w)]


# Polarity read from a band. Not bridges: on the fixture BD1's bevel read the
# wrong way on 2 of 3 good boards (-1.10, -2.01 against golden +1.6..+2.2) on
# detector and hand boxes alike, and with 3 goldens that became a false
# REVERSED; its marking reads the right way round on all of them, so a bridge
# takes the marking check (judge._polarity) like an IC.
STRIPE_TYPES = ("diode", "led")
# The band is read with the ends trimmed harder along the body's length: a
# detector box runs a little onto the leads, and bright lead metal at the ends
# diluted the fixture's reversed D4 (test1) from -1.97 to -1.25, under
# judge.STRIPE_CLEAR. At 0.14 it reads -2.22 on detector boxes, -2.25 on hand
# boxes, and the good diodes read as clearly or more (in-sample, one board).
STRIPE_INSET_LONG = 0.14


def stripe_end(hp: np.ndarray, p: Part) -> float:
    """Which end carries a stripe dark across the whole body (cathode band, bevel).

    Profile along the axis where the stripe lies: diodes along their length, a
    bridge's bevel across its width. Per position, the 80th percentile of the
    other axis: a band stays dark even in its lightest pixels, text does not.
    Returns (first-end minimum - last-end minimum) in profile sd units; the sign
    says which end, the size how clear it is."""
    along_long = p.type != "bridge"
    along_x = (p.horizontal and along_long) or (not p.horizontal and not along_long)
    a = hp if along_x else hp.T
    q = np.percentile(a, 80, axis=0).astype(np.float32)
    q = cv2.GaussianBlur(q.reshape(1, -1), (0, 0), 1.5).ravel()
    q = (q - q.mean()) / (q.std() + 1e-6)
    e = max(2, int(0.3 * len(q)))
    return float(q[:e].min() - q[-e:].min())


def turn_margin(hp_t: np.ndarray, hp_g: np.ndarray) -> float:
    """> 0: the test marking matches the golden the same way round; < 0: turned 180."""
    return ncc(hp_t, hp_g) - ncc(hp_t, np.rot90(hp_g, 2))


# ------------------------------------------------------------------- one part
def coverage(geo: Geo, ref: Frame, img: Frame) -> tuple[bool, np.ndarray, str]:
    """(in view, usable ring, why not).

    In view needs the body and the board just round it (`Geo.near`) to be real
    photo in both golden and test, and at least MIN_RING_COVER of the ring. The
    ring returned leaves out every pixel that is replicated border in the golden,
    or in the test anywhere within the local search: aligning or levelling on
    copied edge pixels is how a part at the photo edge got a false OFFSET."""
    gc, tc = geo.cut(ref.cover), geo.cut(img.cover)
    if gc[geo.near].min() == 0 or tc[geo.near].min() == 0:
        return False, geo.ring, "outside this photo"
    if gc.min() > 0 and tc.min() > 0 and geo.cut(img.cover, extra=SEARCH).min() > 0:
        return True, geo.ring, ""
    k = 2 * SEARCH + 1
    tce = cv2.erode(geo.cut(img.cover, extra=SEARCH), np.ones((k, k), np.uint8),
                    borderType=cv2.BORDER_CONSTANT, borderValue=0)[SEARCH:-SEARCH, SEARCH:-SEARCH]
    ring = geo.ring * ((gc > 0) & (tce > 0)).astype(np.float32)
    if ring.sum() < MIN_RING_COVER * geo.ring.sum():
        return False, ring, "too close to the photo edge to align"
    return True, ring, ""


def _cut_float(geo: Geo, img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """geo.cut(img.astype(float32), dx, dy) without converting the whole frame:
    the same bilinear samples, read from an integer crop wide enough for the shift."""
    e = int(np.ceil(max(abs(dx), abs(dy)))) + 2
    y0, x0 = geo.y0 - e, geo.x0 - e
    if y0 < 0 or x0 < 0 or y0 + geo.H + 2 * e > img.shape[0] or x0 + geo.W + 2 * e > img.shape[1]:
        return geo.cut(img.astype(np.float32), dx, dy)
    crop = img[y0:y0 + geo.H + 2 * e, x0:x0 + geo.W + 2 * e].astype(np.float32)
    M = np.float32([[1, 0, -(e + dx)], [0, 1, -(e + dy)]])
    return cv2.warpAffine(crop, M, (geo.W, geo.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def golden_outline(ref: Frame, p: Part) -> dict:
    """What the golden self-check reads, and nothing else: skew of the body
    against the silkscreen, and the outline and body long axes. A fraction of
    `measure`'s cost, since it skips alignment, presence and placement."""
    geo = Geo(p)
    ok, ring, _ = coverage(geo, ref, ref)
    if not ok:
        return {"in_view": False}
    gl = lab(geo.cut(ref.bgr).astype(np.float32))
    cls = Classes(geo, gl, ring)
    Bg = cls.body_map(gl)
    thin = silk_mask(gl, geo, cls)
    sa, slen = silk_angle(thin, geo)
    ba, _ = body_angle(Bg, geo)
    rg = body_rect(Bg, geo)
    return dict(in_view=True, silk_len=slen,
                skew_silk=float(wrap(ba - sa, 90.0)) if not np.isnan(sa) and not np.isnan(ba) else float("nan"),
                outline_axis=outline_axis(thin, geo), body_axis=rg["axis"] if rg else None, body_g=rg)


def measure(ref: Frame, img: Frame, p: Part, *, is_golden: bool = False) -> dict:
    """Every raw number the judge needs for part `p`, test `img` against golden `ref`.

    dx, dy, dth are NaN when the body maps give nothing to match (see place)."""
    geo = Geo(p)
    out: dict = {}
    ok, ring, why = coverage(geo, ref, img)
    if not ok:
        out.update(in_view=False, view_note=why)
        return out
    out["in_view"] = True
    sx, sy, q = (0.0, 0.0, 1.0) if is_golden else local_shift(geo, ref, img, ring)
    out.update(local_dx=sx, local_dy=sy, local_q=q)

    g_win = geo.cut(ref.bgr).astype(np.float32)
    t_raw = _cut_float(geo, img.bgr, sx, sy)
    out["glare"] = float((t_raw[geo.body].max(-1) >= config.PHOTO_GLARE_LEVEL).mean())
    t_win = t_raw if is_golden else normalise(t_raw, g_win, ring)
    gl, tl = lab(g_win), lab(t_win)
    cls = Classes(geo, gl, ring)
    Bg, Bt = cls.body_map(gl), cls.body_map(tl)

    # presence
    g_grey = cv2.cvtColor(np.clip(g_win, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    t_grey = cv2.cvtColor(np.clip(t_win, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    fg, ft = float(Bg[geo.core].mean()), float(Bt[geo.core].mean())
    gap_g, gap_t = _gap_contrast(cv2.GaussianBlur(gl, (0, 0), 1.0), geo), _gap_contrast(cv2.GaussianBlur(tl, (0, 0), 1.0), geo)
    out.update(dE_med=_dE_med(gl, tl, geo),
               orient_drop=_orient(g_grey, geo) - _orient(t_grey, geo),
               phys_ratio=ft / max(fg, 1e-3),
               gap_ratio=float(np.log(max(gap_t, 1e-3) / max(gap_g, 1e-3))),
               core_green=float(cls.green(tl)[geo.core].mean()))

    # placement
    dx, dy, dth = place(geo, Bg, Bt)
    out.update(dx=dx, dy=dy, dth=dth)
    rg, rt = body_rect(Bg, geo), body_rect(Bt, geo)
    out["body_g"], out["body_t"] = rg, rt
    out["colour_g"], out["colour_t"] = colour_class(gl[geo.core]), colour_class(tl[geo.core])

    # skew against the silkscreen
    thin = silk_mask(tl, geo, cls)
    sa, slen = silk_angle(thin, geo)
    ba, bcoh = body_angle(Bt, geo)
    out.update(silk_angle=sa, silk_len=slen, body_angle=ba, body_coh=bcoh,
               skew_silk=float(wrap(ba - sa, 90.0)) if not np.isnan(sa) and not np.isnan(ba) else float("nan"))
    if is_golden:
        out["outline_axis"] = outline_axis(silk_mask(gl, geo, cls), geo)
        out["body_axis"] = rg["axis"] if rg else None

    # marking and polarity, test body resampled into the golden frame (in place
    # when the placement could not be measured)
    placed = not np.isnan(dx)
    hp_g = highpass(patch_in_golden_frame(g_grey, geo, 0, 0, 0), p)
    hp_t = highpass(patch_in_golden_frame(t_grey, geo, *((dx, dy, dth) if placed else (0.0, 0.0, 0.0))), p)
    out.update(mark_s0=ncc(hp_t, hp_g), mark_s180=ncc(hp_t, np.rot90(hp_g, 2)),
               turn_margin=turn_margin(hp_t, hp_g))
    if p.polarized and p.type in STRIPE_TYPES:
        d = (dx, dy, dth) if placed else (0.0, 0.0, 0.0)
        out.update(stripe_g=stripe_end(highpass(patch_in_golden_frame(g_grey, geo, 0, 0, 0,
                                                                      inset_long=STRIPE_INSET_LONG), p), p),
                   stripe_t=stripe_end(highpass(patch_in_golden_frame(t_grey, geo, *d,
                                                                      inset_long=STRIPE_INSET_LONG), p), p))
    return out
