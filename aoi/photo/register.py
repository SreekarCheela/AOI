"""Whole-photo alignment of a test (or extra golden) photo onto golden 1.

SIFT on a reduced copy, ratio-test matches, RANSAC homography, then a check
that the warped photo actually lies on the reference. A board is full of
identical parts, so many matches are ambiguous and the inlier ratio runs low even
when the alignment is right; what settles it is the overlap correlation, ~0.9
when aligned and near zero when RANSAC locked onto a repeated pattern. A part
that moved does not drag the alignment with it: it is one outlier among hundreds
of features on the rest of the board.

Measured on the 2026-09-25 iPhone photos: 578-665 inliers, overlap 0.92-0.93, and
10-15 px of residual drift that features.local_shift removes per part.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import config

# SIFT on a 12 MP frame is slow and gains nothing; the homography is scaled back.
REG_MAX_SIDE = 1600
REG_MIN_INLIERS = 25
REG_MIN_OVERLAP_NCC = config.REG_MIN_OVERLAP_NCC
MATCH_CHUNKS = 8                  # 8000 SIFT features -> 1000 query rows per matrix product
_MATCH_POOL: ThreadPoolExecutor | None = None


def _small(img: np.ndarray) -> tuple[np.ndarray, float]:
    s = min(1.0, REG_MAX_SIDE / max(img.shape[:2]))
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, s


@dataclass
class Registration:
    H: np.ndarray | None          # test pixels -> reference pixels
    inliers: int
    matches: int
    note: str = ""
    fit: float = 0.0              # correlation of the aligned photo with the reference

    @property
    def ok(self) -> bool:
        return self.H is not None


@dataclass
class Keypoints:
    """One photo's reduced copy and its SIFT features: golden 1's are computed
    once per batch and reused for every test photo."""
    small: np.ndarray
    k: float
    kp: tuple
    desc: np.ndarray | None
    _match: tuple | None = field(default=None, repr=False)   # (desc transposed, |desc|^2), for _knn2

    def match_side(self) -> tuple[np.ndarray, np.ndarray]:
        if self._match is None:
            self._match = (np.ascontiguousarray(self.desc.T), (self.desc * self.desc).sum(1))
        return self._match


def keypoints(grey_img: np.ndarray) -> Keypoints:
    s, k = _small(grey_img)
    # Local contrast equalisation first, so a photo shot a little brighter or
    # under a different lamp still produces the same keypoints.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kp, desc = cv2.SIFT_create(nfeatures=8000).detectAndCompute(clahe.apply(s), None)
    return Keypoints(s, k, kp, desc)


def register(ref: np.ndarray | Keypoints, img: np.ndarray) -> Registration:
    """Homography taking `img` onto `ref`, or a refusal that says why. `ref`
    may be its precomputed keypoints."""
    rk = ref if isinstance(ref, Keypoints) else keypoints(ref)
    r_s, r_k, kr, dr = rk.small, rk.k, rk.kp, rk.desc
    ik = keypoints(img)
    i_s, i_k, ki, di = ik.small, ik.k, ik.kp, ik.desc
    if dr is None or di is None or len(kr) < REG_MIN_INLIERS or len(ki) < REG_MIN_INLIERS:
        return Registration(None, 0, 0, "too few features -- is the photo in focus?")

    if len(kr) < 2:
        return Registration(None, 0, 0, "too few features -- is the photo in focus?")
    idx, dist = _knn2(di, rk)
    good = np.nonzero(dist[:, 0] < 0.75 * dist[:, 1])[0]
    if len(good) < REG_MIN_INLIERS:
        return Registration(None, 0, len(good),
                            f"only {len(good)} feature matches -- is this the same board?")
    src = np.float32([ki[q].pt for q in good])
    dst = np.float32([kr[t].pt for t in idx[good, 0]])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    n_in = int(mask.sum()) if mask is not None else 0
    if H is None or n_in < REG_MIN_INLIERS:
        return Registration(None, n_in, len(good),
                            f"{n_in} of {len(good)} matches agree on one alignment -- "
                            f"too few to trust")
    fit = _overlap_ncc(r_s, cv2.warpPerspective(i_s, H, (r_s.shape[1], r_s.shape[0])),
                       cv2.warpPerspective(np.ones_like(i_s), H, (r_s.shape[1], r_s.shape[0])))
    if fit < REG_MIN_OVERLAP_NCC:
        return Registration(None, n_in, len(good),
                            f"aligned photo does not lie on the reference "
                            f"(overlap correlation {fit:.2f}) -- is this the same board?", fit)

    # Back to full resolution on both sides.
    S_i = np.diag([i_k, i_k, 1.0])
    S_r_inv = np.diag([1.0 / r_k, 1.0 / r_k, 1.0])
    H = S_r_inv @ H @ S_i
    H /= H[2, 2]
    # A homography that scales the board by 3x or mirrors it is RANSAC having
    # agreed on nonsense, not a photo taken from a slightly different spot.
    det = float(np.linalg.det(H[:2, :2]))
    if not 0.25 < det < 4.0:
        return Registration(None, n_in, len(good), f"implausible alignment (area scale {det:.2f})")
    return Registration(H, n_in, len(good), fit=fit)


def _knn2(di: np.ndarray, rk: Keypoints) -> tuple[np.ndarray, np.ndarray]:
    """Two nearest reference descriptors per query descriptor, and their L2
    distances: cv2.BFMatcher(NORM_L2).knnMatch(k=2), 3-4x faster.

    |a|^2 + |b|^2 - 2ab as one matrix product per chunk of rows, on threads.
    OpenCV's SIFT descriptors are whole numbers scaled to norm 512, so every
    product and partial sum is an integer below 2^24 and float32 holds it
    exactly: the distances are BFMatcher's to the bit, and so is the ratio test
    (checked on the 2026-09-28 fixture: same matches, same distances)."""
    global _MATCH_POOL
    if _MATCH_POOL is None:
        _MATCH_POOL = ThreadPoolExecutor(min(MATCH_CHUNKS, os.cpu_count() or 1), thread_name_prefix="aoi-match")
    rT, b2 = rk.match_side()

    def chunk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D = q @ rT
        D *= -2.0
        D += b2
        r = np.arange(len(q))
        i1 = D.argmin(1); v1 = D[r, i1].copy()
        D[r, i1] = np.inf
        i2 = D.argmin(1); v2 = D[r, i2]
        a2 = (q * q).sum(1)
        return np.stack([i1, i2], 1), np.sqrt(np.maximum(np.stack([v1 + a2, v2 + a2], 1), 0.0))

    parts = list(_MATCH_POOL.map(chunk, np.array_split(di, MATCH_CHUNKS)))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _overlap_ncc(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    m = valid > 0
    if m.sum() < 100:
        return 0.0
    x = a[m].astype(np.float64); y = b[m].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def warp_to_ref(img: np.ndarray, H: np.ndarray, shape: tuple[int, ...]
                ) -> tuple[np.ndarray, np.ndarray]:
    """The photo in reference pixels, and which of those pixels it covers.

    Edges are replicated rather than left black, so a window that runs a little
    past the photo does not pull brightness matching toward zero; the coverage
    map is what says a part was not photographed at all."""
    size = (shape[1], shape[0])
    warped = cv2.warpPerspective(img, H, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    cover = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), H, size,
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return warped, cover


def area_scale(reg: Registration) -> float:
    """Test-to-golden area scale of the alignment (1.0 = same distance)."""
    return float(abs(np.linalg.det(reg.H[:2, :2]))) if reg.ok else 0.0


# ------------------------------------------------------ landmark registration
# The fast path (config.PHOTO_REGISTRATION = "landmarks"). SIFT over the whole
# 1600 px copy is ~150 ms of a board on an M5 and would be most of a second on a
# Raspberry Pi 5. Commercial AOI aligns on fiducials instead: find a few known
# marks in small windows at full resolution and fit the transform to them.
#
# Landmarks here are chosen from golden 1 with nobody marking anything: corner-
# like patches (pads, silkscreen text, drill holes) away from every taught part,
# since parts are what go missing or move, spread over the board, and each
# unique within its own search window. With more than one golden, only the ones
# every other golden shot finds again (NCC >= LM_STABLE) are kept, which drops
# anything glare or shading changes.
#
# Per test photo: a rough alignment first (the last board's own alignment on a
# fixed rig, else SIFT on an 800 px copy, ~35 ms), then each landmark found to a
# sub-pixel in a +-LM_SEARCH px window of the test photo warped by that rough
# alignment, then a RANSAC homography through the landmark pairs. Anything that
# looks wrong falls back to register() above: too few landmarks found, a poor
# fit, or an aligned photo that does not lie on the reference.
#
# 2026-09-30, the two fixtures (handheld phone photos): ~40 ms against ~150 ms,
# fit RMS 2-2.5 px on test boards (a handheld shot is not one plane to a
# homography; per-part re-alignment takes the rest), same catches as SIFT,
# false calls 12 -> 13 on Test 2 and 14 -> 12 on Test 1. In-sample, so SIFT stays
# the default until held-out boards say otherwise.
LM_PATCH = 81                   # landmark patch side, px (odd)
LM_SEARCH = 24                  # +-px searched round the rough alignment's prediction
LM_GRID = 4                     # board split LM_GRID x LM_GRID for spread
LM_PER_CELL = 3
LM_MAX = 40                     # candidates tried per program
LM_UNIQUE = 0.7                 # a second match in the window at or above this: ambiguous
LM_SHARP = 0.9                  # 3 px off its own peak the match must fall below this
LM_STABLE = 0.85                # NCC every other golden shot must reach
LM_MATCH = 0.7                  # NCC a test photo's landmark must reach to be used
LM_MIN = 6                      # landmarks needed for a fit
LM_MAX_RMS = 3.0                # px; a worse fit falls back to SIFT
LM_PRIOR_RMS = 1.5              # px; a fit from the last board's alignment must beat this
LM_COARSE_SIDE = 800            # rough SIFT copy, long side px
LM_COARSE_MIN = 20              # rough SIFT inliers needed


@dataclass
class Landmarks:
    points: np.ndarray            # (N, 2) int, golden-1 full-resolution pixels
    grey: np.ndarray              # golden 1, float32
    coarse: Keypoints             # golden 1's SIFT on the LM_COARSE_SIDE copy
    last_H: np.ndarray | None = None      # the last board's alignment (a fixed rig's prior)


def _keypoints_at(grey_img: np.ndarray, side: int, nfeatures: int = 4000) -> Keypoints:
    k = min(1.0, side / max(grey_img.shape[:2]))
    s = cv2.resize(grey_img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else grey_img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kp, desc = cv2.SIFT_create(nfeatures=nfeatures).detectAndCompute(clahe.apply(s), None)
    return Keypoints(s, k, kp, desc)


def landmark_candidates(golden_bgr: np.ndarray, parts, board_mask: np.ndarray | None = None) -> np.ndarray:
    """Distinct, corner-like patch centres away from every part, spread over the board."""
    g = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    h = LM_PATCH // 2
    ok = np.ones((H, W), np.uint8)
    if board_mask is not None:
        ok &= cv2.resize(board_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    for p in parts:
        m = int(0.5 * max(p.w, p.h)) + h
        ok[max(0, p.y - m):p.y + p.h + m, max(0, p.x - m):p.x + p.w + m] = 0
    e = h + LM_SEARCH + 4
    ok[:e] = 0; ok[-e:] = 0; ok[:, :e] = 0; ok[:, -e:] = 0
    ok = cv2.erode(ok, np.ones((LM_PATCH, LM_PATCH), np.uint8))
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    k = (LM_PATCH, LM_PATCH)
    sxx, syy, sxy = (cv2.boxFilter(a, -1, k) for a in (gx * gx, gy * gy, gx * gy))
    tr = sxx + syy
    lmin = tr / 2 - np.sqrt(np.maximum(tr * tr / 4 - (sxx * syy - sxy * sxy), 0))   # Shi-Tomasi strength
    lmin[ok == 0] = 0
    small = lmin[::4, ::4]
    peaks = (small == cv2.dilate(small, np.ones((11, 11), np.uint8))) & (small > 0)
    ys, xs = np.nonzero(peaks)
    out: list[tuple[int, int, tuple[int, int]]] = []
    tried = 0
    for i in np.argsort(-small[ys, xs]):
        y, x = int(ys[i]) * 4, int(xs[i]) * 4
        cell = (min(LM_GRID - 1, y * LM_GRID // H), min(LM_GRID - 1, x * LM_GRID // W))
        if sum(1 for o in out if o[2] == cell) >= LM_PER_CELL:
            continue
        if any(abs(x - o[0]) < LM_PATCH and abs(y - o[1]) < LM_PATCH for o in out):
            continue
        tried += 1
        tpl = g[y - h:y + h + 1, x - h:x + h + 1]
        win = g[y - h - LM_SEARCH:y + h + LM_SEARCH + 1, x - h - LM_SEARCH:x + h + LM_SEARCH + 1]
        r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        c = LM_SEARCH
        peak = r == cv2.dilate(r, np.ones((5, 5), np.uint8))
        peak[c - 3:c + 4, c - 3:c + 4] = False
        second = float(r[peak].max()) if peak.any() else -1.0
        sharp = float(max(r[c, c - 3], r[c, c + 3], r[c - 3, c], r[c + 3, c]))
        if second < LM_UNIQUE and sharp < LM_SHARP:
            out.append((x, y, cell))
        if len(out) >= LM_MAX or tried > 400:
            break
    return np.array([(x, y) for x, y, _ in out], np.int32).reshape(-1, 2)


def _find(lm: Landmarks, img: np.ndarray, H0: np.ndarray, points: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray]:
    """For each landmark: the test point it was found at (via H0) and the match NCC."""
    from .features import subpixel
    h, n = LM_PATCH // 2, LM_PATCH // 2 + LM_SEARCH
    Hinv = np.linalg.inv(H0)
    src, q = [], []
    for x, y in points:
        T = np.array([[1, 0, -(x - n)], [0, 1, -(y - n)], [0, 0, 1]], np.float64) @ H0
        win = cv2.warpPerspective(img, T, (2 * n + 1, 2 * n + 1), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        r = cv2.matchTemplate(win, lm.grey[y - h:y + h + 1, x - h:x + h + 1], cv2.TM_CCOEFF_NORMED)
        _, v, _, loc = cv2.minMaxLoc(r)
        px, py = subpixel(r, loc)
        pt = Hinv @ np.array([x + px - LM_SEARCH, y + py - LM_SEARCH, 1.0])
        src.append(pt[:2] / pt[2])
        q.append(v)
    return np.float32(src).reshape(-1, 2), np.array(q, np.float32)


def make_landmarks(golden_bgr: np.ndarray, parts, other_goldens: list[np.ndarray] = (),
                   board_mask: np.ndarray | None = None) -> Landmarks:
    """Golden 1's landmarks, kept only where every other golden shot finds them again."""
    g = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY)
    lm = Landmarks(landmark_candidates(golden_bgr, parts, board_mask), g.astype(np.float32),
                   _keypoints_at(g, LM_COARSE_SIDE))
    worst = np.ones(len(lm.points), np.float32)
    for other in other_goldens:
        og = cv2.cvtColor(other, cv2.COLOR_BGR2GRAY)
        H0 = _coarse(lm, og)
        if H0 is None:
            continue
        worst = np.minimum(worst, _find(lm, og, H0, lm.points)[1])
    lm.points = lm.points[worst >= LM_STABLE]
    return lm


def _coarse(lm: Landmarks, img: np.ndarray) -> np.ndarray | None:
    """Rough test -> golden homography from SIFT on small copies, full-resolution pixels."""
    tk = _keypoints_at(img, LM_COARSE_SIDE)
    gk = lm.coarse
    if tk.desc is None or gk.desc is None or len(tk.kp) < LM_COARSE_MIN or len(gk.kp) < 2:
        return None
    idx, dist = _knn2(tk.desc, gk)
    good = np.nonzero(dist[:, 0] < 0.75 * dist[:, 1])[0]
    if len(good) < LM_COARSE_MIN:
        return None
    src = np.float32([tk.kp[q].pt for q in good])
    dst = np.float32([gk.kp[t].pt for t in idx[good, 0]])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or int(mask.sum()) < LM_COARSE_MIN:
        return None
    H = np.diag([1.0 / gk.k, 1.0 / gk.k, 1.0]) @ H @ np.diag([tk.k, tk.k, 1.0])
    return H / H[2, 2]


def _fit(lm: Landmarks, img: np.ndarray, H0: np.ndarray) -> tuple[np.ndarray | None, int, float]:
    """(homography, landmarks used, fit RMS px) from the landmarks found via H0."""
    if len(lm.points) < LM_MIN:
        return None, 0, float("inf")
    src, q = _find(lm, img, H0, lm.points)
    keep = q >= LM_MATCH
    if keep.sum() < LM_MIN:
        return None, int(keep.sum()), float("inf")
    H, mask = cv2.findHomography(src[keep], lm.points[keep].astype(np.float32), cv2.RANSAC, 2.0)
    if H is None or int(mask.sum()) < LM_MIN:
        return None, int(keep.sum()), float("inf")
    used = mask.ravel() > 0
    proj = cv2.perspectiveTransform(src[keep][used][None], H)[0]
    rms = float(np.sqrt(((proj - lm.points[keep][used]) ** 2).sum(1).mean()))
    return H / H[2, 2], int(used.sum()), rms


def register_landmarks(lm: Landmarks, img: np.ndarray, ref: Keypoints | None = None) -> Registration:
    """Test -> golden-1 homography from the landmarks, or register() when they
    cannot be trusted. `ref` is golden 1's full SIFT keypoints for that fallback.
    Registration.note says which way it went."""
    tried = []
    for how, H0 in (("last board's alignment", lm.last_H), ("rough SIFT", None)):
        if how == "rough SIFT":
            H0 = _coarse(lm, img)
            if H0 is None:
                tried.append("rough SIFT failed")
                break
        elif H0 is None:
            continue
        H, used, rms = _fit(lm, img, H0)
        limit = LM_PRIOR_RMS if how.startswith("last") else LM_MAX_RMS
        if H is None or rms > limit:
            tried.append(f"{how}: {used} landmark(s), fit {rms:.1f} px")
            continue
        det = float(np.linalg.det(H[:2, :2]))
        s, k = (ref.small, ref.k) if ref is not None else _small(lm.grey.astype(np.uint8))
        i_s = cv2.resize(img, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
        Hs = np.diag([k, k, 1.0]) @ H @ np.diag([1.0 / k, 1.0 / k, 1.0])
        fit = _overlap_ncc(s, cv2.warpPerspective(i_s, Hs, (s.shape[1], s.shape[0])),
                           cv2.warpPerspective(np.ones_like(i_s), Hs, (s.shape[1], s.shape[0])))
        if not 0.25 < det < 4.0 or fit < REG_MIN_OVERLAP_NCC:
            tried.append(f"{how}: aligned photo does not lie on the reference ({fit:.2f})")
            continue
        lm.last_H = H
        return Registration(H, used, len(lm.points), f"landmarks via {how}, fit {rms:.1f} px", fit)
    reg = register(ref if ref is not None else lm.grey.astype(np.uint8), img)
    reg.note = "; ".join(filter(None, [reg.note, "SIFT (landmarks: " + "; ".join(tried) + ")"]))
    return reg
