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
