"""Capture checks, each with a reason a person can act on.

Two stages, because part boxes and scale exist only after teach:

    golden upload   focus, a board in the frame with enough pixels on it, registers onto golden 1
    test photo      focus against the goldens, same scale as the goldens
                    (per-part glare is judged per part, in judge.py)
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from .io import grey


def _work(bgr: np.ndarray) -> np.ndarray:
    s = min(1.0, config.PHOTO_WORK_SIDE / max(bgr.shape[:2]))
    g = grey(bgr)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g


def sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian on a fixed-size copy, so it compares across cameras."""
    return float(cv2.Laplacian(_work(bgr), cv2.CV_64F).var())


def _quick(img: np.ndarray) -> np.ndarray:
    """Grey copy reduced by a whole factor to about PHOTO_WORK_SIDE. A whole
    factor takes OpenCV's fast area path: ~1 ms for a 12 MP photo, against
    ~45 ms (several times that on a Raspberry Pi) for the 2.52x of _work."""
    g = grey(img) if img.ndim == 3 else img
    f = max(1, round(max(g.shape[:2]) / config.PHOTO_WORK_SIDE))
    return cv2.resize(g, None, fx=1.0 / f, fy=1.0 / f, interpolation=cv2.INTER_AREA) if f > 1 else g


def relative_sharpness(img: np.ndarray) -> float:
    """sharpness() on the _quick copy, for comparing a test photo with goldens
    from the same camera (the goldens' value must come from here too). Its
    scale differs from sharpness(), whose absolute limit stays at teach."""
    return float(cv2.Laplacian(_quick(img), cv2.CV_64F).var())


def board_mask(bgr: np.ndarray, side: int = config.PHOTO_WORK_SIDE) -> np.ndarray:
    """1 on the board, 0 off it, on a copy whose long side is `side` px.

    The board is the largest soldermask-green region plus everything it
    encloses (parts, lands, silkscreen, drilled holes). Anything not green that
    connects to the frame border is background, and that includes notches and
    cutouts, which open onto the outside. A convex hull of the green would fill
    them in, and the desk seen through a cutout then looked like a part.
    """
    s = min(1.0, side / max(bgr.shape[:2]))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    green = cv2.morphologyEx((lab[..., 1] < 120).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, st, _ = cv2.connectedComponentsWithStats(green)
    if n < 2:
        return np.zeros(small.shape[:2], np.uint8)
    board = (labels == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    # outside = non-board pixels reachable from the frame border
    free = cv2.copyMakeBorder(1 - board, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    cv2.floodFill(free, np.zeros((free.shape[0] + 2, free.shape[1] + 2), np.uint8), (0, 0), 2)
    return (free[1:-1, 1:-1] != 2).astype(np.uint8)


def board_fill(bgr: np.ndarray) -> float:
    """Fraction of the frame the board covers, cutouts excluded."""
    return float(board_mask(bgr).mean())


def board_width_px(bgr: np.ndarray) -> float:
    """Long side of the board's minimum-area rectangle, in full-resolution pixels."""
    m = board_mask(bgr)
    if not m.any():
        return 0.0
    s = m.shape[1] / bgr.shape[1]
    pts = np.column_stack(np.nonzero(m))[:, ::-1].astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return max(w, h) / s


def golden_problems(bgr: np.ndarray) -> list[str]:
    out = []
    sh = sharpness(bgr)
    if sh < config.PHOTO_MIN_SHARPNESS:
        out.append(f"out of focus (sharpness {sh:.0f}, need {config.PHOTO_MIN_SHARPNESS:.0f}); "
                   "hold still, tap to focus on the board")
    fill = board_fill(bgr)
    if fill < config.PHOTO_MIN_FILL:
        out.append(f"board fills {fill:.0%} of the frame (need {config.PHOTO_MIN_FILL:.0%}); "
                   "is the board in the photo? move closer")
    side = board_width_px(bgr)
    if side and side < config.PHOTO_MIN_BOARD_PX:
        out.append(f"board is only {side:.0f} px long (need {config.PHOTO_MIN_BOARD_PX}); "
                   "move closer or use a higher resolution camera")
    return out


def test_problems(bgr: np.ndarray, golden_sharpness: float, area_scale: float) -> list[str]:
    """golden_sharpness: the goldens' relative_sharpness. `bgr` may be the grey photo."""
    out = []
    sh = relative_sharpness(bgr)
    if golden_sharpness and sh < config.PHOTO_REL_SHARPNESS * golden_sharpness:
        out.append(f"blurrier than the goldens (sharpness {sh:.0f} vs {golden_sharpness:.0f}); reshoot")
    if area_scale and abs(area_scale - 1.0) > config.PHOTO_SCALE_TOL:
        out.append(f"shot from a different distance or resolution (scale {area_scale:.2f}x the goldens, "
                   f"allowed ±{config.PHOTO_SCALE_TOL:.0%}); match the golden distance and camera resolution")
    return out
