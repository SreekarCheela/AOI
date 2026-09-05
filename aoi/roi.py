"""Regions of interest, derived from CAD + package geometry rather than found.

Everything here is the pipeline's answer to "where is this component and which
pixels are its joints" -- computed, never searched for.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .board import Component
from .packages import Package


def local_to_board(comp: Component, pts: np.ndarray) -> np.ndarray:
    """Map component-local mm (origin at centre, +x along the pad axis) to board mm."""
    c, s = np.cos(np.radians(comp.rot)), np.sin(np.radians(comp.rot))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ R.T + np.array([comp.x, comp.y])


def body_rect(pkg: Package) -> np.ndarray:
    """Component-local half-extents of the body window."""
    return np.array([pkg.body_l / 2.0, pkg.body_w / 2.0])


def land_boxes(pkg: Package) -> list[tuple[np.ndarray, np.ndarray]]:
    """(centre, half-extent) of every land, component-local mm."""
    return [(np.array([l.cx, l.cy]), np.array([l.hx, l.hy])) for l in pkg.lands()]


def gap_boxes(pkg: Package) -> list[tuple[np.ndarray, np.ndarray]]:
    """The regions between neighbouring lands -- where solder must NOT be.

    On a two-terminal part this is the single space across the body, which is
    what the bridge test has always measured. On a lead row it is every
    consecutive pair, and those gaps are the ones that actually bridge: a
    0.8 mm-pitch QFP leaves ~0.4 mm between leads against ~6 mm across the body.
    Shrunk to 80% so a fillet creeping to the very edge of its own land does not
    count as solder in the gap.
    """
    out = []
    for a, b in pkg.adjacent_pairs():
        centre = np.array([(a.cx + b.cx) / 2.0, (a.cy + b.cy) / 2.0])
        # the gap spans inner edge to inner edge along the line joining them,
        # and is as wide as the narrower land across it
        dx, dy = abs(b.cx - a.cx), abs(b.cy - a.cy)
        if dx >= dy:
            half = np.array([max(dx / 2.0 - min(a.hx, b.hx), 1e-3),
                             min(a.hy, b.hy)])
        else:
            half = np.array([min(a.hx, b.hx),
                             max(dy / 2.0 - min(a.hy, b.hy), 1e-3)])
        out.append((centre, half * 0.8))
    return out


# ------------------------------------------------------------------- cropping
def _affine(comp: Component, mm_per_px: float, out_px: int, span_mm: float) -> np.ndarray:
    """Affine taking board pixels -> a canonical, axis-aligned component patch."""
    scale = out_px / (span_mm / mm_per_px)          # px_out per px_in
    cx, cy = comp.x / mm_per_px, comp.y / mm_per_px
    a = np.radians(-comp.rot)                        # de-rotate
    c, s = np.cos(a) * scale, np.sin(a) * scale
    M = np.array([[c, -s, 0.0], [s, c, 0.0]], dtype=np.float64)
    M[0, 2] = out_px / 2.0 - (c * cx - s * cy)
    M[1, 2] = out_px / 2.0 - (s * cx + c * cy)
    return M


def extract_patch(stack: np.ndarray, comp: Component, pkg: Package,
                  mm_per_px: float = config.MM_PER_PX,
                  out_px: int = config.PATCH_PX) -> np.ndarray:
    """Scale- and rotation-normalised (out_px, out_px, C) crop for the models.

    Every package lands at the same pixel size, so one model per package family
    sees a canonical view regardless of part size or placement angle.
    """
    span = max(pkg.footprint())
    M = _affine(comp, mm_per_px, out_px, span)
    chans = [cv2.warpAffine(stack[..., i], M, (out_px, out_px),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
             for i in range(stack.shape[2])]
    return np.stack(chans, axis=-1)


def extract_region(stack: np.ndarray, comp: Component, pkg: Package,
                   mm_per_px: float = config.MM_PER_PX,
                   margin_mm: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Native-resolution, de-rotated crop plus the affine used, for rule measurement.

    Rules measure in microns, so unlike `extract_patch` this keeps 1:1 scale.
    """
    fl, fw = pkg.footprint(margin_mm)
    w = int(round(fl / mm_per_px))
    h = int(round(fw / mm_per_px))
    a = np.radians(-comp.rot)
    c, s = np.cos(a), np.sin(a)
    cx, cy = comp.x / mm_per_px, comp.y / mm_per_px
    M = np.array([[c, -s, w / 2.0 - (c * cx - s * cy)],
                  [s,  c, h / 2.0 - (s * cx + c * cy)]], dtype=np.float64)
    chans = [cv2.warpAffine(stack[..., i], M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
             for i in range(stack.shape[2])]
    return np.stack(chans, axis=-1), M


def local_box_px(centre_mm: np.ndarray, half_mm: np.ndarray, shape: tuple[int, int],
                 mm_per_px: float = config.MM_PER_PX) -> tuple[slice, slice]:
    """Component-local mm box -> slices into a de-rotated region crop."""
    h, w = shape[:2]
    cx, cy = w / 2.0 + centre_mm[0] / mm_per_px, h / 2.0 + centre_mm[1] / mm_per_px
    hx, hy = half_mm[0] / mm_per_px, half_mm[1] / mm_per_px
    x0, x1 = int(round(cx - hx)), int(round(cx + hx))
    y0, y1 = int(round(cy - hy)), int(round(cy + hy))
    return (slice(max(0, y0), min(h, max(y0 + 1, y1))),
            slice(max(0, x0), min(w, max(x0 + 1, x1))))
