"""Measure the sampling pitch instead of asserting it.

`config.UM_PER_PX = 12.0` is a design intent, not a measurement. Every number
this machine reports in microns is that constant multiplied by a pixel count --
placement offset, body size, the lens perspective correction, the search window
that bounds the placement match. If the real pitch is 12.4, every one of those
is 3% wrong, consistently, in a way no amount of golden boards reveals: teach
and inspect share the error so it cancels everywhere except in the numbers a
person reads and the tolerances they are compared against.

The fix is a chrome-on-glass dot grid, which is already on the bill of materials
for exactly this. Photograph it at the board plane, measure the spacing in
pixels, divide by the certified spacing in mm. That is the whole method.

What earns its place here is the checking around it. A calibration that is
quietly wrong is worse than none, because it replaces a number somebody chose
with a number nobody looked at, so `measure_pitch` refuses rather than guesses
when the image does not support an answer: too few dots to be a grid, spacings
that disagree with each other, or a result far from what the optics can
physically produce.

Scope: this measures pitch. It does not map lens distortion, which is the other
half of the Phase 1 exit criterion in section 10 and wants the same target
photographed across the whole tile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import config

# A grid needs enough dots for the spacing to mean anything. Nine is three rows
# of three: fewer, and one bad blob moves the answer.
MIN_DOTS = 9

# How far the measured pitch may sit from the design intent before it is
# reported as a failure rather than a calibration. The optics cannot be 25% off
# and still be in focus, so a result outside this is a detection problem -- the
# wrong target, the wrong magnification, or blobs that are not dots.
MAX_DEVIATION = 0.25

# Spacings between neighbouring dots should agree closely on a chrome-on-glass
# target. They will not if the blobs found are not a regular grid.
MAX_SPACING_CV = 0.05


@dataclass(frozen=True)
class Calibration:
    """What one target photograph says about the sampling pitch."""
    um_per_px: float
    dots: int
    spacing_px: float
    spacing_cv: float          # coefficient of variation across neighbour gaps
    deviation: float           # fraction away from config.UM_PER_PX

    @property
    def mm_per_px(self) -> float:
        return self.um_per_px / 1000.0

    def report(self) -> str:
        return (f"{self.um_per_px:.3f} um/px from {self.dots} dots "
                f"({self.spacing_px:.2f} px spacing, {self.spacing_cv:.1%} "
                f"variation, {self.deviation:+.1%} against the "
                f"{config.UM_PER_PX:.1f} design intent)")


class CalibrationError(RuntimeError):
    """The image does not support a pitch measurement."""


def find_dots(image: np.ndarray, min_area_px: float = 12.0) -> np.ndarray:
    """Centroids of the dark dots in a calibration target image, (N, 2) in px.

    Sub-pixel, from image moments rather than from bounding boxes: a bounding
    box quantises to whole pixels, and the whole point of measuring a grid
    rather than one dot is to average away errors smaller than that.
    """
    img = image
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    # Chrome on glass is a hard binary edge; Otsu finds it without a threshold
    # anyone has to tune per shoot.
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):                      # 0 is the background
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        x, y = centroids[i]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # Dots are round. A long thin blob is an edge, a scratch or the frame.
        if min(w, h) == 0 or max(w, h) / min(w, h) > 2.0:
            continue
        out.append((float(x), float(y)))
    return np.array(out, np.float64).reshape(-1, 2)


def _nearest_neighbour_spacings(pts: np.ndarray) -> np.ndarray:
    """Distance from each dot to its closest neighbour."""
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                 pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def measure_pitch(image: np.ndarray, dot_pitch_mm: float,
                  min_area_px: float = 12.0) -> Calibration:
    """Sampling pitch in um/px, from one photograph of a dot grid.

    `dot_pitch_mm` is the target's certified centre-to-centre spacing -- the
    number on its certificate, not a nominal one. The whole measurement is only
    as good as that figure.

    Raises `CalibrationError` rather than returning a number it cannot stand
    behind. That is the point of the function: a silently wrong pitch scales
    every micron this machine reports, consistently, and cancels out of every
    golden-board comparison, so nothing downstream will ever contradict it.
    """
    if dot_pitch_mm <= 0:
        raise ValueError(f"dot pitch must be positive, got {dot_pitch_mm}")

    pts = find_dots(image, min_area_px)
    if len(pts) < MIN_DOTS:
        raise CalibrationError(
            f"found {len(pts)} dots, need at least {MIN_DOTS}. Check the target "
            f"is in frame and in focus, and that the image is not inverted -- "
            f"this expects dark dots on a light ground.")

    gaps = _nearest_neighbour_spacings(pts)
    spacing = float(np.median(gaps))
    cv_ = float(np.std(gaps) / max(spacing, 1e-9))
    if cv_ > MAX_SPACING_CV:
        raise CalibrationError(
            f"neighbour spacings vary by {cv_:.1%}, over the {MAX_SPACING_CV:.0%} "
            f"a regular grid should show. The blobs found are not evenly spaced, "
            f"so something other than the target is being measured.")

    um_per_px = dot_pitch_mm * 1000.0 / spacing
    deviation = um_per_px / config.UM_PER_PX - 1.0
    if abs(deviation) > MAX_DEVIATION:
        raise CalibrationError(
            f"measured {um_per_px:.2f} um/px, {deviation:+.0%} from the "
            f"{config.UM_PER_PX:.1f} the optics were designed for. That is too "
            f"far to be a calibration and is more likely the wrong target pitch "
            f"({dot_pitch_mm} mm), the wrong magnification, or blobs that are "
            f"not dots.")

    return Calibration(um_per_px=um_per_px, dots=len(pts), spacing_px=spacing,
                       spacing_cv=cv_, deviation=deviation)


def apply(cal: Calibration) -> None:
    """Adopt a measured pitch for this process.

    Deliberately explicit rather than automatic on import. Every micron the
    machine reports moves when this is called, so it belongs at a point somebody
    chose -- and the measured value should be written into the station's
    configuration afterwards rather than re-derived on every run from whichever
    target photograph happens to be lying about.
    """
    config.UM_PER_PX = cal.um_per_px
    config.MM_PER_PX = cal.um_per_px / 1000.0


def render_target(dot_pitch_mm: float = 1.0, rows: int = 7, cols: int = 9,
                  um_per_px: float = config.UM_PER_PX,
                  dot_fraction: float = 0.35) -> np.ndarray:
    """A synthetic dot grid at a known pitch, for testing the measurement.

    Not a stand-in for the real target -- it exists so the arithmetic can be
    checked against a ground truth that is known exactly, which a photograph
    never is. If `measure_pitch` cannot recover the pitch from an image this
    clean, it will not recover it from glass.
    """
    mm_per_px = um_per_px / 1000.0
    pitch_px = dot_pitch_mm / mm_per_px
    margin = pitch_px
    h = int(round((rows - 1) * pitch_px + 2 * margin))
    w = int(round((cols - 1) * pitch_px + 2 * margin))
    img = np.full((h, w), 235, np.uint8)
    r = max(1.5, pitch_px * dot_fraction / 2.0)
    # Sub-pixel centres, via cv2's fixed-point `shift`. Rounding them to whole
    # pixels quantises the very spacing this target exists to certify: at
    # 12 um/px a 1 mm pitch is 83.33 px, which rounds to 83 and hands the
    # measurement a 0.4% error that belongs to the target rather than to the
    # method being tested.
    SH = 4                                     # 1/16 px
    S = 1 << SH
    for j in range(rows):
        for i in range(cols):
            cx = margin + i * pitch_px
            cy = margin + j * pitch_px
            cv2.circle(img, (int(round(cx * S)), int(round(cy * S))),
                       int(round(r * S)), 20, -1, lineType=cv2.LINE_AA, shift=SH)
    return img
