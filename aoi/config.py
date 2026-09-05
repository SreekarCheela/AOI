"""Optical and inspection constants, carried straight from the design spec (docs/aoi-station-one.html)."""
from __future__ import annotations

# ---- optics (spec section 2) -------------------------------------------------
UM_PER_PX: float = 12.0                 # sampling pitch at the board plane
MM_PER_PX: float = UM_PER_PX / 1000.0
TILE_PX: tuple[int, int] = (4096, 3000)  # IMX304 active area

# ---- illumination (spec section 1) -------------------------------------------
# A specular surface tilted alpha from horizontal reflects into a vertical
# camera only when light arrives at elevation theta = 90 - 2*alpha.
# Inverted, each ring reports on one slope band.
CHANNELS: tuple[str, ...] = ("low", "mid", "high", "diffuse")
ANGLED_CHANNELS: tuple[str, ...] = ("low", "mid", "high")
ELEVATION_DEG: dict[str, float] = {"low": 20.0, "mid": 45.0, "high": 72.0}
SLOPE_SIGMA_DEG: float = 7.0            # angular acceptance of a ring


def slope_for_channel(channel: str) -> float:
    """Surface slope (deg from horizontal) that this ring reflects into the camera."""
    return (90.0 - ELEVATION_DEG[channel]) / 2.0


SLOPE_DEG: dict[str, float] = {c: slope_for_channel(c) for c in ANGLED_CHANNELS}

# ---- inspection --------------------------------------------------------------
PATCH_PX: int = 96              # model input; CAD-cropped, alignment-normalised
SEARCH_UM: float = 300.0        # local alignment search half-window
BAND_THRESHOLD: float = 0.35    # normalised intensity counted as "lit" in a channel

# placement tolerances (fail thresholds for the rule layer)
MAX_OFFSET_UM: float = 150.0
MAX_SKEW_DEG: float = 12.0

# ---- presence gating ---------------------------------------------------------
# Body correlation is scored against the golden mean for that package, never a
# global constant. Golden NCC is a property of the package, not the machine: an
# 0603 sits at 0.92 while a SOD-123 sits at 0.99, so one absolute threshold
# cannot mean "the body has moved" for both. A fraction of the taught mean does.
#
# Only one level survives: ABSENT, used as a second opinion on "no body at all"
# alongside the body segmentation below. The old FAIL level tried to say "the
# body is not on its footprint" and was retired when body localisation replaced
# it -- correlation cannot say WHERE the body went, and that was the question.
# ABSENT is the second opinion the body segmentation is checked against before a
# part is called missing. It is reached ONLY when the segmentation found no body
# at all, so the question it answers is narrow: does correlation still see a good
# part, meaning the segmentation failed rather than the part being absent?
#
# That is a wide window, which is why this is not the delicate number it once
# was. Measured across every package: a genuinely missing part correlates at
# 0.40 to 0.60 of golden, because the bare lands are there either way and on a
# small-bodied package like SOT-23-6 they dominate the crop; a good part sits at
# ~1.0. 0.78 leaves 30% clear of the missing population and 22% clear of good.
#
# It used to sit at 0.60 and separate "absent" from "walked off its lands", which
# was a 0.572-to-0.612 window and far too tight to hold. Body localisation took
# that job away: a displaced part still HAS a body, so it never reaches here.
NCC_ABSENT_FRACTION: float = 0.78

# ---- fillet grading ----------------------------------------------------------
# Joint quality is scored as a Mahalanobis distance of the pad's three band
# fractions from the golden distribution for that package (spec section 5),
# not as a per-band z-score.
#
# The three bands are strongly correlated on good joints (r = 0.5 to 0.85
# measured across all five package families), and that correlation is the whole
# point. Anything that changes how much light reaches a component -- vignetting,
# a dimmer corner, LED drift, board finish -- moves all three bands together,
# along the axis where the covariance is large, so it costs almost no distance.
# A genuinely starved joint loses the steep face while keeping the shallow one,
# which moves ACROSS that axis where the covariance is small, and the distance
# is large. A univariate test on the low band alone cannot tell the two apart,
# which is why illumination non-uniformity is the largest single source of
# false calls in a rule-based AOI layer.
#
# The distance is already normalised by the covariance, so unlike the NCC gate
# this threshold is package-relative by construction and a fixed number is the
# right shape for it.
#
# Set from measurement, not from a chi-square table. Over 2160 good pad-tests
# the distance ran median 1.56, p99 3.50, p99.9 4.03, max 4.63; a fully starved
# joint scores 36.8 at worst. That leaves a wide empty band between the two
# populations, so the threshold sits above the observed good maximum with ~23%
# margin while still being 6x below the nearest real defect. At 500 components
# (1000 pad tests) per board that budgets well under 0.1 false calls per board
# to this test, leaving room under the 1.0/board target in section 8.
#
# Re-derive this against real golden boards once they exist. Synthetic joints
# are tidier than real ones and the good-side tail will be fatter.
BAND_MAHALANOBIS_FAIL: float = 6.0

# Regularisation for the 3x3 covariance. Not optional: teach runs on 5-20
# boards, so the covariance is estimated from very few samples, and bands
# correlated at 0.8 leave one near-collapsed axis whose small eigenvalue a plain
# inverse turns into an enormous distance. Unregularised, the test meant to cut
# false calls becomes a source of them.
BAND_COV_RIDGE: float = 0.15    # shrink off-diagonals toward a diagonal matrix
BAND_COV_FLOOR: float = 0.01    # absolute floor on per-band std, as for band_std

DEFECT_CLASSES: tuple[str, ...] = (
    "ok", "missing", "offset", "skew", "reversed", "insufficient", "bridge", "tombstone",
)

# ---- body localisation -------------------------------------------------------
# Where the body actually is on the crop, as opposed to how well it correlates.
# NCC answers "does this look like a good part" and collapses everything that is
# not into one low number; it cannot tell an absent body from one sitting
# somewhere else, which is why missing/displaced and tombstone/displaced both
# resisted every threshold tried on it. Segmenting the body and measuring its
# position and extent answers the question directly.
#
# The body is the one large region that is dark in the high-angle channel (so
# not copper, not solder) yet clearly brighter than bare substrate in the
# diffuse channel. Both thresholds are learned per package at teach, midway
# between the two populations they separate.

# The body is segmented on specular_max = max(low, mid, high) rather than the
# high channel alone, so solder lit in ANY ring is excluded rather than only
# solder lying flat. That level is a property of the material: measured over
# every package family the body sits at 0.082-0.086, a polarity marking at
# 0.114-0.118, and copper or solder at 0.62-0.72. The threshold is this multiple
# of the body level measured at teach, which lands above the marking (so a
# marking does not split the body in two) and far below any metal.
#
# It has to be learned this way rather than from a bright reference in the crop:
# on SOD-123 and TANT-A the body nearly covers the land pattern, so there is no
# bare copper left to sample and a midpoint rule collapses onto the body level.
BODY_SPECULAR_MULT: float = 2.2

# Blob area as a fraction of the golden blob. Below this there is no body in the
# crop at all. Measured: a missing part scores 0.00 (no component survives the
# open), the smallest real body is a tombstoned 0402 at 0.29.
BODY_PRESENT_AREA: float = 0.12

# Body extent along the pad axis divided by the body's own width. A part
# standing on one land spans exactly its width along that axis whatever the
# package, so every tombstone lands near 1.0; a part lying flat spans its length
# and lands at the package aspect ratio, 1.6 to 2.0 here.
#
# Normalising by the body's WIDTH rather than its golden LENGTH is what makes a
# single threshold work. Against golden length, a tombstone (0.22-0.61) collides
# with a part displaced far enough to be clipped by the crop edge (0.758 on an
# 0402), which measures short because the crop cut it, not because it stood up.
# Against its own width those separate cleanly: tombstones 0.42-1.00, everything
# lying flat 1.45-2.01, and this sits at the geometric midpoint with ~17% margin
# on each side.
BODY_STANDING_ASPECT: float = 1.20
