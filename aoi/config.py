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
# The two levels answer different questions, which is why the gate needs both:
#   FAIL   -- the body is not on its footprint (moved, stood up, or gone)
#   ABSENT -- there is no body in the window at all
# A part between them has moved but is still present, which is a placement
# fault, so it falls through to the offset and skew tests to be named properly.
NCC_FAIL_FRACTION: float = 0.85
NCC_ABSENT_FRACTION: float = 0.60

# A tombstone stands on one land with that joint fully formed; the other land is
# bare. A part that merely walked off its pads has BOTH joints degraded. This is
# the fraction of the golden low-angle band the surviving joint must still reach
# for "stood up" to beat "displaced".
TOMBSTONE_SURVIVING_BAND: float = 0.75

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

# Used only before any golden model exists (rule layer running from a cold
# start). Package-relative gating is not available then.
NCC_FAIL_UNTAUGHT: float = 0.62

DEFECT_CLASSES: tuple[str, ...] = (
    "ok", "missing", "offset", "skew", "reversed", "insufficient", "bridge", "tombstone",
)
