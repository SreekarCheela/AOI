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

DEFECT_CLASSES: tuple[str, ...] = (
    "ok", "missing", "offset", "skew", "reversed", "insufficient", "bridge", "tombstone",
)
