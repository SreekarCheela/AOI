"""Package library: body and land geometry per package, in mm.

In production this is generated from datasheets (spec section 6); here it is
hand-entered for the handful of two-terminal packages the prototype renders.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Package:
    name: str
    body_l: float       # length, along the pad-to-pad axis
    body_w: float       # width, across
    body_h: float       # height above board
    pad_l: float        # land length along the axis
    pad_w: float        # land width across
    pad_gap: float      # inner-edge to inner-edge gap between the two lands
    polarized: bool = False
    n_pads: int = 2

    @property
    def pad_pitch(self) -> float:
        """Centre-to-centre distance between the two lands."""
        return self.pad_gap + self.pad_l

    @property
    def span(self) -> float:
        """Total land-pattern length."""
        return self.pad_gap + 2 * self.pad_l

    def footprint(self, margin: float = 0.35) -> tuple[float, float]:
        """Bounding box (length, width) to crop for inspection, with margin."""
        return (self.span + 2 * margin, max(self.body_w, self.pad_w) + 2 * margin)


LIBRARY: dict[str, Package] = {
    p.name: p for p in (
        Package("0402", body_l=1.00, body_w=0.50, body_h=0.35,
                pad_l=0.50, pad_w=0.60, pad_gap=0.50),
        Package("0603", body_l=1.60, body_w=0.80, body_h=0.45,
                pad_l=0.75, pad_w=0.90, pad_gap=0.70),
        Package("0805", body_l=2.00, body_w=1.25, body_h=0.55,
                pad_l=0.90, pad_w=1.30, pad_gap=0.90),
        Package("SOD-123", body_l=2.70, body_w=1.60, body_h=1.00,
                pad_l=0.90, pad_w=1.20, pad_gap=1.40, polarized=True),
        Package("TANT-A", body_l=3.20, body_w=1.60, body_h=1.60,
                pad_l=1.20, pad_w=1.40, pad_gap=1.20, polarized=True),
    )
}


def get(name: str) -> Package:
    try:
        return LIBRARY[name]
    except KeyError:
        raise KeyError(f"package {name!r} not in library; known: {sorted(LIBRARY)}") from None
