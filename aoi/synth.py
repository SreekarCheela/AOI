"""Synthetic four-channel board renderer.

Exists so the pipeline can be developed and validated before any hardware
arrives. It is not a pretty picture generator -- it implements the reflection
model the whole machine rests on (spec section 1):

    a specular surface tilted `alpha` from horizontal reflects into a vertical
    camera only when light arrives at elevation `theta = 90 - 2*alpha`

so each ring reports on one slope band, and a solder fillet -- whose slope rises
from shallow at the pad toe to steep at the termination -- produces an ordered
high -> mid -> low sweep. Starve the fillet and the low band collapses. That is
the signal `rules.fillet_bands` measures, and rendering it honestly here is what
makes the prototype worth anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config, packages
from .board import Board, Component
from .packages import Package

# Material appearance: (diffuse albedo, specular strength).
# These have to be genuinely separated, not just plausible. Chip bodies are dark
# ceramic or black plastic; solder is bright and mirror-like. If body and solder
# sit at similar albedo the diffuse channel becomes a near-uniform blob, the body
# match template loses its structure, and placement measurement silently returns
# zero displacement for every part.
SUBSTRATE = (0.20, 0.06)
COPPER = (0.42, 0.55)
SOLDER = (0.60, 1.00)
BODY = (0.26, 0.13)
MARKING = (0.78, 0.06)
AMBIENT = 0.12          # diffuse leak into the angled channels

# Peak slope a fillet reaches at the termination. This -- not body height -- is
# what solder volume and wetting angle actually set, and it is the quantity the
# three rings resolve. A healthy joint sweeps past all three bands; a starved
# one tops out below the low-angle band and never lights it.
FILLET_SLOPE_OK_DEG = 52.0
FILLET_SLOPE_STARVED_DEG = 22.0
FILLET_POWER = 2.0

# Excess solder is not a scaled-up fillet, and rendering it as one would be
# rendering the wrong physics. Solder wets outward only as far as the flux and
# the land allow; volume beyond that piles up instead, and the surface inverts
# from a concave ramp -- shallow at the toe, steep at the termination -- into a
# convex cap that is nearly FLAT over its apex and steep only at the rim.
#
# The consequence the inspector has to deal with is that the cap costs the
# mid-slope face, so a balled joint's low band collapses exactly as a starved
# one's does. What is left to separate them is where the solder went: the cap
# spills past the land onto bare substrate.
EXCESS_SPILL = 0.45      # cap overhang past the land edge, x the land half-extent
EXCESS_WIDEN = 1.30      # cap half-width across the land, x the land half-width
EXCESS_DOME = 0.55       # cap height, x its own half-run -- sets the rim slope


@dataclass
class Defect:
    """A defect to inject into one component."""
    kind: str = "ok"
    pad: int = 0                      # which land, where it matters
    dx: float = 0.0                   # mm
    dy: float = 0.0                   # mm
    dtheta: float = 0.0               # deg
    substitute: str = ""              # kind="wrong_part": the package actually
                                      # fitted, whose BODY is rendered on this
                                      # component's lands. The lands belong to
                                      # the board and do not change when the
                                      # feeder does.

    @property
    def is_ok(self) -> bool:
        return self.kind == "ok"


@dataclass
class RenderOpts:
    noise: float = 0.012
    gain: float = 1.0
    vignette: float = 0.06            # slow illumination falloff across the board
    seed: int = 0
    place_sigma_um: float = 18.0      # residual placement scatter on good joints
    board_shift_mm: tuple[float, float] = (0.0, 0.0)   # global misregistration
    board_rot_deg: float = 0.0
    fields: dict = field(default_factory=dict)


# --------------------------------------------------------------------- helpers
def _rect(X: np.ndarray, Y: np.ndarray, cx: float, cy: float,
          hx: float, hy: float) -> np.ndarray:
    return (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy)


def _dome_slope(u: np.ndarray, v: np.ndarray, a: float, b: float,
                height: float) -> np.ndarray:
    """Slope (deg) of a spherical cap of half-axes (a, b) piled `height` high.

    h = height * sqrt(1 - (u/a)^2 - (v/b)^2), so the surface is FLAT at the apex
    and turns vertical at the rim -- the exact inverse of the concave ramp
    `_fillet_slope` draws. A joint carrying more solder than the land can wet
    stops spreading and starts piling, and this is the shape it takes.
    """
    inside = np.clip(1.0 - (u / a) ** 2 - (v / b) ** 2, 0.0, 1.0)
    # Guard the rim: the true gradient diverges there and arctan would saturate
    # anyway, so clamping the denominator costs nothing and avoids a warning.
    grad = height * np.sqrt((u / a ** 2) ** 2 + (v / b ** 2) ** 2) / np.maximum(
        np.sqrt(inside), 1e-3)
    return np.degrees(np.arctan(grad))


def _fillet_slope(X: np.ndarray, x_toe: float, x_term: float,
                  h_term: float, power: float = 2.0) -> np.ndarray:
    """Slope (deg) of a concave fillet running from toe to termination.

    h(t) = h_term * t**power with t the normalised run, so the surface is
    shallow where solder meets bare pad and steep where it climbs the
    termination -- the ordering the three rings decode.
    """
    run = x_term - x_toe
    if abs(run) < 1e-6:
        return np.zeros_like(X)
    t = np.clip((X - x_toe) / run, 0.0, 1.0)
    dhdx = h_term * power * np.power(t, power - 1.0) / abs(run)
    return np.degrees(np.arctan(dhdx))


class _Patch:
    """Per-pixel albedo / specular / slope maps for one component footprint."""

    def __init__(self, pkg: Package, mm_per_px: float):
        fl, fw = pkg.footprint(margin=0.45)
        self.w = int(round(fl / mm_per_px))
        self.h = int(round(fw / mm_per_px))
        xs = (np.arange(self.w) - self.w / 2.0 + 0.5) * mm_per_px
        ys = (np.arange(self.h) - self.h / 2.0 + 0.5) * mm_per_px
        self.X, self.Y = np.meshgrid(xs, ys)
        self.albedo = np.full((self.h, self.w), SUBSTRATE[0], np.float32)
        self.spec = np.full((self.h, self.w), SUBSTRATE[1], np.float32)
        self.slope = np.zeros((self.h, self.w), np.float32)

    def paint(self, mask: np.ndarray, material: tuple[float, float],
              slope: np.ndarray | float = 0.0) -> None:
        self.albedo[mask] = material[0]
        self.spec[mask] = material[1]
        self.slope[mask] = slope[mask] if isinstance(slope, np.ndarray) else slope


def _render_component(pkg: Package, defect: Defect, rng: np.random.Generator,
                      mm_per_px: float, place_sigma_um: float) -> _Patch:
    p = _Patch(pkg, mm_per_px)
    X, Y = p.X, p.Y
    lands = pkg.lands()
    # The lands are etched into the board and do not change when the wrong reel
    # goes into the feeder; only the body sitting on them does. So the footprint,
    # the crop and every land above come from the EXPECTED package, and the body
    # geometry below comes from whatever was actually fitted.
    fitted = packages.get(defect.substitute) if (
        defect.kind == "wrong_part" and defect.substitute) else pkg
    body_hx, body_hy = fitted.body_l / 2.0, fitted.body_w / 2.0

    # --- lands ---------------------------------------------------------------
    for l in lands:
        p.paint(_rect(X, Y, l.cx, l.cy, l.hx, l.hy), COPPER)

    if defect.kind == "missing":
        return p                                   # bare copper, nothing else

    # small real-world placement scatter on otherwise good parts
    jx = rng.normal(0.0, place_sigma_um / 1000.0)
    jy = rng.normal(0.0, place_sigma_um / 1000.0)
    ox, oy, oth = defect.dx + jx, defect.dy + jy, defect.dtheta

    if defect.kind == "tombstone" and pkg.is_two_terminal:
        # One end lifted: the body stands on its surviving joint, the other land
        # is left bare. The surviving fillet is built with the same signed-axis
        # convention as a normal joint so it lands inside the pad's fillet window
        # and still reads as wetted -- that one lit joint is exactly what
        # separates a tombstone from a part that was never placed.
        d = pkg.pad_pitch / 2.0
        pad_hx = pkg.pad_l / 2.0
        sgn = -1.0 if defect.pad == 0 else 1.0
        stand_x = sgn * d
        p.paint(_rect(X, Y, stand_x, 0.0, pkg.body_w / 2.0, body_hy), BODY)
        x_toe = sgn * (d + pad_hx * 0.95)
        x_term = sgn * (d - pad_hx * 0.20)
        run = abs(x_term - x_toe)
        h = min(np.tan(np.radians(FILLET_SLOPE_OK_DEG)) * run / FILLET_POWER,
                0.85 * pkg.body_h)
        sl = _fillet_slope(X * sgn, x_toe * sgn, x_term * sgn, h, FILLET_POWER)
        lo, hi = sorted((x_toe, x_term))
        p.paint((X >= lo) & (X <= hi) & (np.abs(Y) <= pkg.pad_w / 2.0 * 0.90),
                SOLDER, sl)
        return p

    # --- de-rotated body coordinates for skew --------------------------------
    a = np.radians(oth)
    c, s = np.cos(a), np.sin(a)
    Xb = (X - ox) * c + (Y - oy) * s
    Yb = -(X - ox) * s + (Y - oy) * c

    # --- fillets, one per land, each climbing along its own axis --------------
    for i, l in enumerate(lands):
        starved = defect.kind == "insufficient" and defect.pad == i
        along_x = abs(l.ax) > abs(l.ay)
        # toe at the outer edge of the land, termination at the body edge, which
        # moves with the part when it is displaced
        if along_x:
            coord, perp = X, Y
            toe = l.cx - l.ax * l.hx * 0.95   # toe sits just inside the land
            term = -l.ax * body_hx + ox
            perp_c, perp_h = l.cy, l.hy
        else:
            coord, perp = Y, X
            toe = l.cy - l.ay * l.hy * 0.95
            term = -l.ay * body_hy + oy
            perp_c, perp_h = l.cx, l.hx

        if defect.kind == "excess" and defect.pad == i:
            # Spills outward past the toe onto bare substrate and sideways past
            # the land: that is the half of the signature the inspector can
            # actually separate from starvation.
            sgn = 1.0 if term >= toe else -1.0
            outer = toe - sgn * EXCESS_SPILL * l.half_along
            a = abs(term - outer) / 2.0
            b = perp_h * EXCESS_WIDEN
            sl = _dome_slope(coord - (outer + term) / 2.0, perp - perp_c, a, b,
                             EXCESS_DOME * a)
            p.paint(((coord - (outer + term) / 2.0) / a) ** 2
                    + ((perp - perp_c) / b) ** 2 <= 1.0, SOLDER, sl)
            continue

        run = abs(term - toe)
        # Solve the climb height from the slope the joint should reach, rather
        # than from the body -- otherwise a tall package renders an 80 deg face
        # that overshoots every band and the test inverts.
        target = FILLET_SLOPE_STARVED_DEG if starved else FILLET_SLOPE_OK_DEG
        h = min(np.tan(np.radians(target)) * run / FILLET_POWER, 0.85 * fitted.body_h)
        sl = _fillet_slope(coord, toe, term, h, FILLET_POWER)
        lo, hi = sorted((toe, term))
        mask = ((coord >= lo) & (coord <= hi)
                & (np.abs(perp - perp_c) <= perp_h * 0.92))
        p.paint(mask, SOLDER, sl)

    # --- body ----------------------------------------------------------------
    body = _rect(Xb, Yb, 0.0, 0.0, body_hx, body_hy)
    p.paint(body, BODY, 0.0)

    if pkg.polarized:
        rev = defect.kind == "reversed"
        mx = (+1.0 if rev else -1.0) * (body_hx * 0.66)
        p.paint(body & _rect(Xb, Yb, mx, 0.0, body_hx * 0.20, body_hy * 0.88), MARKING)

    # --- bridge --------------------------------------------------------------
    # Solder spanning one gap between neighbouring lands. On a chip there is only
    # the one; on a lead row `defect.pad` picks which consecutive pair shorts.
    if defect.kind == "bridge":
        pairs = pkg.adjacent_pairs()
        la, lb = pairs[defect.pad % len(pairs)]
        along_x = abs(lb.cx - la.cx) >= abs(lb.cy - la.cy)
        mid_x, mid_y = (la.cx + lb.cx) / 2.0, (la.cy + lb.cy) / 2.0
        # half the clear space between the two lands, inner edge to inner edge,
        # not half their centre spacing
        if along_x:
            g = max(abs(lb.cx - la.cx) / 2.0 - min(la.hx, lb.hx), 1e-6)
            u, v = X - mid_x, Y - mid_y
            hv = min(la.hy, lb.hy) * 0.55
        else:
            g = max(abs(lb.cy - la.cy) / 2.0 - min(la.hy, lb.hy), 1e-6)
            u, v = Y - mid_y, X - mid_x
            hv = min(la.hx, lb.hx) * 0.55
        dome = np.clip(1.0 - (u / g) ** 2, 0.0, 1.0)
        sl = np.degrees(np.arctan(1.6 * np.abs(u) / g))
        p.paint((np.abs(u) <= g) & (np.abs(v) <= hv) & (dome > 0.02), SOLDER, sl)
    return p


def _to_channels(p: _Patch) -> np.ndarray:
    """Apply the reflection model: slope + material -> one plane per ring."""
    out = np.zeros((p.h, p.w, len(config.CHANNELS)), np.float32)
    for i, ch in enumerate(config.CHANNELS):
        if ch == "diffuse":
            out[..., i] = p.albedo
        else:
            da = p.slope - config.SLOPE_DEG[ch]
            out[..., i] = (p.spec * np.exp(-(da ** 2) / (2 * config.SLOPE_SIGMA_DEG ** 2))
                           + AMBIENT * p.albedo)
    return out


# ----------------------------------------------------------------------- board
def render(board: Board, defects: dict[str, Defect] | None = None,
           opts: RenderOpts | None = None,
           mm_per_px: float = config.MM_PER_PX) -> np.ndarray:
    """Render the whole board as a (H, W, 4) uint8 stack."""
    opts = opts or RenderOpts()
    defects = defects or {}
    rng = np.random.default_rng(opts.seed)

    W = int(round(board.width / mm_per_px))
    H = int(round(board.height / mm_per_px))
    nc = len(config.CHANNELS)

    stack = np.zeros((H, W, nc), np.float32)
    stack[..., :3] = AMBIENT * SUBSTRATE[0]
    stack[..., 3] = SUBSTRATE[0]

    # global misregistration -- what fiducial alignment has to take back out
    ca, sa = np.cos(np.radians(opts.board_rot_deg)), np.sin(np.radians(opts.board_rot_deg))

    def place(px: float, py: float) -> tuple[float, float]:
        x = px - board.width / 2.0
        y = py - board.height / 2.0
        return (board.width / 2.0 + ca * x - sa * y + opts.board_shift_mm[0],
                board.height / 2.0 + sa * x + ca * y + opts.board_shift_mm[1])

    for fx, fy in board.fiducials:
        gx, gy = place(fx, fy)
        _blit_disc(stack, gx, gy, 0.5, mm_per_px)

    for comp in board.components:
        pkg = comp.pkg
        patch = _render_component(pkg, defects.get(comp.refdes, Defect()), rng,
                                  mm_per_px, opts.place_sigma_um)
        chans = _to_channels(patch)
        gx, gy = place(comp.x, comp.y)
        _blit(stack, chans, gx, gy, comp.rot + opts.board_rot_deg, mm_per_px)

    # illumination falloff, then sensor noise
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - W / 2.0, yy - H / 2.0) / np.hypot(W / 2.0, H / 2.0)
    stack *= (1.0 - opts.vignette * r ** 2)[..., None].astype(np.float32)
    stack *= opts.gain
    stack += rng.normal(0.0, opts.noise, stack.shape).astype(np.float32)
    return np.clip(stack * 255.0, 0, 255).astype(np.uint8)


def _blit(stack: np.ndarray, patch: np.ndarray, cx_mm: float, cy_mm: float,
          rot_deg: float, mm_per_px: float) -> None:
    """Rotate a component patch into place, keeping the brighter of the two.

    Warps only into the destination bounding box rather than the whole board --
    at ~50 components per render that is the difference between seconds and
    minutes.
    """
    ph, pw = patch.shape[:2]
    a = np.radians(rot_deg)
    c, s = np.cos(a), np.sin(a)
    cx, cy = cx_mm / mm_per_px, cy_mm / mm_per_px

    hw = (abs(c) * pw + abs(s) * ph) / 2.0 + 2.0
    hh = (abs(s) * pw + abs(c) * ph) / 2.0 + 2.0
    H, W = stack.shape[:2]
    x0, y0 = max(0, int(np.floor(cx - hw))), max(0, int(np.floor(cy - hh)))
    x1, y1 = min(W, int(np.ceil(cx + hw))), min(H, int(np.ceil(cy + hh)))
    if x1 <= x0 or y1 <= y0:
        return

    M = np.array([[c, -s, (cx - x0) - (c * pw / 2.0 - s * ph / 2.0)],
                  [s,  c, (cy - y0) - (s * pw / 2.0 + c * ph / 2.0)]], np.float64)
    sub = cv2.warpAffine(patch, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if sub.ndim == 2:
        sub = sub[..., None]
    view = stack[y0:y1, x0:x1]
    np.maximum(view, sub, out=view)


def _blit_disc(stack: np.ndarray, cx_mm: float, cy_mm: float, dia_mm: float,
               mm_per_px: float) -> None:
    """Fiducial: a bare copper disc, composited through the same path as parts."""
    r = max(3, int(round(dia_mm / 2.0 / mm_per_px)))
    n = 4 * r
    disc = np.zeros((n, n), np.float32)
    cv2.circle(disc, (n // 2, n // 2), r, 1.0, -1, lineType=cv2.LINE_AA)
    patch = np.zeros((n, n, len(config.CHANNELS)), np.float32)
    for i, ch in enumerate(config.CHANNELS):
        v = COPPER[0] if ch == "diffuse" else COPPER[1] * 0.6 + AMBIENT * COPPER[0]
        patch[..., i] = disc * v
    _blit(stack, patch, cx_mm, cy_mm, 0.0, mm_per_px)
