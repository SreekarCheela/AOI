"""Rule-based measurements and the golden reference they are compared against.

These produce numbers, not verdicts: band fractions, offsets in microns, NCC
scores. The verdict is `fusion`'s job. Keeping measurement and decision apart is
what makes the machine debuggable eighteen months from now (spec section 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config, roi
from .board import Board, Component
from .packages import Package

# Must extend well past MAX_SKEW_DEG: a search that stops at the tolerance can
# only ever report the tolerance, so a 20 deg skew reads as exactly 12 and never
# trips the test.
ROT_SEARCH_DEG = tuple(float(d) for d in range(-24, 25, 3))

FEATURE_NAMES = (
    "ncc", "dx_um", "dy_um", "dtheta_deg", "pol_margin", "body_contrast",
    "contrast_ratio", "gap_spec", "band_low_0", "band_mid_0", "band_high_0",
    "band_low_1", "band_mid_1", "band_high_1", "band_maha", "anomaly",
)


@dataclass
class Measure:
    refdes: str
    package: str
    ncc: float = 0.0
    dx_um: float = 0.0
    dy_um: float = 0.0
    dtheta_deg: float = 0.0
    pol_margin: float = 0.0
    body_contrast: float = 0.0
    contrast_ratio: float = 1.0
    gap_spec: float = 0.0
    bands: np.ndarray = field(default_factory=lambda: np.zeros((2, 3), np.float32))
    band_maha: float = 0.0   # worst per-pad Mahalanobis distance from golden
    anomaly: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([self.ncc, self.dx_um, self.dy_um, self.dtheta_deg,
                         self.pol_margin, self.body_contrast, self.contrast_ratio,
                         self.gap_spec, *self.bands.ravel(), self.band_maha,
                         self.anomaly], np.float32)


@dataclass
class GoldenPackage:
    template: np.ndarray                 # mean de-rotated diffuse crop
    template_180: np.ndarray
    body: np.ndarray                     # body window only -- see `body_template`
    body_180: np.ndarray
    body_mask: np.ndarray                # body footprint within `body`
    band_mean: np.ndarray                # (2, 3)
    band_std: np.ndarray                 # (2, 3), diagnostics and reporting only
                                         # -- the fillet decision uses band_cov_inv
    gap_mean: float
    gap_std: float
    contrast_mean: float
    contrast_std: float
    ncc_mean: float = 1.0                # how well a GOOD part of this package
    ncc_std: float = 0.0                 # matches its own golden body template
    # (2, 3, 3) regularised inverse covariance of the three band fractions, per
    # pad. Defaults to an uncorrelated matrix at the std floor so a hand-built
    # Golden still scores something sane rather than dividing by zero.
    band_cov_inv: np.ndarray = field(
        default_factory=lambda: np.stack([np.eye(3) / config.BAND_COV_FLOOR ** 2] * 2))

    @property
    def ncc_fail(self) -> float:
        """Below this, the body is not on its footprint."""
        return self.ncc_mean * config.NCC_FAIL_FRACTION

    @property
    def ncc_absent(self) -> float:
        """Below this, there is no body in the window at all."""
        return self.ncc_mean * config.NCC_ABSENT_FRACTION


@dataclass
class Golden:
    packages: dict[str, GoldenPackage] = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return name in self.packages


# ------------------------------------------------------------------ geometry
def joint_boxes(pkg: Package) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-land fillet window: outer pad edge to body edge, where solder climbs."""
    out = []
    d = pkg.pad_pitch / 2.0
    for sgn in (-1.0, +1.0):
        x_outer = sgn * (d + pkg.pad_l / 2.0)
        x_inner = sgn * pkg.body_l / 2.0
        out.append((np.array([(x_outer + x_inner) / 2.0, 0.0]),
                    np.array([abs(x_outer - x_inner) / 2.0, pkg.pad_w / 2.0 * 0.92])))
    return out


# ------------------------------------------------------------- primitive tests
def _norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def band_fractions(region: np.ndarray, pkg: Package,
                   shift_px: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Fraction of each fillet window lit in each angled ring -- the slope signature."""
    out = np.zeros((2, 3), np.float32)
    for i, (centre, half) in enumerate(joint_boxes(pkg)):
        c = centre + np.array([shift_px[0], shift_px[1]]) * config.MM_PER_PX
        ys, xs = roi.local_box_px(c, half, region.shape)
        win = _norm(region[ys, xs])
        if win.size == 0:
            continue
        for j, ch in enumerate(config.ANGLED_CHANNELS):
            k = config.CHANNELS.index(ch)
            out[i, j] = float((win[..., k] > config.BAND_THRESHOLD).mean())
    return out


def _band_cov_inv(samples: np.ndarray) -> np.ndarray:
    """Regularised inverse covariance of one pad's three band fractions.

    `samples` is (N, 3), one row per golden board.

    The regularisation is load-bearing, not hygiene. Teach runs on 5-20 boards,
    so a 3x3 covariance is estimated from very few samples, and the three bands
    are correlated at r = 0.5 to 0.85 on good joints. That leaves one
    near-collapsed axis, and a plain inverse turns its small eigenvalue into an
    enormous distance for a tiny deviation -- so the test meant to cut false
    calls would generate them instead. Shrinking the off-diagonals bounds the
    condition number while keeping the correlation that makes the test work.
    """
    n = int(samples.shape[0])
    floor = config.BAND_COV_FLOOR ** 2
    if n < 2:
        return np.eye(3) / floor
    cov = np.atleast_2d(np.cov(samples.astype(np.float64), rowvar=False))
    r = config.BAND_COV_RIDGE
    cov = (1.0 - r) * cov + r * np.diag(np.diag(cov))
    cov = cov + floor * np.eye(3)
    return np.linalg.inv(cov)


def band_distance(bands: np.ndarray, g: "GoldenPackage") -> np.ndarray:
    """Mahalanobis distance of each pad's band triple from the golden mean.

    Returns (2,), one distance per land. Scoring the three bands jointly rather
    than one at a time is the whole point: anything that changes how much light
    reaches a component moves all three together, along the high-covariance axis
    where it costs almost no distance, while a starved joint loses the steep
    face but keeps the shallow one and moves across that axis where it costs a
    lot (spec section 5).
    """
    out = np.zeros(2, np.float64)
    for pad in range(2):
        d = bands[pad].astype(np.float64) - np.asarray(g.band_mean[pad], np.float64)
        out[pad] = float(np.sqrt(max(float(d @ g.band_cov_inv[pad] @ d), 0.0)))
    return out


def gap_specular(region: np.ndarray, pkg: Package) -> float:
    """Specular response between the lands -- substrate should be dark here."""
    ys, xs = roi.local_box_px(np.array([0.0, 0.0]), roi.gap_rect(pkg), region.shape)
    win = _norm(region[ys, xs])
    if win.size == 0:
        return 0.0
    lo = config.CHANNELS.index("low")
    mid = config.CHANNELS.index("mid")
    return float(np.maximum(win[..., lo], win[..., mid]).mean())


def body_contrast(region: np.ndarray, pkg: Package) -> float:
    """Diffuse contrast between body window and its surround -- gross presence."""
    dif = config.CHANNELS.index("diffuse")
    ys, xs = roi.local_box_px(np.array([0.0, 0.0]), roi.body_rect(pkg), region.shape)
    body = _norm(region[ys, xs, dif])
    whole = _norm(region[..., dif])
    if body.size == 0:
        return 0.0
    return float(body.mean() - whole.mean())


def _match(region_ch: np.ndarray, template: np.ndarray) -> tuple[float, float, float]:
    """NCC of template inside region; returns (score, dx_px, dy_px)."""
    if template.size == 0 or region_ch.shape[0] < template.shape[0] or \
       region_ch.shape[1] < template.shape[1]:
        return 0.0, 0.0, 0.0
    res = cv2.matchTemplate(region_ch, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    cx = (region_ch.shape[1] - template.shape[1]) / 2.0
    cy = (region_ch.shape[0] - template.shape[0]) / 2.0
    return float(score), float(loc[0] - cx), float(loc[1] - cy)


def _rotate_pair(tpl: np.ndarray, mask: np.ndarray, deg: float):
    """Rotate a template and its mask together about their common centre."""
    if abs(deg) < 1e-6:
        return tpl, mask
    h, w = tpl.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    t = cv2.warpAffine(tpl, M, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT101)
    m = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return np.ascontiguousarray(t), np.ascontiguousarray(m)


def _match_masked(region: np.ndarray, tpl: np.ndarray, mask: np.ndarray) -> float:
    if region.shape[0] < tpl.shape[0] or region.shape[1] < tpl.shape[1]:
        return -9.0
    r = cv2.matchTemplate(region, tpl, cv2.TM_CCORR_NORMED, mask=mask)
    return float(np.nan_to_num(r, nan=-9.0, posinf=-9.0, neginf=-9.0).max())


def _rotate(region: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-6:
        return region
    h, w = region.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(region, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def body_mask_for(template: np.ndarray, pkg: Package) -> np.ndarray:
    """Binary mask of the body inside a body-window template.

    Skew is measured by rotating the template and correlating. Without a mask
    the fixed lands sit inside the template too, so rotating it rotates them as
    well and the score always argues for zero. Masking to the body compares only
    the pixels that actually turn with the part.
    """
    h, w = template.shape[:2]
    mask = np.zeros((h, w), np.float32)
    hx = max(2, int(pkg.body_l / 2.0 / config.MM_PER_PX))
    hy = max(2, int(pkg.body_w / 2.0 / config.MM_PER_PX))
    cv2.rectangle(mask, (w // 2 - hx, h // 2 - hy), (w // 2 + hx, h // 2 + hy), 1.0, -1)
    return mask


def body_template(template: np.ndarray, pkg: Package) -> np.ndarray:
    """Crop a golden region down to the component body.

    Matching the whole footprint does not measure placement: the lands are fixed
    to the board and never move, so they anchor the correlation peak at zero no
    matter where the body actually sits. Only the body carries the displacement.
    """
    half = roi.body_rect(pkg) * 1.06
    ys, xs = roi.local_box_px(np.array([0.0, 0.0]), half, template.shape)
    return np.ascontiguousarray(template[ys, xs])


# ------------------------------------------------------------------- measuring
def measure(stack: np.ndarray, comp: Component, golden: Golden) -> Measure:
    pkg = comp.pkg
    m = Measure(comp.refdes, comp.package)
    region, _ = roi.extract_region(stack, comp, pkg)
    dif = config.CHANNELS.index("diffuse")
    reg_d = _norm(region[..., dif])

    m.gap_spec = gap_specular(region, pkg)
    m.body_contrast = body_contrast(region, pkg)

    g = golden.packages.get(comp.package)
    if g is None:
        m.bands = band_fractions(region, pkg)
        return m

    # z-score, not a ratio: the golden mean passes through zero for some
    # packages and dividing by it produces five-figure nonsense.
    m.contrast_ratio = float((m.body_contrast - g.contrast_mean)
                             / max(g.contrast_std, 0.01))

    # placement: match the body against the un-rotated region
    m.ncc, dx_px, dy_px = _match(reg_d, g.body)
    m.dx_um = dx_px * config.UM_PER_PX
    m.dy_um = dy_px * config.UM_PER_PX

    # skew: rotate the masked body template; the peak angle is the negative of
    # the part's own rotation, since turning the template the other way is what
    # brings it into register.
    best = (-9.0, 0.0)
    for deg in ROT_SEARCH_DEG:
        t, mk = _rotate_pair(g.body, g.body_mask, deg)
        v = _match_masked(reg_d, t, mk)
        if v > best[0]:
            best = (v, deg)
    m.dtheta_deg = -best[1]

    if pkg.polarized:
        s180, _, _ = _match(reg_d, g.body_180)
        m.pol_margin = m.ncc - float(s180)

    # fillet windows follow the lands, which do not move with the body
    m.bands = band_fractions(region, pkg)
    m.band_maha = float(band_distance(m.bands, g).max())
    return m


# ---------------------------------------------------------------------- teach
def fit_golden(stacks: list[np.ndarray], board: Board) -> Golden:
    """Learn per-package templates and band statistics from known-good boards."""
    acc: dict[str, dict[str, list]] = {}
    for stack in stacks:
        for comp in board.components:
            pkg = comp.pkg
            region, _ = roi.extract_region(stack, comp, pkg)
            d = acc.setdefault(comp.package, {"tpl": [], "band": [], "gap": [], "con": []})
            dif = config.CHANNELS.index("diffuse")
            d["tpl"].append(_norm(region[..., dif]))
            d["band"].append(band_fractions(region, pkg))
            d["gap"].append(gap_specular(region, pkg))
            d["con"].append(body_contrast(region, pkg))

    packages_for = {c.package: c.pkg for c in board.components}
    out = Golden()
    for name, d in acc.items():
        shp = d["tpl"][0].shape
        tpl = np.mean([t for t in d["tpl"] if t.shape == shp], axis=0).astype(np.float32)
        bands = np.stack(d["band"])
        pkg_obj = packages_for[name]
        body = body_template(tpl, pkg_obj)

        # How well does a KNOWN-GOOD part of this package match its own template?
        # Presence is gated on a fraction of this rather than a global constant,
        # so the same rule means the same thing on an 0603 (golden ~0.92) and a
        # SOD-123 (~0.99). Measured here because teach is the only place the
        # good-part distribution exists.
        nccs = [_match(t, body)[0] for t in d["tpl"] if t.shape == shp]

        out.packages[name] = GoldenPackage(
            template=tpl,
            template_180=np.ascontiguousarray(np.rot90(tpl, 2)),
            body=body,
            body_180=np.ascontiguousarray(np.rot90(body, 2)),
            body_mask=body_mask_for(body, pkg_obj),
            band_mean=bands.mean(axis=0),
            band_std=np.maximum(bands.std(axis=0), 0.01),
            band_cov_inv=np.stack([_band_cov_inv(bands[:, pad, :]) for pad in range(2)]),
            gap_mean=float(np.mean(d["gap"])), gap_std=float(max(np.std(d["gap"]), 0.01)),
            contrast_mean=float(np.mean(d["con"])),
            contrast_std=float(max(np.std(d["con"]), 0.005)),
            ncc_mean=float(np.mean(nccs)) if nccs else 1.0,
            ncc_std=float(np.std(nccs)) if nccs else 0.0,
        )
    return out
