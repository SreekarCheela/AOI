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
    "contrast_ratio", "gap_spec",
    # Band evidence has to be summarised, not listed per land: the vector is one
    # fixed width for every package and a QFP has 32 terminals where a chip has
    # 2. The worst land is what decides the component -- one bad joint fails the
    # part -- and the mean says whether the whole part is off or just one lead.
    "band_low_worst", "band_mid_worst", "band_high_worst",
    "band_low_mean", "band_mid_mean", "band_high_mean", "band_maha",
    "spill_worst",
    "body_area_ratio", "body_aspect", "body_shift_um", "body_clipped",
    "anomaly",
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
    bands: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    band_maha: float = 0.0   # worst per-pad Mahalanobis distance from golden
    spill: np.ndarray = field(    # (n_lands,) specular fraction just outside each
        default_factory=lambda: np.zeros(0, np.float32))   # land, minus golden's
    spill_worst: float = 0.0      # the land whose solder spread furthest
    body_area_ratio: float = 1.0   # segmented body area / golden. 0 = no body
    body_aspect: float = 2.0       # body extent along the pad axis / its own width
    body_shift_um: float = 0.0     # body centroid distance from nominal
    body_clipped: float = 0.0      # 1.0 if the blob runs into the crop edge
    anomaly: float = 0.0

    def band_summary(self) -> tuple[float, ...]:
        """Worst and mean band fraction per channel, over however many lands
        this package has. Fixed width, so one feature vector serves a 2-pad chip
        and a 32-lead QFP alike."""
        if self.bands.size == 0:
            return (float("nan"),) * 6
        b = np.asarray(self.bands, np.float64)
        return (*b.min(axis=0), *b.mean(axis=0))

    def vector(self) -> np.ndarray:
        return np.array([self.ncc, self.dx_um, self.dy_um, self.dtheta_deg,
                         self.pol_margin, self.body_contrast, self.contrast_ratio,
                         self.gap_spec, *self.band_summary(), self.band_maha,
                         self.spill_worst,
                         self.body_area_ratio, self.body_aspect,
                         self.body_shift_um, self.body_clipped,
                         self.anomaly], np.float32)


@dataclass
class GoldenPackage:
    template: np.ndarray                 # mean de-rotated diffuse crop
    template_180: np.ndarray
    body: np.ndarray                     # body window only -- see `body_template`
    body_180: np.ndarray
    body_mask: np.ndarray                # body footprint within `body`
    band_mean: np.ndarray                # (n_lands, 3)
    band_std: np.ndarray                 # (n_lands, 3), diagnostics only
                                         # -- the fillet decision uses band_cov_inv
    gap_mean: float
    gap_std: float
    spill_mean: np.ndarray               # (n_lands,) specular fraction just
                                         # outside each land on a good joint
    contrast_mean: float
    contrast_std: float
    ncc_mean: float = 1.0                # how well a GOOD part of this package
    ncc_std: float = 0.0                 # matches its own golden body template
    # Body segmentation, learned at teach: the two thresholds that separate the
    # body from the lands (high channel) and from bare substrate (diffuse), plus
    # what a correctly placed body of this package measures.
    body_spec_max: float = 0.19
    body_dif_min: float = 0.23
    body_len_mm: float = 1.0     # diagnostics and reporting only
    body_width_mm: float = 1.0   # extent ACROSS the pad axis; a body standing on
                                 # one land spans exactly this along the axis
    body_area_px: float = 1.0
    # (2, 3, 3) regularised inverse covariance of the three band fractions, per
    # pad. Defaults to an uncorrelated matrix at the std floor so a hand-built
    # Golden still scores something sane rather than dividing by zero.
    band_cov_inv: np.ndarray = field(   # (n_lands, 3, 3)
        default_factory=lambda: np.stack([np.eye(3) / config.BAND_COV_FLOOR ** 2] * 2))

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
    """Per-land fillet window: outer land edge to body edge, where solder climbs.

    Built along each land's own axis, so a lead on the top row of a QFP gets a
    window running downward into the body exactly as a chip's left land gets one
    running right. For a two-terminal package this reproduces the original pair
    of boxes exactly.
    """
    out = []
    for l in pkg.lands():
        if abs(l.ax) > abs(l.ay):                      # land on a left/right row
            p_out = l.cx - l.ax * l.hx                 # outer edge, away from body
            p_in = -l.ax * pkg.body_l / 2.0            # body edge
            out.append((np.array([(p_out + p_in) / 2.0, l.cy]),
                        np.array([abs(p_out - p_in) / 2.0, l.hy * 0.92])))
        else:                                          # top/bottom row
            p_out = l.cy - l.ay * l.hy
            p_in = -l.ay * pkg.body_w / 2.0
            out.append((np.array([l.cx, (p_out + p_in) / 2.0]),
                        np.array([l.hx * 0.92, abs(p_out - p_in) / 2.0])))
    return out


# ------------------------------------------------------------- primitive tests
def _norm(a: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Scale to [0,1] if the input looks like 8-bit.

    Empty-safe: every caller clips a window against the crop bounds and can end
    up with nothing, and `max()` on an empty array raises. Guarding here rather
    than at each of the three call sites, all of which checked the size only
    after calling this.

    `ref` supplies the array whose range decides the scale, and every caller
    that normalises several arrays which must stay comparable has to pass it.
    Deciding per-array is wrong exactly when it matters: an angled channel dark
    enough never to exceed 1.5 -- which is what an underexposed high-angle frame
    looks like, and the physics gate expects those frames to be nearly black --
    escapes the divide its siblings receive and comes back 255x too bright. In
    `specular_max` that put every pixel at 1.0, so no body was ever found and
    the whole board reported as `missing`.
    """
    a = a.astype(np.float32)
    if a.size == 0:
        return a
    r = a if ref is None else ref
    return a / 255.0 if (r.size > 0 and r.max() > 1.5) else a


def band_fractions(region: np.ndarray, pkg: Package,
                   shift_px: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Fraction of each fillet window lit in each angled ring -- the slope signature.

    Returns (n_lands, 3). One row per terminal, so a chip gives 2 and a QFP 32.
    """
    boxes = joint_boxes(pkg)
    out = np.zeros((len(boxes), 3), np.float32)
    for i, (centre, half) in enumerate(boxes):
        c = centre + np.array([shift_px[0], shift_px[1]]) * config.MM_PER_PX
        ys, xs = roi.local_box_px(c, half, region.shape)
        win = _norm(region[ys, xs])
        if win.size == 0:
            continue
        for j, ch in enumerate(config.ANGLED_CHANNELS):
            k = config.CHANNELS.index(ch)
            out[i, j] = float((win[..., k] > config.BAND_THRESHOLD).mean())
    return out


def spill_boxes(pkg: Package) -> list[tuple[np.ndarray, np.ndarray]]:
    """The strip just OUTSIDE each land's outer edge, where solder must not be.

    Not the fillet window, and not a grown version of it. Inside the fillet
    window a balled joint scores 1.00: the extra volume displaced the fillet
    rather than adding to it, and a window bounded by the toe and the body edge
    is exactly the region solder is supposed to fill. The evidence for excess is
    solder that went where CAD says bare substrate, which is the same question
    the bridge test asks between two lands, asked on the outer side instead.

    Sized to fit inside the inspection crop. The crop carries a 0.35 mm margin
    past the package extent, and on every package in the library the outermost
    land IS the extent, so the strip has to stay under that or it is silently
    clipped -- which is not a loud failure, it is a test that quietly measures
    nothing. `test_excess.py` pins that relationship.
    """
    out = []
    d = config.SPILL_STRIP_MM
    for l in pkg.lands():
        if abs(l.ax) > abs(l.ay):                      # land on a left/right row
            edge = l.cx - l.ax * l.hx                  # outer edge, away from body
            out.append((np.array([edge - l.ax * d / 2.0, l.cy]),
                        np.array([d / 2.0, l.hy])))
        else:                                          # top/bottom row
            edge = l.cy - l.ay * l.hy
            out.append((np.array([l.cx, edge - l.ay * d / 2.0]),
                        np.array([l.hx, d / 2.0])))
    return out


def spill_fractions(region: np.ndarray, pkg: Package, metal_min: float) -> np.ndarray:
    """Specular fraction of each land's outward spill strip. (n_lands,).

    On a good joint this is bare substrate and reads ~0. It is a fraction rather
    than a ratio against golden because the golden value is zero, and a ratio
    against zero is not a measurement.

    What the threshold actually counts is worth being precise about: a pixel
    passes if it is specular AND its slope falls inside one of the three rings,
    which together span roughly 0 to 45 degrees. A surface steeper than that
    reflects away from the camera and reads dark whatever it is made of. So this
    does not measure "solder is present" -- it measures "a specular surface
    lying at a shallow enough angle to be seen is present, outside the land".
    A balled joint's near-vertical rim is genuinely invisible here, and the
    signal is the shoulder inboard of it.
    """
    out = np.zeros(len(pkg.lands()), np.float32)
    for i, (centre, half) in enumerate(spill_boxes(pkg)):
        ys, xs = roi.local_box_px(centre, half, region.shape)
        win = region[ys, xs]
        if win.size == 0:
            continue
        out[i] = float((specular_max(win) > metal_min).mean())
    return out


@dataclass
class BodyBlob:
    """Where the body actually is on the crop, and how big it is."""
    found: bool = False
    dx_mm: float = 0.0          # centroid offset from nominal, along the pad axis
    dy_mm: float = 0.0
    length_mm: float = 0.0      # extent along the pad axis
    width_mm: float = 0.0
    area_px: int = 0
    clipped: bool = False       # blob runs into the crop edge, so its extent is
                                # a crop boundary rather than the part boundary


def specular_max(region: np.ndarray) -> np.ndarray:
    """Brightest of the three angled rings at each pixel.

    Metal is specular and lights up in whichever ring matches its slope, so the
    max over rings catches a fillet lit only in the low band as readily as flat
    copper lit only in the high band. A component body is not specular and stays
    dark in all three.
    """
    chans = [_norm(region[..., config.CHANNELS.index(c)], ref=region)
             for c in config.ANGLED_CHANNELS]
    return np.maximum(np.maximum(chans[0], chans[1]), chans[2])


def body_blob(region: np.ndarray, spec_max: float, dif_min: float) -> BodyBlob:
    """Segment the component body and measure its position and extent.

    The body is the one large region that is dark in every angled ring -- so
    neither copper nor solder, both of which are specular in at least one of
    them -- yet clearly brighter than bare substrate in the diffuse channel.
    Measured across all five package families the body sits at 0.082-0.086 of
    specular max, a polarity marking at 0.114-0.118, and metal at 0.62-0.72.

    This answers a question NCC cannot. Correlation says how well the crop looks
    like a good part and collapses every way of not being one into a single low
    number; it cannot distinguish a body that is absent from a body that is
    somewhere else. Position and extent distinguish them directly:

        no blob                         nothing was placed
        blob much shorter than golden   the part is standing on one land
        blob full length, displaced     the part slid
    """
    sm = specular_max(region)
    dif = _norm(region[..., config.CHANNELS.index("diffuse")])
    mask = ((sm < spec_max) & (dif > dif_min)).astype(np.uint8)
    # Opening drops speckle from sensor noise without touching a real body,
    # which is hundreds of pixels across at 12 um/px.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return BodyBlob()
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    h, w = mask.shape
    x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
    x1 = x0 + int(stats[i, cv2.CC_STAT_WIDTH])
    y1 = y0 + int(stats[i, cv2.CC_STAT_HEIGHT])
    return BodyBlob(
        clipped=bool(x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h),
        found=True,
        dx_mm=float((cent[i][0] - w / 2.0) * config.MM_PER_PX),
        dy_mm=float((cent[i][1] - h / 2.0) * config.MM_PER_PX),
        length_mm=float(stats[i, cv2.CC_STAT_WIDTH] * config.MM_PER_PX),
        width_mm=float(stats[i, cv2.CC_STAT_HEIGHT] * config.MM_PER_PX),
        area_px=int(stats[i, cv2.CC_STAT_AREA]),
    )


def _band_cov_inv(samples: np.ndarray) -> np.ndarray:
    """Regularised inverse covariance of one pad's three band fractions.

    `samples` is (N, 3), one row per golden board.

    The regularisation is load-bearing, not hygiene. Teach runs on 5-20 boards,
    so a 3x3 covariance is estimated from very few samples, and the three bands
    are correlated at r = 0.5 to 0.85 on good joints. That leaves one
    near-collapsed axis, and a plain inverse turns its small eigenvalue into an
    enormous distance for a tiny deviation -- so the test meant to cut false
    calls would generate them instead.

    TWO things regularise here and they do different jobs, which was worth
    measuring rather than assuming. Over 400 draws at n=5, r=0.85, scoring a
    good joint dimmed 8% -- the common-mode move the test exists to forgive --
    against a fail threshold of 6.0:

        ridge   floor    p99    worst
        0.15    0.01     2.61    3.32     shipped
        0.00    0.01     2.77    3.32     floor alone: almost the whole effect
        0.15    1e-7     3.63   13.12     ridge alone: still generates false calls
        0.00    1e-7    12.72   26.17     neither

    So BAND_COV_FLOOR is what keeps the distance bounded, and this docstring
    used to credit the ridge for it. The ridge earns its place separately, by
    bounding the condition number -- at n=5 that is four figures without it and
    under 20 with it -- which is what keeps the weak axis from dominating the
    direction of the distance even when its magnitude is already capped.

    Both are pinned in tests/test_fillet.py. Re-derive them against real golden
    boards: synthetic joints are tidier than real ones and the good-side tail
    will be fatter.
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

    Returns one distance per land. Scoring the three bands jointly rather
    than one at a time is the whole point: anything that changes how much light
    reaches a component moves all three together, along the high-covariance axis
    where it costs almost no distance, while a starved joint loses the steep
    face but keeps the shallow one and moves across that axis where it costs a
    lot (spec section 5).
    """
    n = min(len(bands), len(g.band_mean))
    out = np.zeros(n, np.float64)
    for pad in range(n):
        d = bands[pad].astype(np.float64) - np.asarray(g.band_mean[pad], np.float64)
        out[pad] = float(np.sqrt(max(float(d @ g.band_cov_inv[pad] @ d), 0.0)))
    return out


def gap_specular(region: np.ndarray, pkg: Package) -> float:
    """Worst specular response in any gap between neighbouring lands.

    Substrate should be dark there. A chip has one such gap; a lead row has one
    per consecutive pair, and the worst of them is what decides the component,
    because a single bridged pair shorts the part however clean the rest is.
    """
    lo = config.CHANNELS.index("low")
    mid = config.CHANNELS.index("mid")
    worst = 0.0
    for centre, half in roi.gap_boxes(pkg):
        ys, xs = roi.local_box_px(centre, half, region.shape)
        win = _norm(region[ys, xs])
        if win.size == 0:
            continue
        worst = max(worst, float(np.maximum(win[..., lo], win[..., mid]).mean()))
    return worst


def body_contrast(region: np.ndarray, pkg: Package) -> float:
    """Diffuse contrast between body window and its surround -- gross presence."""
    dif = config.CHANNELS.index("diffuse")
    ys, xs = roi.local_box_px(np.array([0.0, 0.0]), roi.body_rect(pkg), region.shape)
    body = _norm(region[ys, xs, dif], ref=region[..., dif])
    whole = _norm(region[..., dif])
    if body.size == 0:
        return 0.0
    return float(body.mean() - whole.mean())


def _match(region_ch: np.ndarray, template: np.ndarray,
           max_shift_px: float | None = None) -> tuple[float, float, float]:
    """NCC of template inside region; returns (score, dx_px, dy_px).

    `max_shift_px` bounds how far from nominal the peak is allowed to be. The
    spec calls for a +-0.3 mm search window and `config.SEARCH_UM` recorded it,
    but nothing ever applied it: the peak was taken over the whole correlation
    surface, so the real bound was whatever the crop margin happened to be --
    incidental, and different for every package.

    Bounding it is the same principle the rotation sweep already follows. The
    window has to be wider than the tolerance it tests, or a part outside
    tolerance can only ever measure exactly the tolerance; 300 um against a
    150 um limit is that factor of two. Its value is refusing a peak that is
    nowhere near the part: a bright neighbour or a stretch of bare copper can
    correlate better than a body that has genuinely moved, and an unbounded
    search will take it and report a confident placement for the wrong object.
    """
    if template.size == 0 or region_ch.shape[0] < template.shape[0] or \
       region_ch.shape[1] < template.shape[1]:
        return 0.0, 0.0, 0.0
    res = cv2.matchTemplate(region_ch, template, cv2.TM_CCOEFF_NORMED)
    cx = (region_ch.shape[1] - template.shape[1]) / 2.0
    cy = (region_ch.shape[0] - template.shape[0]) / 2.0

    if max_shift_px is not None:
        r = int(round(max_shift_px))
        x0, x1 = max(0, int(round(cx)) - r), min(res.shape[1], int(round(cx)) + r + 1)
        y0, y1 = max(0, int(round(cy)) - r), min(res.shape[0], int(round(cy)) + r + 1)
        if x1 > x0 and y1 > y0:
            _, score, _, loc = cv2.minMaxLoc(res[y0:y1, x0:x1])
            return float(score), float(loc[0] + x0 - cx), float(loc[1] + y0 - cy)

    _, score, _, loc = cv2.minMaxLoc(res)
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
        # Nothing here can be judged: every test compares against a golden
        # statistic for this package, and there is not one. Say so with NaN
        # rather than leaving the defaults, because 0.0 is a perfectly good
        # measurement meaning "correlates with nothing" and every downstream
        # threshold reads it as a gross fault. An unknown that looks like a
        # defect is how one unprogrammed package flags a whole board.
        m.bands = band_fractions(region, pkg)
        m.ncc = m.dx_um = m.dy_um = m.dtheta_deg = float("nan")
        m.pol_margin = m.contrast_ratio = m.band_maha = float("nan")
        m.spill_worst = float("nan")
        m.body_area_ratio = m.body_aspect = m.body_shift_um = float("nan")
        m.body_clipped = float("nan")
        return m

    # z-score, not a ratio: the golden mean passes through zero for some
    # packages and dividing by it produces five-figure nonsense.
    m.contrast_ratio = float((m.body_contrast - g.contrast_mean)
                             / max(g.contrast_std, 0.01))

    # placement: match the body against the un-rotated region
    m.ncc, dx_px, dy_px = _match(reg_d, g.body,
                                 max_shift_px=config.SEARCH_UM / config.UM_PER_PX)
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

    # Solder outside the land, over and above whatever a good joint of this
    # package shows there. Zero on a good board; positive means solder spread
    # past the copper it was meant to stay on.
    n = min(len(g.spill_mean), len(pkg.lands()))
    m.spill = (spill_fractions(region, pkg, g.body_spec_max)[:n]
               - np.asarray(g.spill_mean[:n], np.float32))
    m.spill_worst = float(m.spill.max()) if m.spill.size else 0.0

    blob = body_blob(region, g.body_spec_max, g.body_dif_min)
    m.body_area_ratio = blob.area_px / max(g.body_area_px, 1.0)
    # Normalised by the body's own WIDTH, not its length. A part standing on one
    # land spans exactly its width along the pad axis, whatever package it is, so
    # this reads ~1.0 for every tombstone and ~1.6-2.0 for anything lying flat.
    # Normalising by golden LENGTH instead confuses a tombstone with a part
    # displaced far enough to be clipped by the crop edge, which also measures
    # short but for an unrelated reason.
    m.body_aspect = blob.length_mm / max(g.body_width_mm, 1e-6)
    m.body_shift_um = float(np.hypot(blob.dx_mm, blob.dy_mm)) * 1000.0
    m.body_clipped = 1.0 if blob.clipped else 0.0
    return m


# ---------------------------------------------------------------------- teach
def fit_golden(stacks: list[np.ndarray], board: Board) -> Golden:
    """Learn per-package templates and band statistics from known-good boards."""
    acc: dict[str, dict[str, list]] = {}
    for stack in stacks:
        for comp in board.components:
            pkg = comp.pkg
            region, _ = roi.extract_region(stack, comp, pkg)
            d = acc.setdefault(comp.package, {"tpl": [], "band": [], "gap": [], "con": [],
                                              "spec_body": [],
                                              "df_body": [], "df_sub": [], "reg": []})
            dif = config.CHANNELS.index("diffuse")
            d["tpl"].append(_norm(region[..., dif]))
            d["reg"].append(region)
            # Levels the body segmentation thresholds are learned from. The body
            # is dark in the high channel where the lands are bright, and bright
            # in the diffuse channel where bare substrate is dark; each threshold
            # goes midway between its own pair.
            sm = specular_max(region)
            dfc = _norm(region[..., dif])
            # Body core only. Kept well inside the footprint so a polarity
            # marking, which sits at 0.46-0.86 of the body half-length, stays
            # mostly out of the sample; the median finishes the job.
            bys, bxs = roi.local_box_px(np.array([0.0, 0.0]), roi.body_rect(pkg) * 0.45,
                                        region.shape)
            d["spec_body"].append(float(np.median(sm[bys, bxs])))
            d["df_body"].append(float(dfc[bys, bxs].mean()))
            # The 5th percentile of the crop is bare substrate -- the footprint
            # margin guarantees some is always in view.
            d["df_sub"].append(float(np.percentile(dfc, 5)))
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

        spec_max = float(np.median(d["spec_body"])) * config.BODY_SPECULAR_MULT
        dif_min = (float(np.mean(d["df_body"])) + float(np.mean(d["df_sub"]))) / 2.0
        blobs = [body_blob(r, spec_max, dif_min) for r in d["reg"]]
        blobs = [b for b in blobs if b.found]
        # Same threshold, used from the other side: below it is body, marking or
        # substrate, above it is metal. Measured here rather than in the pass
        # above because the level is only known once every golden board has
        # contributed to it.
        spill = np.mean([spill_fractions(r, pkg_obj, spec_max) for r in d["reg"]],
                        axis=0)

        out.packages[name] = GoldenPackage(
            template=tpl,
            template_180=np.ascontiguousarray(np.rot90(tpl, 2)),
            body=body,
            body_180=np.ascontiguousarray(np.rot90(body, 2)),
            body_mask=body_mask_for(body, pkg_obj),
            band_mean=bands.mean(axis=0),
            band_std=np.maximum(bands.std(axis=0), 0.01),
            spill_mean=spill,
            band_cov_inv=np.stack([_band_cov_inv(bands[:, pad, :])
                                   for pad in range(bands.shape[1])]),
            gap_mean=float(np.mean(d["gap"])), gap_std=float(max(np.std(d["gap"]), 0.01)),
            contrast_mean=float(np.mean(d["con"])),
            contrast_std=float(max(np.std(d["con"]), 0.005)),
            ncc_mean=float(np.mean(nccs)) if nccs else 1.0,
            ncc_std=float(np.std(nccs)) if nccs else 0.0,
            body_spec_max=spec_max,
            body_dif_min=dif_min,
            body_len_mm=float(np.mean([b.length_mm for b in blobs])) if blobs else 1.0,
            body_width_mm=float(np.mean([b.width_mm for b in blobs])) if blobs else 1.0,
            body_area_px=float(np.mean([b.area_px for b in blobs])) if blobs else 1.0,
        )
    return out
