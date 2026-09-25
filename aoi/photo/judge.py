"""Measurements -> calls -> part state -> board verdict.

Order per part is aoi/fusion.py's: missing, tombstone, wrong part, polarity,
offset, skew. A FAIL stops the part's remaining checks, because a part that is
not there has no polarity and a part standing on end has no offset. A REVIEW
does not stop them: a polarity REVIEW part is still checked for offset and skew.

Two modes set the thresholds:

* Three or more goldens. Every threshold is learned per part from how much a
  good part varies between golden shots (golden_stats, built at teach). Scores
  are z-scores against that spread.
* Single golden (CLI and tests only). A golden cannot measure its own spread, so
  fixed thresholds from the 2026-09-25 run are used and every result is stamped.

Scores are never shown as probabilities until a check has enough operator labels
to calibrate them (Platt scaling, `calibrate`).
"""
from __future__ import annotations

import math

import numpy as np

from .. import config
from .model import BoardResult, Call, Part, PartResult

# Frozen missing-part model, fitted on the 2026-09-25 fixture (37 parts x 3 test
# boards, 14 missing): logistic regression on the three features that separated
# bare footprints from placed parts. Raw-space weights. Leave-one-board-out it
# caught 12 of 14 missing with no false call; the two it missed (test3 R11, R14)
# are bare lands of dull solder that read as a body on every feature tried.
MISSING_WEIGHTS = {"phys_ratio": -2.0694, "gap_ratio": -1.7137, "ext": -6.1625}
MISSING_BIAS = 6.0225
# Decision point: midway between the highest-scoring present part on the fitting
# data (test3 R10, 0.60) and the lowest-scoring missing part it caught (0.94).
MISSING_P = 0.75
MISSING_Z = 4.0                 # multi-golden: z of the missing logit over golden spread
MISSING_SPREAD_MAX = 2.0        # golden logit sd above this -> UNINSPECTED (undecidable)
LOGIT_SD_FLOOR = 0.25

# Polarity, cathode band / bevel: both ends of the profile must be clearly
# different (|contrast| >= 1.5 profile sd) and on opposite ends. On the fixture
# only test1 D4 (the one reversed diode) passes that; the golden D1's smudged
# marking reads 1.39, so D1's band polarity is UNINSPECTED rather than guessed.
STRIPE_CLEAR = 1.5
TURN_REVIEW = -0.20             # marking correlation margin, 0 vs 180 deg
WRONG_PART_Z = 4.0


def _z(v: float, stat) -> float:
    if not stat:
        return 0.0
    mu, sd = stat[0], stat[1]
    return (v - mu) / max(sd, 0.05)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def extent_ratio(m: dict) -> float:
    g, t = m.get("body_g"), m.get("body_t")
    if not g or g["extent"] <= 0:
        return 1.0
    return (t["extent"] / g["extent"]) if t else 0.0


def missing_logit(m: dict) -> float:
    f = {"phys_ratio": m["phys_ratio"], "gap_ratio": m["gap_ratio"], "ext": extent_ratio(m)}
    return MISSING_BIAS + sum(MISSING_WEIGHTS[k] * f[k] for k in MISSING_WEIGHTS)


# ------------------------------------------------------------- board noise
def board_noise(measures: dict[str, dict]) -> dict:
    """Common shift (camera-tilt parallax) and placement noise over the parts that
    are there, so one displaced part cannot drag either."""
    ok = [m for m in measures.values()
          if m.get("in_view") and _sigmoid(missing_logit(m)) < MISSING_P]
    if len(ok) < 5:
        return dict(dx=0.0, dy=0.0, sigma=0.0, sigma_deg=0.0, n=len(ok))
    dx = np.array([m["dx"] for m in ok]); dy = np.array([m["dy"] for m in ok])
    th = np.array([m["dth"] for m in ok])
    mx, my = float(np.median(dx)), float(np.median(dy))
    r = np.hypot(dx - mx, dy - my)
    sigma = float(np.median(r)) / math.sqrt(2 * math.log(2))   # median radius of a 2-D normal is 1.177 sigma
    sdeg = 1.4826 * float(np.median(np.abs(th - np.median(th))))
    return dict(dx=mx, dy=my, sigma=sigma, sigma_deg=sdeg, n=len(ok))


def offset_limit_px(px_per_mm: float, sigma: float) -> float:
    return max(config.MAX_OFFSET_UM / 1000.0 * px_per_mm, config.PHOTO_OFFSET_SIGMA_K * sigma)


# ----------------------------------------------------------------- one part
def judge_part(p: Part, m: dict, *, noise: dict, px_per_mm: float, stats: dict | None,
               single_golden: bool) -> PartResult:
    pr = PartResult(p, measures=m)
    if not m.get("in_view"):
        pr.calls.append(Call("view", "NOT CHECKED", "outside this photo", measured_against="photo coverage"))
        return pr
    if m["glare"] > config.PHOTO_MAX_GLARE_FRAC:
        pr.calls.append(Call("glare", "NOT CHECKED",
                             f"{m['glare']:.0%} of the body is blown out by glare; reshoot with diffuse light",
                             measured_against="photo exposure"))
        return pr
    against = "fixed thresholds (single golden)" if single_golden else "golden spread"

    # 1. missing
    z = missing_logit(m)
    if single_golden or not stats:
        p_miss = _sigmoid(z)
        if p_miss >= MISSING_P:
            pr.calls.append(Call("missing", "MISSING", "no body on the footprint", p_miss, False, against))
            return pr
    else:
        mu, sd = stats.get("missing_logit", (0.0, 0.0, 0))[:2]
        if sd > MISSING_SPREAD_MAX:
            pr.calls.append(Call("missing", "UNINSPECTED",
                                 f"good parts vary too much between golden shots (logit sd {sd:.1f})",
                                 measured_against="golden spread"))
            return pr
        zz = (z - mu) / max(sd, LOGIT_SD_FLOOR)
        if zz > MISSING_Z:
            pr.calls.append(Call("missing", "MISSING", f"{zz:.1f} sd from the goldens",
                                 _sigmoid(zz - MISSING_Z), False, against))
            return pr

    # body is there: record where it sits, net of the board-wide shift
    pr.dx_px, pr.dy_px, pr.dtheta_deg = m["dx"] - noise["dx"], m["dy"] - noise["dy"], m["dth"]

    # 2. tombstone: short along the pad axis and one end of the footprint bare
    g, t = m.get("body_g"), m.get("body_t")
    if g and t:
        ext = t["extent"] / max(g["extent"], 1.0)
        bare_end = min(t["end_a"] / max(g["end_a"], 1e-3), t["end_b"] / max(g["end_b"], 1e-3))
        if ext < config.PHOTO_TOMBSTONE_LENGTH and bare_end < 0.4:
            pr.calls.append(Call("tombstone", "TOMBSTONE",
                                 f"body spans {ext:.0%} of its golden length, one end bare",
                                 _sigmoid(8 * (config.PHOTO_TOMBSTONE_LENGTH - ext)), False, against))
            return pr
        # 3. wrong part by size, only against a measured golden spread: on one golden
        #    the segmented body size of correctly placed parts varied 0.5x-2.7x on the
        #    fixture, so a fixed band would fail good boards. Colour class was worse.
        if stats and not single_golden:
            za = _z(math.log(max(t["area"], 1.0) / max(g["area"], 1.0)), stats.get("log_area"))
            zw = _z(math.log(max(t["short"], 1.0) / max(g["short"], 1.0)), stats.get("log_width"))
            zz = max(abs(za), abs(zw))
            if zz > WRONG_PART_Z:
                pr.calls.append(Call("wrong_part", "WRONG PART",
                                     f"body size {zz:.1f} sd from the goldens (area z {za:+.1f}, width z {zw:+.1f})",
                                     _sigmoid(zz - WRONG_PART_Z), False, against))
                return pr

    # 4. polarity
    if p.polarized:
        pr.calls.extend(_polarity(p, m, single_golden, stats))
        if any(c.state == "REVERSED" for c in pr.calls):
            return pr

    # 5. offset, net of the board-wide shift, against max(150 um, k sigma)
    lim = offset_limit_px(px_per_mm, noise["sigma"])
    off = math.hypot(pr.dx_px, pr.dy_px)
    um = off / px_per_mm * 1000.0 if px_per_mm else float("nan")
    if off > lim:
        pr.calls.append(Call("offset", "OFFSET", f"{um:.0f} um off its lands (limit {lim / px_per_mm * 1000:.0f} um)",
                             _sigmoid((off - lim) / max(noise["sigma"], 1.0)), False,
                             "board placement noise and 150 um"))
        return pr

    # 6. skew: silkscreen outline where there is one, else the golden body
    if not math.isnan(m.get("skew_silk", float("nan"))) and m.get("silk_len", 0) >= 150:
        sk, ref = m["skew_silk"], "silkscreen outline"
    else:
        sk, ref = m["dth"], "golden body"
    quarter = _quarter_turn(g, t)
    lim_deg = max(config.MAX_SKEW_DEG, 3 * noise.get("sigma_deg", 0.0))
    if quarter:
        pr.calls.append(Call("skew", "SKEW", "turned about 90 deg against its golden body", 1.0, False, "golden body"))
    elif abs(sk) > lim_deg:
        pr.calls.append(Call("skew", "SKEW", f"{sk:+.1f} deg against the {ref} (limit {lim_deg:.0f})",
                             _sigmoid((abs(sk) - lim_deg) / 2.0), False, ref))
    return pr


def _quarter_turn(g: dict | None, t: dict | None) -> bool:
    if not g or not t:
        return False
    if g["long"] / max(g["short"], 1) < 1.5 or t["long"] / max(t["short"], 1) < 1.5:
        return False                     # near-square bodies have no long axis to compare
    d = abs(((t["axis"] - g["axis"]) + 90.0) % 180.0 - 90.0)
    return d > 60.0


def _polarity(p: Part, m: dict, single_golden: bool, stats: dict | None) -> list[Call]:
    calls = []
    if "stripe_g" in m:
        sg, st = m["stripe_g"], m["stripe_t"]
        if abs(sg) < STRIPE_CLEAR:
            calls.append(Call("polarity", "UNINSPECTED",
                              f"golden band/bevel unclear ({sg:+.2f}); polarity not judged",
                              measured_against="golden band"))
        elif abs(st) >= STRIPE_CLEAR and np.sign(st) != np.sign(sg):
            consistent = (not single_golden and stats
                          and stats.get("stripe_consistent", (0, 0, 0))[0] >= 1.0)
            state = "REVERSED" if consistent else "POLARITY REVIEW"
            calls.append(Call("polarity", state,
                              f"band/bevel at the opposite end to the golden ({st:+.2f} vs {sg:+.2f})",
                              _sigmoid(abs(st) - STRIPE_CLEAR), False, "golden band"))
        return calls
    if m["turn_margin"] < TURN_REVIEW:
        calls.append(Call("polarity", "POLARITY REVIEW",
                          f"marking matches the golden better turned 180 deg (margin {m['turn_margin']:+.2f})",
                          _sigmoid(-m["turn_margin"] * 5), False, "golden marking"))
    return calls


# ------------------------------------------------------------- board verdict
def board_verdict(result: BoardResult) -> None:
    """The part-state table from the design, applied in place."""
    reasons = []
    fails = [pr for pr in result.parts if any(c.severity == "FAIL" for c in pr.calls)]
    reviews = [pr for pr in result.parts if any(c.severity == "REVIEW" for c in pr.calls)]
    wholly_uninspected = [pr for pr in result.parts
                          if any(c.check == "missing" and c.state == "UNINSPECTED" for c in pr.calls)]
    if result.board_calls:
        reasons += [c.detail for c in result.board_calls]
    if fails:
        reasons.append(f"{len(fails)} part(s) failed: " + ", ".join(f"{p.refdes} {p.state}" for p in fails))
    if reviews:
        reasons.append(f"{len(reviews)} part(s) need review: " + ", ".join(f"{p.refdes} {p.state}" for p in reviews
                                                                           if p not in fails))
    frac = len(wholly_uninspected) / max(len(result.parts), 1)
    if wholly_uninspected:
        reasons.append(f"{len(wholly_uninspected)} part(s) uninspected (golden too variable): "
                       + ", ".join(p.refdes for p in wholly_uninspected))
    if fails:
        result.verdict = "FAIL"
    elif reviews or result.board_calls or frac > config.PHOTO_UNINSPECTED_FRAC:
        result.verdict = "REVIEW"
    else:
        result.verdict = "PASS"
    result.reasons = reasons


# ---------------------------------------------------------------- calibration
def calibrate(labels: list[tuple[float, int]]):
    """Platt scaling once a check has enough labels of both kinds; else None."""
    if len(labels) < config.PHOTO_CALIBRATE_MIN_LABELS:
        return None
    y = np.array([l for _, l in labels])
    if min(int(y.sum()), int(len(y) - y.sum())) < config.PHOTO_CALIBRATE_MIN_PER_CLASS:
        return None
    from sklearn.linear_model import LogisticRegression
    x = np.array([[s] for s, _ in labels])
    lr = LogisticRegression().fit(x, y)
    return lambda s: float(lr.predict_proba([[s]])[0, 1])


def apply_calibration(result: BoardResult, calibrators: dict) -> None:
    for pr in result.parts:
        for c in pr.calls:
            f = calibrators.get(c.check)
            if f is not None and c.score is not None:
                c.score, c.calibrated = f(c.score), True
