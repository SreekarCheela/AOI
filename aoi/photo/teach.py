"""Teaching a board: where the parts are, how big a millimetre is, what good looks like.

One library path serves both the Tk Program screen and `python -m aoi_photo teach`:

    propose_boxes   candidate part bodies on golden 1 (non-soldermask blobs)
    self_check      golden parts that sit off their own silkscreen outline
    build_program   gates, scale, golden statistics, versioned save

Golden statistics are what turn thresholds from guesses into measurements: every
extra golden is aligned onto golden 1 and measured exactly as a test photo would
be, so each part's score spread over good shots is known.
"""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import config
from . import features as F
from . import gate, register
from .io import PhotoError, grey, load_bgr
from .judge import missing_logit
from .model import Part
from .program import (Program, check_golden_count, copy_goldens, new_version_folder, now)


class SelfCheckError(PhotoError):
    """Golden parts off their outline that nobody has accepted as intended."""

    def __init__(self, flags: list[tuple[str, str]]):
        self.flags = flags
        super().__init__("golden self-check flagged " + "; ".join(f"{r}: {d}" for r, d in flags)
                         + ". Fix the golden, or accept them as intended (--accept-selfcheck "
                         + ",".join(r for r, _ in flags) + ")")


# ------------------------------------------------------------ box proposals
def propose_boxes(golden_bgr: np.ndarray, min_side_px: int = 25, max_side_px: int = 400,
                  detector_weights=None) -> list[tuple[int, int, int, int]]:
    """Candidate part bodies for the operator to accept, rename or delete.

    The trained part detector (detect.py) when its weights exist; otherwise
    the colour method below, which cannot tell solder from a ceramic body and
    over-proposes on lands, leads and through-hole pads. Either way, nothing
    whose centre is off the board (outside it, or in a cutout) is proposed.
    """
    from . import detect
    weights = detector_weights or detect.DEFAULT_WEIGHTS
    if detect.PartDetector.available(weights):
        return detector_boxes(golden_bgr, detect.PartDetector(weights))
    return colour_boxes(golden_bgr, min_side_px, max_side_px)


def detector_boxes(golden_bgr: np.ndarray, det, score_min: float | None = None
                   ) -> list[tuple[int, int, int, int]]:
    from . import detect
    m = gate.board_mask(golden_bgr)
    k = m.shape[1] / golden_bgr.shape[1]
    out = []
    for x, y, w, h, _ in det.detect(golden_bgr, detect.SCORE_MIN if score_min is None else score_min):
        cx, cy = int((x + w / 2) * k), int((y + h / 2) * k)
        if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx]:
            out.append((int(round(x)), int(round(y)), int(round(w)), int(round(h))))
    return out


def colour_boxes(golden_bgr: np.ndarray, min_side_px: int = 25, max_side_px: int = 400
                 ) -> list[tuple[int, int, int, int]]:
    """Fallback proposals from body colour alone, used when no detector is trained.

    Bodies on these boards are either neutral dark (resistors, diodes, SOT and
    SOT-223 packages: a* ~131, chroma < 8) or tan (ceramic caps: b* 137-148) on
    green soldermask (a* 104-117). Each colour is segmented on the board region,
    closed, and every blob's box is snapped to the rows and columns it mostly
    fills, which trims the shadows that otherwise merge into it.

    Measured on the 2026-09-25 golden: 18 of 37 taught bodies get a proposal at
    IoU >= 0.5, most of the rest one that is on the part but loose (IoU 0.2-0.5),
    and there are extra proposals to delete. It saves drawing, not reviewing."""
    s = min(1.0, 2000 / max(golden_bgr.shape[:2]))
    small = cv2.resize(golden_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else golden_bgr
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a - 128, b - 128)
    onboard = gate.board_mask(golden_bgr, side=max(small.shape[:2]))
    onboard = cv2.erode(onboard, np.ones((5, 5), np.uint8))       # board edge is not a part
    k = max(3, int(0.006 * max(small.shape[:2])) | 1)
    out = []
    for colour in ((a >= 124) & (chroma < 12) & (L < 130),               # neutral dark bodies
                   (a >= 127) & (b >= 137) & (L > 60) & (L < 175)):      # tan ceramic
        m = (colour & (onboard > 0)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        n, lb, st, _ = cv2.connectedComponentsWithStats(m)
        for j in range(1, n):
            x, y, w, h, _ = st[j]
            sub = m[y:y + h, x:x + w]
            cols = np.nonzero(sub.mean(0) >= 0.5)[0]; rows = np.nonzero(sub.mean(1) >= 0.5)[0]
            if len(cols) < 3 or len(rows) < 3:
                continue
            x2, y2 = x + cols[0], y + rows[0]
            w2, h2 = cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1
            box = tuple(int(round(v / s)) for v in (x2, y2, w2, h2))
            lo, hi = min(box[2], box[3]), max(box[2], box[3])
            if lo < min_side_px or hi > max_side_px or hi / max(lo, 1) > 4.0:
                continue
            if sub[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].mean() < 0.6:
                continue
            out.append(box)
    return out


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


# --------------------------------------------------------------- self-check
def self_check(golden_bgr: np.ndarray, parts: list[Part]) -> tuple[list[tuple[str, str]], list[str]]:
    """Golden parts off their own silkscreen: (flags, not_checkable).

    Skew uses the outline's straight segments (mod 90). A quarter turn needs the
    part's own outline rectangle; where the outline is shared or open the part is
    listed as not checkable rather than passed.

    Reads only what it needs (features.golden_outline): no alignment, presence
    or placement, which made a board with nothing checkable cost ~8 s."""
    ref = F.Frame.of(golden_bgr)
    flags, unchecked = [], []
    for p, m in zip(parts, F.pool().map(lambda q: F.golden_outline(ref, q), parts)):
        sk = m.get("skew_silk", float("nan"))
        if not math.isnan(sk) and m.get("silk_len", 0) >= 150 and abs(sk) > config.MAX_SKEW_DEG:
            flags.append((p.refdes, f"{sk:+.1f} deg against its silkscreen outline"))
            continue
        oa, ba, body = m.get("outline_axis"), m.get("body_axis"), m.get("body_g")
        # a near-square body blob (a SOT-23 with its lead shadows) has no long axis to compare
        if oa is None or ba is None or not body or body["long"] / max(body["short"], 1) < 1.5:
            unchecked.append(p.refdes)
            continue
        d = abs(((ba - oa) + 90.0) % 180.0 - 90.0)
        if d > 60.0:
            flags.append((p.refdes, "turned about 90 deg across its outline"))
    return flags, unchecked


# -------------------------------------------------------------- scale
def scale_from(golden_bgr: np.ndarray, px_per_mm: float | None, board_width_mm: float | None) -> float:
    """px per mm, or 0.0 when no scale is given.

    Scale is optional: without it every offset is judged against the board's
    own placement noise (k sigma) and reported in px, and the program is stamped
    "no scale: offsets in px". Presence, polarity and skew do not need it."""
    if px_per_mm is not None and px_per_mm <= 0:
        raise PhotoError(f"px/mm must be positive, not {px_per_mm}")
    if board_width_mm is not None and board_width_mm <= 0:
        raise PhotoError(f"board width must be positive, not {board_width_mm} mm")
    if px_per_mm:
        return float(px_per_mm)
    if board_width_mm:
        w = gate.board_width_px(golden_bgr)
        if w <= 0:
            raise PhotoError("could not find the board outline to measure its width; give --px-per-mm")
        return w / float(board_width_mm)
    return 0.0


# -------------------------------------------------------- duplicate goldens
# Median grey-level residual (after a gain/offset fit) between golden 1 and
# another golden aligned onto it, on a 1000 px copy. Measured: the same file,
# a re-encoded copy and a warped + brightened copy of the fixture golden read
# 0.00-0.42; burst shots of the same unmoved board 1.14-2.10; re-framed shots
# of the same board 3.5-4.5; a different board 7.9.
DUPLICATE_RESIDUAL = 0.75


def shot_residual(golden1: np.ndarray, other: np.ndarray, H: np.ndarray) -> float:
    w, c = register.warp_to_ref(other, H, golden1.shape)
    s = 1000 / max(golden1.shape[:2])
    ga = cv2.resize(grey(golden1), None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    gb = cv2.resize(grey(w), None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    cm = cv2.erode((cv2.resize(c, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_NEAREST) == 255)
                   .astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if cm.sum() < 100:
        return float("inf")
    x, y = ga[cm], gb[cm]
    k, b = np.polyfit(y, x, 1)
    return float(np.median(np.abs(x - (k * y + b))))


# ---------------------------------------------------------- golden stats
def golden_stats(goldens: list[np.ndarray], parts: list[Part],
                 registrations: dict[int, np.ndarray] | None = None) -> dict:
    """Per part, the spread of every judged score over the golden shots.

    Each golden after the first, aligned onto golden 1 and measured as a test
    photo, is one sample. Golden 1 against itself is not: it reads a perfect
    match by construction (phys and gap ratio 1, zero size change), and counting
    it pulled every mean toward "perfect" and shrank every spread. With three
    goldens that leaves two samples, so the judge floors each spread
    (judge.missing_cut, LOG_AREA_SD, LOG_WIDTH_SD) rather than trust it alone.
    For the band, golden 1's own stripe is a real reading and is counted.

    `registrations` (golden index -> homography onto golden 1) skips aligning
    again when build_program already did."""
    ref = F.Frame.of(goldens[0])
    samples: dict[str, dict[str, list[float]]] = {
        p.refdes: {"missing_logit": [], "stripe": [], "log_area": [], "log_width": []} for p in parts}
    g1_stripe: dict[str, float] = {}
    for i, g in enumerate(goldens[1:], start=1):
        H = (registrations or {}).get(i)
        if H is None:
            reg = register.register(grey(goldens[0]), grey(g))
            if not reg.ok:
                raise PhotoError(f"golden {i + 1} does not align onto golden 1: {reg.note}")
            H = reg.H
        w, c = register.warp_to_ref(g, H, goldens[0].shape)
        img = F.Frame.of(w, c)
        for p, m in zip(parts, F.measure_all(ref, img, parts).values()):
            if not m.get("in_view"):
                continue
            samples[p.refdes]["missing_logit"].append(missing_logit(m))
            g, t = m.get("body_g"), m.get("body_t")
            if g and t:
                samples[p.refdes]["log_area"].append(math.log(max(t["area"], 1.0) / max(g["area"], 1.0)))
                samples[p.refdes]["log_width"].append(math.log(max(t["short"], 1.0) / max(g["short"], 1.0)))
            if "stripe_t" in m:
                g1_stripe[p.refdes] = m["stripe_g"]
                samples[p.refdes]["stripe"].append(m["stripe_t"])
    stats = {}
    for r, d in samples.items():
        s: dict = {}
        for key in ("missing_logit", "log_area", "log_width"):
            v = d[key]
            if v:
                s[key] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) >= 2 else 0.0, len(v))
        st = ([g1_stripe[r]] if r in g1_stripe else []) + d["stripe"]
        if st:
            clear_same = all(abs(v) >= 1.5 and np.sign(v) == np.sign(st[0]) for v in st)
            s["stripe_consistent"] = (1.0 if clear_same and len(st) >= 3 else 0.0, 0.0, len(st))
            # goldens (golden 1 included) whose band is clear and at golden 1's end
            s["stripe_agree"] = (float(sum(1 for v in st if abs(v) >= 1.5 and abs(st[0]) >= 1.5
                                           and np.sign(v) == np.sign(st[0]))), 0.0, len(st))
        stats[r] = s
    return stats


# ---------------------------------------------------------------- build
@dataclass
class TeachReport:
    program: Program
    golden_problems: dict[str, list[str]]
    selfcheck_flags: list[tuple[str, str]]
    selfcheck_unchecked: list[str]


def build_program(name: str, golden_paths: list[str | Path], parts: list[Part], *,
                  px_per_mm: float | None = None, board_width_mm: float | None = None,
                  root: str | Path = config.PHOTO_PROGRAMS_DIR, single_golden: bool = False,
                  accept_selfcheck: list[str] | str = ()) -> TeachReport:
    golden_paths = [Path(p) for p in golden_paths]
    check_golden_count(len(golden_paths), single_golden)
    if not parts:
        raise PhotoError("no parts taught: box at least one part on golden 1")
    goldens = [load_bgr(p) for p in golden_paths]
    problems = {p.name: gate.golden_problems(g) for p, g in zip(golden_paths, goldens)}
    bad = {k: v for k, v in problems.items() if v}
    if bad:
        raise PhotoError("golden photo(s) rejected: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in bad.items()))
    regs: dict[int, np.ndarray] = {}
    for i, (p, g) in enumerate(zip(golden_paths[1:], goldens[1:]), start=1):
        reg = register.register(grey(goldens[0]), grey(g))
        if not reg.ok:
            raise PhotoError(f"{p.name} does not align onto {golden_paths[0].name}: {reg.note}")
        regs[i] = reg.H
    for i, H in regs.items():
        # the same shot twice measures no spread at all; the golden statistics
        # would then be one board's noise, not how good boards vary
        for j in [0] + [k for k in regs if k < i]:
            Hij = H if j == 0 else np.linalg.inv(regs[j]) @ H
            if shot_residual(goldens[j], goldens[i], Hij) < DUPLICATE_RESIDUAL:
                raise PhotoError(f"golden {i + 1} is the same shot as golden {j + 1}; "
                                 "shoot a different good board or re-seat it")
    ppm = scale_from(goldens[0], px_per_mm, board_width_mm)
    small = min(parts, key=lambda q: max(q.w, q.h))
    if max(small.w, small.h) < config.PHOTO_MIN_PART_PX:
        raise PhotoError(f"smallest part {small.refdes} is {max(small.w, small.h)} px long "
                         f"(need {config.PHOTO_MIN_PART_PX}); move the camera closer")
    H, W = goldens[0].shape[:2]
    off = [q.refdes for q in parts if q.x < 0 or q.y < 0 or q.x + q.w > W or q.y + q.h > H]
    if off:
        raise PhotoError(f"part box(es) outside golden 1: {', '.join(off)}")

    flags, unchecked = self_check(goldens[0], parts)
    accepted = [r.strip() for r in (accept_selfcheck.split(",") if isinstance(accept_selfcheck, str)
                                    else accept_selfcheck) if r.strip()]
    pending = [(r, d) for r, d in flags if r not in accepted and "all" not in accepted]
    if pending:
        raise SelfCheckError(pending)

    stats = {} if single_golden else golden_stats(goldens, parts, regs)
    folder, version = new_version_folder(root, name)
    try:
        prog = Program(name=name, version=version, px_per_mm=ppm, parts=parts,
                       goldens=copy_goldens(golden_paths, folder), single_golden=single_golden,
                       accepted_selfcheck=[r for r, _ in flags], golden_stats=stats, created=now(),
                       folder=str(folder))
        prog.save()
        if config.PHOTO_DIFF and len(goldens) >= 2:
            from . import diff
            np.save(prog.spread_path, diff.spread_map(goldens))
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)     # never leave a half-written version behind
        raise
    return TeachReport(prog, problems, flags, unchecked)
