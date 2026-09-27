"""Part names read from the board's own silkscreen.

A board already carries every part's name: R12 printed beside resistor 12, D4
beside diode 4. Teach reads them off golden 1 with an offline OCR (RapidOCR:
the PP-OCR detector, angle classifier and recogniser as ONNX models, run on
onnxruntime; nothing is downloaded at run time) and gives each part box the
label that belongs to it. The name's letter says what the part is, and that
says whether it can go in backwards, so no part needs marking by hand
(designator letters: ASME Y14.44 / IEEE 315; the FPIC PCB dataset uses the
same letter-to-class link, arXiv 2202.08414).

    1  OCR the board at 0/90/180/270 deg: silkscreen text runs every way, and
       the recogniser reads a line best upright. Per-character boxes let a line
       like "R7 R4 T2" or "R10D1" give each name its own position.
    2  A name is a known designator letter plus a number that does not start
       with 0. A line that is not made only of names ("22R0" on a resistor,
       "PT-PCB-004-03") keeps only its stand-alone names; a name whose centre
       lies on a part body is marking, not silkscreen, and is dropped.
    3  Names go to boxes by one optimal assignment (Hungarian) on the squared
       gap between name and box, in units of the name's own character height.
       Squared cost matters: a column of four names under a column of four
       parts is then matched in order, which is how dense blocks are labelled.
    4  Boxes still without a name get a second, closer read of their
       neighbourhood at twice the resolution, then the assignment runs again.

Measured on the 2026-09-25 fixture (PT-PCB-004-03, one iPhone golden) in
tests/test_photo_names.py and the scratch evaluation; those numbers are
in-sample for this board.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import gate
from .model import Part

# Designator letter -> (part type, polarized). Multi-letter prefixes first so
# BD1 is a bridge, not a B-something.
PREFIXES: dict[str, tuple[str, bool]] = {
    "LED": ("led", True), "BD": ("bridge", True), "BR": ("bridge", True), "DB": ("bridge", True),
    "ZD": ("diode", True), "IC": ("ic", True), "RN": ("resistor", False), "FB": ("other", False),
    "TP": ("other", False), "SW": ("other", False), "CN": ("other", False), "VR": ("ic", True),
    "R": ("resistor", False), "C": ("capacitor", False), "L": ("other", False),
    "D": ("diode", True), "Q": ("sot23", True), "T": ("sot23", True), "U": ("ic", True),
    "F": ("other", False), "J": ("other", False), "Y": ("other", False), "X": ("other", False),
    "K": ("other", False), "S": ("other", False), "P": ("other", False), "M": ("other", False),
}
_ALT = "|".join(sorted(PREFIXES, key=len, reverse=True))
_TOKEN = rf"(?:{_ALT})[1-9]\d{{0,2}}"
FULL = re.compile(rf"^(?:{_TOKEN})+$")
TOKEN = re.compile(rf"({_ALT})([1-9]\d{{0,2}})")
ALONE = re.compile(rf"(?<![A-Z0-9])({_ALT})([1-9]\d{{0,2}})(?![A-Z0-9])")

# One-letter misreads seen on silkscreen, used only to fill a gap: a letter
# that names nothing else on the board, read where a common series is missing
# that number (the fixture's R14 read as F14).
CONFUSED = {"F": "R", "P": "R", "K": "R", "E": "C", "G": "C", "O": "D", "J": "T", "I": "T", "Q": "D"}
# digit misreads in silkscreen fonts (and 6/9 on upside-down text)
DIGIT_CONFUSED = {"6": "89", "8": "69", "9": "68", "1": "7", "7": "1", "5": "6", "3": "8", "0": "8"}

MIN_SCORE = 0.5                 # OCR line confidence below this is not a name
GATE = 6.0                      # max gap, in character heights, name to box
LOCAL_SCALE = 1.0               # second read: neighbourhood at this scale
LOCAL_ALL = False               # close-up of every box, not only the unnamed
LOCAL_MAX_MARGIN = 4.0          # close-up margin cap, character heights
LOCAL_READERS = None            # None: every recogniser; ("vision",) or ("rapid",) for one
LOCAL_MARGIN = 2.5              # neighbourhood = box grown this x its long side


@dataclass
class Label:
    """One name read off the board, in golden-1 pixels."""
    name: str
    x: float                    # centre
    y: float
    h: float                    # character height, px
    score: float                # summed confidence of the readings that agree
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # x0, y0, x1, y1
    votes: dict = field(default_factory=dict)                     # every name read here -> confidence

    @property
    def prefix(self) -> str:
        return TOKEN.match(self.name).group(1)


@dataclass
class Naming:
    names: list[str | None]     # per box, None where no name was found
    labels: list[Label] = field(default_factory=list)
    note: str = ""
    margin: list[float] = field(default_factory=list)   # per box: next choice's extra cost
    cost: list[float | None] = field(default_factory=list)

    @property
    def found(self) -> int:
        return sum(n is not None for n in self.names)

    @property
    def unsure(self) -> list[str]:
        """Names worth a glance: close to their next choice, or a poor fit."""
        return [n for n, m, c in zip(self.names, self.margin, self.cost)
                if n and (m < UNSURE_MARGIN or (c is not None and c > 5.0))]


def type_of(name: str) -> tuple[str, bool]:
    """(part type, polarized) from a designator, ("other", False) if unknown."""
    m = TOKEN.fullmatch(name or "")
    return PREFIXES.get(m.group(1), ("other", False)) if m else ("other", False)


# ------------------------------------------------------------------ OCR
# Two recognisers, because each misses what the other reads: on the fixture
# PP-OCR (RapidOCR) read C4, C5, R2, R3, R6 that Apple Vision did not, and Vision
# read R8 (PP-OCR: "R6"), R12, R13, C7, C8. Every reading is a vote.
_RAPID = None
_LOCK = threading.Lock()
_DIGITS = str.maketrans({"O": "0", "I": "1"})
_NUMBERISH = re.compile(rf"({_ALT})([0-9OI]{{1,3}})")


def have_rapid() -> bool:
    import importlib.util
    return importlib.util.find_spec("rapidocr") is not None


def have_vision() -> bool:
    import importlib.util
    import sys
    return sys.platform == "darwin" and importlib.util.find_spec("Vision") is not None


def available() -> bool:
    return have_rapid() or have_vision()


def _rapid():
    global _RAPID
    with _LOCK:
        if _RAPID is None:
            from rapidocr import RapidOCR
            # use_cls off: every quarter turn is read anyway, so each line is
            # upright in one of them; the angle classifier flips a line's
            # crop but not its character boxes, which swaps "R6R3" to R3, R6.
            # Det limit "max": the default blows a small crop up until its
            # short side is 736 px, and text that large is not detected.
            _RAPID = RapidOCR(params={"Global.log_level": "error", "Global.max_side_len": 8192,
                                      "Global.min_side_len": 8, "Global.use_cls": False,
                                      "Global.return_word_box": True,
                                      "Global.return_single_char_box": True,
                                      "Det.limit_type": "max", "Det.limit_side_len": 8192,
                                      # the OCR pool runs several reads at once
                                      "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                                      "EngineConfig.onnxruntime.inter_op_num_threads": 1})
        return _RAPID


def _unrotate(u: np.ndarray, v: np.ndarray, k: int, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel (u, v) of np.rot90(img, k) back to (x, y) of img (H x W)."""
    if k == 0:
        return u, v
    if k == 1:
        return W - 1 - v, u
    if k == 2:
        return W - 1 - u, H - 1 - v
    return v, H - 1 - u


def normalise(text: str) -> str:
    """Upper case, and O/I read as 0/1 where they stand in a designator's
    number (Vision reads R10 as "R1O" and D1 as "Di"). Length is kept."""
    t = text.upper()
    return _NUMBERISH.sub(lambda m: m.group(1) + m.group(2).translate(_DIGITS), t)


def designators(text: str) -> list[tuple[str, int, int]]:
    """(name, start, end) of every designator in one line of text. A line made
    only of names ("R7 R4 T2", "R10D1") gives all of them; any other line only
    its stand-alone names, so "22R0" on a resistor body gives none."""
    t = normalise(text)
    joined = "".join(ch for ch in t if not ch.isspace())
    if joined and FULL.match(joined):
        idx = [i for i, ch in enumerate(t) if not ch.isspace()]
        return [(m.group(0), idx[m.start()], idx[m.end() - 1] + 1) for m in TOKEN.finditer(joined)]
    return [(m.group(0), m.start(), m.end()) for m in ALONE.finditer(t)]


def _read_rapid(img: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
    """(name, score, corner points) from PP-OCR, in `img` pixels."""
    out = []
    r = _rapid()(img)
    for line, score in zip(r.word_results or (), r.scores or ()):
        if score < MIN_SCORE:
            continue
        chars = [(c, np.asarray(b, np.float32)) for c, _, b in line if c.strip()]
        text = "".join(c for c, _ in chars)
        for name, a, b in designators(text):
            out.append((name, float(score), np.concatenate([p for _, p in chars[a:b]])))
    return out


def _read_vision(img: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
    """(name, score, corner points) from Apple Vision (macOS), in `img` pixels."""
    import Vision
    from Foundation import NSData
    ok, buf = cv2.imencode(".png", img)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
        NSData.dataWithBytes_length_(buf.tobytes(), len(buf)), None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(0)                  # accurate
    req.setUsesLanguageCorrection_(False)        # designators are not words
    handler.performRequests_error_([req], None)
    H, W = img.shape[:2]
    out = []
    for o in req.results() or []:
        c = o.topCandidates_(1)[0]
        score = float(c.confidence())
        if score < MIN_SCORE:
            continue
        text = str(c.string())
        for name, a, b in designators(text):
            box, _ = c.boundingBoxForRange_error_((a, b - a), None)
            if box is None:
                continue
            bb = box.boundingBox()                  # normalised, origin bottom left
            x0, x1 = bb.origin.x * W, (bb.origin.x + bb.size.width) * W
            y0, y1 = (1 - bb.origin.y - bb.size.height) * H, (1 - bb.origin.y) * H
            out.append((name, score, np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])))
    return out


OCR_WORKERS = 6                 # both recognisers release the GIL; 6 threads ~ 4x on an M-series Mac
_POOL = None


def _pool():
    global _POOL
    from concurrent.futures import ThreadPoolExecutor
    with _LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(OCR_WORKERS, thread_name_prefix="aoi-ocr")
    return _POOL


def _crop(bgr: np.ndarray, region, scale: float):
    x0, y0 = 0, 0
    img = bgr
    if region is not None:
        x0, y0, w, h = region
        w, h = w + min(x0, 0), h + min(y0, 0)
        x0, y0 = max(0, x0), max(0, y0)
        img = bgr[y0:y0 + h, x0:x0 + w]
    if img.size and scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img, x0, y0


def _read_turn(read, img: np.ndarray, k: int, x0: int, y0: int, scale: float) -> list[Label]:
    H, W = img.shape[:2]
    out = []
    for name, score, pts in read(np.ascontiguousarray(np.rot90(img, k))):
        xs, ys = _unrotate(pts[:, 0], pts[:, 1], k, H, W)
        hh = float(pts[:, 1].max() - pts[:, 1].min())   # character height, upright frame
        xs, ys = x0 + xs / scale, y0 + ys / scale
        out.append(Label(name, float(xs.mean()), float(ys.mean()), hh / scale, score,
                         (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))))
    return out


def read_many(bgr: np.ndarray, regions: list, scale: float = 1.0, turns=(0, 1, 2, 3),
              readers=None) -> list[list[Label]]:
    """read_labels for several regions at once, every (region, turn,
    recogniser) on the OCR pool; one merged label list per region."""
    want = set(readers or ("rapid", "vision"))
    readers = (([_read_rapid] if have_rapid() and "rapid" in want else [])
               + ([_read_vision] if have_vision() and "vision" in want else []))
    jobs = []
    for ri, region in enumerate(regions):
        img, x0, y0 = _crop(bgr, region, scale)
        if img.size == 0:
            continue
        for k in turns:
            for read in readers:
                jobs.append((ri, _pool().submit(_read_turn, read, img, k, x0, y0, scale)))
    found: list[list[Label]] = [[] for _ in regions]
    for ri, f in jobs:
        found[ri] += f.result()
    return [merge(f) for f in found]


def read_labels(bgr: np.ndarray, region: tuple[int, int, int, int] | None = None,
                scale: float = 1.0) -> list[Label]:
    """Every designator in `bgr` (or its region x, y, w, h), read at all four
    quarter turns by every recogniser available, in full-image pixels, the
    readings of one place pooled into one name (see merge)."""
    return read_many(bgr, [region], scale)[0]


def merge(labels: list[Label]) -> list[Label]:
    """One name per place. Readings within 0.8 character heights of each other
    are one place; its name is the one with the most summed confidence over
    every recogniser and turn (R8 read by Vision twice beats PP-OCR's one "R6"),
    its score that sum. The same name at two places close together is one."""
    places: list[list[Label]] = []
    for lb in sorted(labels, key=lambda l: -l.score):
        for pl in places:
            if np.hypot(lb.x - pl[0].x, lb.y - pl[0].y) < 0.8 * max(lb.h, pl[0].h):
                pl.append(lb)
                break
        else:
            places.append([lb])
    out: list[Label] = []
    for pl in places:
        votes: dict[str, float] = {}
        for lb in pl:
            votes[lb.name] = votes.get(lb.name, 0.0) + lb.score
        name = max(votes, key=votes.get)
        same = [lb for lb in pl if lb.name == name]
        box = tuple(float(v) for v in np.mean([l.box for l in same], axis=0))
        out.append(Label(name, float(np.mean([l.x for l in same])), float(np.mean([l.y for l in same])),
                         float(np.median([l.h for l in same])), votes[name], box, votes))
    kept: list[Label] = []
    for lb in sorted(out, key=lambda l: -l.score):
        if not any(o.name == lb.name and np.hypot(lb.x - o.x, lb.y - o.y) < 2.5 * max(lb.h, o.h) for o in kept):
            kept.append(lb)
    return kept


# ------------------------------------------------------------------ assignment
# How labels sit round their parts, from placement practice (IPC-7351 and EDA
# tools' 9-position autoplace; FPIC arXiv 2202.08414 on clustered labels): a
# board repeats a few offset styles, e.g. "centred below, 1.5 character heights
# off". The styles are found by voting, then every name-part pair is scored by
# how well it fits one, plus whether the name's letter fits the part's body.
STYLE_BW = 0.4                  # mean-shift bandwidth, character heights
STYLE_MIN_VOTES = 3.0
STYLE_K = 1                     # nearest parts each name votes for
STYLE_SG, STYLE_SA = 0.5, 0.3   # spread of a style: gap, lateral (character heights)
REACH = 8.0                     # a name further than this from a part is not its name
MISS = 9.0                      # cost of leaving a part unnamed or a name unused (~3 sigma)
NO_STYLE = 4.0                  # extra cost of a pair that fits no style
TYPE_COST = {"ok": 0.0, "unsure": 2.0, "tantalum": 6.0, "wrong": 25.0}


def _look(bgr: np.ndarray, r: tuple[int, int, int, int]) -> str:
    """Body colour of a part box on golden 1: 'tan' (ceramic capacitor),
    'dark' (resistor, diode, transistor, IC, tantalum) or 'other'."""
    x, y, w, h = r
    core = bgr[y + h // 4:y + h - h // 4, x + w // 4:x + w - w // 4]
    if core.size == 0:
        return "other"
    lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    L, a, b = (float(np.median(lab[:, i])) for i in range(3))
    chroma = float(np.hypot(a - 128, b - 128))
    if chroma >= 12 and b > 132:
        return "tan"
    if L < 120 and chroma < 12:
        return "dark"
    return "other"


def _type_fit(name: str, look: str) -> str:
    t, _ = type_of(name)
    if look == "tan":
        return "ok" if t == "capacitor" else ("wrong" if t in ("resistor", "diode", "led", "bridge", "sot23", "ic")
                                               else "unsure")
    if look == "dark":
        if t == "capacitor":
            return "tantalum"
        return "ok" if t in ("resistor", "diode", "led", "bridge", "sot23", "ic") else "unsure"
    return "unsure"


def _geometry(lb: Label, r: tuple[int, int, int, int]) -> tuple[str, float, float]:
    """(side, gap, lateral) of a name against a part box, in the name's
    character heights. Lateral is the smallest of centre, left-edge and
    right-edge misalignment: text is anchored by an edge as often as its centre."""
    x, y, w, h = r
    lx0, ly0, lx1, ly1 = lb.box
    u = max(lb.h, 8.0)
    gx = max(x - lx1, lx0 - (x + w), 0.0)
    gy = max(y - ly1, ly0 - (y + h), 0.0)
    if gy >= gx:
        side = "below" if lb.y > y + h / 2 else "above"
        lat = min(abs(lb.x - (x + w / 2)), abs(lx0 - x), abs(lx1 - (x + w)))
        return side, gy / u, lat / u
    side = "right" if lb.x > x + w / 2 else "left"
    lat = min(abs(lb.y - (y + h / 2)), abs(ly0 - y), abs(ly1 - (y + h)))
    return side, gx / u, lat / u


def _styles(rects, labels) -> list[tuple[str, float, float]]:
    """The board's dominant (side, gap, lateral) offsets: every name votes for
    its nearest part; mean-shift per side; modes with >= STYLE_MIN_VOTES.
    Only the nearest: votes for second and third nearest parts made
    "above, one character to the side" styles that fit a swapped neighbour
    (C2/C3 on the fixture) as well as the true part."""
    votes: dict[str, list[tuple[float, float, float]]] = {}
    for lb in labels:
        geo = sorted((_geometry(lb, r) for r in rects), key=lambda s: np.hypot(s[1], s[2]))[:STYLE_K]
        for k, (side, g, a) in enumerate(geo, 1):
            if np.hypot(g, a) <= REACH:
                votes.setdefault(side, []).append((g, a, 1.0 / k))
    out = []
    for side, v in votes.items():
        P = np.array([(g, a) for g, a, _ in v]); W = np.array([w for *_, w in v])
        modes: list[tuple[np.ndarray, float]] = []
        for p in P:
            m = p.copy()
            for _ in range(30):
                near = np.linalg.norm(P - m, axis=1) < STYLE_BW
                nm = (P[near] * W[near, None]).sum(0) / W[near].sum()
                if np.linalg.norm(nm - m) < 1e-3:
                    break
                m = nm
            if not any(np.linalg.norm(m - q) < STYLE_BW / 2 for q, _ in modes):
                modes.append((m, float(W[np.linalg.norm(P - m, axis=1) < STYLE_BW].sum())))
        out += [(side, float(m[0]), float(m[1])) for m, wsum in modes if wsum >= STYLE_MIN_VOTES]
    return out


def _cost(lb: Label, r, look: str, styles) -> float:
    side, g, a = _geometry(lb, r)
    if np.hypot(g, a) > REACH:
        return np.inf
    fit = [((g - mg) / STYLE_SG) ** 2 + ((a - ma) / STYLE_SA) ** 2 for s, mg, ma in styles if s == side]
    geo = min(fit) if fit else np.inf
    if not np.isfinite(geo) or geo > MISS:
        geo = min(geo, NO_STYLE + (g ** 2 + a ** 2) / 2.0)
    return geo + TYPE_COST[_type_fit(lb.name, look)]


LIST_BONUS = 6.0                # cost taken off a pair matched in order within a list
# A pair the global optimum needs but that fits no style this badly is left
# unnamed. Fixture plus two unseen phone goldens: every right pair cost <= 6.8,
# the wrong ones 9.8-12.1 (and those carried the wrong type, R13 on a capacitor).
ACCEPT = 7.5
UNSURE_MARGIN = 1.0             # a name this close to its next choice is shown as "check"


def _columns(items, centre, lo, hi, tol, step) -> list[list[int]]:
    """Runs of >= 3 items sharing an axis (centres within tol) and evenly
    stacked along the other (gaps under step): a list of names, or a column of
    parts. `centre`, `lo`, `hi` give each item's cross-axis centre and its span
    along the run."""
    order = sorted(range(len(items)), key=lambda i: centre(items[i]))
    groups: list[list[int]] = []
    for i in order:
        for g in groups:
            if abs(centre(items[i]) - np.mean([centre(items[k]) for k in g])) < tol(items[i]):
                g.append(i)
                break
        else:
            groups.append([i])
    runs = []
    for g in groups:
        g.sort(key=lambda i: lo(items[i]))
        cur = [g[0]]
        for i in g[1:]:
            if lo(items[i]) - hi(items[cur[-1]]) < step(items[i]):
                cur.append(i)
            else:
                if len(cur) >= 3:
                    runs.append(cur)
                cur = [i]
        if len(cur) >= 3:
            runs.append(cur)
    return runs


def _list_pairs(rects, labels, looks) -> list[tuple[int, int]]:
    """Names printed as a list beyond the end of a column (or row) of parts,
    in the same order as the parts (the fixture's C7 R12 C8 R13 under its
    block): matched in order, both directions tried, type fit deciding."""
    pairs = []
    for axis in (0, 1):                    # 0: vertical columns, 1: horizontal rows
        a, b = (0, 1) if axis == 0 else (1, 0)
        lab_runs = _columns(labels, lambda l: (l.box[a] + l.box[a + 2]) / 2, lambda l: l.box[b],
                            lambda l: l.box[b + 2], lambda l: 0.6 * l.h, lambda l: 1.2 * l.h)
        part_runs = _columns(rects, lambda r: r[a] + r[a + 2] / 2, lambda r: r[b], lambda r: r[b] + r[b + 2],
                             lambda r: 0.35 * r[a + 2], lambda r: 1.2 * max(r[b + 2], 1))
        for lr in lab_runs:
            L = [labels[j] for j in lr]
            h = float(np.median([l.h for l in L]))
            l_lo, l_hi = min(l.box[b] for l in L), max(l.box[b + 2] for l in L)
            l_c = float(np.mean([(l.box[a] + l.box[a + 2]) / 2 for l in L]))
            best = None
            for pr in part_runs:
                if len(pr) != len(lr):
                    continue
                R = [rects[i] for i in pr]
                p_lo, p_hi = min(r[b] for r in R), max(r[b] + r[b + 2] for r in R)
                p_a0, p_a1 = min(r[a] for r in R), max(r[a] + r[a + 2] for r in R)
                if not (p_a0 - h <= l_c <= p_a1 + h):          # beside the run's own lane
                    continue
                gap = l_lo - p_hi if l_lo >= p_hi else p_lo - l_hi
                if gap < 0 or gap > 10 * h:                    # past its end, not alongside
                    continue
                for order in (pr, pr[::-1]):
                    cost = sum(TYPE_COST[_type_fit(labels[j].name, looks[i])] for i, j in zip(order, lr))
                    if best is None or cost < best[0]:
                        best = (cost, list(zip(order, lr)))
            if best is not None and best[0] < TYPE_COST["wrong"]:
                pairs += best[1]
    return pairs


def assign(rects: list[tuple[int, int, int, int]], labels: list[Label],
           golden_bgr: np.ndarray | None = None, detail: dict | None = None) -> list[str | None]:
    """Each box's name, by one optimal assignment of names to boxes (Hungarian),
    every box and every name free to stay unmatched at cost MISS, then the
    numbering repaired (_repair)."""
    from scipy.optimize import linear_sum_assignment
    labels = [lb for lb in labels if not _on_body(lb, rects)]
    if labels:
        hmed = float(np.median([lb.h for lb in labels]))
        labels = [lb for lb in labels if lb.h >= 0.6 * hmed]      # smaller text is body marking
    names: list[str | None] = [None] * len(rects)
    if not rects or not labels:
        return names
    looks = [_look(golden_bgr, r) if golden_bgr is not None else "other" for r in rects]
    styles = _styles(rects, labels)
    nP, nL = len(rects), len(labels)
    C = np.full((nP + nL, nL + nP), 0.0)
    for i, r in enumerate(rects):
        for j, lb in enumerate(labels):
            c = _cost(lb, r, looks[i], styles)
            C[i, j] = c if np.isfinite(c) else 1e6
    for i, j in _list_pairs(rects, labels, looks):
        C[i, j] = min(C[i, j], TYPE_COST[_type_fit(labels[j].name, looks[i])]) - LIST_BONUS
    C[:nP, nL:] = np.where(np.eye(nP) > 0, MISS, 1e6)             # part unnamed
    C[nP:, :nL] = np.where(np.eye(nL) > 0, MISS, 1e6)             # name unused (unfitted footprint)
    rows, cols = linear_sum_assignment(C)
    got: list[Label | None] = [None] * nP
    margin = [0.0] * nP
    for i, j in zip(rows, cols):
        if i < nP and j < nL and C[i, j] <= ACCEPT:
            got[i] = labels[j]
            # how much worse the next choice is, for this part and for this name
            other_l = min(np.delete(C[i, :nL], j).min(initial=MISS), MISS)
            other_p = min(np.delete(C[:nP, j], i).min(initial=MISS), MISS)
            margin[i] = float(min(other_l, other_p) - C[i, j])
    names = _repair(got, looks, labels)
    if detail is not None:
        detail["margin"] = margin
        detail["cost"] = [float(C[i, cols[list(rows).index(i)]]) if got[i] else None for i in range(nP)]
    return names


def _candidates(lb: Label) -> list[str]:
    """Other readings of a name, most likely first: what else was read at its
    place, a confused letter, a confused digit, a dropped or a lost digit."""
    m = TOKEN.fullmatch(lb.name)
    p, n = m.group(1), m.group(2)
    out = [k for k in sorted(lb.votes, key=lb.votes.get, reverse=True) if k != lb.name]
    if p in CONFUSED:
        out.append(CONFUSED[p] + n)
    for i, d in enumerate(n):
        for e in DIGIT_CONFUSED.get(d, ""):
            out.append(p + n[:i] + e + n[i + 1:])
    if len(n) > 1:
        out.append(p + n[:-1])                          # a stray digit read on the end
    out += [p + n + "1", p + "1" + n]                   # a thin 1 lost
    return [c for c in out if TOKEN.fullmatch(c) and c != lb.name]


def _repair(got: list[Label | None], looks: list[str], labels: list[Label]) -> list[str | None]:
    """Numbering as a check on the reading. A name given to two parts keeps
    the better-read one; the other, and a letter that names nothing else on the
    board, take the first other reading that fills a gap in its series, is not
    in use and fits the body (R2 read "R20" beside R20; R11 read "R1", the
    unfitted footprint's name; F14 for R14; J4 for T4)."""
    names = [lb.name if lb else None for lb in got]
    series: dict[str, set[int]] = {}
    for n in {n for n in names if n}:
        m = TOKEN.fullmatch(n)
        series.setdefault(m.group(1), set()).add(int(m.group(2)))

    def fills_gap(c: str) -> bool:
        m = TOKEN.fullmatch(c)
        s = series.get(m.group(1), set())
        k = int(m.group(2))
        return len(s) >= 2 and k not in s and k <= max(s) + 1

    def conflict(i: int) -> bool:
        """This part's name is also another part's, or another place's."""
        n = names[i]
        return (sum(1 for k in names if k == n) > 1
                or any(o is not got[i] and o.name == n for o in labels))

    # Most telling first: a conflicting name with another reading at its own
    # place (R1 there, and R11 too); then a letter used nowhere else (F14);
    # then a conflicting name read less confidently than its rival. Each is
    # re-checked when its turn comes, so fixing one misread settles the other
    # side of its conflict (a well-read R20 is not "fixed" into R2).
    order = []
    for i, lb in enumerate(got):
        if lb is None:
            continue
        m = TOKEN.fullmatch(lb.name)
        alts = any(k != lb.name for k in lb.votes)
        lonely = len(series.get(m.group(1), ())) == 1 and m.group(1) in CONFUSED
        rivals = [o.score for o in labels if o is not lb and o.name == lb.name]
        rivals += [got[k].score for k, n in enumerate(names) if n == lb.name and k != i]
        if rivals and alts:
            order.append((0, lb.score, i))
        elif lonely:
            order.append((1, lb.score, i))
        elif rivals and max(rivals) > lb.score:
            order.append((2, lb.score, i))
    for prio, _, i in sorted(order):
        if prio != 1 and not conflict(i):
            continue
        used = {n for n in names if n}
        pool = [k for k in sorted(got[i].votes, key=got[i].votes.get, reverse=True) if k != got[i].name] \
            if prio == 0 else _candidates(got[i])
        for c in pool:
            if TOKEN.fullmatch(c) and c not in used and fills_gap(c) \
                    and TYPE_COST[_type_fit(c, looks[i])] < TYPE_COST["wrong"]:
                old = TOKEN.fullmatch(names[i])
                series.get(old.group(1), set()).discard(int(old.group(2)))
                names[i] = c
                m = TOKEN.fullmatch(c)
                series.setdefault(m.group(1), set()).add(int(m.group(2)))
                break
    for i in range(len(names)):                         # a name still on two parts: neither is trusted
        if names[i] and sum(1 for k in names if k == names[i]) > 1:
            best = max((k for k, n in enumerate(names) if n == names[i]), key=lambda k: got[k].score)
            if i != best:
                names[i] = None
    return names


def _on_body(lb: Label, rects) -> bool:
    for x, y, w, h in rects:
        ix, iy = 0.1 * w, 0.1 * h
        if x + ix <= lb.x <= x + w - ix and y + iy <= lb.y <= y + h - iy:
            return True
    return False


def _board_region(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    m = gate.board_mask(bgr)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    k = bgr.shape[1] / m.shape[1]
    x0, x1 = int(xs.min() * k), int((xs.max() + 1) * k)
    y0, y1 = int(ys.min() * k), int((ys.max() + 1) * k)
    pad = int(0.02 * max(bgr.shape[:2]))
    return x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad


def name_boxes(golden_bgr: np.ndarray, rects: list[tuple[int, int, int, int]]) -> Naming:
    """Names for part boxes on golden 1: board-wide read, assignment, a closer
    read round every box still unnamed, and the assignment again."""
    if not available():
        return Naming([None] * len(rects), note="silkscreen names need rapidocr (see requirements.txt)")
    labels = read_labels(golden_bgr, _board_region(golden_bgr))
    detail: dict = {}
    names = assign(rects, labels, golden_bgr, detail)
    close = rects if LOCAL_ALL else [r for r, n in zip(rects, names) if n is None]
    if close:
        # labels sit within a few character heights of their part: a close-up
        # round a big part (U1) need not be the size of the part again
        hmed = float(np.median([lb.h for lb in labels])) if labels else 100.0
        regions = []
        for x, y, w, h in close:
            m = int(min(LOCAL_MARGIN * max(w, h), LOCAL_MAX_MARGIN * hmed))
            regions.append((x - m, y - m, w + 2 * m, h + 2 * m))
        extra = [lb for found in read_many(golden_bgr, regions, LOCAL_SCALE, readers=LOCAL_READERS)
                 for lb in found]
        labels = add_local(labels, merge(extra))
        names = assign(rects, labels, golden_bgr, detail)
    return Naming(names, labels, margin=detail.get("margin", []), cost=detail.get("cost", []))


def add_local(board: list[Label], local: list[Label]) -> list[Label]:
    """Close-up readings added to the board-wide ones without out-voting them:
    overlapping close-ups read the same text several times, so they only name a
    place the board read found nothing at, or extend a name it cut short
    (R1 -> R11: a thin 1 lost)."""
    out = list(board)
    for lb in local:
        near = [k for k, o in enumerate(out) if np.hypot(lb.x - o.x, lb.y - o.y) < 0.8 * max(lb.h, o.h)]
        if not near:
            out.append(lb)
            continue
        k = near[0]
        o = out[k]
        if len(lb.name) > len(o.name) and lb.name.startswith(o.name[:len(o.name)]) and lb.name[:len(o.name)] == o.name:
            votes = dict(o.votes)
            votes[lb.name] = votes.get(lb.name, 0.0) + lb.score
            out[k] = Label(lb.name, o.x, o.y, o.h, votes[lb.name], o.box, votes)
    return out


def named_parts(parts: list[Part], names: list[str | None]) -> list[Part]:
    """Parts renamed, typed and marked polarized from their names; a part with
    no name keeps its own. A name used twice keeps it on the first only."""
    seen: set[str] = set()
    out = []
    for p, n in zip(parts, names):
        if n and n not in seen:
            seen.add(n)
            t, pol = type_of(n)
            out.append(Part(n, p.x, p.y, p.w, p.h, type=t, package=p.package, polarized=pol))
        else:
            out.append(p)
    return out
