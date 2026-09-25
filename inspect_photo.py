"""Missing and misplaced parts, checked on ordinary photographs.

Standalone from the machine pipeline on purpose. `cli.py` inspects channel
stacks at a fixed 12 um/px, registered from fiducials and driven by a centroid
file -- none of which exists for a phone photo of a board on a desk. This asks a
narrower question with nothing but pictures: compared with a board known to be
good, is each part still there, and is it where it was?

    photos/golden/   one or more photos of a known-good board
    photos/test/     photos of the boards to check
    photos/parts.csv where each part is, drawn once on one golden photo

    python inspect_photo.py mark photos/golden/good1.jpg
    python inspect_photo.py check photos/test/

How it works, per test photo:

  1. Register the whole photo onto the reference golden photo (SIFT features,
     RANSAC homography). The photos do not have to be taken from exactly the
     same place; a hand-held reshoot is fine. A part that moved does not drag
     the registration with it, because RANSAC treats it as an outlier among
     hundreds of features on the rest of the board.
  2. For each marked part, search a window around where it should be for the
     golden appearance of that part, over position and rotation.
  3. Weak best match -> missing. Strong match, but displaced or turned past
     tolerance -> offset / skew. Same order as `aoi/fusion.py`: a part that is
     not there has no position to measure.

What it does not do: solder joints, polarity, wrong value. Those need the
lighting rings; this needs only a picture.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aoi import capture, config
from aoi.photo.register import (REG_MAX_SIDE, REG_MIN_INLIERS, REG_MIN_OVERLAP_NCC,  # noqa: F401
                                Registration, register, warp_to_ref, _overlap_ncc, _small)

OUT = Path("out")
PARTS = Path("photos/parts.csv")
GOLDEN = Path("photos/golden")


# How far to look for a part, as a fraction of its own longest side, on top of
# the template padding. A part that slid further than this is not "offset", it
# is somewhere else, and reads as missing -- which is also true of it.
SEARCH_FRACTION = 0.6
# Border of board kept around each marked box in the template. The outline of
# the body against the board is what carries position; a template cut exactly
# on the body edge has almost none.
PAD_FRACTION = 0.15
PAD_MIN_PX = 4
BLUR_FRACTION = 0.06
ANGLE_RANGE_DEG = 30.0
ANGLE_STEP_DEG = 3.0

# With a single golden photo there is no measured spread of good-part scores, so
# presence falls back to an absolute level. A good part between two photos of
# the same board correlates at ~0.85-0.95; a bare footprint keeps its lands and
# solder, which is why this is not lower.
DEFAULT_MISSING_NCC = 0.60
# Without a pixel scale, placement tolerance is a fraction of the part's own
# short side. 0.25 of an 0603 body (0.8 mm) is 200 um, close to the machine's
# 150 um in config.MAX_OFFSET_UM.
DEFAULT_OFFSET_FRACTION = 0.25

COLOURS = {"ok": (90, 190, 90), "missing": (60, 60, 235),
           "offset": (40, 170, 240), "skew": (40, 210, 240),
           "not_in_view": (150, 150, 150)}


# ------------------------------------------------------------------ parts file
@dataclass
class Part:
    name: str
    x: int
    y: int
    w: int
    h: int

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


def save_parts(path: Path, reference: Path, parts: list[Part]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(f"# reference: {reference}\n")
        w = csv.writer(fh)
        w.writerow(["name", "x", "y", "w", "h"])
        for p in parts:
            w.writerow([p.name, p.x, p.y, p.w, p.h])


def load_parts(path: Path) -> tuple[Path, list[Part]]:
    """Read the parts file. The first line names the photo the boxes were drawn
    on; every coordinate is in that photo's pixels, so it is the one every test
    photo gets registered onto."""
    if not path.exists():
        sys.exit(f"{path} not found -- run `python inspect_photo.py mark <golden photo>` first")
    lines = path.read_text().splitlines()
    ref = None
    body = []
    for ln in lines:
        if ln.startswith("#"):
            if ln[1:].strip().lower().startswith("reference:"):
                ref = Path(ln.split(":", 1)[1].strip())
            continue
        if ln.strip():
            body.append(ln)
    if ref is None:
        sys.exit(f"{path}: no '# reference: <photo>' line, so the boxes have no image to belong to")
    parts = []
    for row in csv.DictReader(body):
        parts.append(Part(row["name"].strip(), int(float(row["x"])), int(float(row["y"])),
                          int(float(row["w"])), int(float(row["h"]))))
    if not parts:
        sys.exit(f"{path}: no parts listed")
    return ref, parts


# ---------------------------------------------------------------------- images
def load_grey(path: Path) -> np.ndarray:
    """Greyscale uint8. Display-referred is fine here -- correlation does not
    care about a monotonic tone curve the way the slope measurement does."""
    a = capture.load_image(path, gamma=None)
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def image_files(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if not p.is_dir():
        sys.exit(f"{p}: no such file or folder")
    exts = capture.RAW_EXT | capture.LDR_EXT
    return sorted(f for f in p.iterdir() if f.suffix.lower() in exts)


# ----------------------------------------------------------------- measurement
@dataclass
class Match:
    ncc: float
    dx_px: float
    dy_px: float
    angle_deg: float         # CCW as displayed


def _template(ref: np.ndarray, p: Part) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Golden crop of the part with a border of board, the mask of the body
    inside it, and the body centre in template pixels."""
    pad = max(PAD_MIN_PX, int(round(PAD_FRACTION * min(p.w, p.h))))
    H, W = ref.shape
    x0, y0 = max(0, p.x - pad), max(0, p.y - pad)
    x1, y1 = min(W, p.x + p.w + pad), min(H, p.y + p.h + pad)
    tpl = ref[y0:y1, x0:x1]
    mask = np.zeros(tpl.shape, np.float32)
    mask[p.y - y0:p.y - y0 + p.h, p.x - x0:p.x - x0 + p.w] = 1.0
    return tpl, mask, (p.centre[0] - x0, p.centre[1] - y0)


def _subpixel(res: np.ndarray, loc: tuple[int, int]) -> tuple[float, float]:
    """Parabolic refinement of a correlation peak."""
    x, y = loc
    fx = fy = 0.0
    if 0 < x < res.shape[1] - 1:
        l, c, r = res[y, x - 1], res[y, x], res[y, x + 1]
        d = l - 2 * c + r
        fx = 0.5 * (l - r) / d if abs(d) > 1e-9 else 0.0
    if 0 < y < res.shape[0] - 1:
        u, c, b = res[y - 1, x], res[y, x], res[y + 1, x]
        d = u - 2 * c + b
        fy = 0.5 * (u - b) / d if abs(d) > 1e-9 else 0.0
    return x + float(np.clip(fx, -1, 1)), y + float(np.clip(fy, -1, 1))


def _match_levels(win: np.ndarray, ref_win: np.ndarray) -> np.ndarray:
    """Map the test window's brightness onto the golden one's.

    Two photos are never exposed identically, and the placement match below
    compares brightness directly, so the test window is first given the golden
    window's levels. Percentiles rather than a least-squares fit: most of the
    window is unchanged board, and a missing or moved part must not drag the
    mapping toward itself."""
    t_lo, t_mid, t_hi = np.percentile(win, (5, 50, 95))
    r_lo, r_mid, r_hi = np.percentile(ref_win, (5, 50, 95))
    g = (r_hi - r_lo) / max(t_hi - t_lo, 1e-3)
    return ((win.astype(np.float32) - t_mid) * g + r_mid).astype(np.float32)


def find_part(ref: np.ndarray, img: np.ndarray, p: Part) -> Match:
    """Presence, position and rotation of part `p` in `img`, which is already
    registered to `ref`.

    Two different matches, because the two questions want different pixels.

    Presence correlates the whole template, lands and board border included. A
    bare footprint loses the body but keeps its lands, so the score drops by
    about half; a part that is merely turned or shifted still overlaps its own
    template well enough to stay clear of that.

    Position and rotation compare the body alone (the argument of
    `rules.body_mask_for`). The lands are fixed to the board, so any comparison
    that includes them argues for zero offset and zero degrees whatever the part
    did. And it compares brightness directly rather than correlating: a plain
    body on a plain board has almost no pattern to correlate, but it is a
    different grey from the board and from the lands, so a body pushed half off
    its footprint shows up as a large difference.
    """
    tpl, mask, (bx, by) = _template(ref, p)
    th, tw = tpl.shape
    m = int(round(SEARCH_FRACTION * max(p.w, p.h))) + max(p.w, p.h) // 4
    H, W = img.shape
    x0, y0 = max(0, p.x - m), max(0, p.y - m)
    x1, y1 = min(W, p.x + p.w + m), min(H, p.y + p.h + m)
    win = img[y0:y1, x0:x1]
    if win.shape[0] <= th or win.shape[1] <= tw:
        return Match(0.0, 0.0, 0.0, 0.0)

    ncc = float(cv2.minMaxLoc(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED))[1])

    # Smooth at a scale set by the part, so sensor grain on a plain body does
    # not decide where the minimum falls.
    sig = max(1.0, BLUR_FRACTION * min(p.w, p.h))
    t_f = cv2.GaussianBlur(tpl.astype(np.float32), (0, 0), sig)
    w_f = cv2.GaussianBlur(_match_levels(win, ref[y0:y1, x0:x1]), (0, 0), sig)

    def body(a: float):
        R = cv2.getRotationMatrix2D((bx, by), a, 1.0)
        t = cv2.warpAffine(t_f, R, (tw, th), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT101)
        mk = cv2.warpAffine(mask, R, (tw, th), flags=cv2.INTER_NEAREST)
        res = cv2.matchTemplate(w_f, t, cv2.TM_SQDIFF, mask=mk) / max(float(mk.sum()), 1.0)
        res = np.nan_to_num(res, nan=np.inf, posinf=np.inf)
        v, _, loc, _ = cv2.minMaxLoc(res)
        return float(v), res, loc, t

    coarse = np.arange(-ANGLE_RANGE_DEG, ANGLE_RANGE_DEG + 1e-6, ANGLE_STEP_DEG)
    best_a = min(coarse, key=lambda a: body(a)[0])
    fine = np.arange(best_a - ANGLE_STEP_DEG, best_a + ANGLE_STEP_DEG + 1e-6, 1.0)
    best_a = float(min(fine, key=lambda a: body(a)[0]))
    _, res, loc, t_rot = body(best_a)
    px, py = _subpixel(-res, loc)
    ex, ey = p.centre[0] - x0, p.centre[1] - y0
    # A part that turned also scores better against the turned full template.
    if abs(best_a) >= ANGLE_STEP_DEG:
        t8 = np.clip(t_rot, 0, 255).astype(np.uint8)
        ncc = max(ncc, float(cv2.minMaxLoc(
            cv2.matchTemplate(win, t8, cv2.TM_CCOEFF_NORMED))[1]))
    # getRotationMatrix2D turns counter-clockwise as displayed for a positive
    # angle, and the template had to turn by the part's own rotation to fit it.
    return Match(ncc, float(px + bx - ex), float(py + by - ey), best_a)


@dataclass
class Verdict:
    part: Part
    reason: str
    match: Match
    detail: str

    @property
    def failed(self) -> bool:
        return self.reason not in ("ok", "not_in_view")

    @property
    def checked(self) -> bool:
        return self.reason != "not_in_view"


def in_view(cover: np.ndarray, p: Part) -> bool:
    """Whether the test photo covers the part and a margin round it -- the
    region the search reads from. A part the camera did not see is not
    checked, and is not a defect either."""
    m = int(round(SEARCH_FRACTION * max(p.w, p.h)))
    y0, x0 = p.y - m, p.x - m
    y1, x1 = p.y + p.h + m, p.x + p.w + m
    if y0 < 0 or x0 < 0 or y1 > cover.shape[0] or x1 > cover.shape[1]:
        return False
    return bool(cover[y0:y1, x0:x1].min() > 0)


def judge(p: Part, m: Match, missing_ncc: float, um_per_px: float | None,
          max_offset_um: float, offset_fraction: float, max_skew: float) -> Verdict:
    if m.ncc < missing_ncc:
        return Verdict(p, "missing", m, f"best match {m.ncc:.2f} < {missing_ncc:.2f}")
    off_px = float(np.hypot(m.dx_px, m.dy_px))
    if um_per_px:
        off, lim, unit = off_px * um_per_px, max_offset_um, "um"
    else:
        off, lim, unit = off_px, offset_fraction * min(p.w, p.h), "px"
    if off > lim:
        return Verdict(p, "offset", m, f"{off:.0f} {unit} off (limit {lim:.0f})")
    if abs(m.angle_deg) > max_skew:
        return Verdict(p, "skew", m, f"{m.angle_deg:+.1f} deg (limit {max_skew:.0f})")
    return Verdict(p, "ok", m, f"{off:.0f} {unit}, {m.angle_deg:+.1f} deg")


# --------------------------------------------------------------------- reports
def _rotated_box(p: Part, m: Match) -> np.ndarray:
    cx, cy = p.centre[0] + m.dx_px, p.centre[1] + m.dy_px
    # OpenCV's RotatedRect angle runs clockwise as displayed.
    return cv2.boxPoints(((cx, cy), (p.w, p.h), -m.angle_deg)).astype(np.int32)


def draw(img: np.ndarray, verdicts: list[Verdict], path: Path) -> None:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = max(1, int(round(max(img.shape) / 1500)))
    fs = 0.4 * t + 0.1
    for v in verdicts:
        p, col = v.part, COLOURS[v.reason]
        # Where it should be, thin; where it was found, in the verdict colour.
        cv2.rectangle(out, (p.x, p.y), (p.x + p.w, p.y + p.h), (200, 200, 200), 1)
        if v.reason == "not_in_view":
            pass
        elif v.reason != "missing":
            cv2.polylines(out, [_rotated_box(p, v.match)], True, col, 2 * t if v.failed else t)
        else:
            cv2.line(out, (p.x, p.y), (p.x + p.w, p.y + p.h), col, 2 * t)
            cv2.line(out, (p.x + p.w, p.y), (p.x, p.y + p.h), col, 2 * t)
        label = p.name if not v.failed else f"{p.name} {v.reason}"
        cv2.putText(out, label, (p.x, max(12, p.y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, col, t, cv2.LINE_AA)
    cv2.imwrite(str(path), out)


def montage(ref: np.ndarray, img: np.ndarray, verdicts: list[Verdict], path: Path) -> bool:
    """Golden | test, side by side, for every flagged part -- what the operator
    needs to agree or disagree with a call without opening the full photo."""
    rows = []
    for v in (v for v in verdicts if v.failed):
        p = v.part
        m = int(round(SEARCH_FRACTION * max(p.w, p.h)))
        H, W = ref.shape
        x0, y0 = max(0, p.x - m), max(0, p.y - m)
        x1, y1 = min(W, p.x + p.w + m), min(H, p.y + p.h + m)
        a, b = ref[y0:y1, x0:x1], img[y0:y1, x0:x1]
        s = 160.0 / max(a.shape)
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        row = cv2.cvtColor(np.hstack([a, np.full((a.shape[0], 6), 255, np.uint8), b]),
                           cv2.COLOR_GRAY2BGR)
        bar = np.full((22, row.shape[1], 3), 28, np.uint8)
        cv2.putText(bar, f"{p.name}: {v.reason}  (golden | test)", (6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS[v.reason], 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, row]))
    if not rows:
        return False
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 6, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(255, 255, 255)) for r in rows]
    cv2.imwrite(str(path), np.vstack(rows))
    return True


# -------------------------------------------------------------------- commands
def cmd_mark(a: argparse.Namespace) -> None:
    ref_path = Path(a.image)
    img = load_grey(ref_path)
    s = min(1.0, a.display / max(img.shape))
    show = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    print("Draw a box around each part's BODY, press SPACE or ENTER after each one,")
    print("and ESC when done. Tight on the body is best; a little board margin is added.")
    try:
        boxes = cv2.selectROIs("mark parts -- ESC when done", show, showCrosshair=False)
        cv2.destroyAllWindows()
    except cv2.error:
        sys.exit("This OpenCV build has no GUI, so boxes cannot be drawn here.\n"
                 f"Write {a.parts} by hand instead: a '# reference: {ref_path}' line, "
                 "then name,x,y,w,h per part, in pixels of that photo.")
    parts = [Part(f"P{i + 1}", *(int(round(v / s)) for v in b))
             for i, b in enumerate(np.asarray(boxes).reshape(-1, 4)) if b[2] > 0 and b[3] > 0]
    if not parts:
        sys.exit("no boxes drawn")
    parts_path = Path(a.parts)
    save_parts(parts_path, ref_path, parts)

    OUT.mkdir(exist_ok=True)
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = max(1, int(round(max(img.shape) / 1500)))
    for p in parts:
        cv2.rectangle(out, (p.x, p.y), (p.x + p.w, p.y + p.h), (90, 190, 90), t)
        cv2.putText(out, p.name, (p.x, max(12, p.y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * t + 0.1, (90, 190, 90), t, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "parts_marked.png"), out)
    print(f"\n{len(parts)} parts -> {parts_path}")
    print(f"labelled image -> {OUT / 'parts_marked.png'}")
    print(f"Rename P1, P2, ... to refdes in {parts_path} if you like; order does not matter.")


def cmd_check(a: argparse.Namespace) -> None:
    ref_path, parts = load_parts(Path(a.parts))
    if not ref_path.exists():
        sys.exit(f"reference photo {ref_path} (named in {a.parts}) not found")
    ref = load_grey(ref_path)
    OUT.mkdir(exist_ok=True)

    # Extra golden photos turn presence from a guess into a measurement: how well
    # does a GOOD part of this board match its own template, photo to photo?
    # Same rule as the machine (config.NCC_ABSENT_FRACTION of the golden mean).
    golden_dir = Path(a.golden)
    extra = [f for f in image_files(golden_dir) if f.resolve() != ref_path.resolve()] \
        if golden_dir.is_dir() else []
    thr = {p.name: a.missing_ncc for p in parts}
    if extra and not a.missing_ncc_set:
        print(f"learning presence thresholds from {len(extra)} extra golden photo(s)")
        scores: dict[str, list[float]] = {p.name: [] for p in parts}
        for f in extra:
            g = load_grey(f)
            reg = register(ref, g)
            if not reg.ok:
                print(f"  {f.name}: skipped, {reg.note}")
                continue
            gw, gcov = warp_to_ref(g, reg.H, ref.shape)
            for p in parts:
                if in_view(gcov, p):
                    scores[p.name].append(find_part(ref, gw, p).ncc)
        for p in parts:
            if scores[p.name]:
                thr[p.name] = config.NCC_ABSENT_FRACTION * float(np.mean(scores[p.name]))
        worst = min(parts, key=lambda p: min(scores[p.name] or [1.0]))
        if scores[worst.name]:
            print(f"  weakest good-part match: {worst.name} at "
                  f"{min(scores[worst.name]):.2f}, missing below {thr[worst.name]:.2f}")
    else:
        print(f"presence threshold {a.missing_ncc:.2f} (fixed; add more photos to "
              f"{golden_dir}/ to learn it per part)")

    tests = [f for f in image_files(Path(a.test)) if f.resolve() != ref_path.resolve()]
    if not tests:
        sys.exit(f"no photos in {a.test}")
    scale = f"{a.um_per_px} um/px" if a.um_per_px else \
        f"no pixel scale, offset limit {a.offset_fraction:.0%} of each part's short side"
    print(f"{len(parts)} parts, reference {ref_path.name}, {scale}")

    any_fail = False
    for f in tests:
        print(f"\n{f.name}")
        img = load_grey(f)
        reg = register(ref, img)
        if not reg.ok:
            print(f"  NOT CHECKED: could not align to the reference -- {reg.note}")
            any_fail = True
            continue
        warped, cover = warp_to_ref(img, reg.H, ref.shape)
        print(f"  aligned on {reg.inliers} of {reg.matches} feature matches, "
              f"overlap correlation {reg.fit:.2f}")

        verdicts = [judge(p, find_part(ref, warped, p), thr[p.name], a.um_per_px,
                          a.max_offset_um, a.offset_fraction, a.max_skew)
                    if in_view(cover, p) else
                    Verdict(p, "not_in_view", Match(0.0, 0.0, 0.0, 0.0),
                            "outside this photo, not checked")
                    for p in parts]
        failed = [v for v in verdicts if v.failed]
        unseen = [v for v in verdicts if not v.checked]
        any_fail |= bool(failed)
        print(f"  {len(parts)} parts, {len(failed)} flagged"
              + (f", {len(unseen)} NOT CHECKED (outside the photo: "
                 f"{', '.join(v.part.name for v in unseen)})" if unseen else ""))
        print(f"  {'part':10}{'result':10}{'match':>7}{'dx':>8}{'dy':>8}{'rot':>7}  detail")
        for v in sorted(verdicts, key=lambda v: (not v.failed, v.part.name)):
            if v.failed or a.verbose:
                m = v.match
                print(f"  {v.part.name:10}{v.reason:10}{m.ncc:7.2f}{m.dx_px:8.1f}"
                      f"{m.dy_px:8.1f}{m.angle_deg:7.1f}  {v.detail}")

        stem = f.stem
        draw(warped, verdicts, OUT / f"inspect_{stem}.png")
        wrote = [OUT / f"inspect_{stem}.png"]
        if montage(ref, warped, verdicts, OUT / f"inspect_{stem}_flagged.png"):
            wrote.append(OUT / f"inspect_{stem}_flagged.png")
        with open(OUT / f"inspect_{stem}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["part", "result", "match", "dx_px", "dy_px", "rot_deg", "detail"])
            for v in verdicts:
                m = v.match
                w.writerow([v.part.name, v.reason, f"{m.ncc:.3f}", f"{m.dx_px:.2f}",
                            f"{m.dy_px:.2f}", f"{m.angle_deg:.2f}", v.detail])
        wrote.append(OUT / f"inspect_{stem}.csv")
        print("  wrote " + ", ".join(str(p) for p in wrote))
    sys.exit(1 if any_fail else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("mark", help="draw a box round each part on one golden photo")
    mk.add_argument("image")
    mk.add_argument("--parts", default=str(PARTS))
    mk.add_argument("--display", type=int, default=1400,
                    help="longest side of the marking window, px")
    mk.set_defaults(fn=cmd_mark)

    ck = sub.add_parser("check", help="check test photos against the golden board")
    ck.add_argument("test", help="a photo, or a folder of them")
    ck.add_argument("--parts", default=str(PARTS))
    ck.add_argument("--golden", default=str(GOLDEN),
                    help="folder of good-board photos; the extras teach presence thresholds")
    ck.add_argument("--missing-ncc", type=float, default=None,
                    help=f"fixed presence threshold (default {DEFAULT_MISSING_NCC}, "
                         f"or learned when there are extra golden photos)")
    ck.add_argument("--um-per-px", type=float, default=None,
                    help="pixel scale, e.g. from `cli.py calibrate`; enables limits in um")
    ck.add_argument("--max-offset-um", type=float, default=config.MAX_OFFSET_UM)
    ck.add_argument("--offset-fraction", type=float, default=DEFAULT_OFFSET_FRACTION)
    ck.add_argument("--max-skew", type=float, default=config.MAX_SKEW_DEG)
    ck.add_argument("-v", "--verbose", action="store_true", help="list passing parts too")
    ck.set_defaults(fn=cmd_check)

    a = ap.parse_args()
    if a.cmd == "check":
        a.missing_ncc_set = a.missing_ncc is not None
        if a.missing_ncc is None:
            a.missing_ncc = DEFAULT_MISSING_NCC
    a.fn(a)


if __name__ == "__main__":
    main()
