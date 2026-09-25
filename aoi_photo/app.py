"""Photo AOI desktop app: teach a board once, inspect many, answer the calls.

Native Tk, no network, no browser (spec section 5). The app runs on a line PC
that may have no network at all, and an operator clearing calls cannot wait on
a page load. Tk is in the standard library, so nothing new is installed.

Four tabs, in the order a new board goes through them:

    Program   golden photos, part boxes, scale -> teach.build_program
    Inspect   one photo per board -> inspect, record, report, per board
    Review    the calls that need an answer, keyboard first (D F S R N U)
    Stats     false calls per part group, from the operator's answers

Two rules shape the code:

* The window never waits on the library. Aligning a 12 MP photo takes seconds,
  so every library call that touches a photo runs on a worker thread and
  reports back through a queue that the Tk loop polls with after(). A worker
  never touches Tk: Tk is not thread-safe, and the failure is a hang, not an
  exception.
* Everything that can be decided without a display is a plain function or
  class (View, BoxList, the call list, the stats rows), so the tests run on a
  build machine with no screen. The Tk classes at the bottom only wire them up.
"""
from __future__ import annotations

import math
import os
import queue
import sqlite3
import sys
import threading
import time
import tkinter as tk
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import cv2
import numpy as np

from aoi import config
from aoi.photo import gate, io, judge, program, record, report, teach
from aoi.photo import inspect as photo_inspect
from aoi.photo.model import MARKED_ENDS, PART_TYPES, UNVALIDATED, BoardResult, Call, Part
from aoi.review import BG, DEFECT, DIM, FALSE, INK, PANEL, RULE

AMBER = "#e0a030"            # REVIEW: between the defect red and the pass green
SELECTED = "#34507a"
POLL_MS = 50
SMALL_SIDE = 2048            # long side of the copy drawn when zoomed out
MAX_ZOOM = 8.0               # canvas pixels per image pixel
CROP_SIDE = 220              # long side of each golden | test crop in Review
ALIGNED_KEEP = 6             # aligned test photos kept for crops (about 36 MB each)
AUTOCLOSE_ENV = "AOI_PHOTO_APP_AUTOCLOSE_MS"
LONG_WORK = ("teach", "inspect", "retrain")    # quitting mid-way loses work; ask first
EDITOR_EMPTY = ("Step 1: click \"Upload golden photos...\" on the left and pick at least 3 photos\n"
                "of a known-good board (JPEG, PNG or HEIC). Golden 1 appears here for boxing parts.")

GOLDEN_RULE = (f"at least {config.PHOTO_MIN_GOLDENS} golden photos: shoot the good board, "
               "pick it up, put it back, shoot again")
FCR_NOTE = ("False-call rate = FAIL calls marked F / parts inspected "
            "(spec section 8 target: under 0.2% per part)")
PHOTO_TYPES = [("Photos", "*.jpg *.jpeg *.png *.heic *.JPG *.JPEG *.PNG *.HEIC"), ("All files", "*")]
CALIBRATED_CHECKS = ("missing", "offset", "skew", "polarity", "tombstone", "wrong_part")

VERDICT_COLOURS = {"PASS": FALSE, "FAIL": DEFECT, "REVIEW": AMBER}
ANSWER_TEXT = {"D": "defect", "F": "false call", "S": "skipped", "R": "reshoot"}
ANSWER_COLOURS = {"D": DEFECT, "F": FALSE, "S": DIM, "R": AMBER}


def verdict_colour(verdict: str) -> str:
    """PASS green, FAIL red, REVIEW amber; anything else (SUPERSEDED, ERROR) dim."""
    return VERDICT_COLOURS.get(verdict, DIM)


# ================================================================== the view
@dataclass
class View:
    """Where a large image sits on a canvas: canvas = (image - origin) * scale.

    Every stored coordinate is a full-resolution golden-1 pixel. The canvas only
    ever shows a resampled window of the image, and this is the one place the
    two coordinate systems meet.
    """
    scale: float = 1.0           # canvas pixels per image pixel
    ox: float = 0.0              # image x at the canvas's left edge
    oy: float = 0.0              # image y at the canvas's top edge
    min_scale: float = 0.01
    max_scale: float = MAX_ZOOM

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.ox) * self.scale, (y - self.oy) * self.scale

    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return cx / self.scale + self.ox, cy / self.scale + self.oy

    def _centre_on(self, x: float, y: float, scale: float, cw: float, ch: float) -> None:
        self.scale = scale
        self.ox = x - cw / (2.0 * scale)
        self.oy = y - ch / (2.0 * scale)

    def fit(self, iw: int, ih: int, cw: int, ch: int) -> None:
        """Whole image in the canvas, centred. Zoom out stops at half of this."""
        if min(iw, ih, cw, ch) < 1:
            return
        s = min(cw / iw, ch / ih)
        self.min_scale = s / 2.0
        self.max_scale = max(MAX_ZOOM, s)
        self._centre_on(iw / 2.0, ih / 2.0, s, cw, ch)

    def zoom_at(self, factor: float, cx: float, cy: float) -> None:
        """Zoom keeping the image point under (cx, cy) where it is."""
        ix, iy = self.to_image(cx, cy)
        self.scale = min(self.max_scale, max(self.min_scale, self.scale * factor))
        self.ox = ix - cx / self.scale
        self.oy = iy - cy / self.scale

    def pan(self, dcx: float, dcy: float) -> None:
        self.ox -= dcx / self.scale
        self.oy -= dcy / self.scale

    def show_box(self, x: float, y: float, w: float, h: float, cw: int, ch: int,
                 margin: float = 0.8) -> None:
        """One part filling the canvas with a margin around it (a fraction of
        its long side, as report.crop_pair), so an offset part is still in view."""
        if min(cw, ch) < 1:
            return
        m = margin * max(w, h, 1.0)
        s = min(cw / (w + 2 * m), ch / (h + 2 * m))
        self._centre_on(x + w / 2.0, y + h / 2.0, min(self.max_scale, max(self.min_scale, s)), cw, ch)

    def visible(self, iw: int, ih: int, cw: int, ch: int) -> tuple[int, int, int, int] | None:
        """The image window (x0, y0, x1, y1) that lands on the canvas; None if none does."""
        x0, y0 = max(0, math.floor(self.ox)), max(0, math.floor(self.oy))
        x1 = min(iw, math.ceil(self.ox + cw / self.scale))
        y1 = min(ih, math.ceil(self.oy + ch / self.scale))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1


def downscale(bgr: np.ndarray, side: int = SMALL_SIDE) -> tuple[np.ndarray, float]:
    """A copy no longer than `side`, and its scale; the image itself if already small."""
    k = min(1.0, side / max(bgr.shape[:2]))
    if k >= 1.0:
        return bgr, 1.0
    return cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), k


def render_patch(full: np.ndarray, small: np.ndarray, k: float, view: View,
                 cw: int, ch: int) -> tuple[np.ndarray, float, float] | None:
    """The visible window of `full` resampled to canvas pixels (RGB), and the
    canvas point its top-left goes at.

    Zoomed out, the window is cut from `small`, so a 12 MP photo is never
    resampled whole on every pan. Zoomed in past 2x it is drawn with nearest
    neighbour, so an operator sees the pixels the measurement saw.
    """
    H, W = full.shape[:2]
    vis = view.visible(W, H, cw, ch)
    if vis is None:
        return None
    x0, y0, x1, y1 = vis
    out_w = max(1, int(round((x1 - x0) * view.scale)))
    out_h = max(1, int(round((y1 - y0) * view.scale)))
    if view.scale <= k and small is not full:
        sx0, sy0 = int(x0 * k), int(y0 * k)
        sx1 = max(sx0 + 1, min(small.shape[1], math.ceil(x1 * k)))
        sy1 = max(sy0 + 1, min(small.shape[0], math.ceil(y1 * k)))
        patch = small[sy0:sy1, sx0:sx1]
    else:
        patch = full[y0:y1, x0:x1]
    if out_w >= 2 * patch.shape[1]:
        interp = cv2.INTER_NEAREST
    elif out_w > patch.shape[1]:
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_AREA
    rgb = cv2.cvtColor(cv2.resize(patch, (out_w, out_h), interpolation=interp), cv2.COLOR_BGR2RGB)
    cx, cy = view.to_canvas(x0, y0)
    return rgb, cx, cy


def fit_side(img: np.ndarray, side: int = CROP_SIDE) -> np.ndarray:
    """Scale to `side` on the long edge; up by nearest neighbour, so a small
    crop shows its real pixels rather than a blur of them."""
    s = side / max(img.shape[:2])
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
    w, h = max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))
    return cv2.resize(img, (w, h), interpolation=interp)


# ================================================================ part boxes
@dataclass
class Box:
    """One part box on golden 1, full-resolution pixels. Unnamed while refdes is ''."""
    x: int
    y: int
    w: int
    h: int
    refdes: str = ""
    type: str = "other"
    package: str = ""
    polarized: bool = False
    marked_end: str = "none"
    tag: int = 0                 # P-number, shown while the box has no refdes

    @property
    def candidate(self) -> bool:
        """Not taught until somebody names it: proposals are guesses."""
        return not self.refdes

    @property
    def label(self) -> str:
        return self.refdes or f"P{self.tag}"

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def part(self) -> Part:
        return Part(self.refdes, self.x, self.y, self.w, self.h, self.type, self.package,
                    self.polarized, self.marked_end)


class BoxList:
    """The parts being taught: add, select, rename, delete, and turn into Parts.

    Nothing reaches build_program without a refdes. A proposal is the
    machine's guess at a body; the operator names what is a part and deletes
    what is not, and an unnamed box is refused rather than taught as "P7".
    """

    def __init__(self) -> None:
        self.boxes: list[Box] = []
        self.selected: int | None = None
        self._tag = 0

    def __len__(self) -> int:
        return len(self.boxes)

    @property
    def unnamed(self) -> int:
        return sum(1 for b in self.boxes if b.candidate)

    def add(self, x: int, y: int, w: int, h: int, refdes: str = "", **fields) -> int:
        self._tag += 1
        self.boxes.append(Box(int(x), int(y), int(w), int(h), refdes.strip(), tag=self._tag))
        i = len(self.boxes) - 1
        try:
            self.set_fields(i, **fields)
        except ValueError:
            del self.boxes[i]
            raise
        return i

    def add_proposals(self, rects, overlap: float = 0.5) -> tuple[int, int]:
        """(added, skipped): a proposal on top of an existing box is skipped,
        so proposing twice does not stack duplicates."""
        added = skipped = 0
        for r in rects:
            r = tuple(int(v) for v in r)
            if any(teach.iou(r, b.rect) > overlap for b in self.boxes):
                skipped += 1
                continue
            self.add(*r)
            added += 1
        return added, skipped

    def hit(self, x: float, y: float) -> int | None:
        """The smallest box under an image point, so a part inside a bigger one stays clickable."""
        inside = [i for i, b in enumerate(self.boxes) if b.contains(x, y)]
        if not inside:
            return None
        return min(inside, key=lambda i: self.boxes[i].w * self.boxes[i].h)

    def select(self, i: int | None) -> None:
        if i is not None and not 0 <= i < len(self.boxes):
            raise IndexError(i)
        self.selected = i

    def delete(self, i: int | None = None) -> bool:
        i = self.selected if i is None else i
        if i is None or not 0 <= i < len(self.boxes):
            return False
        del self.boxes[i]
        if self.selected == i:
            self.selected = None
        elif self.selected is not None and self.selected > i:
            self.selected -= 1
        return True

    def rename(self, i: int, refdes: str) -> None:
        self.boxes[i].refdes = refdes.strip()

    def set_fields(self, i: int, *, type: str | None = None, package: str | None = None,
                   polarized: bool | None = None, marked_end: str | None = None) -> None:
        b = self.boxes[i]
        if type is not None:
            if type not in PART_TYPES:
                raise ValueError(f"type {type!r} not one of {', '.join(PART_TYPES)}")
            b.type = type
        if marked_end is not None:
            if marked_end not in MARKED_ENDS:
                raise ValueError(f"marked end {marked_end!r} not one of {', '.join(MARKED_ENDS)}")
            b.marked_end = marked_end
        if package is not None:
            b.package = package.strip()
        if polarized is not None:
            b.polarized = bool(polarized)

    def to_parts(self) -> list[Part]:
        """Parts for build_program, or ValueError naming every problem at once."""
        if not self.boxes:
            raise ValueError("no parts: draw boxes or propose them first")
        problems = []
        unnamed = [b.label for b in self.boxes if b.candidate]
        if unnamed:
            shown = ", ".join(unnamed[:6]) + (", ..." if len(unnamed) > 6 else "")
            problems.append(f"{len(unnamed)} box(es) have no refdes ({shown}): name each or delete it")
        counts: dict[str, int] = {}
        for b in self.boxes:
            if b.refdes:
                counts[b.refdes] = counts.get(b.refdes, 0) + 1
            if b.w <= 0 or b.h <= 0:
                problems.append(f"{b.label} has an empty box")
        problems += [f"{r} is used {n} times" for r, n in counts.items() if n > 1]
        if problems:
            raise ValueError("; ".join(problems))
        return [b.part() for b in self.boxes]

    @classmethod
    def from_parts(cls, parts: list[Part]) -> "BoxList":
        bl = cls()
        for p in parts:
            bl.add(p.x, p.y, p.w, p.h, p.refdes, type=p.type, package=p.package,
                   polarized=p.polarized, marked_end=p.marked_end)
        return bl


def rect_from_points(ax: float, ay: float, bx: float, by: float, W: int, H: int,
                     min_side: int = 4) -> tuple[int, int, int, int] | None:
    """A drag between two image points as (x, y, w, h) clipped to the image;
    None for a click or a sliver, which is a selection, not a part."""
    x0, x1 = sorted((ax, bx))
    y0, y1 = sorted((ay, by))
    x0, y0 = int(round(max(0.0, x0))), int(round(max(0.0, y0)))
    x1, y1 = int(round(min(float(W), x1))), int(round(min(float(H), y1)))
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    return x0, y0, x1 - x0, y1 - y0


# ============================================================ teach helpers
def parse_scale(mode: str, text: str) -> tuple[float | None, float | None]:
    """(px_per_mm, board_width_mm) from the Scale radio and its entry."""
    try:
        v = float(text.strip())
    except ValueError:
        raise ValueError(f"scale {text.strip()!r} is not a number") from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError("scale must be a positive number")
    return (v, None) if mode == "px_per_mm" else (None, v)


def check_name(name: str) -> str:
    """A program name is a folder name under programs/, so it cannot climb out of it."""
    name = name.strip()
    if not name:
        raise ValueError("give the program a name")
    if name.startswith(".") or any(c in name for c in '/\\:'):
        raise ValueError(f"{name!r} cannot be a folder name (no leading dot, no / \\ :)")
    return name


def accepted_selfcheck(flags: list[tuple[str, str]], ticked: set[str]) -> list[str] | None:
    """Every flagged refdes if each one is ticked as intended, else None (cancel).

    Accepting some and not others would save a golden with a known-bad part in
    it; the self-check exists so that never happens by omission."""
    names = [r for r, _ in flags]
    return names if all(r in ticked for r in names) else None


@dataclass
class ProgramInfo:
    """The latest version of one program folder, or why it cannot be loaded."""
    name: str
    version: int = 0
    n_goldens: int = 0
    px_per_mm: float = 0.0
    single_golden: bool = False
    program: program.Program | None = None
    error: str = ""

    @property
    def note(self) -> str:
        return self.error or ("single golden" if self.single_golden else "")

    @property
    def label(self) -> str:
        if self.program is None:
            return f"{self.name}  (cannot load)"
        text = f"{self.name}  v{self.version}  {self.n_goldens} golden(s)  {self.px_per_mm:.1f} px/mm"
        return text + ("  single golden" if self.single_golden else "")


def list_programs(root: str | Path) -> list[ProgramInfo]:
    """Every folder under `root` with a v*/program.json, latest version of each."""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob("v*/program.json"))):
        try:
            prog = program.latest(root, d.name)
        # TypeError: a program.json whose part fields do not match Part.
        except (io.PhotoError, OSError, ValueError, KeyError, TypeError) as e:
            out.append(ProgramInfo(d.name, error=str(e)))
            continue
        out.append(ProgramInfo(prog.name, prog.version, len(prog.goldens), prog.px_per_mm,
                               prog.single_golden, prog))
    return out


# ========================================================== inspect helpers
def new_run_dir(runs: str | Path, name: str, now: datetime | None = None) -> Path:
    """runs/<program>/<YYYYmmdd-HHMMSS>, with a suffix if that second is taken,
    so two batches started in one second do not overwrite each other's reports."""
    base = Path(runs) / name / (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    out, n = base, 1
    while out.exists():
        n += 1
        out = base.with_name(f"{base.name}-{n}")
    return out


def duplicate_stems(photos: list[Path]) -> list[str]:
    """Photo names that would collide: report.write_all names every file after
    the photo's stem, so a/board.jpg and b/board.jpg in one batch would overwrite."""
    seen: dict[str, int] = {}
    for p in photos:
        seen[Path(p).stem] = seen.get(Path(p).stem, 0) + 1
    return sorted(s for s, n in seen.items() if n > 1)


def build_calibrators(rec: record.Record) -> dict:
    """Platt scaling per check, where the labels allow it (as `python -m aoi_photo inspect`)."""
    return {c: f for c in CALIBRATED_CHECKS if (f := judge.calibrate(rec.labels(c))) is not None}


def flagged_count(res: BoardResult) -> int:
    """Parts with a FAIL or REVIEW call, plus board-level calls needing an answer."""
    return (sum(1 for pr in res.parts if pr.flagged)
            + sum(1 for c in res.board_calls if c.severity in record.NEEDS_ANSWER))


# =========================================================== review helpers
def recent_runs(rec: record.Record, limit: int = 300) -> list[dict]:
    """Newest runs first, with their report paths when known."""
    return rec.runs(limit)


def run_label(r: dict) -> str:
    try:
        when = datetime.fromisoformat(r["created"]).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        when = r["created"]
    text = (f"#{r['id']}  {Path(r['image']).name}  {r['program']} v{r['version']}  "
            f"{r['machine_verdict']}  {when}")
    return text + ("  superseded" if r.get("superseded_by") is not None else "")


def review_calls(calls: list[dict]) -> list[dict]:
    """The calls an operator has to answer: FAIL and REVIEW, and every board-level call."""
    return [c for c in calls if c["severity"] in record.NEEDS_ANSWER or c["board_level"]]


def score_text(c: dict) -> str:
    """'score 0.97' until calibrated, 'P=0.97' after: Call.score_text is the one rule."""
    return Call(c["check"], c["state"], score=c["score"], calibrated=c["calibrated"]).score_text()


def call_row(c: dict) -> tuple[str, str, str, str, str]:
    """(answer, part, state, score, note) for the Review call list."""
    note = "(unvalidated)" if c["state"] in UNVALIDATED else ""
    return (ANSWER_TEXT.get(c["adjudication"] or "", ""), c["refdes"] or "board", c["state"],
            score_text(c), note)


def call_summary(c: dict) -> str:
    head = " ".join(s for s in (c["refdes"] or "board", c["state"], score_text(c)) if s)
    if c["state"] in UNVALIDATED:
        head += "  (unvalidated)"
    return f"{head}\n{c['check']}: {c['detail'] or '-'}"


def next_unanswered(calls: list[dict], after: int | None = None) -> int | None:
    """Index of the next call with no answer after `after`, wrapping; None if all answered.

    A skip is an answer ("decide later"), so N moves past skipped calls; they
    stay in the list, and the run stays REVIEW until they are answered."""
    n = len(calls)
    start = -1 if after is None else after
    for step in range(1, n + 1):
        i = (start + step) % n
        if calls[i]["adjudication"] is None:
            return i
    return None


def changed_index(before: list[dict], after: list[dict]) -> int | None:
    """The first call whose answer differs between two readings (what an undo touched)."""
    old = {c["id"]: c["adjudication"] for c in before}
    for i, c in enumerate(after):
        if old.get(c["id"], c["adjudication"]) != c["adjudication"]:
            return i
    return None


# ============================================================ stats helpers
STATS_COLUMNS = (
    ("group", "group", 200),
    ("parts_inspected", "parts inspected", 120),
    ("fail_calls", "FAIL calls", 90),
    ("false_calls", "false calls", 90),
    ("fcr", "FCR", 80),
    ("review_calls", "review calls", 100),
    ("review_resolved_ok", "review resolved OK", 140),
    ("boards", "boards", 70),
    ("false_calls_per_board", "false calls / board", 140),
)


def stats_row(d: dict) -> tuple[str, ...]:
    def fmt(key: str) -> str:
        v = d[key]
        if key == "fcr":
            return f"{v:.2%}"
        if key == "false_calls_per_board":
            return f"{v:.2f}"
        return str(v)
    return tuple(fmt(k) for k, _, _ in STATS_COLUMNS)


def stats_rows(rec: record.Record) -> list[tuple[str, ...]]:
    """One row per part group, then the total."""
    return [stats_row(d) for d in rec.stats()] + [stats_row(rec.stats_total())]


# ================================================================== workers
# Each runs on its own thread and only ever puts tuples on the queue. None of
# them touches Tk.
def check_goldens_worker(q: queue.Queue, gen: int, paths: list[Path]) -> None:
    for p in paths:
        try:
            problems = gate.golden_problems(io.load_bgr(p))
        except (io.PhotoError, OSError) as e:
            problems = [str(e)]
        q.put(("golden_checked", gen, str(p), problems))


def load_view_worker(q: queue.Queue, kind: str, key, path: Path) -> None:
    try:
        bgr = io.load_bgr(path)
    except (io.PhotoError, OSError) as e:
        q.put((kind, key, None, None, 1.0, str(e)))
        return
    small, k = downscale(bgr)
    q.put((kind, key, bgr, small, k, ""))


def propose_worker(q: queue.Queue, gen: int, bgr: np.ndarray) -> None:
    from aoi.photo import detect
    q.put(("proposals", gen, teach.propose_boxes(bgr), detect.PartDetector.available()))


def retrain_worker(q: queue.Queue, programs_root: Path) -> None:
    """Rebuild the part detector from every saved program and labelled shoot."""
    from aoi.photo import detect_train
    try:
        data = detect_train.collect(programs_root)
        n = sum(d.source != "cc0" for d in data)
        q.put(("retrain_log", f"training the part detector on {n} labelled photo(s)..."))
        out = detect_train.train(data, log=lambda m: q.put(("retrain_log", f"detector: {m}")))
    except (io.PhotoError, OSError, ValueError, RuntimeError) as e:
        q.put(("retrain_log", f"detector training failed: {e}"))
        return
    q.put(("retrain_log", f"part detector saved to {out}; Propose boxes now uses it"))


def teach_worker(q: queue.Queue, gen: int, name: str, goldens: list[Path], parts: list[Part],
                 px_per_mm: float | None, board_width_mm: float | None, root: Path,
                 accept: list[str]) -> None:
    try:
        rep = teach.build_program(name, goldens, parts, px_per_mm=px_per_mm,
                                  board_width_mm=board_width_mm, root=root,
                                  accept_selfcheck=list(accept))
    except teach.SelfCheckError as e:            # before PhotoError: it is one
        q.put(("teach_selfcheck", gen, e.flags))
    except (io.PhotoError, OSError, ValueError) as e:
        q.put(("teach_error", gen, str(e)))
    else:
        q.put(("teach_ok", gen, rep))


def inspect_worker(q: queue.Queue, prog: program.Program, photos: list[Path], db: Path,
                   out_dir: Path) -> None:
    """`python -m aoi_photo inspect`'s loop, one queue message per board."""
    try:
        golden = photo_inspect.Golden(prog)
    except (io.PhotoError, OSError) as e:
        q.put(("inspect_fatal", f"golden 1 of {prog.name} v{prog.version}: {e}"))
        return
    q.put(("inspect_golden", (prog.name, prog.version), golden.bgr))
    try:
        rec = record.Record(db)
    except (RuntimeError, sqlite3.Error) as e:
        q.put(("inspect_fatal", f"{db}: {e}"))
        return
    with rec:
        try:
            pv = rec.program_version(prog.name, prog.version, prog.path, len(prog.goldens),
                                     prog.px_per_mm, prog.single_golden)
        except ValueError as e:
            q.put(("inspect_fatal", str(e)))
            return
        calib = build_calibrators(rec)
        for i, f in enumerate(photos):
            t0 = time.perf_counter()
            try:
                res, aligned = photo_inspect.inspect(prog, f, golden, calib)
                run_id = rec.save_run(res, pv)
                paths = report.write_all(res, golden.bgr, aligned if aligned is not None
                                         else np.zeros_like(golden.bgr), out_dir)
                rec.set_report(run_id, paths["annotated"], paths["html"])
            except (io.PhotoError, OSError, ValueError, KeyError) as e:
                q.put(("inspect_error", i, str(e)))
                continue
            q.put(("inspect_row", i, run_id, res, paths, aligned, time.perf_counter() - t0))


# ======================================================================= Tk
def to_photo(rgb: np.ndarray, master: tk.Misc) -> tk.PhotoImage:
    """RGB uint8 -> Tk image via PPM, as aoi/review.py: no image library needed.
    `master` is explicit; an image on the wrong interpreter renders blank."""
    h, w = rgb.shape[:2]
    header = f"P6 {w} {h} 255 ".encode()
    return tk.PhotoImage(master=master, format="PPM",
                         data=header + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())


class ImageView:
    """A canvas showing one large image: fit on first show, wheel or +/- to
    zoom, right-drag or space-drag to pan, 0 to fit. `overlay(canvas, view)`
    draws on top after every redraw."""

    def __init__(self, parent: tk.Misc, overlay=None, empty: str = "") -> None:
        self.canvas = tk.Canvas(parent, bg=PANEL, highlightthickness=1,
                                highlightbackground=RULE, highlightcolor=DIM, takefocus=1)
        self.view = View()
        self.full: np.ndarray | None = None
        self.small: np.ndarray | None = None
        self.k = 1.0
        self.overlay = overlay
        self.empty = empty
        self.space = False
        self._photo: tk.PhotoImage | None = None   # Tk drops unreferenced images
        self._queued = False
        self._fitted = False
        self._pan_at: tuple[int, int] | None = None
        c = self.canvas
        c.bind("<Configure>", self._on_configure)
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Button-4>", lambda e: self.zoom(1.2, e.x, e.y))
        c.bind("<Button-5>", lambda e: self.zoom(1 / 1.2, e.x, e.y))
        for b in ("2", "3"):                       # right button is 2 on older macOS Tk, 3 elsewhere
            c.bind(f"<ButtonPress-{b}>", self.pan_start)
            c.bind(f"<B{b}-Motion>", self.pan_move)
            c.bind(f"<ButtonRelease-{b}>", self.pan_end)
        for k in ("plus", "equal", "KP_Add"):
            c.bind(f"<KeyPress-{k}>", lambda e: self.zoom_centre(1.25))
        for k in ("minus", "KP_Subtract"):
            c.bind(f"<KeyPress-{k}>", lambda e: self.zoom_centre(0.8))
        c.bind("<KeyPress-0>", lambda e: self.fit())
        try:                                       # Tk 9: two-finger trackpad scroll
            c.bind("<TouchpadScroll>", self._on_touchpad)
        except tk.TclError:
            pass
        c.bind("<KeyPress-space>", lambda e: setattr(self, "space", True))
        c.bind("<KeyRelease-space>", lambda e: setattr(self, "space", False))

    def size(self) -> tuple[int, int]:
        return self.canvas.winfo_width(), self.canvas.winfo_height()

    def set_image(self, bgr: np.ndarray, small: np.ndarray | None = None, k: float | None = None) -> None:
        self.full = bgr
        self.small, self.k = (small, k) if small is not None and k else downscale(bgr)
        self._fitted = False
        self.fit()

    def clear(self, text: str = "") -> None:
        self.full = self.small = None
        self.empty = text
        self._fitted = False
        self.redraw()

    def fit(self) -> None:
        if self.full is not None:
            cw, ch = self.size()
            self.view.fit(self.full.shape[1], self.full.shape[0], cw, ch)
            self._fitted = cw > 1 and ch > 1
        self.redraw()

    def show_box(self, x: float, y: float, w: float, h: float) -> None:
        if self.full is None:
            return
        if not self._fitted:
            self.fit()
        cw, ch = self.size()
        self.view.show_box(x, y, w, h, cw, ch)
        self.redraw()

    def zoom(self, factor: float, cx: float, cy: float) -> None:
        if self.full is not None:
            self.view.zoom_at(factor, cx, cy)
            self.redraw()

    def zoom_centre(self, factor: float) -> None:
        cw, ch = self.size()
        self.zoom(factor, cw / 2, ch / 2)

    def pan_start(self, e) -> None:
        self.canvas.focus_set()
        self._pan_at = (e.x, e.y)

    def pan_move(self, e) -> None:
        if self._pan_at is None or self.full is None:
            return
        self.view.pan(e.x - self._pan_at[0], e.y - self._pan_at[1])
        self._pan_at = (e.x, e.y)
        self.redraw()

    def pan_end(self, _e=None) -> None:
        self._pan_at = None

    @property
    def panning(self) -> bool:
        return self._pan_at is not None

    def redraw(self) -> None:
        """Coalesced: a burst of wheel or drag events draws once."""
        if not self._queued:
            self._queued = True
            self.canvas.after_idle(self._draw)

    def _on_configure(self, _e) -> None:
        if self.full is not None and not self._fitted:
            self.fit()
        else:
            self.redraw()

    def _on_wheel(self, e) -> None:
        if e.delta:
            self.zoom(1.15 if e.delta > 0 else 1 / 1.15, e.x, e.y)

    def _on_touchpad(self, e) -> None:
        if self.full is None:
            return
        try:
            dx, dy = (float(v) for v in self.canvas.tk.splitlist(
                self.canvas.tk.call("tk::PreciseScrollDeltas", e.delta)))
        except tk.TclError:
            return
        self.view.pan(dx, dy)
        self.redraw()

    def _draw(self) -> None:
        self._queued = False
        c = self.canvas
        c.delete("all")
        cw, ch = self.size()
        if self.full is None:
            if self.empty:
                c.create_text(cw / 2, ch / 2, text=self.empty, fill=DIM, justify="center",
                              width=max(200, cw - 80))
            return
        out = render_patch(self.full, self.small, self.k, self.view, cw, ch)
        if out is not None:
            rgb, x, y = out
            self._photo = to_photo(rgb, c)
            c.create_image(int(round(x)), int(round(y)), anchor="nw", image=self._photo)
        if self.overlay is not None:
            self.overlay(c, self.view)


@dataclass
class SessionRun:
    """What this session knows about a run beyond the record: the result and its reports."""
    result: BoardResult
    paths: dict
    golden_key: tuple[str, int]


class App:
    """The Photo AOI window."""

    def __init__(self, root: tk.Tk, programs: str | Path, runs: str | Path, db: str | Path) -> None:
        self.root = root
        self.programs_root = Path(programs)
        self.runs_root = Path(runs)
        self.db_path = Path(db)
        self.rec = record.Record(self.db_path)        # main-thread connection; workers open their own
        self.q: queue.Queue = queue.Queue()
        self.busy: set[str] = set()

        # Program tab
        self.boxes = BoxList()
        self.golden_paths: list[Path] = []
        self.golden_status: dict[str, list[str] | None] = {}
        self.editable = False
        self.base_info: ProgramInfo | None = None
        self.program_infos: list[ProgramInfo] = []
        self.teach_gen = 0
        self.accepted: list[str] = []
        self._teach_args: tuple | None = None
        self._loading_fields = False
        self._band: tuple[float, float] | None = None
        self._ed_mode: str | None = None

        # Inspect tab
        self.inspect_photos: list[Path] = []
        self.inspect_iids: list[str] = []
        self.inspect_runs: dict[str, int] = {}
        self.inspect_programs: dict[str, program.Program] = {}
        self.inspect_out: Path | None = None

        # Review tab
        self.session: dict[int, SessionRun] = {}
        self.goldens: dict[tuple[str, int], np.ndarray] = {}
        self.aligned: OrderedDict[int, np.ndarray] = OrderedDict()
        self.run_rows: list[dict] = []
        self.run_id: int | None = None
        self.calls_all: list[dict] = []
        self.calls: list[dict] = []
        self.cur: int | None = None
        self.shown_at = 0.0
        self.annotated: np.ndarray | None = None
        self._crop_photos: list[tk.PhotoImage] = []

        root.title("Photo AOI")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1440, sw - 40)}x{min(900, sh - 90)}+10+10")
        root.minsize(min(1100, sw - 40), min(680, sh - 90))
        self._fonts()
        self._style()
        self._build()
        self._handlers = {
            "worker_done": self._on_worker_done,
            "golden_checked": self._on_golden_checked,
            "editor_image": self._on_editor_image,
            "proposals": self._on_proposals,
            "retrain_log": lambda msg: self.status(msg, error="failed" in msg),
            "teach_ok": self._on_teach_ok,
            "teach_error": self._on_teach_error,
            "teach_selfcheck": self._on_teach_selfcheck,
            "inspect_golden": self._on_inspect_golden,
            "inspect_row": self._on_inspect_row,
            "inspect_error": self._on_inspect_error,
            "inspect_fatal": self._on_inspect_fatal,
            "review_image": self._on_review_image,
        }
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<KeyPress>", self._on_key, add="+")
        self._poll_id = root.after(POLL_MS, self._poll)
        self.refresh_programs()
        self.refresh_runs()
        self.refresh_stats()
        self._sync_program_controls()
        self.new_program()                     # open ready to teach: upload is the first step
        self.status(f"programs {self.programs_root}   runs {self.runs_root}   record {self.db_path}")

    # ------------------------------------------------------------ plumbing
    def _fonts(self) -> None:
        r = self.root
        self.f_head = tkfont.Font(root=r, family="Helvetica", size=16, weight="bold")
        self.f_big = tkfont.Font(root=r, family="Helvetica", size=20, weight="bold")
        self.f_lab = tkfont.Font(root=r, family="Helvetica", size=12)
        self.f_mono = tkfont.Font(root=r, family="Menlo", size=12)
        self.f_small = tkfont.Font(root=r, family="Menlo", size=10)
        self.f_bold = tkfont.Font(root=r, family="Helvetica", size=12, weight="bold")

    def _style(self) -> None:
        # clam, because the native macOS theme ignores background colours and
        # a white table in a dark room undoes the operator's adaptation.
        st = ttk.Style(self.root)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=INK, fieldbackground=PANEL, bordercolor=RULE,
                     lightcolor=PANEL, darkcolor=PANEL, troughcolor=PANEL, selectbackground=SELECTED,
                     selectforeground=INK, insertcolor=INK, font=self.f_lab)
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", background=PANEL, foreground=DIM, padding=(18, 6))
        st.map("TNotebook.Tab", background=[("selected", RULE)], foreground=[("selected", INK)])
        st.configure("TButton", background=PANEL, foreground=INK, padding=(10, 4))
        st.map("TButton", background=[("disabled", BG), ("active", RULE)],
               foreground=[("disabled", DIM)])
        st.configure("Accent.TButton", background=FALSE, foreground="#10131a")
        st.map("Accent.TButton", background=[("disabled", RULE), ("active", "#5cc488")],
               foreground=[("disabled", DIM)])
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=INK,
                     rowheight=24, font=self.f_lab)
        st.map("Treeview", background=[("selected", SELECTED)], foreground=[("selected", INK)])
        st.configure("Treeview.Heading", background=RULE, foreground=DIM, relief="flat")
        st.map("Treeview.Heading", background=[("active", RULE)])
        st.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=INK, arrowcolor=INK)
        st.map("TCombobox", fieldbackground=[("readonly", PANEL), ("disabled", BG)],
               foreground=[("readonly", INK), ("disabled", DIM)],
               selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", INK)])
        st.configure("TEntry", fieldbackground=PANEL, foreground=INK)
        st.map("TEntry", fieldbackground=[("disabled", BG)], foreground=[("disabled", DIM)])
        for w in ("TCheckbutton", "TRadiobutton"):
            st.configure(w, background=BG, foreground=INK, indicatorbackground=PANEL,
                         indicatorforeground=INK)
            st.map(w, background=[("active", BG)], foreground=[("disabled", DIM)])
        st.configure("Horizontal.TProgressbar", background=FALSE, troughcolor=PANEL)
        self.root.option_add("*TCombobox*Listbox.background", PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", INK)
        self.root.option_add("*TCombobox*Listbox.selectBackground", SELECTED)

    def _label(self, parent, text: str = "", fg: str = INK, font=None, **kw) -> tk.Label:
        return tk.Label(parent, text=text, fg=fg, bg=kw.pop("bg", BG), font=font or self.f_lab,
                        anchor=kw.pop("anchor", "w"), justify=kw.pop("justify", "left"), **kw)

    def status(self, msg: str, error: bool = False) -> None:
        self.status_bar.configure(text=msg, fg=DEFECT if error else DIM)

    def _spawn(self, name: str, fn, *args) -> None:
        """Run fn(q, *args) on a thread. worker_done always follows, with ok=False
        if fn raised something unexpected (its traceback goes to the terminal)."""
        self.busy.add(name)

        def run() -> None:
            ok = False
            try:
                fn(self.q, *args)
                ok = True
            finally:
                self.q.put(("worker_done", name, ok))

        threading.Thread(target=run, name=f"aoi-{name}", daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                self._handlers[msg[0]](*msg[1:])
        except queue.Empty:
            pass
        self._poll_id = self.root.after(POLL_MS, self._poll)

    def _on_worker_done(self, name: str, ok: bool) -> None:
        self.busy.discard(name)
        if not ok:
            self.status(f"{name} stopped on an unexpected error; the terminal has the details", error=True)
        if name == "teach":
            self._sync_program_controls()
        elif name == "inspect":
            self._inspect_finished()
        elif name == "propose":
            self.b_propose.configure(state="normal" if self.editable else "disabled")

    def on_close(self) -> None:
        working = sorted(self.busy & set(LONG_WORK))
        if working and not messagebox.askyesno(
                "Photo AOI", f"Still working ({', '.join(working)}). Quit anyway?\n"
                "A program being saved may be left half-written.", parent=self.root):
            return
        self.close()

    def close(self) -> None:
        self.root.after_cancel(self._poll_id)
        self.rec.close()
        self.root.destroy()

    # --------------------------------------------------------------- layout
    def _build(self) -> None:
        self.status_bar = tk.Label(self.root, text="", fg=DIM, bg=PANEL, font=self.f_small,
                                   anchor="w", padx=12, pady=5)
        self.status_bar.pack(side="bottom", fill="x")
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.tab_program = tk.Frame(self.nb, bg=BG)
        self.tab_inspect = tk.Frame(self.nb, bg=BG)
        self.tab_review = tk.Frame(self.nb, bg=BG)
        self.tab_stats = tk.Frame(self.nb, bg=BG)
        for tab, text in ((self.tab_program, "1  Program (golden)"), (self.tab_inspect, "2  Inspect (test)"),
                          (self.tab_review, "3  Review"), (self.tab_stats, "4  Stats")):
            self.nb.add(tab, text=text)
        self._build_program_tab(self.tab_program)
        self._build_inspect_tab(self.tab_inspect)
        self._build_review_tab(self.tab_review)
        self._build_stats_tab(self.tab_stats)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)

    def _tree(self, parent, columns, height: int = 8, stretch_last: bool = True) -> ttk.Treeview:
        tv = ttk.Treeview(parent, columns=[c for c, _, _ in columns], show="headings",
                          height=height, selectmode="browse")
        for i, (c, text, width) in enumerate(columns):
            tv.heading(c, text=text, anchor="w")
            tv.column(c, width=width, anchor="w", stretch=stretch_last and i == len(columns) - 1)
        return tv

    def _on_tab(self, _e=None) -> None:
        tab = self.nb.select()
        if tab == str(self.tab_stats):
            self.refresh_stats()
        elif tab == str(self.tab_review):
            self.refresh_runs()
            self.rv.canvas.focus_set()
        elif tab == str(self.tab_inspect):
            self._refresh_inspect_programs()

    # ========================================================== Program tab
    def _build_program_tab(self, tab: tk.Frame) -> None:
        left = tk.Frame(tab, bg=BG, width=380)
        left.pack(side="left", fill="y", padx=(8, 6), pady=8)
        left.pack_propagate(False)
        right = tk.Frame(tab, bg=BG, width=300)
        right.pack(side="right", fill="y", padx=(6, 8), pady=8)
        right.pack_propagate(False)
        centre = tk.Frame(tab, bg=BG)
        centre.pack(side="left", fill="both", expand=True, pady=8)

        # programs
        self._label(left, "Programs", font=self.f_head).pack(fill="x")
        self.t_programs = self._tree(left, (("name", "name", 130), ("v", "v", 34),
                                            ("g", "goldens", 60), ("ppm", "px/mm", 60),
                                            ("note", "", 80)), height=7)
        self.t_programs.pack(fill="x", pady=(4, 4))
        self.t_programs.bind("<<TreeviewSelect>>", self._on_program_select)
        row = tk.Frame(left, bg=BG)
        row.pack(fill="x")
        ttk.Button(row, text="New program", command=self.new_program).pack(side="left")
        self.b_newver = ttk.Button(row, text="New version from this", command=self.new_version)
        self.b_newver.pack(side="left", padx=4)
        ttk.Button(row, text="Refresh", command=self.refresh_programs).pack(side="left")

        # goldens
        self._label(left, "Golden photos", font=self.f_head).pack(fill="x", pady=(16, 0))
        self.l_golden_count = self._label(left, "", fg=DIM, wraplength=360)
        self.l_golden_count.pack(fill="x")
        self.t_goldens = self._tree(left, (("n", "#", 30), ("file", "file", 120), ("check", "check", 200)),
                                    height=6)
        self.t_goldens.tag_configure("bad", foreground=DEFECT)
        self.t_goldens.tag_configure("ok", foreground=FALSE)
        self.t_goldens.tag_configure("wait", foreground=DIM)
        self.t_goldens.pack(fill="both", expand=True, pady=4)
        row = tk.Frame(left, bg=BG)
        row.pack(fill="x")
        # Always clickable: from a read-only program it starts a new version, so
        # the one thing a first-time user looks for is never greyed out.
        self.b_add_golden = ttk.Button(row, text="Upload golden photos...", style="Accent.TButton",
                                       command=self.add_goldens)
        self.b_add_golden.pack(side="left")
        self.b_rm_golden = ttk.Button(row, text="Remove", command=self.remove_golden)
        self.b_rm_golden.pack(side="left", padx=4)
        self._label(left, "Golden 1 carries the boxes; every other photo is aligned onto it.",
                    fg=DIM, font=self.f_small, wraplength=360).pack(fill="x", pady=(6, 0))

        # editor
        bar = tk.Frame(centre, bg=BG)
        bar.pack(fill="x", pady=(0, 4))
        self.b_propose = ttk.Button(bar, text="Propose boxes", command=self.propose)
        self.b_propose.pack(side="left")
        ttk.Button(bar, text="Retrain detector", command=self.retrain).pack(side="left", padx=(4, 0))
        self.b_import = ttk.Button(bar, text="Import parts.csv...", command=self.import_parts)
        self.b_import.pack(side="left", padx=4)
        ttk.Button(bar, text="Export parts.csv...", command=self.export_parts).pack(side="left")
        ttk.Button(bar, text="Fit", command=lambda: self.ed.fit()).pack(side="left", padx=4)
        self.l_mode = self._label(bar, "", fg=AMBER)
        self.l_mode.pack(side="left", padx=10)
        self.ed = ImageView(centre, overlay=self._draw_boxes, empty=EDITOR_EMPTY)
        self.ed.canvas.pack(fill="both", expand=True)
        self._label(centre, "left-drag: draw a box   click: select   Delete: remove   "
                            "wheel or + -: zoom   right-drag or space-drag: pan   0: fit",
                    fg=DIM, font=self.f_small).pack(fill="x", pady=(4, 0))
        c = self.ed.canvas
        c.bind("<ButtonPress-1>", self._ed_press)
        c.bind("<B1-Motion>", self._ed_drag)
        c.bind("<ButtonRelease-1>", self._ed_release)
        c.bind("<KeyPress-Delete>", self._ed_delete)
        c.bind("<KeyPress-BackSpace>", self._ed_delete)

        # right: name, scale, selected part, save
        self._label(right, "Program", font=self.f_head).pack(fill="x")
        self._label(right, "name").pack(fill="x", pady=(4, 0))
        self.v_name = tk.StringVar(master=self.root)
        self.e_name = ttk.Entry(right, textvariable=self.v_name)
        self.e_name.pack(fill="x")

        self._label(right, "scale").pack(fill="x", pady=(10, 0))
        self.v_scale_mode = tk.StringVar(master=self.root, value="px_per_mm")
        self.v_scale = tk.StringVar(master=self.root)
        self.r_ppm = ttk.Radiobutton(right, text="px per mm", value="px_per_mm", variable=self.v_scale_mode)
        self.r_ppm.pack(anchor="w")
        self.r_bw = ttk.Radiobutton(right, text="board long side (mm)", value="board_mm",
                                    variable=self.v_scale_mode)
        self.r_bw.pack(anchor="w")
        self.e_scale = ttk.Entry(right, textvariable=self.v_scale)
        self.e_scale.pack(fill="x")

        self._label(right, "Selected part", font=self.f_head).pack(fill="x", pady=(18, 0))
        self.v_refdes = tk.StringVar(master=self.root)
        self.v_type = tk.StringVar(master=self.root)
        self.v_pkg = tk.StringVar(master=self.root)
        self.v_pol = tk.BooleanVar(master=self.root)
        self.v_end = tk.StringVar(master=self.root)
        self._label(right, "refdes").pack(fill="x", pady=(4, 0))
        self.e_refdes = ttk.Entry(right, textvariable=self.v_refdes)
        self.e_refdes.pack(fill="x")
        self._label(right, "type").pack(fill="x", pady=(4, 0))
        self.c_type = ttk.Combobox(right, textvariable=self.v_type, values=PART_TYPES, state="readonly")
        self.c_type.pack(fill="x")
        self._label(right, "package").pack(fill="x", pady=(4, 0))
        self.e_pkg = ttk.Entry(right, textvariable=self.v_pkg)
        self.e_pkg.pack(fill="x")
        self.k_pol = ttk.Checkbutton(right, text="polarized", variable=self.v_pol,
                                     command=self._on_field_change)
        self.k_pol.pack(anchor="w", pady=(6, 0))
        self._label(right, "marked end").pack(fill="x", pady=(4, 0))
        self.c_end = ttk.Combobox(right, textvariable=self.v_end, values=MARKED_ENDS, state="readonly")
        self.c_end.pack(fill="x")
        self.v_refdes.trace_add("write", lambda *_: self._on_field_change())
        self.v_pkg.trace_add("write", lambda *_: self._on_field_change())
        self.c_type.bind("<<ComboboxSelected>>", lambda e: self._on_field_change())
        self.c_end.bind("<<ComboboxSelected>>", lambda e: self._on_field_change())
        for w in (self.e_refdes, self.e_pkg):
            w.bind("<Return>", lambda e: self.ed.canvas.focus_set())
        self.l_parts = self._label(right, "", fg=DIM)
        self.l_parts.pack(fill="x", pady=(14, 0))

        self.b_save = ttk.Button(right, text="Save program", style="Accent.TButton",
                                 command=self.save_program)
        self.b_save.pack(side="bottom", fill="x", pady=(8, 0))
        self._label(right, "Saving aligns every golden and measures every part; it takes a while.",
                    fg=DIM, font=self.f_small, wraplength=280).pack(side="bottom", fill="x")

    # -- program list
    def refresh_programs(self) -> None:
        self.program_infos = list_programs(self.programs_root)
        tv = self.t_programs
        tv.delete(*tv.get_children())
        for info in self.program_infos:
            tv.insert("", "end", iid=info.name, values=(
                info.name, info.version or "", info.n_goldens or "",
                f"{info.px_per_mm:.1f}" if info.px_per_mm else "", info.note))
        self._refresh_inspect_programs()

    def _info(self, name: str) -> ProgramInfo | None:
        return next((i for i in self.program_infos if i.name == name), None)

    def _selected_program(self) -> ProgramInfo | None:
        sel = self.t_programs.selection()
        return self._info(sel[0]) if sel else None

    def _dirty(self) -> bool:
        return self.editable and (len(self.boxes) > 0 or bool(self.golden_paths))

    def _confirm_discard(self) -> bool:
        return not self._dirty() or messagebox.askyesno(
            "Photo AOI", "Discard the program being edited?", parent=self.root)

    def _on_program_select(self, _e=None) -> None:
        info = self._selected_program()
        if info is None:
            return
        b = self.base_info
        if not self.editable and b is not None and (b.name, b.version) == (info.name, info.version):
            return
        if info.program is None:
            self.status(f"{info.name}: {info.error}", error=True)
            return
        if not self._confirm_discard():
            return
        self._view_program(info)

    def _view_program(self, info: ProgramInfo) -> None:
        p = info.program
        self._reset_teach(False, p.name, p.golden_paths(), BoxList.from_parts(p.parts),
                          ("px_per_mm", f"{p.px_per_mm:g}"), info)
        self.l_mode.configure(text=f"viewing {p.name} v{p.version} (read-only)", fg=DIM)

    def new_program(self) -> None:
        if not self._confirm_discard():
            return
        self._reset_teach(True, "", [], BoxList(), ("px_per_mm", ""), None)
        self.t_programs.selection_set(())
        self.l_mode.configure(text="new program", fg=AMBER)
        self.e_name.focus_set()

    def new_version(self) -> None:
        info = self._selected_program() or self.base_info
        if info is None or info.program is None:
            self.status("pick a program on the left first", error=True)
            return
        if self.editable and not self._confirm_discard():
            return
        p = info.program
        self._reset_teach(True, p.name, p.golden_paths(), BoxList.from_parts(p.parts),
                          ("px_per_mm", f"{p.px_per_mm:g}"), info)
        self.l_mode.configure(text=f"new version of {p.name} (saves as v{p.version + 1})", fg=AMBER)

    def _reset_teach(self, editable: bool, name: str, goldens: list[Path], boxes: BoxList,
                     scale: tuple[str, str], base: ProgramInfo | None) -> None:
        self.teach_gen += 1
        self.editable = editable
        self.base_info = base
        self.accepted = []
        self.v_name.set(name)
        self.v_scale_mode.set(scale[0])
        self.v_scale.set(scale[1])
        self.boxes = boxes
        self.golden_paths = []
        self.golden_status = {}
        self.ed.clear("loading golden 1..." if goldens else EDITOR_EMPTY)
        self._add_golden_paths(list(goldens), check=editable)
        self._load_fields()
        self._sync_program_controls()

    def _sync_program_controls(self) -> None:
        on = "normal" if self.editable else "disabled"
        saving = "teach" in self.busy
        for w in (self.b_rm_golden, self.b_import, self.e_name, self.e_scale,
                  self.r_ppm, self.r_bw):
            w.configure(state=on)
        self.b_propose.configure(state=on if "propose" not in self.busy else "disabled")
        self.b_save.configure(state="normal" if self.editable and not saving else "disabled")
        self.b_newver.configure(state="normal" if not saving else "disabled")
        self._load_fields()

    # -- goldens
    def add_goldens(self) -> None:
        if not self.editable:
            # viewing a saved program: adding goldens means a new version of it,
            # or a new program when nothing is selected
            info = self._selected_program() or self.base_info
            if info is not None and info.program is not None:
                self.new_version()
            else:
                self.new_program()
            if not self.editable:
                return
        paths = filedialog.askopenfilenames(parent=self.root, title="Golden photos (at least 3)",
                                            filetypes=PHOTO_TYPES)
        if paths:
            self._add_golden_paths(sorted(Path(p) for p in paths), check=True)

    def _add_golden_paths(self, paths: list[Path], check: bool) -> None:
        had_first = self.golden_paths[0] if self.golden_paths else None
        new = [p for p in paths if p not in self.golden_paths]
        self.golden_paths += new
        for p in new:
            self.golden_status[str(p)] = None if check else []
        self._refresh_goldens()
        if check and new:
            self._spawn("golden check", check_goldens_worker, self.teach_gen, new)
        if self.golden_paths and self.golden_paths[0] != had_first:
            self._spawn("load golden", load_view_worker, "editor_image",
                        (self.teach_gen, str(self.golden_paths[0])), self.golden_paths[0])

    def remove_golden(self) -> None:
        sel = self.t_goldens.selection()
        if not sel:
            return
        i = int(sel[0])
        if i == 0 and len(self.boxes):
            self.status("golden 1 carries the boxes; start a new program to change it", error=True)
            return
        del self.golden_paths[i]
        self._refresh_goldens()
        if i == 0:
            self.ed.clear(EDITOR_EMPTY)
            if self.golden_paths:
                self._spawn("load golden", load_view_worker, "editor_image",
                            (self.teach_gen, str(self.golden_paths[0])), self.golden_paths[0])

    def _refresh_goldens(self) -> None:
        tv = self.t_goldens
        tv.delete(*tv.get_children())
        for i, p in enumerate(self.golden_paths):
            probs = self.golden_status.get(str(p))
            if probs is None:
                text, tag = "checking...", "wait"
            elif probs:
                text, tag = "; ".join(probs), "bad"
            else:
                text, tag = ("ok" if self.editable else "in program"), "ok"
            tv.insert("", "end", iid=str(i), values=(i + 1, p.name, text), tags=(tag,))
        n = len(self.golden_paths)
        short = n < config.PHOTO_MIN_GOLDENS
        self.l_golden_count.configure(text=f"{n} golden photo(s). Rule: {GOLDEN_RULE}.",
                                      fg=DEFECT if short and self.editable else DIM)

    def _on_golden_checked(self, gen: int, path: str, problems: list[str]) -> None:
        if gen == self.teach_gen and path in self.golden_status:
            self.golden_status[path] = problems
            self._refresh_goldens()

    def _on_editor_image(self, key, bgr, small, k, err: str) -> None:
        gen, path = key
        if gen != self.teach_gen or not self.golden_paths or str(self.golden_paths[0]) != path:
            return
        if bgr is None:
            self.ed.clear(err)
            self.status(err, error=True)
            return
        self.ed.set_image(bgr, small, k)

    # -- editor
    def _draw_boxes(self, c: tk.Canvas, view: View) -> None:
        for i, b in enumerate(self.boxes.boxes):
            x0, y0 = view.to_canvas(b.x, b.y)
            x1, y1 = view.to_canvas(b.x + b.w, b.y + b.h)
            sel = i == self.boxes.selected
            col = INK if sel else (AMBER if b.candidate else FALSE)
            c.create_rectangle(x0, y0, x1, y1, outline=col, width=3 if sel else 2,
                               dash=(5, 3) if b.candidate else ())
            c.create_text(x0, y0 - 2, anchor="sw", text=b.label, fill=col, font=self.f_small)

    def _ed_press(self, e) -> None:
        c = self.ed.canvas
        c.focus_set()
        if self.ed.space:
            self.ed.pan_start(e)
            self._ed_mode = "pan"
            return
        if self.ed.full is None:
            return
        ix, iy = self.ed.view.to_image(e.x, e.y)
        i = self.boxes.hit(ix, iy)
        if i is not None or not self.editable:
            self._ed_mode = None
            self._select_box(i)
            return
        self._ed_mode = "draw"
        self._band = (ix, iy)
        c.create_rectangle(e.x, e.y, e.x, e.y, outline=INK, dash=(3, 3), tags="band")

    def _ed_drag(self, e) -> None:
        if self._ed_mode == "pan":
            self.ed.pan_move(e)
        elif self._ed_mode == "draw" and self._band is not None:
            x0, y0 = self.ed.view.to_canvas(*self._band)
            self.ed.canvas.coords("band", x0, y0, e.x, e.y)

    def _ed_release(self, e) -> None:
        mode, self._ed_mode = self._ed_mode, None
        if mode == "pan":
            self.ed.pan_end()
            return
        if mode != "draw" or self._band is None or self.ed.full is None:
            return
        self.ed.canvas.delete("band")
        H, W = self.ed.full.shape[:2]
        rect = rect_from_points(*self._band, *self.ed.view.to_image(e.x, e.y), W, H)
        self._band = None
        if rect is None:
            self._select_box(None)
            return
        self._select_box(self.boxes.add(*rect))
        self.e_refdes.focus_set()

    def _ed_delete(self, _e=None) -> None:
        if self.editable and self.boxes.delete():
            self._load_fields()
            self.ed.redraw()

    def _select_box(self, i: int | None) -> None:
        self.boxes.select(i)
        self._load_fields()
        self.ed.redraw()

    def _load_fields(self) -> None:
        """Selected box -> side panel. The traces fire on set(); _loading_fields
        keeps them from writing the values straight back."""
        i = self.boxes.selected
        self._loading_fields = True
        try:
            b = self.boxes.boxes[i] if i is not None else None
            self.v_refdes.set(b.refdes if b else "")
            self.v_type.set(b.type if b else "")
            self.v_pkg.set(b.package if b else "")
            self.v_pol.set(b.polarized if b else False)
            self.v_end.set(b.marked_end if b else "")
        finally:
            self._loading_fields = False
        on = "normal" if (b is not None and self.editable) else "disabled"
        for w in (self.e_refdes, self.e_pkg, self.k_pol):
            w.configure(state=on)
        for w in (self.c_type, self.c_end):
            w.configure(state="readonly" if on == "normal" else "disabled")
        n, u = len(self.boxes), self.boxes.unnamed
        self.l_parts.configure(text=f"{n} box(es)" + (f", {u} unnamed" if u else ""),
                               fg=AMBER if u else DIM)

    def _on_field_change(self) -> None:
        i = self.boxes.selected
        if self._loading_fields or i is None or not self.editable:
            return
        self.boxes.rename(i, self.v_refdes.get())
        try:
            self.boxes.set_fields(i, type=self.v_type.get() or None, package=self.v_pkg.get(),
                                  polarized=self.v_pol.get(), marked_end=self.v_end.get() or None)
        except ValueError as e:
            self.status(str(e), error=True)
        n, u = len(self.boxes), self.boxes.unnamed
        self.l_parts.configure(text=f"{n} box(es)" + (f", {u} unnamed" if u else ""),
                               fg=AMBER if u else DIM)
        self.ed.redraw()

    def propose(self) -> None:
        if self.ed.full is None:
            self.status("add golden photos first: boxes are proposed on golden 1", error=True)
            return
        self.b_propose.configure(state="disabled")
        self.status("proposing boxes on golden 1...")
        self._spawn("propose", propose_worker, self.teach_gen, self.ed.full)

    def _on_proposals(self, gen: int, rects, from_detector: bool = True) -> None:
        if gen != self.teach_gen or not self.editable:
            return
        added, skipped = self.boxes.add_proposals(rects)
        self._load_fields()
        self.ed.redraw()
        note = ("" if from_detector else
                "  No trained part detector yet: these come from colour alone and include solder "
                "and pads; use Retrain detector once a program is saved.")
        self.status(f"{added} proposal(s) added, dashed P1, P2...: name each part you keep, "
                    f"delete the rest" + (f"; {skipped} already boxed" if skipped else "") + note,
                    error=not from_detector)

    def retrain(self) -> None:
        if "retrain" in self.busy:
            self.status("the part detector is already training")
            return
        if not messagebox.askyesno(
                "Retrain detector", "Retrain the part-box detector from every saved program and "
                "labelled photo?\n\nIt runs on this machine, offline, and takes roughly 45 minutes "
                "on an Apple-silicon Mac. You can keep working meanwhile.", parent=self.root):
            return
        self._spawn("retrain", retrain_worker, self.programs_root)

    def import_parts(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Import parts.csv",
                                          filetypes=[("CSV", "*.csv"), ("All files", "*")])
        if not path:
            return
        try:
            parts = program.read_parts_csv(path)
            boxes = BoxList.from_parts(parts)
        except (io.PhotoError, OSError, ValueError) as e:
            messagebox.showerror("Import parts.csv", str(e), parent=self.root)
            return
        if len(self.boxes) and not messagebox.askyesno(
                "Import parts.csv", f"Replace the {len(self.boxes)} box(es) here with "
                f"{len(parts)} from {Path(path).name}?", parent=self.root):
            return
        self.boxes = boxes
        self._load_fields()
        self.ed.redraw()
        self.status(f"{len(parts)} part(s) from {path}")

    def export_parts(self) -> None:
        try:
            parts = self.boxes.to_parts()
        except ValueError as e:
            self.status(f"not exported: {e}", error=True)
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Export parts.csv",
                                            defaultextension=".csv", initialfile="parts.csv")
        if not path:
            return
        try:
            program.write_parts_csv(path, parts)
        except OSError as e:
            self.status(f"not exported: {e}", error=True)
            return
        self.status(f"{len(parts)} part(s) -> {path}")

    # -- save
    def save_program(self) -> None:
        try:
            name = check_name(self.v_name.get())
            parts = self.boxes.to_parts()
            ppm, bw = parse_scale(self.v_scale_mode.get(), self.v_scale.get())
        except ValueError as e:
            self.status(f"not saved: {e}", error=True)
            return
        self._teach_args = (name, list(self.golden_paths), parts, ppm, bw)
        self._start_teach()

    def _start_teach(self) -> None:
        name, goldens, parts, ppm, bw = self._teach_args
        self.busy.add("teach")
        self._sync_program_controls()
        self.status(f"saving {name}: checking {len(goldens)} golden(s), aligning them, "
                    "measuring every part...")
        self._spawn("teach", teach_worker, self.teach_gen, name, goldens, parts, ppm, bw,
                    self.programs_root, list(self.accepted))

    def _on_teach_error(self, gen: int, msg: str) -> None:
        self.status(f"not saved: {msg}", error=True)
        messagebox.showerror("Program not saved", msg, parent=self.root)

    def _on_teach_selfcheck(self, gen: int, flags: list[tuple[str, str]]) -> None:
        if gen != self.teach_gen:
            return
        accepted = self._selfcheck_dialog(flags)
        if accepted is None:
            self.status("not saved: fix the golden, or tick every flagged part as intended", error=True)
            return
        self.accepted = sorted(set(self.accepted) | set(accepted))
        self._start_teach()

    def _selfcheck_dialog(self, flags: list[tuple[str, str]]) -> list[str] | None:
        top = tk.Toplevel(self.root)
        top.title("Golden self-check")
        top.configure(bg=BG, padx=18, pady=14)
        top.transient(self.root)
        self._label(top, "Golden 1 has parts that sit off their own silkscreen outline.",
                    font=self.f_head).pack(fill="x")
        self._label(top, "Fix the golden board and reshoot, or tick each one that is placed "
                         "that way on purpose. Saving needs every flag ticked.",
                    fg=DIM, wraplength=520).pack(fill="x", pady=(2, 10))
        ticks: dict[str, tk.BooleanVar] = {}
        for r, d in flags:
            row = tk.Frame(top, bg=BG)
            row.pack(fill="x", pady=1)
            ticks[r] = tk.BooleanVar(master=top, value=False)
            ttk.Checkbutton(row, text="intended", variable=ticks[r]).pack(side="left")
            self._label(row, f"{r}: {d}", font=self.f_mono).pack(side="left", padx=8)
        out: dict[str, list[str] | None] = {"accepted": None}

        def confirm() -> None:
            out["accepted"] = accepted_selfcheck(flags, {r for r, v in ticks.items() if v.get()})
            top.destroy()

        row = tk.Frame(top, bg=BG)
        row.pack(fill="x", pady=(12, 0))
        ttk.Button(row, text="Save with these accepted", style="Accent.TButton",
                   command=confirm).pack(side="right")
        ttk.Button(row, text="Cancel", command=top.destroy).pack(side="right", padx=6)
        top.bind("<Escape>", lambda e: top.destroy())
        top.grab_set()
        self.root.wait_window(top)
        return out["accepted"]

    def _on_teach_ok(self, gen: int, rep: teach.TeachReport) -> None:
        p = rep.program
        lines = [f"{p.name} v{p.version}: {len(p.parts)} parts, {len(p.goldens)} goldens, "
                 f"{p.px_per_mm:.2f} px/mm", f"saved in {p.path}"]
        if rep.selfcheck_flags:
            lines.append("self-check accepted as intended: " + ", ".join(r for r, _ in rep.selfcheck_flags))
        un = rep.selfcheck_unchecked
        lines.append(f"self-check not possible for {len(un)} part(s) (no outline of their own)"
                     + (": " + ", ".join(un[:12]) + (", ..." if len(un) > 12 else "") if un else ""))
        self.status(f"{lines[0]}; self-check not possible for {len(un)} part(s)")
        self.refresh_programs()
        info = self._info(p.name)
        if info is not None and info.program is not None:
            self._view_program(info)
            self.t_programs.selection_set(info.name)
        messagebox.showinfo("Program saved", "\n".join(lines), parent=self.root)

    # ========================================================== Inspect tab
    def _build_inspect_tab(self, tab: tk.Frame) -> None:
        bar = tk.Frame(tab, bg=BG)
        bar.pack(fill="x", padx=10, pady=(10, 4))
        self._label(bar, "Program").pack(side="left")
        self.v_iprog = tk.StringVar(master=self.root)
        self.c_iprog = ttk.Combobox(bar, textvariable=self.v_iprog, state="readonly", width=56)
        self.c_iprog.pack(side="left", padx=(6, 12))
        self.c_iprog.bind("<<ComboboxSelected>>", lambda e: self._on_iprog())
        ttk.Button(bar, text="Upload test photos...", style="Accent.TButton",
                   command=self.add_tests).pack(side="left")
        self.b_clear = ttk.Button(bar, text="Clear", command=self.clear_tests)
        self.b_clear.pack(side="left", padx=4)
        self.b_run = ttk.Button(bar, text="Run", style="Accent.TButton", command=self.run_inspect)
        self.b_run.pack(side="left", padx=(12, 0))
        self.pb = ttk.Progressbar(bar, mode="determinate", length=240)
        self.pb.pack(side="left", padx=12)
        self.l_iprogress = self._label(bar, "", fg=DIM, font=self.f_mono)
        self.l_iprogress.pack(side="left")
        self.l_istamp = self._label(tab, "", fg=AMBER)
        self.l_istamp.pack(fill="x", padx=10)
        self.t_inspect = self._tree(tab, (("photo", "photo", 260), ("verdict", "verdict", 90),
                                          ("flagged", "flagged", 70), ("time", "time", 70),
                                          ("note", "note", 600)), height=20)
        for v in ("PASS", "FAIL", "REVIEW"):
            self.t_inspect.tag_configure(v, foreground=verdict_colour(v))
        self.t_inspect.tag_configure("ERROR", foreground=DEFECT)
        self.t_inspect.tag_configure("queued", foreground=DIM)
        self.t_inspect.pack(fill="both", expand=True, padx=10, pady=4)
        self.t_inspect.bind("<Double-1>", self._on_inspect_open)
        self.t_inspect.bind("<Return>", self._on_inspect_open)
        self._label(tab, "One photo per board. Double-click a board to review it.",
                    fg=DIM, font=self.f_small).pack(fill="x", padx=10, pady=(0, 8))

    def _refresh_inspect_programs(self) -> None:
        self.inspect_programs = {i.label: i.program for i in self.program_infos if i.program is not None}
        labels = list(self.inspect_programs)
        self.c_iprog.configure(values=labels)
        if self.v_iprog.get() not in self.inspect_programs:
            self.v_iprog.set(labels[0] if labels else "")
        self._on_iprog()

    def _on_iprog(self) -> None:
        prog = self.inspect_programs.get(self.v_iprog.get())
        self.l_istamp.configure(text=(f"every result from this program is stamped: {prog.stamp}"
                                      if prog is not None and prog.single_golden else ""))

    def add_tests(self) -> None:
        paths = filedialog.askopenfilenames(parent=self.root, title="Test photos (one per board)",
                                            filetypes=PHOTO_TYPES)
        for p in sorted(Path(x) for x in paths):
            if p in self.inspect_photos:
                continue
            self.inspect_photos.append(p)
            self.t_inspect.insert("", "end", iid=str(len(self.inspect_photos) - 1),
                                  values=(p.name, "", "", "", str(p.parent)), tags=("queued",))
        self.l_iprogress.configure(text=f"{len(self.inspect_photos)} photo(s)")

    def clear_tests(self) -> None:
        if "inspect" in self.busy:
            return
        self.inspect_photos = []
        self.inspect_runs = {}
        self.t_inspect.delete(*self.t_inspect.get_children())
        self.pb.configure(value=0)
        self.l_iprogress.configure(text="")

    def run_inspect(self) -> None:
        if "inspect" in self.busy:
            return
        prog = self.inspect_programs.get(self.v_iprog.get())
        if prog is None:
            self.status("choose a program to inspect against (teach one on the Program tab)", error=True)
            return
        if not self.inspect_photos:
            self.status("add test photos first: one photo per board", error=True)
            return
        dups = duplicate_stems(self.inspect_photos)
        if dups:
            self.status("two photos share a name and their reports would overwrite each other: "
                        + ", ".join(dups) + "; rename one or run them separately", error=True)
            return
        self.inspect_out = new_run_dir(self.runs_root, prog.name)
        self.inspect_runs = {}
        for i, p in enumerate(self.inspect_photos):
            self.t_inspect.item(str(i), values=(p.name, "queued", "", "", str(p.parent)), tags=("queued",))
        self.pb.configure(maximum=len(self.inspect_photos), value=0)
        self.l_iprogress.configure(text=f"0 / {len(self.inspect_photos)}")
        self.b_run.configure(state="disabled")
        self.b_clear.configure(state="disabled")
        self.status(f"inspecting {len(self.inspect_photos)} board(s) against {prog.name} "
                    f"v{prog.version} -> {self.inspect_out}")
        self._spawn("inspect", inspect_worker, prog, list(self.inspect_photos), self.db_path,
                    self.inspect_out)

    def _step_progress(self) -> None:
        self.pb.step(1)
        self.l_iprogress.configure(text=f"{int(self.pb['value'])} / {len(self.inspect_photos)}")

    def _on_inspect_golden(self, key: tuple[str, int], bgr: np.ndarray) -> None:
        self.goldens[key] = bgr

    def _on_inspect_row(self, i: int, run_id: int, res: BoardResult, paths: dict,
                        aligned: np.ndarray | None, secs: float) -> None:
        iid = str(i)
        note = "; ".join(s for s in (res.stamp, res.reasons[0] if res.reasons else "") if s)
        self.t_inspect.item(iid, values=(Path(res.image).name, res.verdict, flagged_count(res),
                                         f"{secs:.1f} s", note), tags=(res.verdict,))
        self.inspect_runs[iid] = run_id
        self.session[run_id] = SessionRun(res, paths, (res.program, res.program_version))
        if aligned is not None:
            self.aligned[run_id] = aligned
            while len(self.aligned) > ALIGNED_KEEP:
                self.aligned.popitem(last=False)
        self._step_progress()

    def _on_inspect_error(self, i: int, msg: str) -> None:
        p = self.inspect_photos[i]
        self.t_inspect.item(str(i), values=(p.name, "ERROR", "", "", msg), tags=("ERROR",))
        self._step_progress()

    def _on_inspect_fatal(self, msg: str) -> None:
        self.status(f"inspection stopped: {msg}", error=True)
        messagebox.showerror("Inspection stopped", msg, parent=self.root)

    def _inspect_finished(self) -> None:
        self.b_run.configure(state="normal")
        self.b_clear.configure(state="normal")
        verdicts = [self.t_inspect.set(iid, "verdict") for iid in self.t_inspect.get_children()]
        counts = {v: verdicts.count(v) for v in ("PASS", "FAIL", "REVIEW", "ERROR") if v in verdicts}
        if counts and self.status_bar.cget("fg") != DEFECT:
            self.status(f"{len(self.inspect_runs)} board(s) inspected: "
                        + ", ".join(f"{n} {v}" for v, n in counts.items())
                        + f"; reports in {self.inspect_out}")
        self.refresh_runs()
        self.refresh_stats()

    def _on_inspect_open(self, _e=None) -> None:
        sel = self.t_inspect.selection()
        run_id = self.inspect_runs.get(sel[0]) if sel else None
        if run_id is None:
            return
        self.nb.select(self.tab_review)
        self.open_run(run_id)

    # =========================================================== Review tab
    def _build_review_tab(self, tab: tk.Frame) -> None:
        top = tk.Frame(tab, bg=BG)
        top.pack(fill="x", padx=10, pady=(10, 2))
        self._label(top, "Run").pack(side="left")
        self.v_run = tk.StringVar(master=self.root)
        self.c_run = ttk.Combobox(top, textvariable=self.v_run, state="readonly", width=70)
        self.c_run.pack(side="left", padx=6)
        self.c_run.bind("<<ComboboxSelected>>", self._on_run_pick)
        ttk.Button(top, text="Refresh", command=self.refresh_runs).pack(side="left")
        self.l_operator = self._label(top, "", font=self.f_big)
        self.l_operator.pack(side="right")
        self._label(top, "operator", fg=DIM).pack(side="right", padx=(18, 6))
        self.l_machine = self._label(top, "", font=self.f_big)
        self.l_machine.pack(side="right")
        self._label(top, "machine", fg=DIM).pack(side="right", padx=(0, 6))
        self.l_rstamp = self._label(tab, "", fg=AMBER)
        self.l_rstamp.pack(fill="x", padx=10)

        body = tk.Frame(tab, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        side = tk.Frame(body, bg=BG, width=470)
        side.pack(side="right", fill="y", padx=(10, 0))
        side.pack_propagate(False)
        self.rv = ImageView(body, overlay=self._draw_highlight, empty="Pick a run above.")
        self.rv.canvas.pack(side="left", fill="both", expand=True)
        # Left button has nothing to draw here, so a plain left-drag pans too.
        self.rv.canvas.bind("<ButtonPress-1>", self.rv.pan_start)
        self.rv.canvas.bind("<B1-Motion>", self.rv.pan_move)
        self.rv.canvas.bind("<ButtonRelease-1>", self.rv.pan_end)

        self._label(side, "Calls needing an answer", font=self.f_head).pack(fill="x")
        self.t_calls = self._tree(side, (("answer", "answer", 80), ("part", "part", 64),
                                         ("state", "state", 130), ("score", "score", 84),
                                         ("note", "", 100)), height=11)
        for k, col in ANSWER_COLOURS.items():
            self.t_calls.tag_configure(k, foreground=col)
        self.t_calls.pack(fill="x", pady=4)
        self.t_calls.bind("<<TreeviewSelect>>", self._on_call_select)
        self.l_call = self._label(side, "", font=self.f_mono, wraplength=450)
        self.l_call.pack(fill="x", pady=(6, 6))
        crops = tk.Frame(side, bg=BG)
        crops.pack(fill="x")
        self.crop_caps, self.crop_imgs = [], []
        for text in ("GOLDEN", "TEST"):
            col = tk.Frame(crops, bg=BG)
            col.pack(side="left", padx=(0, 10), anchor="n")
            cap = self._label(col, text, fg=DIM, font=self.f_small)
            cap.pack(anchor="w")
            img = tk.Label(col, bg=PANEL, bd=0, fg=DIM, width=CROP_SIDE // 8, height=CROP_SIDE // 18)
            img.pack()
            self.crop_caps.append(cap)
            self.crop_imgs.append(img)
        self._label(side, "D  real defect     F  false call     S  skip\n"
                          "R  reshoot (board calls)     N  next unanswered     U  undo",
                    fg=DIM, font=self.f_mono).pack(side="bottom", fill="x", pady=(8, 0))

    def refresh_runs(self) -> None:
        self.run_rows = recent_runs(self.rec)
        self.c_run.configure(values=[run_label(r) for r in self.run_rows])
        if self.run_id is not None:
            idx = next((i for i, r in enumerate(self.run_rows) if r["id"] == self.run_id), None)
            if idx is not None:
                self.c_run.current(idx)

    def _on_run_pick(self, _e=None) -> None:
        i = self.c_run.current()
        if 0 <= i < len(self.run_rows):
            self.open_run(self.run_rows[i]["id"])

    def _run_row(self, run_id: int) -> dict | None:
        row = next((r for r in self.run_rows if r["id"] == run_id), None)
        if row is None:
            self.refresh_runs()
            row = next((r for r in self.run_rows if r["id"] == run_id), None)
        return row

    def open_run(self, run_id: int) -> None:
        row = self._run_row(run_id)
        if row is None:
            self.status(f"no run {run_id} in {self.db_path}", error=True)
            return
        self.run_id = run_id
        self.cur = None
        self.annotated = None
        self.c_run.current(self.run_rows.index(row))
        self.l_machine.configure(text=row["machine_verdict"], fg=verdict_colour(row["machine_verdict"]))
        self.l_rstamp.configure(text=row["stamp"])
        sess = self.session.get(run_id)
        annotated = (sess.paths.get("annotated") if sess is not None else None) or row.get("annotated")
        if annotated and Path(annotated).exists():
            self.rv.clear("loading the annotated image...")
            self._spawn("load report", load_view_worker, "review_image", run_id, annotated)
        else:
            self.rv.clear("This run's report image is not available (recorded before report "
                          "paths were stored, or moved).\nIts calls are on the right; answer them as usual.\n"
                          f"Reports are under {self.runs_root / row['program']}.")
        self._reload_calls()
        if self.calls:
            j = next_unanswered(self.calls)
            self._select_call(0 if j is None else j)
        else:
            self._clear_call()
        self.rv.canvas.focus_set()

    def _on_review_image(self, run_id: int, bgr, small, k, err: str) -> None:
        if run_id != self.run_id:
            return
        if bgr is None:
            self.rv.clear(err)
            self.status(err, error=True)
            return
        self.annotated = bgr
        self.rv.set_image(bgr, small, k)
        if self.cur is not None:
            self._present_call(self.cur, restart_clock=False)

    def _reload_calls(self) -> None:
        self.calls_all = self.rec.calls(self.run_id)
        self.calls = review_calls(self.calls_all)
        tv = self.t_calls
        tv.delete(*tv.get_children())
        for c in self.calls:
            tv.insert("", "end", iid=str(c["id"]), values=call_row(c), tags=(c["adjudication"] or "open",))
        st = self.rec.operator_status(self.run_id)
        self.l_operator.configure(text=st, fg=verdict_colour(st))
        if not self.calls:
            self.l_call.configure(text="Nothing on this board needs an answer.")

    def _clear_call(self) -> None:
        self.cur = None
        for img in self.crop_imgs:
            img.configure(image="", text="", width=CROP_SIDE // 8, height=CROP_SIDE // 18)
        self.rv.redraw()

    def _select_call(self, i: int) -> None:
        self._present_call(i)
        iid = str(self.calls[i]["id"])
        self.t_calls.selection_set(iid)
        self.t_calls.see(iid)

    def _on_call_select(self, _e=None) -> None:
        sel = self.t_calls.selection()
        if not sel:
            return
        i = next((k for k, c in enumerate(self.calls) if str(c["id"]) == sel[0]), None)
        if i is not None and i != self.cur:
            self._present_call(i)
            self.rv.canvas.focus_set()

    def _part_result(self, c: dict):
        sess = self.session.get(self.run_id)
        if sess is None or not c["refdes"]:
            return None
        try:
            return sess.result.part(c["refdes"])
        except KeyError:
            return None

    def _present_call(self, i: int, restart_clock: bool = True) -> None:
        """Show call i: summary, zoom to the part, golden | test crops. The clock
        behind `seconds` starts here, and not again when the image arrives late."""
        self.cur = i
        c = self.calls[i]
        if restart_clock:
            self.shown_at = time.perf_counter()
        self.l_call.configure(text=call_summary(c))
        pr = self._part_result(c)
        if self.rv.full is not None:
            if pr is not None:
                p = pr.part
                self.rv.show_box(p.x + pr.dx_px, p.y + pr.dy_px, p.w, p.h)
            else:
                self.rv.fit()
        self._show_crops(pr)

    def _draw_highlight(self, c: tk.Canvas, view: View) -> None:
        if self.cur is None or self.cur >= len(self.calls):
            return
        pr = self._part_result(self.calls[self.cur])
        if pr is None:
            return
        p, pad = pr.part, 0.15 * max(pr.part.w, pr.part.h)
        x0, y0 = view.to_canvas(p.x + pr.dx_px - pad, p.y + pr.dy_px - pad)
        x1, y1 = view.to_canvas(p.x + pr.dx_px + p.w + pad, p.y + pr.dy_px + p.h + pad)
        c.create_rectangle(x0, y0, x1, y1, outline=INK, width=2, dash=(6, 4))

    def _show_crops(self, pr) -> None:
        self._crop_photos = []
        pair: list[np.ndarray | None] = [None, None]
        caps = ["GOLDEN", "TEST"]
        if pr is not None:
            sess = self.session[self.run_id]
            g, t = self.goldens.get(sess.golden_key), self.aligned.get(self.run_id)
            if g is not None and t is not None:
                pair = list(report.crop_pair(g, t, pr.part))
            elif self.annotated is not None:
                pair[1] = report.crop_pair(self.annotated, self.annotated, pr.part)[1]
                caps = ["GOLDEN (not kept in memory)", "TEST (annotated)"]
        for cap, lab, img, text in zip(self.crop_caps, self.crop_imgs, pair, caps):
            cap.configure(text=text)
            if img is None or min(img.shape[:2]) == 0:
                lab.configure(image="", text="no crop", width=CROP_SIDE // 8, height=CROP_SIDE // 18)
                continue
            photo = to_photo(cv2.cvtColor(fit_side(img), cv2.COLOR_BGR2RGB), self.root)
            self._crop_photos.append(photo)
            lab.configure(image=photo, text="", width=0, height=0)

    # -- keys
    def _on_key(self, e) -> str | None:
        """D F S R N U on the Review tab. Not while typing into a field, and not
        with Control or Command held, so no shortcut is taken for a decision."""
        if self.nb.select() != str(self.tab_review) or e.state & 0x000C:
            return None
        try:
            focus = self.root.focus_get()
        except KeyError:                          # a combobox popdown has focus
            return None
        if isinstance(focus, (tk.Entry, ttk.Entry, tk.Text)):
            return None
        k = (e.keysym or "").lower()
        if k in ("d", "f", "s", "r"):
            self.answer(k.upper())
        elif k == "n":
            self.next_call()
        elif k == "u":
            self.undo()
        else:
            return None
        return "break"

    def answer(self, key: str) -> None:
        if self.run_id is None or self.cur is None or not self.calls:
            self.status("pick a run and a call first", error=True)
            return
        c = self.calls[self.cur]
        secs = time.perf_counter() - self.shown_at
        try:
            self.rec.adjudicate(c["id"], key, secs)
        except (ValueError, KeyError) as e:
            self.status(str(e), error=True)
            return
        cur = self.cur
        self._reload_calls()
        who = f"{c['refdes'] or 'board'} {c['state']}"
        j = next_unanswered(self.calls, cur)
        if j is None:
            self._select_call(min(cur, len(self.calls) - 1))
            self.status(f"{who}: {ANSWER_TEXT[key]} ({secs:.1f} s). Every call on this board is "
                        f"answered; operator status {self.l_operator.cget('text')}")
        else:
            self._select_call(j)
            self.status(f"{who}: {ANSWER_TEXT[key]} ({secs:.1f} s)")

    def next_call(self) -> None:
        if not self.calls:
            return
        j = next_unanswered(self.calls, self.cur)
        if j is None:
            self.status("no unanswered calls on this board")
        else:
            self._select_call(j)
            self.status(f"next unanswered: {self.calls[j]['refdes'] or 'board'} {self.calls[j]['state']}")

    def undo(self) -> None:
        if self.run_id is None:
            return
        before = self.calls
        if not self.rec.undo_last(self.run_id):
            self.status("nothing to undo on this run")
            return
        self._reload_calls()
        j = changed_index(before, self.calls)
        if self.calls:
            self._select_call(j if j is not None else min(self.cur or 0, len(self.calls) - 1))
        self.status("last answer undone" + (f": {self.calls[j]['refdes'] or 'board'} "
                                            f"{self.calls[j]['state']}" if j is not None else ""))

    # ============================================================ Stats tab
    def _build_stats_tab(self, tab: tk.Frame) -> None:
        bar = tk.Frame(tab, bg=BG)
        bar.pack(fill="x", padx=10, pady=(10, 4))
        self._label(bar, FCR_NOTE, fg=DIM).pack(side="left")
        ttk.Button(bar, text="Refresh", command=self.refresh_stats).pack(side="right")
        self.t_stats = self._tree(tab, STATS_COLUMNS, height=20)
        self.t_stats.tag_configure("total", foreground=INK, font=self.f_bold)
        self.t_stats.pack(fill="both", expand=True, padx=10, pady=(4, 10))

    def refresh_stats(self) -> None:
        tv = self.t_stats
        tv.delete(*tv.get_children())
        rows = stats_rows(self.rec)
        for i, r in enumerate(rows):
            tv.insert("", "end", values=r, tags=("total",) if i == len(rows) - 1 else ())


def main(programs: str | Path = "programs", runs: str | Path = "out/photo_runs",
         db: str | Path = "out/photo_aoi.db") -> int:
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"cannot open a window: {e}", file=sys.stderr)
        return 2
    try:
        app = App(root, programs, runs, db)
    except (RuntimeError, sqlite3.Error) as e:
        messagebox.showerror("Photo AOI", f"cannot open the record {db}: {e}", parent=root)
        root.destroy()
        return 2
    ms = os.environ.get(AUTOCLOSE_ENV, "").strip()
    if ms.isdigit():
        root.after(int(ms), app.close)            # tests and smoke runs only
    root.lift()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
