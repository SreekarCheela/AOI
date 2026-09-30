"""Photo AOI desktop app: set up a board once, inspect many, answer the calls.

Native Tk, no network, no browser (spec section 5). The app runs on a line PC
that may have no network at all, and an operator clearing calls cannot wait on
a page load. Tk is in the standard library, so nothing new is installed.

Three tabs, in the order a new board goes through them. Each opens with one
line saying what to do next, because a first-time user should never have to
guess which of twenty buttons matters:

    1 Set up board    upload golden photos -> parts are found on golden 1 by
                      themselves and kept as P1..Pn -> optionally rename a
                      part and mark the polarized ones -> Save. A live
                      checklist says what Save still needs; Save stays off,
                      with the reason beside it, until nothing does.
    2 Inspect boards  one photo per board -> inspect, record, report. Clicking
                      a board shows its annotated photo and flagged parts in
                      the tab; "Review calls" opens it in Results.
    3 Results         the calls that need an answer, with buttons as well as
                      keys (D F S R N U), and the false-call stats beside them.

Nothing on the set-up tab is mandatory beyond golden photos and at least one
part. One golden is allowed and stamps every result single-golden; the board
length is optional, and without it offsets are reported in pixels. Only the
polarized parts need the operator: which parts can go in backwards, and
whether each is a band part (diode, LED) or a marked one (IC, transistor).

Two rules shape the code:

* The window never waits on the library. Aligning a 12 MP photo takes seconds,
  so every library call that touches a photo runs on a worker thread and
  reports back through a queue that the Tk loop polls with after(). A worker
  never touches Tk: Tk is not thread-safe, and the failure is a hang, not an
  exception. Every worker catches everything it can raise and says so on the
  queue, so a status never hangs on "checking...".
* Everything that can be decided without a display is a plain function or
  class (View, BoxList, the save checklist, the call list, the stats rows), so
  the tests run on a build machine with no screen. The Tk classes at the
  bottom only wire them up.
"""
from __future__ import annotations

import csv
import math
import os
import queue
import re
import sqlite3
import sys
import threading
import time
import tkinter as tk
import traceback
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import cv2
import numpy as np

from aoi import config
from aoi.photo import gate, io, judge, names, program, record, report, teach
from aoi.photo import inspect as photo_inspect
from aoi.photo.model import MARKED_ENDS, PART_TYPES, UNVALIDATED, BoardResult, Part
from aoi.review import BG, DEFECT, DIM, FALSE, INK, PANEL, RULE

AMBER = "#e0a030"            # REVIEW: between the defect red and the pass green
SELECTED = "#34507a"
POLL_MS = 50
SMALL_SIDE = 2048            # long side of the copy drawn when zoomed out
MAX_ZOOM = 8.0               # canvas pixels per image pixel
CROP_SIDE = 220              # long side of each golden | test crop in Results
ALIGNED_KEEP = 6             # aligned test photos kept for crops (about 36 MB each)
AUTOCLOSE_ENV = "AOI_PHOTO_APP_AUTOCLOSE_MS"
LONG_WORK = ("teach", "inspect", "retrain")    # quitting mid-way loses work; ask first
EDITOR_EMPTY = ("Click \"Upload golden photos...\" on the left and pick 1 to 3 photos of a\n"
                "known-good board (JPEG, PNG or HEIC). Golden 1 appears here and its parts\n"
                "are found automatically.")

GOLDEN_RULE = (f"{config.PHOTO_MIN_GOLDENS} goldens from {config.PHOTO_MIN_GOLDENS} different "
               "good boards give real thresholds")
FCR_NOTE = ("False-call rate = FAIL calls marked F / parts inspected "
            "(spec section 8 target: under 0.2% per part)")
PHOTO_TYPES = [("Photos", "*.jpg *.jpeg *.png *.heic *.JPG *.JPEG *.PNG *.HEIC"), ("All files", "*")]
CALIBRATED_CHECKS = ("missing", "offset", "skew", "polarity", "tombstone", "wrong_part")
SCALE_LABELS = {"board_mm": "Board length (mm), measure with a ruler — optional",
                "px_per_mm": "Scale (px per mm) — optional"}

VERDICT_COLOURS = {"PASS": FALSE, "FAIL": DEFECT, "REVIEW": AMBER}
ANSWER_TEXT = {"D": "defect", "F": "false call", "S": "skipped", "R": "reshoot"}
ANSWER_COLOURS = {"D": DEFECT, "F": FALSE, "S": DIM, "R": AMBER}

# Polarity, as the operator is asked it. The library checks a band (diode,
# LED, bridge) or a turned marking (everything else) against golden 1, so all
# it needs is "polarized, and which kind": the dropdown picks the type that
# selects the right check, so nobody has to know the type list. Which end is
# marked is not asked: the check never reads it, and work with no effect is
# how the first user ran out of patience.
POL_NO, POL_DIODE, POL_IC = "No", "Diode / LED", "IC / transistor"
POLARITY_CHOICES = (POL_NO, POL_DIODE, POL_IC)
STRIPE_TYPES = ("diode", "led", "bridge")


def verdict_colour(verdict: str) -> str:
    """PASS green, FAIL red, REVIEW amber; anything else (SUPERSEDED, ERROR) dim."""
    return VERDICT_COLOURS.get(verdict, DIM)


def single_golden_banner(n: int) -> str:
    """The warning shown while fewer goldens than config.PHOTO_MIN_GOLDENS are loaded."""
    if n < 1 or n >= config.PHOTO_MIN_GOLDENS:
        return ""
    k = config.PHOTO_MIN_GOLDENS
    return (f"{n} golden: results are stamped single-golden. For real thresholds add "
            f"{k} goldens from {k} different good boards.")


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
    """One part box on golden 1, full-resolution pixels. Named P<tag> until renamed."""
    x: int
    y: int
    w: int
    h: int
    refdes: str = ""
    type: str = "other"
    package: str = ""
    polarized: bool = False
    marked_end: str = "none"
    tag: int = 0                 # P-number, the part's name while refdes is ''

    @property
    def named(self) -> bool:
        return bool(self.refdes)

    @property
    def label(self) -> str:
        return self.refdes or f"P{self.tag}"

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def part(self) -> Part:
        return Part(self.label, self.x, self.y, self.w, self.h, self.type, self.package,
                    self.polarized, self.marked_end)


class BoxList:
    """The parts being taught: add, select, rename, delete, undo, and turn into Parts.

    A found box is a part from the moment it is found, named P1, P2, ... A
    refdes is an optional rename: presence is judged the same either way, and
    asking for 37 names before the first save is how the first user gave up.
    What the operator removes is what is not a part.
    """

    def __init__(self) -> None:
        self.boxes: list[Box] = []
        self.selected: int | None = None
        self._tag = 0
        self._undo: list[tuple] = []      # ("add", [Box, ...]) or ("delete", index, Box)

    def __len__(self) -> int:
        return len(self.boxes)

    @property
    def unnamed(self) -> int:
        return sum(1 for b in self.boxes if not b.named)

    @property
    def polarized(self) -> int:
        return sum(1 for b in self.boxes if b.polarized)

    def add(self, x: int, y: int, w: int, h: int, refdes: str = "", *, undoable: bool = True,
            **fields) -> int:
        self._tag += 1
        self.boxes.append(Box(int(x), int(y), int(w), int(h), refdes.strip(), tag=self._tag))
        i = len(self.boxes) - 1
        try:
            self.set_fields(i, **fields)
        except ValueError:
            del self.boxes[i]
            raise
        if undoable:
            self._undo.append(("add", [self.boxes[i]]))
        return i

    def add_proposals(self, rects, overlap: float = 0.5) -> tuple[int, int]:
        """(added, skipped): a proposal on top of an existing box is skipped,
        so finding parts twice does not stack duplicates. One undo removes them all."""
        added: list[Box] = []
        skipped = 0
        for r in rects:
            r = tuple(int(v) for v in r)
            if any(teach.iou(r, b.rect) > overlap for b in self.boxes):
                skipped += 1
                continue
            added.append(self.boxes[self.add(*r, undoable=False)])
        if added:
            self._undo.append(("add", added))
        return len(added), skipped

    def apply_names(self, rects, found: list[str | None]) -> tuple[int, int]:
        """Silkscreen names onto the boxes they were read for: (named, left).
        Only a box still carrying its auto name and still where it was found
        takes one, so nothing the operator typed or moved is overwritten; the
        name's letter sets the type and whether it is polarized."""
        taken = {b.refdes for b in self.boxes if b.named}
        named = 0
        for r, n in zip(rects, found):
            if not n or n in taken:
                continue
            r = tuple(int(v) for v in r)
            i = next((k for k, b in enumerate(self.boxes) if not b.named and b.rect == r), None)
            if i is None:
                continue
            t, pol = names.type_of(n)
            self.boxes[i].refdes = n
            self.set_fields(i, type=t, polarized=pol)
            taken.add(n)
            named += 1
        return named, self.unnamed

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
        self._undo.append(("delete", i, self.boxes[i]))
        del self.boxes[i]
        if self.selected == i:
            self.selected = None
        elif self.selected is not None and self.selected > i:
            self.selected -= 1
        return True

    def undo(self) -> str | None:
        """Take back the last add or delete; what was undone, or None if nothing was."""
        if not self._undo:
            return None
        op = self._undo.pop()
        keep = self.boxes[self.selected] if self.selected is not None else None
        if op[0] == "delete":
            _, i, box = op
            self.boxes.insert(min(i, len(self.boxes)), box)
            self.selected = self.boxes.index(box)
            return f"{box.label} is back"
        gone = {id(b) for b in op[1]}
        self.boxes = [b for b in self.boxes if id(b) not in gone]
        self.selected = next((k for k, b in enumerate(self.boxes) if b is keep), None)
        return (f"{op[1][0].label} removed again" if len(op[1]) == 1
                else f"{len(op[1])} found parts removed")

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

    def problems(self) -> list[str]:
        """Everything that stops these boxes being taught, all at once."""
        if not self.boxes:
            return ["no parts yet: draw a box around a part, or find parts again"]
        out = [f"{b.label} has an empty box" for b in self.boxes if b.w <= 0 or b.h <= 0]
        counts = Counter(b.label for b in self.boxes)
        out += [f"two parts are called {r}: rename one" for r, n in counts.items() if n > 1]
        return out

    def to_parts(self) -> list[Part]:
        """Parts for build_program, or ValueError naming every problem at once."""
        problems = self.problems()
        if problems:
            raise ValueError("; ".join(problems))
        return [b.part() for b in self.boxes]

    @classmethod
    def from_parts(cls, parts: list[Part]) -> "BoxList":
        bl = cls()
        for p in parts:
            bl.add(p.x, p.y, p.w, p.h, p.refdes, type=p.type, package=p.package,
                   polarized=p.polarized, marked_end=p.marked_end)
            m = re.fullmatch(r"P(\d+)", p.refdes)
            if m:                        # a saved auto name stays an auto name, with its number
                b = bl.boxes[-1]
                b.refdes, b.tag = "", int(m.group(1))
        bl._tag = max([bl._tag] + [b.tag for b in bl.boxes])
        bl._undo = []
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


def parts_outside(parts: list[Part], W: int, H: int) -> list[str]:
    """Refdes of every part whose box does not lie wholly inside a W x H golden 1."""
    return [p.refdes for p in parts
            if p.w <= 0 or p.h <= 0 or p.x < 0 or p.y < 0 or p.x + p.w > W or p.y + p.h > H]


# ================================================================= polarity
def polarity_choice(b: Box) -> str:
    """What the Polarized dropdown shows for a box."""
    if not b.polarized:
        return POL_NO
    return POL_DIODE if b.type in STRIPE_TYPES else POL_IC


def polarity_fields(choice: str, current_type: str) -> dict:
    """The Box fields one dropdown choice sets.

    A band part keeps diode, led or bridge if it already is one; anything else
    polarized is checked by its marking turned 180 degrees, which the library
    does for every type outside STRIPE_TYPES. "No" keeps the type."""
    if choice == POL_NO:
        return dict(polarized=False)
    if choice == POL_DIODE:
        t = current_type if current_type in STRIPE_TYPES else "diode"
    elif choice == POL_IC:
        t = current_type if current_type not in STRIPE_TYPES + ("other", "resistor", "capacitor") else "ic"
    else:
        raise ValueError(f"polarized {choice!r} not one of {', '.join(POLARITY_CHOICES)}")
    return dict(type=t, polarized=True)


# ============================================================ teach helpers
def parse_scale(mode: str, text: str) -> tuple[float | None, float | None]:
    """(px_per_mm, board_width_mm) from the scale field; (None, None) when blank.

    Blank is legal: the program is saved without a scale and offsets are
    reported in pixels."""
    text = text.strip()
    if not text:
        return None, None
    try:
        v = float(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a number") from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError("the scale must be a positive number")
    return (v, None) if mode == "px_per_mm" else (None, v)


def check_name(name: str) -> str:
    """A program name is a folder name under programs/, so it cannot climb out of it."""
    name = name.strip()
    if not name:
        raise ValueError("give the program a name")
    if name.startswith(".") or any(c in name for c in '/\\:'):
        raise ValueError(f"{name!r} cannot be a folder name (no leading dot, no / \\ :)")
    return name


def default_program_name(goldens: list[Path], taken, today: datetime | None = None) -> str:
    """The golden folder's name (a shoot is usually one folder per board), or
    board-<date>; with -2, -3 ... so a new program never lands on an old one."""
    base = ""
    if goldens:
        base = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(goldens[0]).parent.name).strip("-.")
    if not base:
        base = "board-" + (today or datetime.now()).strftime("%Y%m%d")
    taken = set(taken)
    name, n = base, 1
    while name in taken:
        n += 1
        name = f"{base}-{n}"
    return name


def same_shot(msg: str) -> tuple[int, int] | None:
    """(i, j), 1-based, from the library's refusal "golden i is the same shot as
    golden j", so the app can point at the photo to remove; None for anything else."""
    m = re.search(r"golden (\d+) is the same shot as golden (\d+)", msg)
    return (int(m.group(1)), int(m.group(2))) if m else None


def accepted_selfcheck(flags: list[tuple[str, str]], ticked: set[str]) -> list[str] | None:
    """Every flagged refdes if each one is ticked as intended, else None (cancel).

    Accepting some and not others would save a golden with a known-bad part in
    it; the self-check exists so that never happens by omission."""
    names = [r for r, _ in flags]
    return names if all(r in ticked for r in names) else None


# Checklist marks. The state decides the colour; only "todo" and "wait" block Save.
OK, WARN, TODO, WAIT, OPTIONAL = "ok", "warn", "todo", "wait", "optional"
CHECK_MARKS = {OK: "✓", WARN: "!", TODO: "✗", WAIT: "…", OPTIONAL: "–"}


def save_checklist(name: str, golden_states: list[list[str] | None], box_problems: list[str],
                   n_parts: int, n_polarized: int, scale_text: str, scale_mode: str = "board_mm",
                   finding: bool = False) -> tuple[list[tuple[str, str]], str | None]:
    """[(state, text)] for the checklist beside Save, and the first reason Save
    is blocked (None when it is not).

    golden_states is one entry per golden: None while it is being checked, else
    its list of problems. Every problem shows at once; nothing is revealed only
    after the previous one is fixed."""
    rows: list[tuple[str, str]] = []
    try:
        rows.append((OK, f"Name: {check_name(name)}"))
    except ValueError as e:
        rows.append((TODO, f"Name: {e}"))

    n = len(golden_states)
    bad = [(k + 1, p) for k, p in enumerate(golden_states) if p]
    if n == 0:
        rows.append((TODO, "Golden photos: upload at least 1"))
    elif any(p is None for p in golden_states):
        rows.append((WAIT, f"Golden photos: checking {sum(p is None for p in golden_states)} of {n}..."))
    elif bad:
        k, p = bad[0]
        more = f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""
        rows.append((TODO, f"Golden {k} rejected: {'; '.join(p)}{more}. Remove it"))
    elif n < config.PHOTO_MIN_GOLDENS:
        rows.append((WARN, f"Golden photos: {n} (results stamped single-golden; "
                           f"{config.PHOTO_MIN_GOLDENS} recommended)"))
    else:
        rows.append((OK, f"Golden photos: {n}"))

    if finding:
        rows.append((WAIT, "Parts: finding parts on golden 1..."))
    elif box_problems:
        rows.append((TODO, f"Parts: {'; '.join(box_problems)}"))
    else:
        rows.append((OK, f"Parts found: {n_parts}"))

    rows.append((OPTIONAL if n_polarized == 0 else OK,
                 f"Polarized parts marked: {n_polarized} (optional)"))

    what = "Board length" if scale_mode == "board_mm" else "Scale"
    try:
        ppm, bw = parse_scale(scale_mode, scale_text)
    except ValueError as e:
        rows.append((TODO, f"{what}: {e}"))
    else:
        if ppm is None and bw is None:
            rows.append((OPTIONAL, f"{what}: not given (optional; offsets reported in pixels)"))
        else:
            rows.append((OK, f"{what}: {bw:g} mm" if bw else f"{what}: {ppm:g} px/mm"))

    reason = next((t for s, t in rows if s in (TODO, WAIT)), None)
    return rows, reason


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
        scale = f"{self.px_per_mm:.1f} px/mm" if self.px_per_mm else "no scale"
        text = f"{self.name}  v{self.version}  {self.n_goldens} golden(s)  {scale}"
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


def pick_program(names: list[str], current: str | None, prefer: str | None = None) -> str | None:
    """Which program the Inspect choice should hold after the list changes.

    Keyed by name, never by the label: the label carries the version, and
    saving v2 would otherwise make the choice fall back to the first program."""
    for n in (prefer, current):
        if n in names:
            return n
    return names[0] if names else None


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


def flagged_rows(res: BoardResult) -> list[tuple[str, str, str]]:
    """(part, state, detail) for everything on one board that needs an answer,
    board-level calls first: they say why the parts were not judged."""
    rows = [("board", c.state, c.detail) for c in res.board_calls if c.severity in record.NEEDS_ANSWER]
    for pr in res.parts:
        for c in pr.flagged:
            rows.append((pr.refdes, c.state, c.detail))
    return rows


def board_headline(res: BoardResult) -> str:
    """One line saying why a board got its verdict, for the big label over its photo."""
    board = [c.detail for c in res.board_calls if c.severity in record.NEEDS_ANSWER]
    if board:
        return board[0]
    n = sum(1 for pr in res.parts if pr.flagged)
    if n:
        return f"{n} part(s) flagged: {', '.join(pr.refdes for pr in res.parts if pr.flagged)[:120]}"
    if res.reasons:
        return res.reasons[0]
    return f"all {len(res.parts)} parts passed"


def error_text(e: BaseException) -> str:
    """An exception as a row note. The library's own errors read as sentences;
    anything else keeps its type, since 'error: 0' says nothing."""
    if isinstance(e, (io.PhotoError, OSError, ValueError)):
        return str(e) or type(e).__name__
    return f"{type(e).__name__}: {e}"


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
    """'confidence 0.97 (uncalibrated)' until enough answers calibrate the
    check, 'P=0.97 (calibrated)' after. An uncalibrated score is never shown
    as a probability."""
    if c["score"] is None:
        return ""
    if c["calibrated"]:
        return f"P={c['score']:.2f} (calibrated)"
    return f"confidence {c['score']:.2f} (uncalibrated)"


def call_row(c: dict) -> tuple[str, str, str, str]:
    """(answer, part, problem, score) for the Results call list."""
    score = score_text(c)
    if c["state"] in UNVALIDATED:
        score = (score + "  unvalidated").strip()
    return (ANSWER_TEXT.get(c["adjudication"] or "", ""), c["refdes"] or "board", c["state"], score)


def call_summary(c: dict) -> str:
    head = " ".join(s for s in (c["refdes"] or "whole board", c["state"]) if s)
    lines = [head]
    if score_text(c):
        lines.append(score_text(c) + ("  (unvalidated check)" if c["state"] in UNVALIDATED else ""))
    elif c["state"] in UNVALIDATED:
        lines.append("(unvalidated check)")
    lines.append(f"{c['check']}: {c['detail'] or '-'}")
    return "\n".join(lines)


def review_progress(machine: str, calls: list[dict]) -> str:
    """'machine says FAIL · you have answered 0 of 5'."""
    done = sum(1 for c in calls if c["adjudication"] is not None)
    if not calls:
        return f"machine says {machine} · nothing on this board needs an answer"
    return f"machine says {machine} · you have answered {done} of {len(calls)}"


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


Geometry = dict[str, tuple[Part, float, float]]     # refdes -> (taught box, dx, dy) in golden-1 px


def result_geometry(res: BoardResult) -> Geometry:
    return {pr.refdes: (pr.part, pr.dx_px, pr.dy_px) for pr in res.parts}


def run_geometry(row: dict, programs_root: str | Path) -> tuple[program.Program | None, Geometry, str]:
    """For a run this session did not inspect: every part's taught box from
    the program version on disk, and its measured shift from the run's CSV
    report (zero where the CSV is gone). (program, geometry, why-not)."""
    folder = Path(programs_root) / row["program"] / f"v{row['version']}"
    try:
        prog = program.Program.load(folder)
    except (io.PhotoError, OSError, ValueError, KeyError, TypeError) as e:
        return None, {}, f"program {row['program']} v{row['version']} is not under {programs_root}: {e}"
    shifts: dict[str, tuple[float, float]] = {}
    if row.get("annotated"):
        path = Path(row["annotated"]).with_name(f"{Path(row['image']).stem}.csv")
        try:
            with path.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("refdes") and r.get("dx_px"):
                        shifts[r["refdes"]] = (float(r["dx_px"]), float(r["dy_px"]))
        except (OSError, ValueError, KeyError, csv.Error):
            pass
    return prog, {p.refdes: (p, *shifts.get(p.refdes, (0.0, 0.0))) for p in prog.parts}, ""


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
# them touches Tk, and each catches Exception itself: a message that never
# arrives leaves a row saying "checking..." forever.
def check_goldens_worker(q: queue.Queue, gen: int, paths: list[Path]) -> None:
    for p in paths:
        try:
            problems = gate.golden_problems(io.load_bgr(p))
        except Exception as e:                    # noqa: BLE001 - any failure is this golden's problem
            problems = [error_text(e)]
        q.put(("golden_checked", gen, str(p), problems))


def load_view_worker(q: queue.Queue, kind: str, key, path: Path) -> None:
    try:
        bgr = io.load_bgr(path)
        small, k = downscale(bgr)
    except Exception as e:                        # noqa: BLE001
        q.put((kind, key, None, None, 1.0, f"{Path(path).name}: {error_text(e)}"))
        return
    q.put((kind, key, bgr, small, k, ""))


def propose_worker(q: queue.Queue, gen: int, bgr: np.ndarray) -> None:
    try:
        from aoi.photo import detect
        rects = teach.propose_boxes(bgr)
        from_detector = detect.PartDetector.available()
    except Exception as e:                        # noqa: BLE001
        q.put(("proposals", gen, [], True, error_text(e)))
        return
    q.put(("proposals", gen, rects, from_detector, ""))


def names_worker(q: queue.Queue, gen: int, bgr: np.ndarray, rects) -> None:
    """The board's own silkscreen names for the found boxes (names.py)."""
    try:
        res = names.name_boxes(bgr, [tuple(int(v) for v in r) for r in rects])
    except Exception as e:                        # noqa: BLE001
        q.put(("names", gen, rects, [None] * len(rects), error_text(e)))
        return
    q.put(("names", gen, rects, res.names, res.note, res.unsure))


def retrain_worker(q: queue.Queue, programs_root: Path) -> None:
    """Rebuild the part detector from every saved program and labelled shoot."""
    from aoi.photo import detect_train
    try:
        data = detect_train.collect(programs_root)
        n = sum(d.source != "cc0" for d in data)
        q.put(("retrain_log", f"training the part detector on {n} labelled photo(s)..."))
        out = detect_train.train(data, log=lambda m: q.put(("retrain_log", f"detector: {m}")))
    except Exception as e:                        # noqa: BLE001
        q.put(("retrain_log", f"detector training failed: {error_text(e)}"))
        return
    q.put(("retrain_log", f"part detector saved to {out}; Find parts now uses it"))


def teach_worker(q: queue.Queue, gen: int, name: str, goldens: list[Path], parts: list[Part],
                 px_per_mm: float | None, board_width_mm: float | None, single_golden: bool,
                 root: Path, accept: list[str]) -> None:
    try:
        rep = teach.build_program(name, goldens, parts, px_per_mm=px_per_mm,
                                  board_width_mm=board_width_mm, root=root,
                                  single_golden=single_golden, accept_selfcheck=list(accept))
    except teach.SelfCheckError as e:            # before PhotoError: it is one
        q.put(("teach_selfcheck", gen, e.flags))
    except Exception as e:                        # noqa: BLE001
        q.put(("teach_error", gen, error_text(e)))
    else:
        q.put(("teach_ok", gen, rep))


def _photo_or_blank(path: Path, like: np.ndarray) -> np.ndarray:
    """The test photo as shot, for a board that could not be aligned: its report
    should show what was photographed, not a black frame."""
    try:
        return io.load_bgr(path)
    except Exception:                             # noqa: BLE001
        return np.zeros_like(like)


# build the picked program's golden in the background (off in the test suite)
WARM_GOLDENS = os.environ.get("AOI_WARM_GOLDENS", "1") != "0"


def warm_golden(prog: program.Program) -> None:
    """Build and warm the program's golden before Run is clicked, so the first
    board does not pay for it. A failure here is left for the run to report."""
    try:
        photo_inspect.golden_for(prog)
    except Exception:                             # noqa: BLE001
        pass


def inspect_worker(q: queue.Queue, prog: program.Program, photos: list[Path], db: Path,
                   out_dir: Path) -> None:
    """`python -m aoi_photo inspect`'s loop, one queue message per board.

    Three stages overlap: the next photo loads (inspect.prefetch) while this one
    is inspected, and the last one's record and report are written on a second
    thread meanwhile. Messages still leave in photo order, since every one goes
    through that second thread.

    One bad photo never stops the batch: whatever it raises goes into its own
    row and the loop moves on."""
    try:
        golden = photo_inspect.golden_for(prog)   # kept between runs (and warmed when the program was picked)
    except Exception as e:                        # noqa: BLE001
        q.put(("inspect_fatal", f"golden 1 of {prog.name} v{prog.version}: {error_text(e)}"))
        return
    q.put(("inspect_golden", (prog.name, prog.version), golden.bgr))
    try:
        rec = record.Record(db)
    except Exception as e:                        # noqa: BLE001
        q.put(("inspect_fatal", f"{db}: {error_text(e)}"))
        return
    with rec:
        try:
            pv = rec.program_version(prog.name, prog.version, prog.path, len(prog.goldens),
                                     prog.px_per_mm, prog.single_golden)
            calib = build_calibrators(rec)
        except Exception as e:                    # noqa: BLE001
            q.put(("inspect_fatal", error_text(e)))
            return

    held: dict[str, record.Record] = {}           # the writer thread's own connection

    def failed(i: int, e: Exception) -> None:
        if not isinstance(e, (io.PhotoError, OSError, ValueError)):
            traceback.print_exception(e)
        q.put(("inspect_error", i, error_text(e)))

    def finish(i: int, f: Path, res: BoardResult, aligned: np.ndarray | None, t0: float) -> None:
        try:
            rec = held.get("rec") or held.setdefault("rec", record.Record(db))
            run_id = rec.save_run(res, pv)
            # No parts judged (could not align, wrong distance): the report
            # shows the photo as shot, with the reason on it.
            usable = aligned if aligned is not None and res.parts else None
            shown = usable if usable is not None else _photo_or_blank(Path(f), golden.bgr)
            paths = report.write_all(res, golden.bgr, shown, out_dir)
            rec.set_report(run_id, paths["annotated"], paths["html"])
        except Exception as e:                    # noqa: BLE001
            failed(i, e)
            return
        q.put(("inspect_row", i, run_id, res, paths, usable, time.perf_counter() - t0))

    def close() -> None:
        if "rec" in held:
            held.pop("rec").close()

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1, thread_name_prefix="aoi-report") as writer:
        try:
            for i, (f, load) in enumerate(photo_inspect.prefetch(photos)):
                t0 = time.perf_counter()
                try:
                    res, aligned = photo_inspect.inspect(prog, f, golden, calib, bgr=load())
                except Exception as e:            # noqa: BLE001
                    writer.submit(failed, i, e)
                    continue
                writer.submit(finish, i, f, res, aligned, t0)
        finally:
            writer.submit(close)                  # sqlite: closed on the thread that opened it


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

    def show_box(self, x: float, y: float, w: float, h: float, margin: float = 0.8) -> None:
        if self.full is None:
            return
        if not self._fitted:
            self.fit()
        cw, ch = self.size()
        self.view.show_box(x, y, w, h, cw, ch, margin)
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
        # A count per kind of work, not a set: a save that hits the self-check
        # runs a second teach worker before the first one's worker_done arrives,
        # and a set would forget the second one was still running.
        self.busy: Counter = Counter()

        # Set-up tab
        self.boxes = BoxList()
        self.golden_paths: list[Path] = []
        self.golden_status: dict[str, list[str] | None] = {}
        self.editable = False
        self.base_info: ProgramInfo | None = None
        self.program_infos: list[ProgramInfo] = []
        self.teach_gen = 0
        self.accepted: list[str] = []
        self._teach_args: tuple | None = None
        self._auto_name = ""
        self._saved_note = ""
        self._loading_fields = False
        self._band: tuple[float, float] | None = None
        self._ed_mode: str | None = None
        self._selfcheck_top: tk.Toplevel | None = None

        # Inspect tab
        self.inspect_photos: list[Path] = []
        self.inspect_runs: dict[str, int] = {}
        self.inspect_errors: dict[str, str] = {}
        self.inspect_name: str | None = None
        self.inspect_out: Path | None = None
        self.inspect_done = 0
        self.inspect_shown: str | None = None
        self._auto_iid: str | None = None
        self._auto_pick = False
        self._inspect_fatal = ""

        # Results tab
        self.session: dict[int, SessionRun] = {}
        self.goldens: dict[tuple[str, int], np.ndarray] = {}
        self._golden_loading: set[tuple[str, int]] = set()
        self.aligned: OrderedDict[int, np.ndarray] = OrderedDict()
        self.run_rows: list[dict] = []
        self.run_id: int | None = None
        self.run_row: dict | None = None
        self.geo: Geometry = {}
        self.golden_key: tuple[str, int] | None = None
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
            "names": self._on_names,
            "retrain_log": lambda msg: self.status(msg, error="failed" in msg),
            "teach_ok": self._on_teach_ok,
            "teach_error": self._on_teach_error,
            "teach_selfcheck": self._on_teach_selfcheck,
            "inspect_golden": self._on_inspect_golden,
            "inspect_row": self._on_inspect_row,
            "inspect_error": self._on_inspect_error,
            "inspect_fatal": self._on_inspect_fatal,
            "inspect_image": self._on_inspect_image,
            "review_image": self._on_review_image,
            "review_golden": self._on_review_golden,
        }
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<KeyPress>", self._on_key, add="+")
        for seq in ("<Command-z>", "<Control-z>"):
            try:
                root.bind(seq, self._on_undo_key, add="+")
            except tk.TclError:                   # no Command modifier off macOS
                pass
        self._poll_id = root.after(POLL_MS, self._poll)
        self.refresh_programs()
        self.refresh_runs()
        self.refresh_stats()
        self.new_program()                     # open ready to set up: upload is the first step
        self.status(f"programs {self.programs_root}   runs {self.runs_root}   record {self.db_path}")

    # ------------------------------------------------------------ plumbing
    def _fonts(self) -> None:
        r = self.root
        # Helvetica and Menlo on a Mac; elsewhere (the Pi) Tk's own default and
        # fixed families, since a missing family silently becomes a bitmap font
        sans, mono = (("Helvetica", "Menlo") if sys.platform == "darwin" else
                      (tkfont.nametofont("TkDefaultFont", root=r).actual("family"),
                       tkfont.nametofont("TkFixedFont", root=r).actual("family")))
        self.f_head = tkfont.Font(root=r, family=sans, size=16, weight="bold")
        self.f_big = tkfont.Font(root=r, family=sans, size=20, weight="bold")
        self.f_instr = tkfont.Font(root=r, family=sans, size=14, weight="bold")
        self.f_lab = tkfont.Font(root=r, family=sans, size=12)
        self.f_mono = tkfont.Font(root=r, family=mono, size=12)
        self.f_small = tkfont.Font(root=r, family=mono, size=10)
        self.f_bold = tkfont.Font(root=r, family=sans, size=12, weight="bold")

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
        for name, bg, active in (("Accent", FALSE, "#5cc488"), ("Defect", DEFECT, "#e05555"),
                                 ("Review", AMBER, "#ebb850")):
            st.configure(f"{name}.TButton", background=bg, foreground="#10131a")
            st.map(f"{name}.TButton", background=[("disabled", RULE), ("active", active)],
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

    def _focused(self):
        """The widget with keyboard focus. focus_get is None while another app
        is in front; focus_lastfor still names the widget that will have it."""
        try:
            return self.root.focus_get() or self.root.focus_lastfor()
        except (KeyError, tk.TclError):           # a combobox popdown has focus
            return None

    def _typing(self) -> bool:
        return isinstance(self._focused(), (tk.Entry, ttk.Entry, tk.Text))

    def _spawn(self, name: str, fn, *args) -> None:
        """Run fn(q, *args) on a thread. worker_done always follows, with the
        error text if fn raised something it did not catch itself."""
        self.busy[name] += 1

        def run() -> None:
            err = None
            try:
                fn(self.q, *args)
            except BaseException as e:            # noqa: BLE001 - reported, never swallowed silently
                traceback.print_exc()
                err = error_text(e)
            finally:
                self.q.put(("worker_done", name, err))

        threading.Thread(target=run, name=f"aoi-{name}", daemon=True).start()

    def _busy_done(self, name: str) -> None:
        self.busy[name] -= 1
        if self.busy[name] <= 0:
            del self.busy[name]

    def _poll(self) -> None:
        """Drain the queue. A handler that raises is reported and the loop goes
        on: an exception here would otherwise stop polling, and every later row
        would freeze."""
        try:
            while True:
                msg = self.q.get_nowait()
                try:
                    self._handlers[msg[0]](*msg[1:])
                except Exception as e:            # noqa: BLE001
                    traceback.print_exc()
                    self.status(f"{msg[0]}: {error_text(e)}", error=True)
        except queue.Empty:
            pass
        finally:
            self._poll_id = self.root.after(POLL_MS, self._poll)

    def _on_worker_done(self, name: str, err: str | None = None) -> None:
        self._busy_done(name)
        if err:
            self.status(f"{name} stopped: {err}", error=True)
        if name == "teach":
            self._sync_setup()
        elif name == "inspect":
            self._inspect_finished()
        elif name in ("propose", "names"):
            self._sync_setup()

    def on_close(self) -> None:
        working = sorted(set(self.busy) & set(LONG_WORK))
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
        self._build_menu()
        self.status_bar = tk.Label(self.root, text="", fg=DIM, bg=PANEL, font=self.f_small,
                                   anchor="w", padx=12, pady=5)
        self.status_bar.pack(side="bottom", fill="x")
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.tab_setup = tk.Frame(self.nb, bg=BG)
        self.tab_inspect = tk.Frame(self.nb, bg=BG)
        self.tab_results = tk.Frame(self.nb, bg=BG)
        for tab, text in ((self.tab_setup, "1  Set up board"), (self.tab_inspect, "2  Inspect boards"),
                          (self.tab_results, "3  Results")):
            self.nb.add(tab, text=text)
        self._build_setup_tab(self.tab_setup)
        self._build_inspect_tab(self.tab_inspect)
        self._build_results_tab(self.tab_results)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)

    def _build_menu(self) -> None:
        """Things a first-time user never needs sit here, out of the way."""
        bar = tk.Menu(self.root)
        adv = tk.Menu(bar, tearoff=0)
        adv.add_command(label="Import parts.csv...", command=self.import_parts)
        adv.add_command(label="Export parts.csv...", command=self.export_parts)
        adv.add_separator()
        self.v_scale_mode = tk.StringVar(master=self.root, value="board_mm")
        adv.add_checkbutton(label="Enter the scale as px per mm", onvalue="px_per_mm",
                            offvalue="board_mm", variable=self.v_scale_mode,
                            command=self._on_scale_mode)
        adv.add_command(label="Refresh program list", command=self.refresh_programs)
        adv.add_separator()
        adv.add_command(label="Retrain part detector (about 45 minutes)...", command=self.retrain)
        bar.add_cascade(label="Advanced", menu=adv)
        self.menu_advanced = adv
        self.root.configure(menu=bar)

    def _instruction(self, tab: tk.Frame) -> tk.Label:
        lab = self._label(tab, "", font=self.f_instr, wraplength=1300)
        lab.pack(fill="x", padx=10, pady=(8, 2))
        return lab

    def _tree(self, parent, columns, height: int = 8, stretch_last: bool = True) -> ttk.Treeview:
        tv = ttk.Treeview(parent, columns=[c for c, _, _ in columns], show="headings",
                          height=height, selectmode="browse")
        for i, (c, text, width) in enumerate(columns):
            tv.heading(c, text=text, anchor="w")
            tv.column(c, width=width, anchor="w", stretch=stretch_last and i == len(columns) - 1)
        return tv

    def _on_tab(self, _e=None) -> None:
        tab = self.nb.select()
        if tab == str(self.tab_results):
            self.refresh_runs()
            self._on_results_sub()
        elif tab == str(self.tab_inspect):
            self._refresh_inspect_programs()
            self._sync_inspect()

    # ========================================================= Set-up tab
    def _build_setup_tab(self, tab: tk.Frame) -> None:
        self.l_sinstr = self._instruction(tab)
        self.l_banner = self._label(tab, "", fg=AMBER, font=self.f_bold)
        self.l_banner.pack(fill="x", padx=10)
        body = tk.Frame(tab, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=BG, width=330)
        left.pack(side="left", fill="y", padx=(8, 6), pady=6)
        left.pack_propagate(False)
        right = tk.Frame(body, bg=BG, width=330)
        right.pack(side="right", fill="y", padx=(6, 8), pady=6)
        right.pack_propagate(False)
        centre = tk.Frame(body, bg=BG)
        centre.pack(side="left", fill="both", expand=True, pady=6)

        # left: programs, then goldens
        self._label(left, "Saved programs", font=self.f_head).pack(fill="x")
        self.t_programs = self._tree(left, (("name", "name", 150), ("v", "v", 34),
                                            ("g", "goldens", 60), ("note", "", 80)), height=5)
        self.t_programs.pack(fill="x", pady=(4, 4))
        self.t_programs.bind("<<TreeviewSelect>>", self._on_program_select)
        row = tk.Frame(left, bg=BG)
        row.pack(fill="x")
        self.b_new = ttk.Button(row, text="New program", command=self.new_program)
        self.b_new.pack(side="left")
        self.b_newver = ttk.Button(row, text="New version from this", command=self.new_version)
        self.b_newver.pack(side="left", padx=4)

        self._label(left, "Golden photos", font=self.f_head).pack(fill="x", pady=(14, 0))
        self.l_golden_count = self._label(left, "", fg=DIM, wraplength=320)
        self.l_golden_count.pack(fill="x")
        self.t_goldens = self._tree(left, (("n", "#", 30), ("file", "file", 120), ("check", "check", 160)),
                                    height=4)
        self.t_goldens.tag_configure("bad", foreground=DEFECT)
        self.t_goldens.tag_configure("ok", foreground=FALSE)
        self.t_goldens.tag_configure("wait", foreground=DIM)
        self.t_goldens.pack(fill="both", expand=True, pady=4)
        row = tk.Frame(left, bg=BG)
        row.pack(fill="x")
        # Clickable whenever nothing is saving: from a read-only program it starts
        # a new version, so the one thing a first-time user looks for is never grey.
        self.b_add_golden = ttk.Button(row, text="Upload golden photos...", style="Accent.TButton",
                                       command=self.add_goldens)
        self.b_add_golden.pack(side="left")
        self.b_rm_golden = ttk.Button(row, text="Remove", command=self.remove_golden)
        self.b_rm_golden.pack(side="left", padx=4)
        self._label(left, "Golden 1 carries the part boxes; every other photo is aligned onto it.",
                    fg=DIM, font=self.f_small, wraplength=320).pack(fill="x", pady=(6, 0))

        # centre: photo, then the parts list
        bar = tk.Frame(centre, bg=BG)
        bar.pack(fill="x", pady=(0, 4))
        self.b_propose = ttk.Button(bar, text="Find parts again", command=self.propose)
        self.b_propose.pack(side="left")
        ttk.Button(bar, text="Fit", command=lambda: self.ed.fit()).pack(side="left", padx=4)
        self.l_mode = self._label(bar, "", fg=AMBER)
        self.l_mode.pack(side="left", padx=10)
        self.ed = ImageView(centre, overlay=self._draw_boxes, empty=EDITOR_EMPTY)
        self.ed.canvas.pack(fill="both", expand=True)
        self._label(centre, "drag on empty board: add a part   click a part: select it   "
                            "Delete (photo focused): remove   Cmd/Ctrl+Z: undo   "
                            "wheel or + -: zoom   right-drag: pan   0: fit",
                    fg=DIM, font=self.f_small).pack(fill="x", pady=(4, 2))
        c = self.ed.canvas
        c.bind("<ButtonPress-1>", self._ed_press)
        c.bind("<B1-Motion>", self._ed_drag)
        c.bind("<ButtonRelease-1>", self._ed_release)
        c.bind("<KeyPress-Delete>", self._ed_delete)
        c.bind("<KeyPress-BackSpace>", self._ed_delete)

        plist = tk.Frame(centre, bg=BG)
        plist.pack(fill="x")
        self.t_parts = self._tree(plist, (("name", "part", 90), ("type", "type", 90),
                                          ("pol", "polarized", 130), ("size", "size (px)", 90)),
                                   height=6)
        self.t_parts.tag_configure("pol", foreground=AMBER)
        sb = ttk.Scrollbar(plist, orient="vertical", command=self.t_parts.yview)
        self.t_parts.configure(yscrollcommand=sb.set)
        btns = tk.Frame(plist, bg=BG)
        btns.pack(side="right", fill="y", padx=(6, 0))
        sb.pack(side="right", fill="y")
        self.t_parts.pack(side="left", fill="x", expand=True)
        self.t_parts.bind("<<TreeviewSelect>>", self._on_part_row)
        self.b_add_part = ttk.Button(btns, text="Add part", command=self.add_part_mode)
        self.b_add_part.pack(fill="x")
        self.b_del_part = ttk.Button(btns, text="Delete part", command=self.delete_part)
        self.b_del_part.pack(fill="x", pady=4)
        self.b_rename = ttk.Button(btns, text="Rename", command=self.rename_part)
        self.b_rename.pack(fill="x")
        self.b_undo = ttk.Button(btns, text="Undo", command=self.undo_box)
        self.b_undo.pack(fill="x", pady=4)

        # right: name, scale, checklist + Save, then the selected part
        self._label(right, "Program name").pack(fill="x")
        self.v_name = tk.StringVar(master=self.root)
        self.e_name = ttk.Entry(right, textvariable=self.v_name)
        self.e_name.pack(fill="x")
        self.l_scale = self._label(right, SCALE_LABELS["board_mm"], wraplength=320)
        self.l_scale.pack(fill="x", pady=(8, 0))
        self.v_scale = tk.StringVar(master=self.root)
        self.e_scale = ttk.Entry(right, textvariable=self.v_scale)
        self.e_scale.pack(fill="x")
        self.v_name.trace_add("write", lambda *_: self._sync_setup())
        self.v_scale.trace_add("write", lambda *_: self._sync_setup())

        self._label(right, "Before you save", font=self.f_head).pack(fill="x", pady=(8, 0))
        self.checklist = tk.Frame(right, bg=BG)
        self.checklist.pack(fill="x")
        self.check_rows: list[tk.Label] = []
        for _ in range(5):
            lab = self._label(self.checklist, "", wraplength=310)
            lab.pack(fill="x")
            self.check_rows.append(lab)
        self.b_save = ttk.Button(right, text="Save program", style="Accent.TButton",
                                 command=self.save_program)
        self.b_save.pack(fill="x", pady=(8, 0))
        self.l_save_reason = self._label(right, "", fg=DIM, font=self.f_small, wraplength=310)
        self.l_save_reason.pack(fill="x")

        self._label(right, "Selected part", font=self.f_head).pack(fill="x", pady=(10, 0))
        self._label(right, "name (optional, e.g. R12)").pack(fill="x")
        self.v_refdes = tk.StringVar(master=self.root)
        self.e_refdes = ttk.Entry(right, textvariable=self.v_refdes)
        self.e_refdes.pack(fill="x")
        row = tk.Frame(right, bg=BG)
        row.pack(fill="x", pady=(4, 0))
        self._label(row, "type (optional)").pack(side="left")
        self.v_type = tk.StringVar(master=self.root)
        self.c_type = ttk.Combobox(row, textvariable=self.v_type, values=PART_TYPES, state="readonly",
                                   width=12)
        self.c_type.pack(side="right")

        row = tk.Frame(right, bg=BG)
        row.pack(fill="x", pady=(10, 0))
        self._label(row, "Mark polarized parts", font=self.f_bold).pack(side="left")
        self.l_polcount = self._label(row, "", fg=DIM)
        self.l_polcount.pack(side="right")
        self._label(right, "Only parts that can go in backwards. Pick what the selected part is; "
                           "golden 1 shows the right way round, so no end needs marking.",
                    fg=DIM, font=self.f_small, wraplength=310).pack(fill="x")
        self.v_pol = tk.StringVar(master=self.root, value=POL_NO)
        self.c_pol = ttk.Combobox(right, textvariable=self.v_pol, values=POLARITY_CHOICES,
                                  state="readonly")
        self.c_pol.pack(fill="x", pady=(2, 0))

        self.v_refdes.trace_add("write", lambda *_: self._on_refdes_change())
        self.c_type.bind("<<ComboboxSelected>>", lambda e: self._on_type_change())
        self.c_pol.bind("<<ComboboxSelected>>", lambda e: self._on_polarity_change())
        # Return names the next part: the naming run stays in the entry and
        # never hands the keyboard to the canvas, where BackSpace deletes.
        self.e_refdes.bind("<Return>", lambda e: self._next_part())
        self.e_refdes.bind("<Escape>", lambda e: self.ed.canvas.focus_set())

    # -- program list
    def refresh_programs(self, prefer: str | None = None) -> None:
        self.program_infos = list_programs(self.programs_root)
        tv = self.t_programs
        tv.delete(*tv.get_children())
        for info in self.program_infos:
            tv.insert("", "end", iid=info.name, values=(
                info.name, info.version or "", info.n_goldens or "", info.note))
        self._refresh_inspect_programs(prefer)

    def _info(self, name: str) -> ProgramInfo | None:
        return next((i for i in self.program_infos if i.name == name), None)

    def _selected_program(self) -> ProgramInfo | None:
        sel = self.t_programs.selection()
        return self._info(sel[0]) if sel else None

    def _saving(self) -> bool:
        return "teach" in self.busy

    def _can_edit(self) -> bool:
        """Boxes and fields change only on a program being set up, and not while
        it is being saved: the save already took its copy."""
        return self.editable and not self._saving()

    def _dirty(self) -> bool:
        return self.editable and (len(self.boxes) > 0 or bool(self.golden_paths))

    def _confirm_discard(self) -> bool:
        return not self._dirty() or messagebox.askyesno(
            "Photo AOI", "Discard the program being set up?", parent=self.root)

    def _on_program_select(self, _e=None) -> None:
        if self._saving():
            # A save finishing later would replace whatever was opened here.
            self._show_base_selection()
            self.status("wait for the save to finish", error=True)
            return
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
            self._show_base_selection()
            return
        self._view_program(info)

    def _show_base_selection(self) -> None:
        b = self.base_info
        want = (b.name,) if b is not None and b.name in self.t_programs.get_children() else ()
        if tuple(self.t_programs.selection()) != want:
            self.t_programs.selection_set(want)

    def _view_program(self, info: ProgramInfo) -> None:
        p = info.program
        self._reset_teach(False, p.name, p.golden_paths(), BoxList.from_parts(p.parts),
                          f"{p.px_per_mm:g}" if p.px_per_mm else "", "px_per_mm", info)
        self.l_mode.configure(text=f"viewing {p.name} v{p.version} (read-only)", fg=DIM)

    def new_program(self) -> None:
        if self._saving() or not self._confirm_discard():
            return
        self._reset_teach(True, "", [], BoxList(), "", "board_mm", None)
        self.t_programs.selection_set(())
        self.l_mode.configure(text="new program", fg=AMBER)

    def new_version(self) -> None:
        if self._saving():
            return
        info = self._selected_program() or self.base_info
        if info is None or info.program is None:
            self.status("pick a saved program on the left first", error=True)
            return
        if self.editable and not self._confirm_discard():
            return
        p = info.program
        self._reset_teach(True, p.name, p.golden_paths(), BoxList.from_parts(p.parts),
                          f"{p.px_per_mm:g}" if p.px_per_mm else "", "px_per_mm", info)
        self.l_mode.configure(text=f"new version of {p.name} (saves as v{p.version + 1})", fg=AMBER)

    def _reset_teach(self, editable: bool, name: str, goldens: list[Path], boxes: BoxList,
                     scale: str, scale_mode: str, base: ProgramInfo | None) -> None:
        self.teach_gen += 1
        self.editable = editable
        self.base_info = base
        self.accepted = []
        self._auto_name = ""
        self._saved_note = ""
        self.boxes = boxes
        self.golden_paths = []
        self.golden_status = {}
        self.v_scale_mode.set(scale_mode)
        self.v_scale.set(scale)
        self.v_name.set(name)
        self.ed.clear("loading golden 1..." if goldens else EDITOR_EMPTY)
        self._add_golden_paths(list(goldens), check=editable)
        self._on_scale_mode()
        self._refresh_parts()
        self._load_fields()
        self._sync_setup()

    def _on_scale_mode(self) -> None:
        self.l_scale.configure(text=SCALE_LABELS[self.v_scale_mode.get()])
        self._sync_setup()

    def _checklist(self) -> tuple[list[tuple[str, str]], str | None]:
        return save_checklist(self.v_name.get(), [self.golden_status.get(str(p)) for p in self.golden_paths],
                              self.boxes.problems(), len(self.boxes), self.boxes.polarized,
                              self.v_scale.get(), self.v_scale_mode.get(),
                              finding="propose" in self.busy or "names" in self.busy)

    def _sync_setup(self) -> None:
        """Every control, the checklist, the banner and the instruction, from
        the state. Called after anything on the tab changes."""
        if not hasattr(self, "b_undo"):
            return                                 # a trace firing while the tab is being built
        saving = self._saving()
        edit = "normal" if self._can_edit() else "disabled"
        for w in (self.b_rm_golden, self.e_name, self.e_scale, self.b_add_part, self.b_undo):
            w.configure(state=edit)
        self.b_propose.configure(state=edit if "propose" not in self.busy and self.ed.full is not None
                                 else "disabled")
        self.b_add_golden.configure(state="disabled" if saving else "normal")
        self.b_new.configure(state="disabled" if saving else "normal")
        self.b_newver.configure(state="disabled" if saving or (self._selected_program() or self.base_info)
                                is None else "normal")
        self.t_programs.configure(selectmode="none" if saving else "browse")
        for i in (0, 1):                           # Import / Export parts.csv
            self.menu_advanced.entryconfigure(i, state="normal" if (i == 1 or self._can_edit()) else "disabled")

        rows, reason = self._checklist()
        colours = {OK: FALSE, WARN: AMBER, TODO: DEFECT, WAIT: DIM, OPTIONAL: DIM}
        for lab, (state, text) in zip(self.check_rows, rows):
            lab.configure(text=f"{CHECK_MARKS[state]}  {text}", fg=colours[state])
        ready = self.editable and not saving and reason is None
        self.b_save.configure(state="normal" if ready else "disabled")
        if saving:
            why = "saving: aligning the goldens and measuring every part..."
        elif not self.editable:
            why = "read-only: \"New version from this\" to change it"
        else:
            why = f"Save needs: {reason}" if reason else "Ready to save: a few seconds for one golden, up to a minute for three."
        self.l_save_reason.configure(text=why, fg=DEFECT if reason and self.editable and not saving else DIM)
        self.l_banner.configure(text=single_golden_banner(len(self.golden_paths)) if self.editable else "")
        n = len(self.golden_paths)
        self.l_golden_count.configure(text=f"{n} golden photo(s). {GOLDEN_RULE}.")
        self.l_polcount.configure(text=f"marked: {self.boxes.polarized}")
        self.l_sinstr.configure(text=self._setup_instruction(reason), fg=INK)

    def _setup_instruction(self, reason: str | None) -> str:
        if self._saving():
            return "Saving: aligning the goldens and measuring every part (a few seconds per golden)."
        if not self.editable:
            b = self.base_info
            head = self._saved_note or (f"Viewing {b.name} v{b.version} (read-only)." if b else "")
            return head + " Next: tab 2 \"Inspect boards\". To change this program: \"New version from this\"."
        if not self.golden_paths:
            return "Step 1: click \"Upload golden photos...\" and pick 1 to 3 photos of a known-good board."
        if self.ed.full is None:
            return "Loading golden 1..."
        if "propose" in self.busy:
            return "Finding the parts on golden 1..."
        if "names" in self.busy:
            return "Reading every part's name (R12, C9, D4...) off the board's silkscreen..."
        if reason:
            return f"Before saving: {reason}."
        return ("Parts found. Delete any box that is not a part, mark polarized parts (optional), "
                "then click Save program.")

    # -- goldens
    def add_goldens(self) -> None:
        if self._saving():
            return
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
        paths = filedialog.askopenfilenames(parent=self.root, title="Golden photos (1 to 3, known-good)",
                                            filetypes=PHOTO_TYPES)
        if paths:
            self._add_golden_paths(sorted(Path(p) for p in paths), check=True)

    def _add_golden_paths(self, paths: list[Path], check: bool) -> None:
        had_first = self.golden_paths[0] if self.golden_paths else None
        new = [p for p in paths if p not in self.golden_paths]
        self.golden_paths += new
        for p in new:
            self.golden_status[str(p)] = None if check else []
        if check and new:
            self._spawn("golden check", check_goldens_worker, self.teach_gen, new)
        if self.golden_paths and self.golden_paths[0] != had_first:
            self._load_golden1()
        if self.editable and self.golden_paths:
            # the name follows the photos until somebody types their own
            if not self.v_name.get().strip() or self.v_name.get() == self._auto_name:
                taken = {i.name for i in self.program_infos}
                self._auto_name = default_program_name(self.golden_paths, taken)
                self.v_name.set(self._auto_name)
        self._refresh_goldens()

    def _load_golden1(self) -> None:
        self._spawn("load golden", load_view_worker, "editor_image",
                    (self.teach_gen, str(self.golden_paths[0])), self.golden_paths[0])

    def remove_golden(self) -> None:
        sel = self.t_goldens.selection()
        if not sel or not self._can_edit():
            return
        i = int(sel[0])
        if i == 0 and len(self.boxes):
            if not messagebox.askyesno(
                    "Remove golden 1", f"The {len(self.boxes)} part box(es) sit on golden 1 and are "
                    "cleared with it; the next photo becomes golden 1 and its parts are found again. "
                    "Remove golden 1?", parent=self.root):
                return
            self.boxes = BoxList()
        del self.golden_paths[i]
        # Anything still in flight was for the old golden list: proposals or an
        # image for the old golden 1 must not land on the new one.
        self.teach_gen += 1
        pending = [p for p in self.golden_paths if self.golden_status.get(str(p)) is None]
        if pending:
            self._spawn("golden check", check_goldens_worker, self.teach_gen, pending)
        if i == 0:
            self.ed.clear(EDITOR_EMPTY if not self.golden_paths else "loading golden 1...")
            if self.golden_paths:
                self._load_golden1()
        self._refresh_goldens()
        self._refresh_parts()
        self._load_fields()

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
        self._sync_setup()

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
            self._sync_setup()
            return
        self.ed.set_image(bgr, small, k)
        # Zero-marking set-up: golden 1 is here and nothing is boxed, so find
        # the parts now rather than wait for a button nobody knows to press.
        if self._can_edit() and not len(self.boxes):
            self.propose()
        self._sync_setup()

    # -- editor
    def _draw_boxes(self, c: tk.Canvas, view: View) -> None:
        for i, b in enumerate(self.boxes.boxes):
            x0, y0 = view.to_canvas(b.x, b.y)
            x1, y1 = view.to_canvas(b.x + b.w, b.y + b.h)
            sel = i == self.boxes.selected
            col = INK if sel else (AMBER if b.polarized else FALSE)
            c.create_rectangle(x0, y0, x1, y1, outline=col, width=3 if sel else 2)
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
        if i is not None or not self._can_edit():
            self._ed_mode = None
            self._select_box(i)
            if i is not None and self._can_edit():
                # typing a name straight after the click must land in the name field
                self.e_refdes.focus_set()
                self.e_refdes.select_range(0, "end")
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
        self.l_mode.configure(text="")
        self._select_box(self.boxes.add(*rect))
        self._refresh_parts()
        self.e_refdes.focus_set()

    def _ed_delete(self, _e=None) -> str | None:
        """Delete or BackSpace on the photo. Only there: the keys belong to an
        entry whenever one has the keyboard, and a name being corrected must
        never cost the part."""
        if self._typing() or self._focused() is not self.ed.canvas:
            return None
        self.delete_part()
        return "break"

    def delete_part(self) -> None:
        if not self._can_edit():
            return
        i = self.boxes.selected
        if i is None:
            self.status("select a part first: click it on the photo or in the list", error=True)
            return
        label = self.boxes.boxes[i].label
        if self.boxes.delete():
            self._after_boxes_changed()
            self.status(f"{label} deleted (Cmd/Ctrl+Z or Undo brings it back)")

    def undo_box(self) -> None:
        if not self._can_edit():
            return
        what = self.boxes.undo()
        if what is None:
            self.status("nothing to undo")
            return
        self._after_boxes_changed()
        self.status(f"undone: {what}")

    def _on_undo_key(self, _e=None) -> str | None:
        if self.nb.select() != str(self.tab_setup) or self._typing():
            return None                            # an entry's own editing is not a box undo
        self.undo_box()
        return "break"

    def add_part_mode(self) -> None:
        if not self._can_edit() or self.ed.full is None:
            return
        self.l_mode.configure(text="drag a box around the part on the photo", fg=AMBER)
        self.ed.canvas.focus_set()

    def rename_part(self) -> None:
        if self.boxes.selected is None or not self._can_edit():
            self.status("select a part first: click it on the photo or in the list", error=True)
            return
        self.e_refdes.focus_set()
        self.e_refdes.select_range(0, "end")

    def _next_part(self) -> str:
        if len(self.boxes):
            i = self.boxes.selected
            j = 0 if i is None else (i + 1) % len(self.boxes)
            self._select_box(j, zoom=True)
            self.e_refdes.select_range(0, "end")
        return "break"

    def _after_boxes_changed(self) -> None:
        self._refresh_parts()
        self._load_fields()
        self.ed.redraw()
        self._sync_setup()

    def _select_box(self, i: int | None, zoom: bool = False) -> None:
        self.boxes.select(i)
        self._load_fields()
        self._show_part_row()
        if zoom and i is not None:
            b = self.boxes.boxes[i]
            self.ed.show_box(b.x, b.y, b.w, b.h, margin=3.0)
        self.ed.redraw()

    def _part_values(self, b: Box) -> tuple:
        return (b.label, b.type, polarity_choice(b) if b.polarized else "", f"{b.w} x {b.h}")

    def _refresh_parts(self) -> None:
        tv = self.t_parts
        tv.delete(*tv.get_children())
        for i, b in enumerate(self.boxes.boxes):
            tv.insert("", "end", iid=str(i), values=self._part_values(b),
                      tags=("pol",) if b.polarized else ())
        self._show_part_row()

    def _show_part_row(self) -> None:
        i = self.boxes.selected
        want = (str(i),) if i is not None and str(i) in self.t_parts.get_children() else ()
        if tuple(self.t_parts.selection()) != want:
            self.t_parts.selection_set(want)
        if want:
            self.t_parts.see(want[0])

    def _on_part_row(self, _e=None) -> None:
        sel = self.t_parts.selection()
        i = int(sel[0]) if sel else None
        if i is None or i == self.boxes.selected or i >= len(self.boxes):
            return
        self._select_box(i, zoom=True)

    def _update_part_row(self, i: int) -> None:
        if str(i) in self.t_parts.get_children():
            b = self.boxes.boxes[i]
            self.t_parts.item(str(i), values=self._part_values(b), tags=("pol",) if b.polarized else ())

    def _load_fields(self) -> None:
        """Selected box -> side panel. The traces fire on set(); _loading_fields
        keeps them from writing the values straight back."""
        i = self.boxes.selected
        b = self.boxes.boxes[i] if i is not None and i < len(self.boxes) else None
        self._loading_fields = True
        try:
            self.v_refdes.set(b.refdes if b else "")
            self.v_type.set(b.type if b else "")
            self.v_pol.set(polarity_choice(b) if b else POL_NO)
        finally:
            self._loading_fields = False
        on = b is not None and self._can_edit()
        self.e_refdes.configure(state="normal" if on else "disabled")
        self.b_del_part.configure(state="normal" if on else "disabled")
        self.b_rename.configure(state="normal" if on else "disabled")
        for w in (self.c_type, self.c_pol):
            w.configure(state="readonly" if on else "disabled")

    def _on_refdes_change(self) -> None:
        i = self.boxes.selected
        if self._loading_fields or i is None or not self._can_edit():
            return
        self.boxes.rename(i, self.v_refdes.get())
        self._update_part_row(i)
        self.ed.redraw()
        self._sync_setup()

    def _on_type_change(self) -> None:
        i = self.boxes.selected
        if i is None or not self._can_edit():
            return
        try:
            self.boxes.set_fields(i, type=self.v_type.get())
        except ValueError as e:
            self.status(str(e), error=True)
        self._load_fields()
        self._update_part_row(i)

    def _on_polarity_change(self) -> None:
        i = self.boxes.selected
        if i is None or not self._can_edit():
            return
        b = self.boxes.boxes[i]
        self.boxes.set_fields(i, **polarity_fields(self.v_pol.get(), b.type))
        self._load_fields()
        self._update_part_row(i)
        self.ed.redraw()
        self._sync_setup()

    def propose(self) -> None:
        if self.ed.full is None:
            self.status("upload golden photos first: parts are found on golden 1", error=True)
            return
        if "propose" in self.busy or not self._can_edit():
            return
        self.status("finding parts on golden 1...")
        self._spawn("propose", propose_worker, self.teach_gen, self.ed.full)
        self._sync_setup()

    def _on_proposals(self, gen: int, rects, from_detector: bool = True, err: str = "") -> None:
        if gen != self.teach_gen or not self._can_edit():
            return
        if err:
            self.status(f"finding parts failed: {err}; draw boxes by hand", error=True)
            return
        added, skipped = self.boxes.add_proposals(rects)
        self._after_boxes_changed()
        note = ("" if from_detector else
                "  No trained part detector yet: these come from colour alone and include pads; "
                "delete the boxes that are not parts.")
        reading = names.available() and added
        self.status(f"{added} part(s) found"
                    + ("; reading their names off the board..." if reading else
                       ", named P1, P2, ...; renaming is optional")
                    + (f"; {skipped} already boxed" if skipped else "") + note,
                    error=not from_detector)
        if reading and self.ed.full is not None:
            self._spawn("names", names_worker, gen, self.ed.full, [tuple(r) for r in rects])
            self._sync_setup()

    def _on_names(self, gen: int, rects, found: list[str | None], note: str = "",
                  unsure: list[str] | None = None) -> None:
        if gen != self.teach_gen or not self._can_edit():
            return
        named, left = self.boxes.apply_names(rects, found)
        self._after_boxes_changed()
        pol = self.boxes.polarized
        msg = (f"{named} part(s) named from the board's silkscreen; {pol} polarized (from D, LED, BD, "
               f"Q, T, U names), checked automatically")
        if left:
            msg += f"; {left} without a readable name keep P-numbers: click one and type its name"
        check = [n for n in (unsure or []) if any(b.refdes == n for b in self.boxes.boxes)]
        if check:
            msg += "; check " + ", ".join(check) + " (close calls)"
        self.status(msg + (f" ({note})" if note else ""), error=bool(note))

    def retrain(self) -> None:
        if "retrain" in self.busy:
            self.status("the part detector is already training")
            return
        if not messagebox.askyesno(
                "Retrain part detector", "Retrain the part detector from every saved program and "
                "labelled photo?\n\nIt runs on this machine, offline, and takes roughly 45 minutes "
                "on an Apple-silicon Mac. You can keep working meanwhile.", parent=self.root):
            return
        self._spawn("retrain", retrain_worker, self.programs_root)

    def import_parts(self) -> None:
        if not self._can_edit():
            self.status("import parts into a program being set up (New program or New version)", error=True)
            return
        if self.ed.full is None:
            self.status("upload golden 1 first: imported boxes are checked against it", error=True)
            return
        path = filedialog.askopenfilename(parent=self.root, title="Import parts.csv",
                                          filetypes=[("CSV", "*.csv"), ("All files", "*")])
        if not path:
            return
        try:
            parts = program.read_parts_csv(path)
            boxes = BoxList.from_parts(parts)
        except (io.PhotoError, OSError, ValueError, KeyError, TypeError) as e:
            messagebox.showerror("Import parts.csv", str(e), parent=self.root)
            return
        H, W = self.ed.full.shape[:2]
        off = parts_outside(parts, W, H)
        if off:
            shown = ", ".join(off[:8]) + (", ..." if len(off) > 8 else "")
            messagebox.showerror(
                "Import parts.csv", f"{len(off)} of {len(parts)} part(s) lie outside golden 1 "
                f"({W} x {H} px): {shown}.\n\nThis parts.csv was made for a different photo; "
                "nothing was imported.", parent=self.root)
            return
        if len(self.boxes) and not messagebox.askyesno(
                "Import parts.csv", f"Replace the {len(self.boxes)} box(es) here with "
                f"{len(parts)} from {Path(path).name}?", parent=self.root):
            return
        self.boxes = boxes
        self._after_boxes_changed()
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
        if not self._can_edit():
            return
        _, reason = self._checklist()
        if reason:
            self.status(f"not saved yet: {reason}", error=True)
            return
        name = check_name(self.v_name.get())
        ppm, bw = parse_scale(self.v_scale_mode.get(), self.v_scale.get())
        goldens = list(self.golden_paths)
        single = len(goldens) < config.PHOTO_MIN_GOLDENS
        self._teach_args = (name, goldens, self.boxes.to_parts(), ppm, bw, single)
        self._start_teach()

    def _start_teach(self) -> None:
        name, goldens, parts, ppm, bw, single = self._teach_args
        self.status(f"saving {name}: checking {len(goldens)} golden(s), aligning them, "
                    "measuring every part...")
        self._spawn("teach", teach_worker, self.teach_gen, name, goldens, parts, ppm, bw, single,
                    self.programs_root, list(self.accepted))
        self._load_fields()
        self._sync_setup()

    def _on_teach_error(self, gen: int, msg: str) -> None:
        if gen != self.teach_gen:
            self.status(f"an earlier save failed: {msg}", error=True)
            return
        self.status(f"not saved: {msg}", error=True)
        dup = same_shot(msg)
        if dup is not None and dup[0] <= len(self.golden_paths):
            # Mark the photo itself, so the checklist holds Save until it is gone.
            i, j = dup
            self.golden_status[str(self.golden_paths[i - 1])] = [
                f"same shot as golden {j}: remove it, or pick the board up, put it back and reshoot"]
            self._refresh_goldens()
            msg += (f"\n\nGolden {i} is marked in the list: select it and click Remove. Two photos "
                    "of one unmoved board show no spread, so they count as one golden.")
        messagebox.showerror("Program not saved", msg, parent=self.root)

    def _on_teach_selfcheck(self, gen: int, flags: list[tuple[str, str]]) -> None:
        if gen != self.teach_gen:
            return
        # The dialog is part of the save: Save stays off and quitting still
        # warns until the operator answers it and the next save attempt ends.
        self.busy["teach"] += 1

        def done(accepted: list[str] | None) -> None:
            try:
                if accepted is None:
                    self.status("not saved: fix the golden, or tick every flagged part as intended",
                                error=True)
                elif gen == self.teach_gen:
                    self.accepted = sorted(set(self.accepted) | set(accepted))
                    self._start_teach()
            finally:
                self._busy_done("teach")
                self._load_fields()
                self._sync_setup()

        self._open_selfcheck(flags, done)

    def _open_selfcheck(self, flags: list[tuple[str, str]], done) -> None:
        """The self-check question, modal to the window but not to the poll
        loop: it returns at once and calls done(accepted or None) later, so
        rows keep arriving while it is open."""
        top = tk.Toplevel(self.root)
        self._selfcheck_top = top
        top.title("Golden self-check")
        top.configure(bg=BG, padx=18, pady=14)
        top.transient(self.root)
        self._label(top, "Golden 1 has parts that sit off their own silkscreen outline.",
                    font=self.f_head).pack(fill="x")
        self._label(top, "Fix the golden board and reshoot, or tick each one that is placed "
                         "that way on purpose. Saving needs every flag ticked.",
                    fg=DIM, wraplength=520).pack(fill="x", pady=(2, 10))
        ticks: dict[str, tk.BooleanVar] = {}
        every = tk.BooleanVar(master=top, value=False)

        def tick_all() -> None:
            for v in ticks.values():
                v.set(every.get())

        ttk.Checkbutton(top, text="all of them are intended", variable=every,
                        command=tick_all).pack(anchor="w", pady=(0, 6))
        for r, d in flags:
            row = tk.Frame(top, bg=BG)
            row.pack(fill="x", pady=1)
            ticks[r] = tk.BooleanVar(master=top, value=False)
            ttk.Checkbutton(row, text="intended", variable=ticks[r]).pack(side="left")
            self._label(row, f"{r}: {d}", font=self.f_mono).pack(side="left", padx=8)
        answered = {"done": False}

        def finish(result: list[str] | None) -> None:
            if answered["done"]:
                return
            answered["done"] = True
            self._selfcheck_top = None
            try:
                top.grab_release()
                top.destroy()
            except tk.TclError:
                pass
            done(result)

        row = tk.Frame(top, bg=BG)
        row.pack(fill="x", pady=(12, 0))
        ttk.Button(row, text="Save with these accepted", style="Accent.TButton",
                   command=lambda: finish(accepted_selfcheck(
                       flags, {r for r, v in ticks.items() if v.get()}))).pack(side="right")
        ttk.Button(row, text="Cancel", command=lambda: finish(None)).pack(side="right", padx=6)
        top.bind("<Escape>", lambda e: finish(None))
        top.protocol("WM_DELETE_WINDOW", lambda: finish(None))
        try:
            top.grab_set()
        except tk.TclError:                        # not viewable yet; still answerable
            pass

    def _on_teach_ok(self, gen: int, rep: teach.TeachReport) -> None:
        p = rep.program
        head = f"Saved {p.name} v{p.version}: {len(p.parts)} parts, {len(p.goldens)} golden(s)"
        if rep.selfcheck_flags:
            head += "; accepted as intended: " + ", ".join(r for r, _ in rep.selfcheck_flags)
        if gen != self.teach_gen:
            # The user moved on (not possible from the buttons, which are off
            # during a save, but never let a late save replace their work).
            self.refresh_programs()
            self.status(head)
            return
        self.refresh_programs(prefer=p.name)
        info = self._info(p.name)
        if info is not None and info.program is not None:
            self._view_program(info)
            self.t_programs.selection_set(info.name)
        self._saved_note = head + "."
        self.status(head)
        self._sync_setup()

    # ======================================================== Inspect tab
    def _build_inspect_tab(self, tab: tk.Frame) -> None:
        self.l_iinstr = self._instruction(tab)
        bar = tk.Frame(tab, bg=BG)
        bar.pack(fill="x", padx=10, pady=(4, 4))
        self._label(bar, "Program").pack(side="left")
        self.v_iprog = tk.StringVar(master=self.root)
        self.c_iprog = ttk.Combobox(bar, textvariable=self.v_iprog, state="readonly", width=48)
        self.c_iprog.pack(side="left", padx=(6, 12))
        self.c_iprog.bind("<<ComboboxSelected>>", lambda e: self._on_iprog())
        self.b_add_tests = ttk.Button(bar, text="Upload test photos...", style="Accent.TButton",
                                      command=self.add_tests)
        self.b_add_tests.pack(side="left")
        self.b_clear = ttk.Button(bar, text="Clear", command=self.clear_tests)
        self.b_clear.pack(side="left", padx=4)
        self.b_run = ttk.Button(bar, text="Run", style="Accent.TButton", command=self.run_inspect)
        self.b_run.pack(side="left", padx=(12, 0))
        self.pb = ttk.Progressbar(bar, mode="determinate", length=200)
        self.pb.pack(side="left", padx=12)
        self.l_iprogress = self._label(bar, "", fg=DIM, font=self.f_mono)
        self.l_iprogress.pack(side="left")
        self.l_istamp = self._label(tab, "", fg=AMBER)
        self.l_istamp.pack(fill="x", padx=10)

        body = tk.Frame(tab, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        left = tk.Frame(body, bg=BG, width=520)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._label(left, "Boards", font=self.f_head).pack(fill="x")
        self.t_inspect = self._tree(left, (("photo", "photo", 150), ("verdict", "verdict", 70),
                                           ("flagged", "flagged", 60), ("time", "time", 55),
                                           ("note", "note", 180)), height=12)
        for v in ("PASS", "FAIL", "REVIEW"):
            self.t_inspect.tag_configure(v, foreground=verdict_colour(v))
        self.t_inspect.tag_configure("ERROR", foreground=DEFECT)
        self.t_inspect.tag_configure("queued", foreground=DIM)
        self.t_inspect.pack(fill="both", expand=True, pady=4)
        self.t_inspect.bind("<<TreeviewSelect>>", self._on_board_select)
        self.t_inspect.bind("<Double-1>", lambda e: self.review_board())
        self.t_inspect.bind("<Return>", lambda e: self.review_board())
        self._label(left, "Flagged on this board", font=self.f_bold).pack(fill="x", pady=(6, 0))
        self.t_iflags = self._tree(left, (("part", "part", 70), ("state", "problem", 140),
                                          ("detail", "detail", 280)), height=7)
        self.t_iflags.pack(fill="x", pady=4)
        self.t_iflags.bind("<<TreeviewSelect>>", self._on_iflag_select)
        self.b_review = ttk.Button(left, text="Review calls →", style="Accent.TButton",
                                   command=self.review_board)
        self.b_review.pack(fill="x", pady=(4, 0))

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.l_iverdict = self._label(right, "", font=self.f_big, wraplength=800)
        self.l_iverdict.pack(fill="x")
        self.l_ireason = self._label(right, "", font=self.f_head, wraplength=800)
        self.l_ireason.pack(fill="x", pady=(0, 4))
        self.iv = ImageView(right, overlay=self._draw_iflag,
                            empty="Run a batch, then click a board to see its photo with the defects boxed.")
        self.iv.canvas.pack(fill="both", expand=True)
        self.iv.canvas.bind("<ButtonPress-1>", self.iv.pan_start)
        self.iv.canvas.bind("<B1-Motion>", self.iv.pan_move)
        self.iv.canvas.bind("<ButtonRelease-1>", self.iv.pan_end)

    def _refresh_inspect_programs(self, prefer: str | None = None) -> None:
        infos = [i for i in self.program_infos if i.program is not None]
        self._iprog_infos = {i.name: i for i in infos}
        self.c_iprog.configure(values=[i.label for i in infos])
        self.inspect_name = pick_program([i.name for i in infos], self.inspect_name, prefer)
        info = self._iprog_infos.get(self.inspect_name) if self.inspect_name else None
        self.v_iprog.set(info.label if info else "")
        self._on_iprog(keep=True)

    def _inspect_program(self) -> program.Program | None:
        info = getattr(self, "_iprog_infos", {}).get(self.inspect_name or "")
        return info.program if info else None

    def _on_iprog(self, keep: bool = False) -> None:
        if not keep:
            i = self.c_iprog.current()
            names = list(getattr(self, "_iprog_infos", {}))
            if 0 <= i < len(names):
                self.inspect_name = names[i]
        prog = self._inspect_program()
        self.l_istamp.configure(text=(f"every result from this program is stamped: {prog.stamp}"
                                      if prog is not None and prog.stamp else ""))
        self._sync_inspect()
        if prog is not None and WARM_GOLDENS:
            threading.Thread(target=warm_golden, args=(prog,), name="aoi-warm", daemon=True).start()

    def _sync_inspect(self) -> None:
        running = "inspect" in self.busy
        off = "disabled" if running else "normal"
        # Photos added mid-run would get rows the running batch never fills.
        for w in (self.b_add_tests, self.b_clear, self.b_run):
            w.configure(state=off)
        self.c_iprog.configure(state="disabled" if running else "readonly")
        sel = self.t_inspect.selection()
        self.b_review.configure(state="normal" if sel and sel[0] in self.inspect_runs else "disabled")
        self.l_iinstr.configure(text=self._inspect_instruction())

    def _inspect_instruction(self) -> str:
        if not self.program_infos or self._inspect_program() is None:
            return "Set up a board first (tab 1): nothing here can be inspected without a saved program."
        if "inspect" in self.busy:
            return f"Inspecting {self.inspect_done} of {len(self.inspect_photos)}... click a finished board to see it."
        if not self.inspect_photos:
            return "Upload test photos (one photo per board), then click Run."
        if not self.inspect_runs and not self.inspect_errors:
            return "Click Run to inspect these boards."
        return "Click a board to see its defects; \"Review calls →\" to answer them."

    def add_tests(self) -> None:
        if "inspect" in self.busy:
            self.status("wait for this run to finish, then add more photos", error=True)
            return
        paths = filedialog.askopenfilenames(parent=self.root, title="Test photos (one per board)",
                                            filetypes=PHOTO_TYPES)
        for p in sorted(Path(x) for x in paths):
            if p in self.inspect_photos:
                continue
            self.inspect_photos.append(p)
            self.t_inspect.insert("", "end", iid=str(len(self.inspect_photos) - 1),
                                  values=(p.name, "", "", "", "not run yet"), tags=("queued",))
        self.l_iprogress.configure(text=f"{len(self.inspect_photos)} photo(s)")
        self._sync_inspect()

    def clear_tests(self) -> None:
        if "inspect" in self.busy:
            return
        self.inspect_photos = []
        self.inspect_runs = {}
        self.inspect_errors = {}
        self.inspect_done = 0
        self.inspect_shown = None
        self.t_inspect.delete(*self.t_inspect.get_children())
        self.t_iflags.delete(*self.t_iflags.get_children())
        self.iv.clear("Run a batch, then click a board to see its photo with the defects boxed.")
        self.l_iverdict.configure(text="")
        self.l_ireason.configure(text="")
        self.pb.configure(value=0)
        self.l_iprogress.configure(text="")
        self._sync_inspect()

    def run_inspect(self) -> None:
        if "inspect" in self.busy:
            return
        prog = self._inspect_program()
        if prog is None:
            self.status("choose a program to inspect against (set one up on tab 1)", error=True)
            return
        if not self.inspect_photos:
            self.status("upload test photos first: one photo per board", error=True)
            return
        dups = duplicate_stems(self.inspect_photos)
        if dups:
            self.status("two photos share a name and their reports would overwrite each other: "
                        + ", ".join(dups) + "; rename one or run them separately", error=True)
            return
        self.inspect_out = new_run_dir(self.runs_root, prog.name)
        self.inspect_runs = {}
        self.inspect_errors = {}
        self.inspect_done = 0
        self._inspect_fatal = ""
        self._auto_pick = True
        for i, p in enumerate(self.inspect_photos):
            self.t_inspect.item(str(i), values=(p.name, "queued", "", "", ""), tags=("queued",))
        self.pb.configure(maximum=len(self.inspect_photos), value=0)
        self.l_iprogress.configure(text=f"0 / {len(self.inspect_photos)}")
        self.status(f"inspecting {len(self.inspect_photos)} board(s) against {prog.name} "
                    f"v{prog.version} -> {self.inspect_out}")
        self._spawn("inspect", inspect_worker, prog, list(self.inspect_photos), self.db_path,
                    self.inspect_out)
        self._sync_inspect()

    def _step_progress(self) -> None:
        # Set, never step(): a determinate bar wraps to 0 when it reaches maximum.
        self.inspect_done += 1
        n = len(self.inspect_photos)
        self.pb.configure(maximum=max(n, 1), value=min(self.inspect_done, n))
        self.l_iprogress.configure(text=f"{self.inspect_done} / {n}")
        self.l_iinstr.configure(text=self._inspect_instruction())

    def _on_inspect_golden(self, key: tuple[str, int], bgr: np.ndarray) -> None:
        self.goldens[key] = bgr

    def _on_inspect_row(self, i: int, run_id: int, res: BoardResult, paths: dict,
                        aligned: np.ndarray | None, secs: float) -> None:
        iid = str(i)
        note = "; ".join(s for s in (board_headline(res), res.stamp) if s)
        self.t_inspect.item(iid, values=(Path(res.image).name, res.verdict, flagged_count(res),
                                         f"{secs:.1f} s", note), tags=(res.verdict,))
        self.inspect_runs[iid] = run_id
        self.session[run_id] = SessionRun(res, paths, (res.program, res.program_version))
        if aligned is not None:
            self.aligned[run_id] = aligned
            while len(self.aligned) > ALIGNED_KEEP:
                self.aligned.popitem(last=False)
        self._step_progress()
        # The first finished board shows itself, so the first defect box is on
        # screen without a click; after that the operator's choice stands.
        if self._auto_pick and not self.t_inspect.selection():
            self._auto_pick = False
            self.t_inspect.selection_set(iid)
        elif self.t_inspect.selection() == (iid,):
            self._on_board_select()

    def _on_inspect_error(self, i: int, msg: str) -> None:
        p = self.inspect_photos[i]
        self.inspect_errors[str(i)] = msg
        self.t_inspect.item(str(i), values=(p.name, "ERROR", "", "", msg), tags=("ERROR",))
        self._step_progress()
        if self.t_inspect.selection() == (str(i),):
            self._on_board_select()

    def _on_inspect_fatal(self, msg: str) -> None:
        self._inspect_fatal = msg
        self.status(f"inspection stopped: {msg}", error=True)
        messagebox.showerror("Inspection stopped", msg, parent=self.root)

    def _inspect_finished(self) -> None:
        verdicts = [self.t_inspect.set(iid, "verdict") for iid in self.t_inspect.get_children()]
        counts = {v: verdicts.count(v) for v in ("PASS", "FAIL", "REVIEW", "ERROR") if v in verdicts}
        for iid in self.t_inspect.get_children():  # anything the worker never reached
            if self.t_inspect.set(iid, "verdict") == "queued":
                self.t_inspect.item(iid, values=(self.inspect_photos[int(iid)].name, "not run", "", "",
                                                 self._inspect_fatal or "the run stopped before this photo"),
                                    tags=("ERROR",))
        summary = (f"{len(self.inspect_runs)} board(s) inspected: "
                   + ", ".join(f"{n} {v}" for v, n in counts.items() if v != "queued")
                   + f"; reports in {self.inspect_out}")
        if self._inspect_fatal:
            summary = f"inspection stopped: {self._inspect_fatal}. " + summary
        self.status(summary, error=bool(self._inspect_fatal or counts.get("ERROR")))
        self.refresh_runs()
        self.refresh_stats()
        self._sync_inspect()

    def _on_board_select(self, _e=None) -> None:
        sel = self.t_inspect.selection()
        self._sync_inspect()
        self.t_iflags.delete(*self.t_iflags.get_children())
        if not sel:
            return
        iid = sel[0]
        run_id = self.inspect_runs.get(iid)
        if run_id is None:
            err = self.inspect_errors.get(iid)
            self.inspect_shown = None
            self.l_iverdict.configure(text="ERROR" if err else "not inspected yet",
                                      fg=DEFECT if err else DIM)
            self.l_ireason.configure(text=err or "", fg=DEFECT)
            self.iv.clear(err or "This board has not been inspected yet.")
            return
        res = self.session[run_id].result
        self.l_iverdict.configure(text=f"{res.verdict}  {Path(res.image).name}", fg=verdict_colour(res.verdict))
        self.l_ireason.configure(text=board_headline(res),
                                 fg=AMBER if res.board_calls and not res.parts else INK)
        for k, (part, state, detail) in enumerate(flagged_rows(res)):
            self.t_iflags.insert("", "end", iid=str(k), values=(part, state, detail))
        if self.inspect_shown != iid:
            self.inspect_shown = iid
            self.iv.clear("loading the annotated photo...")
            self._spawn("load report", load_view_worker, "inspect_image", (iid, run_id),
                        self.session[run_id].paths["annotated"])

    def _on_inspect_image(self, key, bgr, small, k, err: str) -> None:
        iid, run_id = key
        if self.inspect_shown != iid:
            return
        if bgr is None:
            self.iv.clear(err)
            return
        self.iv.set_image(bgr, small, k)

    def _iflag_part(self) -> tuple[Part, float, float] | None:
        sel, bsel = self.t_iflags.selection(), self.t_inspect.selection()
        run_id = self.inspect_runs.get(bsel[0]) if bsel else None
        if not sel or run_id is None:
            return None
        part = self.t_iflags.set(sel[0], "part")
        return result_geometry(self.session[run_id].result).get(part)

    def _on_iflag_select(self, _e=None) -> None:
        g = self._iflag_part()
        if g is None:
            self.iv.fit()
            return
        p, dx, dy = g
        self.iv.show_box(p.x + dx, p.y + dy, p.w, p.h, margin=2.0)

    def _draw_iflag(self, c: tk.Canvas, view: View) -> None:
        g = self._iflag_part()
        if g is not None:
            self._outline(c, view, *g)

    def _outline(self, c: tk.Canvas, view: View, p: Part, dx: float, dy: float) -> None:
        pad = 0.15 * max(p.w, p.h)
        x0, y0 = view.to_canvas(p.x + dx - pad, p.y + dy - pad)
        x1, y1 = view.to_canvas(p.x + dx + p.w + pad, p.y + dy + p.h + pad)
        c.create_rectangle(x0, y0, x1, y1, outline=INK, width=2, dash=(6, 4))

    def review_board(self) -> None:
        sel = self.t_inspect.selection()
        run_id = self.inspect_runs.get(sel[0]) if sel else None
        if run_id is None:
            self.status("pick an inspected board first", error=True)
            return
        fsel = self.t_iflags.selection()
        part = self.t_iflags.set(fsel[0], "part") if fsel else None
        self.nb.select(self.tab_results)
        self.results_nb.select(self.sub_review)
        self.open_run(run_id, refdes=None if part in (None, "board") else part)

    # ======================================================== Results tab
    def _build_results_tab(self, tab: tk.Frame) -> None:
        self.l_rinstr = self._instruction(tab)
        self.l_rinstr.configure(text="Answer each call: Real defect (D) or False call (F). "
                                     "Next (N) jumps to the next unanswered call.")
        self.results_nb = ttk.Notebook(tab)
        self.results_nb.pack(fill="both", expand=True, padx=6, pady=(2, 4))
        self.sub_review = tk.Frame(self.results_nb, bg=BG)
        self.sub_stats = tk.Frame(self.results_nb, bg=BG)
        self.results_nb.add(self.sub_review, text="Review calls")
        self.results_nb.add(self.sub_stats, text="False-call stats")
        self.results_nb.bind("<<NotebookTabChanged>>", lambda e: self._on_results_sub())
        self._build_review(self.sub_review)
        self._build_stats(self.sub_stats)

    def _on_results_sub(self) -> None:
        if self.results_nb.select() == str(self.sub_stats):
            self.refresh_stats()
        else:
            self.rv.canvas.focus_set()

    def _build_review(self, tab: tk.Frame) -> None:
        top = tk.Frame(tab, bg=BG)
        top.pack(fill="x", padx=10, pady=(8, 2))
        self._label(top, "Board").pack(side="left")
        self.v_run = tk.StringVar(master=self.root)
        self.c_run = ttk.Combobox(top, textvariable=self.v_run, state="readonly", width=64)
        self.c_run.pack(side="left", padx=6)
        self.c_run.bind("<<ComboboxSelected>>", self._on_run_pick)
        ttk.Button(top, text="Refresh", command=self.refresh_runs).pack(side="left")
        self.l_operator = self._label(top, "", font=self.f_big)
        self.l_operator.pack(side="right")
        self._label(top, "your status", fg=DIM).pack(side="right", padx=(18, 6))
        self.l_machine = self._label(top, "", font=self.f_big)
        self.l_machine.pack(side="right")
        self._label(top, "machine", fg=DIM).pack(side="right", padx=(0, 6))
        self.l_rprogress = self._label(tab, "", font=self.f_head)
        self.l_rprogress.pack(fill="x", padx=10)
        self.l_rstamp = self._label(tab, "", fg=AMBER)
        self.l_rstamp.pack(fill="x", padx=10)

        body = tk.Frame(tab, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        side = tk.Frame(body, bg=BG, width=500)
        side.pack(side="right", fill="y", padx=(10, 0))
        side.pack_propagate(False)
        self.rv = ImageView(body, overlay=self._draw_highlight,
                            empty="Pick a board above, or click \"Review calls →\" on tab 2.")
        self.rv.canvas.pack(side="left", fill="both", expand=True)
        # Left button has nothing to draw here, so a plain left-drag pans too.
        self.rv.canvas.bind("<ButtonPress-1>", self.rv.pan_start)
        self.rv.canvas.bind("<B1-Motion>", self.rv.pan_move)
        self.rv.canvas.bind("<ButtonRelease-1>", self.rv.pan_end)

        self._label(side, "Calls needing an answer", font=self.f_head).pack(fill="x")
        self.t_calls = self._tree(side, (("answer", "your answer", 90), ("part", "part", 64),
                                         ("state", "problem", 130), ("score", "score", 200)), height=9)
        for k, col in ANSWER_COLOURS.items():
            self.t_calls.tag_configure(k, foreground=col)
        self.t_calls.pack(fill="x", pady=4)
        self.t_calls.bind("<<TreeviewSelect>>", self._on_call_select)
        self.l_call = self._label(side, "", font=self.f_mono, wraplength=480)
        self.l_call.pack(fill="x", pady=(4, 4))

        btns = tk.Frame(side, bg=BG)
        btns.pack(fill="x", pady=(0, 6))
        self.b_answer = {}
        for key, text, style in (("D", "Real defect (D)", "Defect.TButton"),
                                 ("F", "False call (F)", "Accent.TButton"),
                                 ("S", "Skip (S)", "TButton"),
                                 ("R", "Reshoot (R)", "Review.TButton")):
            b = ttk.Button(btns, text=text, style=style, command=lambda k=key: self.answer(k))
            b.pack(side="left", padx=(0, 4))
            self.b_answer[key] = b
        btns2 = tk.Frame(side, bg=BG)
        btns2.pack(fill="x", pady=(0, 6))
        self.b_next = ttk.Button(btns2, text="Next unanswered (N)", command=self.next_call)
        self.b_next.pack(side="left", padx=(0, 4))
        self.b_undo_answer = ttk.Button(btns2, text="Undo (U)", command=self.undo)
        self.b_undo_answer.pack(side="left")

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
        self._sync_answer_buttons()

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

    def open_run(self, run_id: int, refdes: str | None = None) -> None:
        row = self._run_row(run_id)
        if row is None:
            self.status(f"no run {run_id} in {self.db_path}", error=True)
            return
        self.run_id = run_id
        self.run_row = row
        self.cur = None
        self.annotated = None
        self.c_run.current(self.run_rows.index(row))
        self.l_machine.configure(text=row["machine_verdict"], fg=verdict_colour(row["machine_verdict"]))
        self.l_rstamp.configure(text=row["stamp"])
        self.golden_key = (row["program"], int(row["version"]))
        sess = self.session.get(run_id)
        if sess is not None:
            self.geo = result_geometry(sess.result)
        else:
            # A run from an earlier session: the boxes come from the program
            # version on disk and the shifts from the run's CSV report.
            prog, self.geo, why = run_geometry(row, self.programs_root)
            if prog is not None and self.golden_key not in self.goldens \
                    and self.golden_key not in self._golden_loading:
                self._golden_loading.add(self.golden_key)
                self._spawn("load golden", load_view_worker, "review_golden", self.golden_key,
                            prog.golden_paths()[0])
            elif why:
                self.status(why, error=True)
        annotated = (sess.paths.get("annotated") if sess is not None else None) or row.get("annotated")
        if annotated and Path(annotated).exists():
            self.rv.clear("loading the annotated image...")
            self._spawn("load report", load_view_worker, "review_image", run_id, annotated)
        else:
            self.rv.clear("This board's report image is not available (recorded before report "
                          "paths were stored, or moved).\nIts calls are on the right; answer them as usual.\n"
                          f"Reports are under {self.runs_root / row['program']}.")
        self._reload_calls()
        if self.calls:
            j = next((k for k, c in enumerate(self.calls) if refdes and c["refdes"] == refdes), None)
            if j is None:
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

    def _on_review_golden(self, key, bgr, small, k, err: str) -> None:
        self._golden_loading.discard(key)
        if bgr is None:
            self.status(f"golden 1 of {key[0]} v{key[1]}: {err}", error=True)
            return
        self.goldens[key] = bgr
        if key == self.golden_key and self.cur is not None:
            self._show_crops(self._geo_for(self.calls[self.cur]))

    def _reload_calls(self) -> None:
        self.calls_all = self.rec.calls(self.run_id)
        self.calls = review_calls(self.calls_all)
        tv = self.t_calls
        tv.delete(*tv.get_children())
        for c in self.calls:
            tv.insert("", "end", iid=str(c["id"]), values=call_row(c), tags=(c["adjudication"] or "open",))
        st = self.rec.operator_status(self.run_id)
        self.l_operator.configure(text=st, fg=verdict_colour(st))
        machine = self.run_row["machine_verdict"] if self.run_row else ""
        self.l_rprogress.configure(text=review_progress(machine, self.calls))
        if not self.calls:
            self.l_call.configure(text="Nothing on this board needs an answer.")
        self._sync_answer_buttons()

    def _sync_answer_buttons(self) -> None:
        c = self.calls[self.cur] if self.cur is not None and self.cur < len(self.calls) else None
        for key, b in self.b_answer.items():
            ok = c is not None and key in (record.BOARD_KEYS if c["board_level"] else record.PART_KEYS)
            b.configure(state="normal" if ok else "disabled")
        self.b_next.configure(state="normal" if self.calls else "disabled")
        self.b_undo_answer.configure(state="normal" if self.run_id is not None else "disabled")

    def _clear_call(self) -> None:
        self.cur = None
        for img in self.crop_imgs:
            img.configure(image="", text="", width=CROP_SIDE // 8, height=CROP_SIDE // 18)
        self._sync_answer_buttons()
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

    def _geo_for(self, c: dict) -> tuple[Part, float, float] | None:
        return self.geo.get(c["refdes"]) if c["refdes"] else None

    def _present_call(self, i: int, restart_clock: bool = True) -> None:
        """Show call i: summary, zoom to the part, golden | test crops. The clock
        behind `seconds` starts here, and not again when the image arrives late."""
        self.cur = i
        c = self.calls[i]
        if restart_clock:
            self.shown_at = time.perf_counter()
        self.l_call.configure(text=call_summary(c))
        g = self._geo_for(c)
        if self.rv.full is not None:
            if g is not None:
                p, dx, dy = g
                self.rv.show_box(p.x + dx, p.y + dy, p.w, p.h)
            else:
                self.rv.fit()
        self._show_crops(g)
        self._sync_answer_buttons()

    def _draw_highlight(self, c: tk.Canvas, view: View) -> None:
        if self.cur is None or self.cur >= len(self.calls):
            return
        g = self._geo_for(self.calls[self.cur])
        if g is not None:
            self._outline(c, view, *g)

    def _show_crops(self, g) -> None:
        self._crop_photos = []
        pair: list[np.ndarray | None] = [None, None]
        caps = ["GOLDEN", "TEST"]
        if g is not None:
            p, dx, dy = g
            gold = self.goldens.get(self.golden_key) if self.golden_key else None
            test = self.aligned.get(self.run_id)
            if test is None and self.annotated is not None:
                test, caps[1] = self.annotated, "TEST (report image)"
            if gold is None:
                caps[0] = ("GOLDEN (loading...)" if self.golden_key in self._golden_loading
                           else "GOLDEN (not available)")
            if gold is not None:
                pair[0] = report.crop_pair(gold, gold, p)[0]
            if test is not None:
                moved = Part(p.refdes, int(round(p.x + dx)), int(round(p.y + dy)), p.w, p.h)
                pair[1] = report.crop_pair(test, test, moved)[1]
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
        """D F S R N U on Review calls. Not while typing into a field, and not
        with Control or Command held, so no shortcut is taken for a decision."""
        if (self.nb.select() != str(self.tab_results) or self.results_nb.select() != str(self.sub_review)
                or e.state & 0x000C):
            return None
        if self._typing():
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
            self.status("pick a board and a call first", error=True)
            return
        c = self.calls[self.cur]
        if key == "R" and not c["board_level"]:
            self.status("Reshoot is for whole-board calls; for a part answer D, F or S", error=True)
            return
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
                        f"answered; your status {self.l_operator.cget('text')}")
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
            self.status("nothing to undo on this board")
            return
        self._reload_calls()
        j = changed_index(before, self.calls)
        if self.calls:
            self._select_call(j if j is not None else min(self.cur or 0, len(self.calls) - 1))
        self.status("last answer undone" + (f": {self.calls[j]['refdes'] or 'board'} "
                                            f"{self.calls[j]['state']}" if j is not None else ""))

    # -- stats
    def _build_stats(self, tab: tk.Frame) -> None:
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
