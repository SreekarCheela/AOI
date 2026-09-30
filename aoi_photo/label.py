"""Labelling window: say which defects were planted on which board photo.

    python -m aoi_photo label --program NAME FOLDER

Writes FOLDER/truth.json (aoi/photo/truth.py), the file `eval` scores against.
Each photo is aligned onto the program's golden 1, so every taught part's box
lands on it. Click a part (or pick it in the list), then press its defect:

    m missing     o offset      s skew        w wrong part
    t tombstone   r reversed    k marking differs
    Delete / BackSpace / c: clear          n / p: next / previous photo
    g: this photo is a golden, not a board

A part the program never boxed can be typed in by name. Every change is saved
at once. A board with nothing labelled counts as a good board.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np

from aoi.photo import inspect as I
from aoi.photo import register
from aoi.photo.io import grey, load_bgr
from aoi.photo.program import Program
from aoi.photo.truth import DEFECTS, Truth
from aoi.review import BG, DEFECT, DIM, FALSE, INK, PANEL
from aoi_photo.app import ImageView, error_text

KEYS = {"m": "missing", "o": "offset", "s": "skew", "w": "wrong_part", "t": "tombstone",
        "r": "reversed", "k": "marking_differs"}
LABEL = {d: d.replace("_", " ") for d in DEFECTS}
KEEP = 4                                  # aligned photos held in memory


def part_at(parts, x: float, y: float):
    """The smallest taught box containing golden-1 point (x, y), or None."""
    hits = [p for p in parts if p.x <= x <= p.x + p.w and p.y <= y <= p.y + p.h]
    return min(hits, key=lambda p: p.w * p.h) if hits else None


def align_worker(golden: I.Golden, photo: Path) -> tuple[np.ndarray, str]:
    """The photo in golden-1 pixels, and a note ("" when aligned)."""
    bgr = load_bgr(photo)
    reg = golden.align(grey(bgr))
    if not reg.ok:
        return bgr, f"could not align ({reg.note}): the boxes will not sit on the parts"
    warped, _ = register.warp_to_ref(bgr, reg.H, golden.bgr.shape)
    return warped, ""


class LabelApp:
    def __init__(self, root: tk.Tk, program: Program, folder: str | Path):
        self.root, self.program = root, program
        self.truth = Truth(folder)
        self.parts = list(program.parts)
        self.photos = self.truth.photos()
        self.i = 0
        self.sel: str | None = None
        self.images: OrderedDict[Path, tuple[np.ndarray, str]] = OrderedDict()
        self.q: queue.Queue = queue.Queue()
        self.golden: I.Golden | None = None
        self._loading: set[Path] = set()
        root.title(f"Label planted defects: {program.name} v{program.version} on {self.truth.folder}")
        root.configure(bg=BG)
        root.geometry("1400x900")
        self._build()
        threading.Thread(target=self._load_golden, daemon=True).start()
        root.after(50, self._poll)
        self.show(0)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        r = self.root
        left = tk.Frame(r, bg=BG, width=300)
        left.pack(side="left", fill="y", padx=10, pady=10)
        tk.Label(left, text="Photos", bg=BG, fg=INK, anchor="w").pack(fill="x")
        self.lb = tk.Listbox(left, height=18, bg=PANEL, fg=INK, selectbackground="#3a4250",
                             exportselection=False, width=34)
        self.lb.pack(fill="x")
        self.lb.bind("<<ListboxSelect>>", lambda e: self.lb.curselection() and self.show(self.lb.curselection()[0]))
        self.v_golden = tk.BooleanVar()
        ttk.Checkbutton(left, text="this photo is a golden (g)", variable=self.v_golden,
                        command=self._toggle_golden).pack(anchor="w", pady=(6, 10))
        tk.Label(left, text="Planted on this board", bg=BG, fg=INK, anchor="w").pack(fill="x")
        self.lb_marks = tk.Listbox(left, height=12, bg=PANEL, fg=INK, exportselection=False, width=34)
        self.lb_marks.pack(fill="x")
        self.lb_marks.bind("<<ListboxSelect>>", self._on_mark_select)
        tk.Label(left, text="Part not boxed by the program:", bg=BG, fg=DIM, anchor="w").pack(fill="x", pady=(10, 0))
        row = tk.Frame(left, bg=BG)
        row.pack(fill="x")
        self.e_ref = ttk.Entry(row, width=8)
        self.e_ref.pack(side="left")
        self.c_def = ttk.Combobox(row, values=[LABEL[d] for d in DEFECTS], state="readonly", width=16)
        self.c_def.current(0)
        self.c_def.pack(side="left", padx=4)
        ttk.Button(row, text="Add", command=self._add_typed).pack(side="left")
        self.l_status = tk.Label(left, text="", bg=BG, fg=DIM, anchor="w", justify="left", wraplength=290)
        self.l_status.pack(fill="x", pady=(12, 0))

        right = tk.Frame(r, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
        bar = tk.Frame(right, bg=BG)
        bar.pack(fill="x")
        for k, d in KEYS.items():
            ttk.Button(bar, text=f"{LABEL[d]} ({k})", command=lambda d=d: self.mark(d)).pack(side="left", padx=2)
        ttk.Button(bar, text="clear (Del)", command=lambda: self.mark(None)).pack(side="left", padx=2)
        self.l_part = tk.Label(right, text="click a part", bg=BG, fg=INK, anchor="w")
        self.l_part.pack(fill="x", pady=4)
        self.iv = ImageView(right, overlay=self._draw, empty="aligning...")
        self.iv.canvas.pack(fill="both", expand=True)
        self.iv.canvas.bind("<ButtonPress-1>", self._on_click)
        for k, d in KEYS.items():
            r.bind(f"<KeyPress-{k}>", lambda e, d=d: self._key(e, d))
        for k in ("Delete", "BackSpace", "c"):
            r.bind(f"<KeyPress-{k}>", lambda e: self._key(e, None))
        r.bind("<KeyPress-n>", lambda e: self._typing(e) or self.show(self.i + 1))
        r.bind("<KeyPress-p>", lambda e: self._typing(e) or self.show(self.i - 1))
        r.bind("<KeyPress-g>", lambda e: self._typing(e) or (self.v_golden.set(not self.v_golden.get()),
                                                             self._toggle_golden()))

    # ------------------------------------------------------------- background
    def _load_golden(self) -> None:
        try:
            self.q.put(("golden", I.golden_for(self.program)))
        except Exception as e:                    # noqa: BLE001
            self.q.put(("error", f"golden 1 of {self.program.name}: {error_text(e)}"))

    def _request(self, photo: Path) -> None:
        if self.golden is None or photo in self.images or photo in self._loading:
            return
        self._loading.add(photo)

        def run() -> None:
            try:
                self.q.put(("image", photo, *align_worker(self.golden, photo)))
            except Exception as e:                # noqa: BLE001
                self.q.put(("image", photo, None, error_text(e)))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "golden":
                    self.golden = msg[1]
                    self.show(self.i)
                elif msg[0] == "error":
                    self.l_status.configure(text=msg[1], fg=DEFECT)
                elif msg[0] == "image":
                    _, photo, img, note = msg
                    self._loading.discard(photo)
                    if img is None:
                        self.images[photo] = (None, note)
                    else:
                        self.images[photo] = (img, note)
                    while len(self.images) > KEEP:
                        self.images.popitem(last=False)
                    if self.photos and photo == self.photos[self.i]:
                        self.show(self.i)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    # ------------------------------------------------------------------ state
    @property
    def photo(self) -> Path | None:
        return self.photos[self.i] if self.photos else None

    def _refresh_list(self) -> None:
        self.lb.delete(0, "end")
        gold = self.truth.goldens
        for p in self.photos:
            n = len(self.truth.on(p.stem))
            tag = "golden" if p.stem in gold else (f"{n} planted" if n else "good board")
            self.lb.insert("end", f"{p.name}   ({tag})")
        if self.photos:
            self.lb.selection_clear(0, "end")
            self.lb.selection_set(self.i)
            self.lb.see(self.i)

    def _refresh_marks(self) -> None:
        self.lb_marks.delete(0, "end")
        self._marks = sorted(self.truth.on(self.photo.stem).items()) if self.photo else []
        for ref, d in self._marks:
            boxed = any(p.refdes == ref for p in self.parts)
            self.lb_marks.insert("end", f"{ref}: {LABEL.get(d, d)}" + ("" if boxed else "   (not boxed)"))

    def show(self, i: int) -> None:
        if not self.photos:
            self.iv.clear(f"no photos in {self.truth.folder}")
            return
        self.i = max(0, min(len(self.photos) - 1, i))
        photo = self.photos[self.i]
        self.v_golden.set(photo.stem in self.truth.goldens)
        self._refresh_list()
        self._refresh_marks()
        got = self.images.get(photo)
        if got is None:
            self.iv.clear("loading the golden..." if self.golden is None else f"aligning {photo.name}...")
            self._request(photo)
        elif got[0] is None:
            self.iv.clear(got[1])
        else:
            if self.iv.full is not got[0]:
                self.iv.set_image(got[0])
            self.iv.redraw()
        note = got[1] if got else ""
        self.l_status.configure(text=note or f"{len(self.truth.boards())} board photo(s); "
                                              f"{len(self.truth.data['truth'])} defect(s) planted",
                                fg=DEFECT if note else DIM)
        if self.golden is not None:                # align the next photo ahead
            for j in (self.i + 1,):
                if j < len(self.photos):
                    self._request(self.photos[j])

    def select(self, refdes: str | None) -> None:
        self.sel = refdes
        d = self.truth.on(self.photo.stem).get(refdes) if refdes and self.photo else None
        self.l_part.configure(text=(f"{refdes}: {LABEL[d]}" if d else f"{refdes}: nothing planted")
                              if refdes else "click a part")
        self.iv.redraw()

    def mark(self, defect: str | None) -> None:
        if self.photo is None or self.sel is None:
            self.l_status.configure(text="click a part first", fg=DEFECT)
            return
        if self.photo.stem in self.truth.goldens:
            self.l_status.configure(text="this photo is marked as a golden; untick it to label it", fg=DEFECT)
            return
        self.truth.set(self.photo.stem, self.sel, defect)
        self._refresh_list()
        self._refresh_marks()
        self.select(self.sel)

    # ----------------------------------------------------------------- events
    def _typing(self, e) -> bool:
        return isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Combobox))

    def _key(self, e, defect: str | None) -> None:
        if not self._typing(e):
            self.mark(defect)

    def _on_click(self, e) -> None:
        self.iv.canvas.focus_set()
        x, y = self.iv.view.to_image(e.x, e.y)
        p = part_at(self.parts, x, y)
        self.select(p.refdes if p else None)

    def _on_mark_select(self, _e=None) -> None:
        s = self.lb_marks.curselection()
        if s:
            ref = self._marks[s[0]][0]
            self.select(ref)
            p = next((q for q in self.parts if q.refdes == ref), None)
            if p is not None and self.iv.full is not None:
                self.iv.show_box(p.x, p.y, p.w, p.h, margin=3.0)

    def _add_typed(self) -> None:
        ref = self.e_ref.get().strip().upper()
        if not ref:
            return
        self.select(ref)
        self.mark(DEFECTS[self.c_def.current()])
        self.e_ref.delete(0, "end")

    def _toggle_golden(self) -> None:
        if self.photo is None:
            return
        try:
            self.truth.set_golden(self.photo, self.v_golden.get())
        except ValueError as e:
            self.v_golden.set(False)
            self.l_status.configure(text=str(e), fg=DEFECT)
        self._refresh_list()

    def _draw(self, c: tk.Canvas, view) -> None:
        if self.photo is None:
            return
        planted = self.truth.on(self.photo.stem)
        for p in self.parts:
            x0, y0 = view.to_canvas(p.x, p.y)
            x1, y1 = view.to_canvas(p.x + p.w, p.y + p.h)
            d = planted.get(p.refdes)
            col = DEFECT if d else (FALSE if p.refdes == self.sel else DIM)
            c.create_rectangle(x0, y0, x1, y1, outline=col, width=3 if (d or p.refdes == self.sel) else 1)
            text = f"{p.refdes}: {LABEL[d]}" if d else p.refdes
            c.create_text(x0, y0 - 3, text=text, anchor="sw", fill=col,
                          font=("TkDefaultFont", 10, "bold" if d else "normal"))


def main(program: Program, folder: str | Path) -> int:
    root = tk.Tk()
    LabelApp(root, program, folder)
    root.mainloop()
    return 0
