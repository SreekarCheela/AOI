"""Operator review screen.

Native desktop, Tk, no network (spec section 5). It sits on a factory floor and
has to work when the network does not, with input latency that does not make an
operator wait. Tk is in the standard library, which keeps the Pi image small.

The screen exists to answer one question per component, as fast as a person can
answer it: was this call real? Section 10 sets the bar in seconds -- an operator
with thirty minutes of training clears a full board's fails in under sixty --
so the design follows from that:

* the flagged part and a known-good one of the same package, side by side. An
  operator deciding from the crop alone is guessing.
* false colour, low->red mid->green high->blue, because that is the view where
  a starved joint is obvious: it has lost its red.
* the measurement that triggered the call, in words, so the decision is informed
  rather than an aesthetic judgement.
* two keys. D for a real defect, F for a false call. No mouse in the path.

Every decision is written through `store.adjudicate`, with the time taken, and
that is the labelled sample the false-call filter trains on.
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import font as tkfont

import cv2
import numpy as np

from . import config

# Dark, because the station sits in a light-tight enclosure and the operator is
# looking at dim specular images; a white UI destroys their adaptation.
BG = "#14171c"
PANEL = "#1c2027"
INK = "#e8eaf0"
DIM = "#8a93a3"
DEFECT = "#e0604a"
FALSE = "#46b073"
RULE = "#2c313a"

ZOOM = 4          # 96 px crop is a postage stamp on a 24" panel


def false_colour(patch: np.ndarray) -> np.ndarray:
    """low -> red, mid -> green, high -> blue, as one RGB image.

    A correctly formed fillet sweeps blue through green to red from the pad toe
    up to the termination. A starved joint never develops the steep face, so its
    red is simply not there -- which is a thing a person can see instantly and a
    number takes a paragraph to explain.
    """
    idx = [config.CHANNELS.index(c) for c in ("low", "mid", "high")]
    lo, mi, hi = (patch[..., i] for i in idx)
    return np.dstack([lo, mi, hi])


def _to_photo(rgb: np.ndarray, master: tk.Misc | None = None,
              zoom: int = ZOOM) -> tk.PhotoImage:
    """RGB uint8 -> a Tk image, without pulling in an image library.

    Tk reads PPM natively, and encoding one is a header plus the bytes, so the
    conversion costs nothing and adds no dependency.

    `master` is explicit rather than left to Tk's implicit default root: an
    image attached to the wrong interpreter renders blank, and silently.
    """
    big = cv2.resize(rgb, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    h, w = big.shape[:2]
    header = f"P6 {w} {h} 255 ".encode()
    return tk.PhotoImage(master=master, format="PPM",
                         data=header + big.astype(np.uint8).tobytes())


class ReviewApp:
    """One board's fails, one at a time."""

    def __init__(self, store, board_id: int, operator: str = "") -> None:
        self.store = store
        self.board_id = board_id
        self.operator = operator
        self.queue = store.review_queue(board_id)
        self.board = store.db.execute(
            "SELECT * FROM board WHERE id = ?", (board_id,)).fetchone()
        self.i = 0
        self.shown_at = 0.0
        self._photos: list[tk.PhotoImage] = []   # Tk drops unreferenced images

        self.root = tk.Tk()
        self.root.title(f"AOI review - {self.board['name']}")
        self.root.configure(bg=BG)
        self._build()
        self._bind()
        self._show()

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        mono = tkfont.Font(family="Menlo", size=12)
        head = tkfont.Font(family="Helvetica", size=22, weight="bold")
        lab = tkfont.Font(family="Helvetica", size=11)

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=24, pady=(18, 8))
        self.title = tk.Label(top, text="", font=head, fg=INK, bg=BG, anchor="w")
        self.title.pack(side="left")
        self.progress = tk.Label(top, text="", font=mono, fg=DIM, bg=BG, anchor="e")
        self.progress.pack(side="right")

        tk.Frame(self.root, bg=RULE, height=1).pack(fill="x", padx=24)

        mid = tk.Frame(self.root, bg=BG)
        mid.pack(padx=24, pady=16)
        self.panels = []
        for caption in ("THIS PART", "KNOWN GOOD"):
            col = tk.Frame(mid, bg=BG)
            col.pack(side="left", padx=(0, 18))
            tk.Label(col, text=caption, font=lab, fg=DIM, bg=BG).pack(anchor="w")
            img = tk.Label(col, bg=PANEL, bd=0)
            img.pack()
            self.panels.append(img)

        self.reason = tk.Label(self.root, text="", font=head, fg=DEFECT, bg=BG)
        self.reason.pack(padx=24, anchor="w")
        self.detail = tk.Label(self.root, text="", font=mono, fg=INK, bg=BG,
                               justify="left", wraplength=760, anchor="w")
        self.detail.pack(padx=24, pady=(2, 14), anchor="w")

        btns = tk.Frame(self.root, bg=BG)
        btns.pack(padx=24, pady=(0, 8), fill="x")
        self._button(btns, "D   Real defect", DEFECT, lambda: self._decide("defect"))
        self._button(btns, "F   False call", FALSE, lambda: self._decide("false_call"))

        self.foot = tk.Label(self.root, text="", font=tkfont.Font(family="Menlo", size=10),
                             fg=DIM, bg=BG)
        self.foot.pack(pady=(0, 14))

    def _button(self, parent, text: str, colour: str, cmd) -> None:
        b = tk.Button(parent, text=text, command=cmd, font=tkfont.Font(
            family="Helvetica", size=15, weight="bold"),
            bg=colour, fg="#10131a", activebackground=colour,
            bd=0, highlightthickness=0, padx=26, pady=14)
        b.pack(side="left", padx=(0, 12))

    def _bind(self) -> None:
        # Keys, not the mouse. Sixty seconds for a board's fails is a keyboard
        # target; hunting a pointer across a 24 inch panel is not.
        self.root.bind("<KeyPress-d>", lambda e: self._decide("defect"))
        self.root.bind("<KeyPress-f>", lambda e: self._decide("false_call"))
        self.root.bind("<KeyPress-s>", lambda e: self._skip())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    # ------------------------------------------------------------------ flow
    def _show(self) -> None:
        if self.i >= len(self.queue):
            return self._finish()
        v = self.queue[self.i]
        self._photos.clear()

        from .store import decode_crop
        crop = decode_crop(v["crop"]) if v["crop"] else None
        ref = self.store.golden_ref(v["package"])
        for panel, patch in zip(self.panels, (crop, ref)):
            if patch is None:
                panel.configure(image="", text="no crop", fg=DIM, width=24, height=8)
                continue
            photo = _to_photo(false_colour(patch), master=self.root)
            self._photos.append(photo)
            panel.configure(image=photo, text="", width=0, height=0)

        self.title.configure(text=f"{v['refdes']}   {v['package']}")
        self.progress.configure(
            text=f"{self.i + 1} / {len(self.queue)}    board {self.board['name']}")
        self.reason.configure(text=v["reason"].upper())
        self.detail.configure(text=v["detail"] or "")
        self.foot.configure(text="D  real defect        F  false call        "
                                 "S  skip        Esc  stop")
        self.shown_at = time.perf_counter()

    def _decide(self, call: str) -> None:
        if self.i >= len(self.queue):
            return
        v = self.queue[self.i]
        self.store.adjudicate(v["id"], call, self.operator,
                              time.perf_counter() - self.shown_at)
        self.i += 1
        self._show()

    def _skip(self) -> None:
        self.i += 1
        self._show()

    def _finish(self) -> None:
        secs = self.store.board_review_seconds(self.board_id)
        gaps = self.store.untestable(self.board_id)
        self.title.configure(text="Board reviewed")
        self.progress.configure(text="")
        for p in self.panels:
            p.configure(image="", text="", width=0, height=0)
        self.reason.configure(text=f"{len(self.queue)} adjudicated in {secs:.0f} s",
                              fg=FALSE if secs < 60 else DEFECT)
        msg = ("under the 60 s target" if secs < 60
               else "over the 60 s target in section 10")
        if gaps:
            pkgs = sorted({g["package"] for g in gaps})
            msg += (f"\n\nNOT INSPECTED: {len(gaps)} components ({', '.join(pkgs)} "
                    "have no golden model). Teach them before releasing the board.")
        self.detail.configure(text=msg)
        self.foot.configure(text="Esc  close")

    def run(self) -> None:
        self.root.mainloop()
