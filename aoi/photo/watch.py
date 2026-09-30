"""Inspect photos as they arrive in a folder.

    python -m aoi_photo inspect --program NAME --watch FOLDER      (Ctrl-C stops)
    or "Watch folder..." on the app's Inspect tab

Anything that drops photos into a folder feeds the inspector: an iPhone by
AirDrop into Downloads, a camera tool saving frames, a network share. Only
photos that arrive after watching starts are inspected; what was already there
is left alone.

A file is taken once it has stopped growing (the same non-zero size on two
polls in a row), so a photo still being copied is never read half-written.
Hidden and temporary names (".x", "x.tmp", "x.part", "x.download", "x.crdownload")
are ignored: copiers write under those and rename when done.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

from .io import PHOTO_EXT

POLL_S = 0.5
TEMP_SUFFIXES = (".tmp", ".part", ".download", ".crdownload")


def _candidates(folder: Path) -> list[Path]:
    try:
        entries = list(folder.iterdir())
    except OSError:
        return []
    return [f for f in entries
            if not f.name.startswith(".") and f.suffix.lower() in PHOTO_EXT
            and not f.name.lower().endswith(TEMP_SUFFIXES) and f.is_file()]


class FolderWatch:
    """Iterate the photos that arrive in `folder` until `stop` is set, oldest
    first. `on_new(path)` is called just before each is yielded."""

    def __init__(self, folder: str | Path, stop: threading.Event | None = None,
                 poll_s: float = POLL_S, on_new: Callable[[Path], None] | None = None):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(f"{self.folder}: not a folder")
        self.stop = stop or threading.Event()
        self.poll_s = poll_s
        self.on_new = on_new
        self.seen: set[Path] = set(_candidates(self.folder))      # already there: not ours

    def ready(self, sizes: dict[Path, int]) -> list[Path]:
        """New files whose size has held since the last poll; updates `sizes`."""
        out = []
        for f in _candidates(self.folder):
            if f in self.seen:
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            if st.st_size > 0 and sizes.get(f) == st.st_size:
                out.append((st.st_mtime, f.name, f))
            sizes[f] = st.st_size
        return [f for _, _, f in sorted(out)]

    def __iter__(self) -> Iterator[Path]:
        sizes: dict[Path, int] = {}
        while not self.stop.is_set():
            for f in self.ready(sizes):
                self.seen.add(f)
                sizes.pop(f, None)
                if self.on_new is not None:
                    self.on_new(f)
                yield f
                if self.stop.is_set():
                    return
            self.stop.wait(self.poll_s)

