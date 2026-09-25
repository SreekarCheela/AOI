"""A taught board: its goldens, its parts, its scale and what good looks like.

A program is a folder:

    <name>/program.json      parts, px/mm, version, golden list, golden statistics
    <name>/golden_01.jpg ... the golden photos, copied in, golden 1 first

Every part coordinate is in golden-1 pixels; every other photo is aligned onto
golden 1 before anything is measured. Re-teaching or adding goldens writes a new
version, and every run records the version it was judged against.
"""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from .io import PhotoError
from .model import MARKED_ENDS, PART_TYPES, Part

PARTS_COLUMNS = ["refdes", "x", "y", "w", "h", "type", "package", "polarized", "marked_end"]
SINGLE_GOLDEN_STAMP = "single golden: fixed thresholds from the 2026-09-25 run"


def _bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def read_parts_csv(path: str | Path) -> list[Part]:
    """refdes,x,y,w,h,type,package,polarized,marked_end -- golden-1 pixels.

    Only refdes,x,y,w,h are required; the rest default to an unpolarized
    'other' part, which is inspected for presence and placement only."""
    path = Path(path)
    if not path.exists():
        raise PhotoError(f"{path}: parts file not found")
    rows = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    parts, seen = [], set()
    for i, r in enumerate(csv.DictReader(rows), start=2):
        name = (r.get("refdes") or r.get("name") or "").strip()
        if not name:
            raise PhotoError(f"{path.name} line {i}: no refdes")
        if name in seen:
            raise PhotoError(f"{path.name} line {i}: {name} listed twice")
        seen.add(name)
        try:
            x, y, w, h = (int(round(float(r[k]))) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            raise PhotoError(f"{path.name} line {i}: {name} needs numeric x,y,w,h") from None
        if w <= 0 or h <= 0:
            raise PhotoError(f"{path.name} line {i}: {name} has an empty box")
        typ = (r.get("type") or "other").strip().lower()
        if typ not in PART_TYPES:
            raise PhotoError(f"{path.name} line {i}: {name} type {typ!r} not one of {', '.join(PART_TYPES)}")
        end = (r.get("marked_end") or "none").strip().lower()
        if end not in MARKED_ENDS:
            raise PhotoError(f"{path.name} line {i}: {name} marked_end {end!r} not one of {', '.join(MARKED_ENDS)}")
        parts.append(Part(name, x, y, w, h, typ, (r.get("package") or "").strip(),
                          _bool(r.get("polarized", "0")), end))
    if not parts:
        raise PhotoError(f"{path.name}: no parts listed")
    return parts


def write_parts_csv(path: str | Path, parts: list[Part]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PARTS_COLUMNS)
        for p in parts:
            w.writerow([p.refdes, p.x, p.y, p.w, p.h, p.type, p.package, int(p.polarized), p.marked_end])


@dataclass
class Program:
    name: str
    version: int
    px_per_mm: float
    parts: list[Part]
    goldens: list[str]                     # file names inside the program folder
    single_golden: bool = False
    accepted_selfcheck: list[str] = field(default_factory=list)
    golden_stats: dict = field(default_factory=dict)   # refdes -> feature -> [mean, sd, n]
    created: str = ""
    folder: str = ""

    @property
    def path(self) -> Path:
        return Path(self.folder)

    def golden_paths(self) -> list[Path]:
        return [self.path / g for g in self.goldens]

    @property
    def stamp(self) -> str:
        return SINGLE_GOLDEN_STAMP if self.single_golden else ""

    def part(self, refdes: str) -> Part:
        for p in self.parts:
            if p.refdes == refdes:
                return p
        raise KeyError(refdes)

    def save(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d.pop("folder")
        (self.path / "program.json").write_text(json.dumps(d, indent=1))
        write_parts_csv(self.path / "parts.csv", self.parts)
        return self.path / "program.json"

    @classmethod
    def load(cls, folder: str | Path) -> "Program":
        folder = Path(folder)
        f = folder / "program.json"
        if not f.exists():
            raise PhotoError(f"{folder}: not a program (no program.json)")
        d = json.loads(f.read_text())
        d["parts"] = [Part(**p) for p in d["parts"]]
        prog = cls(**d, folder=str(folder))
        missing = [g for g in prog.golden_paths() if not g.exists()]
        if missing:
            raise PhotoError(f"{folder}: golden photo(s) missing: {', '.join(m.name for m in missing)}")
        return prog


def new_version_folder(root: str | Path, name: str) -> tuple[Path, int]:
    """<root>/<name>/v<N>, one past the highest version already there."""
    base = Path(root) / name
    base.mkdir(parents=True, exist_ok=True)
    versions = [int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit()]
    v = max(versions, default=0) + 1
    return base / f"v{v}", v


def latest(root: str | Path, name: str) -> Program:
    base = Path(root) / name
    versions = sorted((int(p.name[1:]), p) for p in base.glob("v*") if p.name[1:].isdigit())
    if not versions:
        raise PhotoError(f"no program named {name!r} under {root}")
    return Program.load(versions[-1][1])


def resolve(program: str | Path, root: str | Path = "programs") -> Program:
    """A program folder (…/v3), a program name folder (latest version), or a name under root."""
    p = Path(program)
    if (p / "program.json").exists():
        return Program.load(p)
    if p.is_dir() and any(p.glob("v*/program.json")):
        return latest(p.parent, p.name)
    return latest(root, str(program))


def copy_goldens(paths: list[Path], folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for i, src in enumerate(paths, start=1):
        dst = folder / f"golden_{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        names.append(dst.name)
    return names


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_golden_count(n: int, single_golden: bool) -> None:
    if single_golden:
        if n < 1:
            raise PhotoError("no golden photo given")
        return
    if n < config.PHOTO_MIN_GOLDENS:
        raise PhotoError(
            f"{n} golden photo(s); at least {config.PHOTO_MIN_GOLDENS} are needed. One photo "
            "cannot show how much a good part varies from shot to shot, so every threshold "
            "would be a guess. Pick the board up, put it back, and shoot again.")
