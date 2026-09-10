"""Reading placement files that every tool writes differently.

The inspection program is generated from data the SMT line already produces, and
the centroid file is the spine of it: reference designator, package, position,
rotation, side. Every EDA tool and every contract manufacturer exports that
differently -- different column names, different units, different rotation
conventions, a title block above the header, a decimal comma, a side column that
says "top" or "T" or "1".

Format-tolerant parsing is exactly the kind of tedious work that has to be done
before a machine exists rather than during commissioning, because the failure is
not an exception. A file whose X column is read as Y, or whose mils are read as
millimetres, produces a board full of components in plausible-looking wrong
places, and the first symptom is a false-call rate.

So this module reads what it can and is loud about the rest. It never guesses a
unit it cannot justify, and `ParseReport` carries what was skipped and why, so a
file that half-loads says so instead of quietly inspecting two thirds of a board.

Known limit worth stating: these readers are written from the documented shapes
of the common exports. They have not been fed a real file from a real contract
manufacturer, and until they have, "handles the documented formats" is the
honest claim -- not "validated".
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# Column aliases, lowercased and stripped of punctuation before matching. Order
# matters only in that the first hit wins, so the most specific names come first.
ALIASES: dict[str, tuple[str, ...]] = {
    "refdes": ("refdes", "reference", "designator", "ref", "partref",
               "component", "name", "id"),
    "package": ("package", "footprint", "pattern", "packageref", "pkg",
                "partpattern", "shape"),
    "x": ("midx", "posx", "refx", "centerx", "centrex", "x", "locationx",
          "pinx", "xmm", "xmil"),
    "y": ("midy", "posy", "refy", "centery", "centrey", "y", "locationy",
          "piny", "ymm", "ymil"),
    "rot": ("rotation", "rot", "angle", "orientation", "rotate"),
    "side": ("side", "layer", "tb", "topbottom", "board"),
    "value": ("value", "comment", "val", "partvalue"),
}

# Unit suffixes as they appear inside a value or a column heading.
UNIT_TO_MM = {"mm": 1.0, "millimeter": 1.0, "millimetre": 1.0,
              "mil": 0.0254, "mils": 0.0254, "th": 0.0254, "thou": 0.0254,
              "in": 25.4, "inch": 25.4, '"': 25.4,
              "um": 0.001, "µm": 0.001, "micron": 0.001,
              "cm": 10.0}

_NUM = re.compile(r"[-+]?\d*[.,]?\d+(?:[eE][-+]?\d+)?")
_UNIT = re.compile(r"[a-zµ\"]+", re.I)

# Compared after `_key`, so "TopLayer", "Top Layer" and "top-layer" all reduce
# to "toplayer" and only one spelling has to be listed.
TOP_WORDS = {"top", "t", "1", "toplayer", "topside", "front", "f", "a",
             "component", "componentside"}
BOTTOM_WORDS = {"bottom", "b", "2", "bottomlayer", "bottomside", "back",
                "bot", "solder", "solderside"}


_TRAILING_UNIT = re.compile(
    r"(mm|mil|mils|thou|th|inch|in|um|cm)$", re.I)


def _key(s: str) -> str:
    """Normalise a column heading for alias matching."""
    return re.sub(r"[^a-z0-9]", "", (s or "").strip().lower())


def _keys(s: str) -> tuple[str, ...]:
    """Every form of a heading worth matching against.

    Headings carry their units -- `Center-X(mm)`, `Mid X (mil)`, `PosX_mm` --
    and they normalise to `centerxmm`, `midxmil`, `posxmm`. Listing every
    coordinate name times every unit as a separate alias is how that table gets
    long and still misses one, so the unit is stripped and both forms tried.
    """
    k = _key(s)
    stripped = _TRAILING_UNIT.sub("", k)
    return (k,) if stripped == k else (k, stripped)


_BRACKETED = re.compile(r"[(\[]\s*([a-zµ\"]+)\s*[)\]]", re.I)


def _unit_from(text: str) -> float | None:
    """Millimetre scale implied by a unit written in a heading or a value.

    Looked for in brackets first, then at the end. Taking the FIRST run of
    letters reads `Mid X (mil)` as "Mid" and finds no unit, which silently
    leaves the file in millimetres -- so every coordinate comes out 39x too
    large and the board is nonsense in a way no exception reports.
    """
    text = (text or "").strip()
    if not text:
        return None
    m = _BRACKETED.search(text)
    if m:
        u = UNIT_TO_MM.get(m.group(1).lower())
        if u:
            return u
    tail = re.search(r"([a-zµ\"]+)\s*$", text, re.I)
    if tail:
        u = UNIT_TO_MM.get(tail.group(1).lower())
        if u:
            return u
    return None


def _number(text: str, scale: float) -> float | None:
    """A float out of a messy cell, in mm.

    A unit inside the cell overrides the column's, because a file that writes
    `1.27mm` in one row and `50mil` in the next is telling you something the
    heading cannot.
    """
    if text is None:
        return None
    text = text.strip().strip('"').strip()
    if not text:
        return None
    m = _NUM.search(text)
    if not m:
        return None
    raw = m.group(0)
    # A comma is a decimal point in much of the world, and a thousands separator
    # in the rest. Treat it as a decimal point only when nothing else could be:
    # exactly one comma, no dot, and fewer than three digits after it.
    if "," in raw and "." not in raw:
        head, _, tail = raw.partition(",")
        raw = f"{head}.{tail}" if len(tail) != 3 else f"{head}{tail}"
    else:
        raw = raw.replace(",", "")
    try:
        v = float(raw)
    except ValueError:
        return None
    return v * (_unit_from(text[m.end():]) or scale)


@dataclass
class Placement:
    refdes: str
    package: str
    x: float                 # mm
    y: float                 # mm, in the file's own convention
    rot: float               # deg
    side: str                # "top" or "bottom"
    value: str = ""


@dataclass
class ParseReport:
    """What the file gave up, and what it did not."""
    path: str = ""
    rows_read: int = 0
    placements: list[Placement] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)
    columns: dict[str, str] = field(default_factory=dict)
    unit_scale: float = 1.0
    unit_source: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.placements) and not self.missing

    @property
    def missing(self) -> list[str]:
        return [k for k in ("refdes", "x", "y") if k not in self.columns]

    def summary(self) -> str:
        unit = {1.0: "mm", 0.0254: "mil", 25.4: "inch",
                0.001: "um", 10.0: "cm"}.get(self.unit_scale, f"x{self.unit_scale}")
        lines = [f"{len(self.placements)} placements from {self.rows_read} rows"
                 f"  ({unit}, {self.unit_source})"]
        if self.columns:
            lines.append("  columns: " + ", ".join(
                f"{k}={v!r}" for k, v in sorted(self.columns.items())))
        if self.missing:
            lines.append(f"  MISSING required column(s): {', '.join(self.missing)}")
        if self.skipped:
            lines.append(f"  skipped {len(self.skipped)} row(s):")
            for n, why in self.skipped[:5]:
                lines.append(f"    line {n}: {why}")
            if len(self.skipped) > 5:
                lines.append(f"    ... and {len(self.skipped) - 5} more")
        return "\n".join(lines)


def _find_header(rows: list[list[str]]) -> int:
    """Index of the row that is actually the header.

    Altium and several CM templates put a title block above it -- job name,
    date, units, a blank line. Taking row 0 on faith reads those as column
    names and then finds no columns at all. The header is the first row that
    names at least a designator and one coordinate.
    """
    for i, row in enumerate(rows[:40]):
        keys = {k for c in row for k in _keys(c)}
        has_ref = any(k in ALIASES["refdes"] for k in keys)
        has_xy = any(k in ALIASES["x"] for k in keys) and \
                 any(k in ALIASES["y"] for k in keys)
        if has_ref and has_xy:
            return i
    return 0


def _map_columns(header: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for field_name, names in ALIASES.items():
        for raw in header:
            if field_name not in out and any(k in names for k in _keys(raw)):
                out[field_name] = raw
                break
    return out


def _units(header: list[str], mapping: dict[str, str],
           default: float, forced: str | None) -> tuple[float, str]:
    if forced:
        scale = UNIT_TO_MM.get(forced.lower())
        if scale is None:
            raise ValueError(f"unknown unit {forced!r}")
        return scale, "given on the command line"
    for key in ("x", "y"):
        col = mapping.get(key)
        if col:
            u = _unit_from(col)
            if u:
                return u, f"from the {col!r} heading"
    return default, "assumed -- no unit in the headings or the values"


def read(path: str | Path, unit: str | None = None,
         default_unit_mm: float = 1.0, side: str | None = "top") -> ParseReport:
    """Parse a placement file into a `ParseReport`.

    `side` filters to one side of the board; None keeps everything. The default
    is "top" because a double-sided assembly is two scans and mixing both sides
    into one program puts half the components at coordinates that are mirrored
    with respect to the image.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")

    # Comment lines are dropped -- EXCEPT that KiCad writes its column names in
    # one ("# Ref  Val  Package  PosX  PosY  Rot  Side"). Strip the marker and
    # keep the line; a comment that does not name columns is discarded by the
    # header search anyway, and one that does is the only header the file has.
    kept: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("#", "//", ";")):
            body = s.lstrip("#/; ").strip()
            keys = {k for c in body.split() for k in _keys(c)}
            if any(k in ALIASES["refdes"] for k in keys) and \
               any(k in ALIASES["x"] for k in keys):
                kept.append(body)
            continue
        kept.append(ln)
    if not kept:
        return ParseReport(path=str(path))

    # Try each delimiter and keep whichever actually yields a header. Sniffing
    # is unreliable here because the title block above the header is prose --
    # "Board: DEMO-1", "Units: mm" -- and a sniffer offered those will happily
    # choose ':' or a space, after which every row is shredded and the file
    # reports as having no columns at all rather than as being misread.
    rows: list[list[str]] = []
    best_score = -1
    for delim in (",", ";", "\t", "|"):
        try:
            cand = list(csv.reader(io.StringIO("\n".join(kept)), delimiter=delim))
        except csv.Error:
            continue
        cand = [r for r in cand if any(c.strip() for c in r)]
        if not cand:
            continue
        hdr = _find_header(cand)
        found = _map_columns(cand[hdr])
        # A header that names a designator and both coordinates beats one that
        # merely produced more columns.
        score = (3 if {"refdes", "x", "y"} <= set(found) else 0) * 100 + len(found)
        if score > best_score:
            best_score, rows = score, cand

    # Whitespace-delimited files -- KiCad .pos among them -- sniff as a single
    # column. Splitting on runs of whitespace is the fallback, and it is only
    # taken when it actually produces more columns, so a genuine one-column file
    # is left alone.
    if max((len(r) for r in rows), default=0) < 3:
        split = [r.split() for r in kept]
        if max((len(r) for r in split), default=0) >= 3:
            rows = split

    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ParseReport(path=str(path))

    hdr_i = _find_header(rows)
    header = rows[hdr_i]
    mapping = _map_columns(header)
    rep = ParseReport(path=str(path), columns=dict(mapping))
    if rep.missing:
        return rep

    scale, why = _units(header, mapping, default_unit_mm, unit)
    rep.unit_scale, rep.unit_source = scale, why
    idx = {k: header.index(v) for k, v in mapping.items()}

    for n, row in enumerate(rows[hdr_i + 1:], start=hdr_i + 2):
        rep.rows_read += 1

        def cell(key: str) -> str:
            i = idx.get(key)
            return row[i].strip() if i is not None and i < len(row) else ""

        refdes = cell("refdes").strip('"')
        if not refdes:
            rep.skipped.append((n, "no reference designator"))
            continue
        x, y = _number(cell("x"), scale), _number(cell("y"), scale)
        if x is None or y is None:
            rep.skipped.append((n, f"{refdes}: no usable coordinate "
                                   f"(x={cell('x')!r} y={cell('y')!r})"))
            continue

        raw_side = _key(cell("side").strip('"'))
        if raw_side in BOTTOM_WORDS:
            this_side = "bottom"
        elif raw_side in TOP_WORDS or not raw_side:
            this_side = "top"
        else:
            this_side = raw_side or "top"
        if side is not None and this_side != side:
            rep.skipped.append((n, f"{refdes}: on the {this_side} side"))
            continue

        rot = _number(cell("rot"), 1.0)
        pkg = cell("package").strip('"')
        if not pkg:
            rep.skipped.append((n, f"{refdes}: no package named"))
            continue

        rep.placements.append(Placement(
            refdes=refdes, package=pkg, x=x, y=y,
            rot=(rot or 0.0) % 360.0, side=this_side,
            value=cell("value").strip('"')))
    return rep


def extent(rep: ParseReport, margin_mm: float = 2.0) -> tuple[float, float, float, float]:
    """(min_x, min_y, width, height) covering every placement, plus a margin.

    A centroid file states where parts are, never how big the board is. This is
    a usable stand-in when the outline is not to hand -- but it is a guess, and
    a board with nothing near one edge will come out too small. Prefer the real
    outline when you have it.
    """
    if not rep.placements:
        raise ValueError("no placements to take an extent from")
    xs = [p.x for p in rep.placements]
    ys = [p.y for p in rep.placements]
    return (min(xs) - margin_mm, min(ys) - margin_mm,
            max(xs) - min(xs) + 2 * margin_mm,
            max(ys) - min(ys) + 2 * margin_mm)
