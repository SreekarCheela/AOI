"""Scoring a program against photos whose defects are known.

    python -m aoi_photo eval --program NAME FOLDER [--truth truth.json] [--boxes parts.csv]

FOLDER holds one photo per board and a truth.json (the fixture format):

    {"goldens": ["golden1.heic", ...],          photos in FOLDER that are not boards
     "truth": [{"image": "test1", "refdes": "C6", "defect": "missing"}, ...]}

Every other photo is a board. A board with no truth entry is a good board, so
anything flagged on it is a false call. Defect names: missing, offset, skew,
wrong_part, tombstone, reversed, marking_differs (anything else is caught by
any flag).

Per planted defect the outcome is one of
    caught        flagged with a state that names this defect
    other label   flagged, but as something else (the board still fails)
    missed        not flagged: an escape
    not taught    the program has no box for the part, so no part check ran;
                  with --boxes, still "caught" if an UNEXPECTED DIFFERENCE
                  covers where the part should be
    not checked   the whole board was refused (could not align, blurred ...)

and every flag on a part with no planted defect is a false call. Nothing is
written to the operator record: an evaluation must not become Stats' labels.

The numbers are only as honest as the photos. Photos that were used to choose
a threshold (the 2026-09-25/28 fixtures are, for most of them) score the method
on its own training data; hold boards out before trusting a rate.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .model import DIFF_TYPE, BoardResult, Part

# defect -> the part states that name it
EXPECTED: dict[str, tuple[str, ...]] = {
    "missing": ("MISSING",),
    "offset": ("OFFSET",),
    "skew": ("SKEW",),
    "wrong_part": ("WRONG PART",),
    "tombstone": ("TOMBSTONE",),
    "reversed": ("REVERSED", "POLARITY REVIEW"),
    "marking_differs": ("MARKING DIFFERS",),
}
OUTCOMES = ("caught", "other label", "missed", "not taught", "not checked")
FLAG = ("FAIL", "REVIEW")


@dataclass
class Row:
    image: str
    refdes: str
    defect: str
    outcome: str
    states: list[str] = field(default_factory=list)


@dataclass
class Score:
    rows: list[Row]
    false_calls: list[tuple[str, str, str, str]]     # image, refdes, state, severity
    boards: dict[str, dict]                          # image -> verdict, planted, flagged, seconds
    parts_inspected: int

    def by_defect(self) -> dict[str, Counter]:
        out: dict[str, Counter] = {}
        for r in self.rows:
            out.setdefault(r.defect, Counter())[r.outcome] += 1
        return out

    def summary(self) -> dict:
        n = len(self.rows)
        c = Counter(r.outcome for r in self.rows)
        bad = [b for b in self.boards.values() if b["planted"]]
        good = [b for b in self.boards.values() if not b["planted"]]
        return dict(
            planted=n, caught=c["caught"], other_label=c["other label"], missed=c["missed"],
            not_taught=c["not taught"], not_checked=c["not checked"],
            # a defect the operator is sent to, under any label
            flagged_rate=(c["caught"] + c["other label"]) / n if n else None,
            false_calls=len(self.false_calls),
            false_fail_calls=sum(1 for f in self.false_calls if f[3] == "FAIL"),
            false_calls_per_board=len(self.false_calls) / len(self.boards) if self.boards else None,
            false_call_rate=len(self.false_calls) / self.parts_inspected if self.parts_inspected else None,
            boards=len(self.boards),
            board_escapes=sum(1 for b in bad if b["verdict"] == "PASS"),
            good_boards=len(good),
            good_boards_failed=sum(1 for b in good if b["verdict"] != "PASS"),
            boards_refused=sum(1 for b in self.boards.values() if b["refused"]),
            seconds_per_board=(sum(b["seconds"] for b in self.boards.values()) / len(self.boards)
                               if self.boards else None),
        )


def read_truth(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    if not isinstance(d.get("truth"), list):
        raise ValueError(f"{path}: no \"truth\" list")
    for t in d["truth"]:
        if not {"image", "refdes", "defect"} <= set(t):
            raise ValueError(f"{path}: truth entry {t} needs image, refdes and defect")
    return d


def non_boards(truth: dict) -> set[str]:
    """Stems of the photos in the folder that are goldens, not boards."""
    names = list(truth.get("goldens", [])) + ([truth["golden"]] if truth.get("golden") else [])
    return {Path(n).stem for n in names}


def _covered(box: Part, diffs: list[Part], min_share: float = 0.3) -> bool:
    """An UNEXPECTED DIFFERENCE covers at least min_share of the part's box."""
    for d in diffs:
        ix = max(0, min(box.x + box.w, d.x + d.w) - max(box.x, d.x))
        iy = max(0, min(box.y + box.h, d.y + d.h) - max(box.y, d.y))
        if ix * iy >= min_share * box.w * box.h:
            return True
    return False


def score(results: dict[str, BoardResult], truth: list[dict], taught: set[str],
          boxes: dict[str, Part] | None = None, seconds: dict[str, float] | None = None) -> Score:
    """Score results (keyed by photo stem) against planted defects. `taught` is
    the program's refdes set; `boxes` places untaught parts in golden-1 pixels."""
    planted: dict[tuple[str, str], str] = {(t["image"], t["refdes"]): t["defect"] for t in truth}
    rows: list[Row] = []
    false: list[tuple[str, str, str, str]] = []
    boards: dict[str, dict] = {}
    inspected = 0
    for image, res in results.items():
        refused = not res.parts
        flagged = {pr.refdes: [c for c in pr.calls if c.severity in FLAG] for pr in res.parts}
        diffs = [pr.part for pr in res.parts if pr.part.type == DIFF_TYPE]
        inspected += sum(1 for pr in res.parts if pr.part.type != DIFF_TYPE)
        here = {r: d for (i, r), d in planted.items() if i == image}
        for refdes, defect in sorted(here.items()):
            calls = flagged.get(refdes, [])
            states = [c.state for c in calls]
            if refused:
                out = "not checked"
            elif refdes not in taught:
                box = (boxes or {}).get(refdes)
                out = "caught" if box is not None and _covered(box, diffs) else "not taught"
            elif not calls:
                out = "missed"
            elif set(states) & set(EXPECTED.get(defect, states)):
                out = "caught"
            else:
                out = "other label"
            rows.append(Row(image, refdes, defect, out, states))
        explained = {r for r in here if r not in taught and boxes and r in boxes}
        for refdes, calls in flagged.items():
            if refdes in here:
                continue
            if refdes.startswith("DIFF") and any(_covered(boxes[r], [_part_of(res, refdes)])
                                                 for r in explained):
                continue                                  # the difference is a planted untaught defect
            for c in calls:
                false.append((image, refdes, c.state, c.severity))
        for c in res.board_calls:
            if c.severity in FLAG and not refused:
                false.append((image, "board", c.state, c.severity))
        boards[image] = dict(verdict=res.verdict, planted=len(here), refused=refused,
                             flagged=sum(1 for v in flagged.values() if v),
                             seconds=(seconds or {}).get(image, 0.0))
    return Score(rows, false, boards, inspected)


def _part_of(res: BoardResult, refdes: str) -> Part:
    return next(pr.part for pr in res.parts if pr.refdes == refdes)


def run(program, folder: str | Path, truth_path: str | Path | None = None,
        boxes_path: str | Path | None = None, progress=None) -> Score:
    """Inspect every board photo in `folder` with `program` and score it."""
    from . import inspect as I
    from .io import photo_files
    from .program import read_parts_csv
    folder = Path(folder)
    truth = read_truth(truth_path or folder / "truth.json")
    skip = non_boards(truth)
    photos = [f for f in photo_files(folder) if f.stem not in skip]
    unknown = sorted({t["image"] for t in truth["truth"]} - {f.stem for f in photos})
    if unknown:
        raise ValueError(f"truth names photos not in {folder}: {', '.join(unknown)}")
    golden = I.Golden(program)
    results: dict[str, BoardResult] = {}
    secs: dict[str, float] = {}
    for f, load in I.prefetch(photos):
        t0 = time.perf_counter()
        results[f.stem], _ = I.inspect(program, f, golden, bgr=load())
        secs[f.stem] = time.perf_counter() - t0
        if progress:
            progress(f, results[f.stem], secs[f.stem])
    boxes = {p.refdes: p for p in read_parts_csv(boxes_path)} if boxes_path else None
    return score(results, truth["truth"], {p.refdes for p in program.parts}, boxes, secs)


def report(s: Score) -> str:
    """The score as text: per defect, the escapes and false calls by name, totals."""
    lines = [f"{'defect':18}{'planted':>8}" + "".join(f"{o:>13}" for o in OUTCOMES)]
    for defect, c in sorted(s.by_defect().items()):
        lines.append(f"{defect:18}{sum(c.values()):8d}" + "".join(f"{c[o]:13d}" for o in OUTCOMES))
    esc = [r for r in s.rows if r.outcome in ("missed", "not taught", "not checked")]
    if esc:
        lines.append("")
        lines.append("not caught:")
        lines += [f"  {r.image} {r.refdes} {r.defect}: {r.outcome}" for r in esc]
    wrong = [r for r in s.rows if r.outcome == "other label"]
    if wrong:
        lines.append("caught under another label:")
        lines += [f"  {r.image} {r.refdes} {r.defect}: called {', '.join(r.states)}" for r in wrong]
    if s.false_calls:
        lines.append(f"false calls ({len(s.false_calls)}):")
        lines += [f"  {i} {r}: {st} ({sev})" for i, r, st, sev in s.false_calls]
    m = s.summary()
    lines.append("")
    rate = "n/a" if m["flagged_rate"] is None else f"{m['flagged_rate']:.0%}"
    lines.append(f"{m['planted']} planted defects: {m['caught']} caught, {m['other_label']} under another label, "
                 f"{m['missed']} missed, {m['not_taught']} not taught, {m['not_checked']} on refused boards "
                 f"(flagged {rate})")
    fcr = "n/a" if m["false_call_rate"] is None else f"{m['false_call_rate']:.2%}"
    lines.append(f"{m['false_calls']} false calls ({m['false_fail_calls']} FAIL) on {m['boards']} boards, "
                 f"{fcr} of {s.parts_inspected} part inspections")
    lines.append(f"boards: {m['board_escapes']} defective board(s) passed, "
                 f"{m['good_boards_failed']} of {m['good_boards']} good board(s) not passed, "
                 f"{m['boards_refused']} refused (could not be checked)")
    if m["seconds_per_board"] is not None:
        lines.append(f"{m['seconds_per_board']:.2f} s per board (inspection only, this machine)")
    return "\n".join(lines)


def to_json(s: Score) -> dict:
    return dict(summary=s.summary(),
                by_defect={d: dict(c) for d, c in s.by_defect().items()},
                rows=[r.__dict__ for r in s.rows],
                false_calls=[dict(image=i, refdes=r, state=st, severity=sev) for i, r, st, sev in s.false_calls],
                boards=s.boards)
