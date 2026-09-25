"""photo_aoi.db: every photo-mode run, and what the operator said about it.

A separate file from the station's record (aoi/store.py). The two inspect
different things at different scales -- a registered stack at a fixed pitch
against an ordinary photograph at whatever scale teach measured -- and a
false-call rate that mixed them would describe neither.

Three design points:

* The machine verdict is never overwritten. The operator adjudicates each
  call, not the part and not the board, and `operator_status` is derived from
  those adjudications beside the stored verdict. A board where the machine said
  FAIL and the operator said "false call" to everything is a false call to be
  counted, which it cannot be if the FAIL was edited into a PASS.
* Adjudications are appended, not replaced. Undo marks a row undone instead of
  deleting it, so a mis-keyed D followed by an undo is still on record as a
  hesitation, and the latest non-undone row is the operator's answer.
* The schema carries a version. This file outlives the code that writes it, and
  a newer file opened by older code must stop with a reason rather than write
  rows the newer schema does not expect.

Keys, from the design:

    D  defect       the call is right
    F  false call   the part is fine (for a REVIEW call: "resolved OK")
    S  skip         decide later; the run stays REVIEW
    R  reshoot      board-level only: the photo is the problem, take another
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .model import BoardResult, Call

SCHEMA_VERSION = 2

# One script per version, applied in order. Never edit a shipped entry: a file
# already at that version will not run it again, so the change reaches only new
# files and the two diverge silently. Add the next number instead.
MIGRATIONS: dict[int, str] = {
    1: """
CREATE TABLE programs (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE
);

CREATE TABLE program_versions (
    id            INTEGER PRIMARY KEY,
    program_id    INTEGER NOT NULL REFERENCES programs(id),
    version       INTEGER NOT NULL,
    created       TEXT NOT NULL,
    path          TEXT NOT NULL,
    n_goldens     INTEGER NOT NULL,
    px_per_mm     REAL NOT NULL,
    single_golden INTEGER NOT NULL,
    UNIQUE (program_id, version)
);

CREATE TABLE runs (
    id                 INTEGER PRIMARY KEY,
    program_version_id INTEGER NOT NULL REFERENCES program_versions(id),
    image              TEXT NOT NULL,
    created            TEXT NOT NULL,
    machine_verdict    TEXT NOT NULL,
    reasons_json       TEXT NOT NULL,
    stamp              TEXT NOT NULL,
    superseded_by      INTEGER REFERENCES runs(id)
);

CREATE TABLE part_results (
    id       INTEGER PRIMARY KEY,
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    refdes   TEXT NOT NULL,
    type     TEXT NOT NULL,
    package  TEXT NOT NULL,
    state    TEXT NOT NULL
);
CREATE INDEX part_results_run ON part_results(run_id);

-- Board-level calls (registration, uninspected fraction, ...) have no part, so
-- part_result_id and refdes are NULL for them.
CREATE TABLE calls (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    part_result_id INTEGER REFERENCES part_results(id) ON DELETE CASCADE,
    refdes         TEXT,
    check_name     TEXT NOT NULL,
    state          TEXT NOT NULL,
    severity       TEXT NOT NULL,
    score          REAL,
    calibrated     INTEGER NOT NULL,
    detail         TEXT NOT NULL
);
CREATE INDEX calls_run ON calls(run_id);

CREATE TABLE adjudications (
    id          INTEGER PRIMARY KEY,
    call_id     INTEGER REFERENCES calls(id) ON DELETE CASCADE,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    board_level INTEGER NOT NULL,
    key         TEXT NOT NULL CHECK (key IN ('D', 'F', 'S', 'R')),
    at          TEXT NOT NULL,
    seconds     REAL,
    undone      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX adjudications_run ON adjudications(run_id);
CREATE INDEX adjudications_call ON adjudications(call_id);
""",
    # v2: where each run's report went, so Review can show a run from an earlier
    # session. NULL for runs recorded before the column existed.
    2: """
ALTER TABLE runs ADD COLUMN annotated TEXT;
ALTER TABLE runs ADD COLUMN html TEXT;
"""
}

PART_KEYS = ("D", "F", "S")
BOARD_KEYS = ("D", "F", "R")

# Severities an operator has to answer. INFO and OK are the machine saying
# nothing is wrong; asking a person to confirm them is how a queue gets buried.
NEEDS_ANSWER = ("FAIL", "REVIEW")

# A part in one of these states was not looked at, so it cannot have produced a
# false call and does not belong in the denominator of a false-call rate.
NOT_INSPECTED = ("NOT CHECKED", "UNINSPECTED")

# The latest answer on each call. MAX(id) and not MAX(at): `at` is
# second-precision, and an operator keying D then F inside one second would tie.
_LATEST = """
SELECT call_id, key FROM adjudications
WHERE id IN (SELECT MAX(id) FROM adjudications WHERE undone = 0 GROUP BY call_id)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Record:
    """Photo mode's local record. One SQLite file, no network."""

    def __init__(self, path: str | Path = "out/photo_aoi.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        try:
            self._migrate()
        except RuntimeError:
            self.db.close()
            raise

    def _migrate(self) -> None:
        """Bring the file up to SCHEMA_VERSION, or refuse one that is newer.

        Each version's script runs inside one transaction together with the
        row recording it, so a failure part-way leaves the file at the version
        it started at rather than half-way between two.
        """
        self.db.execute("CREATE TABLE IF NOT EXISTS schema_version ("
                        " version INTEGER NOT NULL, applied TEXT NOT NULL)")
        self.db.commit()
        stored = self.version()
        if stored > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} is schema version {stored}, newer than this code "
                f"understands ({SCHEMA_VERSION}); update the code, do not write "
                f"to this file with it")
        for v in range(stored + 1, SCHEMA_VERSION + 1):
            script = (f"BEGIN;\n{MIGRATIONS[v]}\n"
                      f"INSERT INTO schema_version (version, applied)"
                      f" VALUES ({v}, '{_now()}');\nCOMMIT;")
            try:
                self.db.executescript(script)
            except sqlite3.Error:
                self.db.rollback()
                raise

    def version(self) -> int:
        """The schema version stored in the file; 0 for a new one."""
        row = self.db.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        return int(row["v"] or 0)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Record":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ write
    def program_version(self, name: str, version: int, path: str | Path,
                        n_goldens: int, px_per_mm: float,
                        single_golden: bool) -> int:
        """The id of one taught program version, adding it the first time.

        A second call with the same name and version but a different golden
        count, scale or single-golden flag is refused: that is two different
        programs under one label, and every run recorded against it would be
        counted in stats as if they were the same. Reteaching bumps the version.
        """
        self.db.execute("INSERT OR IGNORE INTO programs (name) VALUES (?)", (name,))
        pid = self.db.execute("SELECT id FROM programs WHERE name = ?",
                              (name,)).fetchone()["id"]
        row = self.db.execute(
            "SELECT * FROM program_versions WHERE program_id = ? AND version = ?",
            (pid, int(version))).fetchone()
        if row is not None:
            stored = (row["n_goldens"], row["px_per_mm"], bool(row["single_golden"]))
            given = (int(n_goldens), float(px_per_mm), bool(single_golden))
            if stored != given:
                raise ValueError(
                    f"program {name!r} version {version} is already recorded with "
                    f"(n_goldens, px_per_mm, single_golden) = {stored}, not {given}")
            self.db.commit()
            return int(row["id"])
        cur = self.db.execute(
            "INSERT INTO program_versions (program_id, version, created, path,"
            " n_goldens, px_per_mm, single_golden) VALUES (?,?,?,?,?,?,?)",
            (pid, int(version), _now(), str(path), int(n_goldens),
             float(px_per_mm), int(bool(single_golden))))
        self.db.commit()
        return int(cur.lastrowid)

    def save_run(self, result: BoardResult, program_version_id: int) -> int:
        """Record one inspected photo: the run, every part, every call.

        The result names its program and version too, and the two have to
        agree; a run filed under the wrong program would put its false calls in
        another program's stats.
        """
        pv = self.db.execute(
            "SELECT p.name, v.version FROM program_versions v"
            " JOIN programs p ON p.id = v.program_id WHERE v.id = ?",
            (int(program_version_id),)).fetchone()
        if pv is None:
            raise KeyError(f"no program version {program_version_id}")
        if (pv["name"], pv["version"]) != (result.program, int(result.program_version)):
            raise ValueError(
                f"result is for {result.program!r} v{result.program_version}, but "
                f"program version {program_version_id} is {pv['name']!r} v{pv['version']}")

        with self.db:
            cur = self.db.execute(
                "INSERT INTO runs (program_version_id, image, created,"
                " machine_verdict, reasons_json, stamp) VALUES (?,?,?,?,?,?)",
                (int(program_version_id), result.image, _now(), result.verdict,
                 json.dumps(list(result.reasons)), result.stamp))
            run_id = int(cur.lastrowid)
            for pr in result.parts:
                p = pr.part
                cur = self.db.execute(
                    "INSERT INTO part_results (run_id, refdes, type, package, state)"
                    " VALUES (?,?,?,?,?)", (run_id, p.refdes, p.type, p.package, pr.state))
                prid = int(cur.lastrowid)
                for c in pr.calls:
                    self._insert_call(run_id, prid, p.refdes, c)
            for c in result.board_calls:
                self._insert_call(run_id, None, None, c)
        return run_id

    def set_report(self, run_id: int, annotated: str | Path, html: str | Path) -> None:
        """Remember where write_all put this run's annotated image and HTML."""
        with self.db:
            cur = self.db.execute("UPDATE runs SET annotated = ?, html = ? WHERE id = ?",
                                  (str(annotated), str(html), int(run_id)))
        if cur.rowcount == 0:
            raise KeyError(f"no run {run_id}")

    def runs(self, limit: int = 300) -> list[dict]:
        """Newest runs first, with program name, version and report paths."""
        rows = self.db.execute(
            "SELECT r.id, r.image, r.created, r.machine_verdict, r.stamp, r.superseded_by,"
            " r.annotated, r.html, p.name AS program, v.version"
            " FROM runs r JOIN program_versions v ON v.id = r.program_version_id"
            " JOIN programs p ON p.id = v.program_id ORDER BY r.id DESC LIMIT ?",
            (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def _insert_call(self, run_id: int, part_result_id: int | None,
                     refdes: str | None, c: Call) -> None:
        self.db.execute(
            "INSERT INTO calls (run_id, part_result_id, refdes, check_name, state,"
            " severity, score, calibrated, detail) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, part_result_id, refdes, c.check, c.state, c.severity,
             None if c.score is None else float(c.score), int(bool(c.calibrated)),
             c.detail))

    def adjudicate(self, call_id: int, key: str, seconds: float | None = None) -> None:
        """Record the operator's answer to one call. This is the labelled sample.

        R exists only at board level: a reshoot answers "the photo cannot be
        judged", which is a statement about the board, never about one part.
        """
        row = self.db.execute(
            "SELECT run_id, part_result_id IS NULL AS board_level FROM calls"
            " WHERE id = ?", (int(call_id),)).fetchone()
        if row is None:
            raise KeyError(f"no call {call_id}")
        board_level = bool(row["board_level"])
        allowed = BOARD_KEYS if board_level else PART_KEYS
        if key not in allowed:
            level = "board-level" if board_level else "part-level"
            raise ValueError(f"key for a {level} call must be one of "
                             f"{', '.join(allowed)}; got {key!r}")
        self.db.execute(
            "INSERT INTO adjudications (call_id, run_id, board_level, key, at, seconds)"
            " VALUES (?,?,?,?,?,?)",
            (int(call_id), int(row["run_id"]), int(board_level), key, _now(),
             None if seconds is None else float(seconds)))
        self.db.commit()

    def undo_last(self, run_id: int) -> bool:
        """Mark the most recent answer on this run undone. False if none is left."""
        row = self.db.execute(
            "SELECT MAX(id) AS id FROM adjudications WHERE run_id = ? AND undone = 0",
            (int(run_id),)).fetchone()
        if row["id"] is None:
            return False
        self.db.execute("UPDATE adjudications SET undone = 1 WHERE id = ?", (row["id"],))
        self.db.commit()
        return True

    def supersede(self, run_id: int, new_run_id: int) -> None:
        """Point a run at the reshoot that replaces it. Both must exist."""
        if int(run_id) == int(new_run_id):
            raise ValueError("a run cannot supersede itself")
        for rid in (run_id, new_run_id):
            if self.db.execute("SELECT 1 FROM runs WHERE id = ?", (int(rid),)).fetchone() is None:
                raise KeyError(f"no run {rid}")
        self.db.execute("UPDATE runs SET superseded_by = ? WHERE id = ?",
                        (int(new_run_id), int(run_id)))
        self.db.commit()

    # ------------------------------------------------------------------- read
    def calls(self, run_id: int) -> list[dict]:
        """Every call on a run, part-level first, each with its current answer."""
        rows = self.db.execute(
            f"SELECT c.*, l.key FROM calls c LEFT JOIN ({_LATEST}) l ON l.call_id = c.id"
            " WHERE c.run_id = ? ORDER BY c.part_result_id IS NULL, c.id",
            (int(run_id),)).fetchall()
        return [{
            "id": r["id"],
            "refdes": r["refdes"],
            "check": r["check_name"],
            "state": r["state"],
            "severity": r["severity"],
            "score": r["score"],
            "calibrated": bool(r["calibrated"]),
            "detail": r["detail"],
            "board_level": r["part_result_id"] is None,
            "adjudication": r["key"],
        } for r in rows]

    def operator_status(self, run_id: int) -> str:
        """The board as the operator has left it, beside the machine verdict.

        The rule, from the design: FAIL if any call is D; PASS once every FAIL,
        REVIEW and NOT CHECKED call (and any board-level REVIEW) is F; REVIEW
        while any is unadjudicated, skipped, or -- at board level -- sent for a
        reshoot. A reshoot is resolved by superseding the run, after which the
        run's status is SUPERSEDED whatever was keyed on it: the new photo is
        the board's record now. INFO and OK calls need no answer, so a run with
        no FAIL or REVIEW call is PASS.
        """
        run = self.db.execute("SELECT superseded_by FROM runs WHERE id = ?",
                              (int(run_id),)).fetchone()
        if run is None:
            raise KeyError(f"no run {run_id}")
        if run["superseded_by"] is not None:
            return "SUPERSEDED"
        keys = [c["adjudication"] for c in self.calls(run_id)
                if c["severity"] in NEEDS_ANSWER]
        if "D" in keys:
            return "FAIL"
        if all(k == "F" for k in keys):
            return "PASS"
        return "REVIEW"

    def stats(self) -> list[dict]:
        """False-call numbers per part group (type and package), live runs only.

        The false-call numerator is FAIL calls the operator answered F. A REVIEW
        answered F is the machine asking and being told "fine", which is what a
        REVIEW is for; it is counted as `review_resolved_ok`, not as a false
        call. The denominator is parts actually inspected: a NOT CHECKED or
        UNINSPECTED part was never judged, so it cannot have been misjudged.

        There is no escape count. A defect the machine passed never reaches the
        operator, so this file cannot see it; only a hand-inspected sample can.

        Board-level calls belong to no group and are left out here and in
        `stats_total`. Superseded runs are left out too: the reshoot that
        replaced one is the board's record, and counting both would count the
        board twice.
        """
        # The same string Part.group builds, so a row here names the group the
        # judge and the report use.
        groups = self._tally("TRIM(p.type || ' ' || p.package)")
        return [dict(group=g, **groups[g]) for g in sorted(groups)]

    def stats_total(self) -> dict:
        """The same numbers as `stats`, over every group at once."""
        total = self._tally("'all'")
        return dict(group="all", **total.get("all", self._blank()))

    def _tally(self, group_expr: str) -> dict[str, dict]:
        """Group-keyed counts behind `stats` and `stats_total`.

        `group_expr` is a fixed SQL expression chosen by the two callers above,
        never user input.
        """
        live = "r.superseded_by IS NULL"
        out: dict[str, dict] = {}
        for r in self.db.execute(
                f"SELECT {group_expr} AS g,"
                f" SUM(p.state NOT IN ({', '.join(repr(s) for s in NOT_INSPECTED)}))"
                f" AS inspected,"
                f" COUNT(DISTINCT p.run_id) AS boards"
                f" FROM part_results p JOIN runs r ON r.id = p.run_id"
                f" WHERE {live} GROUP BY g").fetchall():
            out[r["g"]] = self._blank()
            out[r["g"]]["parts_inspected"] = int(r["inspected"] or 0)
            out[r["g"]]["boards"] = int(r["boards"])
        for r in self.db.execute(
                f"SELECT {group_expr} AS g,"
                f" SUM(c.severity = 'FAIL') AS fails,"
                f" SUM(c.severity = 'FAIL' AND l.key = 'F') AS false_calls,"
                f" SUM(c.severity = 'REVIEW') AS reviews,"
                f" SUM(c.severity = 'REVIEW' AND l.key = 'F') AS reviews_ok"
                f" FROM calls c JOIN part_results p ON p.id = c.part_result_id"
                f" JOIN runs r ON r.id = c.run_id"
                f" LEFT JOIN ({_LATEST}) l ON l.call_id = c.id"
                f" WHERE {live} GROUP BY g").fetchall():
            s = out[r["g"]]
            s["fail_calls"] = int(r["fails"] or 0)
            s["false_calls"] = int(r["false_calls"] or 0)
            s["review_calls"] = int(r["reviews"] or 0)
            s["review_resolved_ok"] = int(r["reviews_ok"] or 0)
        for s in out.values():
            s["fcr"] = s["false_calls"] / s["parts_inspected"] if s["parts_inspected"] else 0.0
            s["false_calls_per_board"] = s["false_calls"] / s["boards"] if s["boards"] else 0.0
        return out

    @staticmethod
    def _blank() -> dict:
        return {"parts_inspected": 0, "fail_calls": 0, "false_calls": 0, "fcr": 0.0,
                "review_calls": 0, "review_resolved_ok": 0, "boards": 0,
                "false_calls_per_board": 0.0}

    def labels(self, check_name: str) -> list[tuple[float, int]]:
        """(score, 1 if defect else 0) for one check, for Platt calibration.

        Only calls with a score and a D or F answer: a skip is not a label, and
        a call with no score has nothing to calibrate. Superseded runs are kept.
        The photo was replaced, but the operator still looked at that crop and
        said what it was, and that is exactly the sample calibration needs.
        """
        rows = self.db.execute(
            f"SELECT c.score, l.key FROM calls c JOIN ({_LATEST}) l ON l.call_id = c.id"
            " WHERE c.check_name = ? AND c.score IS NOT NULL AND l.key IN ('D', 'F')"
            " ORDER BY c.id", (check_name,)).fetchall()
        return [(float(r["score"]), 1 if r["key"] == "D" else 0) for r in rows]
