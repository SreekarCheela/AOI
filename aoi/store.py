"""Local inspection record: SQLite, on the machine, no network (spec section 5).

This is the half of the operator station that matters. The review screen is a
way of getting at it, but the record is what turns a rule-based inspector into
something that improves: every fail an operator adjudicates becomes a labelled
crop, and those labels are what the false-call filter of section 6C trains on.
Rules alone plateau at 3-8% false calls; operator labels are what take it under
1%, and none of that can start until the labels are being stored.

Three design points, all from the spec:

* Inspection stays local. SQLite, one file, no server in the loop. Yield data
  gets pushed upward for reporting, but a network outage must not stop a board
  being inspected or reviewed.
* Store crops, not frames. A full board of raw frames is about a gigabyte and
  fills any disk within a week; the 96x96 patches are about 5 MB compressed and
  are exactly what the retraining loop needs.
* An adjudication is a labelled sample. It is recorded with what the machine
  said, what the operator said, and how long they took -- the last one because
  section 10 sets an exit criterion in seconds.
"""
from __future__ import annotations

import sqlite3
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS board (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    serial        TEXT,
    inspected_at  TEXT NOT NULL,
    seconds       REAL NOT NULL,
    align_dx_mm   REAL,
    align_dy_mm   REAL,
    align_rot_deg REAL,
    n_components  INTEGER NOT NULL,
    n_flagged     INTEGER NOT NULL,
    n_untestable  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS verdict (
    id        INTEGER PRIMARY KEY,
    board_id  INTEGER NOT NULL REFERENCES board(id) ON DELETE CASCADE,
    refdes    TEXT NOT NULL,
    package   TEXT NOT NULL,
    failed    INTEGER NOT NULL,
    testable  INTEGER NOT NULL,
    reason    TEXT NOT NULL,
    score     REAL NOT NULL,
    detail    TEXT,
    features  BLOB,
    crop      BLOB
);
CREATE INDEX IF NOT EXISTS verdict_board ON verdict(board_id);

CREATE TABLE IF NOT EXISTS adjudication (
    verdict_id INTEGER PRIMARY KEY REFERENCES verdict(id) ON DELETE CASCADE,
    call       TEXT NOT NULL CHECK (call IN ('defect', 'false_call')),
    operator   TEXT,
    decided_at TEXT NOT NULL,
    seconds    REAL
);

-- One representative good patch per package, so review can put the part it is
-- asking about next to what correct looks like. An operator deciding from the
-- crop alone is guessing.
CREATE TABLE IF NOT EXISTS golden_ref (
    package TEXT PRIMARY KEY,
    crop    BLOB NOT NULL
);
"""


# ------------------------------------------------------------------ crop codec
def encode_crop(patch: np.ndarray) -> bytes:
    """(H, W, C) uint8 -> PNG bytes of the channels laid out side by side.

    PNG rather than raw so the blob is both compressed and openable by anything;
    a strip rather than four blobs so one crop is one row in one column.
    """
    a = np.asarray(patch, np.uint8)
    if a.ndim == 2:
        a = a[..., None]
    strip = np.hstack([a[..., i] for i in range(a.shape[2])])
    ok, buf = cv2.imencode(".png", strip)
    if not ok:
        raise ValueError("could not encode crop")
    return buf.tobytes()


def decode_crop(blob: bytes, channels: int = 4) -> np.ndarray:
    """PNG strip -> (H, W, C) uint8."""
    strip = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
    if strip is None:
        raise ValueError("could not decode crop")
    w = strip.shape[1] // channels
    return np.stack([strip[:, i * w:(i + 1) * w] for i in range(channels)], axis=-1)


# A component taking far longer than the rest of its board was not being
# reviewed for that whole time: the operator was orienting, or interrupted.
OUTLIER_MULTIPLE = 5.0
OUTLIER_FLOOR_S = 10.0

# Section 10: a full board's fails, under a minute.
REVIEW_TARGET_S = 60.0


@dataclass(frozen=True)
class ReviewTiming:
    """What a board's review actually cost, and whether it met the target."""
    board_id: int
    adjudicated: int
    total_s: float           # wall time, everything included
    median_s: float          # per component
    throughput_s: float      # median x count: the rate actually sustained
    slowest_s: float
    slowest_position: int = 0    # 1-based, in decision order
    outliers: int = 0            # components far above the median

    @property
    def meets_target(self) -> bool:
        """Judged on sustained rate, not on wall time."""
        return self.adjudicated > 0 and self.throughput_s < REVIEW_TARGET_S

    @property
    def note(self) -> str:
        """One line saying what the numbers mean, including when they disagree."""
        if not self.adjudicated:
            return "nothing adjudicated"
        verdict = ("under" if self.meets_target else "OVER") + \
                  f" the {REVIEW_TARGET_S:.0f} s target"
        if not self.outliers:
            return f"{self.throughput_s:.0f} s for {self.adjudicated}, {verdict}"
        where = ("orientation, it was the first component"
                 if self.slowest_position == 1
                 else f"an interruption at component {self.slowest_position}")
        return (f"{self.throughput_s:.0f} s for {self.adjudicated} at the rate "
                f"actually sustained ({self.median_s:.1f} s each), {verdict}. "
                f"Wall time was {self.total_s:.0f} s: {self.outliers} component"
                f"{'s' if self.outliers > 1 else ''} ran long, the slowest "
                f"{self.slowest_s:.0f} s -- {where}.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """The station's local record. One SQLite file."""

    def __init__(self, path: str | Path = "out/station.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ write
    def save_golden(self, refs: dict[str, np.ndarray]) -> None:
        """Store one reference patch per package, from teach."""
        self.db.executemany(
            "INSERT OR REPLACE INTO golden_ref (package, crop) VALUES (?, ?)",
            [(k, encode_crop(v)) for k, v in refs.items()])
        self.db.commit()

    def save_result(self, board, result, patches: np.ndarray | None = None,
                    serial: str | None = None) -> int:
        """Record one inspected board and every component verdict on it.

        Crops are kept for components that FAILED or could not be tested -- the
        ones a person will look at and the ones worth retraining on. Keeping a
        patch for all 500 good components too would be storing the same picture
        of a correct 0402 several hundred times a board.
        """
        cur = self.db.execute(
            "INSERT INTO board (name, serial, inspected_at, seconds, align_dx_mm,"
            " align_dy_mm, align_rot_deg, n_components, n_flagged, n_untestable)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (board.name, serial, _now(), float(result.seconds),
             float(result.align_shift_mm[0]), float(result.align_shift_mm[1]),
             float(result.align_rot_deg), len(result.verdicts),
             len(result.failed), len(result.untestable)))
        board_id = int(cur.lastrowid)

        rows = []
        for i, v in enumerate(result.verdicts):
            keep = v.failed or not v.testable
            crop = encode_crop(patches[i]) if (keep and patches is not None) else None
            feats = (result.measures[i].vector().tobytes()
                     if i < len(result.measures) else None)
            rows.append((board_id, v.refdes, v.package, int(v.failed),
                         int(v.testable), v.reason, float(v.score), v.detail,
                         feats, crop))
        self.db.executemany(
            "INSERT INTO verdict (board_id, refdes, package, failed, testable,"
            " reason, score, detail, features, crop) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows)
        self.db.commit()
        return board_id

    def adjudicate(self, verdict_id: int, call: str, operator: str = "",
                   seconds: float | None = None) -> None:
        """Record an operator's decision. This is the labelled sample."""
        if call not in ("defect", "false_call"):
            raise ValueError(f"call must be 'defect' or 'false_call', got {call!r}")
        self.db.execute(
            "INSERT OR REPLACE INTO adjudication (verdict_id, call, operator,"
            " decided_at, seconds) VALUES (?,?,?,?,?)",
            (int(verdict_id), call, operator, _now(),
             None if seconds is None else float(seconds)))
        self.db.commit()

    # ------------------------------------------------------------------- read
    def latest_board(self) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM board ORDER BY id DESC LIMIT 1").fetchone()

    def review_queue(self, board_id: int) -> list[sqlite3.Row]:
        """Components a person still has to decide on.

        Failed and testable only. A component with no inspection program is a
        coverage gap in the program, not a defect on the board; putting those in
        the queue is how one unprogrammed package buries the real fails.
        """
        return self.db.execute(
            "SELECT v.* FROM verdict v"
            " LEFT JOIN adjudication a ON a.verdict_id = v.id"
            " WHERE v.board_id = ? AND v.failed = 1 AND v.testable = 1"
            "   AND a.verdict_id IS NULL"
            " ORDER BY v.refdes", (board_id,)).fetchall()

    def untestable(self, board_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM verdict WHERE board_id = ? AND testable = 0"
            " ORDER BY refdes", (board_id,)).fetchall()

    def golden_ref(self, package: str) -> np.ndarray | None:
        row = self.db.execute(
            "SELECT crop FROM golden_ref WHERE package = ?", (package,)).fetchone()
        return decode_crop(row["crop"]) if row else None

    # ------------------------------------------------- the point of all of it
    def export_labels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adjudicated crops as training data for the false-call filter.

        Returns (patches, labels, packages) with label 1 = real defect,
        0 = false call. This is a naturally balanced and naturally abundant set
        because it only ever contains components the rule layer already failed
        (spec section 6C).
        """
        rows = self.db.execute(
            "SELECT v.package, v.crop, a.call FROM adjudication a"
            " JOIN verdict v ON v.id = a.verdict_id"
            " WHERE v.crop IS NOT NULL ORDER BY a.verdict_id").fetchall()
        if not rows:
            return (np.zeros((0, 0, 0, 0), np.uint8),
                    np.zeros(0, np.int64), np.zeros(0, dtype="<U16"))
        patches = np.stack([decode_crop(r["crop"]) for r in rows])
        labels = np.array([1 if r["call"] == "defect" else 0 for r in rows], np.int64)
        packages = np.array([r["package"] for r in rows])
        return patches, labels, packages

    def yield_metrics(self) -> list[dict]:
        """Per-package numbers, which is how section 8 says to track this.

        False call rate is the fraction of flagged components an operator judged
        good. It is only meaningful once adjudications exist -- before that the
        machine has an opinion and nobody has checked it.

        Escape rate is deliberately absent: it can only be measured against a
        hand-inspected sample, which is an input this store does not have.
        """
        rows = self.db.execute("""
            SELECT v.package                                   AS package,
                   COUNT(*)                                    AS inspected,
                   SUM(v.failed)                               AS flagged,
                   SUM(CASE WHEN v.testable = 0 THEN 1 ELSE 0 END) AS untestable,
                   SUM(CASE WHEN a.call = 'false_call' THEN 1 ELSE 0 END) AS false_calls,
                   SUM(CASE WHEN a.call = 'defect'     THEN 1 ELSE 0 END) AS defects,
                   AVG(a.seconds)                              AS avg_review_s
            FROM verdict v
            LEFT JOIN adjudication a ON a.verdict_id = v.id
            GROUP BY v.package ORDER BY v.package""").fetchall()
        out = []
        for r in rows:
            judged = (r["false_calls"] or 0) + (r["defects"] or 0)
            out.append({
                "package": r["package"],
                "inspected": r["inspected"],
                "flagged": r["flagged"] or 0,
                "untestable": r["untestable"] or 0,
                "adjudicated": judged,
                "false_calls": r["false_calls"] or 0,
                "defects": r["defects"] or 0,
                # of everything inspected, how much did a person judge to be a
                # needless stop? That is the number section 8 is about.
                "false_call_rate": ((r["false_calls"] or 0) / r["inspected"]
                                    if r["inspected"] else 0.0),
                "avg_review_s": r["avg_review_s"],
            })
        return out

    def board_review_seconds(self, board_id: int) -> float:
        """Wall time an operator spent on one board, outliers and all."""
        row = self.db.execute(
            "SELECT COALESCE(SUM(a.seconds), 0.0) AS s FROM adjudication a"
            " JOIN verdict v ON v.id = a.verdict_id WHERE v.board_id = ?",
            (board_id,)).fetchone()
        return float(row["s"])

    def review_timing(self, board_id: int) -> ReviewTiming:
        """How fast a board was reviewed, measured so one pause cannot decide it.

        Section 10's criterion is a full board's fails in under sixty seconds by
        an operator with thirty minutes of training. A raw sum does not measure
        that. The first component on an operator's first board absorbs all the
        orientation -- reading the layout, learning what the two panels mean,
        finding the keys -- and an interruption anywhere lands entirely on
        whichever component happened to be on screen. Both are wall time and
        neither is review time, and a sum cannot tell any of them apart.

        Observed: a 12-fail board took 85.8 s, of which the FIRST component took
        60.2 s and the other eleven averaged 2.3 s. The sum says the operator
        failed the criterion. The operator was reviewing at roughly 26 s per
        board.

        So the target is judged on the median per component times the number of
        components -- the rate the operator actually sustained. The raw total is
        reported alongside, never hidden, and anything far above the median is
        counted and located rather than quietly dropped: one slow component in
        first position is orientation, the same component in tenth position is
        an interruption, and they call for different responses.
        """
        rows = self.db.execute(
            "SELECT a.seconds FROM adjudication a JOIN verdict v ON v.id = a.verdict_id"
            " WHERE v.board_id = ? AND a.seconds IS NOT NULL"
            " ORDER BY a.decided_at, a.verdict_id", (board_id,)).fetchall()
        secs = [float(r["seconds"]) for r in rows]
        if not secs:
            return ReviewTiming(board_id, 0, 0.0, 0.0, 0.0, 0.0, 0, 0)

        median = float(statistics.median(secs))
        # Scale-relative, with an absolute floor so a board where every decision
        # takes half a second does not flag a three-second one as an anomaly.
        limit = max(OUTLIER_MULTIPLE * median, OUTLIER_FLOOR_S)
        outliers = [i for i, v in enumerate(secs) if v > limit]
        slowest = max(range(len(secs)), key=lambda i: secs[i])
        return ReviewTiming(
            board_id=board_id,
            adjudicated=len(secs),
            total_s=float(sum(secs)),
            median_s=median,
            throughput_s=median * len(secs),
            slowest_s=secs[slowest],
            slowest_position=slowest + 1,
            outliers=len(outliers),
        )
