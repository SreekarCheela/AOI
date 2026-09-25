"""photo_aoi.db: runs, adjudications, and the numbers that come out of them.

Every result here is hand-built. What is under test is the bookkeeping -- which
calls need an answer, which answer is current, what counts as a false call --
and a wrong count there reads as a real false-call rate, so it is pinned down
row by row against the rule in the design.
"""
from __future__ import annotations

import sqlite3

import pytest

from aoi.photo import record
from aoi.photo.model import BoardResult, Call, Part, PartResult
from aoi.photo.record import Record


def _part(refdes, type_="resistor", package="0603"):
    return Part(refdes, x=10, y=10, w=20, h=10, type=type_, package=package)


def _result(parts, board_calls=(), verdict="FAIL", image="board_01.jpg"):
    return BoardResult(image=image, program="demo", program_version=1,
                       verdict=verdict, reasons=["R1 MISSING"], parts=list(parts),
                       board_calls=list(board_calls), stamp="SINGLE GOLDEN")


@pytest.fixture
def rec(tmp_path):
    with Record(tmp_path / "out" / "photo_aoi.db") as r:
        yield r


@pytest.fixture
def pv(rec):
    return rec.program_version("demo", 1, "programs/demo", 3, 45.0, False)


def _save(rec, pv, parts, board_calls=()):
    run = rec.save_run(_result(parts, board_calls), pv)
    return run, {(c["refdes"], c["check"]): c["id"] for c in rec.calls(run)}


# ------------------------------------------------------------------ schema
def test_schema_created_at_current_version(tmp_path):
    path = tmp_path / "nested" / "photo_aoi.db"
    with Record(path) as r:
        assert r.version() == record.SCHEMA_VERSION == 2
        names = {row[0] for row in r.db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"schema_version", "programs", "program_versions", "runs",
            "part_results", "calls", "adjudications"} <= names
    # reopening does not reapply the migration
    with Record(path) as r:
        assert [row[0] for row in r.db.execute("SELECT version FROM schema_version")] == [1, 2]


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "photo_aoi.db"
    Record(path).close()
    db = sqlite3.connect(path)
    db.execute("INSERT INTO schema_version (version, applied) VALUES (99, 'later')")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="newer"):
        Record(path)


def test_program_version_is_insert_or_get(rec):
    a = rec.program_version("demo", 1, "p", 3, 45.0, False)
    assert rec.program_version("demo", 1, "p", 3, 45.0, False) == a
    assert rec.program_version("demo", 2, "p", 3, 45.0, False) != a
    with pytest.raises(ValueError, match="already recorded"):
        rec.program_version("demo", 1, "p", 5, 45.0, False)


def test_save_run_refuses_a_mismatched_program(rec):
    other = rec.program_version("other", 1, "p", 3, 45.0, False)
    with pytest.raises(ValueError, match="demo"):
        rec.save_run(_result([]), other)


# ------------------------------------------------------------------ round trip
def test_save_run_round_trip(rec, pv):
    parts = [
        PartResult(_part("R1"), [Call("missing", "MISSING", "no body", 0.97)]),
        PartResult(_part("C1", "capacitor", "0805"),
                   [Call("polarity", "POLARITY REVIEW", "", 0.6, calibrated=True),
                    Call("offset", "OK")]),
        PartResult(_part("U1", "ic", "SOIC-8"), []),
    ]
    board = [Call("view", "NOT CHECKED", "glare over 12% of parts")]
    run, _ = _save(rec, pv, parts, board)

    calls = rec.calls(run)
    assert [(c["refdes"], c["check"], c["state"], c["severity"]) for c in calls] == [
        ("R1", "missing", "MISSING", "FAIL"),
        ("C1", "polarity", "POLARITY REVIEW", "REVIEW"),
        ("C1", "offset", "OK", "OK"),
        (None, "view", "NOT CHECKED", "REVIEW"),
    ]
    assert calls[0]["score"] == pytest.approx(0.97) and calls[0]["calibrated"] is False
    assert calls[1]["calibrated"] is True and calls[2]["score"] is None
    assert [c["board_level"] for c in calls] == [False, False, False, True]
    assert all(c["adjudication"] is None for c in calls)
    assert rec.db.execute("SELECT COUNT(*) FROM part_results WHERE run_id = ?",
                          (run,)).fetchone()[0] == 3
    row = rec.db.execute("SELECT * FROM runs WHERE id = ?", (run,)).fetchone()
    assert (row["image"], row["machine_verdict"], row["stamp"]) == \
        ("board_01.jpg", "FAIL", "SINGLE GOLDEN")
    assert row["superseded_by"] is None


# ------------------------------------------------------------------ operator status
@pytest.fixture
def flagged_run(rec, pv):
    """A FAIL, a REVIEW, a NOT CHECKED, an INFO and an OK on one board."""
    parts = [
        PartResult(_part("R1"), [Call("missing", "MISSING", score=0.9)]),
        PartResult(_part("D1", "diode", "SOD-123"), [Call("polarity", "POLARITY REVIEW")]),
        PartResult(_part("R2"), [Call("glare", "NOT CHECKED")]),
        PartResult(_part("R3"), [Call("marking", "UNINSPECTED")]),
        PartResult(_part("R4"), [Call("offset", "OK")]),
    ]
    return _save(rec, pv, parts)


def test_defect_on_any_call_is_fail(rec, flagged_run):
    run, ids = flagged_run
    rec.adjudicate(ids["D1", "polarity"], "D")
    assert rec.operator_status(run) == "FAIL"


def test_every_answer_false_call_is_pass(rec, flagged_run):
    run, ids = flagged_run
    for k in [("R1", "missing"), ("D1", "polarity"), ("R2", "glare")]:
        rec.adjudicate(ids[k], "F", seconds=2.0)
    # INFO (UNINSPECTED) and OK were never answered and did not need to be
    assert rec.operator_status(run) == "PASS"


def test_pending_or_skipped_stays_review(rec, flagged_run):
    run, ids = flagged_run
    assert rec.operator_status(run) == "REVIEW"
    rec.adjudicate(ids["R1", "missing"], "F")
    rec.adjudicate(ids["D1", "polarity"], "F")
    assert rec.operator_status(run) == "REVIEW"       # NOT CHECKED still open
    rec.adjudicate(ids["R2", "glare"], "S")
    assert rec.operator_status(run) == "REVIEW"       # a skip is not an answer


def test_run_with_nothing_flagged_is_pass(rec, pv):
    run, _ = _save(rec, pv, [PartResult(_part("R1"), [Call("offset", "OK")]),
                             PartResult(_part("R2"), [Call("marking", "UNINSPECTED")])])
    assert rec.operator_status(run) == "PASS"


def test_board_level_reshoot_then_supersede(rec, pv):
    board = [Call("view", "NOT CHECKED", "registration overlap 0.42")]
    run, ids = _save(rec, pv, [PartResult(_part("R1"), [Call("offset", "OK")])], board)
    rec.adjudicate(ids[None, "view"], "R")
    assert rec.operator_status(run) == "REVIEW"
    new, _ = _save(rec, pv, [PartResult(_part("R1"), [Call("offset", "OK")])])
    rec.supersede(run, new)
    assert rec.operator_status(run) == "SUPERSEDED"
    assert rec.operator_status(new) == "PASS"


def test_board_level_false_call_passes(rec, pv):
    run, ids = _save(rec, pv, [], [Call("view", "NOT CHECKED")])
    rec.adjudicate(ids[None, "view"], "F")
    assert rec.operator_status(run) == "PASS"


# ------------------------------------------------------------------ keys and undo
def test_invalid_keys_are_rejected(rec, pv):
    run, ids = _save(rec, pv, [PartResult(_part("R1"), [Call("missing", "MISSING")])],
                     [Call("view", "NOT CHECKED")])
    with pytest.raises(ValueError, match="D, F, S"):
        rec.adjudicate(ids["R1", "missing"], "R")     # reshoot is board-level only
    with pytest.raises(ValueError, match="D, F, R"):
        rec.adjudicate(ids[None, "view"], "S")
    with pytest.raises(ValueError, match="D, F, S"):
        rec.adjudicate(ids["R1", "missing"], "x")
    with pytest.raises(KeyError):
        rec.adjudicate(99999, "D")


def test_undo_restores_the_previous_answer(rec, flagged_run):
    run, ids = flagged_run
    cid = ids["R1", "missing"]
    rec.adjudicate(cid, "F")
    rec.adjudicate(cid, "D")
    assert rec.operator_status(run) == "FAIL"
    assert rec.undo_last(run) is True
    assert {c["id"]: c["adjudication"] for c in rec.calls(run)}[cid] == "F"
    assert rec.undo_last(run) is True
    assert {c["id"]: c["adjudication"] for c in rec.calls(run)}[cid] is None
    assert rec.undo_last(run) is False
    # undone rows are kept, not deleted
    assert rec.db.execute("SELECT COUNT(*) FROM adjudications WHERE undone = 1"
                          ).fetchone()[0] == 2


def test_supersede_checks_both_runs(rec, flagged_run):
    run, _ = flagged_run
    with pytest.raises(KeyError):
        rec.supersede(run, 99999)
    with pytest.raises(ValueError):
        rec.supersede(run, run)


# ------------------------------------------------------------------ stats
def test_stats_numerator_and_denominator(rec, pv):
    parts = [
        PartResult(_part("R1"), [Call("missing", "MISSING", score=0.9)]),
        PartResult(_part("R2"), [Call("offset", "OFFSET", score=0.7)]),
        PartResult(_part("R3"), [Call("marking", "MARKING DIFFERS", score=0.5)]),
        PartResult(_part("R4"), [Call("glare", "NOT CHECKED")]),
        PartResult(_part("R5"), [Call("marking", "UNINSPECTED")]),
        PartResult(_part("R6"), [Call("offset", "OK")]),
        PartResult(_part("C1", "capacitor", "0805"), [Call("offset", "OK")]),
    ]
    run, ids = _save(rec, pv, parts, [Call("view", "NOT CHECKED")])
    rec.adjudicate(ids["R1", "missing"], "F")     # FAIL answered F: a false call
    rec.adjudicate(ids["R2", "offset"], "D")      # a real defect
    rec.adjudicate(ids["R3", "marking"], "F")     # REVIEW resolved OK, not a false call
    rec.adjudicate(ids["R4", "glare"], "F")       # NOT CHECKED resolved OK
    rec.adjudicate(ids[None, "view"], "F")        # board-level: in no group

    by = {s["group"]: s for s in rec.stats()}
    assert set(by) == {"resistor 0603", "capacitor 0805"}
    r = by["resistor 0603"]
    # R4 NOT CHECKED and R5 UNINSPECTED were never judged: 6 parts -> 4 inspected
    assert r["parts_inspected"] == 4
    assert r["fail_calls"] == 2
    assert r["false_calls"] == 1
    assert r["fcr"] == pytest.approx(1 / 4)
    assert r["review_calls"] == 2
    assert r["review_resolved_ok"] == 2
    assert r["boards"] == 1
    assert r["false_calls_per_board"] == pytest.approx(1.0)
    assert by["capacitor 0805"]["parts_inspected"] == 1
    assert by["capacitor 0805"]["fcr"] == 0.0

    total = rec.stats_total()
    assert total["parts_inspected"] == 5 and total["false_calls"] == 1
    assert total["fcr"] == pytest.approx(1 / 5)
    assert total["boards"] == 1
    assert set(total) == set(r)


def test_stats_follow_undo_and_skip_superseded_runs(rec, pv):
    parts = [PartResult(_part("R1"), [Call("missing", "MISSING", score=0.9)])]
    run, ids = _save(rec, pv, parts)
    rec.adjudicate(ids["R1", "missing"], "F")
    assert rec.stats_total()["false_calls"] == 1
    rec.adjudicate(ids["R1", "missing"], "D")
    assert rec.stats_total()["false_calls"] == 0      # the latest answer counts
    rec.undo_last(run)
    assert rec.stats_total()["false_calls"] == 1

    new, _ = _save(rec, pv, [PartResult(_part("R1"), [Call("offset", "OK")])])
    rec.supersede(run, new)
    total = rec.stats_total()
    assert total["false_calls"] == 0 and total["boards"] == 1
    assert total["parts_inspected"] == 1


def test_stats_on_an_empty_record(rec):
    assert rec.stats() == []
    total = rec.stats_total()
    assert total["parts_inspected"] == 0 and total["fcr"] == 0.0
    assert total["false_calls_per_board"] == 0.0


# ------------------------------------------------------------------ labels
def test_labels_for_calibration(rec, pv):
    parts = [
        PartResult(_part("R1"), [Call("missing", "MISSING", score=0.9)]),
        PartResult(_part("R2"), [Call("missing", "MISSING", score=0.4)]),
        PartResult(_part("R3"), [Call("missing", "MISSING", score=0.8)]),   # skipped
        PartResult(_part("R4"), [Call("missing", "MISSING")]),              # no score
        PartResult(_part("R5"), [Call("missing", "MISSING", score=0.7)]),   # unanswered
        PartResult(_part("R6"), [Call("offset", "OFFSET", score=0.6)]),     # other check
    ]
    run, ids = _save(rec, pv, parts)
    rec.adjudicate(ids["R1", "missing"], "D")
    rec.adjudicate(ids["R2", "missing"], "F")
    rec.adjudicate(ids["R3", "missing"], "S")
    rec.adjudicate(ids["R4", "missing"], "D")
    rec.adjudicate(ids["R6", "offset"], "D")
    assert rec.labels("missing") == [(pytest.approx(0.9), 1), (pytest.approx(0.4), 0)]
    assert rec.labels("offset") == [(pytest.approx(0.6), 1)]
    assert rec.labels("tombstone") == []


def test_v1_file_migrates_to_v2_and_keeps_report_paths(tmp_path):
    """A photo_aoi.db written at schema 1 gains the report-path columns in place."""
    import sqlite3
    from aoi.photo import record
    f = tmp_path / "old.db"
    con = sqlite3.connect(f)
    con.executescript("CREATE TABLE schema_version (version INTEGER NOT NULL, applied TEXT);"
                      + record.MIGRATIONS[1]
                      + "INSERT INTO schema_version (version, applied) VALUES (1, 'then');")
    con.close()
    with record.Record(f) as rec:
        assert rec.version() == record.SCHEMA_VERSION == 2
        cols = {r[1] for r in rec.db.execute("PRAGMA table_info(runs)")}
        assert {"annotated", "html"} <= cols
        assert rec.runs() == []
