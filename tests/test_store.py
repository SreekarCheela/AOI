"""The local inspection record, and the labels that come out of it.

This is the half of the operator station that matters. The screen is a way of
getting at the record; the record is what turns a rule-based inspector into
something that improves. Section 6C's false-call filter trains on adjudicated
crops, and nothing about that can start until adjudications are being stored,
which is why this is tested far harder than the UI on top of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from aoi import pipeline, roi, rules, store, synth

from conftest import render_with


@pytest.fixture
def db(tmp_path):
    with store.Store(tmp_path / "station.db") as s:
        yield s


@pytest.fixture
def inspected(db, brd, golden):
    """One board with a few real defects, recorded exactly as the CLI does."""
    defects = {"R1": synth.Defect("missing"),
               "R3": synth.Defect("offset", dx=0.25),
               "R4": synth.Defect("insufficient", pad=1)}
    stack = synth.render(brd, defects=defects,
                         opts=synth.RenderOpts(seed=61, place_sigma_um=0.0))
    res = pipeline.Inspector(golden).inspect(stack, brd)
    patches = np.stack([roi.extract_patch(res.aligned, c, c.pkg)
                        for c in brd.components])
    bid = db.save_result(brd, res, patches, serial="SN-0001")
    return bid, res


# ------------------------------------------------------------------ crop codec
def test_a_crop_survives_the_round_trip():
    """Crops are the training data. A lossy codec here would quietly degrade
    every model trained downstream."""
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, (96, 96, 4), dtype=np.uint8)
    out = store.decode_crop(store.encode_crop(patch))
    assert out.shape == patch.shape
    assert np.array_equal(out, patch), "crop storage is lossy"


def test_crops_are_compressed():
    """Section 7: keep crops, not frames. A board of raw frames is a gigabyte
    and fills any disk in a week."""
    patch = np.zeros((96, 96, 4), np.uint8)
    patch[20:70, 20:70] = 200
    assert len(store.encode_crop(patch)) < 96 * 96 * 4


# ---------------------------------------------------------------- the record
def test_the_board_and_every_verdict_are_recorded(db, inspected, brd):
    bid, res = inspected
    board = db.db.execute("SELECT * FROM board WHERE id = ?", (bid,)).fetchone()
    assert board["serial"] == "SN-0001"
    assert board["n_components"] == len(brd.components)
    assert board["n_flagged"] == len(res.failed)
    n = db.db.execute("SELECT COUNT(*) c FROM verdict WHERE board_id = ?",
                      (bid,)).fetchone()["c"]
    assert n == len(brd.components)


def test_crops_are_kept_for_fails_and_not_for_every_good_part(db, inspected):
    """Storing a patch for all 500 good components is storing the same picture
    of a correct 0402 several hundred times a board."""
    bid, _ = inspected
    withc = db.db.execute(
        "SELECT failed, testable, crop IS NOT NULL AS has FROM verdict"
        " WHERE board_id = ?", (bid,)).fetchall()
    for r in withc:
        should = bool(r["failed"]) or not bool(r["testable"])
        assert bool(r["has"]) == should, \
            f"failed={r['failed']} testable={r['testable']} crop={bool(r['has'])}"
    assert any(r["has"] for r in withc), "no crops stored at all"


def test_the_review_queue_is_only_what_a_person_must_decide(db, inspected):
    bid, res = inspected
    queue = db.review_queue(bid)
    assert {r["refdes"] for r in queue} == {v.refdes for v in res.failed}
    for r in queue:
        assert r["failed"] == 1 and r["testable"] == 1


def test_adjudicating_removes_a_component_from_the_queue(db, inspected):
    bid, _ = inspected
    first = db.review_queue(bid)[0]
    before = len(db.review_queue(bid))
    db.adjudicate(first["id"], "defect", operator="sam", seconds=3.2)
    assert len(db.review_queue(bid)) == before - 1


def test_an_adjudication_records_who_what_and_how_long(db, inspected):
    bid, _ = inspected
    v = db.review_queue(bid)[0]
    db.adjudicate(v["id"], "false_call", operator="sam", seconds=4.5)
    row = db.db.execute("SELECT * FROM adjudication WHERE verdict_id = ?",
                        (v["id"],)).fetchone()
    assert row["call"] == "false_call"
    assert row["operator"] == "sam"
    assert row["seconds"] == pytest.approx(4.5)
    assert row["decided_at"]


def test_an_invalid_call_is_refused(db, inspected):
    bid, _ = inspected
    v = db.review_queue(bid)[0]
    with pytest.raises(ValueError):
        db.adjudicate(v["id"], "probably fine", "sam", 1.0)


def test_re_adjudicating_replaces_rather_than_duplicates(db, inspected):
    """An operator changing their mind must not put two labels on one crop."""
    bid, _ = inspected
    v = db.review_queue(bid)[0]
    db.adjudicate(v["id"], "defect", "sam", 2.0)
    db.adjudicate(v["id"], "false_call", "sam", 3.0)
    rows = db.db.execute("SELECT * FROM adjudication WHERE verdict_id = ?",
                         (v["id"],)).fetchall()
    assert len(rows) == 1 and rows[0]["call"] == "false_call"


# ------------------------------------------------- the point of the whole thing
def test_adjudications_come_back_out_as_training_data(db, inspected):
    """The section 10 exit criterion: the labels land correctly in the dataset.

    Without this the review screen is a nicer way to look at failures and the
    machine never gets better than its rules.
    """
    bid, _ = inspected
    queue = db.review_queue(bid)
    assert len(queue) >= 3, "need a few fails to make this meaningful"
    calls = ["defect", "false_call", "defect"]
    for v, call in zip(queue, calls):
        db.adjudicate(v["id"], call, "sam", 2.0)

    patches, labels, packages = db.export_labels()
    assert len(patches) == len(calls)
    assert patches.shape[1:] == (96, 96, 4), f"got {patches.shape}"
    assert labels.tolist() == [1, 0, 1]
    assert set(packages) <= {v["package"] for v in queue}
    assert patches.dtype == np.uint8


def test_export_is_empty_before_anyone_has_adjudicated(db, inspected):
    patches, labels, packages = db.export_labels()
    assert len(labels) == 0 and len(patches) == 0 and len(packages) == 0


def test_only_adjudicated_crops_are_exported(db, inspected):
    """A crop nobody has judged is not a labelled sample."""
    bid, _ = inspected
    v = db.review_queue(bid)[0]
    db.adjudicate(v["id"], "defect", "sam", 1.0)
    _, labels, _ = db.export_labels()
    assert len(labels) == 1


# ------------------------------------------------------------------- reporting
def test_yield_metrics_are_reported_per_package(db, inspected):
    """Section 8 asks for these per package type, because it is usually two or
    three troublesome packages generating most of the noise."""
    bid, _ = inspected
    for v in db.review_queue(bid)[:2]:
        db.adjudicate(v["id"], "false_call", "sam", 3.0)
    rows = {r["package"]: r for r in db.yield_metrics()}
    assert rows, "no metrics at all"
    for r in rows.values():
        assert r["inspected"] > 0
        assert 0.0 <= r["false_call_rate"] <= 1.0
    assert sum(r["false_calls"] for r in rows.values()) == 2
    assert any(r["avg_review_s"] for r in rows.values())


def test_board_review_time_is_measured(db, inspected):
    """Section 10 sets the exit criterion in seconds, so it has to be a number
    the station actually records rather than something someone times by hand."""
    bid, _ = inspected
    assert db.board_review_seconds(bid) == 0.0
    for v in db.review_queue(bid)[:2]:
        db.adjudicate(v["id"], "defect", "sam", 5.0)
    assert db.board_review_seconds(bid) == pytest.approx(10.0)


def test_untestable_components_are_reported_apart_from_defects(db, brd, golden):
    """A coverage gap is not a defect. It must reach the operator as a separate
    statement about the program, and must never enter the review queue."""
    from aoi import board as B
    partial = B.Board(brd.name, brd.width, brd.height,
                      [c for c in brd.components if c.package != "0603"], brd.fiducials)
    g = rules.fit_golden(
        [synth.render(partial, opts=synth.RenderOpts(seed=470 + i)) for i in range(4)],
        partial)
    res = pipeline.Inspector(g).inspect(render_with(brd, {}), brd)
    patches = np.stack([roi.extract_patch(res.aligned, c, c.pkg) for c in brd.components])
    bid = db.save_result(brd, res, patches)

    gaps = db.untestable(bid)
    assert gaps, "no coverage gap recorded"
    assert {r["package"] for r in gaps} == {"0603"}
    assert not any(r["package"] == "0603" for r in db.review_queue(bid)), \
        "an uninspected component leaked into the review queue"


def test_a_failed_but_untestable_verdict_never_enters_the_queue(db, inspected):
    """Defence in depth on the queue query.

    Today no untestable verdict is also marked failed, so the `testable = 1`
    clause is unreachable through the pipeline and removing it changes nothing.
    That is exactly why it needs a direct test: the day something does mark one
    failed, the clause is what keeps a component nobody can judge out of the
    queue an operator is being asked to clear.
    """
    bid, _ = inspected
    db.db.execute(
        "INSERT INTO verdict (board_id, refdes, package, failed, testable,"
        " reason, score, detail, crop) VALUES (?,?,?,1,0,'untaught',1.0,'',?)",
        (bid, "X99", "MYSTERY", store.encode_crop(np.zeros((96, 96, 4), np.uint8))))
    db.db.commit()
    assert "X99" not in {r["refdes"] for r in db.review_queue(bid)}, \
        "a component with no inspection program was queued for adjudication"
    assert "X99" in {r["refdes"] for r in db.untestable(bid)}


def test_the_latest_board_is_what_review_opens(db, inspected, brd, golden):
    bid, _ = inspected
    res = pipeline.Inspector(golden).inspect(render_with(brd, {}), brd)
    second = db.save_result(brd, res, None)
    assert db.latest_board()["id"] == second != bid
