"""Photo AOI desktop app: the parts that decide things, tested without a display.

The Tk code only wires these up, so a build machine with no screen still checks
what an operator relies on: a box drawn on the canvas lands on the right golden
pixels, nothing unnamed or duplicated is taught, an uncalibrated score never
reads as a probability, N finds the next unanswered call, and the stats say
what the false-call rate is over.

One smoke test opens the real window, and is skipped where there is no display.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from aoi.photo import program, record
from aoi.photo.model import BoardResult, Call, Part, PartResult
from aoi_photo import app


# ------------------------------------------------------------------ the view
def test_canvas_and_image_coordinates_round_trip():
    v = app.View(scale=0.25, ox=100.0, oy=40.0)
    cx, cy = v.to_canvas(500.0, 300.0)
    assert (cx, cy) == (100.0, 65.0)
    assert v.to_image(cx, cy) == pytest.approx((500.0, 300.0))


def test_fit_puts_the_whole_image_in_the_canvas_centred():
    v = app.View()
    v.fit(4032, 3024, 1000, 800)
    assert v.scale == pytest.approx(1000 / 4032)
    x0, y0 = v.to_canvas(0, 0)
    x1, y1 = v.to_canvas(4032, 3024)
    assert x0 == pytest.approx(0) and x1 == pytest.approx(1000)
    assert y0 == pytest.approx(800 - y1)            # equal bands above and below
    assert v.min_scale == pytest.approx(v.scale / 2)


def test_zoom_keeps_the_point_under_the_cursor_still():
    v = app.View()
    v.fit(4000, 3000, 800, 600)
    before = v.to_image(300, 200)
    v.zoom_at(2.0, 300, 200)
    assert v.to_image(300, 200) == pytest.approx(before)
    assert v.scale == pytest.approx(0.4)


def test_zoom_is_clamped():
    v = app.View()
    v.fit(4000, 3000, 800, 600)
    for _ in range(50):
        v.zoom_at(2.0, 10, 10)
    assert v.scale == v.max_scale
    for _ in range(50):
        v.zoom_at(0.5, 10, 10)
    assert v.scale == v.min_scale


def test_pan_moves_the_image_with_the_drag():
    v = app.View(scale=2.0)
    v.pan(20, -10)
    assert (v.ox, v.oy) == (-10.0, 5.0)
    assert v.to_canvas(0, 0) == (20.0, -10.0)


def test_show_box_frames_the_part_with_a_margin():
    v = app.View()
    v.fit(4000, 3000, 800, 600)
    v.show_box(1000, 1200, 100, 50, 800, 600)
    x0, y0 = v.to_canvas(1000, 1200)
    x1, y1 = v.to_canvas(1100, 1250)
    assert 0 < x0 < x1 < 800 and 0 < y0 < y1 < 600
    assert (x0 + x1) / 2 == pytest.approx(400) and (y0 + y1) / 2 == pytest.approx(300)


def test_visible_window_is_clipped_to_the_image():
    v = app.View(scale=1.0, ox=-50.0, oy=10.0)
    assert v.visible(200, 100, 100, 50) == (0, 10, 50, 60)
    v.ox = 500.0
    assert v.visible(200, 100, 100, 50) is None


def test_render_patch_draws_the_visible_window_at_canvas_size():
    full = np.zeros((1000, 2000, 3), np.uint8)
    full[:, :1000] = (255, 0, 0)                     # left half blue in BGR
    small, k = app.downscale(full, side=500)
    assert small.shape[:2] == (250, 500) and k == pytest.approx(0.25)
    v = app.View()
    v.fit(2000, 1000, 400, 400)
    rgb, cx, cy = app.render_patch(full, small, k, v, 400, 400)
    assert rgb.shape == (200, 400, 3)
    assert (cx, cy) == pytest.approx((0, 100))
    assert tuple(rgb[100, 50]) == (0, 0, 255)       # RGB now: blue
    v.show_box(1500, 500, 10, 10, 400, 400)          # zoomed in: cut from the full image
    rgb, _, _ = app.render_patch(full, small, k, v, 400, 400)
    assert rgb.shape[0] >= 399 and rgb.shape[1] >= 399


def test_fit_side_scales_a_crop_to_the_panel():
    assert app.fit_side(np.zeros((10, 40, 3), np.uint8), 200).shape[:2] == (50, 200)
    assert app.fit_side(np.zeros((1000, 500, 3), np.uint8), 200).shape[:2] == (200, 100)


def test_a_drag_becomes_a_clipped_box_and_a_click_does_not():
    assert app.rect_from_points(50, 60, 10, 20, 100, 100) == (10, 20, 40, 40)
    assert app.rect_from_points(-20, -5, 30, 30, 100, 100) == (0, 0, 30, 30)
    assert app.rect_from_points(90, 90, 150, 150, 100, 100) == (90, 90, 10, 10)
    assert app.rect_from_points(10, 10, 11, 12, 100, 100) is None


# ----------------------------------------------------------------- box list
def test_unnamed_boxes_are_numbered_and_dashed_until_named():
    bl = app.BoxList()
    i = bl.add(10, 10, 50, 30)
    j = bl.add(100, 10, 50, 30)
    assert [bl.boxes[k].label for k in (i, j)] == ["P1", "P2"]
    assert bl.boxes[i].candidate and bl.unnamed == 2
    bl.rename(i, "  R1 ")
    assert bl.boxes[i].label == "R1" and not bl.boxes[i].candidate
    assert bl.unnamed == 1


def test_click_selects_the_smallest_box_under_it():
    bl = app.BoxList()
    big = bl.add(0, 0, 200, 200, "U1")
    small = bl.add(50, 50, 20, 20, "R1")
    assert bl.hit(60, 60) == small
    assert bl.hit(150, 150) == big
    assert bl.hit(300, 300) is None


def test_delete_keeps_the_selection_on_the_same_box():
    bl = app.BoxList()
    for r in ("R1", "R2", "R3"):
        bl.add(0, 0, 10, 10, r)
    bl.select(2)
    assert bl.delete(0)
    assert bl.boxes[bl.selected].refdes == "R3"
    assert bl.delete()                                # the selected one
    assert bl.selected is None and [b.refdes for b in bl.boxes] == ["R2"]
    assert not bl.delete()


def test_fields_are_checked_against_the_model_vocabulary():
    bl = app.BoxList()
    i = bl.add(0, 0, 10, 10, "D1")
    bl.set_fields(i, type="diode", package="SOD-123", polarized=True, marked_end="left")
    assert bl.boxes[i].part() == Part("D1", 0, 0, 10, 10, "diode", "SOD-123", True, "left")
    with pytest.raises(ValueError, match="type"):
        bl.set_fields(i, type="transistor")
    with pytest.raises(ValueError, match="marked end"):
        bl.set_fields(i, marked_end="middle")
    with pytest.raises(ValueError):
        bl.add(0, 0, 5, 5, "X1", type="nonsense")
    assert len(bl) == 1                               # the refused box was not kept


def test_nothing_unnamed_or_duplicated_is_taught():
    bl = app.BoxList()
    with pytest.raises(ValueError, match="no parts"):
        bl.to_parts()
    bl.add(0, 0, 10, 10, "R1")
    bl.add(20, 0, 10, 10, "R1")
    bl.add(40, 0, 10, 10)
    with pytest.raises(ValueError) as e:
        bl.to_parts()
    assert "P3" in str(e.value) and "R1 is used 2 times" in str(e.value)
    bl.rename(1, "R2")
    bl.rename(2, "C1")
    assert [p.refdes for p in bl.to_parts()] == ["R1", "R2", "C1"]


def test_proposals_on_top_of_existing_boxes_are_skipped():
    bl = app.BoxList()
    bl.add(100, 100, 40, 20, "R1")
    added, skipped = bl.add_proposals([(101, 100, 40, 20), (300, 300, 30, 30), (500, 50, 10, 60)])
    assert (added, skipped) == (2, 1)
    assert [b.label for b in bl.boxes] == ["R1", "P2", "P3"]
    assert bl.add_proposals([(300, 300, 30, 30)]) == (0, 1)


def test_parts_round_trip_through_the_box_list_and_parts_csv(tmp_path):
    parts = [Part("BD1", 325, 1520, 180, 230, "bridge", "MBS", True, "right"),
             Part("R19", 768, 1500, 108, 60, "resistor", "1206")]
    bl = app.BoxList.from_parts(parts)
    assert bl.to_parts() == parts
    program.write_parts_csv(tmp_path / "parts.csv", bl.to_parts())
    assert program.read_parts_csv(tmp_path / "parts.csv") == parts


# ------------------------------------------------------------ teach helpers
def test_scale_is_either_px_per_mm_or_the_board_length():
    assert app.parse_scale("px_per_mm", "45.2") == (45.2, None)
    assert app.parse_scale("board_mm", " 100 ") == (None, 100.0)
    for bad in ("", "abc", "0", "-3", "nan", "inf"):
        with pytest.raises(ValueError):
            app.parse_scale("px_per_mm", bad)


def test_program_name_is_a_safe_folder_name():
    assert app.check_name("  pt-pcb-004 ") == "pt-pcb-004"
    for bad in ("", "  ", "../x", "a/b", ".hidden", "c:d"):
        with pytest.raises(ValueError):
            app.check_name(bad)


def test_selfcheck_saves_only_when_every_flag_is_ticked():
    flags = [("R3", "+4.1 deg"), ("D2", "turned about 90 deg")]
    assert app.accepted_selfcheck(flags, {"R3", "D2"}) == ["R3", "D2"]
    assert app.accepted_selfcheck(flags, {"R3"}) is None
    assert app.accepted_selfcheck(flags, set()) is None


def _write_program(root, name, version, *, single=False, goldens=1, parts=None):
    folder = root / name / f"v{version}"
    folder.mkdir(parents=True)
    names = [f"golden_{i:02d}.jpg" for i in range(1, goldens + 1)]
    for n in names:
        (folder / n).write_bytes(b"not decoded by the listing")
    parts = parts if parts is not None else [dict(refdes="R1", x=1, y=2, w=30, h=20)]
    (folder / "program.json").write_text(json.dumps(dict(
        name=name, version=version, px_per_mm=45.2, parts=parts, goldens=names,
        single_golden=single)))


def test_program_list_shows_the_latest_version_and_what_is_broken(tmp_path):
    _write_program(tmp_path, "alpha", 1, single=True)
    _write_program(tmp_path, "alpha", 2, goldens=3)
    _write_program(tmp_path, "beta", 1, parts=[dict(refdes="R1", colour="red")])
    (tmp_path / "not-a-program").mkdir()
    infos = app.list_programs(tmp_path)
    assert [i.name for i in infos] == ["alpha", "beta"]
    a, b = infos
    assert (a.version, a.n_goldens, a.single_golden, a.note) == (2, 3, False, "")
    assert "v2" in a.label and "3 golden" in a.label and "45.2 px/mm" in a.label
    assert b.program is None and b.error and b.note == b.error
    assert app.list_programs(tmp_path / "missing") == []


def test_single_golden_programs_say_so(tmp_path):
    _write_program(tmp_path, "gamma", 1, single=True)
    (info,) = app.list_programs(tmp_path)
    assert info.note == "single golden" and info.label.endswith("single golden")


# ---------------------------------------------------------- inspect helpers
def test_run_folders_never_collide(tmp_path):
    now = datetime(2026, 9, 25, 19, 45, 28)
    first = app.new_run_dir(tmp_path, "demo", now)
    assert first == tmp_path / "demo" / "20260925-194528"
    first.mkdir(parents=True)
    second = app.new_run_dir(tmp_path, "demo", now)
    assert second.name == "20260925-194528-2"


def test_photos_that_would_overwrite_each_others_reports_are_caught(tmp_path):
    photos = [tmp_path / "a" / "board.jpg", tmp_path / "b" / "board.png", tmp_path / "c.jpg"]
    assert app.duplicate_stems(photos) == ["board"]
    assert app.duplicate_stems(photos[1:]) == []


def test_verdict_colours():
    assert app.verdict_colour("PASS") == app.FALSE
    assert app.verdict_colour("FAIL") == app.DEFECT
    assert app.verdict_colour("REVIEW") == app.AMBER
    assert app.verdict_colour("SUPERSEDED") == app.DIM


def _board() -> BoardResult:
    parts = [
        PartResult(Part("R12", 100, 150, 40, 20, "resistor", "0603"),
                   [Call("missing", "MISSING", "no body on the footprint", 0.97)]),
        PartResult(Part("C3", 250, 150, 30, 30, "capacitor", "0805"),
                   [Call("tombstone", "TOMBSTONE", "one end bare", 0.81, calibrated=True)]),
        PartResult(Part("D1", 40, 220, 40, 20, "diode", "SOD-123"),
                   [Call("polarity", "POLARITY REVIEW", "band at the other end", 0.6),
                    Call("offset", "UNINSPECTED", "golden too variable")]),
        PartResult(Part("R1", 40, 260, 40, 20, "resistor", "0603"), [Call("offset", "OK")]),
    ]
    return BoardResult(image="shots/board_07.jpg", program="demo", program_version=1,
                       verdict="FAIL", parts=parts,
                       board_calls=[Call("board", "NOT CHECKED", "whole board shifted 300 um")],
                       stamp="single golden: fixed thresholds from the 2026-09-25 run")


def test_flagged_count_is_flagged_parts_plus_board_calls():
    assert app.flagged_count(_board()) == 4


# ----------------------------------------------------------- review helpers
@pytest.fixture
def rec(tmp_path):
    with record.Record(tmp_path / "photo.db") as r:
        yield r


@pytest.fixture
def run(rec):
    pv = rec.program_version("demo", 1, "programs/demo/v1", 1, 45.2, True)
    return rec.save_run(_board(), pv)


def test_recent_runs_are_listed_newest_first(rec, run):
    pv = rec.program_version("demo", 1, "programs/demo/v1", 1, 45.2, True)
    second = rec.save_run(_board(), pv)
    rows = app.recent_runs(rec)
    assert [r["id"] for r in rows] == [second, run]
    r = rows[0]
    assert (r["program"], r["version"], r["machine_verdict"]) == ("demo", 1, "FAIL")
    label = app.run_label(r)
    assert label.startswith(f"#{second}  board_07.jpg  demo v1  FAIL")
    rec.supersede(run, second)
    assert app.run_label(app.recent_runs(rec)[1]).endswith("superseded")


def test_the_review_list_is_what_needs_an_answer(rec, run):
    calls = app.review_calls(rec.calls(run))
    assert [(c["refdes"], c["state"]) for c in calls] == [
        ("R12", "MISSING"), ("C3", "TOMBSTONE"), ("D1", "POLARITY REVIEW"), (None, "NOT CHECKED")]


def test_an_uncalibrated_score_is_never_shown_as_a_probability(rec, run):
    rows = {c["refdes"]: app.call_row(c) for c in app.review_calls(rec.calls(run))}
    assert rows["R12"] == ("", "R12", "MISSING", "score 0.97", "")
    assert rows["C3"] == ("", "C3", "TOMBSTONE", "P=0.81", "(unvalidated)")
    assert rows[None][1] == "board" and rows[None][3] == ""
    summary = app.call_summary(app.review_calls(rec.calls(run))[1])
    assert "P=0.81" in summary and "(unvalidated)" in summary and "tombstone: one end bare" in summary


def test_next_unanswered_skips_answered_calls_and_wraps(rec, run):
    calls = app.review_calls(rec.calls(run))
    assert app.next_unanswered(calls) == 0
    rec.adjudicate(calls[0]["id"], "D", 1.2)
    rec.adjudicate(calls[2]["id"], "S", 0.8)
    calls = app.review_calls(rec.calls(run))
    assert app.call_row(calls[0])[0] == "defect" and app.call_row(calls[2])[0] == "skipped"
    assert app.next_unanswered(calls) == 1
    assert app.next_unanswered(calls, 1) == 3
    assert app.next_unanswered(calls, 3) == 1        # wraps past the answered ones
    for c in calls:
        if c["adjudication"] is None:
            rec.adjudicate(c["id"], "F")
    assert app.next_unanswered(app.review_calls(rec.calls(run))) is None
    assert app.next_unanswered([]) is None


def test_undo_is_traced_to_the_call_it_reopened(rec, run):
    calls = app.review_calls(rec.calls(run))
    rec.adjudicate(calls[0]["id"], "D")
    rec.adjudicate(calls[1]["id"], "F")
    before = app.review_calls(rec.calls(run))
    assert rec.undo_last(run)
    after = app.review_calls(rec.calls(run))
    assert app.changed_index(before, after) == 1
    assert app.changed_index(after, after) is None


def test_reshoot_is_refused_on_a_part_call(rec, run):
    part_call = app.review_calls(rec.calls(run))[0]
    with pytest.raises(ValueError, match="part-level"):
        rec.adjudicate(part_call["id"], "R")


# ------------------------------------------------------------ stats helpers
def test_stats_rows_state_the_rate_as_a_percentage(rec, run):
    calls = app.review_calls(rec.calls(run))
    rec.adjudicate(calls[0]["id"], "F")               # R12 MISSING was a false call
    rows = app.stats_rows(rec)
    assert [r[0] for r in rows] == ["capacitor 0805", "diode SOD-123", "resistor 0603", "all"]
    res = dict(zip([k for k, _, _ in app.STATS_COLUMNS], rows[2]))
    assert res["parts_inspected"] == "2" and res["false_calls"] == "1" and res["fcr"] == "50.00%"
    total = dict(zip([k for k, _, _ in app.STATS_COLUMNS], rows[-1]))
    assert total["fcr"] == "25.00%" and total["false_calls_per_board"] == "1.00"
    assert "under 0.2% per part" in app.FCR_NOTE


def test_stats_row_formats_small_rates_with_enough_digits():
    d = dict(group="resistor 0402", parts_inspected=5000, fail_calls=12, false_calls=9,
             fcr=9 / 5000, review_calls=3, review_resolved_ok=2, boards=100,
             false_calls_per_board=0.09)
    assert app.stats_row(d) == ("resistor 0402", "5000", "12", "9", "0.18%", "3", "2", "100", "0.09")


# ----------------------------------------------------------------- the window
def test_the_window_opens_and_closes(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display available: {e}")
    try:
        a = app.App(root, tmp_path / "programs", tmp_path / "runs", tmp_path / "photo.db")
        root.update()
        assert root.title() == "Photo AOI"
        assert [a.nb.tab(t, "text") for t in a.nb.tabs()] == [
            "1  Program (golden)", "2  Inspect (test)", "3  Review", "4  Stats"]
        # a first-time user must be able to upload straight away
        assert a.editable and str(a.b_add_golden.cget("state")) == "normal"
        a.close()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass                                      # already destroyed by close()
