"""Photo AOI desktop app: the parts that decide things, tested without a display,
then the window itself, driven directly, where a display exists.

The Tk code only wires the plain helpers up, so a build machine with no screen
still checks what an operator relies on: a box drawn on the canvas lands on the
right golden pixels, found parts are taught as P1..Pn without anyone naming
them, the save checklist says every missing thing at once, an uncalibrated score
never reads as a probability, N finds the next unanswered call, and one bad
photo never stops a batch.

The window tests (skipped where Tk cannot open a window) are one regression
test per bug the 2026-09-26 audit confirmed; each names its letter from the
fix plan. They stub the file dialogs and the workers, never the Tk loop.
"""
from __future__ import annotations

import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from aoi import config
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
def test_found_boxes_are_parts_named_p1_p2_until_renamed():
    bl = app.BoxList()
    i = bl.add(10, 10, 50, 30)
    j = bl.add(100, 10, 50, 30)
    assert [bl.boxes[k].label for k in (i, j)] == ["P1", "P2"]
    assert bl.unnamed == 2
    bl.rename(i, "  R1 ")
    assert bl.boxes[i].label == "R1" and bl.unnamed == 1
    # nobody has to name anything: an unnamed box is taught under its P-number
    assert [p.refdes for p in bl.to_parts()] == ["R1", "P2"]
    assert bl.to_parts()[1] == Part("P2", 100, 10, 50, 30, "other")


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


def test_undo_brings_back_a_deleted_box_and_takes_back_an_added_one():
    bl = app.BoxList()
    bl.add(0, 0, 10, 10, "R1")
    bl.add(20, 0, 10, 10, "R2")
    bl.select(0)
    bl.delete()
    assert [b.label for b in bl.boxes] == ["R2"]
    assert "R1" in bl.undo()
    assert [b.label for b in bl.boxes] == ["R1", "R2"] and bl.selected == 0
    bl.select(1)
    bl.add(40, 0, 10, 10)                             # P3
    assert bl.undo() is not None
    assert [b.label for b in bl.boxes] == ["R1", "R2"] and bl.selected == 1
    added, _ = bl.add_proposals([(100, 100, 20, 20), (200, 200, 20, 20)])
    assert added == 2 and len(bl) == 4
    assert "2 found parts" in bl.undo()               # one undo for one Find parts
    assert len(bl) == 2


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


def test_duplicate_names_are_refused_all_at_once():
    bl = app.BoxList()
    assert "no parts" in bl.problems()[0]
    with pytest.raises(ValueError, match="no parts"):
        bl.to_parts()
    bl.add(0, 0, 10, 10, "R1")
    bl.add(20, 0, 10, 10, "R1")
    bl.add(40, 0, 10, 10, "P1")                       # collides with nothing: P1 is taken by no box
    bl.add(60, 0, 10, 10)                             # P4
    probs = bl.problems()
    assert probs == ["two parts are called R1: rename one"]
    bl.rename(0, "P4")                                # a typed name on top of an auto name
    assert bl.problems() == ["two parts are called P4: rename one"]
    bl.rename(0, "R2")
    assert [p.refdes for p in bl.to_parts()] == ["R2", "R1", "P1", "P4"]


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


def test_a_saved_auto_name_stays_an_auto_name_and_new_boxes_do_not_reuse_it():
    bl = app.BoxList.from_parts([Part("P7", 0, 0, 10, 10), Part("R1", 20, 0, 10, 10)])
    assert bl.boxes[0].label == "P7" and not bl.boxes[0].named
    i = bl.add(40, 0, 10, 10)
    assert bl.boxes[i].label == "P8"
    assert bl.undo() is not None and len(bl) == 2    # loading a program is not an undo step
    assert bl.undo() is None


def test_boxes_outside_golden_1_are_named():
    parts = [Part("R1", 10, 10, 20, 20), Part("R2", 190, 10, 20, 20), Part("R3", -1, 0, 5, 5),
             Part("R4", 0, 0, 0, 5)]
    assert app.parts_outside(parts, 200, 100) == ["R2", "R3", "R4"]
    assert app.parts_outside(parts[:1], 200, 100) == []


# ----------------------------------------------------------------- polarity
def test_the_polarized_dropdown_sets_the_type_that_selects_the_check():
    assert app.polarity_fields(app.POL_DIODE, "other") == dict(type="diode", polarized=True)
    assert app.polarity_fields(app.POL_DIODE, "led")["type"] == "led"
    assert app.polarity_fields(app.POL_DIODE, "bridge")["type"] == "bridge"
    assert app.polarity_fields(app.POL_IC, "other") == dict(type="ic", polarized=True)
    assert app.polarity_fields(app.POL_IC, "sot23")["type"] == "sot23"
    assert app.polarity_fields(app.POL_IC, "diode")["type"] == "ic"
    assert app.polarity_fields(app.POL_NO, "diode") == dict(polarized=False)
    with pytest.raises(ValueError):
        app.polarity_fields("maybe", "other")
    b = app.Box(0, 0, 10, 10, type="led", polarized=True)
    assert app.polarity_choice(b) == app.POL_DIODE
    b.type = "sot23"
    assert app.polarity_choice(b) == app.POL_IC
    b.polarized = False
    assert app.polarity_choice(b) == app.POL_NO


# ------------------------------------------------------------ teach helpers
def test_scale_is_optional_and_otherwise_a_positive_number():
    assert app.parse_scale("board_mm", "") == (None, None)
    assert app.parse_scale("board_mm", "   ") == (None, None)
    assert app.parse_scale("px_per_mm", "45.2") == (45.2, None)
    assert app.parse_scale("board_mm", " 100 ") == (None, 100.0)
    for bad in ("abc", "0", "-3", "nan", "inf"):
        with pytest.raises(ValueError):
            app.parse_scale("board_mm", bad)


def test_program_name_is_a_safe_folder_name():
    assert app.check_name("  pt-pcb-004 ") == "pt-pcb-004"
    for bad in ("", "  ", "../x", "a/b", ".hidden", "c:d"):
        with pytest.raises(ValueError):
            app.check_name(bad)


def test_the_default_name_is_the_golden_folder_and_never_an_existing_program():
    g = [Path("/shoots/pt pcb 004/IMG_1.jpg")]
    assert app.default_program_name(g, []) == "pt-pcb-004"
    assert app.default_program_name(g, ["pt-pcb-004"]) == "pt-pcb-004-2"
    assert app.default_program_name(g, ["pt-pcb-004", "pt-pcb-004-2"]) == "pt-pcb-004-3"
    assert app.default_program_name([], [], datetime(2026, 9, 26)) == "board-20260926"
    app.check_name(app.default_program_name([Path("/.x/a.jpg")], []))


def test_selfcheck_saves_only_when_every_flag_is_ticked():
    flags = [("R3", "+4.1 deg"), ("D2", "turned about 90 deg")]
    assert app.accepted_selfcheck(flags, {"R3", "D2"}) == ["R3", "D2"]
    assert app.accepted_selfcheck(flags, {"R3"}) is None
    assert app.accepted_selfcheck(flags, set()) is None


def test_one_golden_is_allowed_with_a_banner():
    assert "1 golden: results are stamped single-golden" in app.single_golden_banner(1)
    assert f"add {config.PHOTO_MIN_GOLDENS} goldens from {config.PHOTO_MIN_GOLDENS} different" \
        in app.single_golden_banner(1)
    assert app.single_golden_banner(0) == ""
    assert app.single_golden_banner(config.PHOTO_MIN_GOLDENS) == ""


def _checklist(**kw):
    args = dict(name="demo", golden_states=[[]], box_problems=[], n_parts=37, n_polarized=0,
                scale_text="", scale_mode="board_mm", finding=False)
    args.update(kw)
    return app.save_checklist(**args)


def test_bug_n_the_checklist_says_what_save_needs_all_at_once():
    rows, reason = _checklist()
    assert reason is None                             # one golden, no names, no scale: savable
    states = [s for s, _ in rows]
    assert states == [app.OK, app.WARN, app.OK, app.OPTIONAL, app.OPTIONAL]
    assert "Parts found: 37" in rows[2][1] and "3 recommended" in rows[1][1]
    rows, reason = _checklist(name="", golden_states=[], box_problems=["no parts yet"],
                              scale_text="abc")
    todo = [t for s, t in rows if s == app.TODO]
    assert len(todo) == 4                              # name, goldens, parts, scale: every one shown
    assert reason.startswith("Name")
    _, reason = _checklist(golden_states=[[], None])
    assert "checking" in reason
    _, reason = _checklist(golden_states=[[], ["out of focus"]])
    assert "Golden 2 rejected: out of focus" in reason
    _, reason = _checklist(finding=True)
    assert "finding parts" in reason
    rows, reason = _checklist(scale_text="100", n_polarized=2,
                              golden_states=[[]] * config.PHOTO_MIN_GOLDENS)
    assert reason is None and all(s == app.OK for s, _ in rows)
    assert "100 mm" in rows[4][1]


def _write_program(root, name, version, *, single=False, goldens=1, parts=None, image=None):
    folder = root / name / f"v{version}"
    folder.mkdir(parents=True)
    names = [f"golden_{i:02d}.png" for i in range(1, goldens + 1)]
    for n in names:
        if image is not None:
            cv2.imwrite(str(folder / n), image)
        else:
            (folder / n).write_bytes(b"not decoded by the listing")
    parts = parts if parts is not None else [dict(refdes="R1", x=1, y=2, w=30, h=20)]
    (folder / "program.json").write_text(json.dumps(dict(
        name=name, version=version, px_per_mm=45.2, parts=parts, goldens=names,
        single_golden=single)))
    return folder


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
    info.px_per_mm = 0.0
    assert "no scale" in info.label


def test_bug_e_the_inspect_choice_is_kept_by_name_not_label():
    assert app.pick_program(["alpha", "zeta"], "zeta") == "zeta"
    assert app.pick_program(["alpha", "zeta"], "gone") == "alpha"
    assert app.pick_program(["alpha", "zeta"], "alpha", prefer="zeta") == "zeta"
    assert app.pick_program([], "zeta") is None


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


def test_a_board_says_why_in_one_line_and_lists_what_is_flagged():
    res = _board()
    assert app.board_headline(res) == "whole board shifted 300 um"
    assert app.flagged_rows(res)[0] == ("board", "NOT CHECKED", "whole board shifted 300 um")
    assert [r[0] for r in app.flagged_rows(res)[1:]] == ["R12", "C3", "D1"]
    res.board_calls = []
    assert app.board_headline(res).startswith("3 part(s) flagged: R12, C3, D1")
    unaligned = BoardResult("t.jpg", "demo", 1, "REVIEW",
                            board_calls=[Call("board", "NOT CHECKED", "could not align: 3 inliers")])
    assert app.board_headline(unaligned) == "could not align: 3 inliers"


# ----------------------------------------------------------------- workers
def test_bug_m_the_image_and_golden_check_workers_always_answer(monkeypatch, tmp_path):
    def boom(_p):
        raise RuntimeError("decoder crashed")
    monkeypatch.setattr(app.io, "load_bgr", boom)
    q = queue.Queue()
    app.load_view_worker(q, "editor_image", (1, "g.jpg"), tmp_path / "g.jpg")
    kind, key, bgr, _, _, err = q.get_nowait()
    assert (kind, key, bgr) == ("editor_image", (1, "g.jpg"), None)
    assert "RuntimeError: decoder crashed" in err
    app.check_goldens_worker(q, 3, [tmp_path / "a.jpg", tmp_path / "b.jpg"])
    got = [q.get_nowait() for _ in range(2)]
    assert [m[2] for m in got] == [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
    assert all("decoder crashed" in m[3][0] for m in got)


def test_propose_worker_reports_a_failure_instead_of_dying(monkeypatch):
    monkeypatch.setattr(app.teach, "propose_boxes", lambda bgr: 1 / 0)
    q = queue.Queue()
    app.propose_worker(q, 5, np.zeros((10, 10, 3), np.uint8))
    kind, gen, rects, _, err = q.get_nowait()
    assert (kind, gen, rects) == ("proposals", 5, []) and "ZeroDivisionError" in err


def test_teach_worker_saves_one_golden_without_a_scale(monkeypatch):
    seen = {}
    monkeypatch.setattr(app.teach, "build_program", lambda *a, **k: seen.update(k) or "report")
    q = queue.Queue()
    app.teach_worker(q, 2, "demo", [Path("g.jpg")], [Part("P1", 0, 0, 50, 50)], None, None, True,
                     Path("programs"), [])
    assert q.get_nowait() == ("teach_ok", 2, "report")
    assert seen["single_golden"] is True and seen["px_per_mm"] is None and seen["board_width_mm"] is None
    monkeypatch.setattr(app.teach, "build_program", lambda *a, **k: [][1])
    app.teach_worker(q, 2, "demo", [], [], None, None, False, Path("programs"), [])
    kind, gen, msg = q.get_nowait()
    assert kind == "teach_error" and "IndexError" in msg


def _fake_inspect_setup(monkeypatch, tmp_path, inspect_fn):
    golden = np.zeros((60, 80, 3), np.uint8)
    monkeypatch.setattr(app.photo_inspect, "Golden", lambda prog: SimpleNamespace(bgr=golden))
    monkeypatch.setattr(app.photo_inspect, "inspect", inspect_fn)
    monkeypatch.setattr(app.io, "load_bgr", lambda p: np.full((30, 40, 3), 7, np.uint8))
    monkeypatch.setattr(app.photo_inspect, "load_bgr", lambda p: np.full((30, 40, 3), 7, np.uint8))
    written = []

    def write_all(res, g, shown, out_dir):
        written.append((Path(res.image).name, shown))
        stem = Path(res.image).stem
        return {"annotated": tmp_path / f"{stem}.png", "csv": tmp_path / f"{stem}.csv",
                "html": tmp_path / f"{stem}.html"}
    monkeypatch.setattr(app.report, "write_all", write_all)
    prog = SimpleNamespace(name="demo", version=1, path=tmp_path, goldens=["g"], px_per_mm=0.0,
                           single_golden=True)
    return prog, written


def test_bug_d_k_one_bad_photo_never_stops_the_batch_and_unaligned_shows_the_photo(monkeypatch, tmp_path):
    def inspect(prog, f, golden, calib, bgr=None):
        name = Path(f).name
        if name == "b.jpg":
            raise cv2.error("simulated OpenCV assertion")
        if name == "c.jpg":                            # could not align
            return BoardResult(str(f), "demo", 1, "REVIEW",
                               board_calls=[Call("board", "NOT CHECKED", "could not align")]), None
        res = BoardResult(str(f), "demo", 1, "PASS",
                          parts=[PartResult(Part("R1", 1, 1, 10, 10), [Call("offset", "OK")])])
        return res, np.full((60, 80, 3), 200, np.uint8)
    prog, written = _fake_inspect_setup(monkeypatch, tmp_path, inspect)
    q = queue.Queue()
    photos = [tmp_path / n for n in ("a.jpg", "b.jpg", "c.jpg")]
    app.inspect_worker(q, prog, photos, tmp_path / "photo.db", tmp_path / "out")
    msgs = [q.get_nowait() for _ in range(q.qsize())]
    kinds = [(m[0], m[1]) for m in msgs if m[0] != "inspect_golden"]
    assert kinds == [("inspect_row", 0), ("inspect_error", 1), ("inspect_row", 2)]
    err = next(m for m in msgs if m[0] == "inspect_error")[2]
    assert "error" in err and "simulated OpenCV assertion" in err     # the reason, not "see terminal"
    shown = dict(written)
    assert shown["a.jpg"].max() == 200                               # aligned photo
    assert shown["c.jpg"].shape == (30, 40, 3) and shown["c.jpg"].max() == 7   # the photo, not black
    row_c = next(m for m in msgs if m[0] == "inspect_row" and m[1] == 2)
    assert row_c[5] is None                                          # nothing aligned to crop from
    with record.Record(tmp_path / "photo.db") as rec:
        assert len(rec.runs()) == 2


def test_inspect_worker_reports_a_golden_that_will_not_load(monkeypatch, tmp_path):
    def bad(prog):
        raise MemoryError("golden too big")
    monkeypatch.setattr(app.photo_inspect, "Golden", bad)
    q = queue.Queue()
    prog = SimpleNamespace(name="demo", version=1, path=tmp_path)
    app.inspect_worker(q, prog, [tmp_path / "a.jpg"], tmp_path / "photo.db", tmp_path)
    kind, msg = q.get_nowait()
    assert kind == "inspect_fatal" and "MemoryError: golden too big" in msg


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
    assert rows["R12"] == ("", "R12", "MISSING", "confidence 0.97 (uncalibrated)")
    assert rows["C3"] == ("", "C3", "TOMBSTONE", "P=0.81 (calibrated)  unvalidated")
    assert rows[None][1] == "board" and rows[None][3] == ""
    assert "P=" not in rows["R12"][3]
    summary = app.call_summary(app.review_calls(rec.calls(run))[1])
    assert "P=0.81" in summary and "unvalidated" in summary and "tombstone: one end bare" in summary


def test_the_review_state_reads_as_a_sentence(rec, run):
    calls = app.review_calls(rec.calls(run))
    assert app.review_progress("FAIL", calls) == "machine says FAIL · you have answered 0 of 4"
    rec.adjudicate(calls[0]["id"], "D")
    assert app.review_progress("FAIL", app.review_calls(rec.calls(run))).endswith("answered 1 of 4")
    assert "nothing on this board" in app.review_progress("PASS", [])


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


def test_bug_h_a_run_from_an_earlier_session_gets_its_boxes_back(tmp_path):
    img = np.zeros((300, 400, 3), np.uint8)
    _write_program(tmp_path / "programs", "demo", 2, image=img,
                   parts=[dict(refdes="R12", x=100, y=150, w=40, h=20),
                          dict(refdes="C3", x=250, y=150, w=30, h=30)])
    out = tmp_path / "runs"
    out.mkdir()
    with (out / "board_07.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["refdes", "dx_px", "dy_px"])
        w.writeheader()
        w.writerow(dict(refdes="R12", dx_px="3.50", dy_px="-2.00"))
        w.writerow(dict(refdes="", dx_px="", dy_px=""))
    row = dict(program="demo", version=2, image="shots/board_07.jpg",
               annotated=str(out / "board_07_annotated.png"))
    prog, geo, why = app.run_geometry(row, tmp_path / "programs")
    assert why == "" and prog.version == 2
    assert geo["R12"] == (Part("R12", 100, 150, 40, 20), 3.5, -2.0)
    assert geo["C3"][1:] == (0.0, 0.0)               # no CSV row: the taught position
    prog, geo, why = app.run_geometry(dict(row, version=9), tmp_path / "programs")
    assert prog is None and geo == {} and "demo v9" in why


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
# Driven directly: no screen capture, no synthetic mouse unless a test is about
# the mouse. Dialogs are stubbed so nothing waits for a person.
class Dialogs:
    def __init__(self, monkeypatch):
        self.log: list[tuple[str, str]] = []
        self.files: list[str] = []
        self.file = ""
        self.yes = True
        mb, fd = app.messagebox, app.filedialog
        monkeypatch.setattr(mb, "showerror", lambda t, m, **k: self.log.append(("error", m)))
        monkeypatch.setattr(mb, "showinfo", lambda t, m, **k: self.log.append(("info", m)))
        monkeypatch.setattr(mb, "askyesno", lambda t, m, **k: self.log.append(("ask", m)) or self.yes)
        monkeypatch.setattr(fd, "askopenfilenames", lambda **k: list(self.files))
        monkeypatch.setattr(fd, "askopenfilename", lambda **k: self.file)
        monkeypatch.setattr(fd, "asksaveasfilename", lambda **k: "")


@pytest.fixture
def gui(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display available: {e}")
    dialogs = Dialogs(monkeypatch)
    a = app.App(root, tmp_path / "programs", tmp_path / "runs", tmp_path / "photo.db")
    root.update()
    a.dialogs = dialogs
    yield a
    try:
        a.close()
    except tk.TclError:
        pass                                          # already closed by the test


def pump(a, until=lambda: False, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        a.root.update()
        if until():
            return True
        time.sleep(0.01)
    return False


def idle(a):
    return not a.busy and a.q.empty()


def _golden_png(tmp_path, name="golden.png") -> Path:
    img = np.full((300, 400, 3), 40, np.uint8)
    cv2.rectangle(img, (100, 100), (160, 130), (200, 200, 200), -1)
    p = tmp_path / "shoot-a" / name
    p.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(p), img)
    return p


def _ready(a, tmp_path, monkeypatch, rects=((100, 100, 60, 30), (200, 50, 40, 40))):
    """One golden uploaded, checked clean, and its parts found."""
    monkeypatch.setattr(app.gate, "golden_problems", lambda bgr: [])
    monkeypatch.setattr(app.teach, "propose_boxes", lambda bgr: list(rects))
    # no OCR on a synthetic board with no text: loading the ~100 MB models on
    # the suite's first read outlasted pump's timeout on a busy machine
    monkeypatch.setattr(app.names, "available", lambda: False)
    a.dialogs.files = [str(_golden_png(tmp_path))]
    a.add_goldens()
    assert pump(a, lambda: idle(a) and len(a.boxes) == len(rects))


def test_the_window_opens_on_three_plain_tabs(gui):
    a = gui
    assert a.root.title() == "Photo AOI"
    assert [a.nb.tab(t, "text") for t in a.nb.tabs()] == [
        "1  Set up board", "2  Inspect boards", "3  Results"]
    assert [a.results_nb.tab(t, "text") for t in a.results_nb.tabs()] == [
        "Review calls", "False-call stats"]
    # a first-time user must be able to upload straight away, and is told to
    assert a.editable and str(a.b_add_golden.cget("state")) == "normal"
    assert "Upload golden photos" in a.l_sinstr.cget("text")
    # the 45-minute retrain is not a button on any tab
    labels = [a.menu_advanced.entrycget(i, "label") for i in range(a.menu_advanced.index("end") + 1)
              if a.menu_advanced.type(i) != "separator"]
    assert any("Retrain" in s and "45 minutes" in s for s in labels)


def test_zero_marking_setup_finds_parts_and_saves_one_golden_unnamed(gui, tmp_path, monkeypatch):
    a = gui
    _ready(a, tmp_path, monkeypatch)
    assert [b.label for b in a.boxes.boxes] == ["P1", "P2"]
    assert a.v_name.get() == "shoot-a"                # defaulted from the golden folder
    assert "1 golden: results are stamped single-golden" in a.l_banner.cget("text")
    assert str(a.b_save.cget("state")) == "normal"
    assert len(a.t_parts.get_children()) == 2
    # boxes are solid: a found part is a part, not a dashed guess
    a.ed.redraw()
    a.root.update()
    rects = [i for i in a.ed.canvas.find_all() if a.ed.canvas.type(i) == "rectangle"]
    assert len(rects) == 2 and all(a.ed.canvas.itemcget(i, "dash") == "" for i in rects)
    seen = {}

    def fake_teach(q, gen, name, goldens, parts, ppm, bw, single, root, accept):
        seen.update(name=name, parts=[p.refdes for p in parts], ppm=ppm, bw=bw, single=single)
        q.put(("teach_error", gen, "stopped here"))
    monkeypatch.setattr(app, "teach_worker", fake_teach)
    a.save_program()
    assert pump(a, lambda: idle(a) and seen)
    assert seen == dict(name="shoot-a", parts=["P1", "P2"], ppm=None, bw=None, single=True)


def test_bug_n_save_is_off_with_the_reason_until_the_checklist_is_met(gui, tmp_path, monkeypatch):
    a = gui
    assert str(a.b_save.cget("state")) == "disabled"
    assert "Save needs" in a.l_save_reason.cget("text")
    _ready(a, tmp_path, monkeypatch)
    a.v_name.set("a/b")
    assert str(a.b_save.cget("state")) == "disabled" and "folder name" in a.l_save_reason.cget("text")
    a.v_name.set("demo")
    a.v_scale.set("abc")
    assert str(a.b_save.cget("state")) == "disabled" and "not a number" in a.l_save_reason.cget("text")
    a.v_scale.set("")
    assert str(a.b_save.cget("state")) == "normal"
    texts = [r.cget("text") for r in a.check_rows]
    assert any("Parts found: 2" in t for t in texts)
    assert not any("self-check" in t for t in texts)


def test_bug_a_busy_stays_set_through_the_second_save_after_the_self_check(gui, tmp_path, monkeypatch):
    a = gui
    _ready(a, tmp_path, monkeypatch)
    release = threading.Event()
    calls = []

    def fake_teach(q, gen, *rest):
        calls.append(rest[-1])
        if len(calls) == 1:
            q.put(("teach_selfcheck", gen, [("P1", "+4 deg")]))
            return
        release.wait(10)
        q.put(("teach_error", gen, "fake end"))
    monkeypatch.setattr(app, "teach_worker", fake_teach)
    monkeypatch.setattr(a, "_open_selfcheck", lambda flags, done: done([r for r, _ in flags]))
    a.save_program()
    assert pump(a, lambda: len(calls) == 2)
    pump(a, lambda: a.q.empty() and a.busy["teach"] == 1, timeout=2)
    assert calls[1] == ["P1"]                          # accepted flag passed on
    assert "teach" in a.busy and str(a.b_save.cget("state")) == "disabled"
    assert set(a.busy) & set(app.LONG_WORK)            # closing would warn
    a.save_program()
    pump(a, timeout=0.2)
    assert len(calls) == 2                             # no double save
    release.set()
    assert pump(a, lambda: idle(a))
    assert "teach" not in a.busy and str(a.b_save.cget("state")) == "normal"


def test_bug_b_navigation_is_off_during_a_save_and_a_stale_save_changes_nothing(gui, tmp_path, monkeypatch):
    a = gui
    _ready(a, tmp_path, monkeypatch)
    release = threading.Event()

    def fake_teach(q, gen, *rest):
        release.wait(10)
        q.put(("teach_ok", gen, SimpleNamespace(
            program=SimpleNamespace(name="elsewhere", version=1, parts=[], goldens=[]),
            selfcheck_flags=[])))
    monkeypatch.setattr(app, "teach_worker", fake_teach)
    a.save_program()
    pump(a, timeout=0.1)
    for b in (a.b_new, a.b_newver, a.b_add_golden, a.b_rm_golden, a.b_save):
        assert str(b.cget("state")) == "disabled"
    assert str(a.t_programs.cget("selectmode")) == "none"
    gen = a.teach_gen
    a.new_program()                                    # a keyboard route still cannot reset
    assert a.teach_gen == gen and len(a.boxes) == 2
    a.teach_gen += 1                                   # as if the work had moved on anyway
    boxes, name = list(a.boxes.boxes), a.v_name.get()
    release.set()
    assert pump(a, lambda: idle(a))
    assert a.boxes.boxes == boxes and a.v_name.get() == name and a.editable
    assert not [m for m in a.dialogs.log if m[0] == "error"]


def test_bug_c_clicking_a_part_puts_the_keyboard_in_its_name_and_never_deletes_it(gui, tmp_path, monkeypatch):
    a = gui
    _ready(a, tmp_path, monkeypatch)
    cx, cy = a.ed.view.to_canvas(130, 115)             # inside P1
    a._ed_press(SimpleNamespace(x=cx, y=cy))
    a._ed_release(SimpleNamespace(x=cx, y=cy))
    a.root.update()
    assert a.boxes.selected == 0 and a._focused() is a.e_refdes
    a.e_refdes.insert("end", "R1")                     # typing lands in the entry
    assert a.boxes.boxes[0].refdes == "R1"
    assert a._ed_delete() is None and len(a.boxes) == 2   # BackSpace in the entry: text, not box
    a.e_refdes.focus_force()                           # key events go to the focused window
    a.root.update()
    a.e_refdes.event_generate("<Return>")              # Return names the next part
    a.root.update()
    assert a.boxes.selected == 1 and a._focused() is a.e_refdes
    assert a._ed_delete() is None and len(a.boxes) == 2
    a.ed.canvas.focus_set()                            # only with the photo focused
    a.root.update()
    assert a._ed_delete() == "break" and len(a.boxes) == 1
    assert a._on_undo_key() == "break" and len(a.boxes) == 2   # Cmd/Ctrl+Z brings it back
    a.e_refdes.focus_set()
    a.root.update()
    assert a._on_undo_key() is None                    # not while typing


def test_polarized_parts_are_marked_with_one_dropdown(gui, tmp_path, monkeypatch):
    a = gui
    _ready(a, tmp_path, monkeypatch)
    a._select_box(0)
    a.v_pol.set(app.POL_DIODE)
    a._on_polarity_change()
    b = a.boxes.boxes[0]
    assert (b.type, b.polarized) == ("diode", True)
    assert a.l_polcount.cget("text") == "marked: 1"
    assert a.t_parts.item("0", "values")[2] == app.POL_DIODE
    a._select_box(1)
    a.v_pol.set(app.POL_IC)
    a._on_polarity_change()
    assert a.boxes.to_parts()[1].type == "ic" and a.boxes.to_parts()[1].polarized
    assert any("Polarized parts marked: 2" in r.cget("text") for r in a.check_rows)


def test_a_repeated_golden_is_pointed_at_and_holds_save(gui, tmp_path, monkeypatch):
    assert app.same_shot("golden 2 is the same shot as golden 1; shoot a different good board") == (2, 1)
    assert app.same_shot("golden 2 does not align") is None
    a = gui
    _ready(a, tmp_path, monkeypatch)
    a._add_golden_paths([_golden_png(tmp_path, "again.png")], check=True)
    assert pump(a, lambda: idle(a))
    assert str(a.b_save.cget("state")) == "normal"
    a._on_teach_error(a.teach_gen, "golden 2 is the same shot as golden 1; shoot a different good board")
    assert "Select it and click Remove".lower() in a.dialogs.log[-1][1].lower()
    assert "same shot as golden 1" in a.t_goldens.item("1", "values")[2]
    assert str(a.b_save.cget("state")) == "disabled"
    assert "Golden 2 rejected" in a.l_save_reason.cget("text")


def test_bug_f_removing_golden_1_drops_everything_in_flight_for_it(gui, tmp_path, monkeypatch):
    a = gui
    _ready(a, tmp_path, monkeypatch)
    second = _golden_png(tmp_path, "golden2.png")
    a._add_golden_paths([second], check=True)
    assert pump(a, lambda: idle(a))
    old = a.teach_gen
    a.t_goldens.selection_set("0")
    a.dialogs.yes = True
    a.remove_golden()
    assert a.teach_gen > old and len(a.boxes) == 0
    a._on_proposals(old, [(1, 1, 50, 50)])               # a stale Find parts result
    a._on_editor_image((old, str(second)), np.zeros((5, 5, 3), np.uint8), None, 1.0, "")
    assert len(a.boxes) == 0 and (a.ed.full is None or a.ed.full.shape[0] != 5)
    assert pump(a, lambda: idle(a) and len(a.boxes) == 2)   # the new golden 1 gets its own parts


def test_bug_l_an_import_outside_golden_1_is_refused_with_the_reason(gui, tmp_path, monkeypatch):
    a = gui
    _ready(a, tmp_path, monkeypatch)
    csv_path = tmp_path / "far.csv"
    csv_path.write_text("refdes,x,y,w,h,type,package,polarized,marked_end\n"
                        "R1,9000,9000,80,40,resistor,0603,0,none\nR2,10,10,80,40,resistor,0603,0,none\n")
    a.dialogs.file = str(csv_path)
    a.import_parts()
    errors = [m for k, m in a.dialogs.log if k == "error"]
    assert errors and "outside golden 1" in errors[0] and "R1" in errors[0]
    assert [b.label for b in a.boxes.boxes] == ["P1", "P2"]


def test_bug_i_the_self_check_dialog_does_not_stop_the_queue(gui, tmp_path, monkeypatch):
    a = gui
    _ready(a, tmp_path, monkeypatch)
    a._on_teach_selfcheck(a.teach_gen, [("P1", "+4 deg")])
    assert a._selfcheck_top is not None
    handled = []
    a._handlers["probe"] = lambda: handled.append(1)
    a.q.put(("probe",))
    assert pump(a, lambda: handled, timeout=2)         # rows keep arriving with it open
    assert a._selfcheck_top is not None and "teach" in a.busy
    cancel = [w for w in a._selfcheck_top.winfo_children() for w in [w, *w.winfo_children()]
              if isinstance(w, app.ttk.Button) and w.cget("text") == "Cancel"]
    cancel[0].invoke()
    assert a._selfcheck_top is None and "teach" not in a.busy


def test_bug_e_saving_a_new_version_keeps_the_inspect_choice(gui, tmp_path):
    a = gui
    root = tmp_path / "programs"
    for n in ("alpha", "zeta"):
        _write_program(root, n, 1)
    a.refresh_programs()
    a.nb.select(a.tab_inspect)
    a.c_iprog.current(1)
    a._on_iprog()
    assert a.inspect_name == "zeta"
    _write_program(root, "zeta", 2)
    a.refresh_programs()
    assert a.inspect_name == "zeta" and "zeta  v2" in a.v_iprog.get()
    assert a._inspect_program().version == 2


def _fake_run(monkeypatch, n_photos, tmp_path):
    release = threading.Event()

    def fake_inspect(q, prog, photos, db, out):
        release.wait(10)
        for i in range(len(photos)):
            q.put(("inspect_error", i, f"ValueError: photo {i} is unreadable"))
    monkeypatch.setattr(app, "inspect_worker", fake_inspect)
    return release


def test_bug_g_j_progress_ends_full_and_no_photo_joins_a_running_batch(gui, tmp_path, monkeypatch):
    a = gui
    _write_program(tmp_path / "programs", "demo", 1)
    a.refresh_programs()
    a.nb.select(a.tab_inspect)
    a.dialogs.files = [str(tmp_path / f"t{i}.jpg") for i in range(3)]
    a.add_tests()
    release = _fake_run(monkeypatch, 3, tmp_path)
    a.run_inspect()
    assert str(a.b_add_tests.cget("state")) == "disabled"
    a.dialogs.files = [str(tmp_path / "late.jpg")]
    a.add_tests()
    assert len(a.inspect_photos) == 3                  # refused, not a blank row
    release.set()
    assert pump(a, lambda: idle(a))
    assert a.pb["value"] == 3 and a.pb["maximum"] == 3 and a.l_iprogress.cget("text") == "3 / 3"
    rows = [a.t_inspect.item(i, "values") for i in a.t_inspect.get_children()]
    assert [r[1] for r in rows] == ["ERROR"] * 3 and "unreadable" in rows[0][4]
    assert "0 board(s) inspected" in a.status_bar.cget("text")
    assert str(a.b_add_tests.cget("state")) == "normal"


def _recorded_run(tmp_path):
    """A program on disk and one run of it in the record, as an earlier session left them."""
    img = np.full((300, 400, 3), 90, np.uint8)
    _write_program(tmp_path / "programs", "demo", 1, image=img,
                   parts=[dict(refdes="R12", x=100, y=150, w=40, h=20)])
    out = tmp_path / "runs" / "demo" / "old"
    out.mkdir(parents=True)
    cv2.imwrite(str(out / "board_07_annotated.png"), img)
    (out / "board_07.html").write_text("<html></html>")
    with (out / "board_07.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["refdes", "dx_px", "dy_px"])
        w.writeheader()
        w.writerow(dict(refdes="R12", dx_px="2", dy_px="1"))
    res = BoardResult("shots/board_07.jpg", "demo", 1, "FAIL", parts=[PartResult(
        Part("R12", 100, 150, 40, 20), [Call("missing", "MISSING", "no body", 0.97)])],
        board_calls=[Call("board", "NOT CHECKED", "glare")])
    with record.Record(tmp_path / "photo.db") as rec:
        pv = rec.program_version("demo", 1, tmp_path / "programs/demo/v1", 1, 45.2, False)
        run_id = rec.save_run(res, pv)
        rec.set_report(run_id, out / "board_07_annotated.png", out / "board_07.html")
    return run_id


def test_bug_h_results_for_an_earlier_run_show_crops_and_the_highlight(tmp_path, monkeypatch):
    run_id = _recorded_run(tmp_path)
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display available: {e}")
    Dialogs(monkeypatch)
    a = app.App(root, tmp_path / "programs", tmp_path / "runs", tmp_path / "photo.db")
    try:
        a.nb.select(a.tab_results)
        a.open_run(run_id)
        assert pump(a, lambda: idle(a) and a.annotated is not None and a.goldens)
        a._present_call(0)
        assert a._geo_for(a.calls[0]) == (Part("R12", 100, 150, 40, 20), 2.0, 1.0)
        caps = [c.cget("text") for c in a.crop_caps]
        assert caps == ["GOLDEN", "TEST (report image)"]
        assert all(str(i.cget("image")) for i in a.crop_imgs)
        assert "answered 0 of 2" in a.l_rprogress.cget("text")
        # R is offered on the board call only
        assert str(a.b_answer["R"].cget("state")) == "disabled"
        a.answer("R")
        assert a.calls[0]["adjudication"] is None
        a._select_call(1)
        assert str(a.b_answer["R"].cget("state")) == "normal"
        assert str(a.b_answer["S"].cget("state")) == "disabled"
        a.b_answer["D"].invoke()
        assert "answered 1 of 2" in a.l_rprogress.cget("text")
    finally:
        a.close()
