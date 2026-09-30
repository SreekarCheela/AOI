"""aoi.photo.evaluate: scoring planted defects against inspection results."""
import json

import pytest

from aoi.photo import evaluate
from aoi.photo.model import DIFF_TYPE, BoardResult, Call, Part, PartResult


def _board(image, verdict, calls, extra=(), board_calls=()):
    parts = [PartResult(Part(r, 10 * i, 0, 8, 8), [Call(c, s)]) for i, (r, c, s) in enumerate(calls)]
    parts += [PartResult(p, [Call("difference", "UNEXPECTED DIFFERENCE")]) for p in extra]
    return BoardResult(image, "p", 1, verdict, parts=parts, board_calls=list(board_calls))


TAUGHT = {"R1", "R2", "C1", "D1"}


def test_each_outcome_and_false_calls():
    results = {
        "b1": _board("b1", "FAIL", [("R1", "missing", "MISSING"), ("R2", "offset", "OFFSET"),
                                     ("C1", "offset", "OK"), ("D1", "polarity", "POLARITY REVIEW")]),
        "good": _board("good", "FAIL", [("R1", "offset", "OFFSET"), ("R2", "offset", "OK")]),
    }
    truth = [dict(image="b1", refdes="R1", defect="missing"),       # caught
             dict(image="b1", refdes="R2", defect="skew"),          # other label
             dict(image="b1", refdes="C1", defect="offset"),        # missed
             dict(image="b1", refdes="D1", defect="reversed"),      # caught (REVIEW names it)
             dict(image="b1", refdes="R9", defect="missing")]       # not taught
    s = evaluate.score(results, truth, TAUGHT)
    out = {(r.refdes): r.outcome for r in s.rows}
    assert out == {"R1": "caught", "R2": "other label", "C1": "missed", "D1": "caught", "R9": "not taught"}
    assert s.false_calls == [("good", "R1", "OFFSET", "FAIL")]
    m = s.summary()
    assert (m["planted"], m["caught"], m["missed"], m["false_calls"]) == (5, 2, 1, 1)
    assert m["good_boards"] == 1 and m["good_boards_failed"] == 1 and m["board_escapes"] == 0
    assert m["false_call_rate"] == pytest.approx(1 / 6)


def test_untaught_part_is_caught_by_a_difference_over_its_box_and_that_difference_is_not_false():
    diff = Part("DIFF1", 100, 100, 20, 20, type=DIFF_TYPE)
    far = Part("DIFF2", 500, 500, 20, 20, type=DIFF_TYPE)
    results = {"b": _board("b", "REVIEW", [("R1", "missing", "OK")], extra=[diff, far])}
    truth = [dict(image="b", refdes="R9", defect="missing")]
    boxes = {"R9": Part("R9", 105, 105, 10, 10)}
    s = evaluate.score(results, truth, TAUGHT, boxes)
    assert [r.outcome for r in s.rows] == ["caught"]
    assert s.false_calls == [("b", "DIFF2", "UNEXPECTED DIFFERENCE", "REVIEW")]
    assert s.parts_inspected == 1                     # differences are not part inspections


def test_refused_board_is_not_checked_and_a_passed_defective_board_is_an_escape():
    refused = BoardResult("r", "p", 1, "REVIEW", board_calls=[Call("board", "NOT CHECKED", "could not align")])
    passed = _board("e", "PASS", [("R1", "missing", "OK")])
    truth = [dict(image="r", refdes="R1", defect="missing"), dict(image="e", refdes="R1", defect="missing")]
    s = evaluate.score({"r": refused, "e": passed}, truth, TAUGHT)
    assert [r.outcome for r in s.rows] == ["not checked", "missed"]
    assert s.false_calls == []
    m = s.summary()
    assert m["board_escapes"] == 1 and m["boards_refused"] == 1


def test_truth_file_checks_and_goldens_are_not_boards(tmp_path):
    p = tmp_path / "truth.json"
    p.write_text(json.dumps({"goldens": ["golden1.heic"], "golden": "G.jpg",
                             "truth": [{"image": "t", "refdes": "R1", "defect": "missing"}]}))
    t = evaluate.read_truth(p)
    assert evaluate.non_boards(t) == {"golden1", "G"}
    p.write_text(json.dumps({"truth": [{"image": "t"}]}))
    with pytest.raises(ValueError, match="needs image, refdes and defect"):
        evaluate.read_truth(p)


def test_report_names_escapes_and_false_calls():
    results = {"b": _board("b", "FAIL", [("R1", "missing", "OK"), ("R2", "offset", "OFFSET")])}
    s = evaluate.score(results, [dict(image="b", refdes="R1", defect="missing")], TAUGHT)
    text = evaluate.report(s)
    assert "b R1 missing: missed" in text and "b R2: OFFSET (FAIL)" in text
    assert json.dumps(evaluate.to_json(s))            # serialisable


def test_golden_is_kept_between_runs_per_program_version(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from aoi.photo import inspect as I
    built = []

    class FakeGolden:
        def __init__(self, prog):
            built.append(prog.version)
            self.warmed = False

        def warm(self):
            self.warmed = True
    monkeypatch.setattr(I, "Golden", FakeGolden)
    monkeypatch.setattr(I, "_GOLDENS", type(I._GOLDENS)())
    v1, v2 = (SimpleNamespace(path=tmp_path / "p" / f"v{v}", version=v) for v in (1, 2))
    a = I.golden_for(v1)
    assert I.golden_for(v1) is a and a.warmed
    assert I.golden_for(v2) is not a              # a re-teach is a new version: a new golden
    assert built == [1, 2]


def test_doctor_reports_every_dependency_with_a_status():
    from aoi.photo import doctor
    rows = doctor.checks()
    assert {s for s, _, _ in rows} <= {"OK", "WARN", "FAIL"}
    names = {w for _, w, _ in rows}
    assert {"Python", "numpy", "opencv-python-headless", "tkinter", "part detector", "HEIC"} <= names
    assert "ready" in doctor.text([("OK", "x", "y")])
    assert "1 problem(s)" in doctor.text([("FAIL", "x", "y")])


def test_pi_requirements_pin_what_the_main_set_pins():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    pins = lambda f: {l.split("==")[0]: l.split("==")[1].strip()                     # noqa: E731
                      for l in (root / f).read_text().splitlines() if "==" in l and not l.startswith("#")}
    main, pi = pins("requirements.txt"), pins("requirements-pi.txt")
    shared = set(main) & set(pi)
    assert {"numpy", "opencv-python-headless", "onnxruntime", "torch"} <= shared
    assert {k: main[k] for k in shared} == {k: pi[k] for k in shared}   # one verified version of each
