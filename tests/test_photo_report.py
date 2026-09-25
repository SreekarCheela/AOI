"""Photo-mode exports: the annotated image, the CSV, the one-file HTML.

Synthetic images only. The checks are the ones a reader of the export relies
on: the flagged part is marked where it was found, every call is a CSV row, the
HTML opens with no network, and an unvalidated check says so.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from aoi.photo import report
from aoi.photo.model import BoardResult, Call, Part, PartResult


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    golden = rng.integers(90, 110, (300, 400, 3), dtype=np.uint8)
    test = golden.copy()
    return golden, test


@pytest.fixture
def result():
    parts = [
        PartResult(Part("R12", 100, 150, 40, 20, "resistor", "0603"),
                   [Call("missing", "MISSING", "no body in window", 0.97)]),
        PartResult(Part("C3", 250, 150, 30, 30, "capacitor", "0805"),
                   [Call("tombstone", "TOMBSTONE", "length 0.52 of golden", 0.81),
                    Call("offset", "OFFSET", "0.21 mm", 0.66)],
                   dx_px=12.0, dy_px=-5.0, dtheta_deg=1.5),
        PartResult(Part("R1", 40, 220, 40, 20, "resistor", "0603"),
                   [Call("offset", "OK")]),
    ]
    return BoardResult(image="shots/board_07.jpg", program="demo", program_version=2,
                       verdict="FAIL", reasons=["R12 MISSING", "C3 TOMBSTONE"],
                       parts=parts, board_calls=[Call("view", "NOT CHECKED", "glare")],
                       px_per_mm=44.5, common_shift_px=(1.5, -0.5), noise_px=0.8,
                       stamp="SINGLE GOLDEN")


def test_annotate_marks_the_flagged_part(images, result):
    _, test = images
    out = report.annotate(test, result)
    assert out.shape == test.shape and out.dtype == test.dtype
    assert not np.shares_memory(out, test)
    # the thick FAIL box on R12 is drawn along its taught top-left corner
    edge = out[150:170, 100:104].astype(int)
    red = (edge[..., 2] > 200) & (edge[..., 0] < 60) & (edge[..., 1] < 60)
    assert red.any()
    # C3 is drawn where it was found (moved +12, -5), not where it was taught
    found = out[145:175, 262:265].astype(int)
    assert ((found[..., 2] > 200) & (found[..., 0] < 60)).any()
    # the input is untouched
    assert np.array_equal(test, images[1])


def test_annotate_banner_changes_the_top_rows(images, result):
    _, test = images
    out = report.annotate(test, result)
    assert not np.array_equal(out[:10], test[:10])


def test_label_text(result):
    assert report._label(result.part("R12")) == "R12 MISSING score 0.97"
    assert report._label(result.part("C3")) == "C3 TOMBSTONE score 0.81 (unvalidated) +1"


def test_crop_pair_same_window_clipped(images):
    golden, test = images
    g, t = report.crop_pair(golden, test, Part("E", 0, 0, 20, 10))
    assert g.shape == t.shape == (26, 36, 3)      # margin 16, clipped at 0
    g, t = report.crop_pair(golden, test, Part("F", 380, 290, 20, 10))
    assert g.shape == t.shape == (26, 36, 3)      # clipped at the far edge
    g, t = report.crop_pair(golden, test, Part("M", 100, 100, 20, 10), margin_frac=0.5)
    assert g.shape == (30, 40, 3)
    assert np.array_equal(g, golden[90:120, 90:130])


def test_csv_has_one_row_per_call(tmp_path, result):
    path = tmp_path / "r.csv"
    report.write_csv(result, path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5                     # 1 + 2 + 1 part calls, 1 board call
    assert tuple(rows[0]) == report.CSV_FIELDS
    assert rows[1]["refdes"] == "C3" and rows[1]["unvalidated"] == "1"
    assert rows[1]["dx_px"] == "12.00"
    assert rows[2]["unvalidated"] == "0"
    assert rows[-1]["refdes"] == "" and rows[-1]["state"] == "NOT CHECKED"
    assert rows[-1]["dx_px"] == ""
    assert rows[3]["score"] == ""


def test_write_all_html_is_one_self_contained_file(tmp_path, images, result):
    golden, test = images
    paths = report.write_all(result, golden, test, tmp_path / "exports")
    assert paths == {"annotated": tmp_path / "exports" / "board_07_annotated.png",
                     "csv": tmp_path / "exports" / "board_07.csv",
                     "html": tmp_path / "exports" / "board_07.html"}
    assert all(p.exists() for p in paths.values())
    assert sorted(p.name for p in (tmp_path / "exports").iterdir()) == \
        ["board_07.csv", "board_07.html", "board_07_annotated.png"]

    doc = paths["html"].read_text(encoding="utf-8")
    assert "http" not in doc
    assert "data:image/jpeg;base64," in doc
    # board image + a golden and a test crop for each of the two flagged parts
    assert doc.count("data:image/jpeg;base64,") == 5
    assert "unvalidated on real boards" in doc
    assert "not calibrated" in doc
    assert "SINGLE GOLDEN" in doc and "44.50 px/mm" in doc


def test_html_escapes_text(tmp_path, images, result):
    _, test = images
    result.reasons = ["<script>alert(1)</script>"]
    result.part("R12").calls[0].detail = 'a "quoted" & <b>bold</b> detail'
    path = tmp_path / "x.html"
    report.write_html(result, test, {}, path)
    doc = path.read_text(encoding="utf-8")
    assert "<script>" not in doc and "<b>bold" not in doc
    assert "&lt;script&gt;" in doc


def test_html_calibrated_scores_are_not_marked(tmp_path, images):
    _, test = images
    r = BoardResult(image="b.jpg", program="demo", program_version=1, verdict="FAIL",
                    parts=[PartResult(Part("R1", 10, 10, 20, 10),
                                      [Call("missing", "MISSING", score=0.9,
                                            calibrated=True)])])
    path = tmp_path / "b.html"
    report.write_html(r, test, {}, path)
    doc = path.read_text(encoding="utf-8")
    assert "P=0.90" in doc and "not calibrated" not in doc
    assert "unvalidated on real boards" not in doc
