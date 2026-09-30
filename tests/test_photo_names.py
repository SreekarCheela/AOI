"""Part names read off the silkscreen (aoi/photo/names.py).

The logic tests build labels by hand; the fixture test runs the real OCR on the
2026-09-25 golden and is skipped where the photos or a recogniser are missing.
"""
from pathlib import Path

import numpy as np
import pytest

from aoi.photo import names as N
from aoi.photo import program, teach
from aoi.photo.model import Part

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "photos/fixtures/2026-09-25"


def lab(name, x, y, h=40.0, score=1.0, w=None, votes=None):
    w = w if w is not None else 0.6 * h * len(name)
    return N.Label(name, x, y, h, score, (x - w / 2, y - h / 2, x + w / 2, y + h / 2),
                   votes if votes is not None else {name: score})


# ------------------------------------------------------------------ grammar
@pytest.mark.parametrize("text, want", [
    ("R7 R4 T2", ["R7", "R4", "T2"]),
    ("R6R3", ["R6", "R3"]),
    ("R10D1", ["R10", "D1"]),
    ("R1O D1", ["R10", "D1"]),             # O read for 0 in a number
    ("Di", ["D1"]),
    ("22R0", []),                          # a resistor's value marking
    ("PT-PCB-004-03", []),
    ("MADE IN INDIA", []),
    ("R8-", ["R8"]),
    ("BD1", ["BD1"]),
    ("LED3", ["LED3"]),
    ("R0", []),                            # designator numbers do not start at 0
])
def test_designators_are_read_out_of_ocr_lines(text, want):
    assert [n for n, _, _ in N.designators(text)] == want


def test_designator_spans_point_at_their_own_characters():
    t = "R7 R4 T2"
    assert [t[a:b] for _, a, b in N.designators(t)] == ["R7", "R4", "T2"]


@pytest.mark.parametrize("name, want", [
    ("R12", ("resistor", False)), ("C9", ("capacitor", False)), ("D4", ("diode", True)),
    ("LED1", ("led", True)), ("BD1", ("bridge", True)), ("T3", ("sot23", True)),
    ("Q2", ("sot23", True)), ("U1", ("ic", True)), ("L1", ("other", False)), ("junk", ("other", False)),
])
def test_the_letter_sets_type_and_polarity(name, want):
    assert N.type_of(name) == want


def test_merge_pools_readings_of_one_place_by_summed_confidence():
    # PP-OCR's one "R6" against Vision's two "R8" at the same place
    got = N.merge([lab("R6", 100, 100, score=0.93), lab("R8", 102, 101, score=0.9), lab("R8", 99, 99, score=0.95)])
    assert [g.name for g in got] == ["R8"] and set(got[0].votes) == {"R6", "R8"}


def test_merge_keeps_a_sideways_list_of_names_apart():
    # Test 2's C7 R12 C8 R13 printed rotated: neighbouring names one line
    # (~0.8 character heights) apart, side by side, not one text read twice
    got = N.merge([lab(n, 100 + 80 * k, 100, h=100, w=75) for k, n in enumerate(["C7", "R12", "C8", "R13"])])
    assert sorted(g.name for g in got) == ["C7", "C8", "R12", "R13"]


# ------------------------------------------------------------------ assignment
def row_board():
    """A row of ten parts with their names printed as a row centred below
    each one, closer to the neighbours' names than a nearest-point match can
    tell apart (the fixture's R9 T3 R5 R8 ... row)."""
    rects, labels, truth = [], [], []
    for k in range(10):
        x = 100 + 90 * k
        rects.append((x, 100, 40, 60))
        truth.append(f"R{k + 1}")
        labels.append(lab(f"R{k + 1}", x + 20, 100 + 60 + 45))
    return rects, labels, truth


def test_a_row_of_names_below_a_row_of_parts_is_matched_one_to_one():
    rects, labels, truth = row_board()
    assert N.assign(rects, labels) == truth


def test_a_name_with_no_part_stays_unused_and_a_far_name_is_not_taken():
    rects, labels, truth = row_board()
    labels.append(lab("R99", 3000, 3000))                  # an unfitted footprint far away
    del labels[4]                                          # R5's label not read
    got = N.assign(rects, labels)
    assert got[4] is None and "R99" not in got
    assert got[:4] == truth[:4] and got[5:] == truth[5:]


def test_marking_on_a_body_is_not_a_name():
    rects, labels, truth = row_board()
    labels.append(lab("R50", 120, 130))                    # centred on the first body
    assert N.assign(rects, labels) == truth


def test_a_list_of_names_past_the_end_of_a_column_is_matched_in_order():
    # four parts stacked, their names listed below the column in the same order
    rects = [(500, 100 + 50 * k, 120, 40) for k in range(4)]
    names = ["C7", "R12", "C8", "R13"]
    labels = [lab(n, 560, 360 + 45 * k) for k, n in enumerate(names)]
    assert N.assign(rects, labels) == names


def test_numbering_repairs_a_lone_misread_letter():
    rects, labels, truth = row_board()
    labels[3] = lab("F4", labels[3].x, labels[3].y)        # R4 read as F4, the only F
    assert N.assign(rects, labels) == truth


def test_a_duplicate_takes_its_other_reading_when_that_fills_a_gap():
    rects, labels, truth = row_board()
    # R11 read as "R1" (a thin 1 lost), with an R11 reading at the same place
    rects.append((100 + 90 * 10, 100, 40, 60))
    truth.append("R11")
    labels.append(lab("R1", 100 + 90 * 10 + 20, 205, score=0.9, votes={"R1": 0.9, "R11": 0.5}))
    assert N.assign(rects, labels) == truth


def test_a_ceramic_body_does_not_take_a_resistor_name():
    assert N._type_fit("R3", "tan") == "wrong" and N._type_fit("C3", "tan") == "ok"
    assert N._type_fit("C3", "dark") == "tantalum" and N._type_fit("D1", "dark") == "ok"


def test_named_parts_take_type_and_polarity_and_keep_a_name_once():
    parts = [Part("P1", 0, 0, 10, 10), Part("P2", 20, 0, 10, 10), Part("P3", 40, 0, 10, 10)]
    out = N.named_parts(parts, ["D4", None, "D4"])
    assert [(p.refdes, p.type, p.polarized) for p in out] == [
        ("D4", "diode", True), ("P2", "other", False), ("P3", "other", False)]


# ------------------------------------------------------------------ the real board
needs_ocr = pytest.mark.skipif(not ((FIXTURE / "Golden.jpg").exists() and N.available()),
                               reason="needs photos/fixtures/2026-09-25, rapidocr and models/ocr")


# ------------------------------------------------------------------ offline models
def test_missing_models_leave_parts_unnamed_with_a_reason_and_never_download(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "OCR_MODEL_DIR", tmp_path)
    monkeypatch.setattr(N, "_ENGINE", {})
    assert set(N.missing_models()) >= {"PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_medium.onnx"}
    assert not N.available()
    res = N.name_boxes(np.zeros((40, 40, 3), np.uint8), [(1, 1, 8, 8)])
    assert res.names == [None] and "fetch" not in res.note and "models/ocr/README.md" in res.note
    with pytest.raises(N.OCRModelsMissing):
        N._engine("medium")


def test_install_models_refuses_a_file_that_does_not_match_its_pinned_hash(tmp_path, monkeypatch):
    src = tmp_path / "usb"; src.mkdir()
    name = N.CLS_MODEL
    (src / name).write_bytes(b"not the model")
    monkeypatch.setattr(N, "MODELS", {name: N.MODELS[name]})
    with pytest.raises(N.OCRModelsMissing, match="checksum"):
        N.install_models(src, tmp_path / "ocr")
    assert not (tmp_path / "ocr" / name).exists()


def test_readers_are_the_same_on_every_platform():
    # no platform-specific recogniser: a Mac names parts exactly as the production machine
    assert all(r.split("@")[0] in N.ENGINES for r in N.BOARD_READERS + N.LOCAL_READERS)
    assert not hasattr(N, "have_vision")


def test_an_enlarged_read_reports_points_in_the_image_it_was_given(monkeypatch):
    class Out:
        word_results = [[("R", 0.9, [[20, 10], [30, 10], [30, 30], [20, 30]]),
                         ("8", 0.9, [[30, 10], [40, 10], [40, 30], [30, 30]])]]
        scores = [0.9]
    seen = {}

    def fake(img):
        seen["shape"] = img.shape
        return Out()
    monkeypatch.setattr(N, "_engine", lambda name: fake)
    got = N._reader("small@2")(np.zeros((50, 60, 3), np.uint8))
    assert seen["shape"][:2] == (100, 120)
    assert got[0][0] == "R8" and got[0][2][:, 0].max() == 20.0     # 40 px at 2x -> 20 px


@needs_ocr
def test_fixture_golden_names_nearly_every_part_and_never_with_the_wrong_type():
    from aoi.photo.io import load_bgr
    g = load_bgr(FIXTURE / "Golden.jpg")
    truth = program.read_parts_csv(FIXTURE / "parts.csv")
    rects = [(q.x, q.y, q.w, q.h) for q in truth]
    res = N.name_boxes(g, rects)
    right = sum(n == q.refdes for n, q in zip(res.names, truth))
    wrong = [(q.refdes, n) for n, q in zip(res.names, truth) if n and n != q.refdes]
    # measured 2026-09-27: 33-36 right, 0-1 wrong; 2026-09-30 (medium PP-OCR, no Apple
    # Vision): 33 right, 1 wrong (R11 as R1), 3 unnamed (in-sample: this board shaped the rules)
    assert right >= 32, (right, wrong)
    assert len(wrong) <= 2, wrong
    assert all(N.type_of(n)[0] == N.type_of(t)[0] for t, n in wrong), wrong
