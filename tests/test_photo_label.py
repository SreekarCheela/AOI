"""truth.json editing (aoi/photo/truth.py) and the labelling window's logic."""
import json

import cv2
import numpy as np
import pytest

from aoi.photo.model import Part
from aoi.photo.truth import DEFECTS, Truth


def _folder(tmp_path, names=("golden1.jpg", "b1.jpg", "b2.jpg")):
    for n in names:
        cv2.imwrite(str(tmp_path / n), np.zeros((8, 8, 3), np.uint8))
    return tmp_path


def test_set_clear_and_keep_other_keys(tmp_path):
    f = _folder(tmp_path)
    (f / "truth.json").write_text(json.dumps({"source": "shoot 3", "truth": []}))
    t = Truth(f)
    t.set("b1", "r7", "skew")                                   # names are upper-cased
    t.set("b1", "C2", "missing")
    t.set("b1", "R7", "tombstone")                              # one defect per part: replaced
    assert Truth(f).on("b1") == {"C2": "missing", "R7": "tombstone"}
    t.set("b1", "C2", None)
    d = json.loads((f / "truth.json").read_text())
    assert d["source"] == "shoot 3" and d["truth"] == [{"image": "b1", "refdes": "R7", "defect": "tombstone"}]
    with pytest.raises(ValueError):
        t.set("b1", "R1", "bent")
    assert set(DEFECTS) >= {"missing", "skew", "wrong_part", "tombstone"}


def test_goldens_are_not_boards_and_a_labelled_board_cannot_become_one(tmp_path):
    f = _folder(tmp_path)
    t = Truth(f)
    t.set_golden(f / "golden1.jpg", True)
    assert [p.name for p in t.boards()] == ["b1.jpg", "b2.jpg"]
    t.set("b1", "R1", "missing")
    with pytest.raises(ValueError, match="clear them"):
        t.set_golden(f / "b1.jpg", True)
    t.set_golden(f / "golden1.jpg", False)
    assert len(t.boards()) == 3
    from aoi.photo import evaluate
    assert evaluate.non_boards(evaluate.read_truth(f / "truth.json")) == set()


def test_part_at_picks_the_smallest_box_under_the_click():
    pytest.importorskip("tkinter")
    from aoi_photo.label import part_at
    parts = [Part("U1", 0, 0, 100, 100), Part("R1", 40, 40, 10, 10)]
    assert part_at(parts, 45, 45).refdes == "R1"
    assert part_at(parts, 5, 5).refdes == "U1"
    assert part_at(parts, 500, 5) is None
