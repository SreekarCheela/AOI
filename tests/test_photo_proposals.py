"""Box proposals: regressions for the failure classes found on real photos.

Root causes (investigation 2026-09-25, Golden.jpg of PT-PCB-004-03):
  * cutouts: the board region was the convex hull of the soldermask, which
    swallows notches and cutouts; beige desk seen through them (b* ~140) then
    passed the tan "ceramic cap" colour test and was proposed as a part.
"""
from __future__ import annotations

import cv2
import numpy as np

from aoi.photo import gate, teach

DESK = (110, 140, 160)        # BGR beige: L* ~150, b* ~140, dim enough to pass L < 175
MASK = (40, 110, 30)          # BGR soldermask green
BODY = (35, 38, 40)           # black resistor body


def _board_with_notch() -> np.ndarray:
    img = np.full((900, 1200, 3), DESK, np.uint8)
    cv2.rectangle(img, (100, 100), (1100, 800), MASK, -1)
    cv2.rectangle(img, (500, 600), (700, 801), DESK, -1)          # notch open to the bottom edge
    for x in (250, 850):                                           # two parts on the board
        cv2.rectangle(img, (x, 300), (x + 90, 345), BODY, -1)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def test_cutout_is_not_board():
    img = _board_with_notch()
    m = gate.board_mask(img)
    s = m.shape[1] / img.shape[1]
    assert m[int(700 * s), int(600 * s)] == 0        # inside the notch
    assert m[int(200 * s), int(600 * s)] == 1        # on the board
    assert m[int(320 * s), int(290 * s)] == 1        # a part body counts as board


def test_no_box_proposed_in_a_cutout():
    boxes = teach.propose_boxes(_board_with_notch())
    for x, y, w, h in boxes:
        cx, cy = x + w / 2, y + h / 2
        assert not (500 <= cx <= 700 and 600 <= cy <= 800), (x, y, w, h)


def test_board_fill_excludes_cutout():
    img = _board_with_notch()
    frac = (1000 * 700 - 200 * 200) / (1200 * 900)
    assert abs(gate.board_fill(img) - frac) < 0.03


class _FakeDetector:
    """Stands in for the trained model: returns fixed boxes in photo pixels."""

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, bgr, score_min=0.3, board_long_px=None):
        return [(*b, 0.9) for b in self.boxes]


def test_detector_boxes_in_a_cutout_or_off_board_are_dropped():
    img = _board_with_notch()
    on_board = (250, 300, 90, 45)
    in_notch = (560, 650, 80, 60)
    off_board = (20, 20, 50, 30)
    kept = teach.detector_boxes(img, _FakeDetector([on_board, in_notch, off_board]))
    assert kept == [on_board]


def test_decode_finds_a_peak_and_its_size():
    import math
    import torch
    from aoi.photo import detect as D
    hm = torch.full((1, 1, 32, 32), -8.0)
    hm[0, 0, 10, 20] = 4.0                               # one confident peak
    wh = torch.zeros((1, 2, 32, 32)); wh[0, 0, 10, 20] = math.log(25 / D.STRIDE); wh[0, 1, 10, 20] = math.log(12 / D.STRIDE)
    off = torch.zeros((1, 2, 32, 32)); off[0, :, 10, 20] = 0.5
    (x, y, w, h, s), = D.decode(hm, wh, off, 0.3)
    assert abs(w - 25) < 1e-3 and abs(h - 12) < 1e-3 and s > 0.95
    assert abs((x + w / 2) - 20.5 * D.STRIDE) < 1e-3 and abs((y + h / 2) - 10.5 * D.STRIDE) < 1e-3


def test_training_targets_peak_at_the_box_centre():
    import math
    from aoi.photo import detect as D, detect_train as T
    hm, wh, off, mask = T.targets([(100, 60, 40, 20)])
    cy, cx = int((60 + 10) / D.STRIDE), int((100 + 20) / D.STRIDE)
    assert hm[cy, cx] == 1.0 and mask.sum() == 1
    assert abs(wh[0, cy, cx] - math.log(40 / D.STRIDE)) < 1e-6


def test_proposals_fall_back_to_colour_without_weights(tmp_path):
    boxes = teach.propose_boxes(_board_with_notch(), detector_weights=tmp_path / "none.pt")
    assert len(boxes) >= 2                                # the two black bodies, from colour
