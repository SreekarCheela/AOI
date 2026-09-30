"""Landmark registration (register.register_landmarks) and its SIFT fallback."""
import cv2
import numpy as np
import pytest

from aoi.photo import register as R
from aoi.photo.model import Part


def _board(seed=3, h=1100, w=1500):
    """Texture with corners everywhere: blobs, lines and text on green."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (40, 110, 30), np.uint8)
    for _ in range(260):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        c = tuple(int(v) for v in rng.integers(60, 255, 3))
        k = rng.integers(0, 3)
        if k == 0:
            cv2.circle(img, (x, y), int(rng.integers(6, 22)), c, -1)
        elif k == 1:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 50)), y + int(rng.integers(10, 50))), c, -1)
        else:
            cv2.putText(img, f"R{int(rng.integers(1, 99))}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


PARTS = [Part("U1", 600, 450, 200, 120), Part("R1", 300, 300, 60, 30)]


def _warp(img, H_test_to_ref):
    """The test photo: the golden seen through the inverse of H."""
    return cv2.warpPerspective(img, np.linalg.inv(H_test_to_ref), (img.shape[1], img.shape[0]),
                               borderMode=cv2.BORDER_REPLICATE)


def _corner_error(Ha, Hb, shape):
    h, w = shape[:2]
    c = np.float32([[[100, 100], [w - 100, 100], [w - 100, h - 100], [100, h - 100]]])
    return float(np.abs(cv2.perspectiveTransform(c, Ha) - cv2.perspectiveTransform(c, Hb)).max())


def test_landmarks_avoid_parts_and_recover_a_known_alignment_to_a_fraction_of_a_pixel():
    g = _board()
    lm = R.make_landmarks(g, PARTS)
    assert len(lm.points) >= R.LM_MIN
    for p in PARTS:                                         # none inside a part's box
        inside = ((lm.points[:, 0] >= p.x) & (lm.points[:, 0] < p.x + p.w)
                  & (lm.points[:, 1] >= p.y) & (lm.points[:, 1] < p.y + p.h))
        assert not inside.any()
    th = np.deg2rad(2.0)
    H = np.array([[np.cos(th), -np.sin(th), 18.0], [np.sin(th), np.cos(th), -11.0], [2e-6, -1e-6, 1.0]])
    grey = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)            # noqa: E731
    reg = R.register_landmarks(lm, grey(_warp(g, H)), R.keypoints(grey(g)))
    assert reg.ok and "rough SIFT" in reg.note
    assert _corner_error(reg.H, H, g.shape) < 0.3


def test_second_board_starts_from_the_last_alignment_and_a_foreign_photo_falls_back_to_sift():
    g = _board()
    grey = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)            # noqa: E731
    lm = R.make_landmarks(g, PARTS)
    ref = R.keypoints(grey(g))
    H = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    assert "rough SIFT" in R.register_landmarks(lm, grey(_warp(g, H)), ref).note
    H2 = np.array([[1.0, 0.0, 9.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])       # the next board, a few px on
    reg = R.register_landmarks(lm, grey(_warp(g, H2)), ref)
    assert "last board" in reg.note and _corner_error(reg.H, H2, g.shape) < 0.3
    other = grey(_board(seed=11))
    reg = R.register_landmarks(lm, other, ref)
    assert "SIFT (landmarks:" in reg.note                             # never trusted: fell back


def test_landmarks_unstable_across_golden_shots_are_dropped():
    g = _board()
    moved = g.copy()
    lm_all = R.make_landmarks(g, PARTS)
    x, y = (int(v) for v in lm_all.points[0])
    cv2.rectangle(moved, (x - 45, y - 45), (x + 45, y + 45), (40, 110, 30), -1)   # gone on the second shot
    lm = R.make_landmarks(g, PARTS, [moved])
    assert not any((p == (x, y)).all() for p in lm.points)
    assert len(lm.points) == len(lm_all.points) - 1


@pytest.mark.parametrize("mode", ["sift", "landmarks"])
def test_golden_align_follows_the_config(monkeypatch, mode):
    from types import SimpleNamespace
    from aoi import config
    from aoi.photo import inspect as I
    monkeypatch.setattr(config, "PHOTO_REGISTRATION", mode)
    g = object.__new__(I.Golden)
    b = _board()
    g.bgr, g.grey = b, cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    g.keypoints = R.keypoints(g.grey)
    g.program = SimpleNamespace(parts=PARTS, golden_paths=lambda: [])
    g._landmarks, g._board_mask = None, np.ones((10, 10), np.uint8)
    reg = g.align(g.grey)
    assert reg.ok and (("landmarks via" in reg.note) == (mode == "landmarks"))
