"""The photo checker: missing and misplaced parts from ordinary pictures.

`inspect_photo.py` sees only one grey image per board -- the diffuse channel
stands in for a photograph -- and a box per part drawn on a golden photo. The
boxes here come from CAD so the test knows exactly where each body is.

The regressions these pin were all found building it:

  * a skewed part read as 0 deg, because the fixed lands in the template argue
    for zero rotation whatever the body did
  * a plain body on a plain board matched a stretch of empty board a body-height
    away, because correlation throws away the brightness difference that is the
    only thing separating them
  * a part near the edge of the photo read as offset, because the black border
    of the warped photo skewed the brightness matching
"""
from __future__ import annotations

import numpy as np
import pytest

import inspect_photo as ip
from aoi import board as B
from aoi import config, roi, synth

DIF = config.CHANNELS.index("diffuse")


def photo(brd: B.Board, defects: dict | None = None, seed: int = 0, **opts) -> np.ndarray:
    return synth.render(brd, defects=defects or {},
                        opts=synth.RenderOpts(seed=seed, **opts))[..., DIF]


def parts_for(brd: B.Board) -> list[ip.Part]:
    out = []
    for c in brd.components:
        h = roi.body_rect(c.pkg)
        corners = np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]])
        pts = roi.local_to_board(c, corners) / config.MM_PER_PX
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        out.append(ip.Part(c.refdes, int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return out


def check(ref: np.ndarray, img: np.ndarray, parts: list[ip.Part]) -> dict[str, ip.Verdict]:
    reg = ip.register(ref, img)
    assert reg.ok, reg.note
    warped, cover = ip.warp_to_ref(img, reg.H, ref.shape)
    return {p.name: ip.judge(p, ip.find_part(ref, warped, p), ip.DEFAULT_MISSING_NCC,
                             config.UM_PER_PX, config.MAX_OFFSET_UM, 0.25,
                             config.MAX_SKEW_DEG)
            for p in parts if ip.in_view(cover, p)}


@pytest.fixture(scope="module")
def board() -> B.Board:
    # Dense enough to give the feature registration something to lock onto;
    # the suite's hand-built board is too sparse for that.
    return B.random_board(40.0, 30.0, seed=1)


@pytest.fixture(scope="module")
def ref(board: B.Board) -> np.ndarray:
    return photo(board, seed=100)


def test_clean_reshoot_passes(board, ref):
    """A second photo of a good board, moved and turned on the desk, flags nothing."""
    img = photo(board, seed=9, board_shift_mm=(-0.3, 0.2), board_rot_deg=-0.4)
    v = check(ref, img, parts_for(board))
    assert len(v) >= len(board.components) - 3        # a few fall off the frame edge
    assert [n for n, x in v.items() if x.failed] == []


def test_defects_are_found_and_named(board, ref):
    cs = board.components
    truth = {cs[0].refdes: synth.Defect("missing"),
             cs[3].refdes: synth.Defect("missing"),
             cs[5].refdes: synth.Defect("offset", dx=0.25),
             cs[8].refdes: synth.Defect("offset", dx=-0.2),
             cs[11].refdes: synth.Defect("skew", dtheta=20.0)}
    img = photo(board, truth, seed=7, board_shift_mm=(0.42, -0.31), board_rot_deg=0.6)
    v = check(ref, img, parts_for(board))

    flagged = {n: x.reason for n, x in v.items() if x.failed}
    assert flagged == {k: d.kind for k, d in truth.items()}

    # The measured angle is the injected one, not a fraction of it.
    assert abs(abs(v[cs[11].refdes].match.angle_deg) - 20.0) <= 3.0
    # Offsets read somewhat short -- the body blur and the lands under the body
    # edge both pull toward nominal -- but they must clear the 150 um limit.
    for r in (cs[5].refdes, cs[8].refdes):
        m = v[r].match
        assert np.hypot(m.dx_px, m.dy_px) * config.UM_PER_PX > config.MAX_OFFSET_UM


def test_unrelated_photo_is_refused(ref):
    """A different board must not be 'aligned' and then judged part by part."""
    other = photo(B.random_board(40.0, 30.0, seed=5), seed=3)
    assert not ip.register(ref, other).ok
