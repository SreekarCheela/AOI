"""Turning real photographs into the four-channel stack.

The prototype shoot is a camera on a tripod and one lamp moved between frames.
Two things quietly ruin it, and both are handled here rather than in the
operator's head: display gamma, which makes JPEG pixel values not proportional
to the light that arrived, and registration between frames.

Registration is the one that does not announce itself. Each angled frame lights
a DIFFERENT slope band, so the frames genuinely do not look alike -- a fillet
blazing in the low frame is dark in the high one. Correlating them on raw image
gradient locks onto the specular highlights, which are the brightest thing in
the picture and the part that moves between frames rather than staying put. The
result is a false-colour image that is merely smeared, and the natural reading
of that is "the optical design does not work" rather than "the frames do not
line up" -- which would be an expensive conclusion to draw wrongly.
"""
from __future__ import annotations

import sys

import numpy as np
import cv2
import pytest

from aoi import board as B
from aoi import capture, config, synth

TRUE_SHIFTS = [(5.0, 3.0), (-7.0, 2.0), (12.0, -9.0)]
PAIRS = [("low", "mid"), ("low", "high"), ("low", "diffuse"),
         ("mid", "high"), ("high", "diffuse")]
# The lamp is moved between shots, so frames differ in overall brightness by
# several times even with the camera's exposure locked.
LAMP = {"low": 1.0, "mid": 0.55, "high": 0.30, "diffuse": 1.4}


@pytest.fixture(scope="module")
def frames() -> dict[str, np.ndarray]:
    """One linear frame per channel, off a board with realistic structure."""
    brd = B.random_board(30.0, 22.0, seed=1)
    stack = synth.render(brd, opts=synth.RenderOpts(seed=5))
    return {c: stack[..., config.CHANNELS.index(c)].astype(np.float32) / 255.0
            for c in config.CHANNELS}


def _shift_image(a: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- the measure
def test_a_shift_is_recovered_within_a_frame(frames):
    """Baseline: the correlator and the sign convention are not the problem."""
    for ch, img in frames.items():
        dx, dy = capture._shift(img, _shift_image(img, 5.0, 3.0))
        assert dx == pytest.approx(5.0, abs=0.6), f"{ch} dx {dx:+.2f}"
        assert dy == pytest.approx(3.0, abs=0.6), f"{ch} dy {dy:+.2f}"


@pytest.mark.parametrize("ref,mov", PAIRS)
@pytest.mark.parametrize("dx,dy", TRUE_SHIFTS)
def test_frames_lit_differently_still_register(frames, ref, mov, dx, dy):
    """The bug this file exists for. Frames lighting different slope bands share
    almost no bright structure, and that is the normal case here, not an edge."""
    got_x, got_y = capture._shift(frames[ref], _shift_image(frames[mov], dx, dy))
    assert got_x == pytest.approx(dx, abs=1.5), \
        f"{ref}->{mov} dx {got_x:+.2f}, wanted {dx:+.1f}"
    assert got_y == pytest.approx(dy, abs=1.5), \
        f"{ref}->{mov} dy {got_y:+.2f}, wanted {dy:+.1f}"


def test_the_specular_highlights_are_what_had_to_be_suppressed(frames):
    """Pins WHY `_prep` compresses before taking a gradient, by measuring the
    thing it replaced. Correlating raw image gradient is the obvious approach
    and it is wrong here: the gradient is dominated by the specular highlights,
    which are precisely the content that moves from frame to frame.

    If this ever stops failing, the compression is no longer earning its place
    and the simpler code should come back.
    """
    def raw_gradient_prep(a):
        a = cv2.GaussianBlur(a, (0, 0), 1.2)
        m = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3),
                          cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))
        return np.ascontiguousarray((m - m.mean()) / (m.std() + 1e-8), np.float64)

    worst_old = 0.0
    for ref, mov in PAIRS:
        moved = _shift_image(frames[mov], 5.0, 3.0)
        (dx, _), _ = cv2.phaseCorrelate(raw_gradient_prep(frames[ref]),
                                        raw_gradient_prep(moved))
        worst_old = max(worst_old, abs(dx - 5.0))
    assert worst_old > 2.0, (
        f"raw-gradient correlation is only {worst_old:.2f} px out across every "
        "channel pair; the compression in _prep may no longer be needed")


def test_prep_is_invariant_to_overall_frame_brightness(frames):
    """Why each frame is normalised before the log, tested as the property
    rather than through its effect.

    The lamp moves between shots, so the frames differ several-fold in overall
    level. A log is not scale invariant, so without the normalisation the SAME
    scene compresses to a different shape in each frame and the correlation is
    being asked to match two things that no longer look alike.

    Measured through registration this is board-dependent -- 4.6 px of error on
    a 54-component board, none at all on a 24-component one -- so a downstream
    assertion would pass or fail on the fixture rather than on the code. The
    property itself does not care: with the normalisation the two preps are
    bit-identical, without it they differ by 0.18 whatever the board.
    """
    for ch, img in frames.items():
        a = capture._prep(img)
        b = capture._prep(img * 0.30)
        assert np.abs(a - b).mean() < 1e-3, (
            f"{ch}: dimming the frame changed its prepared form by "
            f"{np.abs(a - b).mean():.4f}; the log is seeing a different scale")


def test_registration_survives_the_lamp_changing_the_frame_brightness(frames):
    """The lamp moves, so frames differ several-fold in overall brightness. The
    compression in `_prep` is a log, which is not scale invariant, so each frame
    is normalised to its own level first. Without that step this returns."""
    worst = 0.0
    for ref, mov in PAIRS:
        a = frames[ref] * LAMP[ref]
        b = _shift_image(frames[mov] * LAMP[mov], 5.0, 3.0)
        dx, dy = capture._shift(a, b)
        worst = max(worst, abs(dx - 5.0), abs(dy - 3.0))
    assert worst < 1.5, f"{worst:.2f} px of error once the frames differ in brightness"


def test_all_channels_end_up_on_the_same_grid(frames):
    """What the whole thing is for. Each frame is nudged differently, as a
    tripod and a mirror do, and afterwards every channel has to sit on one grid
    -- band_fractions reads all three angled channels at the SAME pixels, so
    misregistration reads straight through as a wrong joint measurement."""
    nudge = {"low": (3.0, 1.0), "mid": (0.0, -2.0),
             "high": (-2.0, -4.0), "diffuse": (-4.0, -4.0)}
    order = list(config.CHANNELS)
    moved = [_shift_image(frames[c], *nudge[c]) for c in order]
    out, shifts = capture.align(moved)

    assert all(ok for *_, ok in shifts), f"a shift was refused: {shifts}"
    residual = [capture._shift(frames[c], o) for c, o in zip(order, out)]
    xs = [r[0] for r in residual]
    ys = [r[1] for r in residual]
    assert max(xs) - min(xs) < 1.5, \
        f"channels still {max(xs) - min(xs):.2f} px apart in x: {dict(zip(order, xs))}"
    assert max(ys) - min(ys) < 1.5, \
        f"channels still {max(ys) - min(ys):.2f} px apart in y: {dict(zip(order, ys))}"


# ------------------------------------------------------------------- guarding
def test_an_impossible_shift_is_refused_rather_than_applied(frames):
    """A structureless frame -- blown out, or the lens cap left on -- gives a
    correlation peak that is meaningless and enormous. Warping by it turns a
    frame that was nearly right into one that is unusable, so refuse, keep the
    frame as shot, and report it. Loud beats silent."""
    blank = np.full_like(frames["low"], 0.5)
    out, shifts = capture.align([frames["low"], blank])
    dx, dy, ok = shifts[1]
    assert not ok, f"a {dx:+.0f},{dy:+.0f} px shift was accepted as a tripod nudge"
    assert np.array_equal(out[1], blank), "the refused frame was warped anyway"


def test_alignment_can_be_turned_off(frames):
    order = list(config.CHANNELS)
    imgs = [frames[c] for c in order]
    out, shifts = capture.align(imgs, enable=False)
    assert all(a is b for a, b in zip(out, imgs)), "frames were touched"
    assert shifts == [], (
        "disabled alignment reported per-frame shifts. A caller cannot then "
        "tell 'not attempted' from 'attempted and refused', and will warn about "
        "frames that are perfectly fine")


# ---------------------------------------------------------------- end to end
def _write_shoot(folder, stack, gamma_encode=True, nudge=None):
    """Write a rendered stack out as camera JPEGs, display-encoded."""
    for i, ch in enumerate(config.CHANNELS):
        img = stack[..., i].astype(np.float32) / 255.0
        if nudge:
            img = _shift_image(img, *nudge[ch])
        if gamma_encode:
            img = np.power(np.clip(img, 0, 1), 1 / 2.2)
        cv2.imwrite(str(folder / f"{ch}.jpg"), (img * 255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


@pytest.fixture(scope="module")
def shoot_stack():
    brd = B.Board("SHOOT", 18.0, 8.0, [
        B.Component("R1", "0805", x=4.0, y=4.0),
        B.Component("R2", "0805", x=9.0, y=4.0),
        B.Component("R3", "0603", x=14.0, y=4.0)], [])
    return synth.render(brd, defects={"R2": synth.Defect("insufficient", pad=0)},
                        opts=synth.RenderOpts(seed=3, place_sigma_um=0.0))


def test_a_locked_camera_reproduces_the_stack_it_was_shot_from(tmp_path, shoot_stack):
    """The path the first shoot should actually use: nothing moves but the lamp,
    so there is nothing to register and the arithmetic is all that is being
    asked of the code. What has to survive is the LIT FRACTION, because that is
    the number band_fractions computes, not the mean or the peak."""
    _write_shoot(tmp_path, shoot_stack)
    stack = capture.build_stack(capture.find_shots(tmp_path), gamma=2.2,
                                do_align=False)
    for ch in config.ANGLED_CHANNELS:
        k = config.CHANNELS.index(ch)
        want = (shoot_stack[..., k] / 255.0 > config.BAND_THRESHOLD).mean()
        got = (stack[..., k] / 255.0 > config.BAND_THRESHOLD).mean()
        assert got == pytest.approx(want, abs=0.002), (
            f"{ch} lit fraction {got * 100:.2f}% against {want * 100:.2f}% in the "
            "frames it was built from")


def test_build_stack_reports_what_registration_did(tmp_path, shoot_stack):
    """Misregistration is the silent failure here, so the numbers have to come
    back to the caller rather than staying inside the function."""
    nudge = {"low": (3.0, 1.0), "mid": (0.0, -2.0),
             "high": (-2.0, -4.0), "diffuse": (-4.0, -4.0)}
    _write_shoot(tmp_path, shoot_stack, nudge=nudge)
    shifts: list = []
    capture.build_stack(capture.find_shots(tmp_path), gamma=2.2, do_align=True,
                        shifts_out=shifts)
    assert [s[0] for s in shifts] == list(config.CHANNELS)
    by_ch = {c: (dx, dy) for c, dx, dy, _ in shifts}
    for ch in ("mid", "high", "diffuse"):
        want_x = nudge[ch][0] - nudge["low"][0]
        assert by_ch[ch][0] == pytest.approx(want_x, abs=1.5), (
            f"{ch} reported dx {by_ch[ch][0]:+.2f}, but it was shot "
            f"{want_x:+.1f} px from the reference frame")


def test_display_gamma_is_undone(tmp_path, shoot_stack):
    """A JPEG carries a display gamma, so its values are not proportional to the
    light that arrived. The whole method compares brightness BETWEEN frames, so
    leaving it in is not a cosmetic error."""
    _write_shoot(tmp_path, shoot_stack)
    shots = capture.find_shots(tmp_path)
    linear = capture.build_stack(shots, gamma=2.2, do_align=False)
    as_shot = capture.build_stack(shots, gamma=None, do_align=False)
    k = config.CHANNELS.index("diffuse")
    want = (shoot_stack[..., k] / 255.0 > config.BAND_THRESHOLD).mean()
    got_lin = (linear[..., k] / 255.0 > config.BAND_THRESHOLD).mean()
    got_raw = (as_shot[..., k] / 255.0 > config.BAND_THRESHOLD).mean()
    assert got_lin == pytest.approx(want, abs=0.01)
    assert abs(got_raw - want) > abs(got_lin - want), (
        "leaving the display gamma in made no difference to the lit fraction; "
        "this test is not measuring what it claims to")


# --------------------------------------------------------------- file handling
def test_frames_are_matched_to_channels_by_name(tmp_path):
    for name in ("IMG_01_low.jpg", "IMG_02_mid.JPG", "shot-high.png",
                 "diffuse_bounce.jpg", "notes.txt"):
        if name.endswith(".txt"):
            (tmp_path / name).write_text("x")
        else:
            cv2.imwrite(str(tmp_path / name), np.zeros((8, 8), np.uint8))
    shots = capture.find_shots(tmp_path)
    assert set(shots) == set(config.CHANNELS)
    assert shots["low"].name == "IMG_01_low.jpg"


def test_a_missing_diffuse_frame_is_synthesised(tmp_path, shoot_stack):
    """Three lamp positions and no flat frame is a plausible way to come back
    from a first shoot. It should still produce a usable stack."""
    _write_shoot(tmp_path, shoot_stack)
    (tmp_path / "diffuse.jpg").unlink()
    shots = capture.find_shots(tmp_path)
    assert "diffuse" not in shots
    stack = capture.build_stack(shots, gamma=2.2, do_align=False)
    assert stack.shape[2] == len(config.CHANNELS)
    assert stack[..., config.CHANNELS.index("diffuse")].max() > 0, \
        "no diffuse frame was synthesised, so the body template has nothing to work from"


def test_a_raw_frame_without_rawpy_says_what_to_do(tmp_path, monkeypatch):
    """rawpy is optional, so a DNG off a phone is a normal thing to arrive with
    and a bare ModuleNotFoundError is the wrong answer to it.

    Simulated rather than skipped when rawpy IS installed: the message is what
    the person standing over the board reads, so it has to be pinned on every
    machine, not only on the ones that happen to lack the package.
    """
    monkeypatch.setitem(sys.modules, "rawpy", None)      # makes `import rawpy` fail
    dng = tmp_path / "low.dng"
    dng.write_bytes(b"not really a dng")

    with pytest.raises(ImportError) as e:
        capture.load_image(dng)

    msg = str(e.value)
    assert "low.dng" in msg, "the message does not say which file it choked on"
    assert "pip install rawpy" in msg, "no remedy given"
    assert "JPEG" in msg, "the no-install way out is not offered"


def test_raw_files_are_still_recognised_as_frames(tmp_path):
    """The gate check in cli.py keys off find_shots, so a DNG has to be found as
    a frame even when it cannot be read -- otherwise the shoot reports 'no
    frames' and the real reason never surfaces."""
    for name in ("low.dng", "mid.DNG", "high.jpg"):
        (tmp_path / name).write_bytes(b"x")
    shots = capture.find_shots(tmp_path)
    assert set(shots) == {"low", "mid", "high"}
    assert [p for p in shots.values() if p.suffix.lower() in capture.RAW_EXT], \
        "no RAW frame detected, so cli.py would never print the rawpy message"
