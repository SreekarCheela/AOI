"""Photo mode: loader, program, judge rules, and the real-photo pass bar.

The judge tests drive `judge_part` with hand-built measurement dicts, so every row
of the part-state table is pinned without a photo. The real-photo tests use the
2026-09-25 iPhone shoot of PT-PCB-004-03 (photos/fixtures/2026-09-25, gitignored)
and skip with a stated reason when it is absent. That fixture has ONE golden, so
it runs in single-golden mode, and the frozen missing-part model was fitted on
these same photos: the missing-part numbers below are in-sample. Out of sample
(leave one board out) the same features caught 12 of 14 with no false call.
"""
from __future__ import annotations

import json
import math
import socket
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi import capture, config
from aoi.photo import io, judge, program
from aoi.photo.model import BoardResult, Call, Part, PartResult

FIXTURE = Path("photos/fixtures/2026-09-25")
needs_fixture = pytest.mark.skipif(not (FIXTURE / "Golden.jpg").exists(),
                                   reason=f"real-photo fixture {FIXTURE} absent (photos/ is gitignored)")


# ------------------------------------------------------------------- loading
def _exif_rotated_jpeg(path: Path) -> np.ndarray:
    """A JPEG whose pixels are stored upside down with EXIF orientation 3."""
    from PIL import Image
    upright = np.zeros((40, 60, 3), np.uint8)
    upright[:10, :20] = (0, 0, 255)                      # red block top-left when upright (BGR)
    stored = np.rot90(upright, 2)
    im = Image.fromarray(stored[..., ::-1].copy())
    exif = Image.Exif(); exif[0x0112] = 3
    im.save(path, exif=exif, quality=95)
    return upright


def test_exif_rotated_jpeg_loads_upright(tmp_path):
    f = tmp_path / "rot.jpg"
    _exif_rotated_jpeg(f)
    bgr = io.load_bgr(f)
    assert bgr[:10, :20, 2].mean() > 150 and bgr[-10:, -20:, 2].mean() < 60
    g = capture.load_image(f)                            # the station loader, same fix
    assert g[:10, :20].mean() > g[-10:, -20:].mean() + 0.05


def test_sixteen_bit_tiff_keeps_depth(tmp_path):
    a = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 13) % 65535
    f = tmp_path / "deep.tif"
    cv2.imwrite(str(f), a)
    g = capture.load_image(f)
    assert g.dtype == np.float32
    assert np.allclose(g, a.astype(np.float32) / 65535.0, atol=1e-6)


def test_heic_without_pillow_heif_says_what_to_do(tmp_path):
    if io.have_heic():
        pytest.skip("pillow-heif is installed")
    f = tmp_path / "x.heic"; f.write_bytes(b"\0" * 16)
    with pytest.raises(io.PhotoError, match="pillow-heif"):
        io.load_bgr(f)


def test_unknown_extension_rejected(tmp_path):
    f = tmp_path / "x.gif"; f.write_bytes(b"GIF89a")
    with pytest.raises(io.PhotoError, match="unsupported"):
        io.load_bgr(f)


# ------------------------------------------------------------------ programs
def test_parts_csv_round_trip_and_errors(tmp_path):
    parts = [Part("R1", 1, 2, 30, 15, "resistor", "0603", False, "none"),
             Part("D1", 50, 60, 40, 20, "diode", "SOD-123FL", True, "right")]
    f = tmp_path / "parts.csv"
    program.write_parts_csv(f, parts)
    assert program.read_parts_csv(f) == parts
    f.write_text("refdes,x,y,w,h\nR1,1,2,3,4\nR1,5,6,7,8\n")
    with pytest.raises(io.PhotoError, match="twice"):
        program.read_parts_csv(f)
    f.write_text("refdes,x,y,w,h,type\nR1,1,2,3,4,blob\n")
    with pytest.raises(io.PhotoError, match="type"):
        program.read_parts_csv(f)


def test_fewer_than_three_goldens_refused_unless_single_golden():
    with pytest.raises(io.PhotoError, match="at least 3"):
        program.check_golden_count(2, single_golden=False)
    program.check_golden_count(1, single_golden=True)
    program.check_golden_count(3, single_golden=False)


def test_program_versions_and_resolve(tmp_path):
    for _ in range(2):
        folder, v = program.new_version_folder(tmp_path, "board")
        g = folder / "golden_01.jpg"; cv2.imwrite(str(g), np.zeros((8, 8, 3), np.uint8))
        program.Program("board", v, 45.0, [Part("R1", 0, 0, 4, 4)], [g.name], folder=str(folder)).save()
    p = program.resolve("board", tmp_path)
    assert p.version == 2 and p.parts[0].refdes == "R1"
    assert program.resolve(tmp_path / "board" / "v1").version == 1


# ---------------------------------------------------------------- judge rules
PPM = 45.2                                   # ~22 um/px, the phone scale
QUIET = dict(dx=0.0, dy=0.0, sigma=1.0, sigma_deg=0.5, n=30)


def body(long=100.0, short=50.0, extent=100.0, area=5000.0, axis=0.0, ea=1.0, eb=1.0):
    return dict(cx=0, cy=0, long=long, short=short, axis=axis, area=area, extent=extent, end_a=ea, end_b=eb)


def meas(**kw) -> dict:
    """A placed, well-aligned part; override what a test needs."""
    m = dict(in_view=True, glare=0.0, phys_ratio=1.0, gap_ratio=0.0, dx=0.0, dy=0.0, dth=0.0,
             body_g=body(), body_t=body(), skew_silk=0.5, silk_len=400.0, turn_margin=0.5)
    m.update(kw)
    return m


PLACED = judge.missing_logit(meas())               # a good part's golden logit


def part(**kw) -> Part:
    d = dict(refdes="R1", x=100, y=100, w=100, h=50, type="resistor", package="0603")
    d.update(kw)
    return Part(**d)


def state(m, p=None, noise=QUIET, stats=None, single=True) -> str:
    return judge.judge_part(p or part(), m, noise=noise, px_per_mm=PPM, stats=stats, single_golden=single).state


def test_placed_part_is_ok():
    assert state(meas()) == "OK"


def test_missing_from_bare_footprint():
    m = meas(phys_ratio=0.4, gap_ratio=-2.0, body_t=body(extent=40.0))
    assert state(m) == "MISSING"


def test_not_in_view_and_glare_are_not_checked():
    assert state(dict(in_view=False)) == "NOT CHECKED"
    assert state(meas(glare=0.4)) == "NOT CHECKED"


def test_offset_limit_is_150um_when_the_photo_is_quiet():
    lim = 0.150 * PPM                                   # 6.8 px
    assert state(meas(dx=lim * 1.3)) == "OFFSET"
    assert state(meas(dx=lim * 0.8)) == "OK"


def test_offset_limit_follows_board_noise_when_noisier_than_spec():
    noisy = dict(QUIET, sigma=4.5)                      # 3 sigma = 13.5 px ~ 300 um
    assert state(meas(dx=10.0), noise=noisy) == "OK"    # over 150 um, inside the noise
    assert state(meas(dx=19.0), noise=noisy) == "OFFSET"


def test_offset_uses_the_programs_scale_not_the_stations():
    pr = judge.judge_part(part(), meas(dx=20.0), noise=QUIET, px_per_mm=PPM, stats=None, single_golden=True)
    c = pr.calls[0]
    assert c.state == "OFFSET"
    assert f"{20.0 / PPM * 1000:.0f} um" in c.detail          # not 20 px x 12 um/px
    assert config.UM_PER_PX == 12.0


def test_common_shift_is_removed_before_offset():
    shifted = dict(QUIET, dx=15.0, dy=0.0)
    assert state(meas(dx=15.0), noise=shifted) == "OK"


def test_skew_against_silkscreen_and_quarter_turn():
    assert state(meas(skew_silk=14.0)) == "SKEW"
    assert state(meas(skew_silk=8.0)) == "OK"
    turned = meas(body_t=body(long=100, short=50, axis=90.0))
    assert state(turned) == "SKEW"


def test_tombstone_short_body_one_end_bare():
    m = meas(body_t=body(extent=50.0, area=2500.0, ea=0.0, eb=1.0))
    assert state(m) == "TOMBSTONE"


def test_wrong_part_only_against_golden_spread():
    big = meas(body_t=body(area=30000.0, short=90.0))
    assert state(big, single=True) == "OK"             # one golden cannot say what normal size is
    stats = {"log_area": (0.0, 0.05, 3), "log_width": (0.0, 0.05, 3), "missing_logit": (PLACED, 0.5, 3)}
    assert state(big, stats=stats, single=False) == "WRONG PART"
    # 2.4x the area is within what good parts varied against one golden on the
    # fixture (up to 4x), so a golden set with almost no spread cannot call it
    assert state(meas(body_t=body(area=12000.0, short=60.0)), stats=stats, single=False) == "OK"


def test_polarity_band_rules():
    d = part(refdes="D1", type="diode", polarized=True, marked_end="right")
    assert state(meas(stripe_g=2.0, stripe_t=-1.9), d) == "POLARITY REVIEW"    # one golden: never REVERSED
    assert state(meas(stripe_g=2.0, stripe_t=1.2), d) == "OK"
    assert state(meas(stripe_g=1.2, stripe_t=-2.0), d) == "UNINSPECTED"         # golden band unclear
    stats = {"stripe_consistent": (1.0, 0.0, 3), "missing_logit": (PLACED, 0.5, 3)}
    assert state(meas(stripe_g=2.0, stripe_t=-1.9), d, stats=stats, single=False) == "REVERSED"


def test_uninspected_when_golden_spread_is_too_wide():
    stats = {"missing_logit": (-3.0, 3.0, 3)}
    assert state(meas(), stats=stats, single=False) == "UNINSPECTED"


def _board(*states_: str, board_calls=()) -> BoardResult:
    res = BoardResult("t.jpg", "b", 1, "?")
    for i, s in enumerate(states_):
        pr = PartResult(part(refdes=f"P{i}"))
        if s != "OK":
            pr.calls.append(Call("x" if s != "UNINSPECTED" else "missing", s))
        res.parts.append(pr)
    res.board_calls = list(board_calls)
    judge.board_verdict(res)
    return res


def test_board_verdict_table():
    assert _board("OK", "OK").verdict == "PASS"
    assert _board("OK", "MISSING").verdict == "FAIL"
    assert _board("OK", "TOMBSTONE").verdict == "FAIL"
    assert _board("OK", "POLARITY REVIEW").verdict == "REVIEW"
    assert _board("OK", "NOT CHECKED").verdict == "REVIEW"
    assert _board("MISSING", "POLARITY REVIEW").verdict == "FAIL"
    assert _board(*(["OK"] * 19 + ["UNINSPECTED"])).verdict == "PASS"         # 5% uninspected
    assert _board(*(["OK"] * 8 + ["UNINSPECTED"] * 2)).verdict == "REVIEW"    # 20%
    assert _board("OK", board_calls=[Call("board", "NOT CHECKED", "could not align")]).verdict == "REVIEW"


def test_scores_become_probabilities_only_with_enough_labels_of_both_kinds():
    few = [(0.9, 1)] * 20 + [(0.2, 0)] * 9
    assert judge.calibrate(few) is None
    one_sided = [(0.9, 1)] * 29 + [(0.2, 0)] * 5
    assert judge.calibrate(one_sided) is None
    enough = [(0.9, 1)] * 15 + [(0.8, 0)] * 5 + [(0.2, 0)] * 10
    f = judge.calibrate(enough)
    assert f is not None and f(0.95) > f(0.1)
    res = _board("MISSING"); res.parts[0].calls[0].score = 0.9
    judge.apply_calibration(res, {"x": f})
    assert res.parts[0].calls[0].calibrated and res.parts[0].calls[0].score_text().startswith("P=")


# ---------------------------------------------------------- offline, real photos
def test_no_remote_urls_in_photo_mode_code():
    roots = [Path("aoi/photo"), Path("aoi_photo")]
    for f in (p for r in roots if r.exists() for p in r.rglob("*.py")):
        text = f.read_text()
        assert "http://" not in text and "https://" not in text, f


@pytest.fixture(scope="module")
def fixture_runs(tmp_path_factory):
    if not (FIXTURE / "Golden.jpg").exists():
        pytest.skip(f"real-photo fixture {FIXTURE} absent (photos/ is gitignored)")
    from aoi.photo import inspect as I, teach
    root = tmp_path_factory.mktemp("programs")
    real_socket = socket.socket

    def refuse(*a, **k):
        raise AssertionError("photo mode tried to open a network socket")
    socket.socket = refuse                              # the whole pipeline runs offline
    try:
        rep = teach.build_program("pt-pcb-004-03", [FIXTURE / "Golden.jpg"],
                                  program.read_parts_csv(FIXTURE / "parts.csv"),
                                  px_per_mm=45.2, root=root, single_golden=True)
        golden = I.Golden(rep.program)
        runs = {n: I.inspect(rep.program, FIXTURE / f"{n}.jpg", golden)[0] for n in ("test1", "test2", "test3")}
    finally:
        socket.socket = real_socket
    truth = json.loads((FIXTURE / "truth.json").read_text())["truth"]
    return rep, runs, truth


@needs_fixture
def test_fixture_missing_parts(fixture_runs):
    _, runs, truth = fixture_runs
    missing = [t for t in truth if t["defect"] == "missing"]
    caught = [t for t in missing if runs[t["image"]].part(t["refdes"]).state == "MISSING"]
    assert len(missing) == 14
    assert len(caught) >= 12, [(t["image"], t["refdes"]) for t in missing if t not in caught]


@needs_fixture
def test_fixture_reversed_diode_and_offset_transistor(fixture_runs):
    _, runs, _ = fixture_runs
    assert runs["test1"].part("D4").state == "POLARITY REVIEW"     # REVERSED needs >= 3 goldens
    assert runs["test1"].part("T1").state == "OFFSET"


@needs_fixture
def test_fixture_false_calls(fixture_runs):
    _, runs, truth = fixture_runs
    for name, res in runs.items():
        known = {t["refdes"] for t in truth if t["image"] == name}
        false = [pr.refdes for pr in res.parts
                 if pr.refdes not in known and any(c.severity == "FAIL" for c in pr.calls)]
        assert len(false) <= 2, (name, false)
        assert res.verdict == "FAIL"


@needs_fixture
def test_fixture_results_are_stamped_and_note_offset_resolution(fixture_runs):
    rep, runs, _ = fixture_runs
    assert rep.program.single_golden
    for res in runs.values():
        assert res.stamp == program.SINGLE_GOLDEN_STAMP
        # 3 sigma of these photos is ~230-320 um, above the 150 um spec
        assert any("offset resolution worse than spec" in r for r in res.reasons)


@needs_fixture
@pytest.mark.xfail(strict=True, reason="known limit: a one-digit marking change (R19 3303 vs golden 3203) "
                   "is below what marking correlation can see at ~45 px/mm; the marking check is off "
                   "in single-golden mode and OCR is deferred (TODOS E8)")
def test_fixture_wrong_value_marking(fixture_runs):
    _, runs, _ = fixture_runs
    assert runs["test1"].part("R19").state == "MARKING DIFFERS"


def _proposal_ious(propose):
    g = io.load_bgr(FIXTURE / "Golden.jpg")
    props = propose(g)
    parts = program.read_parts_csv(FIXTURE / "parts.csv")
    from aoi.photo import teach
    return [max((teach.iou((p.x, p.y, p.w, p.h), q) for q in props), default=0.0) for p in parts], props


@needs_fixture
def test_colour_fallback_proposals():
    """The colour-only fallback, used when no detector is trained: finds most parts
    loosely but over-proposes on solder (measured: 18 of 37 tight, 75 proposals)."""
    from aoi.photo import teach
    ious, props = _proposal_ious(teach.colour_boxes)
    assert sum(v >= 0.2 for v in ious) >= 33
    assert sum(v >= 0.5 for v in ious) >= 16


needs_detector = pytest.mark.skipif(
    not Path("out/models/part_detector.pt").exists(),
    reason="no trained part detector (python -m aoi_photo train-detector)")


@needs_fixture
@needs_detector
def test_detector_proposals_on_fixture_golden():
    """In-sample: the shipped detector was trained on these photos. Held out
    (trained without any iPhone photo) it found 27 of 37 tight with 1 false box."""
    from aoi.photo import detect, teach
    det = detect.PartDetector()
    ious, props = _proposal_ious(lambda g: teach.detector_boxes(g, det))
    tight = sum(v >= 0.5 for v in ious)
    assert tight >= 33, tight
    assert len(props) - tight <= 3, len(props)
