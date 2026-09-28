"""Photo mode fix pass (2026-09-26 core audit): one test per fix, each red without it.

Synthetic where the fix is geometry or bookkeeping; the 2026-09-25 fixture
where it is about real photos (skipped with a reason when photos/ is absent).
"""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi import config
from aoi.photo import features as F
from aoi.photo import gate, io, judge, program, record, report
from aoi.photo.model import BoardResult, Call, Part, PartResult

REPO = config.REPO_ROOT
FIXTURE = REPO / "photos/fixtures/2026-09-25"
needs_fixture = pytest.mark.skipif(not (FIXTURE / "Golden.jpg").exists(),
                                   reason=f"real-photo fixture {FIXTURE} absent (photos/ is gitignored)")
QUIET = dict(dx=0.0, dy=0.0, sigma=1.0, sigma_deg=0.5, n=30)


def _texture(h: int, w: int, seed: int = 0, blur: float = 2.0) -> np.ndarray:
    a = np.random.default_rng(seed).normal(0, 1, (h, w)).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), blur)
    return (128 + 40 * a / a.std()).astype(np.float32)


def _meas(**kw) -> dict:
    body = dict(cx=0, cy=0, long=100.0, short=50.0, axis=0.0, area=5000.0, extent=100.0, end_a=1.0, end_b=1.0)
    m = dict(in_view=True, glare=0.0, phys_ratio=1.0, gap_ratio=0.0, dx=0.0, dy=0.0, dth=0.0,
             body_g=dict(body), body_t=dict(body), skew_silk=0.5, silk_len=400.0, turn_margin=0.5)
    m.update(kw)
    return m


def _part(**kw) -> Part:
    d = dict(refdes="R1", x=100, y=100, w=100, h=50, type="resistor", package="0603")
    d.update(kw)
    return Part(**d)


# ------------------------------------------------ 1. rotation sign
def test_patch_in_golden_frame_undoes_the_measured_turn():
    p = _part(w=120, h=60)
    geo = F.Geo(p)
    g = _texture(geo.H, geo.W)
    c = (geo.bx + p.w / 2.0, geo.by + p.h / 2.0)
    M = cv2.getRotationMatrix2D(c, 10.0, 1.0); M[0, 2] += 5; M[1, 2] -= 3
    t = cv2.warpAffine(g, M, (geo.W, geo.H), borderMode=cv2.BORDER_REPLICATE)
    # place() reports that turn as +10 (golden turned by +10 lies on the test)
    body = np.zeros_like(g); body[geo.body] = 1
    dx, dy, dth = F.place(geo, body, cv2.warpAffine(body, M, (geo.W, geo.H)))
    assert dth == pytest.approx(10.0, abs=0.6) and dx == pytest.approx(5, abs=0.5) and dy == pytest.approx(-3, abs=0.5)
    ref = F.patch_in_golden_frame(g, geo, 0, 0, 0)
    back = F.patch_in_golden_frame(t, geo, 5, -3, 10.0)
    assert F.ncc(back, ref) > 0.95


# ------------------------------------------------ 2. golden spread
def test_missing_cutoff_never_tighter_than_single_golden_nor_unbounded():
    assert judge.missing_cut(-2.2, 0.0) == pytest.approx(judge.MISSING_CUT_SINGLE)
    assert judge.missing_cut(-2.2, 0.01) == pytest.approx(judge.MISSING_CUT_SINGLE)
    assert judge.missing_cut(0.0, 1.5) == judge.MISSING_CUT_MAX
    # a present part with a logit good boards reach (0.4 on fixture test3 R10)
    # against goldens that barely differ: single golden passes it, so must this
    m = _meas(phys_ratio=0.5, gap_ratio=-0.3)
    z = judge.missing_logit(m)
    assert -1.0 < z < judge.MISSING_CUT_SINGLE
    stats = {"missing_logit": (-2.2, 0.02, 2)}
    assert judge.judge_part(_part(), m, noise=QUIET, px_per_mm=45.2, stats=stats,
                            single_golden=False).state == "OK"


# ------------------------------------------------ marking holes (2026-09-28, Test 2 R12/R13)
def test_marking_text_does_not_hollow_a_body_but_a_land_gap_stays_open():
    p = _part(w=100, h=40)
    geo = F.Geo(p)
    GREEN, METAL, BODY = (120, 100, 140), (230, 128, 128), (60, 128, 128)   # Lab
    gl = np.zeros((geo.H, geo.W, 3), np.float32); gl[:] = GREEN
    gl[geo.body] = BODY
    cls = F.Classes(geo, gl, geo.ring)
    marked = gl.copy()
    for k in range(4):                       # four thin white strokes of "010"-style text
        x = geo.bx + 30 + 12 * k
        marked[geo.by + 12:geo.by + 28, x:x + 3] = METAL
    assert cls.body_map(marked)[geo.core].mean() == pytest.approx(1.0, abs=0.02)
    bare = np.zeros_like(gl); bare[:] = GREEN                   # missing part: two lands, green gap
    bare[geo.by:geo.by + p.h, geo.bx:geo.bx + 25] = METAL
    bare[geo.by:geo.by + p.h, geo.bx + 75:geo.bx + 100] = METAL
    assert cls.body_map(bare)[geo.core].mean() < 0.05


# ------------------------------------------------ open axis (2026-09-28, Test 2 R18/R10)
def test_an_axis_the_golden_body_runs_off_is_not_measured():
    p = _part(w=100, h=40)
    geo = F.Geo(p)
    closed = np.zeros((geo.H, geo.W), np.float32); closed[geo.body] = 1
    strip = np.zeros_like(closed); strip[geo.by:geo.by + p.h, :] = 1     # solder read as body past both ends
    assert F.open_axes(geo, closed) == "" and F.open_axes(geo, strip) == "x"
    # a 40 px slide along the strip is no offset; the same shift across it is
    m = _meas(dx=40.0, open_axes="x")
    pr = judge.judge_part(p, m, noise=QUIET, px_per_mm=45.2, stats=None, single_golden=True)
    assert pr.state == "OK" and pr.dx_px == 0.0 and pr.measures["open_axes"] == "x"
    m = _meas(dy=40.0, open_axes="x")
    assert judge.judge_part(p, m, noise=QUIET, px_per_mm=45.2, stats=None, single_golden=True).state == "OFFSET"


@needs_fixture
def test_golden_stats_leave_out_golden_one_against_itself(tmp_path):
    from aoi.photo import teach
    g = io.load_bgr(FIXTURE / "Golden.jpg")
    M = np.float32([[1, 0, 6], [0, 1, -4]])
    g2 = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    parts = [p for p in program.read_parts_csv(FIXTURE / "parts.csv") if p.refdes in ("R6", "C3")]
    st = teach.golden_stats([g, g2], parts)
    for r in ("R6", "C3"):
        assert st[r]["missing_logit"][2] == 1          # one sample: golden 2 only


@needs_fixture
def test_same_shot_twice_is_refused(tmp_path):
    from aoi.photo import teach
    g = io.load_bgr(FIXTURE / "Golden.jpg")
    dup = tmp_path / "golden again.jpg"
    cv2.imwrite(str(dup), g, [cv2.IMWRITE_JPEG_QUALITY, 85])
    parts = [p for p in program.read_parts_csv(FIXTURE / "parts.csv")][:3]
    with pytest.raises(io.PhotoError, match="golden 2 is the same shot as golden 1; shoot a different "
                                            "good board or re-seat it"):
        teach.build_program("dup", [FIXTURE / "Golden.jpg", dup, FIXTURE / "test2.jpg"], parts,
                            px_per_mm=45.2, root=tmp_path / "progs")
    assert not any((tmp_path / "progs").rglob("program.json"))


# ------------------------------------------------ 3. flat body maps
def test_flat_body_maps_give_no_placement_and_no_offset():
    geo = F.Geo(_part())
    z = np.zeros((geo.H, geo.W), np.float32)
    assert all(math.isnan(v) for v in F.place(geo, z, z))
    assert all(math.isnan(v) for v in F.place(geo, z + 1, z + 1))
    m = _meas(dx=float("nan"), dy=float("nan"), dth=float("nan"))
    pr = judge.judge_part(_part(), m, noise=QUIET, px_per_mm=45.2, stats=None, single_golden=True)
    assert not any(c.state == "OFFSET" for c in pr.calls)
    assert [c.state for c in pr.calls if c.check == "offset"] == ["UNINSPECTED"]
    assert (pr.dx_px, pr.dy_px) == (0.0, 0.0)
    noise = judge.board_noise({f"R{i}": _meas(dx=1.0, dy=1.0) for i in range(6)} | {"X": m})
    assert noise["n"] == 6 and noise["dx"] == 1.0


# ------------------------------------------------ 4. big boxes, photo edge
def test_big_box_at_the_photo_edge_is_measured_not_crashed():
    img = np.full((1600, 2000, 3), (60, 140, 60), np.uint8)
    img[20:320, 20:520] = (40, 40, 40)
    ref = F.Frame.of(img)
    m = F.measure(ref, ref, Part("U1", 10, 10, 500, 300, "ic"), is_golden=True)
    assert "in_view" in m
    with pytest.raises(io.PhotoError, match="too large"):
        F.Geo(Part("U9", 100, 100, 1400, 1300))


def test_ring_on_replicated_border_is_not_trusted():
    img = np.full((800, 1200, 3), (60, 140, 60), np.uint8)
    img += np.random.default_rng(1).integers(0, 30, img.shape, dtype=np.uint8)
    img[400:440, 600:680] = (30, 30, 30)
    p = Part("R1", 600, 400, 80, 40)
    geo = F.Geo(p)
    ref = F.Frame.of(img)
    # test photo covers everything: the whole ring is used
    ok, ring, _ = F.coverage(geo, ref, F.Frame.of(img))
    assert ok and ring.sum() == geo.ring.sum()
    # the test photo ends 10 px left of the body: the board beside it is border
    cover = np.full(img.shape[:2], 255, np.uint8); cover[:, :590] = 0
    ok, ring, why = F.coverage(geo, ref, F.Frame.of(img, cover))
    assert not ok and "outside" in why
    m = F.measure(ref, F.Frame.of(img, cover), p)
    assert m["in_view"] is False
    assert judge.judge_part(p, m, noise=QUIET, px_per_mm=45.2, stats=None,
                            single_golden=True).state == "NOT CHECKED"
    # ending 30 px out: in view, but no ring pixel within the local search of the
    # border is used for aligning or levelling
    cover = np.full(img.shape[:2], 255, np.uint8); cover[:, :570] = 0
    ok, ring, _ = F.coverage(geo, ref, F.Frame.of(img, cover))
    assert ok and 0 < ring.sum() < geo.ring.sum()
    xs = np.nonzero(ring.any(axis=0))[0] + geo.x0 - F.PAD
    assert xs.min() >= 570 + F.SEARCH


# ------------------------------------------------ 5. record
@pytest.fixture
def rec(tmp_path):
    with record.Record(tmp_path / "r.db") as r:
        yield r


def _run(rec, pv, calls_by_part, verdict):
    r = BoardResult("t.jpg", "b", 1, verdict)
    for i, calls in enumerate(calls_by_part):
        r.parts.append(PartResult(Part(f"R{i}", 0, 0, 50, 20, "resistor", "0603"), list(calls)))
    return rec.save_run(r, pv)


def test_operator_status_keeps_a_review_with_nothing_to_answer(rec):
    pv = rec.program_version("b", 1, "x", 3, 45.0, False)
    rid = _run(rec, pv, [[Call("missing", "UNINSPECTED")], [Call("missing", "UNINSPECTED")], []], "REVIEW")
    assert rec.operator_status(rid) == "REVIEW"
    rid = _run(rec, pv, [[]], "PASS")
    assert rec.operator_status(rid) == "PASS"


def test_supersede_refuses_a_cycle(rec):
    pv = rec.program_version("b", 1, "x", 3, 45.0, False)
    a, b, c = (_run(rec, pv, [[]], "PASS") for _ in range(3))
    rec.supersede(a, b)
    with pytest.raises(ValueError, match="cycle"):
        rec.supersede(b, a)
    rec.supersede(b, c)
    with pytest.raises(ValueError, match="cycle"):
        rec.supersede(c, a)
    assert rec.operator_status(c) == "PASS"


def test_false_call_rate_counts_parts_judged_for_presence(rec):
    pv = rec.program_version("b", 1, "x", 3, 45.0, False)
    _run(rec, pv, [[Call("polarity", "UNINSPECTED")],       # presence judged: counts
                   [Call("offset", "UNINSPECTED")],         # presence judged: counts
                   [Call("missing", "UNINSPECTED")],        # never judged
                   [Call("view", "NOT CHECKED")],           # never judged
                   []], "REVIEW")
    assert rec.stats_total()["parts_inspected"] == 3


# ------------------------------------------------ 6. reasons count
def test_review_reason_count_matches_the_parts_listed():
    r = BoardResult("t.jpg", "b", 1, "PASS")
    both = PartResult(Part("D1", 0, 0, 50, 20, "diode"), [Call("polarity", "POLARITY REVIEW"), Call("offset", "OFFSET")])
    rev = PartResult(Part("D2", 0, 0, 50, 20, "diode"), [Call("polarity", "POLARITY REVIEW")])
    r.parts += [both, rev]
    judge.board_verdict(r)
    assert "1 part(s) need review: D2 POLARITY REVIEW" in r.reasons
    r2 = BoardResult("t.jpg", "b", 1, "PASS"); r2.parts = [both]
    judge.board_verdict(r2)
    assert not any("need review" in x for x in r2.reasons)


# ------------------------------------------------ 7. versions
def test_version_folders_are_claimed_and_unsaved_ones_skipped(tmp_path):
    f1, v1 = program.new_version_folder(tmp_path, "b")
    f2, v2 = program.new_version_folder(tmp_path, "b")
    assert (v1, v2) == (1, 2) and f1.is_dir() and f2.is_dir()
    cv2.imwrite(str(f1 / "golden_01.jpg"), np.zeros((8, 8, 3), np.uint8))
    program.Program("b", 1, 45.0, [Part("R1", 0, 0, 4, 4)], ["golden_01.jpg"], folder=str(f1)).save()
    assert program.latest(tmp_path, "b").version == 1          # v2 has no program.json
    assert program.resolve("b", tmp_path).version == 1
    assert program.new_version_folder(tmp_path, "b")[1] == 3   # v2 stays claimed


# ------------------------------------------------ 8. paths
def test_default_paths_do_not_depend_on_the_working_directory():
    from aoi.photo import detect
    import aoi_photo.__main__ as M
    for p in (detect.DEFAULT_WEIGHTS, M.PROGRAMS, M.DB, M.RUNS):
        assert Path(p).is_absolute(), p
        assert REPO in Path(p).parents
    assert (REPO / "aoi" / "photo").is_dir()


# ------------------------------------------------ 9. EXIF on TIFF
def test_tiff_with_exif_orientation_loads_upright(tmp_path):
    from PIL import Image
    stored = np.zeros((40, 80, 3), np.uint8); stored[:, :20] = (0, 0, 255)   # red on the left, BGR
    im = Image.fromarray(stored[..., ::-1].copy()); ex = im.getexif(); ex[0x0112] = 6
    f = tmp_path / "turned.tif"; im.save(f, exif=ex)
    bgr = io.load_bgr(f)
    assert bgr.shape == (80, 40, 3)                    # tag 6: turn 90 deg clockwise to view
    assert bgr[:20, :, 2].mean() > 200 and bgr[-20:, :, 2].mean() < 50


# ------------------------------------------------ 10. golden gate
@pytest.mark.skipif(not (REPO / "photos/phone/golden").is_dir() or not (REPO / "photos/golden").is_dir(),
                    reason="phone / sony photos absent (photos/ is gitignored)")
def test_gate_passes_sharp_phone_goldens_and_refuses_tiny_boards():
    for f in sorted((REPO / "photos/phone/golden").glob("*.jpg"))[::5]:
        assert gate.golden_problems(io.load_bgr(f)) == [], f.name
    if (FIXTURE / "Golden.jpg").exists():
        assert gate.golden_problems(io.load_bgr(FIXTURE / "Golden.jpg")) == []
    probs = gate.golden_problems(io.load_bgr(REPO / "photos/golden/sony_g1.png"))
    assert any("move closer or use a higher resolution" in p for p in probs), probs


def test_scale_message_mentions_resolution():
    img = np.full((400, 600, 3), 128, np.uint8)
    probs = gate.test_problems(img, 0.0, 0.5)
    assert any("different distance or resolution" in p for p in probs)


# ------------------------------------------------ 11/12. no scale, auto names
def test_offset_without_scale_is_k_sigma_in_px():
    assert judge.offset_limit_px(0.0, 2.0) == pytest.approx(config.PHOTO_OFFSET_SIGMA_K * 2.0)
    m = _meas(dx=12.0)
    pr = judge.judge_part(_part(), m, noise=dict(QUIET, sigma=2.0), px_per_mm=0.0, stats=None, single_golden=True)
    c = [c for c in pr.calls if c.state == "OFFSET"][0]
    assert "12.0 px" in c.detail and "um" not in c.detail
    # no scale and no noise measured: offset cannot be judged, never called
    pr = judge.judge_part(_part(), m, noise=dict(QUIET, sigma=0.0), px_per_mm=0.0, stats=None, single_golden=True)
    assert [c.state for c in pr.calls] == ["UNINSPECTED"]


def test_stamp_says_no_scale():
    p = program.Program("b", 1, 0.0, [], [], single_golden=True)
    assert program.NO_SCALE_STAMP in p.stamp and program.SINGLE_GOLDEN_STAMP in p.stamp
    assert program.Program("b", 1, 45.0, [], [], single_golden=True).stamp == program.SINGLE_GOLDEN_STAMP


def test_polarity_follows_type_and_polarized():
    m = _meas(turn_margin=-0.6)
    other = judge.judge_part(_part(refdes="P1", type="other"), m, noise=QUIET, px_per_mm=45.2,
                             stats=None, single_golden=True)
    assert not any(c.check == "polarity" for c in other.calls)       # auto-named box: not polarized
    for typ in ("ic", "sot23", "regulator", "other"):
        pr = judge.judge_part(_part(type=typ, polarized=True), m, noise=QUIET, px_per_mm=45.2,
                              stats=None, single_golden=True)
        assert pr.state == "POLARITY REVIEW", typ
    band = _meas(stripe_g=-2.0, stripe_t=2.0)
    pr = judge.judge_part(_part(type="diode", polarized=True), band, noise=QUIET, px_per_mm=45.2,
                          stats=None, single_golden=True)
    assert pr.state == "POLARITY REVIEW"
    assert set(F.STRIPE_TYPES) == {"diode", "led"}         # a bridge's bevel misread; marking check


@needs_fixture
def test_auto_named_parts_teach_and_inspect_without_scale(tmp_path):
    from aoi.photo import inspect as I, teach
    hand = program.read_parts_csv(FIXTURE / "parts.csv")
    pick = [p for p in hand if p.refdes in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "C6", "C3")]
    parts = [Part(f"P{i}", p.x, p.y, p.w, p.h) for i, p in enumerate(pick, 1)]
    rep = teach.build_program("auto", [FIXTURE / "Golden.jpg"], parts, root=tmp_path, single_golden=True)
    prog = rep.program
    assert prog.px_per_mm == 0.0 and program.NO_SCALE_STAMP in prog.stamp
    assert program.Program.load(prog.path).px_per_mm == 0.0
    res, aligned = I.inspect(prog, FIXTURE / "test1.jpg")
    assert program.NO_SCALE_STAMP in res.stamp
    c6 = res.part(f"P{[p.refdes for p in pick].index('C6') + 1}")
    assert c6.state == "MISSING"                                   # C6 is missing on test1
    assert not any("um" in c.detail for pr in res.parts for c in pr.calls)
    assert not any("offset resolution" in r for r in res.reasons)
    paths = report.write_all(res, I.Golden(prog).bgr, aligned, tmp_path / "out")
    assert "none (offsets in px)" in paths["html"].read_text()


# ------------------------------------------------ 13. self-check cost
@needs_fixture
def test_self_check_does_not_run_the_full_measure(monkeypatch):
    from aoi.photo import teach

    def boom(*a, **k):
        raise AssertionError("self_check ran the full measurement")
    g = io.load_bgr(FIXTURE / "Golden.jpg")
    parts = program.read_parts_csv(FIXTURE / "parts.csv")
    ref = F.Frame.of(g)
    for p in parts[::6]:                   # the light read gives the full read's numbers
        full, light = F.measure(ref, ref, p, is_golden=True), F.golden_outline(ref, p)
        for k in ("silk_len", "outline_axis", "body_axis"):
            assert light[k] == full[k], (p.refdes, k)
        assert light["skew_silk"] == full["skew_silk"] or (math.isnan(light["skew_silk"])
                                                            and math.isnan(full["skew_silk"]))
    monkeypatch.setattr(F, "measure", boom)
    t0 = time.time()
    teach.self_check(g, parts)
    assert time.time() - t0 < 3.0


# ------------------------------------------------ 14. CLI errors
def test_cli_unknown_program_is_a_message_not_a_traceback(tmp_path, capsys):
    import aoi_photo.__main__ as M
    photo = tmp_path / "board.jpg"; cv2.imwrite(str(photo), np.zeros((8, 8, 3), np.uint8))
    rc = M.main(["--programs", str(tmp_path / "progs"), "--db", str(tmp_path / "r.db"),
                 "--out", str(tmp_path / "out"), "inspect", "--program", "nope", str(photo)])
    assert rc == 2
    assert "no program named 'nope'" in capsys.readouterr().err


def test_cli_teach_refuses_test_photos_in_the_golden_folder(tmp_path, capsys):
    import aoi_photo.__main__ as M
    d = tmp_path / "shoot"; d.mkdir()
    for n in ("Golden.jpg", "test1.jpg"):
        cv2.imwrite(str(d / n), np.zeros((8, 8, 3), np.uint8))
    csv = tmp_path / "parts.csv"; csv.write_text("refdes,x,y,w,h\nR1,1,1,4,4\n")
    rc = M.main(["--programs", str(tmp_path / "progs"), "teach", "--golden", str(d),
                 "--parts", str(csv), "--name", "b", "--single-golden"])
    assert rc == 2 and "test1.jpg" in capsys.readouterr().err
    assert not (tmp_path / "progs").exists()


# ------------------------------------------------ 15. report box position
def test_flagged_box_is_drawn_where_the_part_was_found():
    img = np.full((400, 600, 3), 90, np.uint8)
    p = Part("R1", 200, 150, 60, 30)
    pr = PartResult(p, [Call("offset", "OFFSET", "x")], dx_px=4.0, dy_px=-2.0,
                    measures={"local_dx": 10.0, "local_dy": 6.0})
    res = BoardResult("t.jpg", "b", 1, "FAIL", parts=[pr], common_shift_px=(8.0, 5.0))
    assert report.found_at(pr, res) == (222.0, 159.0)
    out = report.annotate(img, res)
    col = np.array(report.COLOURS["FAIL"])
    assert np.abs(out[159 + 15, 222].astype(int) - col).sum() < 40      # left edge, at the found spot
    assert np.abs(out[165, 200].astype(int) - col).sum() > 100          # not at the taught spot


# ------------------------------------------------ 16. dE search
def _dE_exhaustive(gl, tl, geo):
    p = geo.p
    g = cv2.GaussianBlur(gl, (0, 0), 1.5); t = cv2.GaussianBlur(tl, (0, 0), 1.5)
    s = max(2, int(0.08 * max(p.w, p.h)))
    return min(F._dE_at(g, t, geo, ox, oy) for oy in range(-s, s + 1) for ox in range(-s, s + 1))


@pytest.mark.parametrize("seed", range(4))
def test_coarse_to_fine_colour_change_matches_exhaustive(seed):
    p = _part(w=150, h=70)
    geo = F.Geo(p)
    base = np.dstack([_texture(geo.H, geo.W, seed * 3 + k, 3.0) for k in range(3)])
    M = np.float32([[1, 0, 3 + seed], [0, 1, -2 - seed]])
    moved = cv2.warpAffine(base, M, (geo.W, geo.H), borderMode=cv2.BORDER_REPLICATE)
    moved += np.random.default_rng(seed).normal(0, 4, moved.shape).astype(np.float32)
    assert F._dE_med(base, moved, geo) == pytest.approx(_dE_exhaustive(base, moved, geo), abs=1e-4)
