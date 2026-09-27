"""One test photo, judged against a taught program.

    load -> gate -> align onto golden 1 -> measure every part -> board noise
         -> judge every part -> whole-board difference -> board-level checks
         -> board verdict

The whole-board difference (diff.py) needs no part boxes. Where it finds the
test board differing from golden 1 away from every taught part, the result
carries an UNEXPECTED DIFFERENCE part (DIFF1, DIFF2...): a part the detector
never boxed and that is now missing, or something on the board that should not
be there.

A photo that cannot be aligned, or was shot at another distance or out of focus,
is REVIEW with the reason and no part is judged: every per-part number would be
measured on the wrong pixels.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .. import config
from . import features as F
from . import gate, judge, register
from .io import grey, load_bgr
from .model import DIFF_PREFIX, DIFF_TYPE, BoardResult, Call, Part, PartResult
from .program import Program


class Golden:
    """Golden 1 of a program, loaded once for any number of test photos."""

    def __init__(self, program: Program):
        self.program = program
        self.bgr = load_bgr(program.golden_paths()[0])
        self.grey = grey(self.bgr)
        self.frame = F.Frame.of(self.bgr)
        self.keypoints = register.keypoints(self.grey)
        self.sharpness = float(np.median([gate.sharpness(load_bgr(p)) for p in program.golden_paths()]))
        sp = program.spread_path
        self.spread = np.load(sp) if sp.exists() else None
        self._board_mask: np.ndarray | None = None
        self.diff_cache: dict = {}                 # diff.diff_blobs' golden-only work

    @property
    def board_mask(self) -> np.ndarray:
        if self._board_mask is None:
            self._board_mask = gate.board_mask(self.bgr)
        return self._board_mask


def _grown(p: Part, pad: float) -> tuple[int, int, int, int]:
    m = pad * max(p.w, p.h)
    return int(p.x - m), int(p.y - m), int(p.w + 2 * m), int(p.h + 2 * m)


def _diff_blobs(program: Program, golden: Golden, warped: np.ndarray, cover: np.ndarray | None):
    """diff.diff_blobs for this board, or the ValueError it raised (returned, so
    it can run on the pool beside the parts and be reported afterwards)."""
    from . import diff
    min_side = min((max(p.w, p.h) for p in program.parts), default=0) or None
    try:
        return diff.diff_blobs(golden.bgr, warped, cover, spread=golden.spread,
                               board_mask=golden.board_mask, min_side_px=min_side,
                               cache=golden.diff_cache)
    except ValueError as e:
        return e


def differences(res: BoardResult, program: Program, golden: Golden, warped: np.ndarray,
                cover: np.ndarray | None, blobs=None) -> None:
    """Whole-board difference blobs off every taught part -> UNEXPECTED DIFFERENCE parts."""
    from . import diff
    if blobs is None:
        blobs = _diff_blobs(program, golden, warped, cover)
    if isinstance(blobs, ValueError):
        res.reasons.append(f"whole-board difference not run: {blobs}")
        return
    grown = [_grown(p, config.PHOTO_DIFF_PART_PAD) for p in program.parts]
    off = [b for b in blobs
           if not any(diff.overlap(b, g) >= config.PHOTO_DIFF_ON_PART for g in grown)
           and diff.metal_fraction(golden.bgr, b) < config.PHOTO_DIFF_MAX_GOLDEN_METAL]
    shown = off[:config.PHOTO_DIFF_MAX_CALLS]
    for i, b in enumerate(shown, 1):
        part = Part(f"{DIFF_PREFIX}{i}", b.x, b.y, b.w, b.h, type=DIFF_TYPE)
        res.parts.append(PartResult(part, calls=[Call(
            "difference", "UNEXPECTED DIFFERENCE",
            "differs from golden 1 where no part was taught: an unboxed part missing or moved, "
            "or something on the board that should not be there",
            score=min(1.0, b.score / 2.0), measured_against="golden 1, whole board")]))
    if len(off) > len(shown):
        res.board_calls.append(Call("board", "NOT CHECKED",
                                    f"{len(off)} places differ from golden 1 off the taught parts; "
                                    f"the {len(shown)} strongest are listed. Is this the same board, "
                                    "lit the same way?", measured_against="golden 1, whole board"))


def _board_review(res: BoardResult, detail: str) -> BoardResult:
    res.board_calls.append(Call("board", "NOT CHECKED", detail, measured_against="whole photo"))
    judge.board_verdict(res)
    return res


def inspect(program: Program, image: str | Path, golden: Golden | None = None,
            calibrators: dict | None = None) -> tuple[BoardResult, np.ndarray | None]:
    """BoardResult and the test photo aligned into golden-1 pixels (None if it
    could not be aligned)."""
    golden = golden or Golden(program)
    res = BoardResult(image=str(image), program=program.name, program_version=program.version,
                      verdict="REVIEW", px_per_mm=program.px_per_mm,
                      single_golden=program.single_golden, stamp=program.stamp)
    bgr = load_bgr(image)
    reg = register.register(golden.keypoints, grey(bgr))
    res.registration = dict(ok=reg.ok, inliers=reg.inliers, matches=reg.matches, fit=reg.fit, note=reg.note)
    if not reg.ok:
        return _board_review(res, f"could not align: {reg.note}; does not match program "
                                  f"{program.name} v{program.version}, or the photo is unusable"), None
    problems = gate.test_problems(bgr, golden.sharpness, register.area_scale(reg))
    if problems:
        warped, _ = register.warp_to_ref(bgr, reg.H, golden.bgr.shape)
        return _board_review(res, "; ".join(problems)), warped

    warped, cover = register.warp_to_ref(bgr, reg.H, golden.bgr.shape)
    img = F.Frame.of(warped, cover)
    _ = golden.board_mask                          # built once, before the pool shares it
    blobs = F.pool().submit(_diff_blobs, program, golden, warped, cover) if config.PHOTO_DIFF else None
    measures = F.measure_all(golden.frame, img, program.parts)
    noise = judge.board_noise(measures)
    res.common_shift_px = (noise["dx"], noise["dy"])
    res.noise_px = noise["sigma"]
    res.offset_limit_px = judge.offset_limit_px(program.px_per_mm, noise["sigma"])
    for p in program.parts:
        res.parts.append(judge.judge_part(p, measures[p.refdes], noise=noise, px_per_mm=program.px_per_mm,
                                          stats=program.golden_stats.get(p.refdes),
                                          single_golden=program.single_golden))
    if blobs is not None:
        differences(res, program, golden, warped, cover, blobs.result())
    shift = math.hypot(noise["dx"], noise["dy"])
    if res.offset_limit_px > 0 and shift > res.offset_limit_px:
        res.board_calls.append(Call("board", "NOT CHECKED",
                                    f"whole board shifted {judge.length_text(shift, program.px_per_mm)}: "
                                    "camera tilt or a systematic placement offset",
                                    measured_against="all parts together"))
    # only a program with a scale has a spec in um to fall short of
    if program.px_per_mm > 0 and res.offset_limit_px > 0.150 * program.px_per_mm + 1e-6:
        res.reasons.append("offset resolution worse than spec on this photo "
                           f"(3 sigma = {res.offset_limit_px / program.px_per_mm * 1000:.0f} um > 150 um)")
    if calibrators:
        judge.apply_calibration(res, calibrators)
    notes = list(res.reasons)
    judge.board_verdict(res)
    res.reasons += [n for n in notes if n not in res.reasons]
    return res, warped
