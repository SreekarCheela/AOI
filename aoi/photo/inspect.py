"""One test photo, judged against a taught program.

    load -> gate -> align onto golden 1 -> measure every part -> board noise
         -> judge every part -> board-level checks -> board verdict

A photo that cannot be aligned, or was shot at another distance or out of focus,
is REVIEW with the reason and no part is judged: every per-part number would be
measured on the wrong pixels.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import features as F
from . import gate, judge, register
from .io import grey, load_bgr
from .model import BoardResult, Call
from .program import Program


class Golden:
    """Golden 1 of a program, loaded once for any number of test photos."""

    def __init__(self, program: Program):
        self.program = program
        self.bgr = load_bgr(program.golden_paths()[0])
        self.grey = grey(self.bgr)
        self.frame = F.Frame.of(self.bgr)
        self.sharpness = float(np.median([gate.sharpness(load_bgr(p)) for p in program.golden_paths()]))


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
    reg = register.register(golden.grey, grey(bgr))
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
    measures = {p.refdes: F.measure(golden.frame, img, p) for p in program.parts}
    noise = judge.board_noise(measures)
    res.common_shift_px = (noise["dx"], noise["dy"])
    res.noise_px = noise["sigma"]
    res.offset_limit_px = judge.offset_limit_px(program.px_per_mm, noise["sigma"])
    for p in program.parts:
        res.parts.append(judge.judge_part(p, measures[p.refdes], noise=noise, px_per_mm=program.px_per_mm,
                                          stats=program.golden_stats.get(p.refdes),
                                          single_golden=program.single_golden))
    shift = math.hypot(noise["dx"], noise["dy"])
    if shift > res.offset_limit_px:
        um = shift / program.px_per_mm * 1000
        res.board_calls.append(Call("board", "NOT CHECKED",
                                    f"whole board shifted {um:.0f} um: camera tilt or a systematic "
                                    "placement offset", measured_against="all parts together"))
    if program.px_per_mm and res.offset_limit_px > 0.150 * program.px_per_mm + 1e-6:
        res.reasons.append("offset resolution worse than spec on this photo "
                           f"(3 sigma = {res.offset_limit_px / program.px_per_mm * 1000:.0f} um > 150 um)")
    if calibrators:
        judge.apply_calibration(res, calibrators)
    notes = list(res.reasons)
    judge.board_verdict(res)
    res.reasons += [n for n in notes if n not in res.reasons]
    return res, warped
