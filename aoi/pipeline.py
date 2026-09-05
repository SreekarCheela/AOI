"""End-to-end inspection: align, measure, score, report.

Order matters and is the whole argument of the design: the board is registered
from its fiducials, every component is then *computed* from CAD rather than
searched for, and only the resulting small aligned patch is shown to a model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config, models, roi
from .board import Board
from .fusion import FusionModel, Verdict, rule_verdict
from .rules import Golden, Measure, measure


@dataclass
class Result:
    verdicts: list[Verdict] = field(default_factory=list)
    measures: list[Measure] = field(default_factory=list)
    aligned: np.ndarray | None = None   # registered stack the verdicts came from
    align_shift_mm: tuple[float, float] = (0.0, 0.0)
    align_rot_deg: float = 0.0
    seconds: float = 0.0

    @property
    def failed(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.failed]

    @property
    def untestable(self) -> list[Verdict]:
        """Components with no inspection program, so never judged either way.

        Kept apart from `failed` on purpose. These are a coverage gap in the
        program, not defects on the board, and mixing them into the review queue
        is how one unprogrammed package buries the real defects.
        """
        return [v for v in self.verdicts if not v.testable]

    @property
    def untaught_packages(self) -> list[str]:
        return sorted({v.package for v in self.untestable})

    def summary(self) -> str:
        f, u = self.failed, self.untestable
        s = (f"{len(self.verdicts)} components, {len(f)} flagged "
             f"({100.0 * len(f) / max(1, len(self.verdicts)):.1f}%) in {self.seconds:.2f}s")
        if u:
            s += (f"   NOT INSPECTED {len(u)} "
                  f"({', '.join(self.untaught_packages)} not taught)")
        return s


# ------------------------------------------------------------------ alignment
def _disc_template(dia_mm: float = 0.5) -> np.ndarray:
    r = max(3, int(round(dia_mm / 2.0 / config.MM_PER_PX)))
    t = np.zeros((4 * r, 4 * r), np.float32)
    cv2.circle(t, (2 * r, 2 * r), r, 1.0, -1, lineType=cv2.LINE_AA)
    return t


def align_board(stack: np.ndarray, board: Board,
                search_mm: float = 2.5) -> tuple[np.ndarray, tuple[float, float], float]:
    """Register the board from its fiducials and warp it back to nominal."""
    if len(board.fiducials) < 2:
        return stack, (0.0, 0.0), 0.0
    dif = config.CHANNELS.index("diffuse")
    img = stack[..., dif].astype(np.float32) / 255.0
    tpl = _disc_template()
    pad = int(round(search_mm / config.MM_PER_PX))
    H, W = img.shape

    src, dst = [], []
    for fx, fy in board.fiducials:
        cx, cy = fx / config.MM_PER_PX, fy / config.MM_PER_PX
        x0, y0 = int(max(0, cx - pad)), int(max(0, cy - pad))
        x1, y1 = int(min(W, cx + pad)), int(min(H, cy + pad))
        win = img[y0:y1, x0:x1]
        if win.shape[0] <= tpl.shape[0] or win.shape[1] <= tpl.shape[1]:
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < 0.35:
            continue
        src.append([x0 + loc[0] + tpl.shape[1] / 2.0, y0 + loc[1] + tpl.shape[0] / 2.0])
        dst.append([cx, cy])
    if len(src) < 2:
        return stack, (0.0, 0.0), 0.0

    M, _ = cv2.estimateAffinePartial2D(np.array(src, np.float32),
                                       np.array(dst, np.float32), method=cv2.LMEDS)
    if M is None:
        return stack, (0.0, 0.0), 0.0
    out = np.stack([cv2.warpAffine(stack[..., i], M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
                    for i in range(stack.shape[2])], axis=-1)
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    # Report the correction where it means something -- at the board centre.
    # The raw affine translation is measured from the pixel origin and is
    # dominated by the rotation lever arm, so it reads as a much larger shift
    # than the board actually moved.
    cx, cy = W / 2.0, H / 2.0
    dx = (M[0, 0] * cx + M[0, 1] * cy + M[0, 2]) - cx
    dy = (M[1, 0] * cx + M[1, 1] * cy + M[1, 2]) - cy
    return out, (float(dx * config.MM_PER_PX), float(dy * config.MM_PER_PX)), rot


# ----------------------------------------------------------------- inspection
class Inspector:
    def __init__(self, golden: Golden, backend: models.Backend | None = None,
                 fusion: FusionModel | None = None, fail_threshold: float = 0.5) -> None:
        self.golden = golden
        self.backend = backend or models.NullBackend()
        self.fusion = fusion
        self.fail_threshold = fail_threshold

    def inspect(self, stack: np.ndarray, board: Board, align: bool = True) -> Result:
        t0 = time.perf_counter()
        res = Result()
        if align:
            stack, res.align_shift_mm, res.align_rot_deg = align_board(stack, board)
        # Hand back the registered stack. The defect map has to be drawn on the
        # same pixels the verdicts were computed from -- draw CAD boxes over the
        # raw capture and every one of them sits off its component by the
        # misregistration the pipeline just corrected.
        res.aligned = stack

        patches = np.stack([roi.extract_patch(stack, c, c.pkg) for c in board.components])
        anom = self.backend.anomaly(patches, [c.package for c in board.components])

        for i, comp in enumerate(board.components):
            m = measure(stack, comp, self.golden)
            m.anomaly = float(anom[i])
            res.measures.append(m)

        if self.fusion is not None:
            X = np.stack([m.vector() for m in res.measures])
            scores = self.fusion.score(X)
            for m, s in zip(res.measures, scores):
                rv = rule_verdict(m, self.golden)
                if not rv.testable:
                    # The tree cannot judge what the rules could not measure --
                    # every feature it would use is NaN. Untestable survives the
                    # fusion layer unchanged.
                    res.verdicts.append(rv)
                    continue
                res.verdicts.append(Verdict(m.refdes, m.package, bool(s >= self.fail_threshold),
                                            float(s), rv.reason if rv.failed else "ok",
                                            rv.detail))
        else:
            res.verdicts = [rule_verdict(m, self.golden) for m in res.measures]

        res.seconds = time.perf_counter() - t0
        return res


# --------------------------------------------------------------------- report
_COL = {"ok": (90, 190, 90), "untaught": (150, 150, 150),
        "missing": (60, 60, 235), "offset": (40, 170, 240),
        "skew": (40, 210, 240), "reversed": (200, 80, 220), "insufficient": (70, 70, 240),
        "bridge": (240, 120, 60), "tombstone": (90, 40, 200)}


def defect_map(stack: np.ndarray, board: Board, result: Result, path: str,
               scale: float = 0.55) -> None:
    """Render a reviewable board image with every flagged component boxed."""
    dif = config.CHANNELS.index("diffuse")
    base = (result.aligned if result.aligned is not None else stack)[..., dif]
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    img = (img * 0.75).astype(np.uint8)

    for comp, v in zip(board.components, result.verdicts):
        pkg = comp.pkg
        fl, fw = pkg.footprint(0.15)
        h = np.array([[-fl / 2, -fw / 2], [fl / 2, -fw / 2], [fl / 2, fw / 2], [-fl / 2, fw / 2]])
        pts = (roi.local_to_board(comp, h) / config.MM_PER_PX).astype(np.int32)
        col = _COL.get(v.reason, (90, 190, 90))
        cv2.polylines(img, [pts], True, col, 2 if v.failed else 1, cv2.LINE_AA)
        if not v.testable:
            # Grey and crossed through: the operator has to be able to see at a
            # glance which parts the machine never actually looked at.
            cv2.line(img, tuple(pts[0]), tuple(pts[2]), col, 1, cv2.LINE_AA)
            cv2.line(img, tuple(pts[1]), tuple(pts[3]), col, 1, cv2.LINE_AA)
        if v.failed:
            p = pts.min(axis=0)
            cv2.putText(img, f"{comp.refdes} {v.reason}", (int(p[0]), int(p[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    bar = f"{result.summary()}   align dx={result.align_shift_mm[0]:+.3f}mm " \
          f"dy={result.align_shift_mm[1]:+.3f}mm rot={result.align_rot_deg:+.2f}deg"
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (28, 28, 28), -1)
    cv2.putText(img, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, out)


def channel_strip(stack: np.ndarray, path: str, scale: float = 0.5) -> None:
    """The four rings side by side -- the raw material every test works from."""
    tiles = []
    for i, name in enumerate(config.CHANNELS):
        t = cv2.cvtColor(stack[..., i], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(t, (0, 0), (t.shape[1], 22), (28, 28, 28), -1)
        cv2.putText(t, name.upper(), (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(t)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(path, cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
