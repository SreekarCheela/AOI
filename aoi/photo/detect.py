"""Part-body detector for teach-time box proposals.

Why a learned model: colour thresholds cannot tell a part from the solder
around it. On the 2026-09-25 photos, tinned lands, leads, fillets and
through-hole pads measured a* 131-134, b* 140-144, the same colour as the
ceramic capacitor bodies (a* 134, b* 143). That one fact produced about 30 of 39
false proposals. What separates a body from solder is shape, texture and
context, which is what a small CNN learns.

The model is class-agnostic ("a part body is here"), CenterNet-style: a
heatmap peak at each body centre plus the body's width and height, predicted at
1/4 resolution. It runs on the CPU (or Apple GPU) with torch, which is already
pinned in requirements.txt; nothing is downloaded at run time.

Photos are normalised to a fixed board size (BOARD_PX along the board's long
side) before detection, so a phone photo and a DSLR photo of the same board look
the same size to the model. Training data and procedure: detect_train.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .. import config

BOARD_PX = 2048                 # board long side after normalisation
STRIDE = 4
DEFAULT_WEIGHTS = Path(os.environ.get("AOI_PART_DETECTOR") or config.PHOTO_DETECTOR_WEIGHTS)
SCORE_MIN = 0.30
NMS_IOU = 0.4
MEAN = np.array([0.45, 0.45, 0.45], np.float32)
STD = np.array([0.25, 0.25, 0.25], np.float32)


def have_torch() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def _torch():
    import torch
    return torch


def build_model():
    """~2 M parameters: residual encoder to 1/32, top-down fusion back to 1/4."""
    torch = _torch()
    nn = torch.nn

    def cbr(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class Res(nn.Module):
        def __init__(self, i, o, s):
            super().__init__()
            self.a = cbr(i, o, s)
            self.b = nn.Sequential(nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
            self.skip = (nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))
                         if (i != o or s != 1) else nn.Identity())
            self.c = nn.Sequential(cbr(o, o), nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))

        def forward(self, x):
            y = torch.relu(self.b(self.a(x)) + self.skip(x))
            return torch.relu(self.c(y) + y)

    class Net(nn.Module):
        def __init__(self, w=96):
            super().__init__()
            self.stem = nn.Sequential(cbr(3, 32, 2), cbr(32, 32))        # /2
            self.s1, self.s2 = Res(32, 64, 2), Res(64, 128, 2)          # /4, /8
            self.s3, self.s4 = Res(128, 192, 2), Res(192, 256, 2)       # /16, /32
            self.l1, self.l2 = nn.Conv2d(64, w, 1), nn.Conv2d(128, w, 1)
            self.l3, self.l4 = nn.Conv2d(192, w, 1), nn.Conv2d(256, w, 1)
            self.fuse = nn.Sequential(cbr(w, w), cbr(w, w))

            def head(o, bias=0.0):
                h = nn.Sequential(cbr(w, w), nn.Conv2d(w, o, 1))
                nn.init.constant_(h[-1].bias, bias)
                return h
            self.hm = head(1, -2.19)                                     # prior p ~ 0.1
            self.wh = head(2)
            self.off = head(2)

        def forward(self, x):
            up = torch.nn.functional.interpolate
            c1 = self.s1(self.stem(x)); c2 = self.s2(c1); c3 = self.s3(c2); c4 = self.s4(c3)
            p = self.l4(c4)
            p = self.l3(c3) + up(p, size=c3.shape[-2:], mode="nearest")
            p = self.l2(c2) + up(p, size=c2.shape[-2:], mode="nearest")
            p = self.l1(c1) + up(p, size=c1.shape[-2:], mode="nearest")
            f = self.fuse(p)
            return self.hm(f), self.wh(f), self.off(f)

    return Net()


def device():
    torch = _torch()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(bgr: np.ndarray):
    torch = _torch()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


def normalise_scale(bgr: np.ndarray, board_long_px: float | None = None) -> float:
    """Factor that brings the board's long side to BOARD_PX."""
    if board_long_px is None:
        from .gate import board_width_px
        board_long_px = board_width_px(bgr)
    if board_long_px <= 0:                       # no board found: assume it fills the frame
        board_long_px = 0.95 * max(bgr.shape[:2])
    return BOARD_PX / float(board_long_px)


def decode(hm, wh, off, score_min: float = SCORE_MIN, top_k: int = 400) -> list[tuple[float, float, float, float, float]]:
    """(x, y, w, h, score) in network-input pixels from one image's head outputs."""
    torch = _torch()
    h = torch.sigmoid(hm[0, 0])
    peaks = (h == torch.nn.functional.max_pool2d(h[None, None], 3, 1, 1)[0, 0]) & (h >= score_min)
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    scores = h[ys, xs]
    if len(scores) > top_k:
        keep = torch.topk(scores, top_k).indices
        ys, xs, scores = ys[keep], xs[keep], scores[keep]
    out = []
    for y, x, s in zip(ys.tolist(), xs.tolist(), scores.tolist()):
        w, hh = (float(np.exp(v)) * STRIDE for v in wh[0, :, y, x].tolist())
        ox, oy = off[0, :, y, x].tolist()
        cx, cy = (x + ox) * STRIDE, (y + oy) * STRIDE
        out.append((cx - w / 2, cy - hh / 2, w, hh, s))
    return nms(out, NMS_IOU)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter > 0 else 0.0


def nms(boxes, iou: float):
    kept = []
    for b in sorted(boxes, key=lambda t: -t[4]):
        if all(_iou(b, k) < iou for k in kept):
            kept.append(b)
    return kept


class PartDetector:
    """Loaded once, then `detect(bgr)` -> [(x, y, w, h, score)] in the photo's pixels."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS):
        torch = _torch()
        self.weights = Path(weights)
        ck = torch.load(self.weights, map_location="cpu", weights_only=False)
        self.model = build_model()
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.meta = {k: v for k, v in ck.items() if k != "model"}
        self.dev = device()
        self.model.to(self.dev)

    @classmethod
    def available(cls, weights: str | Path = DEFAULT_WEIGHTS) -> bool:
        return have_torch() and Path(weights).exists()

    def detect(self, bgr: np.ndarray, score_min: float = SCORE_MIN,
               board_long_px: float | None = None) -> list[tuple[float, float, float, float, float]]:
        torch = _torch()
        k = normalise_scale(bgr, board_long_px)
        img = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        H, W = img.shape[:2]
        ph, pw = (-H) % 32, (-W) % 32
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        with torch.no_grad():
            hm, wh, off = self.model(to_tensor(img)[None].to(self.dev))
        dets = decode(hm.cpu(), wh.cpu(), off.cpu(), score_min)
        return [(x / k, y / k, w / k, h / k, s) for x, y, w, h, s in dets
                if x + w / 2 < W and y + h / 2 < H]
