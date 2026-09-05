"""Load real photographs into the four-channel stack the pipeline expects.

For a first look at whether real solder behaves the way the design assumes, a
DSLR on a tripod and one hand-held lamp are enough. Take one frame per lamp
elevation, drop them in a folder, and this turns them into the same array the
synthetic renderer produces.

Two things it has to get right, both of which quietly ruin the result if
ignored:

* **Linearity.** The whole method compares brightness *between* frames. A JPEG
  has a display gamma of roughly 2.2 baked in, so its pixel values are not
  proportional to the light that arrived. Undo it, or shoot RAW, which is linear
  already.
* **Registration.** Even on a tripod, the mirror slap and a nudged lamp shift
  things by a few pixels. Frames are aligned by phase correlation, which keys on
  structure rather than brightness and so survives the lighting changing
  completely between shots.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config

RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".dng", ".orf", ".pef"}
LDR_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_image(path: str | Path, gamma: float | None = None) -> np.ndarray:
    """Read one frame as a linear float greyscale image in [0, 1].

    `gamma` is the display gamma to undo; pass 2.2 for JPEG/PNG, None for RAW
    (already linear) or for files you have linearised yourself.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in RAW_EXT:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True,
                                  output_bps=16, use_camera_wb=True)
        img = rgb.astype(np.float32) / 65535.0
        img = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        return img                                   # already linear

    if ext not in LDR_EXT:
        raise ValueError(f"unsupported file type {ext!r} ({path.name})")

    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(f"could not read {path}")
    a = a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.0722, 0.7152, 0.2126], np.float32)   # BGR
    if gamma:
        a = np.power(np.clip(a, 0.0, 1.0), gamma)
    return a


def _shift(ref: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    """Translation between two frames, robust to them being lit differently."""
    def prep(a):
        a = cv2.GaussianBlur(a, (0, 0), 1.2)
        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
        m = cv2.magnitude(gx, gy)
        return (m - m.mean()) / (m.std() + 1e-8)
    (dx, dy), _ = cv2.phaseCorrelate(prep(ref).astype(np.float64),
                                     prep(img).astype(np.float64))
    return float(dx), float(dy)


def align(images: list[np.ndarray], enable: bool = True) -> list[np.ndarray]:
    if not enable or len(images) < 2:
        return images
    ref = images[0]
    out = [ref]
    for img in images[1:]:
        dx, dy = _shift(ref, img)
        M = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], np.float64)
        out.append(cv2.warpAffine(img, M, (ref.shape[1], ref.shape[0]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE))
    return out


def find_shots(folder: str | Path) -> dict[str, Path]:
    """Match files to channels by name: low / mid / high / diffuse."""
    folder = Path(folder)
    found: dict[str, Path] = {}
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in RAW_EXT | LDR_EXT:
            continue
        stem = f.stem.lower()
        for ch in config.CHANNELS:
            if ch in stem and ch not in found:
                found[ch] = f
    return found


def build_stack(shots: dict[str, Path], gamma: float | None = 2.2,
                do_align: bool = True, normalise: bool = True) -> np.ndarray:
    """Photographs -> (H, W, 4) uint8 stack, ordered as config.CHANNELS."""
    order = [c for c in config.CHANNELS if c in shots]
    if not order:
        raise ValueError("no frames matched a channel name")
    imgs = [load_image(shots[c], gamma) for c in order]

    h = min(i.shape[0] for i in imgs)
    w = min(i.shape[1] for i in imgs)
    imgs = [i[:h, :w] for i in imgs]
    imgs = align(imgs, do_align)

    if normalise:
        # One common scale across channels -- scaling each on its own would
        # erase the between-channel brightness differences the test measures.
        peak = max(float(np.percentile(i, 99.9)) for i in imgs) or 1.0
        imgs = [i / peak for i in imgs]

    stack = np.zeros((h, w, len(config.CHANNELS)), np.float32)
    for c, img in zip(order, imgs):
        stack[..., config.CHANNELS.index(c)] = img
    if "diffuse" not in order:      # no flat frame shot: approximate one
        idx = [config.CHANNELS.index(c) for c in order]
        stack[..., config.CHANNELS.index("diffuse")] = stack[..., idx].max(axis=2)
    return np.clip(stack * 255.0, 0, 255).astype(np.uint8)


def false_colour(stack: np.ndarray) -> np.ndarray:
    """Low -> red, mid -> green, high -> blue, as one BGR image.

    This is the whole experiment in a single picture. A correctly formed fillet
    sweeps blue (flat pad) through green to red (the steep face against the
    termination). A starved joint never develops that steep face, so its red
    simply is not there.
    """
    lo = stack[..., config.CHANNELS.index("low")]
    mi = stack[..., config.CHANNELS.index("mid")]
    hi = stack[..., config.CHANNELS.index("high")]
    return cv2.merge([hi, mi, lo])          # BGR
