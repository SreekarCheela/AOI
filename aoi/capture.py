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

RAW_HELP = (
    "{name} is a RAW file, and reading RAW needs rawpy, which is not installed.\n"
    "\n"
    "  pip install rawpy        then re-run with --linear\n"
    "\n"
    "Or shoot JPEG instead. `cli.py shots` defaults to --gamma 2.2, which is the\n"
    "right correction for camera and phone JPEG, and the gate is judged by eye,\n"
    "so RAW buys little here beyond skipping that one step."
)


def have_rawpy() -> bool:
    """Whether RAW frames can be read. Checked up front so a shoot fails on the
    first line rather than after loading the frames that happened to be JPEG."""
    import importlib.util
    return importlib.util.find_spec("rawpy") is not None


def load_image(path: str | Path, gamma: float | None = None) -> np.ndarray:
    """Read one frame as a linear float greyscale image in [0, 1].

    `gamma` is the display gamma to undo; pass 2.2 for JPEG/PNG, None for RAW
    (already linear) or for files you have linearised yourself.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in RAW_EXT:
        # rawpy is deliberately NOT a hard dependency. It builds against libraw,
        # and it is needed only for the physics gate -- a one-off weekend shoot.
        # Making every install carry a native build for that is the wrong trade,
        # and a build failure would break the README's one-command quickstart for
        # someone who only wants to run the pipeline. So it is optional, and this
        # is the message that says so rather than a bare ModuleNotFoundError.
        try:
            import rawpy
        except ImportError as exc:
            raise ImportError(RAW_HELP.format(name=path.name)) from exc
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


# A tripod nudge or a mirror slap moves a frame by a few pixels. A three-figure
# answer is not a nudge, it is the correlation having failed, and applying it
# would take a frame that was very nearly right and destroy it. Refuse instead,
# and say so -- a frame left unaligned is recoverable, one warped by 1600 px is
# not.
MAX_ALIGN_SHIFT_PX = 64.0


def _prep(a: np.ndarray) -> np.ndarray:
    """Reduce a frame to the structure that every frame shares.

    Phase correlation assumes the two images look alike. These do not, and that
    is the entire difficulty: each angled frame lights a different slope band,
    so a fillet blazing in the low frame is dark in the high one. The specular
    highlights are the brightest thing in the picture and therefore dominate any
    gradient, yet they are exactly the part that MOVES between frames rather
    than staying put. Correlating on them locks onto a wrong peak -- measured on
    a 54-component board, 4.7 px of error between low and mid and 10.7 px
    between low and high, silently, with a correlation response no worse than
    the correct answers scored.

    What every frame does share is the diffuse part of the scene: soldermask,
    silkscreen, component bodies. They are lit at every lamp elevation, only
    faintly. Two steps bring them out.

    Normalise each frame by its own high percentile first. The lamp is moved
    between shots, so the frames genuinely differ in overall brightness by three
    to four times, and the log below is not scale invariant -- without this the
    same scene compresses differently in each frame and the error comes back
    (measured 4.6 px). Percentile rather than max because one blown highlight
    would otherwise set the scale for the whole frame.

    Then compress. Log squashes the specular highlights into a narrow band at
    the top while stretching the faint diffuse detail underneath, so the
    gradient that follows is driven by structure that is present in every frame.
    """
    scale = float(np.percentile(a, 99.5))
    b = np.log1p(a / max(scale, 1e-6) * 50.0)
    b = cv2.GaussianBlur(b, (0, 0), 1.2)
    m = cv2.magnitude(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3),
                      cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3))
    return (m - m.mean()) / (m.std() + 1e-8)


def _shift(ref: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    """Translation between two frames, robust to them being lit differently."""
    (dx, dy), _ = cv2.phaseCorrelate(
        np.ascontiguousarray(_prep(ref), np.float64),
        np.ascontiguousarray(_prep(img), np.float64))
    return float(dx), float(dy)


def align(images: list[np.ndarray], enable: bool = True,
          max_shift_px: float = MAX_ALIGN_SHIFT_PX
          ) -> tuple[list[np.ndarray], list[tuple[float, float, bool]]]:
    """Register frames onto the first one.

    Returns the frames and, per frame, the (dx, dy) measured and whether it was
    applied. The caller is expected to show that: misregistration is the failure
    mode of this whole method that does not announce itself -- the false-colour
    image just looks smeared, and the natural conclusion is that the optics do
    not work rather than that the frames do not line up.
    """
    if not enable or len(images) < 2:
        # No shifts, rather than a list of refusals: nothing was rejected here,
        # registration simply was not attempted, and the caller must be able to
        # tell those apart before it warns anybody about a frame.
        return images, []
    ref = images[0]
    out = [ref]
    shifts: list[tuple[float, float, bool]] = [(0.0, 0.0, True)]
    for img in images[1:]:
        dx, dy = _shift(ref, img)
        ok = abs(dx) <= max_shift_px and abs(dy) <= max_shift_px
        shifts.append((dx, dy, ok))
        if not ok:
            out.append(img)
            continue
        M = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], np.float64)
        out.append(cv2.warpAffine(img, M, (ref.shape[1], ref.shape[0]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE))
    return out, shifts


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
                do_align: bool = True, normalise: bool = True,
                shifts_out: list | None = None) -> np.ndarray:
    """Photographs -> (H, W, 4) uint8 stack, ordered as config.CHANNELS.

    Pass a list as `shifts_out` to receive one (channel, dx, dy, applied) tuple
    per frame, so the caller can show what registration actually did.
    """
    order = [c for c in config.CHANNELS if c in shots]
    if not order:
        raise ValueError("no frames matched a channel name")
    imgs = [load_image(shots[c], gamma) for c in order]

    h = min(i.shape[0] for i in imgs)
    w = min(i.shape[1] for i in imgs)
    imgs = [i[:h, :w] for i in imgs]
    imgs, shifts = align(imgs, do_align)
    if shifts_out is not None:
        shifts_out.extend((c, dx, dy, ok) for c, (dx, dy, ok) in zip(order, shifts))

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
