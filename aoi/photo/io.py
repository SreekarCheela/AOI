"""Loading photographs as 8-bit BGR, upright.

Every photo-mode measurement needs colour (body colour class, soldermask
green) and 8-bit levels (glare is max(B, G, R) >= 250), so this is not
`aoi.capture.load_image`, which returns linear grey for the slope physics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

LDR_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
HEIC_EXT = {".heic", ".heif"}
PHOTO_EXT = LDR_EXT | HEIC_EXT

HEIC_HELP = "HEIC needs pillow-heif (pip install pillow-heif) on this system, or export the photo as JPEG"


class PhotoError(ValueError):
    """A photo that cannot be used, with a reason a person can act on."""


def have_heic() -> bool:
    import importlib.util
    return importlib.util.find_spec("pillow_heif") is not None


def have_sips() -> bool:
    """macOS ships `sips`, which reads HEIC with no extra install and no network."""
    import shutil
    return sys.platform == "darwin" and shutil.which("sips") is not None


def _heic_via_sips(path: Path) -> np.ndarray:
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / (path.stem + ".jpg")
        r = subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "95",
                            str(path), "--out", str(out)], capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            raise PhotoError(f"{path.name}: could not convert HEIC ({r.stderr.strip() or 'sips failed'}); "
                             f"{HEIC_HELP}")
        a = cv2.imread(str(out), cv2.IMREAD_COLOR)     # IMREAD_COLOR applies the EXIF orientation
    if a is None:
        raise PhotoError(f"{path.name}: HEIC converted but could not be decoded")
    return a


def load_bgr(path: str | Path) -> np.ndarray:
    """8-bit BGR, EXIF orientation applied."""
    path = Path(path)
    ext = path.suffix.lower()
    if not path.exists():
        raise PhotoError(f"{path}: no such file")
    if ext in HEIC_EXT:
        if not have_heic():
            if have_sips():
                return _heic_via_sips(path)
            raise PhotoError(f"{path.name}: {HEIC_HELP}")
        import pillow_heif
        from PIL import Image, ImageOps
        pillow_heif.register_heif_opener()
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    if ext not in LDR_EXT:
        raise PhotoError(f"{path.name}: unsupported file type {ext!r} "
                         f"(use {', '.join(sorted(PHOTO_EXT))})")
    # The EXIF orientation is applied here, not left to cv2. For JPEG and PNG,
    # cv2 decodes the stored pixels and the tag, read by PIL, turns them. cv2's
    # own handling is not dependable elsewhere: on a TIFF carrying the tag it
    # fails to decode at all for turned orientations (OpenCV 5.0: "Internal
    # imread issue") and turns the mirrored ones itself. So any other format
    # with a tag, and anything cv2 cannot decode, goes through PIL with
    # ImageOps.exif_transpose.
    orient = _orientation(path)
    a = None
    if orient == 1 or ext in (".jpg", ".jpeg", ".png"):
        a = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_IGNORE_ORIENTATION)
        if a is not None:
            a = _orient(a, orient)
    if a is None:
        a = _via_pil(path)
    if a.dtype != np.uint8:
        a = (a.astype(np.float32) * (255.0 / np.iinfo(a.dtype).max)).round().astype(np.uint8)
    return a


def _orientation(path: Path) -> int:
    """The EXIF orientation tag (1 = upright), 1 when there is none or PIL cannot read it."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            v = int(im.getexif().get(0x0112, 1) or 1)
    except Exception:
        return 1
    return v if 1 <= v <= 8 else 1


def _orient(a: np.ndarray, tag: int) -> np.ndarray:
    """Stored pixels turned upright, as PIL.ImageOps.exif_transpose would."""
    t = {1: lambda x: x,
         2: lambda x: x[:, ::-1],
         3: lambda x: x[::-1, ::-1],
         4: lambda x: x[::-1],
         5: lambda x: x.swapaxes(0, 1),
         6: lambda x: np.rot90(x, -1),
         7: lambda x: x[::-1, ::-1].swapaxes(0, 1),
         8: lambda x: np.rot90(x, 1)}[tag](a)
    return np.ascontiguousarray(t)


def _via_pil(path: Path) -> np.ndarray:
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                a = np.asarray(im).astype(np.float32)
                a = (a * (255.0 / max(float(a.max()), 1.0)) if im.mode == "I" else a / 257.0)
                g = np.clip(a, 0, 255).astype(np.uint8)
                return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise PhotoError(f"{path.name}: could not be decoded ({type(e).__name__})") from None


def grey(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def photo_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    if not folder.is_dir():
        raise PhotoError(f"{folder}: no such file or folder")
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in PHOTO_EXT)
