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
    # IMREAD_COLOR applies the EXIF orientation tag; IMREAD_UNCHANGED does not.
    a = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if a is None:
        raise PhotoError(f"{path.name}: could not be decoded")
    if a.dtype != np.uint8:
        a = (a.astype(np.float32) * (255.0 / np.iinfo(a.dtype).max)).round().astype(np.uint8)
    return a


def grey(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def photo_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    if not folder.is_dir():
        raise PhotoError(f"{folder}: no such file or folder")
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in PHOTO_EXT)
