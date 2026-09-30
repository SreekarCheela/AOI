"""Is this machine ready to run the photo AOI? `python -m aoi_photo doctor`.

Written for the first boot of the production machine (a Raspberry Pi 5 running
Raspberry Pi OS, offline): every check says what is wrong and what to do, and
the exit code is 1 when something would stop the app from working at all.

    OK    ready
    WARN  the app runs, with a feature off or at risk (names, detector, HEIC ...)
    FAIL  the app will not run, or will give wrong answers
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import platform
import shutil
import sys
from pathlib import Path

from .. import config

# import name -> (distribution, what it is for, FAIL or WARN when missing)
PACKAGES = {
    "numpy": ("numpy", "everything", "FAIL"),
    "cv2": ("opencv-python-headless", "everything", "FAIL"),
    "scipy": ("scipy", "part names at teach", "FAIL"),
    "sklearn": ("scikit-learn", "calibration from operator labels", "FAIL"),
    "PIL": ("pillow", "photo orientation", "FAIL"),
    "onnxruntime": ("onnxruntime", "silkscreen names at teach", "WARN"),
    "rapidocr": ("rapidocr", "silkscreen names at teach", "WARN"),
    "torch": ("torch", "the part detector at teach", "WARN"),
}
PIN_FILE = config.REPO_ROOT / "requirements-pi.txt"


def _pins() -> dict[str, str]:
    pins = {}
    if PIN_FILE.exists():
        for line in PIN_FILE.read_text().splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                n, v = line.split("==")
                pins[n.strip().lower()] = v.strip()
    pins.setdefault("rapidocr", "3.9.2")
    return pins


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def pi_model() -> str | None:
    p = Path("/proc/device-tree/model")
    try:
        return p.read_bytes().rstrip(b"\0").decode() if p.exists() else None
    except OSError:
        return None


def checks() -> list[tuple[str, str, str]]:
    """(status, what, detail) for everything the app depends on."""
    out: list[tuple[str, str, str]] = []
    add = lambda s, w, d: out.append((s, w, d))            # noqa: E731

    v = sys.version_info
    add("OK" if v >= (3, 12) else "FAIL", "Python", f"{platform.python_version()}"
        + ("" if v >= (3, 12) else ": numpy and scipy need 3.12+; on a Pi use the Trixie-based "
                                     "Raspberry Pi OS (Python 3.13), not Bookworm (3.11)"))
    arch = platform.machine().lower()
    add("OK" if arch in ("aarch64", "arm64", "x86_64", "amd64") else "FAIL", "64-bit OS", arch
        + ("" if arch in ("aarch64", "arm64", "x86_64", "amd64") else ": install the 64-bit OS"))
    model = pi_model()
    if model:
        add("OK", "machine", model)

    pins = _pins()
    for mod, (dist, why, level) in PACKAGES.items():
        try:
            importlib.import_module(mod)
        except Exception as e:                              # noqa: BLE001 - a broken install is still reported
            add(level, dist, f"not importable ({type(e).__name__}); needed for {why}")
            continue
        have = _version(dist) or "?"
        want = pins.get(dist.lower())
        if want and have != want:
            add("WARN", dist, f"{have}, pinned {want}: results were verified on the pinned version")
        else:
            add("OK", dist, have)
    if _version("opencv-python") or _version("opencv-contrib-python"):
        add("FAIL", "opencv", "the GUI opencv-python is installed beside the headless build and one "
            "overwrites the other: pip uninstall opencv-python, then reinstall opencv-python-headless")

    try:
        import tkinter
        add("OK", "tkinter", f"Tk {tkinter.TkVersion}")
    except Exception:                                       # noqa: BLE001
        add("FAIL", "tkinter", "missing: sudo apt install python3-tk (the app window needs it)")
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        add("WARN", "display", "no DISPLAY/WAYLAND_DISPLAY: the app window needs the desktop session "
            "(the command-line teach/inspect/eval/bench work without one)")

    from . import names
    if names.have_rapid():
        bad = names.verify_models()
        add("OK" if not bad else "WARN", "OCR models",
            f"{names.OCR_MODEL_DIR}" + (f": missing or corrupt {', '.join(bad)}; parts keep P-numbers "
                                        "until installed (models/ocr/README.md)" if bad else ", hashes verified"))
    w = Path(config.PHOTO_DETECTOR_WEIGHTS)
    add("OK" if w.exists() else "WARN", "part detector",
        str(w) if w.exists() else f"{w} not found: teach falls back to colour proposals; copy "
                                  "part_detector.pt from the training machine")
    heif = _version("pillow-heif")
    if heif:
        add("OK", "HEIC", f"pillow-heif {heif}")
    elif sys.platform == "darwin":
        add("OK", "HEIC", "macOS sips")
    else:
        add("WARN", "HEIC", "pillow-heif missing: iPhone .heic photos cannot be opened; "
            "JPEG/PNG and camera frames are fine")

    for label, path in (("programs", config.PHOTO_PROGRAMS_DIR), ("reports", config.PHOTO_RUNS_DIR)):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(p).free / 1e9
        ok = os.access(p, os.W_OK)
        add("OK" if ok and free >= 2 else ("FAIL" if not ok else "WARN"), f"{label} folder",
            f"{p} ({free:.1f} GB free)" + ("" if ok else ": not writable") + ("" if free >= 2 else ": under 2 GB free"))

    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        add("OK" if mem >= 3.5 else "WARN", "memory", f"{mem:.1f} GB"
            + ("" if mem >= 3.5 else ": a 12 MP board and its golden need ~1.5 GB; get the 8 GB Pi"))
    except (ValueError, OSError, AttributeError):
        pass
    add("OK", "cores", f"{os.cpu_count()}")
    from .bench import throttled
    t = throttled()
    if t is not None:
        clean = t.split("=")[-1] in ("0x0", "0")
        add("OK" if clean else "WARN", "throttling", t + ("" if clean else
            ": the Pi has slowed down for heat or low voltage since boot; fit the active cooler and "
            "use the official 27 W supply"))
    return out


def text(rows: list[tuple[str, str, str]]) -> str:
    lines = [f"  {s:5} {w:24} {d}" for s, w, d in rows]
    fails = sum(1 for s, _, _ in rows if s == "FAIL")
    warns = sum(1 for s, _, _ in rows if s == "WARN")
    lines.append("")
    lines.append("ready" if not fails and not warns else
                 f"{fails} problem(s) that stop the app, {warns} warning(s)")
    return "\n".join(lines)
