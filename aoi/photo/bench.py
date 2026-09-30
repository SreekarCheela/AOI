"""Where a board's time goes on this machine, stage by stage.

    python -m aoi_photo bench --program NAME PHOTOS [--repeat 3] [--json out.json]

Runs the real inspection (inspect.inspect, the same code the app runs) on each
photo, `repeat` times after one warm-up board, and reports the median seconds of
every stage, then load and report writing timed separately. Writes nothing to
the operator record; the reports go to a temporary folder and are deleted.

Made to be run on the production machine (a Raspberry Pi 5 is the likely one)
so its numbers replace estimates scaled from a laptop.

The part measurement and the whole-board diff run at the same time, so
"measure parts" is their overlap and "wait for diff" is what the diff adds after
it; "diff (own thread)" is its full run time for reference, not part of the total.
"""
from __future__ import annotations

import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ORDER = ("register", "gate", "warp", "frame", "measure parts", "judge", "wait for diff", "finish")


def machine() -> dict:
    """What the numbers were measured on."""
    import cv2
    import numpy as np

    from .. import config
    from . import features
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=5).stdout.strip() or cpu
        elif Path("/proc/cpuinfo").exists():
            info = Path("/proc/cpuinfo").read_text()
            for key in ("Model", "model name"):             # Raspberry Pi, then x86
                line = next((ln for ln in info.splitlines() if ln.startswith(key)), None)
                if line:
                    cpu = line.split(":", 1)[1].strip()
                    break
    except (OSError, subprocess.SubprocessError):
        pass
    return dict(cpu=cpu, arch=platform.machine(), system=platform.platform(), cores=os.cpu_count(),
                python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                opencv_threads=cv2.getNumThreads(), part_workers=features.WORKERS,
                registration=config.PHOTO_REGISTRATION)


def throttled() -> str | None:
    """Raspberry Pi firmware's throttle flags (non-zero: it slowed down for heat
    or power during the run, and the numbers are not its best)."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run(program, photos: list[Path], repeat: int = 3, progress=None) -> dict:
    from . import inspect as I
    from . import report
    from .io import load_bgr
    golden = I.Golden(program)
    I.inspect(program, photos[0], golden)                  # warm-up: pools, golden-side caches
    stages: dict[str, list[float]] = {}
    with tempfile.TemporaryDirectory() as out:
        for r in range(repeat):
            for f in photos:
                t0 = time.perf_counter()
                bgr = load_bgr(f)
                load = time.perf_counter() - t0
                t: dict[str, float] = {}
                t0 = time.perf_counter()
                res, aligned = I.inspect(program, f, golden, bgr=bgr, timings=t)
                total = time.perf_counter() - t0
                t0 = time.perf_counter()
                if aligned is not None:
                    report.write_all(res, golden.bgr, aligned, out)
                rep = time.perf_counter() - t0
                t.pop("load", None)
                t.update({"inspect total": total, "load photo": load, "write report": rep})
                for k, v in t.items():
                    stages.setdefault(k, []).append(v)
                if progress:
                    progress(r, f, res, total)
    med = {k: statistics.median(v) for k, v in stages.items()}
    return dict(machine=machine(), throttled=throttled(), program=f"{program.name} v{program.version}",
                parts=len(program.parts), photos=[str(p) for p in photos], repeat=repeat,
                median_s=med, runs=len(stages.get("inspect total", [])))


def text(b: dict) -> str:
    m, med = b["machine"], b["median_s"]
    lines = [f"{m['cpu']} ({m['arch']}, {m['cores']} cores), {m['system']}",
             f"Python {m['python']}, numpy {m['numpy']}, OpenCV {m['opencv']} ({m['opencv_threads']} threads), "
             f"{m['part_workers']} part workers, registration: {m['registration']}",
             f"program {b['program']}, {b['parts']} parts, {len(b['photos'])} photo(s) x {b['repeat']} "
             f"= {b['runs']} runs, medians:", ""]
    for k in ORDER:
        if k in med:
            lines.append(f"  {k:22}{med[k] * 1000:8.0f} ms")
    lines.append(f"  {'inspect total':22}{med['inspect total'] * 1000:8.0f} ms")
    if "diff (own thread)" in med:
        lines.append(f"  {'(diff on its thread)':22}{med['diff (own thread)'] * 1000:8.0f} ms, overlapped")
    lines += ["", f"  {'load photo':22}{med['load photo'] * 1000:8.0f} ms   (the app overlaps it with inspection)",
              f"  {'write report':22}{med['write report'] * 1000:8.0f} ms   (the app overlaps it with the next board)"]
    batch = max(med["inspect total"], med["load photo"], med["write report"])
    lines.append(f"\n  per board in a batch: ~{batch * 1000:.0f} ms (the slowest of the three overlapped stages)")
    if b.get("throttled") and b["throttled"].split("=")[-1] not in ("0x0", "0"):
        lines.append(f"  WARNING: the Pi throttled during the run ({b['throttled']}): cooling or power supply")
    return "\n".join(lines)
