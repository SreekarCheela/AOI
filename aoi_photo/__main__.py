"""Photo AOI entry point.

    python -m aoi_photo                         the desktop app
    python -m aoi_photo teach   --golden DIR --parts parts.csv --name NAME [--px-per-mm X | --board-width-mm W]
    python -m aoi_photo inspect --program NAME DIR_OR_PHOTO
    python -m aoi_photo stats
    python -m aoi_photo train-detector          retrain the part-box detector from taught boards

Everything runs on this machine. Nothing is sent anywhere.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from aoi import config

# Resolved against the repository, so launching from another folder uses the same files.
PROGRAMS = config.PHOTO_PROGRAMS_DIR
RUNS = config.PHOTO_RUNS_DIR
DB = config.PHOTO_DB


def cmd_teach(a: argparse.Namespace) -> int:
    from aoi.photo import io, program, teach
    try:
        goldens = io.photo_files(a.golden)
        parts = program.read_parts_csv(a.parts)
    except io.PhotoError as e:
        print(f"NOT SAVED: {e}", file=sys.stderr)
        return 2
    if not goldens:
        print(f"NOT SAVED: no photos in {a.golden}", file=sys.stderr)
        return 2
    tests = [f.name for f in goldens if f.stem.lower().startswith("test")]
    if tests:
        print(f"NOT SAVED: the golden photos include {', '.join(tests)}, which look like test boards. "
              "Put only golden photos in the --golden folder, or pass golden files one at a time.",
              file=sys.stderr)
        return 2
    try:
        rep = teach.build_program(a.name, goldens, parts, px_per_mm=a.px_per_mm,
                                  board_width_mm=a.board_width_mm, root=a.programs,
                                  single_golden=a.single_golden, accept_selfcheck=a.accept_selfcheck or "")
    except teach.SelfCheckError as e:
        print(f"NOT SAVED: {e}", file=sys.stderr)
        return 2
    except io.PhotoError as e:
        print(f"NOT SAVED: {e}", file=sys.stderr)
        return 2
    p = rep.program
    scale = f"{p.px_per_mm:.1f} px/mm" if p.px_per_mm > 0 else "no scale (offsets in px)"
    print(f"program {p.name} v{p.version}: {len(p.parts)} parts, {len(p.goldens)} golden(s), "
          f"{scale} -> {p.path}")
    if p.stamp:
        print(f"  {p.stamp}")
    if rep.selfcheck_flags:
        print("  self-check accepted as intended: " + ", ".join(r for r, _ in rep.selfcheck_flags))
    if rep.selfcheck_unchecked:
        print(f"  self-check not possible for {len(rep.selfcheck_unchecked)} part(s) "
              "(no outline of their own)")
    return 0


def cmd_inspect(a: argparse.Namespace) -> int:
    from aoi.photo import inspect as I, io, judge, program, record, report
    try:
        prog = program.resolve(a.program, a.programs)
        golden = I.Golden(prog)
        photos = io.photo_files(a.photos)
    except io.PhotoError as e:
        print(f"cannot inspect: {e}", file=sys.stderr)
        return 2
    if not photos:
        print(f"no photos in {a.photos}", file=sys.stderr)
        return 2
    dup = sorted({f.stem for f in photos if sum(g.stem == f.stem for g in photos) > 1})
    if dup:
        print(f"photos share a name and their reports would overwrite each other: {', '.join(dup)}",
              file=sys.stderr)
        return 2
    out = Path(a.out) / prog.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    base, n = out, 2
    while out.exists():
        out, n = base.with_name(f"{base.name}-{n}"), n + 1
    worst = 0
    with record.Record(a.db) as rec:
        pv = rec.program_version(prog.name, prog.version, prog.path, len(prog.goldens),
                                 prog.px_per_mm, prog.single_golden)
        calib = {c: f for c in ("missing", "offset", "skew", "polarity", "tombstone", "wrong_part")
                 if (f := judge.calibrate(rec.labels(c))) is not None}
        for f, load in I.prefetch(photos):
            t0 = time.time()
            res, aligned = I.inspect(prog, f, golden, calib, bgr=load())
            run_id = rec.save_run(res, pv)
            paths = report.write_all(res, golden.bgr, aligned if aligned is not None else golden.bgr * 0, out)
            rec.set_report(run_id, paths["annotated"], paths["html"])
            print(f"{f.name}: {res.verdict} ({time.time() - t0:.1f} s) -> {paths['html']}")
            if res.stamp:
                print(f"  {res.stamp}")
            for r in res.reasons:
                print(f"  {r}")
            worst = max(worst, {"PASS": 0, "REVIEW": 1, "FAIL": 2}[res.verdict])
    return worst


def cmd_stats(a: argparse.Namespace) -> int:
    from aoi.photo import record
    with record.Record(a.db) as rec:
        rows = rec.stats() + [rec.stats_total()]
    print(f"{'group':24}{'parts':>7}{'fail':>6}{'false':>7}{'FCR':>8}{'review':>8}{'ok':>5}{'boards':>8}")
    for r in rows:
        print(f"{r['group']:24}{r['parts_inspected']:7d}{r['fail_calls']:6d}{r['false_calls']:7d}"
              f"{r['fcr']:8.2%}{r['review_calls']:8d}{r['review_resolved_ok']:5d}{r['boards']:8d}")
    return 0


def cmd_train_detector(a: argparse.Namespace) -> int:
    """Rebuild the part detector from every taught program, the labelled photo
    shoots and (unless --no-cc0) the CC0 public set. Runs offline."""
    from aoi.photo import detect, detect_train
    data = detect_train.collect(a.programs, a.photos, cc0=not a.no_cc0)
    own = [d for d in data if d.source != "cc0"]
    print(f"{len(own)} labelled photo(s), {sum(len(d.boxes) for d in own)} part boxes"
          + ("" if a.no_cc0 else f", {sum(d.source == 'cc0' for d in data)} CC0 frames"))
    if not own:
        print("nothing to train on: teach a program first", file=sys.stderr)
        return 2
    out = detect_train.train(data, out=a.weights, iters=a.iters)
    r = detect_train.evaluate(detect.PartDetector(out), own)
    print(f"saved {out}. On its own training photos: recall {r['recall']:.0%} at IoU 0.5, "
          f"{r['fp']} extra boxes over {len(own)} photos (in-sample: a new board will score lower)")
    return 0


def cmd_gui(a: argparse.Namespace) -> int:
    from aoi_photo.app import main as app_main
    return app_main(programs=a.programs, runs=a.out, db=a.db) or 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aoi_photo", description=__doc__.split("\n\n")[0])
    ap.add_argument("--programs", default=str(PROGRAMS), help="where programs live")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--out", default=str(RUNS), help="where reports go")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("teach", help="build a program from golden photos and a parts.csv")
    t.add_argument("--golden", required=True, help="folder of golden photos (golden 1 = first by name)")
    t.add_argument("--parts", required=True, help="refdes,x,y,w,h,type,package,polarized,marked_end")
    t.add_argument("--name", required=True)
    s = t.add_mutually_exclusive_group()
    s.add_argument("--px-per-mm", type=float, help="optional; without a scale offsets are judged and shown in px")
    s.add_argument("--board-width-mm", type=float, help="optional: board's long side, measured once")
    t.add_argument("--accept-selfcheck", help="REFDES,... or 'all': golden parts off their outline are intended")
    t.add_argument("--single-golden", action="store_true",
                   help="allow one golden (tests/CLI only): fixed thresholds, every result stamped")
    t.set_defaults(fn=cmd_teach)
    i = sub.add_parser("inspect", help="inspect test photos (one per board)")
    i.add_argument("--program", required=True, help="program name or folder")
    i.add_argument("photos", help="a photo or a folder of them")
    i.set_defaults(fn=cmd_inspect)
    st = sub.add_parser("stats", help="false calls per package from operator labels")
    st.set_defaults(fn=cmd_stats)
    td = sub.add_parser("train-detector", help="retrain the part-box detector from taught boards (offline)")
    td.add_argument("--photos", default=str(config.PHOTO_SHOOTS_DIR), help="labelled photo shoots")
    td.add_argument("--weights", default=str(config.PHOTO_DETECTOR_WEIGHTS))
    td.add_argument("--iters", type=int, default=4000)
    td.add_argument("--no-cc0", action="store_true", help="leave out the CC0 public PCB set")
    td.set_defaults(fn=cmd_train_detector)
    a = ap.parse_args(argv)
    return (a.fn if a.cmd else cmd_gui)(a)


if __name__ == "__main__":
    sys.exit(main())
