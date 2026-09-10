#!/usr/bin/env python
"""AOI Station One -- prototype command line.

    python cli.py demo                 end-to-end on synthetic boards
    python cli.py dataset               build training crops + feature vectors
    python cli.py train                 train anomaly + classifier + fusion
    python cli.py export                write ONNX for the Pi
    python cli.py bench                 torch vs onnx parity and timing
    python cli.py demo --save           inspect and record to the station log
    python cli.py review                operator review screen for the last board
    python cli.py yield                 per-package false call rate and review time
    python cli.py shots --dir photos/   turn real photographs into a channel stack

Train on the workstation, run `export`, copy out/models to the Pi, then run
with `--backend onnx`. Nothing else changes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from aoi import board as B                     # noqa: E402
from aoi import capture, config, fusion, models, pipeline, roi, rules, store, synth  # noqa: E402

OUT = Path("out")
MODELS = OUT / "models"
DB = OUT / "station.db"
INJECTABLE = ("missing", "offset", "skew", "reversed", "insufficient", "bridge",
              "tombstone", "excess", "wrong_part")

# What a feeder mix-up actually looks like: a part one size step away, or a
# neighbour in the same family. Only the body changes -- the land pattern is
# etched into the board and does not care what was loaded into the machine.
#
# The pairs are chosen to differ enough in body area to be separable. Two that
# are NOT here are the point: a 0603 body on a SOD-123 footprint spans exactly
# the width a standing SOD-123 would, and a SOD-123 on a TANT-A footprint sits
# 16% off in area against a good population that spreads 20%. Both are pinned
# as known limits in tests/test_known_limits.py rather than hidden in the demo.
WRONG_FITS = {
    "0402": "0603", "0603": "0402", "0805": "0603",
    "SOD-123": "TANT-A", "TANT-A": "0805",
    "SOIC-8": "SOT-23-6", "SOT-23-6": "0603", "QFP-32": "SOIC-8",
}


# --------------------------------------------------------------------- helpers
def make_board(seed: int = 1, mixed: bool = False) -> B.Board:
    return B.mixed_board(seed) if mixed else B.random_board(40.0, 30.0, seed=seed)


def golden_stacks(brd: B.Board, n: int, seed0: int = 100) -> list[np.ndarray]:
    """Known-good scans -- the only data the anomaly model ever needs."""
    return [synth.render(brd, opts=synth.RenderOpts(seed=seed0 + i)) for i in range(n)]


def random_defects(brd: B.Board, n: int, rng: random.Random) -> dict[str, synth.Defect]:
    picks = rng.sample(brd.components, min(n, len(brd.components)))
    out: dict[str, synth.Defect] = {}
    for c in picks:
        # Only a polarised part can be reversed, and only a two-terminal part can
        # stand on one land. Injecting a tombstone into an eight-lead SOIC would
        # render a perfectly good part and then count it as a miss.
        kinds = [k for k in INJECTABLE
                 if (k != "reversed" or c.pkg.polarized)
                 and (k != "tombstone" or c.pkg.is_two_terminal)
                 and (k != "wrong_part" or c.package in WRONG_FITS)]
        k = rng.choice(kinds)
        n_targets = c.pkg.n_pads if k != "bridge" else len(c.pkg.adjacent_pairs())
        out[c.refdes] = synth.Defect(
            kind=k, pad=rng.randrange(max(1, n_targets)),
            dx=rng.choice([-1, 1]) * rng.uniform(0.18, 0.30) if k == "offset" else 0.0,
            dtheta=rng.choice([-1, 1]) * rng.uniform(15.0, 25.0) if k == "skew" else 0.0,
            substitute=WRONG_FITS.get(c.package, "") if k == "wrong_part" else "")
    return out


def _teach(brd: B.Board, n: int) -> rules.Golden:
    t = time.perf_counter()
    g = rules.fit_golden(golden_stacks(brd, n), brd)
    print(f"  taught {len(g.packages)} package families from {n} golden boards "
          f"({time.perf_counter() - t:.1f}s)")
    return g


# ------------------------------------------------------------------------ demo
def cmd_demo(a: argparse.Namespace) -> None:
    OUT.mkdir(exist_ok=True)
    rng = random.Random(a.seed)
    brd = make_board(a.board_seed, getattr(a, "mixed", False))
    print(f"board {brd.name}: {brd.width:.0f} x {brd.height:.0f} mm, "
          f"{len(brd.components)} components, {len(brd.fiducials)} fiducials")
    print(f"sampling {config.UM_PER_PX:.0f} um/px -> "
          f"{int(brd.width / config.MM_PER_PX)} x {int(brd.height / config.MM_PER_PX)} px, "
          f"{len(config.CHANNELS)} channels")

    print("\nteach")
    golden = _teach(brd, a.golden)

    defects = random_defects(brd, a.defects, rng)
    print(f"\ninject {len(defects)} defects: "
          + ", ".join(f"{k}:{v.kind}" for k, v in sorted(defects.items())[:8])
          + (" ..." if len(defects) > 8 else ""))

    stack = synth.render(brd, defects=defects, opts=synth.RenderOpts(
        seed=a.seed, board_shift_mm=(0.42, -0.31), board_rot_deg=0.6))
    print("  board deliberately misregistered by (+0.42, -0.31) mm, +0.60 deg")

    backend = models.load_backend(a.backend, MODELS)
    fus = fusion.FusionModel.load(MODELS / "fusion.pkl") if (MODELS / "fusion.pkl").exists() \
        and a.backend != "none" else None
    insp = pipeline.Inspector(golden, backend, fus)
    res = insp.inspect(stack, brd)

    print(f"\nalignment recovered dx={res.align_shift_mm[0]:+.3f} "
          f"dy={res.align_shift_mm[1]:+.3f} mm, rot={res.align_rot_deg:+.3f} deg")
    print(f"inspect: {res.summary()}")

    if res.untestable:
        print(f"\nNOT INSPECTED: {len(res.untestable)} components have no golden "
              f"model ({', '.join(res.untaught_packages)}).")
        print("  These are a coverage gap in the program, not defects. Teach them "
              "before releasing the board.")

    truth = set(defects)
    flagged = {v.refdes for v in res.failed}
    caught, missed, false = truth & flagged, truth - flagged, flagged - truth
    print(f"\ncaught {len(caught)}/{len(truth)}   missed {len(missed)}   "
          f"false calls {len(false)}  ({100.0 * len(false) / max(1, len(brd.components)):.2f}% "
          f"of components)")

    print(f"\n{'refdes':8}{'package':10}{'injected':14}{'reported':14}{'score':>7}  detail")
    for v in sorted(res.failed, key=lambda v: v.refdes):
        inj = defects[v.refdes].kind if v.refdes in defects else "-"
        mark = " " if v.refdes in truth else "!"
        print(f"{mark}{v.refdes:7}{v.package:10}{inj:14}{v.reason:14}{v.score:7.2f}  {v.detail}")
    for r in sorted(missed):
        print(f"?{r:7}{'':10}{defects[r].kind:14}{'MISSED':14}")

    if getattr(a, "save", False):
        with store.Store(DB) as st:
            # One reference patch per package, from the golden scans, so review
            # can show the operator what correct looks like alongside the fail.
            gs = golden_stacks(brd, a.golden)
            refs = {}
            for c in brd.components:
                if c.package in refs:
                    continue
                refs[c.package] = np.median(
                    np.stack([roi.extract_patch(g, c, c.pkg) for g in gs]),
                    axis=0).astype(np.uint8)
            st.save_golden(refs)
            patches = np.stack([roi.extract_patch(res.aligned, c, c.pkg)
                                for c in brd.components])
            bid = st.save_result(brd, res, patches, serial=a.serial)
            n = len(st.review_queue(bid))
        print(f"\nsaved board {bid} to {DB}: {n} components queued for review")
        print(f"  review them with:  python cli.py review")

    pipeline.defect_map(stack, brd, res, str(OUT / "defect_map.png"))
    pipeline.channel_strip(stack, str(OUT / "channels.png"))
    print(f"\nwrote {OUT / 'defect_map.png'} and {OUT / 'channels.png'}")


# --------------------------------------------------------------------- dataset
def cmd_dataset(a: argparse.Namespace) -> None:
    OUT.mkdir(exist_ok=True)
    rng = random.Random(a.seed)
    brd = make_board(a.board_seed)
    golden = _teach(brd, a.golden)

    patches, labels, feats, fy, pkgs = [], [], [], [], []
    print(f"\nrendering {a.boards} boards")
    for i in range(a.boards):
        defects = random_defects(brd, a.defects, rng) if i else {}
        stack = synth.render(brd, defects=defects, opts=synth.RenderOpts(
            seed=1000 + i, board_shift_mm=(rng.uniform(-.4, .4), rng.uniform(-.4, .4)),
            board_rot_deg=rng.uniform(-.7, .7)))
        stack, _, _ = pipeline.align_board(stack, brd)
        for c in brd.components:
            bad = c.refdes in defects
            patches.append(roi.extract_patch(stack, c, c.pkg))
            labels.append(1 if bad else 0)
            pkgs.append(c.package)
            m = rules.measure(stack, c, golden)
            feats.append(m.vector())
            fy.append(1 if bad else 0)
        print(f"  board {i + 1}/{a.boards}  {len(defects)} defects", end="\r")

    # Free labels: a polarised part rotated 180 deg IS a reversed part, exactly
    # labelled, at zero annotation cost (spec section 6).
    extra_p, extra_l, extra_pk = [], [], []
    for pat, lab, pk in zip(patches, labels, pkgs):
        if lab == 0 and B.packages.get(pk).polarized:
            extra_p.append(np.rot90(pat, 2).copy())
            extra_l.append(1)
            extra_pk.append(pk)          # keep the package -- models are per-family
    print(f"\n  + {len(extra_p)} synthesised 'reversed' patches from 180 deg rotation")

    P = np.stack(patches + extra_p).astype(np.uint8)
    L = np.array(labels + extra_l, np.int64)
    X = np.stack(feats).astype(np.float32)
    Y = np.array(fy, np.int64)
    np.savez_compressed(OUT / "dataset.npz", patches=P, labels=L,
                        features=X, feature_labels=Y,
                        packages=np.array(pkgs + extra_pk))
    print(f"  patches {P.shape} ({int(L.sum())} defect / {int((L == 0).sum())} ok)")
    print(f"  features {X.shape}")
    print(f"wrote {OUT / 'dataset.npz'} ({(OUT / 'dataset.npz').stat().st_size / 1e6:.1f} MB)")


# ----------------------------------------------------------------------- train
def cmd_train(a: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    MODELS.mkdir(parents=True, exist_ok=True)
    d = np.load(OUT / "dataset.npz", allow_pickle=True)
    P, L = d["patches"], d["labels"]
    X, Y = d["features"], d["feature_labels"]
    torch.manual_seed(0)

    xt = torch.from_numpy(models.to_tensor(P))
    yt = torch.from_numpy(L)
    n = len(xt)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, va = idx[: int(n * 0.8)], idx[int(n * 0.8):]

    # --- anomaly: per-package subspace, fitted from good patches only --------
    PK = d["packages"]
    tr_set, va_set = set(tr.tolist()), set(va.tolist())
    print(f"anomaly subspace (rank {a.pca_k}), per package family")
    seps = []
    for pkg in sorted(set(PK.tolist())):
        sel = np.where(PK == pkg)[0]
        good = [i for i in sel if i in tr_set and yt[i] == 0]
        if len(good) < 20:
            print(f"  {pkg:10} only {len(good)} good patches -- skipped")
            continue
        mu, V = models.fit_subspace(P[good], a.pca_k)
        torch.save({"mu": mu, "V": V}, MODELS / f"anomaly_{models.safe_name(pkg)}.pt")
        net = models.build_subspace_anomaly(mu, V)
        va_sel = [i for i in sel if i in va_set]
        if not va_sel:
            continue
        with torch.no_grad():
            sv = net(xt[va_sel]).numpy()
        yv_ = yt[va_sel].numpy()
        if (yv_ == 0).any() and (yv_ == 1).any():
            o, f = sv[yv_ == 0].mean(), sv[yv_ == 1].mean()
            seps.append(f / max(o, 1e-9))
            print(f"  {pkg:10} {len(good):4d} good  ->  ok {o:.4f}  defect {f:.4f}"
                  f"  separation x{f / max(o, 1e-9):6.2f}")
    if seps:
        print(f"  {'MEAN':10} separation x{np.mean(seps):.2f}")
    yv = yt[va].numpy()

    # --- classifier -----------------------------------------------------------
    clf = models.build_classifier(2)
    opt = torch.optim.Adam(clf.parameters(), 2e-3)
    w = torch.tensor([1.0, float((yt[tr] == 0).sum()) / max(1, int((yt[tr] == 1).sum()))])
    print(f"classifier: {len(tr)} patches, positive weight {w[1]:.1f}")
    for ep in range(a.epochs):
        clf.train(); tot = 0.0
        for xb, yb in DataLoader(TensorDataset(xt[tr], yt[tr]), batch_size=64, shuffle=True):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(clf(xb), yb, weight=w)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        print(f"  epoch {ep + 1:2d}  loss {tot / len(tr):.4f}", end="\r")
    clf.eval(); torch.save(clf.state_dict(), MODELS / "classifier.pt")
    with torch.no_grad():
        pred = clf(xt[va]).argmax(1).numpy()
    print(f"\n  val accuracy {100.0 * (pred == yv).mean():.1f}%  "
          f"recall {100.0 * (pred[yv == 1] == 1).mean():.1f}%")

    # --- fusion ---------------------------------------------------------------
    # Backfill the anomaly column before fitting. `dataset` builds the feature
    # vectors before any model exists, so that column is zeros on disk -- fit the
    # tree on it as-is and the anomaly score contributes nothing, which is
    # exactly what its importance reported.
    be = models.TorchBackend(MODELS)
    fpk = [str(x) for x in d["packages"][: len(X)]]
    X = X.copy()
    X[:, rules.FEATURE_NAMES.index("anomaly")] = be.anomaly(P[: len(X)], fpk)
    print(f"fusion: anomaly column backfilled for {len(X)} feature rows")

    ntr = int(len(X) * 0.8)
    fm = fusion.FusionModel().fit(X[:ntr], Y[:ntr])
    p = fm.score(X[ntr:]) >= 0.5
    t = Y[ntr:]
    tp, fp = int((p & (t == 1)).sum()), int((p & (t == 0)).sum())
    fn = int((~p & (t == 1)).sum())
    print(f"fusion tree: recall {100.0 * tp / max(1, tp + fn):.1f}%  "
          f"false calls {fp} of {int((t == 0).sum())} good "
          f"({100.0 * fp / max(1, int((t == 0).sum())):.2f}%)")
    print("  top features: " + ", ".join(f"{k}={v:.3f}" for k, v in fm.top_features(5)))
    fm.save(MODELS / "fusion.pkl")
    print(f"wrote per-package anomaly_*.pt, classifier.pt, fusion.pkl to {MODELS}")


# ---------------------------------------------------------------------- export
def cmd_export(a: argparse.Namespace) -> None:
    import torch
    MODELS.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, len(config.CHANNELS), config.PATCH_PX, config.PATCH_PX)
    def _load_classifier():
        net = models.build_classifier(2)
        net.load_state_dict(torch.load(MODELS / "classifier.pt", map_location="cpu",
                                       weights_only=True))
        return net.eval()

    if (MODELS / "classifier.pt").exists():
        out = MODELS / "classifier.onnx"
        torch.onnx.export(_load_classifier(), (dummy,), str(out), input_names=["patch"],
                          output_names=["out"], opset_version=17,
                          dynamic_axes={"patch": {0: "n"}, "out": {0: "n"}})
        print(f"  classifier -> {out.name} ({out.stat().st_size / 1e3:.0f} kB)")
    else:
        print("  skip classifier: run train first")

    pts = sorted(MODELS.glob("anomaly_*.pt"))
    if not pts:
        print("  skip anomaly: run train first")
    for pt in pts:
        sd = torch.load(pt, map_location="cpu", weights_only=True)
        net = models.build_subspace_anomaly(sd["mu"], sd["V"])
        out = MODELS / f"{pt.stem}.onnx"
        torch.onnx.export(net, (dummy,), str(out), input_names=["patch"],
                          output_names=["score"], opset_version=17,
                          dynamic_axes={"patch": {0: "n"}, "score": {0: "n"}})
        print(f"  {pt.stem} -> {out.name} ({out.stat().st_size / 1e3:.0f} kB)")
    print("copy out/models to the Pi and run with --backend onnx")


# ----------------------------------------------------------------------- bench
def cmd_bench(a: argparse.Namespace) -> None:
    d = np.load(OUT / "dataset.npz", allow_pickle=True)
    P = d["patches"][: a.n]
    PK = [str(x) for x in d["packages"][: a.n]]
    tb, ob = models.TorchBackend(MODELS), models.OnnxBackend(MODELS)
    print(f"{a.n} patches, {config.PATCH_PX}x{config.PATCH_PX}x{len(config.CHANNELS)}\n")
    res = {}
    for name, be in (("torch", tb), ("onnx", ob)):
        be.classify(P[:8]); be.anomaly(P[:8], PK[:8])   # warm up
        t = time.perf_counter(); c = be.classify(P); a1 = be.anomaly(P, PK)
        dt = time.perf_counter() - t
        res[name] = (c, a1)
        print(f"  {name:6} {dt * 1000:7.1f} ms total  {dt / len(P) * 1000:6.3f} ms/component"
              f"   -> {len(P) / dt:7.0f} comp/s")
    (ct, at), (co, ao) = res["torch"], res["onnx"]
    print(f"\nparity  classify max|delta| {np.abs(ct - co).max():.2e}"
          f"   anomaly max|delta| {np.abs(at - ao).max():.2e}")
    print(f"agreement on argmax: {100.0 * (ct.argmax(1) == co.argmax(1)).mean():.2f}%")
    print("\n500-component board projection:")
    for name, be in (("torch", tb), ("onnx", ob)):
        t = time.perf_counter(); be.classify(P); be.anomaly(P, PK)
        per = (time.perf_counter() - t) / len(P)
        print(f"  {name:6} {per * 500:.2f} s of model time per board (this Mac)")


# ----------------------------------------------------------------------- shots
def cmd_shots(a: argparse.Namespace) -> None:
    """Turn real photographs into a channel stack and show the banding."""
    OUT.mkdir(exist_ok=True)
    folder = Path(a.dir)
    shots = capture.find_shots(folder)
    if not shots:
        print(f"No frames found in {folder}. Name files so each contains one of:")
        print("  " + ", ".join(config.CHANNELS) + "    e.g. low.jpg  mid.jpg  high.jpg")
        return

    print("frames:")
    for ch in config.CHANNELS:
        if ch in shots:
            print(f"  {ch:8} {shots[ch].name}")
    missing = [c for c in config.ANGLED_CHANNELS if c not in shots]
    if missing:
        print(f"  missing angled frames: {', '.join(missing)} -- results will be partial")

    # Check before doing any work. Loading two JPEGs and then failing on the
    # third frame because it is a DNG wastes the time and buries the reason.
    raw = [p for p in shots.values() if p.suffix.lower() in capture.RAW_EXT]
    if raw and not capture.have_rawpy():
        print("\n" + capture.RAW_HELP.format(name=raw[0].name))
        return

    gamma = None if a.linear else a.gamma
    print(f"\nlinearising with gamma {gamma}" if gamma else "\ntreating input as linear")
    shifts: list = []
    stack = capture.build_stack(shots, gamma=gamma, do_align=not a.no_align,
                                shifts_out=shifts)
    if shifts:
        print("\nregistration (each frame onto the first):")
        for ch, dx, dy, ok in shifts:
            note = "" if ok else "   REFUSED, too large to be a tripod nudge -- frame left as shot"
            print(f"  {ch:8} dx={dx:+7.2f} dy={dy:+7.2f} px{note}")
        if any(not ok for *_, ok in shifts):
            print("  A refused frame is still in the stack, unregistered. Reshoot it,")
            print("  or pass --no-align if the camera genuinely did not move.")
    print(f"\nstack {stack.shape}")
    for i, ch in enumerate(config.CHANNELS):
        v = stack[..., i]
        print(f"  {ch:8} mean {v.mean():6.1f}   p99 {np.percentile(v, 99):6.1f}")

    np.save(OUT / "real_stack.npy", stack)
    cv2.imwrite(str(OUT / "real_false_colour.png"), capture.false_colour(stack))
    pipeline.channel_strip(stack, str(OUT / "real_channels.png"), scale=a.scale)
    print(f"\nwrote {OUT}/real_false_colour.png   <- look at this one")
    print(f"      {OUT}/real_channels.png")
    print(f"      {OUT}/real_stack.npy")
    print("\nOn a good joint you should see a blue -> green -> red sweep from the")
    print("pad out to the component. A starved joint loses the red.")


# ---------------------------------------------------------------------- review
def cmd_review(a: argparse.Namespace) -> None:
    """Open the operator review screen on the most recent board."""
    from aoi import review
    with store.Store(DB) as st:
        board = st.latest_board()
        if board is None:
            print(f"No inspections in {DB}. Run:  python cli.py demo --save")
            return
        bid = a.board or board["id"]
        queue = st.review_queue(bid)
        gaps = st.untestable(bid)
        print(f"board {bid} ({board['name']}): {len(queue)} to review, "
              f"{len(gaps)} not inspected")
        if not queue:
            print("nothing queued -- every fail on this board is already adjudicated")
            return
        review.ReviewApp(st, bid, operator=a.operator).run()
        left = len(st.review_queue(bid))
        t = st.review_timing(bid)
        print(f"adjudicated {len(queue) - left} of {len(queue)}")
        print(f"  {t.note}")


# ----------------------------------------------------------------------- yield
def cmd_yield(a: argparse.Namespace) -> None:
    """The numbers section 8 says to track, per package."""
    with store.Store(DB) as st:
        rows = st.yield_metrics()
        if not rows:
            print(f"No inspections in {DB}. Run:  python cli.py demo --save")
            return
        print(f"{'package':10}{'seen':>7}{'flagged':>9}{'not insp':>10}"
              f"{'judged':>8}{'false':>7}{'real':>6}{'FCR':>8}{'review':>9}")
        for r in rows:
            rev = f"{r['avg_review_s']:.1f}s" if r["avg_review_s"] else "-"
            print(f"{r['package']:10}{r['inspected']:7}{r['flagged']:9}"
                  f"{r['untestable']:10}{r['adjudicated']:8}{r['false_calls']:7}"
                  f"{r['defects']:6}{100 * r['false_call_rate']:7.2f}%{rev:>9}")
        tot = sum(r["inspected"] for r in rows)
        fc = sum(r["false_calls"] for r in rows)
        adj = sum(r["adjudicated"] for r in rows)
        print(f"\n{'ALL':10}{tot:7}{sum(r['flagged'] for r in rows):9}"
              f"{sum(r['untestable'] for r in rows):10}{adj:8}{fc:7}"
              f"{sum(r['defects'] for r in rows):6}{100 * fc / max(tot, 1):7.2f}%")
        if adj == 0:
            print("\nNo adjudications yet, so false call rate is not yet meaningful:")
            print("the machine has an opinion and nobody has checked it.")
            print("Run:  python cli.py review")
        print("\nEscape rate needs a hand-inspected sample and is not derivable here.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo"); d.set_defaults(fn=cmd_demo)
    d.add_argument("--golden", type=int, default=6)
    d.add_argument("--defects", type=int, default=12)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--board-seed", type=int, default=1)
    d.add_argument("--backend", default="none", choices=["none", "torch", "onnx"])
    d.add_argument("--save", action="store_true",
                   help="record the inspection in the local SQLite station log")
    d.add_argument("--serial", default=None, help="board serial from the scanner")
    d.add_argument("--mixed", action="store_true",
                   help="board with leaded parts (SOIC, SOT-23, QFP) as well as passives")

    s = sub.add_parser("dataset"); s.set_defaults(fn=cmd_dataset)
    s.add_argument("--boards", type=int, default=14)
    s.add_argument("--golden", type=int, default=6)
    s.add_argument("--defects", type=int, default=14)
    s.add_argument("--seed", type=int, default=3)
    s.add_argument("--board-seed", type=int, default=1)

    t = sub.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--pca-k", type=int, default=16, dest="pca_k")

    e = sub.add_parser("export"); e.set_defaults(fn=cmd_export)

    b = sub.add_parser("bench"); b.set_defaults(fn=cmd_bench)
    b.add_argument("--n", type=int, default=256)

    rv = sub.add_parser("review"); rv.set_defaults(fn=cmd_review)
    rv.add_argument("--board", type=int, default=None, help="board id (default: latest)")
    rv.add_argument("--operator", default="", help="who is reviewing")

    y = sub.add_parser("yield"); y.set_defaults(fn=cmd_yield)

    sh = sub.add_parser("shots"); sh.set_defaults(fn=cmd_shots)
    sh.add_argument("--dir", default="photos", help="folder of low/mid/high[/diffuse] frames")
    sh.add_argument("--gamma", type=float, default=2.2, help="display gamma to undo (JPEG/PNG)")
    sh.add_argument("--linear", action="store_true", help="input is already linear (RAW)")
    sh.add_argument("--no-align", action="store_true")
    sh.add_argument("--scale", type=float, default=0.5)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
