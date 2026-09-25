"""Training the part-body detector, offline, from boards you have already taught.

Data, all licence-clean (decision 2026-09-25: own boards + CC0 only):

    programs/<name>/v*/        every saved program: golden 1's taught boxes, and
                               the same boxes carried onto the other goldens by
                               registration
    photos/...                 earlier labelled shoots of PT-PCB-004-03 (iPhone
                               fixture, Pixel, Sony), same mapping
    out/detector_data/         "PCB Component Detection" (Kaggle, CC0 1.0): video
                               frames of through-hole power boards. Different
                               parts, so it only teaches "part vs board vs
                               background", at low weight.

Every board you teach adds data; `python -m aoi_photo train-detector` rebuilds
the model. On a machine without an Apple or NVIDIA GPU it runs on the CPU, slowly.
"""
from __future__ import annotations

import glob
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import detect as D
from .io import PhotoError, grey, load_bgr

TILE = 512
CC0_DIR = Path("out/detector_data")


# ------------------------------------------------------------------ data
@dataclass
class Labelled:
    """One photo and its part-body boxes (x, y, w, h) in the photo's own pixels."""
    path: Path
    boxes: list[tuple[float, float, float, float]]
    source: str
    board_long_px: float | None = None       # None: measure it
    group: str = ""                          # photos of one shoot share a group
    _img: np.ndarray | None = field(default=None, repr=False)
    _k: float = 1.0

    def normalised(self) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
        """The photo at the detector's board scale, and its boxes at that scale."""
        if self._img is None:
            bgr = load_bgr(self.path)
            k = D.normalise_scale(bgr, self.board_long_px) if self.source != "cc0" else min(1.0, 1400 / max(bgr.shape[:2]))
            self._img = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
            self._k = k
        k = self._k
        return self._img, [(x * k, y * k, w * k, h * k) for x, y, w, h in self.boxes]


def _map_boxes(boxes, H_ref_to_img) -> list[tuple[float, float, float, float]]:
    out = []
    for x, y, w, h in boxes:
        c = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(c, H_ref_to_img).reshape(-1, 2)
        (x0, y0), (x1, y1) = q.min(0), q.max(0)
        out.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
    return out


def shoot(ref: Path, boxes: list[tuple[float, float, float, float]], others: list[Path], source: str,
          exclude: dict[str, list[int]] | None = None, min_scale: float = 0.6) -> list[Labelled]:
    """The reference photo plus every other photo of the same board, boxes carried
    over by registration. `exclude[name]` drops box indices (parts absent or moved
    on that board)."""
    from .register import register
    out = [Labelled(ref, list(boxes), source, group=str(ref.parent))]
    rg = grey(load_bgr(ref))
    for o in others:
        img = load_bgr(o)
        reg = register(rg, grey(img))
        if not reg.ok:
            continue
        if abs(np.linalg.det(reg.H[:2, :2])) ** 0.5 > 1 / min_scale:
            continue                                  # far lower resolution than the reference
        mapped = _map_boxes(boxes, np.linalg.inv(reg.H))
        drop = set((exclude or {}).get(o.stem, []))
        H, W = img.shape[:2]
        keep = [b for i, b in enumerate(mapped) if i not in drop
                and b[0] >= 0 and b[1] >= 0 and b[0] + b[2] <= W and b[1] + b[3] <= H]
        out.append(Labelled(o, keep, source, group=str(ref.parent)))
    return out


def _read_boxes(csv_path: Path) -> tuple[Path | None, list[str], list[tuple[float, float, float, float]]]:
    """Both parts.csv dialects: the photo-mode columns and inspect_photo's `# reference:` file."""
    ref, names, boxes = None, [], []
    rows = []
    for ln in csv_path.read_text().splitlines():
        if ln.startswith("#") and "reference:" in ln:
            ref = Path(ln.split(":", 1)[1].strip())
        elif ln.strip() and not ln.startswith("#"):
            rows.append(ln)
    import csv
    for r in csv.DictReader(rows):
        names.append((r.get("refdes") or r.get("name")).strip())
        boxes.append(tuple(float(r[k]) for k in ("x", "y", "w", "h")))
    return ref, names, boxes


def collect(programs_root: str | Path = "programs", photos_root: str | Path = "photos",
            cc0: bool = True) -> list[Labelled]:
    from .io import photo_files
    data: list[Labelled] = []
    # saved programs: goldens of each latest version
    for pj in sorted(glob.glob(str(Path(programs_root) / "*" / "v*" / "program.json"))):
        folder = Path(pj).parent
        try:
            prog = json.loads(Path(pj).read_text())
        except (OSError, ValueError):
            continue
        goldens = [folder / g for g in prog.get("goldens", [])]
        boxes = [(p["x"], p["y"], p["w"], p["h"]) for p in prog.get("parts", [])]
        if goldens and boxes and all(g.exists() for g in goldens):
            data += shoot(goldens[0], boxes, goldens[1:], "program")
    photos_root = Path(photos_root)
    # the 2026-09-25 iPhone fixture, with its known-absent and moved parts dropped
    fx = photos_root / "fixtures" / "2026-09-25"
    if (fx / "Golden.jpg").exists():
        _, names, boxes = _read_boxes(fx / "parts.csv")
        truth = json.loads((fx / "truth.json").read_text())["truth"]
        excl: dict[str, list[int]] = {}
        for t in truth:
            if t["defect"] in ("missing", "offset"):
                excl.setdefault(t["image"], []).append(names.index(t["refdes"]))
        data += shoot(fx / "Golden.jpg", boxes, [fx / f"test{i}.jpg" for i in (1, 2, 3)], "fixture", excl)
    # earlier shoots of the same board: carry the fixture golden's body boxes onto
    # every golden photo that registers onto it, so all labels share one
    # convention (body only). Their own inspect_photo labels boxed some parts
    # with terminations and pads and some without; shots that do not register
    # (the 1080p and 720p ones) are left out rather than mixed in.
    if (fx / "Golden.jpg").exists():
        _, _, fboxes = _read_boxes(fx / "parts.csv")
        for folder in (photos_root / "phone" / "golden", photos_root / "golden",
                       photos_root / "pixel_20260923" / "golden"):
            if folder.is_dir():
                data += [d for d in shoot(fx / "Golden.jpg", fboxes, photo_files(folder), folder.parent.name)[1:]
                         if d.boxes]
    if cc0:
        data += cc0_frames()
    return data


def cc0_frames(root: Path = CC0_DIR, limit: int | None = None) -> list[Labelled]:
    out = []
    for ann in sorted(glob.glob(str(root / "*" / "ann" / "*.json"))):
        a = json.loads(Path(ann).read_text())
        img = Path(ann).parent.parent / "img" / Path(ann).name[:-5]
        boxes = []
        for o in a.get("objects", []):
            pts = np.array(o["points"]["exterior"], np.float32)
            (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
            boxes.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        if img.exists():
            out.append(Labelled(img, boxes, "cc0", group="cc0"))
    return out[:limit] if limit else out


# ------------------------------------------------------------ augmentation
def _photometric(t: np.ndarray, rng: random.Random) -> np.ndarray:
    a = t.astype(np.float32)
    a *= np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32)          # white balance
    a = (a - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-35, 35)              # contrast, exposure
    a = np.clip(a, 0, 255)
    a = 255.0 * (a / 255.0) ** rng.uniform(0.75, 1.35)                              # gamma
    if rng.random() < 0.3:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.5, 1.6))
    if rng.random() < 0.3:
        a += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 8), a.shape)
    if rng.random() < 0.25:                                                          # a glare blob
        h, w = a.shape[:2]
        m = np.zeros((h, w), np.float32)
        cv2.circle(m, (rng.randrange(w), rng.randrange(h)), rng.randrange(20, 90), 1.0, -1)
        a += cv2.GaussianBlur(m, (0, 0), 25)[..., None] * rng.uniform(60, 160)
    a = np.clip(a, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, rng.randrange(40, 90)])
        a = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return a


def sample_tile(item: Labelled, rng: random.Random) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    img, boxes = item.normalised()
    s = rng.uniform(0.7, 1.4) if item.source != "cc0" else rng.uniform(0.5, 1.2)
    ang = rng.choice([0, 90, 180, 270]) + rng.uniform(-8, 8)
    if boxes and rng.random() < 0.75:
        x, y, w, h = rng.choice(boxes)
        cx, cy = x + w / 2 + rng.uniform(-150, 150), y + h / 2 + rng.uniform(-150, 150)
    else:
        cx, cy = rng.uniform(0, img.shape[1]), rng.uniform(0, img.shape[0])
    M = cv2.getRotationMatrix2D((cx, cy), ang, s)
    M[0, 2] += TILE / 2 - cx; M[1, 2] += TILE / 2 - cy
    if rng.random() < 0.5:                                                          # mirror
        M = np.array([[-1, 0, TILE], [0, 1, 0]], np.float64) @ np.vstack([M, [0, 0, 1]])
    tile = cv2.warpAffine(img, M, (TILE, TILE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    out = []
    for x, y, w, h in boxes:
        c = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        # rotate the box's inscribed ellipse, not its corners: an axis-aligned body turned
        # 8 degrees should stay body-sized, not grow by its corners
        q = c @ M[:, :2].T + M[:, 2]
        (x0, y0), (x1, y1) = q.min(0), q.max(0)
        a_ = math.radians(ang % 90 if ang % 90 < 45 else ang % 90 - 90)
        shrink = 1.0 / (abs(math.cos(a_)) + abs(math.sin(a_)))
        bw, bh = (x1 - x0) * shrink, (y1 - y0) * shrink
        bcx, bcy = (x0 + x1) / 2, (y0 + y1) / 2
        if 0 <= bcx < TILE and 0 <= bcy < TILE and min(bw, bh) >= 4:
            out.append((bcx - bw / 2, bcy - bh / 2, bw, bh))
    return _photometric(tile, rng), out


# ------------------------------------------------------------------ targets
def _radius(w: float, h: float, min_overlap: float = 0.7) -> float:
    """CenterNet's gaussian radius: how far a centre can move and keep the IoU."""
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return max(0.0, min(r1, r2, r3))


def targets(boxes, size: int = TILE // D.STRIDE):
    hm = np.zeros((size, size), np.float32)
    wh = np.zeros((2, size, size), np.float32)
    off = np.zeros((2, size, size), np.float32)
    mask = np.zeros((size, size), np.float32)
    for x, y, w, h in boxes:
        cx, cy = (x + w / 2) / D.STRIDE, (y + h / 2) / D.STRIDE
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < size and 0 <= iy < size):
            continue
        r = max(0, int(_radius(w / D.STRIDE, h / D.STRIDE)))
        sig = (2 * r + 1) / 6.0
        ys, xs = np.ogrid[-r:r + 1, -r:r + 1]
        g = np.exp(-(xs * xs + ys * ys) / (2 * sig * sig))
        y0, y1, x0, x1 = max(0, iy - r), min(size, iy + r + 1), max(0, ix - r), min(size, ix + r + 1)
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g[y0 - iy + r:y1 - iy + r, x0 - ix + r:x1 - ix + r])
        wh[:, iy, ix] = (math.log(max(w, 1) / D.STRIDE), math.log(max(h, 1) / D.STRIDE))
        off[:, iy, ix] = (cx - ix, cy - iy)
        mask[iy, ix] = 1.0
    return hm, wh, off, mask


# -------------------------------------------------------------------- train
def _loss(torch, out, hm_t, wh_t, off_t, mask):
    hm, wh, off = out
    p = torch.sigmoid(hm[:, 0]).clamp(1e-4, 1 - 1e-4)
    pos = hm_t.eq(1).float()
    neg_w = (1 - hm_t) ** 4
    focal = -(pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * neg_w * p ** 2 * torch.log(1 - p)).sum()
    n = pos.sum().clamp(min=1)
    m = mask[:, None]
    l_wh = (torch.abs(wh - wh_t) * m).sum() / n
    l_off = (torch.abs(off - off_t) * m).sum() / n
    return focal / n + 0.5 * l_wh + l_off


def train(data: list[Labelled], out: Path = D.DEFAULT_WEIGHTS, iters: int = 4000, batch: int = 12,
          lr: float = 2e-3, cc0_frac: float = 0.15, seed: int = 0, log=print,
          init: str | Path | None = None) -> Path:
    """`init`: start from an existing model (fine-tune) instead of from scratch."""
    torch = D._torch()
    rng = random.Random(seed); torch.manual_seed(seed)
    own = [d for d in data if d.source != "cc0" and d.boxes]
    cc0 = [d for d in data if d.source == "cc0" and d.boxes]
    if not own:
        raise PhotoError("no labelled photos to train on: teach a program first")
    dev = D.device()
    model = D.build_model()
    if init is not None and Path(init).exists():
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["model"])
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    t0 = time.time()
    for it in range(iters):
        xs, hms, whs, offs, ms = [], [], [], [], []
        for _ in range(batch):
            item = rng.choice(cc0) if (cc0 and rng.random() < cc0_frac) else rng.choice(own)
            tile, boxes = sample_tile(item, rng)
            hm, wh, off, mk = targets(boxes)
            xs.append(D.to_tensor(tile)); hms.append(hm); whs.append(wh); offs.append(off); ms.append(mk)
        x = torch.stack(xs).to(dev)
        tt = [torch.from_numpy(np.stack(v)).to(dev) for v in (hms, whs, offs, ms)]
        model.train()
        loss = _loss(torch, model(x), *tt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
        if it % 200 == 0 or it == iters - 1:
            log(f"iter {it:5d}/{iters}  loss {loss.item():.3f}  {time.time() - t0:.0f}s")
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "board_px": D.BOARD_PX, "iters": iters,
                "images": len(own), "boxes": sum(len(d.boxes) for d in own),
                "sources": sorted({d.source for d in data}), "trained": time.strftime("%Y-%m-%d %H:%M")}, out)
    return out


# --------------------------------------------------------------------- eval
def evaluate(detector, items: list[Labelled], score_min: float = D.SCORE_MIN) -> dict:
    """Recall and precision at IoU 0.5, and loose recall at IoU 0.3, over `items`."""
    tp = fp = n_gt = hit30 = 0
    per = []
    for it in items:
        bgr = load_bgr(it.path)
        dets = detector.detect(bgr, score_min)
        used = set()
        for g in it.boxes:
            best, bi = 0.0, -1
            for j, d in enumerate(dets):
                v = D._iou(g, d)
                if v > best:
                    best, bi = v, j
            n_gt += 1
            hit30 += best >= 0.3
            if best >= 0.5 and bi not in used:
                used.add(bi); tp += 1
        f = len(dets) - len(used)
        fp += f
        per.append((it.path.name, len(it.boxes), len(used), f))
    return dict(recall=tp / max(n_gt, 1), recall30=hit30 / max(n_gt, 1),
                precision=tp / max(tp + fp, 1), gt=n_gt, tp=tp, fp=fp, per_image=per)
