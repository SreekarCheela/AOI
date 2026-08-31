"""Convert raw downloaded datasets into the unified manifest.csv schema
consumed by data/datasets.py (see that file's docstring for the schema).

Each raw dataset has a different label format on disk, and neither has been
inspected first-hand yet in this environment (Kaggle auth is unavailable
here — see download_public_datasets.py). The class-name mappings below are
best-effort, built from each dataset's published documentation, NOT from
directly inspecting the downloaded files. Before trusting a real training
run on this manifest, run this script with `--inspect` first and eyeball
that the discovered class folders match what SOLDEF_AI_CLASS_MAP /
PCB_AOI_CLASS_MAP expect -- fix the maps if not.

Usage:
    python data/prepare_manifest.py --inspect --raw-dir data/raw
    python data/prepare_manifest.py --raw-dir data/raw --out data/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from labels import DEFECT_CLASSES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# SolDef_AI classes per the dataset's published documentation:
# background, exc_solder, good, no_good, poor_solder, spike.
# `no_good` is a generic "defective" label with no specific defect subtype,
# and `background` isn't a board-level class at all -- both are skipped
# rather than guessed at, so we don't inject wrong defect-type labels.
SOLDEF_AI_CLASS_MAP: dict[str, tuple[int, list[str]] | None] = {
    "good": (1, []),
    "exc_solder": (0, ["excess_solder_bridging"]),
    "poor_solder": (0, ["insufficient_solder"]),
    "spike": (0, ["excess_solder_bridging"]),  # closest existing-AOI category to a solder icicle/spike
    "no_good": None,  # generic fail, subtype unknown -- skipped, see module docstring
    "background": None,  # not a board-level label
}

# PCB-AoI (KubeEdge-Ianvs) published defect classes: Bad_slope (insufficient/
# uneven solder paste), Bad_bridge (paste bridging), plus a good/pass class.
# TODO: verify exact folder/label naming once the raw data can be inspected
# (--inspect mode) -- these are the names from the dataset's documentation,
# not confirmed against the actual downloaded directory structure.
PCB_AOI_CLASS_MAP: dict[str, tuple[int, list[str]] | None] = {
    "good": (1, []),
    "pass": (1, []),
    "Bad_slope": (0, ["insufficient_solder"]),
    "Bad_bridge": (0, ["excess_solder_bridging"]),
}


def discover_class_folders(raw_dir: Path) -> dict[str, list[Path]]:
    """Best-effort: assume `raw_dir/**/<class_name>/*.jpg` layout and group
    images by their immediate parent folder name. Datasets that use a flat
    layout with a separate annotation file instead of folder-per-class will
    show up here as zero discovered images -- that's a signal this function
    needs a dataset-specific adapter, not that the data is missing.
    """
    by_class: dict[str, list[Path]] = {}
    for path in raw_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS:
            class_name = path.parent.name
            by_class.setdefault(class_name, []).append(path)
    return by_class


def inspect(raw_dir: Path) -> None:
    for dataset_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        print(f"\n{dataset_dir.name}:")
        by_class = discover_class_folders(dataset_dir)
        if not by_class:
            print("  no folder-per-class images found (flat layout or annotation-file-based -- needs a custom adapter)")
            continue
        for class_name, files in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
            print(f"  {class_name}: {len(files)} images")


def build_rows(raw_dir: Path, class_map: dict, manifest_dir: Path) -> list[dict]:
    rows = []
    by_class = discover_class_folders(raw_dir)
    unmapped = set(by_class) - set(class_map)
    if unmapped:
        print(f"  WARNING: unmapped class folders in {raw_dir.name}, skipped: {sorted(unmapped)}")

    for class_name, files in by_class.items():
        mapping = class_map.get(class_name)
        if mapping is None:
            continue
        pass_fail, defect_names = mapping
        defect_flags = {c: (1 if c in defect_names else 0) for c in DEFECT_CLASSES}
        for f in files:
            rows.append(
                {
                    "image_path": str(f.relative_to(manifest_dir)),
                    "pass_fail": pass_fail,
                    **defect_flags,
                }
            )
    return rows


def assign_splits(rows: list[dict], val_fraction: float, test_fraction: float, seed: int) -> None:
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_val = int(len(rows) * val_fraction)
    n_test = int(len(rows) * test_fraction)
    val_idx = set(indices[:n_val])
    test_idx = set(indices[n_val:n_val + n_test])
    for i, row in enumerate(rows):
        if i in val_idx:
            row["split"] = "val"
        elif i in test_idx:
            row["split"] = "test"
        else:
            row["split"] = "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/manifest.csv")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", action="store_true", help="List discovered class folders and exit")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if args.inspect:
        inspect(raw_dir)
        return

    manifest_path = Path(args.out)
    manifest_dir = manifest_path.parent
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += build_rows(raw_dir / "soldef_ai", SOLDEF_AI_CLASS_MAP, manifest_dir)
    rows += build_rows(raw_dir / "pcb_aoi", PCB_AOI_CLASS_MAP, manifest_dir)

    if not rows:
        raise SystemExit(
            f"No labeled images found under {raw_dir}. Run download_public_datasets.py "
            "first, and --inspect to verify class folder names match the maps in this file."
        )

    assign_splits(rows, args.val_fraction, args.test_fraction, args.seed)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "pass_fail", *DEFECT_CLASSES, "split"])
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    n_test = sum(1 for r in rows if r["split"] == "test")
    print(f"Wrote {manifest_path}: {len(rows)} rows (train={n_train} val={n_val} test={n_test})")


if __name__ == "__main__":
    main()
