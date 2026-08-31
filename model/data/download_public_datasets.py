"""Download the public pretraining datasets from Kaggle.

Datasets (identifiers confirmed via Kaggle search, see ARCHITECTURE.md context):
  - SolDef_AI: kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection
    1,150 SMT solder joint images, 6 classes (background, exc_solder, good,
    no_good, poor_solder, spike), 3 viewpoints per joint.
  - PCB-AoI:   kaggle.com/datasets/kubeedgeianvs/pcb-aoi
    230+ boards, 1,200+ images, solder-paste-inspection defects (bridging,
    slope/insufficient paste, etc.), from the KubeEdge-Ianvs benchmark.

Requires Kaggle API credentials (this environment has none configured):
  Option A: `kaggle auth login` (OAuth, interactive)
  Option B: generate a token at https://www.kaggle.com/settings/api and set
            KAGGLE_API_TOKEN, or save it to ~/.kaggle/access_token
  Option C (older kaggle.json flow): ~/.kaggle/kaggle.json with
            {"username": "...", "key": "..."}

Usage:
    python data/download_public_datasets.py --out data/raw
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DATASETS = {
    "soldef_ai": "mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
    "pcb_aoi": "kubeedgeianvs/pcb-aoi",
}


def has_kaggle_credentials() -> bool:
    import os

    if os.environ.get("KAGGLE_API_TOKEN") or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        return True
    home = Path.home()
    return (home / ".kaggle" / "access_token").exists() or (
        home / ".kaggle" / "kaggle.json"
    ).exists()


def download_dataset(slug: str, dest_dir: Path) -> Path:
    import kagglehub

    if not has_kaggle_credentials():
        # kagglehub/kaggle fall back to an interactive OAuth device-flow
        # prompt when no credentials are found, which hangs forever in a
        # non-interactive environment. Fail fast instead.
        raise RuntimeError(
            "No Kaggle credentials found (checked KAGGLE_API_TOKEN, "
            "KAGGLE_USERNAME/KAGGLE_KEY, ~/.kaggle/access_token, "
            "~/.kaggle/kaggle.json). See this file's module docstring for "
            "setup options."
        )

    cache_path = Path(kagglehub.dataset_download(slug))
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in cache_path.iterdir():
        target = dest_dir / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw", help="Output directory for raw downloads")
    parser.add_argument(
        "--only", choices=list(DATASETS), default=None,
        help="Download just one dataset instead of both",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS

    failures = []
    for name, slug in targets.items():
        dest = out_root / name
        print(f"Downloading {name} ({slug}) -> {dest} ...")
        try:
            download_dataset(slug, dest)
            print(f"  done: {sum(1 for _ in dest.rglob('*') if _.is_file())} files")
        except Exception as e:  # kaggle raises a plain Exception on auth failure
            print(f"  FAILED: {e}", file=sys.stderr)
            failures.append(name)

    if failures:
        print(
            f"\n{len(failures)} dataset(s) failed to download: {failures}. "
            "Most likely cause: no Kaggle API credentials configured in this "
            "environment. See this file's module docstring for setup options.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
