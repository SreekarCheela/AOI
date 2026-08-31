# Pretraining datasets

Per issue #4, two public Kaggle datasets are used for pretraining/transfer
learning before any real line data exists.

## SolDef_AI

- Kaggle: https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection
- Paper: Calabrese et al., *SolDef_AI: An Open Source PCB Dataset for Mask
  R-CNN Defect Detection in Soldering Processes of Electronic Components*
  (MDPI, 2024). https://www.mdpi.com/2504-4494/8/3/117
- ~1150 SMT solder-joint images (0603/0805/1206 resistors, 0805 capacitors),
  each component shot from 3 viewpoints (top, 45°, axonometric) under
  varying lighting.
- Two defect families documented: component placement (misalignment) and
  solder-joint quality (excess/insufficient solder).
- Registered in code as `SOLDEF_AI_SPEC` in
  `src/aoi_pipeline/datasets/soldef_ai.py`.

## PCB-AoI

- Kaggle: https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi
- Origin: KubeEdge-Ianvs benchmark (China Telecom / Raisecom Technology).
  https://ianvs-dev.readthedocs.io/en/latest/proposals/scenarios/industrial-defect-detection/pcb-aoi.html
- ~1271 solder-paste-inspection images (600x600px) across 230+ boards.
- Two defect categories documented: `Bad_qiaojiao` (lifted lead) and
  `Bad_podu` (solder slope defect).
- Registered in code as `PCB_AOI_SPEC` in
  `src/aoi_pipeline/datasets/pcb_aoi.py`.

## Status: needs a real download to close out acceptance criterion #1

This scaffolding was built without Kaggle credentials in the dev
environment, so `DatasetPreparer.download()`/`inspect()` have **not been
run against the real archives yet**. The `raw_label_map` in each dataset
module is seeded from public documentation, not from opening the archives.

To finish acceptance criterion #1 ("Public datasets downloaded, inspected,
and confirmed usable — image count, label format documented"):

1. Configure Kaggle credentials on the machine that will run this:
   `~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars. You
   must also accept each dataset's terms on kaggle.com at least once.
2. `pip install -e ".[dev]"` (installs `kagglehub`).
3. `aoi-pipeline download-datasets` — downloads into `data/raw/`, inspects,
   and preps a normalized manifest into `data/processed/<name>/manifest.jsonl`.
4. Check the printed `raw_labels` / `unmapped_raw_labels` for each dataset
   against `raw_label_map` in `soldef_ai.py` / `pcb_aoi.py`, and extend the
   map if the real on-disk folder names differ from what's seeded here (the
   labeling logic itself — folder-name-based, generic across both datasets
   — does not need to change, only the map).
5. Update this doc with the confirmed image counts and any map corrections.

## How prep works

Both datasets follow the common Kaggle image-classification layout,
`<class_name>/<image>.png`. `DatasetPreparer` (in
`src/aoi_pipeline/datasets/base.py`) is generic over that layout:

- `download()` fetches via `kagglehub.dataset_download()` into `data/raw/<name>/`.
- `inspect()` walks the raw folder, counts images, and reports which parent
  folder names aren't yet in `raw_label_map`.
- `prep()` writes `data/processed/<name>/manifest.jsonl`, one line per
  image, each a `BoardLabel` (see `src/aoi_pipeline/schema/labels.py`) with
  `source=LabelSource.PUBLIC_DATASET` and the original folder name preserved
  as `raw_label` for traceability.

This is exercised end-to-end (without network access) in
`tests/test_datasets.py` using synthetic placeholder images.
