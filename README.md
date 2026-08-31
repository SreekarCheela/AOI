# AOI data pipeline

Data plumbing for the AOI-replacement project (issue #4, part of epic #1):
pretraining data prep, edge-device capture logging, and correlation with the
existing AOI's verdicts. The ESP32 capture rig doesn't exist yet (depends on
issue #2), so this covers everything that doesn't need physical hardware:
dataset prep, schema-as-code, and pipeline scaffolding exercised against
synthetic/sample data.

## Layout

```
src/aoi_pipeline/
  schema/            Labeling schema (Verdict, DefectType, BoardLabel) and
                      capture record schema (CaptureRecord), as code.
  logging_pipeline/   CaptureLogger (image + timestamp + sequence number,
                      JSONL) and StorageMonitor (disk-usage alerting).
  correlation/        Pluggable MatchStrategy (sequence+timestamp by
                      default, ID-based when available) + CorrelationMatcher.
  datasets/           Download/inspect/prep for the SolDef_AI and PCB-AoI
                      Kaggle datasets -> normalized BoardLabel manifests.
  cli.py              `aoi-pipeline` command line entry point.
tests/                 Unit tests (schema, logging, correlation, datasets),
                        all runnable offline against synthetic data.
docs/datasets.md        Dataset provenance, label format, and the remaining
                        steps to close out the "downloaded + inspected" step.
```

## Setup

```
pip install -e ".[dev]"
pytest
```

## CLI

```
# Download + inspect + prep the public pretraining datasets.
# Requires Kaggle credentials — see docs/datasets.md.
aoi-pipeline download-datasets

# Exercise the capture-logging + correlation pipeline against synthetic
# data (no ESP32 rig required).
aoi-pipeline demo-capture-and-correlate
```

## Status against issue #4's acceptance criteria

1. **Public datasets downloaded/inspected/confirmed usable** — dataset specs
   and prep pipeline are implemented and unit-tested against synthetic data;
   the actual download+inspection against Kaggle needs to be run once with
   real credentials (see `docs/datasets.md`).
2. **Logging pipeline captures timestamp + sequence + optional board ID** —
   done (`logging_pipeline/capture_logger.py`), demoed via
   `aoi-pipeline demo-capture-and-correlate`.
3. **Correlation matches >=95% within a 1-day run** — done
   (`correlation/`), `CorrelationReport.match_rate` is the metric to check
   against real captured data once it exists; demo shows the mechanism at
   95% on synthetic data.
4. **Labeling schema documented and applied to a week of line data** —
   schema is implemented (`schema/labels.py`); applying it to real line data
   is blocked on issue #2 (rig installed).
5. **Storage plan with disk-usage monitoring** — `StorageMonitor` implements
   warning/critical thresholds and can halt logging before the device fills
   up; local/attached storage only, per the on-prem IP decision.
