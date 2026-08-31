"""CLI entry points for the data pipeline.

    aoi-pipeline download-datasets [--force]
        Downloads + inspects + preps SolDef_AI and PCB-AoI into data/raw and
        data/processed. Requires Kaggle credentials (see docs/datasets.md).

    aoi-pipeline demo-capture-and-correlate
        Runs the logging pipeline and correlation module against synthetic
        capture/verdict data, so the scaffolding can be exercised without
        the ESP32 rig (which doesn't exist yet — see issue #4).
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from aoi_pipeline.correlation.matcher import CorrelationMatcher
from aoi_pipeline.correlation.strategies import ExistingAoiVerdict
from aoi_pipeline.datasets.pcb_aoi import PcbAoiPreparer
from aoi_pipeline.datasets.soldef_ai import SolDefAIPreparer
from aoi_pipeline.logging_pipeline.capture_logger import CaptureLogger
from aoi_pipeline.schema.labels import DefectType, Verdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _download_datasets(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    preparers = [
        SolDefAIPreparer(raw_dir=raw_dir, processed_dir=processed_dir),
        PcbAoiPreparer(raw_dir=raw_dir, processed_dir=processed_dir),
    ]
    for preparer in preparers:
        print(f"--- {preparer.spec.name} ---")
        preparer.download(force=args.force)
        report = preparer.inspect()
        print(
            f"images={report.image_count} usable={report.is_usable(preparer.spec)} "
            f"raw_labels={report.raw_label_values}"
        )
        if report.unmapped_raw_labels:
            print(f"WARNING: unmapped raw labels: {report.unmapped_raw_labels}")
        manifest_path = preparer.prep()
        print(f"manifest written to {manifest_path}")


def _demo_capture_and_correlate(args: argparse.Namespace) -> None:
    """Exercises the logging + correlation scaffolding end-to-end with
    synthetic data, since there's no ESP32 rig yet to generate real
    captures. Useful as a smoke test / example of how the pieces fit."""
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        capture_logger = CaptureLogger(log_dir=log_dir, station_id="demo-station")

        verdicts = []
        for i in range(20):
            record = capture_logger.log_capture(
                image_path=f"/data/raw/demo_img_{i}.png",
                metadata={"demo": True},
            )
            is_fail = i % 5 == 0
            # The existing AOI reports its verdict a beat after the image is
            # captured — well within SequenceTimestampMatchStrategy's
            # default 2s tolerance.
            verdicts.append(
                ExistingAoiVerdict(
                    verdict_id=f"existing-aoi-{i}",
                    timestamp=record.timestamp + timedelta(milliseconds=400),
                    verdict=Verdict.FAIL if is_fail else Verdict.PASS,
                    defect_types=[DefectType.SOLDER_BRIDGE] if is_fail else [],
                )
            )
        # Simulate one dropped verdict (existing AOI missed a frame) so the
        # demo also shows the match_rate falling below 100%.
        verdicts.pop()

        log_path = next(log_dir.glob("captures_*.jsonl"))
        from aoi_pipeline.logging_pipeline.capture_logger import iter_records

        captures = list(iter_records(log_path))
        report = CorrelationMatcher().run(captures, verdicts)

        print(f"captures={len(captures)} verdicts={len(verdicts)}")
        print(f"matched={len(report.matches)} match_rate={report.match_rate:.2%}")
        print(f"unmatched_captures={len(report.unmatched_captures)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aoi-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-datasets", help="Download + inspect + prep SolDef_AI and PCB-AoI"
    )
    download_parser.add_argument("--raw-dir", default="data/raw")
    download_parser.add_argument("--processed-dir", default="data/processed")
    download_parser.add_argument("--force", action="store_true")
    download_parser.set_defaults(func=_download_datasets)

    demo_parser = subparsers.add_parser(
        "demo-capture-and-correlate",
        help="Run the logging + correlation pipeline against synthetic data",
    )
    demo_parser.set_defaults(func=_demo_capture_and_correlate)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
