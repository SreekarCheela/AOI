"""CLI entry point: point at real ground-truth/model JSONL logs and get a report.

Usage:
    python -m shadow_analysis.cli \\
        --ground-truth existing_aoi_log.jsonl \\
        --model new_system_log.jsonl \\
        --out report.md \\
        --run-label "2026-Wk1-2 shadow run"
"""

from __future__ import annotations

import argparse

from shadow_analysis.comparison import compare
from shadow_analysis.io import load_labeled_captures
from shadow_analysis.report import RunMetadata, generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, help="JSONL file of existing-AOI LabeledCapture records")
    parser.add_argument("--model", required=True, help="JSONL file of new-system LabeledCapture records")
    parser.add_argument("--out", required=True, help="Path to write the markdown report")
    parser.add_argument("--run-label", default="(unlabeled run)")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    ground_truth = load_labeled_captures(args.ground_truth)
    model_output = load_labeled_captures(args.model)

    result = compare(ground_truth, model_output)
    report = generate_report(
        result,
        RunMetadata(run_label=args.run_label, start_date=args.start_date, end_date=args.end_date),
    )

    with open(args.out, "w") as f:
        f.write(report)

    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
