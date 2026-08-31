"""Evaluate a checkpoint on the held-out test split and write a report.

Usage:
    python eval.py --checkpoint checkpoints/best.pt --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.augmentations import build_eval_transform
from data.datasets import WholeBoardDataset
from labels import DEFECT_CLASSES
from metrics import defect_type_metrics, pass_fail_rates
from model import build_model
from train import resolve_device


@torch.no_grad()
def run_inference(model, loader, device, pf_threshold: float, dt_threshold: float):
    all_pf_preds, all_pf_targets = [], []
    all_dt_preds, all_dt_targets = [], []

    model.eval()
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)

        pf_probs = torch.sigmoid(outputs["pass_fail_logits"]).cpu().numpy()
        dt_probs = torch.sigmoid(outputs["defect_type_logits"]).cpu().numpy()

        all_pf_preds.append((pf_probs >= pf_threshold).astype(int))
        all_pf_targets.append(targets["pass_fail"].numpy().astype(int))
        all_dt_preds.append((dt_probs >= dt_threshold).astype(int))
        all_dt_targets.append(targets["defect_type"].numpy().astype(int))

    return (
        np.concatenate(all_pf_preds),
        np.concatenate(all_pf_targets),
        np.concatenate(all_dt_preds),
        np.concatenate(all_dt_targets),
    )


def render_markdown_report(rates: dict, defect_metrics: dict, meta: dict) -> str:
    epic_targets = {
        "agreement_rate": (">=", 0.98),
        "escape_rate": ("<=", 0.005),
        "false_reject_rate": ("<=", 0.03),
    }

    def check(name, value):
        op, target = epic_targets[name]
        met = (value >= target) if op == ">=" else (value <= target)
        return "PASS" if met else "FAIL", target

    lines = [
        "# Baseline Eval Report — Whole-Board Solder-Joint Classifier",
        "",
        f"Generated: {meta['generated_at']}",
        f"Checkpoint: `{meta['checkpoint']}`",
        f"Test split size: {rates['total']} boards",
        "",
        "**Note:** this evaluates against the held-out split of the *public* "
        "pretraining datasets (SolDef_AI / PCB-AoI), not real production line "
        "data. The epic's shadow-mode acceptance bar (issue #1) is reproduced "
        "below as a reference target, not a claim that it's been met on real "
        "line data — that check happens after fine-tuning (issue #6, currently "
        "stubbed, see finetune_line_data.py).",
        "",
        "## Pass/Fail Metrics",
        "",
        "| Metric | Value | Epic target | Status |",
        "|---|---|---|---|",
    ]
    for name in ["agreement_rate", "escape_rate", "false_reject_rate"]:
        status, target = check(name, rates[name])
        lines.append(f"| {name} | {rates[name]:.4f} | {epic_targets[name][0]} {target} | {status} |")

    lines += [
        "",
        f"- Escape rate, normalized to actual-fail boards only (miss rate): "
        f"{rates['escape_rate_of_defects']:.4f}",
        f"- False-reject rate, normalized to actual-pass boards only (false-positive rate): "
        f"{rates['false_reject_rate_of_passes']:.4f}",
        "",
        "### Raw confusion counts",
        "",
        f"- True positive (pass, predicted pass): {rates['true_positive_pass']}",
        f"- True negative (fail, predicted fail): {rates['true_negative_fail']}",
        f"- Escapes (fail, predicted pass): {rates['escapes']}",
        f"- False rejects (pass, predicted fail): {rates['false_rejects']}",
        "",
        "## Defect-Type Metrics (multi-label)",
        "",
        f"Micro-averaged: precision={defect_metrics['micro']['precision']:.4f}, "
        f"recall={defect_metrics['micro']['recall']:.4f}, "
        f"f1={defect_metrics['micro']['f1']:.4f}",
        "",
        "| Defect type | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for name in DEFECT_CLASSES:
        m = defect_metrics["per_class"][name]
        lines.append(
            f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    data_cfg, eval_cfg = config["data"], config["eval"]

    device = resolve_device(config["train"].get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = build_model(checkpoint.get("config", config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_ds = WholeBoardDataset(
        data_cfg["manifest"], split="test", transform=build_eval_transform(data_cfg["image_size"])
    )
    test_loader = DataLoader(
        test_ds, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"]
    )

    pf_preds, pf_targets, dt_preds, dt_targets = run_inference(
        model, test_loader, device, eval_cfg["pass_fail_threshold"], eval_cfg["defect_type_threshold"]
    )

    rates = pass_fail_rates(pf_preds, pf_targets)
    defect_metrics = defect_type_metrics(dt_preds, dt_targets)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
    }

    report_dir = Path(eval_cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "baseline_eval_report.md").write_text(
        render_markdown_report(rates, defect_metrics, meta)
    )
    (report_dir / "baseline_eval_metrics.json").write_text(
        json.dumps({"pass_fail": rates, "defect_type": defect_metrics, "meta": meta}, indent=2)
    )
    print(f"Wrote {report_dir}/baseline_eval_report.md and baseline_eval_metrics.json")
    print(f"agreement_rate={rates['agreement_rate']:.4f} "
          f"escape_rate={rates['escape_rate']:.4f} "
          f"false_reject_rate={rates['false_reject_rate']:.4f}")


if __name__ == "__main__":
    main()
