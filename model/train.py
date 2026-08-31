"""Transfer-learning training loop for the whole-board classifier.

Usage:
    python train.py --config configs/default.yaml

Two-phase transfer learning (controlled by train.freeze_backbone_epochs in config):
  Phase 1: backbone frozen, only the two new heads train (fast, stabilizes the
           randomly-initialized heads before touching pretrained features).
  Phase 2: full network unfrozen, fine-tuned end-to-end at the configured LR.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.augmentations import build_eval_transform, build_train_transform
from data.datasets import WholeBoardDataset
from model import build_model


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_backbone_trainable(model, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def run_epoch(model, loader, device, optimizer, criteria, weights, train: bool) -> dict:
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
    with torch.set_grad_enabled(train):
        for images, targets in loader:
            images = images.to(device)
            pass_fail_t = targets["pass_fail"].to(device)
            defect_t = targets["defect_type"].to(device)

            outputs = model(images)
            loss_pf = criteria["pass_fail"](outputs["pass_fail_logits"], pass_fail_t)
            loss_dt = criteria["defect_type"](outputs["defect_type_logits"], defect_t)
            loss = weights["pass_fail"] * loss_pf + weights["defect_type"] * loss_dt

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    return {"loss": total_loss / max(n_batches, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    data_cfg, train_cfg = config["data"], config["train"]

    torch.manual_seed(train_cfg.get("seed", 42))
    device = resolve_device(train_cfg.get("device", "auto"))
    print(f"Using device: {device}")

    image_size = data_cfg["image_size"]
    train_ds = WholeBoardDataset(
        data_cfg["manifest"], split="train", transform=build_train_transform(image_size)
    )
    val_ds = WholeBoardDataset(
        data_cfg["manifest"], split="val", transform=build_eval_transform(image_size)
    )
    train_loader = DataLoader(
        train_ds, batch_size=data_cfg["batch_size"], shuffle=True,
        num_workers=data_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    model = build_model(config).to(device)

    criteria = {
        "pass_fail": torch.nn.BCEWithLogitsLoss(),
        "defect_type": torch.nn.BCEWithLogitsLoss(),
    }
    weights = {
        "pass_fail": train_cfg["pass_fail_loss_weight"],
        "defect_type": train_cfg["defect_type_loss_weight"],
    }

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    freeze_epochs = train_cfg.get("freeze_backbone_epochs", 0)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    optimizer = None

    for epoch in range(train_cfg["epochs"]):
        # Rebuild the optimizer only at phase transitions (epoch 0, and the
        # frozen->unfrozen switch) so Adam's running moment estimates persist
        # within a phase instead of resetting every epoch.
        if epoch == 0 or epoch == freeze_epochs:
            set_backbone_trainable(model, trainable=epoch >= freeze_epochs)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
            )

        t0 = time.time()
        train_metrics = run_epoch(
            model, train_loader, device, optimizer, criteria, weights, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, device, optimizer, criteria, weights, train=False
        )
        elapsed = time.time() - t0

        phase = "frozen-backbone" if epoch < freeze_epochs else "full-finetune"
        print(
            f"epoch {epoch+1}/{train_cfg['epochs']} [{phase}] "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"({elapsed:.1f}s)"
        )
        history.append({"epoch": epoch + 1, "phase": phase, **{
            f"train_{k}": v for k, v in train_metrics.items()
        }, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config, "epoch": epoch + 1},
                checkpoint_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg.get("early_stopping_patience", 10**9):
                print(f"Early stopping at epoch {epoch+1} (no val improvement).")
                break

    torch.save(
        {"model_state_dict": model.state_dict(), "config": config, "epoch": epoch + 1},
        checkpoint_dir / "last.pt",
    )
    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val loss: {best_val_loss:.4f}. Checkpoints written to {checkpoint_dir}/")


if __name__ == "__main__":
    main()
