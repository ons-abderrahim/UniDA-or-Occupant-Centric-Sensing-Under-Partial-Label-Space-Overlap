#!/usr/bin/env python3
"""
train.py — Train one UniDA method on one task.

Usage examples
--------------
# Occupancy Estimation with PPOT
python train.py --config configs/occupancy.yaml --method ppot

# Activity Recognition with LEAD
python train.py --config configs/activity.yaml --method lead

# Run all four methods on both tasks
bash scripts/run_all.sh
"""

import argparse
import json
import os
import sys

import yaml
import torch

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(__file__))

from src.datasets import build_dataloaders
from src.utils import set_seed, get_device
from src.metrics import print_metrics
from src.methods import (
    PPOT,   train_ppot,
    MLNet,  train_mlnet,
    EIAKDA, train_eiakda,
    LEAD,   train_lead,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge(base: dict, override: dict) -> dict:
    out = dict(base)
    out.update(override)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UniDA Smart-Building Trainer")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config (configs/occupancy.yaml or configs/activity.yaml)")
    parser.add_argument("--method", required=True,
                        choices=["ppot", "mlnet", "eiakda", "lead"],
                        help="UniDA method to train")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save results JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-epoch metric printing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = cfg.get(args.method, {})

    # CLI overrides
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"Device: {device}")
    print(f"Task  : {cfg['task']}")
    print(f"Method: {args.method.upper()}")
    print(f"Source: {cfg['source']}")
    print(f"Target: {cfg['target']}")

    # ── Data ─────────────────────────────────────────────────────────────────
    src_loader, tgt_loader, tgt_eval_loader, n_features, seq_len = build_dataloaders(
        source_path   = cfg["source"],
        target_path   = cfg["target"],
        known_classes = cfg["known_classes"],
        seq_len       = cfg.get("seq_len", 10),
        stride        = cfg.get("stride", 1),
        batch_size    = cfg.get("batch_size", 64),
    )
    num_classes = cfg["num_classes"]
    print(f"Features: {n_features}, Seq len: {seq_len}, Known classes: {num_classes}")
    print(f"Source batches: {len(src_loader)}  |  Target batches: {len(tgt_eval_loader)}")

    # ── Build model + train ───────────────────────────────────────────────────
    verbose = not args.quiet

    if args.method == "ppot":
        model = PPOT(
            in_channels = n_features,
            seq_len     = seq_len,
            num_classes = num_classes,
            ot_mass = method_cfg.get("ot_mass", 0.5),
            ot_eps  = method_cfg.get("ot_eps",  0.05),
            tau_ent = method_cfg.get("tau_ent", 0.5),
        )
        best = train_ppot(
            model, src_loader, tgt_loader, tgt_eval_loader,
            epochs       = cfg.get("epochs", 80),
            lr           = cfg.get("lr", 1e-3),
            weight_decay = cfg.get("weight_decay", 1e-4),
            device       = device,
            verbose      = verbose,
        )

    elif args.method == "mlnet":
        model = MLNet(
            in_channels = n_features,
            seq_len     = seq_len,
            num_classes = num_classes,
            knn_k       = method_cfg.get("knn_k", 5),
            mixup_alpha = method_cfg.get("mixup_alpha", 0.2),
            tau         = method_cfg.get("tau", 0.5),
        )
        best = train_mlnet(
            model, src_loader, tgt_loader, tgt_eval_loader,
            epochs       = cfg.get("epochs", 80),
            lr           = cfg.get("lr", 1e-3),
            weight_decay = cfg.get("weight_decay", 1e-4),
            device       = device,
            verbose      = verbose,
        )

    elif args.method == "eiakda":
        model = EIAKDA(
            in_channels = n_features,
            seq_len     = seq_len,
            num_classes = num_classes,
            knn_k  = method_cfg.get("knn_k",  7),
            q_low  = method_cfg.get("q_low",  0.33),
            q_high = method_cfg.get("q_high", 0.67),
        )
        best = train_eiakda(
            model, src_loader, tgt_loader, tgt_eval_loader,
            epochs       = cfg.get("epochs", 80),
            lr           = cfg.get("lr", 1e-3),
            weight_decay = cfg.get("weight_decay", 1e-4),
            device       = device,
            verbose      = verbose,
        )

    elif args.method == "lead":
        model = LEAD(
            in_channels = n_features,
            seq_len     = seq_len,
            num_classes = num_classes,
        )
        best = train_lead(
            model, src_loader, tgt_loader, tgt_eval_loader,
            src_epochs   = method_cfg.get("src_epochs", cfg.get("epochs", 80) // 2),
            tgt_epochs   = method_cfg.get("tgt_epochs", cfg.get("epochs", 80) // 2),
            lr           = cfg.get("lr", 1e-3),
            weight_decay = cfg.get("weight_decay", 1e-4),
            device       = device,
            verbose      = verbose,
        )

    # ── Print & save results ─────────────────────────────────────────────────
    print("\n── Best metrics ──")
    print_metrics(best, prefix=f"{cfg['task'].upper()} | {args.method.upper()}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"{cfg['task']}_{args.method}.json",
    )
    result = {
        "task":   cfg["task"],
        "method": args.method,
        **best,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
