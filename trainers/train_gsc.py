#!/usr/bin/env python3
"""Train any registered preset under one frozen GSC protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.gsc_dataset import get_loader
from models import MODEL_NAMES, create_model


class OHEMLabelSmoothingLoss(nn.Module):
    def __init__(self, rate: float = 0.7, smoothing: float = 0.1):
        super().__init__()
        self.rate = rate
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction="none")

    def forward(self, prediction, target):
        losses = self.criterion(prediction, target)
        keep = max(int(losses.size(0) * self.rate), 1)
        return torch.topk(losses, keep).values.mean()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # AdaptiveAvgPool2d backward has no strict deterministic CUDA kernel in
    # torch 2.11.  Keep deterministic mode as an audited warning so the fixed
    # architecture can run; the exact limitation is frozen in the manifest.
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_scheduler(optimizer, warmup_epochs, total_epochs, lr_max, lr_min):
    lr_start = lr_max * 0.1

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return lr_start / lr_max + (1.0 - lr_start / lr_max) * epoch / max(
                warmup_epochs, 1
            )
        progress = epoch - warmup_epochs
        horizon = max(total_epochs - warmup_epochs, 1)
        return lr_min / lr_max + (1.0 - lr_min / lr_max) * 0.5 * (
            1.0 + math.cos(math.pi * progress / horizon)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, (features, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            prediction = model(features).argmax(1)
            correct += prediction.eq(labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise RuntimeError("Evaluation consumed zero samples")
    return 100.0 * correct / total


def train_one(args, seed):
    set_seed(seed)
    train_loader, validation_loader, test_loader, num_classes = get_loader(
        args.data_dir, args.batch_size, args.num_workers, seed=seed
    )
    if num_classes not in {30, 35}:
        raise RuntimeError(f"Expected GSC V1 (30) or V2 (35) classes, found {num_classes}")
    device = resolve_device(args.device)
    model = create_model(args.model, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(
        optimizer, args.warmup_epochs, args.epochs, args.lr_max, args.lr_min
    )
    criterion = OHEMLabelSmoothingLoss(
        rate=args.ohem_rate, smoothing=args.label_smoothing
    )
    best_validation = float("-inf")
    best_epoch = 0
    history = []
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"{args.model}_seed{seed}.pth")
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        criterion.rate = 1.0 if epoch < args.warmup_epochs else args.ohem_rate
        epoch_started = time.time()
        samples_seen = 0
        for batch_index, (features, labels) in enumerate(
            tqdm(
                train_loader,
                desc=f"{args.model} seed={seed} epoch={epoch + 1}/{args.epochs}",
                leave=False,
            )
        ):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            samples_seen += labels.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds = time.time() - epoch_started
        validation_started = time.time()
        validation_accuracy = evaluate(
            model, validation_loader, device, args.max_eval_batches
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        validation_seconds = time.time() - validation_started
        scheduler.step()
        epoch_seconds = time.time() - epoch_started
        history.append(
            {
                "epoch": epoch + 1,
                "val_acc": round(validation_accuracy, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "samples": samples_seen,
                "epoch_time_s": round(epoch_seconds, 3),
                "train_time_s": round(training_seconds, 3),
                "validation_time_s": round(validation_seconds, 3),
                "samples_per_s": round(samples_seen / training_seconds, 3),
            }
        )
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
        print(
            f"epoch {epoch + 1:03d}: val={validation_accuracy:.4f}% "
            f"best={best_validation:.4f}% best_epoch={best_epoch} "
            f"samples/s={samples_seen / training_seconds:.2f}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_started = time.time()
    test_accuracy = evaluate(model, test_loader, device, args.max_eval_batches)
    if device.type == "cuda":
        torch.cuda.synchronize()
    test_seconds = time.time() - test_started
    result = {
        "model": args.model,
        "seed": seed,
        "best_val_acc": round(best_validation, 4),
        "best_epoch": best_epoch,
        "test_acc": round(test_accuracy, 4),
        "train_time_s": round(time.time() - started, 1),
        "test_time_s": round(test_seconds, 3),
        "checkpoint": os.path.abspath(checkpoint_path),
        "device": str(device),
        "precision": "fp32",
        "history": history,
    }
    result_path = os.path.join(args.output_dir, f"result_seed{seed}.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/gsc_v2")
    parser.add_argument("--model", choices=MODEL_NAMES, default="brace_full")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-max", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ohem-rate", type=float, default=0.7)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--frequency-mask", type=int, default=5)
    parser.add_argument("--time-mask", type=int, default=20)
    parser.add_argument("--noise-probability", type=float, default=0.8)
    parser.add_argument("--max-time-shift-ms", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    args = parser.parse_args()
    frozen_loader_values = {
        "frequency_mask": 5,
        "time_mask": 20,
        "noise_probability": 0.8,
        "max_time_shift_ms": 100,
    }
    for name, expected in frozen_loader_values.items():
        if getattr(args, name) != expected:
            parser.error(
                f"--{name.replace('_', '-')} must remain {expected} under the frozen protocol"
            )
    if not 0.0 < args.ohem_rate <= 1.0:
        parser.error("--ohem-rate must be in (0, 1]")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1)")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    summary = {}
    for run_seed in arguments.seeds:
        summary[str(run_seed)] = train_one(arguments, run_seed)
    summary_path = os.path.join(arguments.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}")
