#!/usr/bin/env python3
"""Measure Table 3 pure training wall-clock on the full GSC V2 train split."""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.gsc_dataset import GSC_Dataset, _collate_without_paths
from models import create_model
from trainers.train_gsc import OHEMLabelSmoothingLoss


CONFIGURATIONS = {
    "widened BC-ResNet": "widened_bc_only",
    "deep SE": "deep_se",
    "deep SE + local residual": "deep_se_exres",
    "deep TACE": "deep_tace",
    "Full-BRACE-Net": "brace_full",
    "serial local→TACE": "serial_local_tace",
    "serial TACE→local": "serial_tace_local",
    "TF-DBPResNet reconstructed control": "tfdbp_style_topology_control",
    "compute-matched BC-ResNet": "bc_resnet_compute76",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_epoch(model, optimizer, criterion, loader, measured):
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    for features, labels in loader:
        features = features.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        criterion(model(features), labels).backward()
        optimizer.step()
    torch.cuda.synchronize()
    return time.perf_counter() - started if measured else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default="outputs/train_time.json")
    args = parser.parse_args()
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise SystemExit("This paper protocol requires one RTX 4090")
    seed_everything(42)
    loader = DataLoader(
        GSC_Dataset(args.data_dir, "train"), batch_size=128, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(42),
        collate_fn=_collate_without_paths,
    )
    states = {}
    for label, preset in CONFIGURATIONS.items():
        seed_everything(42)
        model = create_model(preset).cuda()
        states[label] = (
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
            OHEMLabelSmoothingLoss(rate=0.7, smoothing=0.1),
        )
    labels = tuple(CONFIGURATIONS)
    for label in labels:
        train_epoch(*states[label], loader, measured=False)
    timings = {label: [] for label in labels}
    for measured_epoch in range(3):
        offset = measured_epoch * 3
        for label in labels[offset:] + labels[:offset]:
            timings[label].append(train_epoch(*states[label], loader, measured=True))
    result = {
        "hardware": torch.cuda.get_device_name(0),
        "seed": 42,
        "batch_size": 128,
        "precision": "FP32",
        "num_workers": args.num_workers,
        "scope": "data/augmentation + H2D + forward + backward + optimizer step",
        "validation_test_checkpoint_io": False,
        "measurements": {
            label: {"epochs_s": values, "median_s_per_epoch": statistics.median(values)}
            for label, values in timings.items()
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
