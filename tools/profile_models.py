#!/usr/bin/env python3
"""Count exact parameters and Conv/Linear multiply-accumulates."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import MODEL_NAMES, create_model


def count_macs(model, sample):
    totals = {"conv1d": 0, "conv2d": 0, "linear": 0}
    handles = []

    def hook(module, _inputs, output):
        if isinstance(module, nn.Conv1d):
            totals["conv1d"] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
        elif isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            totals["conv2d"] += output.numel() * (module.in_channels // module.groups) * kernel
        elif isinstance(module, nn.Linear):
            totals["linear"] += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(sample)
    for handle in handles:
        handle.remove()
    return {**totals, "total": sum(totals.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    report = {
        "input_shape": [1, 1, 80, 101],
        "mac_scope": "Conv1d + Conv2d + Linear; pooling/norm/activation/elementwise excluded",
        "models": {},
    }
    for name in args.models:
        model = create_model(name).to(device).eval()
        report["models"][name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "macs": count_macs(model, torch.randn(1, 1, 80, 101, device=device)),
        }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
