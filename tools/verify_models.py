#!/usr/bin/env python3
"""Structural checks for every paper model preset."""

import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import MODEL_NAMES, create_model


EXPECTED_PARAMETERS = {
    "bc_resnet6": 192251,
    "bc_resnet8": 326979,
    "brace_lite": 191281,
    "brace_full": 325991,
    "widened_bc_only": 332225,
    "deep_se": 307175,
    "deep_se_exres": 307175,
    "deep_tace": 325991,
    "serial_local_tace": 325991,
    "serial_tace_local": 312831,
    "tfdbp_style_topology_control": 325991,
    "bc_resnet_compute76": 451209,
}


def main():
    sample = torch.randn(2, 1, 80, 101)
    for name in MODEL_NAMES:
        torch.manual_seed(42)
        first = create_model(name)
        torch.manual_seed(42)
        second = create_model(name)
        assert sum(p.numel() for p in first.parameters()) == EXPECTED_PARAMETERS[name]
        for (key_a, value_a), (key_b, value_b) in zip(
            first.state_dict().items(), second.state_dict().items()
        ):
            assert key_a == key_b and torch.equal(value_a, value_b)
        assert first(sample).shape == (2, 35)
        first.train()
        first(sample).sum().backward()
        missing = [key for key, value in first.named_parameters() if value.requires_grad and value.grad is None]
        assert not missing, (name, missing)
        print(f"PASS {name}: parameters={EXPECTED_PARAMETERS[name]}")


if __name__ == "__main__":
    main()
