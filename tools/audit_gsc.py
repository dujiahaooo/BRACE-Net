#!/usr/bin/env python3
"""Audit the exact GSC V2 35-class official split and write stable hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path


EXPECTED_COUNTS = {"train": 84_843, "val": 9_981, "test": 11_005}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--output", default="outputs/topology_control/data_audit.json"
    )
    parser.add_argument(
        "--split-hash-dir",
        help="Optional directory for train_files.sha256/val_files.sha256/test_files.sha256",
    )
    args = parser.parse_args()
    root = Path(args.data_dir).expanduser().resolve()
    validation_file = root / "validation_list.txt"
    testing_file = root / "testing_list.txt"
    noise_dir = root / "_background_noise_"
    if not validation_file.is_file() or not testing_file.is_file():
        raise SystemExit("Missing official validation/testing list")
    if not noise_dir.is_dir():
        raise SystemExit("Missing _background_noise_ directory")

    validation_paths = {
        line.strip()
        for line in validation_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    testing_paths = {
        line.strip()
        for line in testing_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    labels = sorted(
        path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")
    )
    split_paths = {name: [] for name in EXPECTED_COUNTS}
    invalid_sample_rates = []
    for label in labels:
        for wav_path in sorted((root / label).glob("*.wav")):
            relative = wav_path.relative_to(root).as_posix()
            if relative in validation_paths:
                split = "val"
            elif relative in testing_paths:
                split = "test"
            else:
                split = "train"
            split_paths[split].append(relative)
            with wave.open(str(wav_path), "rb") as handle:
                sample_rate = handle.getframerate()
            if sample_rate != 16_000:
                invalid_sample_rates.append({"path": relative, "sample_rate": sample_rate})

    counts = {name: len(paths) for name, paths in split_paths.items()}
    output = {
        "data_dir": str(root),
        "class_count": len(labels),
        "classes": labels,
        "counts": counts,
        "expected_counts": EXPECTED_COUNTS,
        "counts_match": counts == EXPECTED_COUNTS,
        "sample_rate_hz": 16_000,
        "invalid_sample_rates": invalid_sample_rates,
        "validation_list_sha256": sha256_bytes(validation_file.read_bytes()),
        "testing_list_sha256": sha256_bytes(testing_file.read_bytes()),
        "split_path_list_sha256": {
            name: sha256_bytes(("\n".join(paths) + "\n").encode("utf-8"))
            for name, paths in split_paths.items()
        },
        "background_noise_files": sorted(path.name for path in noise_dir.glob("*.wav")),
        "passed": len(labels) == 35
        and counts == EXPECTED_COUNTS
        and not invalid_sample_rates,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.split_hash_dir:
        hash_dir = Path(args.split_hash_dir)
        hash_dir.mkdir(parents=True, exist_ok=True)
        for split, digest in output["split_path_list_sha256"].items():
            (hash_dir / f"{split}_files.sha256").write_text(
                f"{digest}\n", encoding="utf-8"
            )
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit("GSC audit failed")


if __name__ == "__main__":
    main()

