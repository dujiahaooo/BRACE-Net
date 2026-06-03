# BRACE-Net: Dual-Stream Keyword Spotting with Time-Aware Coordinate Excitation

Drawing on the observation that local spectral convolutions alone cannot fully capture the long-range temporal dependencies critical for robust keyword spotting, we propose BRACE-Net, a dual-stream architecture that augments the factorized frequency–temporal backbone with a parallel global context stream. Within each dual-stream block, a TACE2D (Time-Aware Coordinate Excitation) module applies independent frequency-axis and time-axis coordinate attention through depthwise convolutions with Hardswish gating, producing channel-wise recalibration that is sensitive to both spectral shape and temporal position. The global stream output is fused with the local stream via a learnable residual scale and GroupNorm, letting the network jointly attend to fine-grained spectral patterns and broad temporal structure without additional encoder depth. Training further employs an Online Hard Example Mining label-smoothing loss (OHEM-LS) that focuses gradient updates on the hardest samples in each batch while preventing overconfident predictions on easy ones. Experiments on Google Speech Commands V1 and V2 show competitive accuracy under a compact parameter budget, with additional generalization evaluation on ESC-50 and UrbanSound8K.

## Repository Structure

```
anonymous_code_release/
  models/
    ├── bracenet.py            # BRACE-Net: BCDualNet, TACE2D, BCDualStreamBlock
    └── subspectralnorm.py     # Sub-Spectral Normalization module
  datasets/
    └── gsc_dataset.py         # Google Speech Commands V2 loader with SpecAugment
  trainers/
    └── train_gsc.py           # Training script: OHEM-LS, AdamW, warmup–cosine schedule
  requirements.txt
```

Loaders for GSC V1, ESC-50, and UrbanSound8K are excluded from this release.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. CPU training is supported via `--cpu`.

## Datasets

- **Google Speech Commands V2** (primary) — 35-class, ~105 k clips, 16 kHz.
  ```
  http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
  ```
- **Google Speech Commands V1** — 30-class variant, same format.
  ```
  http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
  ```
- **ESC-50** — 50-class environmental sound classification (generalization evaluation).
  ```
  https://github.com/karolpiczak/ESC-50
  ```
- **UrbanSound8K** — 10-class urban sound classification (generalization evaluation).
  ```
  https://zenodo.org/records/1203745
  ```

The GSC V2 training script expects the standard directory layout:

```
speech_commands/
  backward/
  bed/
  ...
  validation_list.txt
  testing_list.txt
```

## Training

```bash
python trainers/train_gsc.py \
  --data-dir /path/to/speech_commands \
  --output-dir outputs/gsc_v2_main
```


Key training settings: AdamW optimizer, 300 epochs, 5-epoch linear warmup followed by cosine decay, batch size 128. Run `python trainers/train_gsc.py --help` for the full list of options.

Per-seed checkpoints and JSON logs are written to `--output-dir`. A `summary.json` aggregating all seeds is saved on completion.

## Feature Extraction

80-dimensional log-Mel spectrograms computed at 16 kHz (25 ms window, 10 ms hop). SpecAugment (FrequencyMasking F=15, TimeMasking T=35) is applied during training only.

## Blind Review Notice

This repository is submitted for the double-blind review process. Author names, affiliations, and contact information will be added upon acceptance.

## Citation

Anonymous
