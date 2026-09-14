# BRACE-Net

Minimal reference implementation for **“BRACE-Net: Broadcasted Residual
Network with Axis-Coordinate Excitation for Lightweight Keyword Spotting.”**

BRACE keeps a complete BCResBlock local representation and an independently
shortcut-conditioned Time–frequency Axis Coordinate Excitation (TACE)
representation until learned 1×1 fusion. Full-BRACE-Net uses BRACE blocks in
Stages 3–4; Lite-BRACE-Net uses them throughout a narrower encoder.

This repository intentionally contains only the GSC data pipeline, paper model
definitions, training entry point, and scripts needed to audit the reported
parameters, operations and per-epoch training time. Checkpoints, datasets,
paper-writing files and unrelated transfer-learning code are excluded.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download and extract the official Google Speech Commands archive:

```bash
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p data/speech_commands_v0.02
tar -xzf speech_commands_v0.02.tar.gz -C data/speech_commands_v0.02
```

The GSC V2 official 35-class split contains 84,843 training, 9,981 validation
and 11,005 test clips. Verify it before training:

```bash
python tools/audit_gsc.py --data-dir data/speech_commands_v0.02 \
  --output outputs/data_audit.json
```

Inputs are 80-bin log-Mel spectrograms at 16 kHz (`n_fft=512`, `hop=160`).
Training augmentation is ±100 ms time shift, background-noise mixing with
probability 0.8 and amplitude in `[0, 0.1]`, FrequencyMasking 5 and
TimeMasking 20.

## Paper configurations

All configurations are registered in `models/factory.py`.

| Preset | Purpose |
|---|---|
| `bc_resnet6` | Lite parameter-matched baseline |
| `brace_lite` | Lite-BRACE-Net |
| `bc_resnet8` | Full parameter-matched baseline |
| `brace_full` | Full-BRACE-Net |
| `widened_bc_only` | widened local-only control |
| `deep_se` | deep SE control |
| `deep_se_exres` | deep SE with local residual |
| `deep_tace` | deep TACE control |
| `serial_local_tace` | local→TACE serial topology control |
| `serial_tace_local` | TACE→local serial topology control |
| `tfdbp_style_topology_control` | reconstructed DBBR→TFCA topology control |
| `bc_resnet_compute76` | compute-matched BC-ResNet |

Run structural, initialization and gradient checks:

```bash
python tools/verify_models.py
```

## Training

The paper protocol uses AdamW, 300 epochs, batch 128, five linear warm-up
epochs followed by cosine decay from `1e-3` to `1e-5`, weight decay `1e-4`,
OHEM keep rate 0.7 and label smoothing 0.1. Validation chooses the checkpoint;
the test split is evaluated once.

```bash
python trainers/train_gsc.py \
  --data-dir data/speech_commands_v0.02 \
  --output-dir outputs/brace_full/seed42 \
  --model brace_full --seeds 42 --epochs 300 \
  --batch-size 128 --num-workers 8 --device cuda
```

Use independent output directories for seeds 42, 1234 and 2025.

## Parameter/MAC audit

Convolution and linear MAM are measured for a `1×80×101` input. Pooling,
normalization, activations and element-wise gates are intentionally excluded.

```bash
python tools/profile_models.py \
  --models bc_resnet6 brace_lite bc_resnet8 brace_full \
  --device cpu --output outputs/profile.json
```

## Table 3 pure-training timing

The timing script uses one RTX 4090, FP32 and the full GSC V2 training split.
It runs one unmeasured warm-up epoch and three measured epochs in rotating
model order. Measurements include data loading/augmentation, host-to-device
transfer, forward, backward and optimizer step; validation, testing and
checkpoint I/O are excluded.

```bash
python tools/benchmark_train_time.py \
  --data-dir data/speech_commands_v0.02 \
  --num-workers 8 --output outputs/train_time.json
```

## Reproducibility notes

- Random seed controls Python, NumPy and PyTorch initialization.
- The default paper implementation is FP32 with TF32 disabled.
- Data are augmented online; augmented features are not cached.
- `tfdbp_style_topology_control` is an independent reconstruction used only as
  a common-implementation topology control. It is not an official
  TF-DBPResNet reproduction.

## Citation

Citation metadata will be added after the ICASSP 2027 bibliographic record is
available.
