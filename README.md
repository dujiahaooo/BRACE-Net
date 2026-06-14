# BRACE-Net: Dual-Stream Keyword Spotting with Time-Aware Coordinate Excitation

Drawing on the observation that local spectral convolutions alone cannot fully capture the long-range temporal dependencies critical for robust keyword spotting, we propose BRACE-Net, a dual-stream architecture that augments the factorized frequency–temporal backbone with a parallel global context stream. Within each dual-stream block, a TACE2D (Time-Aware Coordinate Excitation) module applies independent frequency-axis and time-axis coordinate attention through depthwise convolutions with Hardswish gating, producing channel-wise recalibration that is sensitive to both spectral shape and temporal position. The global stream output is fused with the local stream via a learnable residual scale and GroupNorm, letting the network jointly attend to fine-grained spectral patterns and broad temporal structure without additional encoder depth. Training further employs an Online Hard Example Mining label-smoothing loss (OHEM-LS) that focuses gradient updates on the hardest samples in each batch while preventing overconfident predictions on easy ones. Experiments on Google Speech Commands V1 and V2 show competitive accuracy under a compact parameter budget, with additional generalization evaluation on ESC-50 and UrbanSound8K.

## Repository Structure

```
anonymous_code_release/
  models/
    ├── bracenet.py             # BRACE-Net: BCDualNet, TACE2D, BCDualStreamBlock
    └── subspectralnorm.py      # Sub-Spectral Normalization module
  datasets/
    ├── gsc_dataset.py          # Google Speech Commands V2 loader with SpecAugment
    ├── esc50_dataloader.py     # ESC-50 environmental sound classification loader
    └── urbansound_dataloader.py # UrbanSound8K urban sound classification loader
  trainers/
    └── train_gsc.py            # Training script: OHEM-LS, AdamW, warmup–cosine schedule
  requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. CPU training is supported via `--cpu`.

## Datasets

### Primary Benchmark: Google Speech Commands (KWS)

- **Google Speech Commands V2** — 35-class, 105,829 utterances, 16 kHz.
  - Download: `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
  - Official split: 84,843 train / 9,981 val / 11,005 test utterances
  - Dataset loader: `datasets/gsc_dataset.py`
  
- **Google Speech Commands V1** — 30-class, 65,000 utterances, 16 kHz.
  - Download: `http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz`
  - Purpose: Cross-dataset consistency evaluation
  - Dataset loader: `datasets/gsc_dataset.py` (use same loader)

The GSC dataset loaders expect the standard directory structure:
```
speech_commands/
  _background_noise_/
  backward/
  bed/
  ...
  validation_list.txt
  testing_list.txt
```

### Generalization Benchmarks

- **ESC-50** — 50-class environmental sound classification, 5-fold cross-validation.
  - Download: `https://github.com/karolpiczak/ESC-50`
  - Setup: Extract to `ESC-50/` with `meta/esc50.csv` and `audio/` subdirectory
  - Dataset loader: `datasets/esc50_dataloader.py`
  - Transfer learning: Initialize from GSC V2 checkpoint, fine-tune for 100 epochs

- **UrbanSound8K** — 10-class urban sound classification, 10-fold cross-validation.
  - Download: `https://zenodo.org/records/1203745`
  - Setup: Extract to `UrbanSound8K/` with `metadata/UrbanSound8K.csv` and `audio/fold1-fold10/` subdirectories
  - Dataset loader: `datasets/urbansound_dataloader.py`
  - Transfer learning: Initialize from GSC V2 checkpoint, fine-tune for 100 epochs

## Training

### Google Speech Commands V2 (Main Task)

```bash
python trainers/train_gsc.py \
  --data-dir /path/to/speech_commands_v0.02 \
  --output-dir outputs/gsc_v2_main \
  --seeds 42 1234 2025
```

**Training configuration:**
- Optimizer: AdamW with learning rate $10^{-3}$ and weight decay $10^{-4}$
- Schedule: 5-epoch linear warmup, then cosine annealing to $10^{-5}$ over 300 epochs
- Batch size: 128
- Loss: OHEM cross-entropy (keep ratio 0.7) + label smoothing ($\epsilon=0.1$)
- Data augmentation:
  - SpecAugment: FrequencyMasking F=15, TimeMasking T=35
  - Time shifting: ±100 ms
  - Background noise mixing: probability 0.8

**Outputs:**
- Per-seed checkpoints: `outputs/gsc_v2_main/bracenet_seed{42,1234,2025}.pth`
- Per-seed results: `outputs/gsc_v2_main/result_seed{42,1234,2025}.json`
- Summary: `outputs/gsc_v2_main/summary.json` (aggregated over all seeds)

Run `python trainers/train_gsc.py --help` for full list of command-line options.

### Transfer Learning: ESC-50

To fine-tune a trained GSC V2 model on ESC-50 (shown in Table 3 of paper):

```python
import torch
from datasets.esc50_dataloader import get_esc50_loader
from models import BCDualNet

# Load pretrained GSC V2 checkpoint
model = BCDualNet(base_c=40, num_classes=35, use_dual=True, use_tfca=True, use_ssn=True, use_extra_res=True)
model.load_state_dict(torch.load('outputs/gsc_v2_main/bracenet_seed42.pth'))

# Fine-tune last layer for ESC-50 (50 classes)
model.classifier[-1] = torch.nn.Conv2d(model.channels[-1], 50, 1)
model = model.cuda()

# Get ESC-50 loaders
train_loader, val_loader, num_classes = get_esc50_loader(
    csv_path='ESC-50/meta/esc50.csv',
    audio_dir='ESC-50/audio',
    fold=1,  # Use fold 1 for validation
    batch_size=32
)

# Fine-tune with AdamW, lr=5e-4, 100 epochs
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
# ... standard fine-tuning loop
```

### Transfer Learning: UrbanSound8K

Similarly, transfer-learn to UrbanSound8K (10 classes):

```python
from datasets.urbansound_dataloader import get_urbansound_loader

# Modify classifier for 10 classes
model.classifier[-1] = torch.nn.Conv2d(model.channels[-1], 10, 1)

# Get UrbanSound8K loaders
train_loader, val_loader, num_classes = get_urbansound_loader(
    csv_path='UrbanSound8K/metadata/UrbanSound8K.csv',
    audio_root='UrbanSound8K/audio',
    fold=10,  # Use fold 10 for validation
    batch_size=32
)
```

## Feature Extraction

**Log-Mel Spectrogram:**
- Sampling rate: 16 kHz
- Window: 512 samples (25 ms)
- Hop length: 160 samples (10 ms)
- Mel bins: 80
- Frequency range: 20–7800 Hz

**Data augmentation (training only):**
- SpecAugment: FrequencyMasking (F=15), TimeMasking (T=35)

## Blind Review Notice

This repository is submitted for the double-blind review process. Author names, affiliations, and contact information will be added upon acceptance.

## Citation

Anonymous (to be updated upon acceptance)
