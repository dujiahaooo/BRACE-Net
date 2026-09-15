# BRACE-Net

BRACE-Net is a lightweight keyword-spotting model that combines a complete
BCResBlock local representation with shortcut-conditioned Time–frequency Axis
Coordinate Excitation (TACE), followed by learned feature fusion.
The design preserves local spectro-temporal evidence while introducing
axis-aware global context under a compact parameter budget.

## Configurations

```text
brace_lite                       Lite-BRACE-Net
brace_full                       Full-BRACE-Net
bc_resnet6                       Lite parameter-scale baseline
bc_resnet8                       Full parameter-scale baseline
widened_bc_only                  widened local-only control
deep_se                          deep SE control
deep_se_exres                    deep SE with local residual
deep_tace                        deep TACE control
serial_local_tace                local→TACE topology control
serial_tace_local                TACE→local topology control
tfdbp_style_topology_control     reconstructed DBBR→TFCA control
bc_resnet_compute76              compute-matched BC-ResNet
```

All configurations are registered in `models/factory.py`.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Data

Download and extract Google Speech Commands V2:

```bash
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p data/speech_commands_v0.02
tar -xzf speech_commands_v0.02.tar.gz -C data/speech_commands_v0.02
```

The loader uses the official 35-class split, 16-kHz audio, and 80-bin log-Mel
features. Dataset counts and split hashes can be checked with:

```bash
python tools/audit_gsc.py \
  --data-dir data/speech_commands_v0.02 \
  --output outputs/data_audit.json
```

## Training

```bash
python trainers/train_gsc.py \
  --data-dir data/speech_commands_v0.02 \
  --output-dir outputs/brace_full/seed42 \
  --model brace_full --seeds 42 --epochs 300 \
  --batch-size 128 --num-workers 8 --device cuda
```

The training entry point implements AdamW, linear warm-up followed by cosine
decay, online hard-example mining, label smoothing, validation-based checkpoint
selection, and one final test evaluation.

## Repository layout

```text
datasets/       Google Speech Commands data pipeline
models/         BRACE-Net, BC-ResNet, and control configurations
trainers/       GSC training entry point
tools/          data audit, model checks, profiling, and train-time measurement
```

## Verification

```bash
python tools/verify_models.py
python tools/profile_models.py --help
python tools/benchmark_train_time.py --help
python trainers/train_gsc.py --help
```

## License

This project is released under the [BSD 3-Clause Clear License](LICENSE).
The SubSpectralNorm implementation retains the copyright and attribution of
the original [Qualcomm BC-ResNet repository](https://github.com/Qualcomm-AI-research/bcresnet).

## Citation

Citation metadata will be added when the bibliographic record is available.
