"""Google Speech Commands V2 35-class loader used by all paired runs."""

from __future__ import annotations

import os
import random
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset


SAMPLE_RATE = 16_000
SAMPLE_LENGTH = 16_000
TIME_SHIFT_SAMPLES = 1_600
BACKGROUND_NOISE_PROBABILITY = 0.8
BACKGROUND_NOISE_MAX_AMPLITUDE = 0.1
FREQUENCY_MASK_PARAM = 5
TIME_MASK_PARAM = 20


class GSC_Dataset(Dataset):
    def __init__(self, data_dir: str | os.PathLike[str], split: str = "train"):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.split = split
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"GSC directory does not exist: {self.data_dir}")

        self.labels = sorted(
            path.name
            for path in self.data_dir.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        )
        self.label_to_idx = {label: index for index, label in enumerate(self.labels)}
        validation_file = self.data_dir / "validation_list.txt"
        testing_file = self.data_dir / "testing_list.txt"
        if not validation_file.is_file() or not testing_file.is_file():
            raise FileNotFoundError("Official validation_list.txt/testing_list.txt are required")
        validation_paths = {
            line.strip() for line in validation_file.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        testing_paths = {
            line.strip() for line in testing_file.read_text(encoding="utf-8").splitlines() if line.strip()
        }

        self.file_list: list[tuple[Path, int, str]] = []
        for label in self.labels:
            for wav_path in sorted((self.data_dir / label).glob("*.wav")):
                relative_path = wav_path.relative_to(self.data_dir).as_posix()
                belongs = {
                    "val": relative_path in validation_paths,
                    "test": relative_path in testing_paths,
                    "train": relative_path not in validation_paths and relative_path not in testing_paths,
                }[split]
                if belongs:
                    self.file_list.append((wav_path, self.label_to_idx[label], relative_path))

        noise_dir = self.data_dir / "_background_noise_"
        self.background_noise_files = sorted(noise_dir.glob("*.wav"))
        if split == "train" and not self.background_noise_files:
            raise FileNotFoundError(f"No background-noise WAV files found in {noise_dir}")

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=80,
            n_fft=512,
            hop_length=160,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        self.frequency_mask = torchaudio.transforms.FrequencyMasking(FREQUENCY_MASK_PARAM)
        self.time_mask = torchaudio.transforms.TimeMasking(TIME_MASK_PARAM)

    def __len__(self) -> int:
        return len(self.file_list)

    @staticmethod
    def _read_mono(path: Path) -> torch.Tensor:
        waveform_array, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz but found {sample_rate} Hz: {path}")
        waveform = torch.from_numpy(waveform_array).transpose(0, 1)
        return waveform.mean(dim=0, keepdim=True)

    @staticmethod
    def _fit_one_second(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[-1] < SAMPLE_LENGTH:
            return F.pad(waveform, (0, SAMPLE_LENGTH - waveform.shape[-1]))
        return waveform[..., :SAMPLE_LENGTH]

    @staticmethod
    def _time_shift(waveform: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-TIME_SHIFT_SAMPLES, TIME_SHIFT_SAMPLES)
        if shift == 0:
            return waveform
        if shift < 0:
            return F.pad(waveform[..., :shift], (-shift, 0))
        return F.pad(waveform[..., shift:], (0, shift))

    def _sample_noise(self) -> torch.Tensor:
        noise = self._read_mono(random.choice(self.background_noise_files))
        if noise.shape[-1] < SAMPLE_LENGTH:
            repeats = (SAMPLE_LENGTH + noise.shape[-1] - 1) // noise.shape[-1]
            noise = noise.repeat(1, repeats)
        start = random.randint(0, noise.shape[-1] - SAMPLE_LENGTH)
        return noise[..., start : start + SAMPLE_LENGTH]

    def __getitem__(self, index: int):
        path, label, relative_path = self.file_list[index]
        waveform = self._fit_one_second(self._read_mono(path))
        if self.split == "train":
            waveform = self._time_shift(waveform)
            if random.random() < BACKGROUND_NOISE_PROBABILITY:
                amplitude = random.uniform(0.0, BACKGROUND_NOISE_MAX_AMPLITUDE)
                waveform = torch.clamp(waveform + amplitude * self._sample_noise(), -1.0, 1.0)

        log_mel = self.db_transform(self.mel_transform(waveform))
        if self.split == "train":
            log_mel = self.frequency_mask(log_mel)
            log_mel = self.time_mask(log_mel)
        return log_mel, label, relative_path


def _collate_without_paths(batch):
    features, labels, _ = zip(*batch)
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def get_loader(
    data_dir: str | os.PathLike[str],
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
    include_paths: bool = False,
):
    train_dataset = GSC_Dataset(data_dir, "train")
    validation_dataset = GSC_Dataset(data_dir, "val")
    test_dataset = GSC_Dataset(data_dir, "test")
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    collate_fn = None if include_paths else _collate_without_paths
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        persistent_workers=num_workers > 0,
        generator=generator,
        collate_fn=collate_fn,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, collate_fn=collate_fn, **common
    )
    test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, **common)
    return train_loader, validation_loader, test_loader, len(train_dataset.labels)
