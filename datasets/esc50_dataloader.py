"""ESC-50 Environmental Sound Classification dataset loader.

ESC-50 is a 50-class environmental sound classification dataset with
normalized fold-based cross-validation splits.
"""

import os
import torch
import torchaudio
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass


class ESC50Dataset(Dataset):
    """ESC-50 Environmental Sound Classification Dataset.
    
    Dataset: https://github.com/karolpiczak/ESC-50
    Metadata: ESC-50/meta/esc50.csv
    Audio: ESC-50/audio/
    
    Each audio file is 5 seconds at 16 kHz with 50 environmental sound classes.
    Fold-based cross-validation: 1-5 folds available.
    """

    def __init__(self, csv_path, audio_dir, target_sample_rate=16000, 
                 target_length=80000, fold=1, train=True):
        """
        Args:
            csv_path: Path to esc50.csv metadata file.
            audio_dir: Root directory containing audio files.
            target_sample_rate: Resample to this rate (default 16 kHz).
            target_length: Pad/trim to this many samples (default 80000 = 5 sec).
            fold: Which fold to use for validation (1-5). Other folds used for training.
            train: If True, return training split; else validation split.
        """
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.train = train

        df = pd.read_csv(csv_path)
        
        # Split by fold: train uses other folds, val uses specified fold
        if train:
            self.df = df[df['fold'] != fold]
        else:
            self.df = df[df['fold'] == fold]
            
        self.categories = sorted(df['category'].unique())
        self.cat2id = {cat: i for i, cat in enumerate(self.categories)}

        # Feature extraction: 80-dim log-Mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=1024,
            win_length=400,
            hop_length=160,
            n_mels=80,
            f_min=20,
            f_max=7800
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # Data augmentation (training only)
        self.aug_transform = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        label_str = row['category']
        label = self.cat2id[label_str]

        # Read audio using soundfile (more robust than torchaudio)
        wav_numpy, sr = sf.read(file_path)
        waveform = torch.from_numpy(wav_numpy).float()
        
        # Ensure [Channels, Time] format
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        # Resample if necessary
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)

        # Pad/trim to target length
        current_len = waveform.shape[1]
        if current_len > self.target_length:
            start = (current_len - self.target_length) // 2
            waveform = waveform[:, start:start+self.target_length]
        elif current_len < self.target_length:
            padding = self.target_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Extract log-Mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)

        # Apply augmentation during training
        if self.train:
            mel_spec = self.aug_transform(mel_spec)

        return mel_spec, label


def get_esc50_loader(csv_path, audio_dir, batch_size=32, num_workers=4, 
                     fold=1, target_length=80000):
    """Get ESC-50 train/val DataLoaders.
    
    Args:
        csv_path: Path to esc50.csv
        audio_dir: Path to audio directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        fold: Fold index for cross-validation (1-5)
        target_length: Audio length in samples
    
    Returns:
        train_loader, val_loader, num_classes
    """
    train_ds = ESC50Dataset(csv_path, audio_dir, fold=fold, train=True, 
                           target_length=target_length)
    val_ds = ESC50Dataset(csv_path, audio_dir, fold=fold, train=False,
                         target_length=target_length)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    num_classes = len(train_ds.categories)
    return train_loader, val_loader, num_classes
