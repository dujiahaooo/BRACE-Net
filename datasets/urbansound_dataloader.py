"""UrbanSound8K urban sound classification dataset loader.

UrbanSound8K is a 10-class urban sound classification dataset with
10-fold cross-validation splits.
"""

import os
import random
import torch
import torchaudio
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass


class UrbanSoundDataset(Dataset):
    """UrbanSound8K Urban Sound Classification Dataset.
    
    Dataset: https://zenodo.org/records/1203745
    Metadata: UrbanSound8K/metadata/UrbanSound8K.csv
    Audio: UrbanSound8K/audio/
    
    10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling,
    engine_idling, gun_shot, jackhammer, siren, street_music.
    10-fold cross-validation: fold 1-10 available.
    """

    def __init__(self, csv_path, audio_root, target_sample_rate=16000,
                 target_length=64000, fold=10, train=True):
        """
        Args:
            csv_path: Path to UrbanSound8K.csv metadata file.
            audio_root: Root directory containing audio (foldX/ subdirs).
            target_sample_rate: Resample to this rate (default 16 kHz).
            target_length: Pad/trim to this many samples (default 64000 = 4 sec).
            fold: Which fold to use for validation (1-10). Other folds used for training.
            train: If True, return training split; else validation split.
        """
        self.audio_root = audio_root
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.train = train
        
        df = pd.read_csv(csv_path)
        
        # Split by fold: train uses other folds, val uses specified fold
        if train:
            self.df = df[df['fold'] != fold]
        else:
            self.df = df[df['fold'] == fold]
            
        # Class mapping for 10 urban sound classes
        self.class_map = {
            'air_conditioner': 0, 'car_horn': 1, 'children_playing': 2,
            'dog_bark': 3, 'drilling': 4, 'engine_idling': 5,
            'gun_shot': 6, 'jackhammer': 7, 'siren': 8, 'street_music': 9
        }

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
        # Audio path: audio/foldX/filename.wav
        file_path = os.path.join(self.audio_root, f"fold{row['fold']}", 
                                row['slice_file_name'])
        label = self.class_map[row['class_name']]

        # Read audio using soundfile (more robust)
        try:
            wav_numpy, sr = sf.read(file_path)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}, returning silence")
            return torch.zeros(1, 80, 401), label

        waveform = torch.from_numpy(wav_numpy).float()
        
        # Ensure [Channels, Time] format
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
            # UrbanSound8K has some stereo files; convert to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary (original may be 44.1k or 48k)
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)

        # Pad/trim to target length
        current_len = waveform.shape[1]
        if current_len > self.target_length:
            if self.train:
                # Random crop during training
                start = random.randint(0, current_len - self.target_length)
            else:
                # Center crop during validation
                start = (current_len - self.target_length) // 2
            waveform = waveform[:, start:start+self.target_length]
        else:
            padding = self.target_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Extract log-Mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)

        # Apply augmentation during training
        if self.train:
            mel_spec = self.aug_transform(mel_spec)

        return mel_spec, label


def get_urbansound_loader(csv_path, audio_root, batch_size=32, num_workers=4,
                          fold=10, target_length=64000):
    """Get UrbanSound8K train/val DataLoaders.
    
    Args:
        csv_path: Path to UrbanSound8K.csv
        audio_root: Path to audio directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        fold: Fold index for cross-validation (1-10)
        target_length: Audio length in samples
    
    Returns:
        train_loader, val_loader, num_classes
    """
    train_ds = UrbanSoundDataset(csv_path, audio_root, fold=fold, train=True,
                                target_length=target_length)
    val_ds = UrbanSoundDataset(csv_path, audio_root, fold=fold, train=False,
                              target_length=target_length)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    num_classes = 10  # UrbanSound8K always has 10 classes
    return train_loader, val_loader, num_classes
