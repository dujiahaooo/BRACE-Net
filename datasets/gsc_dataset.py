import os
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class GSC_Dataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('_')]
        self.labels.sort()
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

        val_list_file = os.path.join(data_dir, 'validation_list.txt')
        test_list_file = os.path.join(data_dir, 'testing_list.txt')
        val_files = set()
        test_files = set()

        if os.path.exists(val_list_file):
            with open(val_list_file, 'r', encoding='utf-8') as handle:
                val_files = {line.strip() for line in handle if line.strip()}
        if os.path.exists(test_list_file):
            with open(test_list_file, 'r', encoding='utf-8') as handle:
                test_files = {line.strip() for line in handle if line.strip()}

        self.file_list = []
        for label in self.labels:
            label_dir = os.path.join(data_dir, label)
            for wav_file in os.listdir(label_dir):
                if not wav_file.endswith('.wav'):
                    continue
                rel_path = f'{label}/{wav_file}'
                full_path = os.path.join(label_dir, wav_file)
                if split == 'val' and rel_path in val_files:
                    self.file_list.append((full_path, self.label_to_idx[label]))
                elif split == 'test' and rel_path in test_files:
                    self.file_list.append((full_path, self.label_to_idx[label]))
                elif split == 'train' and rel_path not in val_files and rel_path not in test_files:
                    self.file_list.append((full_path, self.label_to_idx[label]))

        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=512, hop_length=160)
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        self.train_mask = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35),
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]
        try:
            waveform_np, _ = sf.read(filepath)
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2:
                waveform = waveform.t()
        except Exception:
            waveform = torch.zeros(1, 16000)

        if waveform.shape[1] < 16000:
            waveform = F.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]

        log_mel = self.db_transform(self.mel_transform(waveform))
        if self.split == 'train':
            log_mel = self.train_mask(log_mel)
        return log_mel, label


def get_loader(data_dir, batch_size=128, num_workers=4):
    train_ds = GSC_Dataset(data_dir, 'train')
    val_ds = GSC_Dataset(data_dir, 'val')
    test_ds = GSC_Dataset(data_dir, 'test')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, len(train_ds.labels)
