#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.gsc_dataset import get_loader
from models import BCDualNet


class OHEMLabelSmoothingLoss(nn.Module):
    def __init__(self, rate=0.7, smoothing=0.1):
        super().__init__()
        self.rate = rate
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction='none')

    def forward(self, pred, target):
        losses = self.criterion(pred, target)
        keep = max(int(losses.size(0) * self.rate), 1)
        return torch.topk(losses, keep).values.mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_scheduler(optimizer, warmup_epochs, total_epochs, lr_max, lr_min):
    lr_start = lr_max * 0.1

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return lr_start / lr_max + (1.0 - lr_start / lr_max) * epoch / max(warmup_epochs, 1)
        progress = epoch - warmup_epochs
        horizon = max(total_epochs - warmup_epochs, 1)
        return lr_min / lr_max + (1.0 - lr_min / lr_max) * 0.5 * (1.0 + math.cos(math.pi * progress / horizon))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            pred = model(features).argmax(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_one(args, seed):
    set_seed(seed)
    train_loader, val_loader, test_loader, num_classes = get_loader(args.data_dir, args.batch_size, args.num_workers)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = BCDualNet(
        base_c=args.base_c,
        num_classes=num_classes,
        use_dual=True,
        use_tfca=True,
        use_ssn=True,
        use_extra_res=True,
        dual_start_stage=args.dual_start_stage,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs, args.lr_max, args.lr_min)
    criterion = OHEMLabelSmoothingLoss(rate=0.7, smoothing=0.1)
    best_val = 0.0
    best_epoch = 0
    history = []
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f'bracenet_seed{seed}.pth')
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        criterion.rate = 1.0 if epoch < args.warmup_epochs else 0.7
        for features, labels in tqdm(train_loader, desc=f'seed={seed} epoch={epoch + 1}/{args.epochs}', leave=False):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        history.append({'epoch': epoch + 1, 'val_acc': round(val_acc, 4), 'lr': optimizer.param_groups[0]['lr']})
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'epoch {epoch + 1:03d}: val={val_acc:.4f}% best={best_val:.4f}% best_epoch={best_epoch}')

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    result = {
        'seed': seed,
        'best_val_acc': round(best_val, 4),
        'best_epoch': best_epoch,
        'test_acc': round(test_acc, 4),
        'train_time_s': round(time.time() - start, 1),
        'checkpoint': ckpt_path,
        'history': history,
    }
    result_path = os.path.join(args.output_dir, f'result_seed{seed}.json')
    with open(result_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Train BRACE-Net on Google Speech Commands V2')
    parser.add_argument('--data-dir', required=True, help='Path to speech_commands_v2/speech_commands')
    parser.add_argument('--output-dir', default='outputs/gsc_v2')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--base-c', type=int, default=40)
    parser.add_argument('--dual-start-stage', type=int, default=2)
    parser.add_argument('--lr-max', type=float, default=1e-3)
    parser.add_argument('--lr-min', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    summary = {}
    for seed in args.seeds:
        summary[str(seed)] = train_one(args, seed)
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(f'Saved summary: {summary_path}')
