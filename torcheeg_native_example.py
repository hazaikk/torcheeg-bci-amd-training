"""
TorchEEG 原生 API 使用示例
==========================
展示 torcheeg 标准工作流:
  DEAPDataset → transforms → model → DataLoader → train/evaluate

参考:
  torcheeg/models/cnn/ccnn.py 中的示例用法

用法:
    python torcheeg_native_example.py

要求:
    DEAP 原始数据 (.dat 文件) 在 ./data/deap/ 或通过 --data-dir 指定
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── 兼容性修复 ──
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)

import scipy.signal
from scipy.signal.windows import hann, hamming, blackman
scipy.signal.hann = hann
scipy.signal.hamming = hamming
scipy.signal.blackman = blackman

from utils.fixes import apply_all_fixes
apply_all_fixes()

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms as T
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT
from torcheeg.models import (
    CCNN, EEGNet, TSCeption, FBCNet, FBMSNet,
)


def main():
    parser = argparse.ArgumentParser(
        description='TorchEEG 原生 API 使用示例')
    parser.add_argument('--data-dir', type=str, default='./data/deap',
                        help='DEAP .dat 文件目录')
    parser.add_argument('--model', type=str, default='CCNN',
                        choices=['CCNN', 'EEGNet', 'TSCeption', 'FBCNet', 'FBMSNet'],
                        help='要演示的模型')
    parser.add_argument('--chunk-size', type=int, default=128,
                        help='时间窗口样本数 (128=1s, 256=2s)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数 (示例设小, 快速演示)')
    parser.add_argument('--gpu', action='store_true',
                        help='使用 GPU')
    parser.add_argument('--num-subjects', type=int, default=2,
                        help='使用的受试者数 (示例取前 N 个, 加快速度)')
    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f'[Device] {device}')
    if device == 'cuda':
        print(f'         GPU: {torch.cuda.get_device_name(0)}')

    # ════════════════════════════════════════════════
    # 1. 数据集加载 + transforms
    # ════════════════════════════════════════════════
    print(f'\n{"="*60}')
    print(f'  Step 1: 加载 DEAPDataset')
    print(f'  Model:   {args.model}')
    print(f'  Window:  {args.chunk_size}pt ({args.chunk_size/128:.0f}s)')
    print(f'  Subjects: {args.num_subjects} (前 {args.num_subjects} 个)')
    print(f'{"="*60}')

    # ── 不同模型的 transform ──
    if args.model == 'CCNN':
        # CCNN: (32, chunk_size) → BandDifferentialEntropy(4频带) → ToGrid(9×9) → ToTensor
        offline_transform = T.Compose([
            T.BandDifferentialEntropy(),
            T.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
        ])
        online_transform = T.ToTensor()
        model_kwargs = dict(
            num_classes=2,
            in_channels=4,
            grid_size=(9, 9),
        )

    elif args.model == 'EEGNet':
        # EEGNet: (32, chunk_size) → To2d → ToTensor
        offline_transform = T.Compose([
            T.To2d(),
        ])
        online_transform = T.ToTensor()
        model_kwargs = dict(
            num_classes=2,
            chunk_size=args.chunk_size,
            num_electrodes=32,
        )

    elif args.model == 'TSCeption':
        offline_transform = T.Compose([
            T.To2d(),
        ])
        online_transform = T.ToTensor()
        model_kwargs = dict(
            num_classes=2,
            num_electrodes=32,
        )

    elif args.model == 'FBCNet':
        offline_transform = T.Compose([
            T.BandSignal(sampling_rate=128, band_dict={
                f'band{i}': [4*i, 4*(i+1)] for i in range(1, 10)
            }),
        ])
        online_transform = T.ToTensor()
        model_kwargs = dict(
            num_classes=2,
            num_electrodes=32,
            chunk_size=args.chunk_size,
            in_channels=9,
        )

    elif args.model == 'FBMSNet':
        offline_transform = T.Compose([
            T.BandSignal(sampling_rate=128, band_dict={
                f'band{i}': [4*i, 4*(i+1)] for i in range(1, 10)
            }),
        ])
        online_transform = T.ToTensor()
        model_kwargs = dict(
            num_classes=2,
            num_electrodes=32,
            chunk_size=args.chunk_size,
            in_channels=9,
        )

    # ── 创建数据集 ──
    t0 = time.time()
    dataset = DEAPDataset(
        root_path=args.data_dir,
        chunk_size=args.chunk_size,
        overlap=0,                         # 无重叠窗口
        num_channel=32,
        offline_transform=offline_transform,
        online_transform=online_transform,
        label_transform=T.Compose([
            T.Select('valence'),
            T.Binary(5.0),                 # valence > 5 → high (1), else → low (0)
        ]),
        num_worker=2,
        verbose=True,
        io_path=f'./_deap_cache_example_{args.model}_{args.chunk_size}',
        io_mode='pickle',
    )
    print(f'  Dataset created: {len(dataset)} samples ({time.time()-t0:.1f}s)')

    # 限制受试者数 (演示用)
    if args.num_subjects and args.num_subjects < 32:
        subjects = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            subj = (i // 40) + 1  # 按 subject 分组 (每 subject 40 trials × n_windows)
            if subj not in subjects:
                subjects.append(subj)
            if len(subjects) > args.num_subjects and all(s <= args.num_subjects for s in subjects):
                break
        # 只保留前 num_subjects 个受试者的样本
        from torch.utils.data import Subset
        keep_idx = []
        for i in range(len(dataset)):
            subj = (i // 40) + 1
            if subj <= args.num_subjects:
                keep_idx.append(i)
        dataset = Subset(dataset, keep_idx)
        print(f'  Subset: {len(dataset)} samples ({args.num_subjects} subjects)')

    # ════════════════════════════════════════════════
    # 2. DataLoader + 模型创建
    # ════════════════════════════════════════════════
    print(f'\n{"="*60}')
    print(f'  Step 2: 创建 DataLoader + 模型')
    print(f'{"="*60}')

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )

    # 创建模型
    ModelClass = {
        'CCNN': CCNN, 'EEGNet': EEGNet,
        'TSCeption': TSCeption, 'FBCNet': FBCNet,
        'FBMSNet': FBMSNet,
    }[args.model]
    model = ModelClass(**model_kwargs)
    model = model.to(device)
    print(f'  Model: {args.model}')
    print(f'         Params: {sum(p.numel() for p in model.parameters()):,}')

    # ════════════════════════════════════════════════
    # 3. 前向传播示例 (单 batch)
    # ════════════════════════════════════════════════
    print(f'\n{"="*60}')
    print(f'  Step 3: 前向传播 (1 batch)')
    print(f'{"="*60}')

    x, y = next(iter(dataloader))
    x, y = x.to(device), y.to(device).long()
    print(f'  Input:  {tuple(x.shape)}  ({x.dtype})')
    print(f'  Label:  {tuple(y.shape)}  ({y.tolist()[:10]})')
    print(f'  Class distribution: {y.sum().item()}/{len(y)} positive')

    with torch.no_grad():
        model.eval()
        output = model(x)
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean().item()
    print(f'  Output: {tuple(output.shape)}')
    print(f'  Acc:    {acc*100:.1f}%')

    # ════════════════════════════════════════════════
    # 4. 训练循环 (演示)
    # ════════════════════════════════════════════════
    print(f'\n{"="*60}')
    print(f'  Step 4: 训练 {args.epochs} epochs')
    print(f'{"="*60}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t_epoch = time.time()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100.0 * correct / total
        epoch_loss = running_loss / len(dataloader)
        epoch_time = time.time() - t_epoch

        print(f'  Epoch {epoch+1:2d}/{args.epochs} | '
              f'Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | '
              f'{epoch_time:.1f}s')

    # ════════════════════════════════════════════════
    # 5. 总结
    # ════════════════════════════════════════════════
    print(f'\n{"="*60}')
    print(f'  完成!')
    print(f'  模型: {args.model}')
    print(f'  最终准确率: {epoch_acc:.2f}%')
    print(f'{"="*60}')
    print(f'\n用法总结:')
    print(f'  from torcheeg.datasets import DEAPDataset')
    print(f'  from torcheeg import transforms as T')
    print(f'  from torcheeg.models import {args.model}')
    print(f'  from torch.utils.data import DataLoader')
    print(f'')
    print(f'  dataset = DEAPDataset(root_path=...,')
    print(f'                        offline_transform=...,')
    print(f'                        online_transform=T.ToTensor(),')
    print(f'                        label_transform=...)')
    print(f'  model = {args.model}(num_classes=2, ...)')
    print(f'  x, y = next(iter(DataLoader(dataset, batch_size=64)))')
    print(f'  model(x)')


if __name__ == '__main__':
    main()
