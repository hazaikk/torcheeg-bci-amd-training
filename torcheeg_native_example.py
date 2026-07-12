"""
TorchEEG 原生 API 完整示例
==========================
展示 torcheeg 标准工作流:
  DEAPDataset → transforms → model_selection → model → DataLoader → train/evaluate

功能:
  - 支持所有 5 种模型 (CCNN, EEGNet, TSCeption, FBCNet, FBMSNet)
  - 支持所有 TorchEEG 数据划分方法 (KFold, KFoldGroupbyTrial, ...)
  - 支持离线/在线/标签变换参数
  - 完整交叉验证训练 + 评估

用法:
    # 默认: CCNN, 前2个受试者, 5 epochs, 无CV
    python torcheeg_native_example.py

    # 指定模型 + 数据划分
    python torcheeg_native_example.py --model EEGNet --cv kfold_groupby_trial --n-splits 5

    # 全数据 + 留一法
    python torcheeg_native_example.py --model CCNN --cv leave_one_subject_out --num-subjects 32 --epochs 50

    # GPU + 离线变换参数
    python torcheeg_native_example.py --model FBCNet --gpu --offline bandpass

参考:
    torcheeg/models/cnn/ccnn.py 中的示例用法
"""

import os
import sys
import argparse
import time
import copy
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings('ignore')

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

from utils.training_strategies import EarlyStopping

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms as T
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT
from torcheeg.models import (
    CCNN, EEGNet, TSCeption, FBCNet, FBMSNet,
)
from torcheeg.model_selection import (
    KFoldGroupbyTrial,
    KFold,
    LeaveOneSubjectOut,
    KFoldPerSubjectGroupbyTrial,
)

# ── 可用模型 ──
AVAILABLE_MODELS = ['CCNN', 'EEGNet', 'TSCeption', 'FBCNet', 'FBMSNet']
CV_STRATEGIES = {
    'kfold':                    KFold,
    'kfold_groupby_trial':      KFoldGroupbyTrial,
    'leave_one_subject_out':    LeaveOneSubjectOut,
    'kfold_per_subject':        KFoldPerSubjectGroupbyTrial,
}


def get_model_and_transform(model_name: str, chunk_size: int,
                            offline_type: str = 'auto'):
    """获取模型类及其对应的 transforms

    Args:
        model_name: 模型名
        chunk_size: 窗口样本数
        offline_type: 离线变换类型
            'auto'  = 各模型默认
            'bandpass' = BandSignal (9频带)
            'de'  = BandDifferentialEntropy
            'none' = 无离线变换

    Returns:
        (ModelClass, offline_transform, online_transform, model_kwargs)
    """
    online_transform = T.ToTensor()

    # ── CCNN ──
    if model_name == 'CCNN':
        if offline_type == 'auto' or offline_type == 'de':
            offline = T.Compose([
                T.BandDifferentialEntropy(),
                T.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
            ])
        elif offline_type == 'bandpass':
            offline = T.Compose([
                T.BandSignal(sampling_rate=128, band_dict={
                    f'band{i}': [4*i, 4*(i+1)] for i in range(1, 10)
                }),
                T.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
            ])
        else:
            offline = T.Compose([
                T.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
            ])
        return CCNN, offline, online_transform, dict(
            num_classes=2, in_channels=4, grid_size=(9, 9),
        )

    # ── EEGNet ──
    elif model_name == 'EEGNet':
        if offline_type == 'auto':
            offline = T.Compose([T.To2d()])
        elif offline_type == 'bandpass':
            offline = T.Compose([
                T.BandSignal(sampling_rate=128, band_dict={
                    f'band{i}': [4*i, 4*(i+1)] for i in range(1, 10)
                }),
                T.To2d(),
            ])
        else:
            offline = T.Compose([T.To2d()])
        return EEGNet, offline, online_transform, dict(
            num_classes=2, chunk_size=chunk_size, num_electrodes=32,
        )

    # ── TSCeption ──
    elif model_name == 'TSCeption':
        offline = T.Compose([T.To2d()])
        return TSCeption, offline, online_transform, dict(
            num_classes=2, num_electrodes=32,
        )

    # ── FBCNet ──
    elif model_name == 'FBCNet':
        offline = T.Compose([
            T.BandSignal(sampling_rate=128, band_dict={
                f'band{i}': [4*i, 4*(i+1)] for i in range(1, 10)
            }),
        ])
        return FBCNet, offline, online_transform, dict(
            num_classes=2, num_electrodes=32,
            chunk_size=chunk_size, in_channels=9,
        )

    # ── FBMSNet ──
    elif model_name == 'FBMSNet':
        offline = T.Compose([
            T.BandSignal(sampling_rate=128, band_dict={
                f'band{i}': [4*i, 4*(i+1)] for i in range(1, 10)
            }),
        ])
        return FBMSNet, offline, online_transform, dict(
            num_classes=2, num_electrodes=32,
            chunk_size=chunk_size, in_channels=9,
        )

    else:
        raise ValueError(f'Unknown model: {model_name}')


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
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

    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item() * 100
    return running_loss / len(dataloader), accuracy


def main():
    parser = argparse.ArgumentParser(
        description='TorchEEG 原生 API 完整示例 (含数据划分)')
    parser.add_argument('--data-dir', type=str, default='./data/deap',
                        help='DEAP .dat 文件目录')
    parser.add_argument('--model', type=str, default='CCNN',
                        choices=AVAILABLE_MODELS,
                        help='要演示的模型')
    parser.add_argument('--chunk-size', type=int, default=128,
                        choices=[128, 256],
                        help='时间窗口样本数 (128=1s, 256=2s)')
    parser.add_argument('--overlap', type=int, default=0,
                        help='窗口重叠样本数 (0=无重叠)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early-patience', type=int, default=15,
                        help='早停容忍轮数 (默认 15, 0=不早停)')
    parser.add_argument('--gpu', action='store_true',
                        help='使用 GPU')

    # 数据划分参数
    parser.add_argument('--cv', type=str, default=None,
                        choices=list(CV_STRATEGIES.keys()) + [None],
                        help='数据划分策略 (None=不划分, 直接训练)')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='KFold 折数')

    # 受试者限制
    parser.add_argument('--num-subjects', type=int, default=2,
                        help='使用的受试者数 (取前 N 个, 默认 2 个快速演示)')

    # 离线变换选择
    parser.add_argument('--offline', type=str, default='auto',
                        choices=['auto', 'de', 'bandpass', 'none'],
                        help='离线变换类型 (auto=各模型默认)')

    # 标签处理
    parser.add_argument('--label', type=str, default='valence',
                        choices=['valence', 'arousal', 'dominance', 'liking'],
                        help='分类目标维度')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='二分类阈值')
    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f'[Device] {device}')
    if device == 'cuda':
        print(f'         GPU: {torch.cuda.get_device_name(0)}')

    # ════════════════════════════════════════════════
    # 1. 模型 + transforms
    # ════════════════════════════════════════════════
    print(f'\n{"="*60}')
    print(f'  Step 1: 模型 & transforms')
    print(f'  Model:       {args.model}')
    print(f'  Window:      {args.chunk_size}pt ({args.chunk_size/128:.0f}s)')
    print(f'  Offline:     {args.offline}')
    print(f'  Label:       {args.label} > {args.threshold}')
    print(f'  CV:          {args.cv or "none (single train/val split)"}')
    print(f'{"="*60}')

    ModelClass, offline_transform, online_transform, model_kwargs = \
        get_model_and_transform(args.model, args.chunk_size, args.offline)

    label_transform = T.Compose([
        T.Select(args.label),
        T.Binary(args.threshold),
    ])

    # ════════════════════════════════════════════════
    # 2. 数据集
    # ════════════════════════════════════════════════
    print(f'\n{"="*60}')
    print(f'  Step 2: 加载 DEAPDataset')
    print(f'{"="*60}')

    t0 = time.time()
    dataset = DEAPDataset(
        root_path=args.data_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        num_channel=32,
        offline_transform=offline_transform,
        online_transform=online_transform,
        label_transform=label_transform,
        num_worker=2,
        verbose=True,
        io_path=f'./_deap_cache_example_{args.model}_{args.chunk_size}_{args.offline}',
        io_mode='pickle',
    )
    print(f'  Dataset: {len(dataset)} samples ({time.time()-t0:.1f}s)')

    # 限制受试者数 (演示用快速加载)
    if args.num_subjects and args.num_subjects < 32:
        keep_idx = []
        for i in range(len(dataset)):
            subj = dataset[i][1].item()  # label
            subj_id = i // (40 * ((8064 - args.chunk_size) // args.chunk_size + 1))
            if subj_id < args.num_subjects:
                keep_idx.append(i)
        dataset = Subset(dataset, keep_idx)
        print(f'  Subset:  {len(dataset)} samples ({args.num_subjects} subjects)')

    # ════════════════════════════════════════════════
    # 3. 数据划分 + 交叉验证
    # ════════════════════════════════════════════════
    print(f'\n{"="*60}')

    if args.cv:
        # ── TorchEEG model_selection 划分 ──
        print(f'  Step 3: TorchEEG 数据划分 — {args.cv}')
        print(f'{"="*60}')

        CVClass = CV_STRATEGIES[args.cv]
        cv = CVClass(n_splits=args.n_splits, shuffle=True, random_state=42)

        fold_results = []
        for fold_idx, (train_dataset, val_dataset) in enumerate(cv.split(dataset)):
            n_folds = args.n_splits if args.cv != 'leave_one_subject_out' else min(32, args.num_subjects)
            print(f'\n  --- Fold {fold_idx+1}/{n_folds} ---')
            print(f'      Train: {len(train_dataset)} | Val: {len(val_dataset)}')

            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=0)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=0)

            # 模型
            model = ModelClass(**model_kwargs).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            early_stopping = EarlyStopping(
                patience=args.early_patience, mode='max',
                verbose=True) if args.early_patience > 0 else None
            best_val_acc = 0.0
            best_model_state = None

            for epoch in range(args.epochs):
                t_epoch = time.time()
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = evaluate(
                    model, val_loader, criterion, device)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())

                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
                    print(f'      Ep {epoch+1:3d}/{args.epochs} | '
                          f'T_loss:{train_loss:.4f} T_acc:{train_acc:.2f}% | '
                          f'V_loss:{val_loss:.4f} V_acc:{val_acc:.2f}% | '
                          f'{time.time()-t_epoch:.1f}s')

                # 早停检查
                if early_stopping and early_stopping(val_acc, epoch):
                    print(f'      >>> Early stop @ epoch {epoch+1}')
                    break

            fold_results.append(best_val_acc)
            print(f'      >>> Fold {fold_idx+1} best val acc: {best_val_acc:.2f}%')

            # 清理 GPU 显存
            if device != 'cpu':
                del model
                torch.cuda.empty_cache()

        # ── CV 汇总 ──
        print(f'\n{"="*60}')
        print(f'  CV Results ({args.cv}, {len(fold_results)} folds):')
        for i, acc in enumerate(fold_results):
            print(f'    Fold {i+1}: {acc:.2f}%')
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        print(f'    Mean ± Std: {mean_acc:.2f}% ± {std_acc:.2f}%')
        print(f'{"="*60}')

    else:
        # ── 无 CV: 单次 train/val 划分 (80/20) ──
        print(f'  Step 3: 单次划分 (80% train / 20% val)')
        print(f'{"="*60}')

        n_total = len(dataset)
        indices = list(range(n_total))
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(n_total * 0.8)
        train_dataset = Subset(dataset, indices[:split])
        val_dataset = Subset(dataset, indices[split:])

        print(f'  Train: {len(train_dataset)} | Val: {len(val_dataset)}')

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=0)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=0)

        # 模型
        model = ModelClass(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        early_stopping = EarlyStopping(
            patience=args.early_patience, mode='max',
            verbose=True) if args.early_patience > 0 else None
        best_val_acc = 0.0

        for epoch in range(args.epochs):
            t_epoch = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            print(f'  Ep {epoch+1:3d}/{args.epochs} | '
                  f'T_loss:{train_loss:.4f} T_acc:{train_acc:.2f}% | '
                  f'V_loss:{val_loss:.4f} V_acc:{val_acc:.2f}% | '
                  f'{time.time()-t_epoch:.1f}s')

            # 早停检查
            if early_stopping and early_stopping(val_acc, epoch):
                print(f'  >>> Early stop @ epoch {epoch+1}')
                break

        print(f'\n  Best val acc: {best_val_acc:.2f}%')

    # ════════════════════════════════════════════════
    # 总结
    # ════════════════════════════════════════════════
    print(f'\n{"="*60}')
    print(f'  完成!')
    print(f'{"="*60}')
    print(f'\nAPI 使用总结:')
    print(f'  from torcheeg.datasets import DEAPDataset')
    print(f'  from torcheeg import transforms as T')
    print(f'  from torcheeg.models import {args.model}')
    print(f'  from torcheeg.model_selection import {args.cv or "KFoldGroupbyTrial"}')
    print(f'')
    print(f'  dataset = DEAPDataset(root_path=...,')
    print(f'                        offline_transform=...,')
    print(f'                        online_transform=T.ToTensor(),')
    print(f'                        label_transform=...)')
    print(f'  model = {args.model}(num_classes=2, ...)')
    print(f'  cv = {args.cv or "KFoldGroupbyTrial"}(n_splits=5)')
    print(f'  for train_ds, val_ds in cv.split(dataset):')
    print(f'      loader = DataLoader(train_ds, batch_size=64, shuffle=True)')
    print(f'      for x, y in loader:')
    print(f'          output = model(x.to(device))')
    print(f'          ...')


if __name__ == '__main__':
    main()
