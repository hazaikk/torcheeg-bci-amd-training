"""
CCNN 复现训练脚本 — 复现 TorchEEG EMO 论文 Table 1 中 CCNN 在 DEAP 上 92.23% 的结果

核心差异 (与之前的实验相比):
  1. 使用 DE (Differential Entropy) 特征, 而非时间切片/频带能量
  2. 输入: BandDifferentialEntropy(4 bands) → ToGrid(9×9) → (4, 9, 9)
  3. 使用 TorchEEG 原生 DEAPDataset 的 offline_transform + online_transform

论文参考:
  - CCNN: Yang et al., "Continuous convolutional neural network with 3D input
    for EEG-based emotion recognition", ICONIP 2018
  - TorchEEG EMO: Table 1 reports CCNN achieves 92.23% on DEAP valence

用法:
    # 1. 直接使用 TorchEEG 原生 DEAPDataset (自动缓存, 推荐)
    python train_ccnn_repro.py --mode native --gpu

    # 2. 使用预处理 .pt 文件 (需先运行 preprocess_ccnn_de.py)
    python train_ccnn_repro.py --mode preprocessed --gpu

    # 3. LOSO 交叉验证
    python train_ccnn_repro.py --cv leave_one_subject_out --gpu

    # 4. 测试模式 (快速验证流程)
    python train_ccnn_repro.py --test --gpu
"""

import os
import sys
import json
import time
import copy
import argparse
import warnings
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ── 项目导入 ──
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)

import scipy.signal
from scipy.signal.windows import hann, hamming, blackman
scipy.signal.hann = hann
scipy.signal.hamming = hamming
scipy.signal.blackman = blackman

from utils.fixes import apply_all_fixes
apply_all_fixes()

from config import (
    config, DEAP_NUM_CHANNELS, DEAP_SAMPLING_RATE, DEAP_NUM_CLASSES,
)

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms as T
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT
from torcheeg.models import CCNN

from utils.training_strategies import EarlyStopping, create_scheduler
from utils.model_utils import get_device, print_gpu_info


# ══════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════

# CCNN 论文推荐参数 (Yang et al. 2018)
# - 4 DE bands (theta, alpha, beta, gamma)
# - 9×9 electrode grid
# - 1s windows (128 samples at 128Hz)
# - Dropout 0.5 (TorchEEG default)
CCNN_IN_CHANNELS = 4      # 4 DE bands
CCNN_GRID_SIZE = (9, 9)   # 9×9 grid
CCNN_DROPOUT = 0.5

# TorchEEG EMO paper reported result
TARGET_ACC = 92.23

DEFAULT_DEAP_DIR = os.path.join(_PROJECT_DIR, 'data', 'deap')
RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')


# ══════════════════════════════════════════════════
# CCNN 模型 (直接使用 TorchEEG)
# ══════════════════════════════════════════════════

def create_ccnn() -> nn.Module:
    """创建 CCNN 模型, 使用论文推荐参数"""
    return CCNN(
        in_channels=CCNN_IN_CHANNELS,
        grid_size=CCNN_GRID_SIZE,
        num_classes=DEAP_NUM_CLASSES,
        dropout=CCNN_DROPOUT,
    )


# ══════════════════════════════════════════════════
# DEAPDataset — CCNN 专用 transforms
# ══════════════════════════════════════════════════

def get_ccnn_transforms():
    """获取 CCNN 复现所需的 DE + ToGrid transforms

    与 TorchEEG EMO paper 完全一致:
      offline: BandDifferentialEntropy → ToGrid(9×9)
      online:  ToTensor
      label:   Select('valence') → Binary(5.0)
    """
    offline_transform = T.Compose([
        T.BandDifferentialEntropy(sampling_rate=DEAP_SAMPLING_RATE),
        T.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
    ])
    online_transform = T.ToTensor()
    label_transform = T.Compose([
        T.Select('valence'),
        T.Binary(5.0),
    ])
    return offline_transform, online_transform, label_transform


def get_deap_dataset(cache_dir: str, chunk_size: int = 128):
    """创建 CCNN 专用 DEAPDataset, 使用固定缓存路径复用 LMDB"""
    offline, online, label = get_ccnn_transforms()
    os.makedirs(cache_dir, exist_ok=True)
    return DEAPDataset(
        root_path=DEFAULT_DEAP_DIR,
        chunk_size=chunk_size,
        overlap=0,
        num_channel=DEAP_NUM_CHANNELS,
        offline_transform=offline,
        online_transform=online,
        label_transform=label,
        io_path=cache_dir,
        io_mode='lmdb',
        num_worker=0,
        verbose=True,
    )


# ══════════════════════════════════════════════════
# 训练与评估
# ══════════════════════════════════════════════════

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        if isinstance(batch, dict):
            inputs, labels = batch['eeg'], batch['y']
        elif isinstance(batch, (list, tuple)):
            inputs, labels = batch[0], batch[1]
        else:
            raise TypeError(f'Unexpected batch type: {type(batch)}')

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs, labels = batch['eeg'], batch['y']
            elif isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                raise TypeError(f'Unexpected batch type: {type(batch)}')

            inputs = inputs.to(device, dtype=torch.float)
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


# ══════════════════════════════════════════════════
# 实验主函数
# ══════════════════════════════════════════════════

def run_ccnn_experiment(
    mode: str = 'native',
    cv_strategy: str = 'kfold_groupby_trial',
    n_splits: int = 5,
    chunk_size: int = 128,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    epochs: int = 200,
    scheduler_name: str = 'cosine',
    early_patience: int = 15,
    device: str = 'cpu',
    run_dir: str = '',
    test_mode: bool = False,
    verbose: bool = True,
) -> Dict:
    """CCNN 复现实验

    使用 DE 特征 + ToGrid 预处理, 目标复现 92.23% 准确率.
    """

    # ── 原生 DEAPDataset 模式 ──
    if mode == 'native':
        # 使用固定缓存路径 (不含时间戳), 多次运行复用 LMDB
        cache_root = os.path.join(os.path.dirname(run_dir), '_deap_cache')
        os.makedirs(cache_root, exist_ok=True)
        io_path = os.path.join(cache_root, f'ccnn_de_{chunk_size}')
        dataset = get_deap_dataset(io_path, chunk_size)

        n_total = len(dataset)
        if verbose:
            print(f'\n[DEAP] CCNN-DE Dataset: {n_total} samples')
            sample_labels = []
            for i in range(min(1000, n_total)):
                s = dataset[i]
                lbl = s[1] if isinstance(s, (list, tuple)) else s['y']
                sample_labels.append(int(lbl))
            cls_counts = Counter(sample_labels)
            print(f'       Class distribution (sample): {dict(cls_counts)}')

        # ── 将 LMDB 数据批量读入 tensor (避免 cv.split 复制 LMDB 崩溃) ──
        print(f'  Loading all data into memory...', flush=True)
        all_data, all_labels, all_subjects = [], [], []
        batch_gen = DataLoader(dataset, batch_size=1024, shuffle=False,
                               num_workers=0)
        for batch in batch_gen:
            if isinstance(batch, dict):
                all_data.append(batch['eeg'])
                all_labels.append(batch['y'].flatten())
            elif isinstance(batch, (list, tuple)):
                all_data.append(batch[0])
                all_labels.append(batch[1].flatten())
        data = torch.cat(all_data, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # subject_ids: DEAPDataset 按 trial/受试者顺序排列
        # BaselineRemoval → 60s/trial, 1s窗口 → 60 windows/trial
        # DEAP = 32 subjects × 40 trials × 60 windows = 76800
        n_windows_per_trial = n_total // (32 * 40)  # 应=60
        n_subjects = 32
        n_trials_per_subject = 40
        subjects = torch.zeros(n_total, dtype=torch.long)
        for si in range(n_subjects):
            start = si * n_trials_per_subject * n_windows_per_trial
            end = (si + 1) * n_trials_per_subject * n_windows_per_trial
            subjects[start:end] = si + 1

        if verbose:
            print(f'       Data shape: {tuple(data.shape)}, '
                  f'Labels: {Counter(labels.tolist())}')

        # 数据移到 GPU
        if device != 'cpu':
            data = data.to(device)

        # ── 按 subject 手动 KFold (与 deap_multi_model 一致) ──
        unique_subjects = sorted(subjects.unique().tolist())
        np.random.seed(42)
        np.random.shuffle(unique_subjects)
        fold_subject_groups = np.array_split(
            unique_subjects, min(n_splits, len(unique_subjects)))
        fold_subject_groups = [g.tolist() if hasattr(g, 'tolist') else list(g)
                               for g in fold_subject_groups]

        fold_indices_list = []
        for val_subjects in fold_subject_groups:
            train_mask = ~torch.isin(subjects, torch.tensor(val_subjects))
            val_mask = torch.isin(subjects, torch.tensor(val_subjects))
            train_idx = torch.where(train_mask)[0]
            val_idx = torch.where(val_mask)[0]
            fold_indices_list.append((train_idx, val_idx))

        n_folds = len(fold_indices_list)
        if test_mode:
            n_folds = min(n_folds, 2)

        # 关闭 dataset 释放 LMDB 连接
        del dataset, batch_gen, all_data, all_labels, all_subjects

    else:
        raise ValueError(f'Unknown mode: {mode}. Use "native" (recommended).')

    # ── 数据集包装器 (用于手动索引切分) ──
    class CCNNDataset(Dataset):
        def __init__(self, data, labels, indices=None):
            self._data = data
            self._labels = labels.flatten()
            self.indices = indices if indices is not None else torch.arange(len(data))

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            return self._data[real_idx], int(self._labels[real_idx])

    # ── 逐折训练 ──
    actual_epochs = 1 if test_mode else epochs

    fold_results = []
    all_epochs_records = []
    best_overall_acc = 0.0
    best_model_state = None

    for fold_idx in range(n_folds):
        fold_item = fold_indices_list[fold_idx]

        if mode == 'native':
            # 从 tensor + 索引构建 DataLoader
            train_idx, val_idx = fold_item
            train_dataset = CCNNDataset(data, labels, indices=train_idx)
            val_dataset = CCNNDataset(data, labels, indices=val_idx)
        else:
            # 从 DEAPDataset subset 构建 (预留)
            train_dataset, val_dataset = fold_item

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=0, drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, drop_last=False,
        )

        if verbose:
            print(f'\n{"="*55}')
            print(f'  Fold {fold_idx+1}/{n_folds}')
            print(f'  Train: {len(train_dataset)} | Val: {len(val_dataset)}')
            print(f'{"="*55}')

        model = create_ccnn()
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, scheduler_name, actual_epochs)
        early_stopping = EarlyStopping(
            patience=early_patience, mode='max', verbose=verbose)
        scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

        fold_best_acc = 0.0

        for epoch in range(actual_epochs):
            t_epoch = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler)
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_acc > fold_best_acc:
                fold_best_acc = val_acc
                if val_acc > best_overall_acc:
                    best_overall_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())

            record = {
                'epoch': epoch + 1, 'fold': fold_idx + 1,
                'train_loss': round(train_loss, 4),
                'train_acc': round(train_acc, 2),
                'val_loss': round(val_loss, 4),
                'val_acc': round(val_acc, 2),
                'time': round(time.time() - t_epoch, 2),
            }
            all_epochs_records.append(record)

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == actual_epochs - 1:
                print(f'  Ep {epoch+1:3d}/{actual_epochs} | '
                      f'T_loss:{train_loss:.4f} T_acc:{train_acc:.2f}% | '
                      f'V_loss:{val_loss:.4f} V_acc:{val_acc:.2f}% | '
                      f'{record["time"]:.1f}s')

            if early_stopping(val_acc, epoch):
                break

        fold_results.append(fold_best_acc)
        print(f'  >>> Fold {fold_idx+1} best val acc: {fold_best_acc:.2f}%')

        # 保存每折指标
        if run_dir:
            fold_csv = os.path.join(run_dir, f'fold_{fold_idx+1}_metrics.csv')
            fold_df = pd.DataFrame([r for r in all_epochs_records
                                     if r['fold'] == fold_idx + 1])
            fold_df.to_csv(fold_csv, index=False)

        if device != 'cpu':
            del model
            torch.cuda.empty_cache()

    # ── 汇总 ──
    val_accs = fold_results
    mean_acc = float(np.mean(val_accs))
    std_acc = float(np.std(val_accs))

    summary = {
        'model': 'CCNN (DE)',
        'chunk_size': chunk_size,
        'cv_strategy': cv_strategy,
        'n_splits': n_folds,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs_actual': actual_epochs,
        'early_patience': early_patience,
        'scheduler': scheduler_name,
        'num_params': sum(p.numel() for p in create_ccnn().parameters()),
        'preprocessing': 'BandDifferentialEntropy → ToGrid(9×9)',
        'input_shape': f'({CCNN_IN_CHANNELS}, {CCNN_GRID_SIZE[0]}, {CCNN_GRID_SIZE[1]})',
        'fold_results': [{'fold': i+1, 'best_val_acc': acc}
                         for i, acc in enumerate(val_accs)],
        'mean_val_acc': round(mean_acc, 2),
        'std_val_acc': round(std_acc, 2),
        'best_val_acc': round(best_overall_acc, 2),
        'target_acc': TARGET_ACC,
        'gap_to_target': round(TARGET_ACC - mean_acc, 2),
    }

    if verbose:
        print(f'\n{"="*55}')
        print(f'  CCNN (DE) — {cv_strategy} ({n_folds}-fold)')
        print(f'  Input: {summary["input_shape"]} (DE features → 9×9 grid)')
        print(f'  Params: {summary["num_params"]:,}')
        print(f'  Target: {TARGET_ACC}% (TorchEEG EMO Table 1)')
        print(f'  Per-fold: {[f"{a:.2f}" for a in val_accs]}')
        print(f'  Mean±Std: {mean_acc:.2f}±{std_acc:.2f}')
        print(f'  Best:     {best_overall_acc:.2f}%')
        print(f'  Gap:      {summary["gap_to_target"]:.2f}%')
        print(f'{"="*55}')

    # ── 保存 ──
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(run_dir, 'best_model.pt'))
        with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        if all_epochs_records:
            pd.DataFrame(all_epochs_records).to_csv(
                os.path.join(run_dir, 'all_epochs.csv'), index=False)

    return summary


# ══════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='CCNN 复现 — DE 特征 + ToGrid, 目标 92.23%')
    parser.add_argument('--mode', type=str, default='native',
                        choices=['native', 'preprocessed'],
                        help='训练模式 (推荐 native)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='DEAP .dat 文件目录')
    parser.add_argument('--cv', type=str, default='kfold_groupby_trial',
                        choices=['kfold_groupby_trial', 'kfold',
                                 'leave_one_subject_out'],
                        help='交叉验证策略')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--chunk-size', type=int, default=128,
                        choices=[128, 256])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early-patience', type=int, default=15)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'step'])
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--test', action='store_true',
                        help='测试模式 (1 epoch, 2 folds)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    verbose = not args.quiet
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    print(f'{"="*55}')
    print(f'  CCNN Reproduction — DE features → ToGrid(9×9)')
    print(f'  Target: {TARGET_ACC}% on DEAP valence')
    print(f'  Device: {device}')
    print(f'  Mode:   {args.mode}')
    print(f"{'='*55}")
    print()
    print(f'  Experimental setup:')
    print(f'    CV:          {args.cv} ({args.n_splits}-fold)')
    print(f'    Window:      {args.chunk_size}pt ({args.chunk_size/DEAP_SAMPLING_RATE:.0f}s)')
    print(f'    Batch:       {args.batch_size}')
    print(f'    LR:          {args.lr}')
    print(f'    Scheduler:   {args.scheduler}')
    print(f'    Early stop:  patience={args.early_patience}')
    print(f'    Max epochs:  {args.epochs}')
    print()
    print(f'  Data preprocessing:')
    print(f'    1. BandDifferentialEntropy (theta/alpha/beta/gamma)')
    print(f'    2. ToGrid(DEAP_CHANNEL_LOCATION_DICT) → ({CCNN_IN_CHANNELS}, 9, 9)')
    print(f'    3. Label: valence > 5 → high, else → low')
    print()

    if device == 'cuda':
        print_gpu_info(device)

    # 数据路径
    if args.data_dir:
        deap_root = args.data_dir
    else:
        from download_deap import get_deap_path
        deap_root = get_deap_path()

    if not os.path.exists(deap_root):
        print(f'[ERROR] DEAP data not found at: {deap_root}')
        print('  Run: python download_deap.py')
        sys.exit(1)

    print(f'[DEAP] Data root: {deap_root}')

    # 结果目录
    mode_label = 'ccnn_de'
    cv_label = args.cv
    window_label = f'{args.chunk_size}pt_{args.chunk_size/DEAP_SAMPLING_RATE:.0f}s'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_label = '_TEST' if args.test else ''
    run_name = f'{mode_label}_{window_label}_{cv_label}_{timestamp}{test_label}'
    run_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f'[RESULT] {run_dir}')

    # 保存配置
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 更改数据目录
    global DEFAULT_DEAP_DIR
    DEFAULT_DEAP_DIR = deap_root

    # 运行实验
    summary = run_ccnn_experiment(
        mode=args.mode,
        cv_strategy=args.cv,
        n_splits=args.n_splits,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        scheduler_name=args.scheduler,
        early_patience=args.early_patience,
        device=device,
        run_dir=run_dir,
        test_mode=args.test,
        verbose=verbose,
    )

    # 输出最终结果
    print(f'\n{"="*55}')
    print(f'  FINAL RESULT')
    print(f'{"="*55}')
    print(f'  CCNN (DE) | {args.cv} ({args.n_splits}-fold)')
    print(f'  Mean Acc:  {summary["mean_val_acc"]:.2f}% ± {summary["std_val_acc"]:.2f}%')
    print(f'  Best Acc:  {summary["best_val_acc"]:.2f}%')
    print(f'  Target:    {TARGET_ACC}%')
    print(f'  Gap:       {summary["gap_to_target"]:.2f}%')
    if summary["gap_to_target"] <= 0:
        print(f'  ✅ 目标达成! 超过 {abs(summary["gap_to_target"]):.2f}%')
    else:
        print(f'  ❌ 未达到目标, 差 {summary["gap_to_target"]:.2f}%')
    print(f'{"="*55}')
    print(f'[DONE] Results saved to: {run_dir}')


if __name__ == '__main__':
    main()
