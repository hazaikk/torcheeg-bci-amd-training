"""
DEAP 数据集训练脚本 — 复现 TorchEEG EMO 论文 Table 1
=====================================================
训练 EEGNet / TSCeption / FBCNet / FBMSNet / CCNN 五个模型
在 DEAP 数据集上的 Valence 二分类准确率。

支持两种模式:
  1. 原生模式: 使用 TorchEEG DEAPDataset + model_selection (自动缓存)
  2. 预处理模式 (推荐): 加载 preprocess_deap.py 生成的 .pt 文件

用法:
    # 1. 先预处理 (只需一次)
    python preprocess_deap.py --models all --chunk-size 128

    # 2. 训练 (预处理模式, 最快)
    python train_deap.py --models EEGNet TSCeption --use-preprocessed

    # 3. 训练 (原生模式, 带自动缓存)
    python train_deap.py --models FBCNet FBMSNet

    # 4. 快速测试 (1个epoch, 1折)
    python train_deap.py --models EEGNet --test

    # 5. 指定 CV 策略和窗口
    python train_deap.py --models ALL --chunk-size 256 --cv leave_one_subject_out

参考:
    TorchEEG: A deep learning toolbox towards EEG-based emotion recognition
    Table 1: Valence dimension accuracies on DEAP
"""

# =====================================
# 兼容性修复
# =====================================
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)

import scipy.signal
from scipy.signal.windows import hann, hamming, blackman
scipy.signal.hann = hann
scipy.signal.hamming = hamming
scipy.signal.blackman = blackman

from utils.fixes import apply_all_fixes
apply_all_fixes()

# =====================================
# 标准库
# =====================================
import argparse
import json
import math
import time
import copy
import warnings
from collections import Counter
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# =====================================
# 项目内部
# =====================================
from config import (
    config, EEGNetParams, FBCNetParams, FBMSNetParams, TSCeptionParams,
    DEAP_NUM_CHANNELS, DEAP_SAMPLING_RATE, DEAP_NUM_CLASSES,
)
from utils.model_utils import get_device, print_gpu_info
from utils.training_strategies import EarlyStopping, create_scheduler
from torcheeg.datasets import DEAPDataset

# =====================================
# 常量
# =====================================
DEFAULT_DEAP_DIR = os.path.join(_PROJECT_DIR, 'data', 'deap')
RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')
AVAILABLE_MODELS = ['EEGNet', 'TSCeption', 'FBCNet', 'FBMSNet', 'CCNN']


# =====================================
# CCNN 包装器 — 处理 ToGrid 输出 reshape
# =====================================

class CCNNWrapper(nn.Module):
    """CCNN 包装器: 将 ToGrid 输出 (t, 9, 9) → (in_channels, 9, 9)

    在 chunk_size=128 时, ToGrid 输出 (128, 9, 9),
    分为 segments=4 段 (每段 32个时间点), 每段取均值.
    """

    def __init__(self, in_channels: int = 4, grid_size: Tuple[int, int] = (9, 9),
                 num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        from torcheeg.models import CCNN
        self.ccnn = CCNN(
            in_channels=in_channels,
            grid_size=grid_size,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.segments = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, t, h, w) — t=timepoints, h=w=9"""
        batch, t, h, w = x.shape
        x = x.view(batch, self.segments, t // self.segments, h, w)
        x = x.mean(dim=2)  # (batch, segments, h, w)
        return self.ccnn(x)


# =====================================
# 预处理加载 Dataset (最快模式)
# =====================================

def get_preprocessed_path(model_name: str, data_dir: str) -> str:
    """获取模型预处理 .pt 文件路径"""
    return os.path.join(data_dir, f'{model_name}_data.pt')


def get_meta_path(data_dir: str) -> str:
    return os.path.join(data_dir, 'meta.pt')


def check_preprocessed(model_name: str, data_dir: str) -> bool:
    """检查预处理文件是否存在"""
    return (os.path.exists(get_preprocessed_path(model_name, data_dir))
            and os.path.exists(get_meta_path(data_dir)))


def load_preprocessed(model_name: str, data_dir: str,
                      subject_ids: Optional[List[int]] = None,
                      device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """加载预处理 .pt 文件

    Returns:
        (data_tensor, labels, subjects)
    """
    data_path = get_preprocessed_path(model_name, data_dir)
    meta_path = get_meta_path(data_dir)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f'Preprocessed data not found: {data_path}\n'
            f'Run: python preprocess_deap.py --models {model_name}')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'Metadata not found: {meta_path}\n'
            f'Run: python preprocess_deap.py')

    data = torch.load(data_path, map_location='cpu', weights_only=True)
    meta = torch.load(meta_path, map_location='cpu', weights_only=True)
    labels = meta['labels']
    subjects = meta['subjects']

    if subject_ids is not None:
        mask = torch.isin(subjects, torch.tensor(subject_ids))
        data = data[mask]
        labels = labels[mask]
        subjects = subjects[mask]

    if device != 'cpu':
        data = data.to(device)
        labels = labels.to(device)
        subjects = subjects.to(device)

    return data, labels, subjects


class CombinedDEAPDataset(Dataset):
    """加载预处理 .pt 文件的 Dataset

    支持按 subject_ids 或 indices 筛选, 用于交叉验证.
    核心优化: 传 indices 但不提前切片, 避免 GPU 显存翻倍.
    数据形态:
        EEGNet/TSCeption: (N, 1, 32, chunk_size)
        FBCNet/FBMSNet:   (N, 9, 32, chunk_size)
        CCNN:             (N, chunk_size, 9, 9)
    """

    def __init__(self, data, labels, subjects,
                 subject_ids: Optional[List[int]] = None,
                 indices: Optional[torch.Tensor] = None):
        # 数据引用 (可能是 CPU 或 GPU tensor)
        self._data = data
        self._labels = labels.flatten()
        self._subjects = subjects.flatten()

        if subject_ids is not None:
            mask = torch.isin(self._subjects, torch.tensor(subject_ids))
            self.indices = torch.where(mask)[0]
        elif indices is not None:
            self.indices = indices
        else:
            self.indices = torch.arange(len(self._data))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self._data[real_idx], int(self._labels[real_idx]), int(self._subjects[real_idx])


# =====================================
# 原生 DEAPDataset (标准模式)
# =====================================

def get_deap_dataset(root_path: str,
                     chunk_size: int = 128,
                     overlap: int = 0,
                     online_transform=None,
                     label_transform=None,
                     io_path: Optional[str] = None,
                     io_mode: str = 'pickle',
                     num_worker: int = 0,
                     verbose: bool = True):
    """创建 TorchEEG DEAPDataset"""
    if io_path is None:
        io_path = f'./_deap_cache_{chunk_size}_{overlap}'
    return DEAPDataset(
        root_path=root_path, chunk_size=chunk_size, overlap=overlap,
        num_channel=DEAP_NUM_CHANNELS,
        online_transform=online_transform,
        label_transform=label_transform,
        io_path=io_path, io_mode=io_mode,
        num_worker=num_worker, verbose=verbose,
    )


def get_transform(model_name: str, chunk_size: int = 128) -> Callable:
    """online_transform 工厂"""
    if model_name == 'CCNN':
        from torcheeg import transforms as T
        from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT
        return T.Compose([
            T.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
            T.ToTensor(),
        ])
    elif model_name in ('EEGNet', 'TSCeption'):
        from torcheeg import transforms as T
        return T.Compose([T.To2d(), T.ToTensor()])
    elif model_name in ('FBCNet', 'FBMSNet'):
        from torcheeg import transforms as T
        fbc_bands = {f'band{i}': [4*i, 4*(i+1)] for i in range(1, 10)}
        return T.Compose([
            T.BandSignal(sampling_rate=DEAP_SAMPLING_RATE, band_dict=fbc_bands),
            T.ToTensor(),
        ])
    else:
        raise ValueError(f'Unknown model: {model_name}')


def get_label_transform() -> Callable:
    """标签 transform: 取 valence 列, 二值化 (阈值 5)"""
    from torcheeg import transforms as T
    return T.Compose([
        T.Select('valence'),
        T.Binary(5.0),
    ])


def get_criterion(model_name: str) -> nn.Module:
    return nn.NLLLoss() if model_name in ('FBCNet', 'FBMSNet') else nn.CrossEntropyLoss()


def get_model(model_name: str, num_classes: int = 2,
              chunk_size: int = 128,
              offline_type: str = 'auto') -> nn.Module:
    """创建模型, 支持不同 offline 预处理模式

    Args:
        model_name: 模型名
        num_classes: 分类数
        chunk_size: 窗口样本数 (仅对有时域维度的模式有效)
        offline_type: 预处理类型, 影响模型参数 (in_channels 等)
    """
    params_map = {'EEGNet': EEGNetParams, 'FBCNet': FBCNetParams,
                  'FBMSNet': FBMSNetParams, 'TSCeption': TSCeptionParams}

    if model_name == 'CCNN':
        # CCNN: 4D输入 (N, t, 9, 9). in_channels = segment 数
        if offline_type == 'de':
            cnn_in_channels = 4  # DE 4 频带
        else:
            cnn_in_channels = chunk_size // 32  # 默认: 128/32=4
        return CCNNWrapper(
            in_channels=cnn_in_channels, grid_size=(9, 9),
            num_classes=num_classes, dropout=0.5)

    elif model_name == 'EEGNet':
        from torcheeg.models import EEGNet
        p = params_map['EEGNet']()
        if offline_type == 'de':
            raise ValueError(
                'EEGNet 不支持 DE 特征: 时间维度太小 (4) 小于卷积核大小 (64). '
                '请使用 --offline bandpass 或 auto.')
        # bandpass 模式: BandSignal → To2d 输出 (1, 9, 32, 128), EEGNet 用默认通道
        return EEGNet(chunk_size=chunk_size, num_electrodes=DEAP_NUM_CHANNELS,
                      F1=p.F1, F2=p.F2, D=p.D,
                      kernel_1=p.kernel_1, kernel_2=p.kernel_2,
                      dropout=p.dropout, num_classes=num_classes)

    elif model_name == 'TSCeption':
        from torcheeg.models import TSCeption
        p = params_map['TSCeption']()
        if offline_type == 'de':
            raise ValueError(
                'TSCeption 不支持 DE 特征: 时间维度太小. '
                '请使用 --offline bandpass 或 auto.')
        return TSCeption(num_electrodes=DEAP_NUM_CHANNELS, in_channels=1,
                         num_classes=num_classes, sampling_rate=DEAP_SAMPLING_RATE,
                         num_T=p.num_T, num_S=p.num_S,
                         hid_channels=p.hid_channels, dropout=p.dropout)

    elif model_name == 'FBCNet':
        from torcheeg.models import FBCNet
        p = params_map['FBCNet']()
        if offline_type == 'de':
            raise ValueError(
                'FBCNet 不支持 DE 特征: 输出无频带维度且时间太小. '
                '请使用 --offline bandpass 或 auto.')
        if offline_type == 'none':
            # 原始 EEG 作为单频带输入
            return FBCNet(num_electrodes=DEAP_NUM_CHANNELS, chunk_size=chunk_size,
                          in_channels=1, num_S=p.num_S, num_classes=num_classes,
                          temporal=p.temporal, stride_factor=p.stride_factor)
        return FBCNet(num_electrodes=DEAP_NUM_CHANNELS, chunk_size=chunk_size,
                      in_channels=9, num_S=p.num_S, num_classes=num_classes,
                      temporal=p.temporal, stride_factor=p.stride_factor)

    elif model_name == 'FBMSNet':
        from torcheeg.models import FBMSNet
        p = params_map['FBMSNet']()
        if offline_type == 'de':
            raise ValueError(
                'FBMSNet 不支持 DE 特征: 输出无频带维度且时间太小. '
                '请使用 --offline bandpass 或 auto.')
        if offline_type == 'none':
            return FBMSNet(num_electrodes=DEAP_NUM_CHANNELS, chunk_size=chunk_size,
                           in_channels=1, num_classes=num_classes,
                           stride_factor=p.stride_factor, temporal=p.temporal,
                           num_feature=p.num_feature, dilatability=p.dilatability)
        return FBMSNet(num_electrodes=DEAP_NUM_CHANNELS, chunk_size=chunk_size,
                       in_channels=9, num_classes=num_classes,
                       stride_factor=p.stride_factor, temporal=p.temporal,
                       num_feature=p.num_feature, dilatability=p.dilatability)

    else:
        raise ValueError(f'Unknown model: {model_name}')


# =====================================
# 训练 & 评估
# =====================================

def train_one_epoch(model, dataloader, optimizer, criterion,
                    device: str, scaler=None) -> Dict:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    data_time = 0.0
    compute_time = 0.0
    t_start = time.time()

    for batch in dataloader:
        t_data = time.time()

        # 兼容三种数据格式:
        # 1. DEAPDataset tuple: (eeg, label)     → len=2
        # 2. CombinedDEAPDataset triplet: (eeg, label, subj) → len=3
        # 3. dict: {'eeg':..., 'y':...}
        if isinstance(batch, dict):
            inputs, labels = batch['eeg'], batch['y']
        elif isinstance(batch, (list, tuple)):
            inputs, labels = batch[0], batch[1]
        else:
            raise TypeError(f'Unexpected batch type: {type(batch)}')

        inputs = inputs.to(device)
        labels = labels.to(device).long()

        data_time += time.time() - t_data
        t_comp = time.time()

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

        compute_time += time.time() - t_comp
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return {
        'loss': running_loss / len(dataloader),
        'acc': 100.0 * correct / total if total > 0 else 0.0,
        'time': time.time() - t_start,
        'data_time': data_time,
        'compute_time': compute_time,
    }


def evaluate(model, dataloader, criterion, device: str) -> Dict:
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
    n_classes = len(torch.unique(all_labels))
    cm = torch.zeros(n_classes, n_classes)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    per_class_acc = cm.diag() / cm.sum(1).clamp(min=1) * 100
    precision = cm.diag() / cm.sum(0).clamp(min=1)
    recall = cm.diag() / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    return {
        'loss': running_loss / len(dataloader),
        'acc': accuracy,
        'f1': f1.mean().item() * 100,
        'per_class_acc': per_class_acc.tolist(),
        'cm': cm.tolist(),
    }


# =====================================
# 交叉验证辅助
# =====================================

def get_kfold_split_indices(n_total: int, n_splits: int,
                            shuffle: bool = True,
                            random_state: int = 42):
    """生成 KFold 折索引 (不依赖 TorchEEG 内部信息)"""
    from sklearn.model_selection import KFold
    return KFold(n_splits=n_splits, shuffle=shuffle,
                 random_state=random_state).split(range(n_total))


def get_group_kfold_indices(subjects: torch.Tensor,
                            n_splits: int = 5,
                            random_state: int = 42):
    """按 subject 分组的 KFold"""
    unique_subjects = torch.unique(subjects).tolist()
    np.random.seed(random_state)
    np.random.shuffle(unique_subjects)
    fold_splits = np.array_split(unique_subjects, n_splits)
    folds = []
    for fi in range(n_splits):
        test_subjects = set(fold_splits[fi])
        train_mask = ~torch.isin(subjects, torch.tensor(list(test_subjects)))
        test_mask = torch.isin(subjects, torch.tensor(list(test_subjects)))
        folds.append((train_mask, test_mask))
    return folds


# =====================================
# 实验主函数
# =====================================

def run_experiment(
    model_name: str,
    deap_root: str,
    chunk_size: int = 128,
    cv_strategy: str = 'kfold_groupby_trial',
    n_splits: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    epochs: int = 100,
    scheduler_name: str = 'cosine',
    early_patience: int = 15,
    device: str = 'cpu',
    run_dir: str = '',
    use_preprocessed: bool = False,
    preproc_dir: str = '',
    test_mode: bool = False,
    test_ratio: float = 0.0,
    verbose: bool = True,
) -> Dict:
    """运行 DEAP 实验

    当 test_ratio > 0 时:
      1. 先按 subject 划分 held-out test set (如 20%)
      2. 剩余数据做 K-fold CV (train/val)
      3. CV 结束后用最佳模型在 test set 上评估
      最终指标以 test accuracy 为准。

    当 test_ratio = 0: 纯 K-fold CV, 以 val 均值为准 (论文 Table 1 方式)。

    use_preprocessed=True: 加载 .pt 文件, 最快模式.
    """
    if use_preprocessed:
        # ── 预处理模式 (最快) ──
        if not preproc_dir:
            preproc_dir = os.path.join(DEFAULT_DEAP_DIR, 'preprocessed')
            if not os.path.exists(preproc_dir):
                # 尝试默认的预处理输出位置
                alt = os.path.join(os.path.dirname(DEFAULT_DEAP_DIR), 'deap_preprocessed')
                if os.path.exists(alt):
                    preproc_dir = alt

        if not check_preprocessed(model_name, preproc_dir):
            print(f'[ERROR] Preprocessed data not found for {model_name}')
            print(f'        Run: python preprocess_deap.py --models {model_name}')
            if run_dir:
                return {'model': model_name, 'error': 'no_preprocessed_data'}
            return {'model': model_name, 'error': 'no_preprocessed_data'}

        # 加载元数据获取 offline_type
        meta_path = get_meta_path(preproc_dir)
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location='cpu', weights_only=True)
            preproc_offline = meta.get('offline_type', 'auto')
        else:
            preproc_offline = 'auto'
        if verbose:
            print(f'       Offline:   {preproc_offline}')

        # 模型兼容性检查 (DE 模式仅 CCNN 支持)
        INCOMPATIBLE_OFFLINE = {
            'EEGNet': ['de'], 'TSCeption': ['de'],
            'FBCNet': ['de'], 'FBMSNet': ['de'],
        }
        if preproc_offline in INCOMPATIBLE_OFFLINE.get(model_name, []):
            print(f'\n  [SKIP] {model_name} 与 offline_type={preproc_offline} 不兼容.')
            print(f'         跳过该模型. 请使用 --offline bandpass 或 auto 重新预处理.')
            return {'model': model_name,
                    'error': f'incompatible_offline_{preproc_offline}'}

        # 加载数据
        data, labels, subjects = load_preprocessed(
            model_name, preproc_dir, device='cpu')
        n_total = len(data)

        # 数据形状适配: 'none' 模式 FBCNet/FBMSNet 输出 3D (N,32,T), 需要 4D (N,1,32,T)
        if preproc_offline == 'none' and model_name in ('FBCNet', 'FBMSNet'):
            if data.dim() == 3:
                data = data.unsqueeze(1)  # (N, 32, T) → (N, 1, 32, T)

        if verbose:
            cls_counts = Counter(labels.tolist())
            print(f'\n[DEAP] Preprocessed: {n_total} samples ({model_name})')
            print(f'       Shape: {tuple(data.shape)}, Class: {dict(cls_counts)}')

        # ── 持出 test set (按 subject 划分) ──
        test_subject_ids = None
        if test_ratio > 0 and use_preprocessed:
            unique_subjects = sorted(subjects.unique().tolist())
            n_test = max(1, int(len(unique_subjects) * test_ratio))
            np.random.seed(42)
            test_subject_ids = set(np.random.choice(unique_subjects, n_test, replace=False).tolist())
            train_val_subjects = [s for s in unique_subjects if s not in test_subject_ids]
            if verbose:
                print(f'       Hold-out test: {len(test_subject_ids)} subjects {sorted(test_subject_ids)}')
                print(f'       Train/Val:     {len(train_val_subjects)} subjects')
        else:
            unique_subjects = sorted(subjects.unique().tolist())
            train_val_subjects = unique_subjects

        # 按 subject 的 KFold 分组
        if cv_strategy == 'leave_one_subject_out':
            fold_subject_groups = [[s] for s in train_val_subjects]
        elif 'kfold' in cv_strategy:
            np.random.seed(42)
            shuffled = train_val_subjects.copy()
            np.random.shuffle(shuffled)
            fold_subject_groups = np.array_split(shuffled, min(n_splits, len(shuffled)))
            fold_subject_groups = [g.tolist() if hasattr(g, 'tolist') else list(g)
                                   for g in fold_subject_groups]
        else:
            fold_subject_groups = np.array_split(train_val_subjects, n_splits)
            fold_subject_groups = [g.tolist() if hasattr(g, 'tolist') else list(g)
                                   for g in fold_subject_groups]

        fold_indices = []
        for val_subjects in fold_subject_groups:
            train_mask = ~torch.isin(subjects, torch.tensor(val_subjects))
            if test_subject_ids:
                # 也从 train 中排除 test subjects
                test_mask_tensor = torch.tensor(list(test_subject_ids))
                train_mask = train_mask & ~torch.isin(subjects, test_mask_tensor)
            val_mask = torch.isin(subjects, torch.tensor(val_subjects))
            train_idx = torch.where(train_mask)[0]
            val_idx = torch.where(val_mask)[0]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            # 校验: 确保 train/val/test 无重叠
            train_subs = set(subjects[train_idx].tolist())
            val_subs = set(subjects[val_idx].tolist())
            assert train_subs & val_subs == set(), f'Overlap: {train_subs & val_subs}'
            if test_subject_ids:
                assert train_subs & test_subject_ids == set(), f'Train-test overlap!'
                assert val_subs & test_subject_ids == set(), f'Val-test overlap!'
            fold_indices.append((train_idx, val_idx))

        # 数据移到 GPU 加速训练 (每个模型独立加载, 见外层循环)
        if device != 'cpu':
            data = data.to(device)
    else:
        # ── 原生 DEAPDataset 模式 ──
        online_transform = get_transform(model_name, chunk_size)
        label_transform = get_label_transform()

        io_path = f'./_deap_cache_{model_name}_{chunk_size}'
        dataset = get_deap_dataset(
            root_path=deap_root, chunk_size=chunk_size, overlap=0,
            online_transform=online_transform,
            label_transform=label_transform,
            io_path=io_path, io_mode='pickle',
            num_worker=0, verbose=False,
        )

        n_total = len(dataset)
        if verbose:
            print(f'\n[DEAP] Dataset: {n_total} samples ({model_name})')
            cls_counts = Counter()
            for i in range(min(500, n_total)):
                sample = dataset[i]
                # DEAPDataset returns (eeg, label)
                lbl = sample[1] if isinstance(sample, (list, tuple)) else sample['y']
                cls_counts[int(lbl)] += 1
            print(f'       Class distribution (sample): {dict(cls_counts)}')

        # 使用 TorchEEG 的 model_selection
        from torcheeg.model_selection import (
            KFoldGroupbyTrial, KFold, LeaveOneSubjectOut,
            KFoldPerSubjectGroupbyTrial,
        )

        split_dir = f'./_deap_split_{model_name}_{chunk_size}'
        if cv_strategy == 'leave_one_subject_out':
            cv = LeaveOneSubjectOut(split_path=f'{split_dir}_loso')
        elif cv_strategy == 'kfold':
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42,
                       split_path=f'{split_dir}_k{n_splits}')
        elif cv_strategy == 'kfold_groupby_trial':
            cv = KFoldGroupbyTrial(n_splits=n_splits, shuffle=True, random_state=42,
                                   split_path=f'{split_dir}_kgt{n_splits}')
        elif cv_strategy == 'kfold_per_subject_groupby_trial':
            cv = KFoldPerSubjectGroupbyTrial(
                n_splits=n_splits, shuffle=True, random_state=42,
                split_path=f'{split_dir}_kpsgt{n_splits}')
        else:
            raise ValueError(f'Unknown CV: {cv_strategy}')

    # ── 如果只是测试模式, 只用 1 折, epochs=1 ──
    actual_epochs = 1 if test_mode else epochs

    # ── 逐 fold 训练 ──
    fold_results = []
    all_fold_metrics = []
    best_val_acc = 0.0
    best_model_state = None

    n_folds = n_splits if not use_preprocessed else len(fold_indices)
    if use_preprocessed and cv_strategy == 'leave_one_subject_out':
        n_folds = len(unique_subjects)

    for fold_idx in range(min(n_folds, 2 if test_mode else n_folds)):
        if use_preprocessed:
            train_idx, test_idx = fold_indices[fold_idx]
            # 传 indices 而非切片, 避免 GPU 显存翻倍
            train_dataset = CombinedDEAPDataset(
                data, labels, subjects, indices=train_idx)
            val_dataset = CombinedDEAPDataset(
                data, labels, subjects, indices=test_idx)
        else:
            # TorchEEG native: cv.split(dataset) yields (train_ds, val_ds)
            fold_gen = cv.split(dataset)
            for fi, (tds, vds) in enumerate(fold_gen):
                if fi == fold_idx:
                    train_dataset, val_dataset = tds, vds
                    break

        # 数据已在 GPU (data.to(device)), 不需要 pin_memory
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=0, drop_last=False)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, drop_last=False)

        if verbose:
            print(f'\n{"="*60}')
            print(f'  Fold {fold_idx+1}/{n_folds}')
            print(f'  Train: {len(train_dataset)} | Val: {len(val_dataset)}')
            print(f'{"="*60}')

        # 模型
        model = get_model(model_name, num_classes=DEAP_NUM_CLASSES,
                          chunk_size=chunk_size,
                          offline_type=preproc_offline if use_preprocessed else 'auto')
        model = model.to(device)
        criterion = get_criterion(model_name)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, scheduler_name, actual_epochs)
        early_stopping = EarlyStopping(
            patience=early_patience, mode='max', verbose=verbose)
        scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

        fold_best_val_acc = 0.0
        fold_metrics = []

        for epoch in range(actual_epochs):
            t_epoch = time.time()
            train_m = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler)
            val_m = evaluate(model, val_loader, criterion, device)
            epoch_time = time.time() - t_epoch

            metrics = {
                'epoch': epoch + 1,
                'train_loss': round(train_m['loss'], 4),
                'train_acc': round(train_m['acc'], 2),
                'val_loss': round(val_m['loss'], 4),
                'val_acc': round(val_m['acc'], 2),
                'val_f1': round(val_m['f1'], 2),
                'time': round(epoch_time, 2),
            }
            fold_metrics.append(metrics)

            if scheduler_name == 'plateau':
                scheduler.step(val_m['loss'])
            else:
                scheduler.step()

            if val_m['acc'] > fold_best_val_acc:
                fold_best_val_acc = val_m['acc']
                if val_m['acc'] > best_val_acc:
                    best_val_acc = val_m['acc']
                    best_model_state = copy.deepcopy(model.state_dict())

            if early_stopping(val_m['acc'], epoch):
                if verbose:
                    print(f'       Early stop @ epoch {epoch+1}')
                break

            # 日志
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0
                            or epoch == actual_epochs - 1):
                print(f'  Ep {epoch+1:3d}/{actual_epochs} | '
                      f'T_loss:{train_m["loss"]:.4f} T_acc:{train_m["acc"]:.2f}% | '
                      f'V_loss:{val_m["loss"]:.4f} V_acc:{val_m["acc"]:.2f}% '
                      f'V_f1:{val_m["f1"]:.2f}% | {epoch_time:.1f}s')

        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_acc': fold_best_val_acc,
            'epochs_trained': len(fold_metrics),
        })
        all_fold_metrics.append(fold_metrics)

        if verbose:
            print(f'  >>> Fold {fold_idx+1} best val acc: {fold_best_val_acc:.2f}%')

        # 保存 per-fold 指标
        if run_dir:
            fold_csv = os.path.join(run_dir, f'fold_{fold_idx+1}_metrics.csv')
            pd.DataFrame(fold_metrics).to_csv(fold_csv, index=False)

    # ── 在 hold-out test set 上评估 ──
    test_metrics = None
    if test_subject_ids and best_model_state is not None and use_preprocessed:
        print(f'\n  {"="*60}')
        print(f'  Evaluating on held-out test set ({len(test_subject_ids)} subjects)...')
        print(f'  {"="*60}')

        # 用最佳模型参数重新建模型
        test_model = get_model(model_name, num_classes=DEAP_NUM_CLASSES,
                               chunk_size=chunk_size)
        test_model.load_state_dict(best_model_state)
        test_model = test_model.to(device)
        test_criterion = get_criterion(model_name)

        # 构建 test dataset (只需对应 subject 的数据)
        test_mask = torch.isin(subjects, torch.tensor(list(test_subject_ids)))
        test_data = data[test_mask]
        test_lbls = labels[test_mask]
        test_subs = subjects[test_mask]

        test_dataset = CombinedDEAPDataset(test_data, test_lbls, test_subs)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, drop_last=False)

        test_metrics = evaluate(test_model, test_loader, test_criterion, device)
        print(f'  Test acc:  {test_metrics["acc"]:.2f}%')
        print(f'  Test f1:   {test_metrics["f1"]:.2f}%')
        print(f'  Test loss: {test_metrics["loss"]:.4f}')
        print(f'  Test CM:\n{np.array(test_metrics["cm"])}')

        # 保存 predictions
        test_model.eval()
        all_preds, all_gt = [], []
        with torch.no_grad():
            for batch in test_loader:
                inp = batch[0].to(device)
                out = test_model(inp)
                _, pred = torch.max(out, 1)
                all_preds.extend(pred.cpu().tolist())
                all_gt.extend(batch[1].tolist())
        if run_dir:
            np.savez(os.path.join(run_dir, 'test_predictions.npz'),
                     predictions=all_preds, ground_truth=all_gt)

        del test_model

    # ── 汇总 ──
    val_accs = [r['best_val_acc'] for r in fold_results]
    summary = {
        'model': model_name,
        'chunk_size': chunk_size,
        'cv_strategy': cv_strategy,
        'n_splits': n_folds,
        'batch_size': batch_size,
        'lr': lr,
        'epochs_actual': actual_epochs,
        'early_patience': early_patience,
        'scheduler': scheduler_name,
        'use_preprocessed': use_preprocessed,
        'test_mode': test_mode,
        'test_ratio': test_ratio,
        'fold_results': fold_results,
        'mean_val_acc': round(float(np.mean(val_accs)), 2),
        'std_val_acc': round(float(np.std(val_accs)), 2),
        'best_val_acc': round(float(np.max(val_accs)), 2),
    }
    if test_metrics:
        summary['test_acc'] = round(test_metrics['acc'], 2)
        summary['test_f1'] = round(test_metrics['f1'], 2)
        summary['test_loss'] = round(test_metrics['loss'], 4)

    if verbose:
        print(f'\n{"="*60}')
        print(f'  {model_name} — {cv_strategy} ({n_folds}-fold)')
        print(f'  Window: {chunk_size}pt')
        if test_metrics:
            print(f'  Test acc: {test_metrics["acc"]:.2f}%  (held-out {len(test_subject_ids)} subjects)')
        print(f'  CV Per-fold: {[f"{a:.2f}" for a in val_accs]}')
        print(f'  CV Mean±Std: {summary["mean_val_acc"]:.2f}±{summary["std_val_acc"]:.2f}')
        print(f'  CV Best: {summary["best_val_acc"]:.2f}%')
        print(f'{"="*60}')

    # ── 保存 ──
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(run_dir, 'best_model.pt'))
        with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        all_rows = []
        for fi, fm in enumerate(all_fold_metrics):
            for row in fm:
                row['fold'] = fi + 1
                all_rows.append(row)
        if all_rows:
            pd.DataFrame(all_rows).to_csv(
                os.path.join(run_dir, 'all_epochs.csv'), index=False)

    return summary


# =====================================
# 主入口
# =====================================

def parse_args():
    p = argparse.ArgumentParser(
        description='DEAP 训练 — 复现 TorchEEG EMO Table 1')
    p.add_argument('--models', type=str, nargs='+', default=['EEGNet'],
                    choices=AVAILABLE_MODELS + ['all'],
                    help=f'Model(s): {AVAILABLE_MODELS}, all')
    p.add_argument('--data-dir', type=str, default=None)
    p.add_argument('--download', action='store_true')
    p.add_argument('--use-preprocessed', action='store_true',
                    help='使用预计算 .pt 文件 (需先运行 preprocess_deap.py)')
    p.add_argument('--preproc-dir', type=str, default=None,
                    help='预计算 .pt 目录')
    p.add_argument('--chunk-size', type=int, default=128,
                    choices=[128, 256])
    p.add_argument('--cv', type=str, default='kfold_groupby_trial',
                    choices=['kfold', 'kfold_groupby_trial',
                             'kfold_per_subject_groupby_trial',
                             'leave_one_subject_out'])
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--early-patience', type=int, default=15)
    p.add_argument('--scheduler', type=str, default='cosine',
                    choices=['cosine', 'cosine_warm', 'step', 'plateau'])
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--results-dir', type=str, default=RESULTS_DIR)
    p.add_argument('--test', action='store_true',
                    help='快速测试模式 (1 epoch, 2 folds)')
    p.add_argument('--test-ratio', type=float, default=0.0,
                    help='held-out test set 比例 (如 0.2=20% subjects 做最终测试)')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    # ── 设备 ──
    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = get_device()
    print(f'[DEVICE] {device}  |  Mode: {"preprocessed" if args.use_preprocessed else "native"}'
          f'  |  Test: {args.test}')
    if device != 'cpu':
        print_gpu_info(device)

    # ── 数据路径 ──
    if args.use_preprocessed:
        if args.preproc_dir:
            deap_root = args.preproc_dir
        else:
            # 自动查找预处理目录
            candidates = [
                os.path.join(DEFAULT_DEAP_DIR, 'preprocessed'),
                os.path.join(os.path.dirname(DEFAULT_DEAP_DIR), 'deap_preprocessed'),
            ]
            deap_root = None
            for c in candidates:
                if os.path.exists(c) and os.path.exists(os.path.join(c, 'meta.pt')):
                    deap_root = c
                    break
            if deap_root is None:
                print('[ERROR] Preprocessed data not found. '
                      'Run: python preprocess_deap.py --models all')
                sys.exit(1)
        print(f'[DEAP] Preprocessed data: {deap_root}')
    else:
        if args.data_dir is None:
            from download_deap import get_deap_path
            deap_root = get_deap_path()
        else:
            deap_root = args.data_dir
        if not os.path.exists(deap_root) or \
           not any(f.endswith('.dat') for f in os.listdir(deap_root)):
            if args.download:
                from download_deap import ensure_deap_dataset
                deap_root = ensure_deap_dataset(args.data_dir)
            else:
                print(f'[ERROR] DEAP data not found at {deap_root}')
                print('        Use --download to auto-download')
                sys.exit(1)
        print(f'[DEAP] Data root: {deap_root}')

    # ── 模型列表 ──
    models = AVAILABLE_MODELS if 'all' in args.models else args.models

    # ── 结果目录 ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    window_label = f'{args.chunk_size}pt_{args.chunk_size/DEAP_SAMPLING_RATE:.0f}s'
    mode_label = 'pre' if args.use_preprocessed else 'native'
    test_label = '_TEST' if args.test else ''
    results_dir = os.path.join(
        args.results_dir,
        f'DEAP_{window_label}_{args.cv}_{mode_label}{test_label}_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f'[CONFIG] -> {results_dir}')

    # ── 逐模型训练 ──
    all_summaries = []
    for model_name in models:
        print(f'\n{"#"*65}')
        print(f'#  {model_name} | {args.chunk_size}pt | {args.cv} | {device}')
        print(f'{"#"*65}')

        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        summary = run_experiment(
            model_name=model_name, deap_root=deap_root,
            chunk_size=args.chunk_size,
            cv_strategy=args.cv, n_splits=args.n_splits,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs, scheduler_name=args.scheduler,
            early_patience=args.early_patience,
            device=device, run_dir=model_dir,
            use_preprocessed=args.use_preprocessed,
            preproc_dir=deap_root if args.use_preprocessed else '',
            test_mode=args.test,
            test_ratio=args.test_ratio,
            verbose=verbose,
        )
        all_summaries.append(summary)
        # Clear GPU cache between models
        if device != "cpu":
            torch.cuda.empty_cache()

    # ── Table 1 汇总 ──
    print(f'\n\n{"="*65}')
    print(f'  Table 1 — Valence Acc on DEAP')
    print(f'  Window: {args.chunk_size}pt ({args.chunk_size/DEAP_SAMPLING_RATE:.0f}s)')
    print(f'  CV: {args.cv}{" (TEST)" if args.test else ""}')
    print(f'{"="*65}')
    has_test = any(s.get('test_acc') for s in all_summaries)
    if has_test:
        print(f'  {"Model":<12s}  {"CV Mean":<8s}  {"Test Acc":<9s}  {"Test F1":<8s}')
        print(f'  {"-"*42}')
        for s in all_summaries:
            if 'error' in s:
                print(f'  {s["model"]:<12s}  ERROR')
            else:
                ta = s.get('test_acc', 0)
                tf1 = s.get('test_f1', 0)
                print(f'  {s["model"]:<12s}  {s["mean_val_acc"]:<6.2f}%   '
                      f'{ta:<6.2f}%    {tf1:<6.2f}%')
    else:
        print(f'  {"Model":<12s}  {"CV Mean":<8s}  {"Std":<7s}  {"Best":<7s}')
        print(f'  {"-"*35}')
        for s in all_summaries:
            if 'error' in s:
                print(f'  {s["model"]:<12s}  ERROR')
            else:
                print(f'  {s["model"]:<12s}  {s["mean_val_acc"]:<6.2f}%   '
                      f'{s["std_val_acc"]:<5.2f}%  {s["best_val_acc"]:<6.2f}%')
    print(f'{"="*65}')

    table_path = os.path.join(results_dir, 'table1_summary.csv')
    rows = [{
        'Model': s['model'],
        'CV_Mean_Acc': s.get('mean_val_acc', 0),
        'CV_Std': s.get('std_val_acc', 0),
        'CV_Best_Acc': s.get('best_val_acc', 0),
        'Test_Acc': s.get('test_acc', ''),
        'Test_F1': s.get('test_f1', ''),
        'CV': s.get('cv_strategy', args.cv),
        'Window_pts': s.get('chunk_size', args.chunk_size),
    } for s in all_summaries if 'error' not in s]
    if rows:
        pd.DataFrame(rows).to_csv(table_path, index=False)
        print(f'[TABLE] -> {table_path}')
    print(f'[DONE] -> {results_dir}')


if __name__ == '__main__':
    main()
