"""
工具模块: 训练策略 + scipy 兼容性修复 + 辅助函数
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import scipy.signal
from scipy.signal.windows import hann, hamming, blackman

# ── scipy 兼容性修复 (Python 3.12) ──
scipy.signal.hann = hann
scipy.signal.hamming = hamming
scipy.signal.blackman = blackman

warnings.filterwarnings('ignore')


def format_channel_location_dict(
    channel_list: List[str],
    location_list: List[List[str]],
) -> Dict[str, Tuple[int, int]]:
    """将电极名称列表 + 布局网格转换为 TorchEEG 的 location dict

    Args:
        channel_list: 电极名称列表, 如 ['FP1', 'AF3', 'F7', ...]
        location_list: 二维网格布局, 每个位置是电极名或 '-'

    Returns:
        Dict[str, Tuple[int, int]]: {channel_name: (row, col)}
    """
    location_dict = {}
    for row_idx, row in enumerate(location_list):
        for col_idx, ch_name in enumerate(row):
            if ch_name != '-' and ch_name in channel_list:
                location_dict[ch_name] = (row_idx, col_idx)
    return location_dict


class EarlyStopping:
    """早停策略 — 监控验证集指标, 连续 patience 轮无改善则停止"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = 'max', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = -math.inf if mode == 'max' else math.inf
        self.early_stop = False
        self.best_epoch = -1

    def __call__(self, score: float, epoch: int = 0) -> bool:
        if self.mode == 'max':
            improved = (score - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - score) > self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'  [EarlyStopping] STOP @ epoch {epoch}, '
                          f'best={self.best_score:.4f} @ epoch {self.best_epoch}')
                return True
        return False

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def load_state_dict(self, state):
        self.__dict__.update(state)


def create_scheduler(optimizer, name: str, epochs: int, **kwargs):
    """创建 LR scheduler"""
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
    elif name == 'plateau':
        patience = kwargs.get('lr_patience', 5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, factor=0.5, min_lr=1e-6)
    elif name == 'step':
        step_size = kwargs.get('step_size', max(1, epochs // 3))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=0.5)
    else:
        raise ValueError(f'Unknown scheduler: {name}')


def set_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
