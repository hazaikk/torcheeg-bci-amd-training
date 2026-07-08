"""
训练策略模块 — EarlyStopping + LR Scheduler 工厂
=================================================
"""

import math
import torch
import numpy as np


class EarlyStopping:
    """早停策略 — 监控验证集指标, 连续 patience 轮无改善则停止

    Args:
        patience: 容忍无改善的 epoch 数 (默认 10)
        min_delta: 最小改善阈值 (默认 1e-4)
        mode: 'max' 监控指标越大越好, 'min' 越小越好
        verbose: 是否打印停止信息
    """

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
        """检查是否应停止训练, 返回 True=停止"""
        if self.mode == 'max':
            delta = score - self.best_score
            improved = delta > self.min_delta
        else:
            delta = self.best_score - score
            improved = delta > self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose and self.counter % 5 == 1:
                print(f'  [EarlyStopping] {self.counter}/{self.patience} '
                      f'no improvement (best={self.best_score:.4f} @ epoch {self.best_epoch})')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'  [EarlyStopping] STOP at epoch {epoch}, '
                          f'best={self.best_score:.4f} @ epoch {self.best_epoch}')
                return True
        return False

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def load_state_dict(self, state):
        self.__dict__.update(state)


def create_scheduler(optimizer, name: str, epochs: int, **kwargs):
    """创建 LR scheduler

    Args:
        name: 'cosine' | 'cosine_warm' | 'step' | 'plateau'
        epochs: 总训练 epoch 数 (cosine 用 T_max)
    """
    if name == 'cosine':
        # CosineAnnealingLR: 从 lr → 0, T_max=epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)

    elif name == 'cosine_warm':
        # CosineAnnealingWarmRestarts: 带重启的余弦退火
        T_0 = kwargs.get('T_0', max(10, epochs // 3))
        T_mult = kwargs.get('T_mult', 2)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6)

    elif name == 'step':
        # StepLR: 每 step_size 轮衰减 gamma
        step_size = kwargs.get('step_size', max(1, epochs // 3))
        gamma = kwargs.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

    elif name == 'plateau':
        # ReduceLROnPlateau: 验证指标停滞时衰减
        patience = kwargs.get('lr_patience', 5)
        factor = kwargs.get('factor', 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience,
            factor=factor, min_lr=1e-6)

    else:
        raise ValueError(f'Unknown scheduler: {name}')
