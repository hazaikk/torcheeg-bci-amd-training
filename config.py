"""
训练配置 — 集中管理所有超参数和路径

AMD 开发者云专用配置
=====================
- data_dir: 可在 Notebook 中通过环境变量 AMT_DATA_DIR 覆盖
- device: 自动检测 ROCm / CUDA / CPU
- 内存优化: batch_size 默认 128 (AMD 16GiB GPU)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================
# 数据路径
# =============================================
# 可以通过环境变量覆盖 (在 AMD Cloud Notebook 中设置)
DATA_DIR = os.environ.get('AMT_DATA_DIR', 'data')
DATA_COMBINED_FILE = 'BCICIV2a.mat'

# 单受试者文件的 BNCI 下载地址
BNCI_BASE_URL = 'http://bnci-horizon-2020.eu/database/data-sets/001-2014/'

# 受试者列表
SUBJECTS = [f'A{str(i).zfill(2)}' for i in range(1, 10)]  # A01~A09
SESSIONS = ['T', 'E']  # Training / Evaluation


# =============================================
# 模型配置
# =============================================
MODELS_TO_TRAIN = ['EEGNet', 'FBCNet', 'FBMSNet']


@dataclass
class ModelConfig:
    """单模型超参数"""
    # 通用
    chunk_size: int = 800
    num_electrodes: int = 22
    num_classes: int = 4
    in_channels: int = 1        # EEGNet/CSPNet/LMDA 用
    in_channels_band: int = 9   # FBCNet/FBMSNet 用 (9 频带)

    # 训练
    epochs: int = 50
    batch_size: int = 128       # AMD 16GiB GPU 可处理 128
    learning_rate: float = 0.001
    weight_decay: float = 0.001

    # 设备
    device: str = 'auto'        # 'auto' → 自动检测 cuda / rocm / cpu

    # 数据加载
    num_workers: int = 0        # AMD Cloud Notebook 保持 0

    # 可视化
    save_plots: bool = True
    plot_dir: str = 'results'


@dataclass
class EEGNetParams:
    F1: int = 8
    F2: int = 16
    D: int = 2
    kernel_1: int = 64
    kernel_2: int = 16
    dropout: float = 0.25


@dataclass
class FBCNetParams:
    num_S: int = 32
    temporal: str = 'LogVarLayer'
    stride_factor: int = 4


@dataclass
class FBMSNetParams:
    num_feature: int = 36
    dilatability: int = 8
    temporal: str = 'LogVarLayer'
    stride_factor: int = 4


@dataclass
class CSPNetParams:
    num_filters_t: int = 20
    filter_size_t: int = 25
    num_filters_s: int = 2
    pool_size_1: int = 100
    pool_stride_1: int = 25


@dataclass
class LMDAConfig:
    depth: int = 9
    kernel: int = 75
    hid_channels_1: int = 24
    hid_channels_2: int = 9
    pool_size: int = 5


@dataclass
class TSCeptionParams:
    num_T: int = 15
    num_S: int = 15
    hid_channels: int = 32
    dropout: float = 0.5


# =============================================
# 默认配置实例
# =============================================
config = ModelConfig()
