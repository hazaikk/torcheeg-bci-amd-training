"""
数据加载工具
=============
支持两种数据格式：
1. 预组装 BCICIV2a.mat（更快，推荐）
2. TorchEEG BCICIV2aDataset（单受试者 .mat 文件）

GPU 优化:
  - precompute=True: 对原始 numpy 数据批量预应用 transforms
    (如 BandSignal FFT 滤波只跑一次，不在每个 epoch 重复)
  - 预计算后 data 为 torch.float32 tensor，支持 pin_memory
"""

import os
import sys
import time
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torcheeg.datasets import BCICIV2aDataset
from torcheeg import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, DATA_COMBINED_FILE, SUBJECTS, SESSIONS, config

PREPROCESSED_DIR = 'preprocessed'


# ── 预处理数据持久化接口 ──

def get_preprocessed_path(model_name: str, data_dir: str) -> str:
    """获取某模型预处理 .pt 文件路径"""
    return os.path.join(data_dir, PREPROCESSED_DIR, f'{model_name}_data.pt')


def get_meta_path(data_dir: str) -> str:
    """获取元数据 .pt 文件路径"""
    return os.path.join(data_dir, PREPROCESSED_DIR, 'meta.pt')


def check_preprocessed(model_name: str, data_dir: str) -> bool:
    """检查预处理文件是否存在"""
    return (os.path.exists(get_preprocessed_path(model_name, data_dir))
            and os.path.exists(get_meta_path(data_dir)))


def load_preprocessed(model_name: str, data_dir: str,
                      subject_ids: Optional[List[int]] = None,
                      device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从磁盘加载预处理后的数据和元数据，可选按受试者筛选

    Returns:
        (data_tensor, labels, subjects)
        - data_tensor: (N, C, 22, 800) float32
        - labels: (N,) long, 0-3
        - subjects: (N,) long, 1-9
    """
    data_path = get_preprocessed_path(model_name, data_dir)
    meta_path = get_meta_path(data_dir)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f'Preprocessed data not found: {data_path}\n'
            f'Run: python preprocess_dataset.py --models {model_name}')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'Metadata not found: {meta_path}\n'
            f'Run: python preprocess_dataset.py')

    data = torch.load(data_path, map_location='cpu', weights_only=True)
    meta = torch.load(meta_path, map_location='cpu', weights_only=True)
    labels = meta['labels'] - 1  # 1-4 → 0-3
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


def precompute_transforms(data: np.ndarray, transform, model_name: str,
                          batch_size: int = 256,
                          device: str = 'cpu') -> torch.Tensor:
    """对完整数据集批量预应用 transforms，避免每个 epoch 重复计算。

    主要加速 FBCNet/FBMSNet 的 BandSignal (多频带 FFT 滤波):
      原始: 每个 epoch 对 4608 样本 × 9 频带做 FFT = 重复 50+ 次
      优化: 训练前做一次，后续 epoch 直接取 tensor

    Args:
        data: numpy (N, 22, 800)
        transform: torch EEG transforms.Compose
        model_name: 模型名（仅用于打印）
        batch_size: 批大小，控制内存占用
        device: 目标设备 — 'cuda' 时预计算后直接搬到 GPU

    Returns:
        torch.float32 tensor, shape 如 (N, 1, 22, 800) 或 (N, 9, 22, 800)
    """
    n = len(data)
    print(f'[DATA] Pre-computing transforms for {model_name} on {n} samples... ',
          end='', flush=True)
    t0 = time.time()

    transformed_batches = []
    for i in range(0, n, batch_size):
        batch = data[i:i + batch_size]
        batch_out = []
        for j, sample in enumerate(batch):
            result = transform(eeg=sample)['eeg']  # transforms return dict
            batch_out.append(result)
        # stack and ensure contiguous float32
        tb = torch.stack(batch_out).contiguous().float()
        transformed_batches.append(tb)

    result = torch.cat(transformed_batches, dim=0)
    elapsed = time.time() - t0

    # 如果 GPU 可用，把预计算数据搬到 GPU 训练（消除每 batch 的 .to(device) 开销）
    moved = ''
    if device != 'cpu' and result.device.type != device:
        result = result.to(device)
        moved = f', moved to {device}'

    print(f'done ({elapsed:.1f}s), shape={tuple(result.shape)}, '
          f'dtype={result.dtype}{moved}')
    return result


# =============================================
# 1. 预组装格式数据集 (BCICIV2a.mat)
# =============================================
class CombinedBCIDataset(Dataset):
    """读取预组装 BCICIV2a.mat 并封装为 PyTorch Dataset。

    支持两种模式:
      1. 原始 numpy → 应用 transforms (precompute=True 时预计算后缓存)
      2. 预加载 tensor → 跳过所有 transforms（从 .pt 文件加载）

    模式 2 最快：数据已经是 (N, C, 22, 800) 的 float32 tensor。
    """

    def __init__(self, data, labels: np.ndarray,
                 subjects: np.ndarray,
                 subject_ids: Optional[List[int]] = None,
                 transform: Optional[callable] = None,
                 precompute: bool = True,
                 precompute_device: str = 'cpu',
                 model_name: str = ''):
        """

        Args:
            data: numpy (N,22,800) / torch tensor (N,C,22,800)
            labels: numpy (N,) 或 tensor (N,)
            subjects: numpy (N,) 或 tensor (N,)
            subject_ids: 筛选的受试者 ID 列表
            transform: transforms（precompute=True 时预应用后设为 None）
            precompute: 是否预计算 transforms
            precompute_device: 预计算目标设备
            model_name: 用于日志
        """
        # 筛选受试者
        if subject_ids is not None:
            if isinstance(subjects, np.ndarray):
                mask = np.isin(subjects.flatten(), subject_ids)
            else:
                mask = torch.isin(subjects.flatten(),
                                  torch.tensor(subject_ids, device=subjects.device))

            if isinstance(data, np.ndarray):
                self.data = data[mask]
            else:
                self.data = data[mask]

            if isinstance(labels, np.ndarray):
                self.labels = labels[mask].flatten()
            else:
                self.labels = labels[mask].flatten()

            self.subjects = subjects[mask].flatten()
        else:
            self.data = data
            self.labels = labels.flatten()
            self.subjects = subjects.flatten()

        # 标签 1-4 → 0-3 (仅 numpy 格式需要)
        if isinstance(self.labels, np.ndarray):
            self.labels = self.labels - 1

        # 预计算或保留原始 transform
        if precompute and transform is not None:
            if isinstance(self.data, np.ndarray):
                self.data = precompute_transforms(
                    self.data, transform,
                    model_name or 'Unknown',
                    device=precompute_device)
                self.transform = None
            else:
                self.transform = None  # 已是 tensor，跳过 transforms
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        if isinstance(x, np.ndarray):
            if self.transform:
                x = self.transform(eeg=x)['eeg']
            else:
                x = torch.from_numpy(x).float()

        y = int(self.labels[idx])

        # subjects 可能是 tensor 或 numpy
        s = self.subjects[idx]
        if hasattr(s, 'item'):
            s = int(s.item())
        else:
            s = int(s)

        return x, y, s


# =============================================
# 2. 数据加载主函数
# =============================================
def load_data(data_dir: Optional[str] = None) -> Tuple[np.ndarray, ...]:
    """
    加载 BCICIV2a 数据。

    返回: (data, labels, subjects, runs)
      - data: (5184, 22, 800) float64
      - labels: (5184,) int, 1-4
      - subjects: (5184,) int, 1-9
      - runs: (5184,) int, 1-6
    """
    import scipy.io as sio

    if data_dir is None:
        data_dir = DATA_DIR

    # 委托给 download_data 的 ensure_dataset（处理查找/下载/组装）
    from download_data import ensure_dataset
    data_dir = ensure_dataset(data_dir, download_if_missing=True, assemble=True)

    combined = os.path.join(data_dir, DATA_COMBINED_FILE)
    if not os.path.exists(combined):
        raise FileNotFoundError(
            f'Dataset not found at {combined}.\n'
            f'Download from: http://bnci-horizon-2020.eu/database/data-sets/')

    mat = sio.loadmat(combined)
    data = mat['all_sub_data3']
    labels = mat['all_sub_label3'].flatten().astype(int)
    subjects = mat['all_sub_index3'].flatten().astype(int)
    runs = mat.get('all_sub_run3', np.ones_like(labels)).flatten().astype(int)
    print(f'[DATA] Loaded: {combined} — {data.shape}')
    return data, labels, subjects, runs


def get_subject_split(data, labels, subjects,
                       test_subject: int,
                       train_subjects: Optional[List[int]] = None):
    """按受试者拆分训练/测试集"""
    if train_subjects is None:
        train_subjects = [s for s in range(1, 10) if s != test_subject]

    train_mask = np.isin(subjects.flatten(), train_subjects)
    test_mask = subjects.flatten() == test_subject

    return {
        'train': (data[train_mask], labels[train_mask], subjects[train_mask]),
        'test': (data[test_mask], labels[test_mask], subjects[test_mask])
    }


def make_dataloader(data, labels, subjects,
                    model_name: str,
                    batch_size: int = 64,
                    shuffle: bool = True,
                    chunk_size: int = 800,
                    use_augmentation: bool = False,
                    precompute: bool = True,
                    device: str = 'cpu',
                    use_preprocessed: bool = False,
                    data_dir: str = '') -> DataLoader:
    """根据模型类型创建 DataLoader（自动应用正确的 transform）

    Args:
        precompute: 预计算 transforms (GPU 训练时启用，避免每 epoch 重复 FFT)
        device: 目标设备 — 'cuda' 时预计算后数据直接驻留 GPU
        use_preprocessed: 优先从磁盘加载预处理 .pt 文件（跳过 transforms 最快）
        data_dir: 数据目录（use_preprocessed=True 时需要）
    """
    # ===== 优先使用磁盘预处理器数据 =====
    if use_preprocessed and data_dir:
        _pt = check_preprocessed(model_name, data_dir)
        if _pt:
            # 从 subject tensors 提取 subject_id
            _subs = subjects
            if hasattr(_subs, 'numpy'):
                _sub_ids = sorted(set(int(s) for s in _subs.flatten().tolist()))
            else:
                _sub_ids = sorted(set(int(s) for s in np.unique(_subs.flatten())))

            _tdata, _tlabels, _tsubs = load_preprocessed(
                model_name, data_dir, subject_ids=_sub_ids, device=device)

            dataset = CombinedBCIDataset(
                _tdata, _tlabels, _tsubs, precompute=False)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=0, pin_memory=False, drop_last=False)

    # ===== 常规路径（原始 numpy + transforms / 预计算） =====
    _aug = use_augmentation if shuffle else False
    transform = get_transform(model_name, chunk_size, _aug)

    dataset = CombinedBCIDataset(data, labels, subjects,
                                  transform=transform,
                                  precompute=precompute,
                                  precompute_device=device,
                                  model_name=model_name)

    _pin = device == 'cpu'  # 数据已在 GPU 时不用 pin_memory
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=_pin, drop_last=False)


def get_transform(model_name: str, chunk_size: int = 800, use_augmentation: bool = False):
    """根据模型类型获取对应的 transforms"""
    aug_list = []
    if use_augmentation:
        aug_list = [
            transforms.BaselineRemoval(),
            transforms.GaussianNoise(std=0.01),
            transforms.TimeMask(max_mask_size=50),
        ]

    if model_name in ['EEGNet', 'CSPNet', 'LMDA', 'TSCeption']:
        base = aug_list + [transforms.To2d(), transforms.ToTensor()]
        return transforms.Compose(base)
    elif model_name in ['FBCNet', 'FBMSNet']:
        fbc_bands = {
            f'band{i}': [4 * i, 4 * (i + 1)]
            for i in range(1, 10)
        }
        base = aug_list + [
            transforms.BandSignal(sampling_rate=250, band_dict=fbc_bands),
            transforms.ToTensor()
        ]
        return transforms.Compose(base)
    else:
        raise ValueError(f'Unknown model: {model_name}')
