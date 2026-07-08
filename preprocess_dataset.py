"""
数据集预处理脚本 — 将 BCICIV2a 转换为各模型所需的格式并保存到磁盘
=================================================================

功能:
  对原始 EEG 数据应用模型对应的 transforms (To2d / BandSignal 等)，
  将结果保存为 .pt 文件。训练时直接加载预处理好的 tensor，
  彻底省去 CPU 端的 BandSignal FFT 滤波等重计算。

用法:
  python preprocess_dataset.py                          # 预处理所有模型
  python preprocess_dataset.py --models EEGNet FBCNet   # 只处理指定模型
  python preprocess_dataset.py --force                  # 强制重新预处理

输出:
  data/preprocessed/
    EEGNet_data.pt      (5184, 1, 22, 800)    EEGNet / TSCeption 格式
    FBCNet_data.pt      (5184, 9, 22, 800)    FBCNet / FBMSNet 格式
    FBMSNet_data.pt     (5184, 9, 22, 800)    FBMSNet 单独保存
    meta.pt             标签和受试者信息
"""

import os
import sys
import time
import argparse
from typing import List

import numpy as np
import torch

# 兼容性修复: 必须在任何可能加载 torcheeg 的 import 之前执行
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)

import scipy.signal
from scipy.signal.windows import hann, hamming, blackman
scipy.signal.hann = hann
scipy.signal.hamming = hamming
scipy.signal.blackman = blackman

from utils.fixes import apply_all_fixes
apply_all_fixes()

from utils.data_utils import load_data, get_transform, precompute_transforms
from download_data import ensure_dataset


PREPROCESSED_DIR = 'preprocessed'


def get_preprocessed_dir(data_dir: str) -> str:
    """获取预处理数据目录"""
    return os.path.join(data_dir, PREPROCESSED_DIR)


def get_preprocessed_path(model_name: str, data_dir: str) -> str:
    """获取某模型预处理数据的 .pt 文件路径"""
    d = get_preprocessed_dir(data_dir)
    return os.path.join(d, f'{model_name}_data.pt')


def get_meta_path(data_dir: str) -> str:
    """获取元数据 .pt 文件路径"""
    return os.path.join(get_preprocessed_dir(data_dir), 'meta.pt')


def check_preprocessed(model_name: str, data_dir: str) -> bool:
    """检查某模型的预处理数据是否存在"""
    data_path = get_preprocessed_path(model_name, data_dir)
    meta_path = get_meta_path(data_dir)
    return os.path.exists(data_path) and os.path.exists(meta_path)


def preprocess_model(model_name: str, data: np.ndarray,
                     labels: np.ndarray, subjects: np.ndarray,
                     data_dir: str, batch_size: int = 256,
                     force: bool = False):
    """预处理单个模型的数据并保存"""
    out_path = get_preprocessed_path(model_name, data_dir)
    meta_path = get_meta_path(data_dir)

    # 跳过已存在的
    if not force and os.path.exists(out_path):
        print(f'  [SKIP] {model_name} — already exists: {out_path}')
        return

    # 获取 transforms（无 augmentation — 训练时再随机加）
    transform = get_transform(model_name, chunk_size=800, use_augmentation=False)

    print(f'  [PROC] {model_name}: transforming {len(data)} samples... ', end='', flush=True)
    t0 = time.time()

    # 批量预计算（复用 precompute_transforms）
    tensor = precompute_transforms(data, transform, model_name,
                                    batch_size=batch_size, device='cpu')

    # 保存
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(tensor, out_path)

    # 元数据只保存一次
    if not os.path.exists(meta_path):
        meta = {
            'labels': torch.from_numpy(labels.flatten()).long(),
            'subjects': torch.from_numpy(subjects.flatten()).long(),
            'num_samples': len(data),
            'num_classes': len(np.unique(labels)),
            'num_subjects': len(np.unique(subjects)),
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        torch.save(meta, meta_path)
        print(f'meta saved', end='')

    elapsed = time.time() - t0
    print(f' — {elapsed:.1f}s, shape={tuple(tensor.shape)}, saved to {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='BCICIV2a dataset preprocessing')
    parser.add_argument('--models', nargs='+',
                        default=['EEGNet', 'TSCeption', 'FBCNet', 'FBMSNet'],
                        help='Models to preprocess')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--force', action='store_true',
                        help='Force re-preprocess even if files exist')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for transform pipeline')
    args = parser.parse_args()

    # 确保原始数据
    print('[INFO] Ensuring raw dataset...')
    data_dir = ensure_dataset(args.data_dir, download_if_missing=True, assemble=True)
    data_dir = os.path.abspath(data_dir)

    # 加载数据
    print('[INFO] Loading BCICIV2a data...')
    data, labels, subjects, _ = load_data(data_dir)
    print(f'  Data: {data.shape}, Subjects: {list(np.unique(subjects).astype(int))}')

    # 预处理每个模型
    print(f'\n[INFO] Preprocessing {len(args.models)} models...')
    print(f'  Output: {os.path.join(data_dir, PREPROCESSED_DIR)}/')
    print(f'  Force: {args.force}\n')

    t_start = time.time()
    for model_name in args.models:
        preprocess_model(model_name, data, labels, subjects,
                         data_dir, args.batch_size, args.force)

    elapsed = time.time() - t_start
    print(f'\n[OK] Preprocessing done in {elapsed:.1f}s')

    # 显示磁盘占用
    pre_dir = get_preprocessed_dir(data_dir)
    if os.path.exists(pre_dir):
        total_size = 0
        print(f'\n[INFO] Preprocessed files:')
        for f in sorted(os.listdir(pre_dir)):
            fpath = os.path.join(pre_dir, f)
            size_mb = os.path.getsize(fpath) / 1024 ** 2
            total_size += os.path.getsize(fpath)
            print(f'  {f:<30} {size_mb:.1f} MB')
        print(f'  {"TOTAL":<30} {total_size / 1024 ** 2:.1f} MB')

    print(f'\n提示: 训练时 --use-preprocessed 会直接加载这些文件，跳过 transforms')


if __name__ == '__main__':
    main()
