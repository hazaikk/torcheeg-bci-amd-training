"""
DEAP 数据集预处理脚本
======================
将原始 DEAP .dat 文件批量转换 transforms 并保存为 .pt 文件。

用法:
    python preprocess_deap.py --models EEGNet TSCeption FBCNet FBMSNet CCNN
    python preprocess_deap.py --models EEGNet --chunk-size 256 --data-dir /path/to/deap
    python preprocess_deap.py --models all --offline de
    python preprocess_deap.py --models FBCNet --offline none

输出:
    data/deap/preprocessed/EEGNet_data.pt     — (N, 1, 32, 128)  tensor
    data/deap/preprocessed/FBCNet_data.pt     — (N, 9, 32, 128)  tensor (auto/bandpass)
    data/deap/preprocessed/FBCNet_data.pt     — (N, n_bands, 32, 128)  tensor (de/none)
    data/deap/preprocessed/meta.pt            — 标签 + 元数据

设计思路:
    - 预处理后的数据直接加载为 torch tensor，训练时跳过所有 transforms
    - 和 BCIC 的 preprocess_dataset.py 风格一致
    - 保留 subject_id / trial_id 信息，支持各种交叉验证策略
"""

import os
import sys
import time
import pickle
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.signal
from scipy.signal import butter, sosfilt
from scipy.signal.windows import hann, hamming, blackman
scipy.signal.hann = hann
scipy.signal.hamming = hamming
scipy.signal.blackman = blackman

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)

from utils.fixes import apply_all_fixes
apply_all_fixes()

from torcheeg import transforms as T
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT

# ── 常量 ──
DEAP_SAMPLING_RATE = 128
DEAP_NUM_CHANNELS = 32
PREPROCESSED_DIR = 'preprocessed'

# 可用模型 (与 train_deap.py 保持一致)
AVAILABLE_MODELS = ['EEGNet', 'TSCeption', 'FBCNet', 'FBMSNet', 'CCNN']


def get_transform(model_name: str, chunk_size: int = 128,
                  offline_type: str = 'auto') -> callable:
    """获取模型对应的数据 transform

    Args:
        model_name: 模型名
        chunk_size: 时间窗口样本数
        offline_type: 离线变换类型
            'auto'     = 各模型默认 (CCNN→ToGrid, EEGNet→To2d, FBCNet→BandSignal)
            'de'       = BandDifferentialEntropy
            'bandpass' = BandSignal (9 频带)
            'none'     = 无频带分解, 仅基础格式转换

    Returns:
        transform: 接收 (eeg: np.ndarray of (32, chunk_size)) 返回 tensor
    """
    fbc_bands = {
        f'band{i}': [4 * i, 4 * (i + 1)]
        for i in range(1, 10)
    }

    if model_name == 'CCNN':
        # CCNN: (32, chunk_size) → (chunk_size, 9, 9)
        if offline_type == 'de':
            return T.Compose([
                T.BandDifferentialEntropy(),
                T.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
                T.ToTensor(),
            ])
        elif offline_type == 'bandpass':
            return T.Compose([
                T.BandSignal(sampling_rate=DEAP_SAMPLING_RATE, band_dict=fbc_bands),
                T.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
                T.ToTensor(),
            ])
        else:  # 'auto' or 'none'
            return T.Compose([
                T.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
                T.ToTensor(),
            ])

    elif model_name in ('EEGNet', 'TSCeption'):
        # (32, chunk_size) → (1, 32, chunk_size)
        if offline_type == 'de':
            return T.Compose([
                T.BandDifferentialEntropy(),
                T.To2d(),
                T.ToTensor(),
            ])
        elif offline_type == 'bandpass':
            return T.Compose([
                T.BandSignal(sampling_rate=DEAP_SAMPLING_RATE, band_dict=fbc_bands),
                T.To2d(),
                T.ToTensor(),
            ])
        else:  # 'auto' or 'none'
            return T.Compose([
                T.To2d(),
                T.ToTensor(),
            ])

    elif model_name in ('FBCNet', 'FBMSNet'):
        # (32, chunk_size) → (n_bands, 32, chunk_size)
        if offline_type in ('auto', 'bandpass'):
            return T.Compose([
                T.BandSignal(sampling_rate=DEAP_SAMPLING_RATE, band_dict=fbc_bands),
                T.ToTensor(),
            ])
        elif offline_type == 'de':
            return T.Compose([
                T.BandDifferentialEntropy(),
                T.ToTensor(),
            ])
        else:  # 'none'
            # 无频带分解: 加回通道维度作为伪频带
            return T.Compose([
                T.ToTensor(),
            ])
    else:
        raise ValueError(f'Unknown model: {model_name}')


def _bandpass_trials(trials_data: np.ndarray,
                     band_dict: Dict[str, List[int]],
                     sampling_rate: int = 128,
                     order: int = 4) -> Dict[str, np.ndarray]:
    """对完整 trials 应用带通滤波 (比 per-window BandSignal 快 ~60倍)

    用 scipy.sosfilt (second-order sections) 滤波, 数值稳定性更好。

    Args:
        trials_data: (n_trials, n_channels, n_times) — 未切窗的原始 trial
        band_dict: {'band1': [4,8], 'band2': [8,12], ...}
        sampling_rate: 采样率
        order: 滤波器阶数

    Returns:
        {band_name: filtered_data (n_trials, n_channels, n_times)}
    """
    results = {}
    for band_name, (low, high) in band_dict.items():
        # 设计一次滤波器, 应用到所有 trial
        sos = butter(order, [low, high], btype='band', fs=sampling_rate, output='sos')
        filtered = np.zeros_like(trials_data)
        for ti in range(trials_data.shape[0]):
            # (n_channels, n_times) 每通道独立滤波
            for ch in range(trials_data.shape[1]):
                filtered[ti, ch] = sosfilt(sos, trials_data[ti, ch])
        results[band_name] = filtered
        print(f'    {band_name} ({low}-{high}Hz): filtered {trials_data.shape}', flush=True)
    return results


def precompute_transforms(data: np.ndarray, transform,
                           model_name: str, chunk_size: int,
                           batch_size: int = 256,
                           device: str = 'cpu',
                           full_trials: Optional[np.ndarray] = None,
                           n_windows_per_trial: int = 63) -> torch.Tensor:
    # FBCNet/FBMSNet 的 BandSignal 内存占用大, 缩小 batch
    if model_name in ('FBCNet', 'FBMSNet'):
        batch_size = min(batch_size, 128)
    """对完整数据集批量预应用 transforms

    Args:
        data: numpy (N, 32, chunk_size) — 已切窗的数据
        transform: torcheeg transforms
        model_name: 仅用于日志
        chunk_size: 窗口大小
        batch_size: 批大小
        device: 计算设备
        full_trials: numpy (n_trials, 32, 8064) 未切窗的原始 trial
                     仅 FBCNet/FBMSNet 使用, 先滤波再切窗加速
        n_windows_per_trial: 每 trial 的窗口数

    Returns:
        torch.float32 tensor
    """
    # ── FBCNet/FBMSNet 加速: 先滤波整段 trial, 再切窗 ──
    if model_name in ('FBCNet', 'FBMSNet') and full_trials is not None:
        n_trials = full_trials.shape[0]
        t0 = time.time()
        print(f'[PREP] {model_name}: bandpass filtering {n_trials} full trials...', flush=True)

        fbc_bands = {
            f'band{i}': [4 * i, 4 * (i + 1)]
            for i in range(1, 10)
        }
        band_data = _bandpass_trials(full_trials, fbc_bands,
                                     sampling_rate=DEAP_SAMPLING_RATE)

        # 切窗: 每个 filtered trial → n_windows_per_trial 个窗口
        n_total = n_trials * n_windows_per_trial
        n_bands = len(band_data)
        n_ch = full_trials.shape[1]
        result = np.zeros((n_total, n_bands, n_ch, chunk_size), dtype=np.float32)

        for bi, (band_name, fdata) in enumerate(band_data.items()):
            idx = 0
            for ti in range(n_trials):
                for wi in range(n_windows_per_trial):
                    start = wi * chunk_size
                    end = start + chunk_size
                    result[idx, bi] = fdata[ti, :, start:end]
                    idx += 1

        result_tensor = torch.from_numpy(result)
        elapsed = time.time() - t0
        print(f'[PREP] {model_name} done ({elapsed:.1f}s), '
              f'shape={tuple(result_tensor.shape)}, dtype={result_tensor.dtype}',
              flush=True)

        if device != 'cpu':
            result_tensor = result_tensor.to(device)
        return result_tensor

    # ── 通用路径 (EEGNet, TSCeption, CCNN) ──
    n = len(data)
    t0 = time.time()
    print(f'[PREP] {model_name}: transforming {n} samples...', flush=True)

    transformed_batches = []
    log_interval = max(1, n // 20)  # 每 5% 打印一次

    for i in range(0, n, batch_size):
        batch = data[i:i + batch_size]
        batch_out = []
        for j, sample in enumerate(batch):
            result = transform(eeg=sample)['eeg']
            batch_out.append(result)
        tb = torch.stack(batch_out).contiguous().float()
        transformed_batches.append(tb)

        # 进度输出
        processed = min(i + batch_size, n)
        if processed % log_interval == 0 or processed == n:
            elapsed = time.time() - t0
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = (n - processed) / speed if speed > 0 else 0
            print(f'  [{processed}/{n}] {100*processed//n}%  '
                  f'{speed:.0f} samples/s  '
                  f'est: {remaining/60:.1f}min remaining',
                  flush=True)

    result = torch.cat(transformed_batches, dim=0)
    elapsed = time.time() - t0

    if device != 'cpu':
        result = result.to(device)

    print(f'[PREP] {model_name} done ({elapsed:.1f}s), '
          f'shape={tuple(result.shape)}, dtype={result.dtype}')
    return result


def process_deap(data_dir: str,
                 model_names: List[str],
                 chunk_size: int = 128,
                 overlap: int = 0,
                 num_channel: int = 32,
                 output_dir: str = '',
                 num_subjects: Optional[int] = None,
                 device: str = 'cpu',
                 offline_type: str = 'auto') -> Dict[str, str]:
    """主处理函数

    加载原始 DEAP .dat 文件 → 分割窗口 → 应用 transforms → 保存 .pt

    Args:
        data_dir: 包含 s01.dat ~ s32.dat 的目录
        model_names: 要处理的模型列表
        chunk_size: 窗口样本数 (128=1s, 256=2s)
        overlap: 窗口重叠
        num_channel: 使用的 EEG 通道数
        output_dir: 输出目录 (默认 data_dir/preprocessed)
        num_subjects: 处理的受试者数 (None=全部)
        device: 计算设备

    Returns:
        dict: {model_name: saved_path}
    """
    if output_dir:
        preproc_dir = output_dir
    else:
        preproc_dir = os.path.join(os.path.dirname(data_dir), 'deap_preprocessed')

    os.makedirs(preproc_dir, exist_ok=True)
    print(f'[PREP] Output dir: {preproc_dir}')
    print(f'[PREP] Window: {chunk_size}pt (overlap={overlap})')

    # ── 加载原始数据 ──
    all_eeg = []      # list of (trials, 32, 8064) per subject
    all_labels = []   # list of (trials, 4) per subject (valence/arousal/dominance/liking)
    all_subject_ids = []
    all_trial_ids = []

    # 枚举受试者 .dat 文件
    dat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.dat')])
    if num_subjects is not None:
        dat_files = dat_files[:num_subjects]

    print(f'[PREP] Loading {len(dat_files)} subjects...')
    for fi, fname in enumerate(dat_files):
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'rb') as f:
            sub_data = pickle.load(f, encoding='latin1')

        eeg = sub_data['data']  # (40, 40, 8064)
        labels = sub_data['labels']  # (40, 4)

        # 只用前 num_channel 个 EEG 通道
        eeg = eeg[:, :num_channel, :]  # (40, num_channel, 8064)

        all_eeg.append(eeg)
        all_labels.append(labels)
        subj_id = int(fname.replace('s', '').replace('.dat', ''))
        all_subject_ids.extend([subj_id] * eeg.shape[0])
        all_trial_ids.extend(list(range(1, eeg.shape[0] + 1)))

        if (fi + 1) % 5 == 0:
            print(f'  Loaded {fi+1}/{len(dat_files)} subjects...')

    # 拼接所有受试者
    data_all = np.concatenate(all_eeg, axis=0)  # (N*40, 32, 8064)
    labels_all = np.concatenate(all_labels, axis=0)  # (N*40, 4)
    subject_ids = np.array(all_subject_ids, dtype=np.int64)
    trial_ids = np.array(all_trial_ids, dtype=np.int64)

    # ── 切分窗口 ──
    step = chunk_size - overlap
    n_trials, n_ch, n_times = data_all.shape
    n_windows_per_trial = (n_times - chunk_size) // step + 1

    print(f'[PREP] Chunking {n_trials} trials × {n_windows_per_trial} windows '
          f'(step={step}, chunk={chunk_size})...')

    # 预分配窗口数组
    n_total_windows = n_trials * n_windows_per_trial
    windows = np.zeros((n_total_windows, n_ch, chunk_size), dtype=np.float64)

    # 扩展标签和元数据
    expanded_labels_v = np.zeros(n_total_windows, dtype=np.float64)  # valence
    expanded_subjects = np.zeros(n_total_windows, dtype=np.int64)
    expanded_trials = np.zeros(n_total_windows, dtype=np.int64)

    idx = 0
    for ti in range(n_trials):
        trial_data = data_all[ti]  # (32, 8064)
        trial_label_v = labels_all[ti, 0]  # valence
        trial_subj = subject_ids[ti]
        trial_id = trial_ids[ti]

        for wi in range(n_windows_per_trial):
            start = wi * step
            end = start + chunk_size
            windows[idx] = trial_data[:, start:end]
            expanded_labels_v[idx] = trial_label_v
            expanded_subjects[idx] = trial_subj
            expanded_trials[idx] = trial_id
            idx += 1

    assert idx == n_total_windows, f'{idx} != {n_total_windows}'

    print(f'[PREP] Total windows: {n_total_windows}')

    # ── 标签二值化: valence > 5 → 1 (high), else → 0 (low) ──
    labels_binary = (expanded_labels_v > 5.0).astype(np.int64)

    # ── 保存元数据 ──
    meta = {
        'labels': torch.from_numpy(labels_binary),
        'labels_continuous': torch.from_numpy(expanded_labels_v),
        'subjects': torch.from_numpy(expanded_subjects),
        'trial_ids': torch.from_numpy(expanded_trials),
        'n_windows_per_trial': n_windows_per_trial,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'num_channels': num_channel,
        'sampling_rate': DEAP_SAMPLING_RATE,
        'window_step': step,
    }
    meta_path = os.path.join(preproc_dir, 'meta.pt')
    torch.save(meta, meta_path)
    print(f'[PREP] Meta saved: {meta_path}')

    # ── 按模型预处理并保存 ──
    # 注意: FBCNet/FBMSNet 用 data_all (未切窗的完整 trial) 先滤波再切窗, 提速 ~60x
    results = {}
    windows_f32 = windows.astype(np.float32)

    for model_name in model_names:
        print()
        transform = get_transform(model_name, chunk_size, offline_type)
        # 快速路径: 仅 FBCNet/FBMSNet + bandpass 模式用 full_trials 先滤波再切窗
        use_fast_path = (model_name in ('FBCNet', 'FBMSNet')
                         and offline_type in ('auto', 'bandpass'))
        transformed = precompute_transforms(
            windows_f32, transform, model_name, chunk_size,
            device=device,
            full_trials=data_all if use_fast_path else None,
            n_windows_per_trial=n_windows_per_trial)

        data_path = os.path.join(preproc_dir, f'{model_name}_data.pt')
        torch.save(transformed.contiguous(), data_path)
        print(f'[PREP] {model_name} data saved: {data_path} '
              f'shape={tuple(transformed.shape)}')
        results[model_name] = data_path

    # 释放内存
    del data_all, all_eeg, windows, windows_f32

    print(f'\n[PREP] All done! Files saved to: {preproc_dir}')
    print(f'  Meta: meta.pt')
    for m in model_names:
        print(f'  {m}: {m}_data.pt')

    return results


def main():
    parser = argparse.ArgumentParser(
        description='DEAP 数据集预计算 transforms')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='DEAP .dat 文件目录 (默认 auto-detect)')
    parser.add_argument('--models', type=str, nargs='+',
                        default=AVAILABLE_MODELS,
                        choices=AVAILABLE_MODELS + ['all'],
                        help='要预处理的模型列表')
    parser.add_argument('--chunk-size', type=int, default=128,
                        choices=[128, 256],
                        help='时间窗口样本数 (128=1s, 256=2s)')
    parser.add_argument('--overlap', type=int, default=0,
                        help='窗口重叠')
    parser.add_argument('--num-subjects', type=int, default=None,
                        help='处理的受试者数 (默认全部 32)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--offline', type=str, default='auto',
                        choices=['auto', 'de', 'bandpass', 'none'],
                        help='离线变换类型 (auto=各模型默认, de=微分熵, '
                             'bandpass=9频带滤波, none=仅格式转换)')
    parser.add_argument('--gpu', action='store_true',
                        help='使用 GPU 加速预计算')
    args = parser.parse_args()

    # 数据目录
    if args.data_dir is None:
        from download_deap import get_deap_path
        data_dir = get_deap_path()
    else:
        data_dir = args.data_dir

    if not os.path.exists(data_dir):
        print(f'[ERROR] DEAP data not found: {data_dir}')
        sys.exit(1)

    # 模型列表
    models = args.models
    if 'all' in models:
        models = AVAILABLE_MODELS

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f'[PREP] Using GPU: {torch.cuda.get_device_name(0)}')

    print(f'[PREP] Offline transform: {args.offline}')

    process_deap(
        data_dir=data_dir,
        model_names=models,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        output_dir=args.output_dir,
        num_subjects=args.num_subjects,
        device=device,
        offline_type=args.offline,
    )


if __name__ == '__main__':
    main()
