"""
DEAP 数据集预处理 — 支持多类型模型 (Transformer/RNN/GNN/Lightweight/CNN)

为每个模型生成对应输入形状的 .pt 文件:

  4D 通用 (1,32,T)       → Conformer, LMDA, CSPNet, LGGNet, TSLANet
  3D 时序 (32,T)          → VanillaTransformer, LSTM, GRU
  3D 节点特征 (32,F)      → DGCNN
  4D 网格 (T,9,9)         → STNet
  4D 特征网格 (8,8,9)     → MTCNN
  4D 特征网格 (36,16,16)  → SSTEmotionNet

用法:
    python preprocess.py --models all
    python preprocess.py --models Conformer LSTM
    python preprocess.py --models all --chunk-size 256
"""

import os
import sys
import time
import pickle
import argparse
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.signal import butter, sosfilt

from torcheeg import transforms as T
from torcheeg.datasets.constants import (
    DEAP_CHANNEL_LOCATION_DICT,
    DEAP_CHANNEL_LIST,
)

from config import *
from utils import format_channel_location_dict

warnings.filterwarnings('ignore')


# ── MTCNN 专用 8×9 电极布局 → location dict ──
MTCNN_LOCATION_DICT = format_channel_location_dict(
    DEAP_CHANNEL_LIST, MTCNN_GRID_8x9,
)


def get_transform(model_name: str) -> callable:
    """获取模型对应的 TorchEEG transform

    Returns:
        transform: 接收 (eeg: ndarray of (32, chunk_size)) 返回 dict
    """
    # ── Group 1: 4D 通用输入 (1, 32, T) ──
    if model_name in ('Conformer', 'LMDA', 'CSPNet', 'LGGNet'):
        return T.Compose([T.To2d(), T.ToTensor()])

    # ── Group 2: 3D 时序输入 (32, T) ──
    # TSLANet 期望 (batch, num_electrodes, chunk_size) = (32, 128)
    elif model_name in ('VanillaTransformer', 'LSTM', 'GRU', 'TSLANet'):
        return T.Compose([T.ToTensor()])

    # ── Group 3: 节点特征输入 (32, F) — 需后处理 ──
    elif model_name == 'DGCNN':
        # BandSignal → 输出 (9, 32, T); 后处理转方差特征
        return T.Compose([
            T.BandSignal(sampling_rate=DEAP_SAMPLING_RATE, band_dict=FBC_BANDS),
            T.ToTensor(),
        ])

    # ── Group 4: 网格输入 (T, 9, 9) ──
    elif model_name == 'STNet':
        return T.Compose([
            T.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
            T.ToTensor(),
        ])

    # ── Group 5: MTCNN — DE + PSD → Concatenate → ToGrid(8×9) ──
    elif model_name == 'MTCNN':
        return T.Compose([
            T.Concatenate([
                T.Compose([
                    T.BandDifferentialEntropy(sampling_rate=DEAP_SAMPLING_RATE),
                ]),
                T.Compose([
                    T.BandPowerSpectralDensity(sampling_rate=DEAP_SAMPLING_RATE),
                ]),
            ]),
            T.ToGrid(MTCNN_LOCATION_DICT),
            T.ToTensor(),
        ])

    # ── Group 6: SSTEmotionNet — DE + Downsample → Concatenate → ToInterpolatedGrid → Resize ──
    elif model_name == 'SSTEmotionNet':
        return T.Compose([
            T.Concatenate([
                T.Compose([
                    T.BandDifferentialEntropy(sampling_rate=DEAP_SAMPLING_RATE),
                    T.MeanStdNormalize(),
                ]),
                T.Compose([
                    T.Downsample(num_points=32),
                    T.MinMaxNormalize(),
                ]),
            ]),
            T.ToInterpolatedGrid(DEAP_CHANNEL_LOCATION_DICT),
            T.ToTensor(),
            T.Resize(size=(16, 16)),
        ])

    else:
        raise ValueError(f'Unknown model: {model_name}')


def _bandpass_trials(trials_data: np.ndarray,
                     band_dict: Dict[str, List[int]],
                     sampling_rate: int = 128,
                     order: int = 4) -> Dict[str, np.ndarray]:
    """对完整 trials 应用带通滤波"""
    results = {}
    for band_name, (low, high) in band_dict.items():
        sos = butter(order, [low, high], btype='band',
                     fs=sampling_rate, output='sos')
        filtered = np.zeros_like(trials_data)
        for ti in range(trials_data.shape[0]):
            for ch in range(trials_data.shape[1]):
                filtered[ti, ch] = sosfilt(sos, trials_data[ti, ch])
        results[band_name] = filtered
    return results


def postprocess_dgcnn(transformed: torch.Tensor) -> torch.Tensor:
    """DGCNN 后处理: (N, 9, 32, T) → 时域方差 → (N, 32, 9)"""
    var = transformed.var(dim=3, keepdim=False)  # (N, 9, 32)
    result = var.permute(0, 2, 1)  # (N, 32, 9)
    return result.contiguous()


def precompute_model_data(windows: np.ndarray,
                           model_name: str,
                           batch_size: int = 256,
                           device: str = 'cpu',
                           full_trials: Optional[np.ndarray] = None,
                           n_windows_per_trial: int = 63) -> torch.Tensor:
    """对一个模型应用 transforms, 返回 tensor

    Args:
        windows: (N, 32, chunk_size) numpy array
        model_name: 模型名
        batch_size: 批大小
        device: 计算设备
        full_trials: (n_trials, 32, 8064) — 用于 BandSignal 快速路径
        n_windows_per_trial: 每 trial 窗口数

    Returns:
        torch.Tensor — 模型对应的输入形状
    """
    # ── DGCNN 特殊路径: 先提取时域方差特征 ──
    if model_name == 'DGCNN':
        t0 = time.time()
        print(f'[PREP] DGCNN: computing band features...', flush=True)

        if full_trials is not None:
            n_trials = full_trials.shape[0]
            band_data = _bandpass_trials(full_trials, FBC_BANDS,
                                         sampling_rate=DEAP_SAMPLING_RATE)

            n_bands = len(band_data)
            n_ch = full_trials.shape[1]
            n_total = n_trials * n_windows_per_trial
            band_var = np.zeros((n_total, n_bands, n_ch), dtype=np.float32)

            for bi, (band_name, fdata) in enumerate(band_data.items()):
                idx = 0
                for ti in range(n_trials):
                    for wi in range(n_windows_per_trial):
                        start = wi * DEAP_CHUNK_SIZE
                        end = start + DEAP_CHUNK_SIZE
                        seg = fdata[ti, :, start:end]
                        band_var[idx, bi] = seg.var(axis=1)
                        idx += 1
            result = torch.from_numpy(band_var.transpose(0, 2, 1)).float()
        else:
            n = len(windows)
            all_var = []
            for i in range(0, n, batch_size):
                batch = windows[i:i + batch_size]
                batch_var = []
                for sample in batch:
                    result_bs = T.BandSignal(sampling_rate=DEAP_SAMPLING_RATE,
                                              band_dict=FBC_BANDS)(eeg=sample)
                    bs_tensor = T.ToTensor()(eeg=result_bs['eeg'])['eeg']
                    var = bs_tensor.var(dim=2)
                    batch_var.append(var.permute(1, 0))
                all_var.append(torch.stack(batch_var))
                processed = min(i + batch_size, n)
                if processed % max(1, n // 10) == 0:
                    print(f'  [{processed}/{n}]', flush=True)
            result = torch.cat(all_var, dim=0)

        elapsed = time.time() - t0
        print(f'[PREP] DGCNN done ({elapsed:.1f}s), '
              f'shape={tuple(result.shape)}', flush=True)
        if device != 'cpu':
            result = result.to(device)
        return result.float()

    # ── 通用路径 ──
    n = len(windows)
    transform = get_transform(model_name)
    t0 = time.time()
    print(f'[PREP] {model_name}: transforming {n} samples...', flush=True)

    transformed_batches = []
    log_interval = max(1, n // 20)

    for i in range(0, n, batch_size):
        batch = windows[i:i + batch_size]
        batch_out = []
        for sample in batch:
            result = transform(eeg=sample)['eeg']
            batch_out.append(result)
        tb = torch.stack(batch_out).contiguous().float()
        transformed_batches.append(tb)

        processed = min(i + batch_size, n)
        if processed % log_interval == 0 or processed == n:
            elapsed = time.time() - t0
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = (n - processed) / speed if speed > 0 else 0
            print(f'  [{processed}/{n}] {speed:.0f} samples/s  '
                  f'est: {remaining/60:.1f}min', flush=True)

    result = torch.cat(transformed_batches, dim=0)
    elapsed = time.time() - t0
    print(f'[PREP] {model_name} done ({elapsed:.1f}s), '
          f'shape={tuple(result.shape)}', flush=True)

    if device != 'cpu':
        result = result.to(device)
    return result


def process_all(data_dir: str,
                model_names: List[str],
                chunk_size: int = 128,
                overlap: int = 0,
                output_dir: str = '',
                num_subjects: Optional[int] = None,
                device: str = 'cpu') -> Dict[str, str]:
    """主处理函数

    加载 DEAP .dat → 分割窗口 → 为每个模型应用 transforms → 保存 .pt
    """
    if output_dir:
        preproc_dir = output_dir
    else:
        preproc_dir = os.path.join(os.path.dirname(os.path.abspath(data_dir)),
                                   'deap_preprocessed')
    os.makedirs(preproc_dir, exist_ok=True)

    print(f'[PREP] Output: {preproc_dir}')
    print(f'[PREP] Window: {chunk_size}pt (overlap={overlap})')

    # ── 加载原始数据 ──
    all_eeg = []
    all_labels = []
    all_subject_ids = []
    all_trial_ids = []

    dat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.dat')])
    if num_subjects is not None:
        dat_files = dat_files[:num_subjects]

    print(f'[PREP] Loading {len(dat_files)} subjects...')
    for fi, fname in enumerate(dat_files):
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'rb') as f:
            sub_data = pickle.load(f, encoding='latin1')
        eeg = sub_data['data'][:, :DEAP_NUM_CHANNELS, :]
        labels = sub_data['labels']
        all_eeg.append(eeg)
        all_labels.append(labels)
        subj_id = int(fname.replace('s', '').replace('.dat', ''))
        all_subject_ids.extend([subj_id] * eeg.shape[0])
        all_trial_ids.extend(list(range(1, eeg.shape[0] + 1)))
        if (fi + 1) % 5 == 0:
            print(f'  Loaded {fi+1}/{len(dat_files)} subjects...')

    data_all = np.concatenate(all_eeg, axis=0)        # (N*40, 32, 8064)
    labels_all = np.concatenate(all_labels, axis=0)    # (N*40, 4)
    subject_ids = np.array(all_subject_ids, dtype=np.int64)
    trial_ids = np.array(all_trial_ids, dtype=np.int64)

    # ── 切分窗口 ──
    step = chunk_size - overlap
    n_trials, n_ch, n_times = data_all.shape
    n_windows_per_trial = (n_times - chunk_size) // step + 1
    n_total_windows = n_trials * n_windows_per_trial

    print(f'[PREP] Chunking {n_trials} trials × {n_windows_per_trial} windows...')

    windows = np.zeros((n_total_windows, n_ch, chunk_size), dtype=np.float32)
    expanded_labels_v = np.zeros(n_total_windows, dtype=np.float64)
    expanded_subjects = np.zeros(n_total_windows, dtype=np.int64)
    expanded_trials = np.zeros(n_total_windows, dtype=np.int64)

    idx = 0
    for ti in range(n_trials):
        trial_data = data_all[ti]
        for wi in range(n_windows_per_trial):
            start = wi * step
            windows[idx] = trial_data[:, start:start + chunk_size]
            expanded_labels_v[idx] = labels_all[ti, 0]  # valence
            expanded_subjects[idx] = subject_ids[ti]
            expanded_trials[idx] = trial_ids[ti]
            idx += 1

    # ── 二值化标签 ──
    labels_binary = (expanded_labels_v > 5.0).astype(np.int64)
    print(f'[PREP] Total: {n_total_windows} windows')
    cls0 = (labels_binary == 0).sum()
    cls1 = (labels_binary == 1).sum()
    print(f'       Class 0 (low):  {cls0} ({cls0/n_total_windows*100:.1f}%)')
    print(f'       Class 1 (high): {cls1} ({cls1/n_total_windows*100:.1f}%)')

    # ── 保存元数据 ──
    meta = {
        'labels': torch.from_numpy(labels_binary),
        'subjects': torch.from_numpy(expanded_subjects),
        'trial_ids': torch.from_numpy(expanded_trials),
        'n_windows_per_trial': n_windows_per_trial,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'num_channels': DEAP_NUM_CHANNELS,
        'sampling_rate': DEAP_SAMPLING_RATE,
        'window_step': step,
    }
    meta_path = os.path.join(preproc_dir, 'meta.pt')
    torch.save(meta, meta_path)
    print(f'[PREP] Meta saved: {meta_path}')

    # ── 按模型预处理 ──
    results = {}
    for model_name in model_names:
        print(f'\n--- {model_name} ---')

        if model_name == 'DGCNN':
            transformed = precompute_model_data(
                windows, model_name, device=device,
                full_trials=data_all, n_windows_per_trial=n_windows_per_trial)
        else:
            transformed = precompute_model_data(
                windows, model_name, device=device)

        data_path = os.path.join(preproc_dir, f'{model_name}_data.pt')
        torch.save(transformed.contiguous(), data_path)
        print(f'[PREP] Saved: {data_path}  shape={tuple(transformed.shape)}')
        results[model_name] = data_path

    # 打印模型输入形状汇总
    print(f'\n{"="*55}')
    print(f'  模型输入形状汇总')
    print(f'{"="*55}')
    for model_name in model_names:
        data_path = os.path.join(preproc_dir, f'{model_name}_data.pt')
        if os.path.exists(data_path):
            t = torch.load(data_path, map_location='cpu', weights_only=True)
            print(f'  {model_name:25s} → {str(tuple(t.shape)):20s}')

    print(f'\n[PREP] All done! Files in: {preproc_dir}')
    return results


def main():
    parser = argparse.ArgumentParser(
        description='DEAP 多模型预计算 transforms')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='DEAP .dat 文件目录')
    parser.add_argument('--models', type=str, nargs='+',
                        default=AVAILABLE_MODELS,
                        choices=AVAILABLE_MODELS + ['all'],
                        help='要预处理的模型列表')
    parser.add_argument('--chunk-size', type=int, default=128,
                        choices=[128, 256])
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--num-subjects', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir or DEFAULT_DATA_DIR
    if not os.path.exists(data_dir):
        print(f'[ERROR] DEAP data not found: {data_dir}')
        sys.exit(1)

    models = args.models
    if 'all' in models:
        models = AVAILABLE_MODELS

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f'[PREP] GPU: {torch.cuda.get_device_name(0)}')

    process_all(
        data_dir=data_dir,
        model_names=models,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        output_dir=args.output_dir,
        num_subjects=args.num_subjects,
        device=device,
    )


if __name__ == '__main__':
    main()
