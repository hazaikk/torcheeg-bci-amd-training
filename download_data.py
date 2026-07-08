"""
数据下载模块
=============
从 BNCI Horizon 2020 下载 BCICIV2a 单受试者 .mat 文件，
并可选组装为预合并格式 BCICIV2a.mat。
"""

import os
import sys
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import scipy.io as sio
import scipy.signal
from scipy.signal.windows import hann, hamming, blackman
scipy.signal.hann = hann
scipy.signal.hamming = hamming
scipy.signal.blackman = blackman

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, BNCI_BASE_URL, SUBJECTS, SESSIONS, DATA_COMBINED_FILE


def _ssl_ctx():
    """创建宽松的 SSL 上下文（某些环境证书不全）"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _progress_hook(block_num, block_size, total_size):
    """简单的下载进度回调"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        bar = '#' * (percent // 5) + '.' * (20 - percent // 5)
        print(f'\r  [{bar}] {percent:3d}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)',
              end='', flush=True)
        if percent >= 100:
            print()


def download_single_file(url: str, save_path: str) -> bool:
    """下载单个 .mat 文件（使用 urlopen + write，兼容各 Python 版本）"""
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
        print(f'  [SKIP] {os.path.basename(save_path)} already exists')
        return True
    try:
        print(f'  [DL] {os.path.basename(save_path)}', end='', flush=True)
        ctx = _ssl_ctx()
        resp = urllib.request.urlopen(url, timeout=300, context=ctx)
        total_size = int(resp.headers.get('Content-Length', 0))
        chunk_size = 8192
        downloaded = 0
        last_pct = -1
        with open(save_path, 'wb') as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded * 100 // total_size
                    if pct >= last_pct + 5:  # 每 5% 打印一次
                        print(f'...{pct}%', end='', flush=True)
                        last_pct = pct
        if total_size > 0:
            print(f'...100% ({downloaded/1024/1024:.1f} MB)')
        else:
            print(f'...{downloaded/1024/1024:.1f} MB')
        return os.path.getsize(save_path) > 1024
    except Exception as e:
        print(f'  [FAIL] {os.path.basename(save_path)}: {e}')
        return False


def download_individual_files(data_dir: Optional[str] = None,
                               max_workers: int = 4) -> bool:
    """下载 18 个单受试者 .mat 文件 (A01T ~ A09E)"""
    if data_dir is None:
        data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    tasks = []
    for subj in SUBJECTS:
        for sess in SESSIONS:
            filename = f'{subj}{sess}.mat'
            url = f'{BNCI_BASE_URL}{filename}'
            save_path = os.path.join(data_dir, filename)
            tasks.append((url, save_path))

    print(f'Downloading {len(tasks)} files to {os.path.abspath(data_dir)} ...')
    success = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_single_file, url, path): path
                   for url, path in tasks}
        for f in as_completed(futures):
            if f.result():
                success += 1

    print(f'\nDownloaded {success}/{len(tasks)} files successfully.')
    return success == len(tasks)


def assemble_combined_mat(data_dir: Optional[str] = None,
                           save_path: Optional[str] = None,
                           chunk_size: int = 800) -> str:
    """
    将 18 个单受试者 .mat 文件组装为预合并格式 BCICIV2a.mat。

    返回生成的 .mat 文件绝对路径。
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if save_path is None:
        save_path = os.path.join(data_dir, DATA_COMBINED_FILE)

    if os.path.exists(save_path):
        print(f'[SKIP] {save_path} already exists')
        return os.path.abspath(save_path)

    print('Assembling combined BCICIV2a.mat from individual files...')

    all_data = []
    all_labels = []
    all_subjects = []
    all_runs = []

    for subj_idx, subj in enumerate(SUBJECTS):
        for sess_idx, sess in enumerate(SESSIONS):
            file_path = os.path.join(data_dir, f'{subj}{sess}.mat')
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f'Missing file: {file_path}. Run download_data.py first.')

            mat = sio.loadmat(file_path)
            a_data = mat['data']

            # 文件有 7-9 列 (run 0~8), 部分受试者可能只有 7 列 (run 0~6)
            # 找出实际包含 trial 数据的 runs
            num_runs = a_data.shape[1]
            available_run_ids = []
            for run_id in range(num_runs):
                run_struct = a_data[0, run_id][0, 0]
                trial_markers = run_struct[1].flatten()
                if len(trial_markers) > 0:
                    available_run_ids.append(run_id)

            if len(available_run_ids) == 0:
                print(f'  [WARN] No MI runs found in {file_path}, skipping')
                continue

            # 标准: runs 3-8 是 MI runs, 如果文件列数不同则用检测到的
            if len(available_run_ids) == 6:
                run_iter = available_run_ids  # 直接用检测到的 6 个 runs
            else:
                run_iter = [r for r in range(3, num_runs) if r < num_runs]

            for run_id in run_iter:
                run_struct = a_data[0, run_id][0, 0]
                X = run_struct[0]           # (time, 25)
                trial_markers = run_struct[1].flatten()  # (48,)
                y = run_struct[2].flatten()  # (48,)

                if len(trial_markers) == 0:
                    continue  # 无 trial 数据则跳过

                for t_id in range(len(trial_markers)):
                    start = int(trial_markers[t_id])
                    end = start + chunk_size
                    trial_data = X[start:end, :22]  # only EEG channels
                    if trial_data.shape[0] < chunk_size:
                        continue  # skip incomplete trials

                    all_data.append(trial_data[np.newaxis, :])  # (1, 800, 22)
                    all_labels.append(y[t_id])
                    all_subjects.append(subj_idx + 1)
                    all_runs.append(run_id - 2)  # 1-6

    # Stack: (N, 800, 22) → (N, 22, 800)
    data_arr = np.concatenate(all_data, axis=0).transpose(0, 2, 1)
    labels_arr = np.array(all_labels, dtype=np.float64).reshape(-1, 1)
    subjects_arr = np.array(all_subjects, dtype=np.float64).reshape(-1, 1)
    runs_arr = np.array(all_runs, dtype=np.float64).reshape(-1, 1)

    sio.savemat(save_path, {
        'all_sub_data3': data_arr,
        'all_sub_label3': labels_arr,
        'all_sub_index3': subjects_arr,
        'all_sub_run3': runs_arr,
    })

    print(f'[OK] Assembled {save_path}')
    print(f'     Data: {data_arr.shape}, Labels: {labels_arr.shape}, '
          f'Subjects: {np.unique(subjects_arr)}')
    return os.path.abspath(save_path)


def _probe_any_data(data_dir: str) -> Optional[str]:
    """探测任何可用的数据源，返回可用的 data_dir"""
    # 1) 目标目录下的合并文件
    p = os.path.join(data_dir, DATA_COMBINED_FILE)
    if os.path.exists(p) and os.path.getsize(p) > 1024:
        return data_dir

    # 2) 目标目录下的个体文件
    if all(os.path.exists(os.path.join(data_dir, f'{s}{e}.mat'))
           for s in SUBJECTS for e in SESSIONS):
        return data_dir

    # 3) 父目录 (项目原有 data/) 下的合并文件
    parent = os.path.join(os.path.dirname(data_dir), 'data', DATA_COMBINED_FILE)
    if os.path.exists(parent) and os.path.getsize(parent) > 1024:
        print(f'[DATA] Found existing dataset at: {parent}')
        # 软链接/复制到目标目录
        os.makedirs(data_dir, exist_ok=True)
        import shutil
        shutil.copy2(parent, os.path.join(data_dir, DATA_COMBINED_FILE))
        return data_dir

    return None


def ensure_dataset(data_dir: Optional[str] = None,
                   download_if_missing: bool = True,
                   assemble: bool = True) -> str:
    """
    确保数据集可用。返回数据目录的绝对路径。

    策略：
    1. 检查父目录 data/ 下已有的 BCICIV2a.mat（项目原有数据）
    2. 检查 data_dir 下已有的文件
    3. 如果 download_if_missing=True，自动从 BNCI 下载
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = os.path.abspath(data_dir)

    # 优先查找已有数据
    existing = _probe_any_data(data_dir)
    if existing is not None:
        print(f'[DATA] Using dataset: {os.path.join(existing, DATA_COMBINED_FILE)}')
        return existing

    # 没有可用数据 → 下载
    if not download_if_missing:
        raise FileNotFoundError(
            f'Dataset not found in {data_dir} or ../data/. '
            'Set download_if_missing=True or download manually from '
            f'{BNCI_BASE_URL}')

    os.makedirs(data_dir, exist_ok=True)
    combined_path = os.path.join(data_dir, DATA_COMBINED_FILE)

    print('[INFO] No local dataset found. Starting download from BNCI...')
    print(f'  URL: {BNCI_BASE_URL}')
    print(f'  This will download ~700 MB of data.\n')

    success = download_individual_files(data_dir)
    if not success:
        print(f'\n[WARN] Some downloads failed. Files may need manual download:\n'
              f'  {BNCI_BASE_URL}')
        # 检查是否有部分文件可用
        individual_exists = any(
            os.path.exists(os.path.join(data_dir, f'{s}{e}.mat'))
            for s in SUBJECTS for e in SESSIONS
        )
        if not individual_exists:
            raise RuntimeError(
                'Download failed completely. Please check your network or '
                'manually download the dataset.')
    else:
        print('[OK] All files downloaded.')

    if assemble and success:
        assemble_combined_mat(data_dir, combined_path)

    return data_dir


if __name__ == '__main__':
    data_dir = ensure_dataset(download_if_missing=True, assemble=True)
    print(f'\nData directory: {data_dir}')
