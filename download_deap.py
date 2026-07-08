"""
DEAP 数据集下载模块
===================
使用 kagglehub 从 Kaggle 下载 DEAP 数据集，
或使用提供的 data_preprocessed_python 目录。

用法:
    python download_deap.py                          # 下载 DEAP 数据集
    python download_deap.py --data-dir /path/to/dir  # 指定自定义路径
"""

import os
import sys
import shutil
from typing import Optional

# =============================================
# 路径配置
# =============================================
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_KAGGLE_CACHE = os.path.join(_PROJECT_DIR, 'kagglehub_cache')
_DEFAULT_DEAP_DIR = os.path.join(_PROJECT_DIR, 'data', 'deap')

# Kaggle dataset identifier
KAGGLE_DATASET = "manh123df/deap-dataset"
DATASET_SUBDIR = "data_preprocessed_python"


def download_via_kagglehub(kaggle_cache_dir: Optional[str] = None) -> str:
    """使用 kagglehub 下载 DEAP 数据集

    Args:
        kaggle_cache_dir: kagglehub 缓存目录 (默认 project_dir/kagglehub_cache)

    Returns:
        str: 下载后数据所在目录的路径 (data_preprocessed_python)
    """
    if kaggle_cache_dir is None:
        kaggle_cache_dir = _DEFAULT_KAGGLE_CACHE

    os.makedirs(kaggle_cache_dir, exist_ok=True)

    # 设置 kagglehub 缓存
    os.environ["KAGGLEHUB_CACHE"] = kaggle_cache_dir

    try:
        import kagglehub
    except ImportError:
        print('[DEAP] kagglehub not installed. Installing...')
        import subprocess
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'kagglehub'])
        import kagglehub

    print(f'[DEAP] Downloading {KAGGLE_DATASET} via kagglehub...')
    print(f'       Cache dir: {kaggle_cache_dir}')

    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)

    print(f'[DEAP] Downloaded to: {dataset_path}')

    # 确认数据文件存在
    data_dir = os.path.join(dataset_path, DATASET_SUBDIR)
    if not os.path.exists(data_dir):
        # 直接检查下载目录
        contents = os.listdir(dataset_path)
        print(f'[DEAP] Contents of {dataset_path}: {contents}')
        for d in contents:
            sub = os.path.join(dataset_path, d)
            if os.path.isdir(sub):
                sub_contents = os.listdir(sub)
                print(f'  {d}/: {sub_contents[:5]}...')
        raise FileNotFoundError(
            f'Expected {DATASET_SUBDIR} not found in {dataset_path}. '
            f'Please check the dataset structure.')

    # 验证至少包含一个 .dat 文件
    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    if not dat_files:
        raise FileNotFoundError(
            f'No .dat files found in {data_dir}. '
            f'Contents: {os.listdir(data_dir)}')

    print(f'[DEAP] Found {len(dat_files)} subject files in {data_dir}')
    return os.path.abspath(data_dir)


def copy_to_project(data_dir: str,
                    target_dir: Optional[str] = None) -> str:
    """将 DEAP 数据复制到项目目录

    Args:
        data_dir: 原始数据路径 (kagglehub 缓存)
        target_dir: 目标目录 (默认 project/data/deap)

    Returns:
        str: 目标目录绝对路径
    """
    if target_dir is None:
        target_dir = _DEFAULT_DEAP_DIR

    os.makedirs(target_dir, exist_ok=True)

    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    for f in dat_files:
        src = os.path.join(data_dir, f)
        dst = os.path.join(target_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f'  [COPY] {f}')

    print(f'[DEAP] Copied {len(dat_files)} files to {target_dir}')
    return os.path.abspath(target_dir)


def ensure_deap_dataset(data_dir: Optional[str] = None,
                         use_cache: bool = True,
                         force_redownload: bool = False) -> str:
    """确保 DEAP 数据集可用

    策略:
    1. 检查 data_dir 下是否有 .dat 文件
    2. 从 kagglehub 下载并复制

    Args:
        data_dir: 目标数据目录 (默认 project/data/deap)
        use_cache: 是否使用已有下载缓存
        force_redownload: 强制重新下载

    Returns:
        str: 包含 .dat 文件的目录绝对路径
    """
    if data_dir is None:
        data_dir = _DEFAULT_DEAP_DIR

    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # 检查已有数据
    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    if dat_files and not force_redownload:
        print(f'[DEAP] Found existing dataset at: {data_dir}')
        print(f'       {len(dat_files)} subject files')
        return data_dir

    # 从 kagglehub 下载
    kaggle_cache = os.path.join(_PROJECT_DIR, 'kagglehub_cache')
    source_dir = download_via_kagglehub(kaggle_cache)
    data_dir = copy_to_project(source_dir, data_dir)

    return data_dir


def get_deap_path(data_dir: Optional[str] = None) -> str:
    """返回 DEAP 数据目录

    供 train_deap.py 调用 — 只返回路径，不下载。
    如果数据不存在，调用 ensure_deap_dataset() 触发下载。
    """
    if data_dir is None:
        data_dir = _DEFAULT_DEAP_DIR

    data_dir = os.path.abspath(data_dir)

    # 检查数据是否存在
    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    if not dat_files:
        # 自动下载
        return ensure_deap_dataset(data_dir)

    return data_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download DEAP dataset')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Target data directory')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download')
    args = parser.parse_args()

    path = ensure_deap_dataset(
        data_dir=args.data_dir,
        force_redownload=args.force
    )
    print(f'\n[DEAP] DEAP dataset ready at: {path}')
