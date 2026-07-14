"""
DEAP 多类型模型训练脚本

支持模型: Conformer, VanillaTransformer, LSTM, GRU, DGCNN, LGGNet, STNet, LMDA, CSPNet

用法:
    python train.py --models Conformer LMDA
    python train.py --models all --gpu
    python train.py --models LSTM --lr 0.0005 --batch-size 64
    python train.py --models all --chunk-size 256
"""

import os
import sys
import json
import time
import copy
import argparse
import warnings
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

from config import *
from utils import EarlyStopping, create_scheduler, set_seed


# =====================================
# Dataset
# =====================================

class PreprocessedDataset(Dataset):
    """加载预计算 .pt 文件的 Dataset, 索引式避免显存翻倍"""

    def __init__(self, data, labels, subjects,
                 indices: Optional[torch.Tensor] = None):
        self._data = data
        self._labels = labels.flatten()
        self._subjects = subjects.flatten()
        if indices is not None:
            self.indices = indices
        else:
            self.indices = torch.arange(len(self._data))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self._data[real_idx], int(self._labels[real_idx]), int(self._subjects[real_idx])


# =====================================
# 模型工厂
# =====================================

def create_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """创建模型实例"""
    params = MODEL_PARAMS[model_name].copy()
    params['num_classes'] = num_classes

    if model_name == 'Conformer':
        from torcheeg.models import Conformer
        return Conformer(**params)

    elif model_name == 'VanillaTransformer':
        from torcheeg.models import VanillaTransformer
        return VanillaTransformer(**params)

    elif model_name == 'LSTM':
        from torcheeg.models import LSTM
        return LSTM(**params)

    elif model_name == 'GRU':
        from torcheeg.models import GRU
        return GRU(**params)

    elif model_name == 'DGCNN':
        from torcheeg.models import DGCNN
        return DGCNN(**params)

    elif model_name == 'LGGNet':
        from torcheeg.models import LGGNet
        return LGGNet(**params)

    elif model_name == 'STNet':
        from torcheeg.models import STNet
        param_copy = params.copy()
        grid_size = param_copy.pop('grid_size')
        return STNet(**param_copy, grid_size=grid_size)

    elif model_name == 'LMDA':
        from torcheeg.models import LMDA
        return LMDA(**params)

    elif model_name == 'CSPNet':
        from torcheeg.models import CSPNet
        return CSPNet(**params)

    else:
        raise ValueError(f'Unknown model: {model_name}')


def count_params(model: nn.Module) -> int:
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================
# 训练 & 评估
# =====================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            inputs, labels = batch[0], batch[1]
        else:
            raise TypeError(f'Unexpected batch type: {type(batch)}')

        inputs = inputs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
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

    return running_loss / len(dataloader), accuracy


# =====================================
# 交叉验证
# =====================================

def run_experiment(
    model_name: str,
    preproc_dir: str,
    results_dir: str,
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
    test_mode: bool = False,
    verbose: bool = True,
) -> Dict:
    """运行单个模型的完整实验"""

    # ── 加载预计算数据 ──
    data_path = os.path.join(preproc_dir, f'{model_name}_data.pt')
    meta_path = os.path.join(preproc_dir, 'meta.pt')

    if not os.path.exists(data_path):
        return {'model': model_name, 'error': f'no_data: {data_path}'}

    data = torch.load(data_path, map_location='cpu', weights_only=True)
    meta = torch.load(meta_path, map_location='cpu', weights_only=True)
    labels = meta['labels'].flatten()
    subjects = meta['subjects'].flatten()
    n_total = len(data)

    cls_counts = Counter(labels.tolist())
    if verbose:
        print(f'\n[LOAD] {model_name}')
        print(f'       Data shape: {tuple(data.shape)}')
        print(f'       Classes: {dict(cls_counts)}')

    # 数据移到 GPU
    if device != 'cpu':
        data = data.to(device)

    # ── 按 subject 划分 CV ──
    unique_subjects = sorted(subjects.unique().tolist())
    np.random.seed(42)
    np.random.shuffle(unique_subjects)
    fold_subject_groups = np.array_split(
        unique_subjects, min(n_splits, len(unique_subjects)))
    fold_subject_groups = [g.tolist() if hasattr(g, 'tolist') else list(g)
                           for g in fold_subject_groups]

    fold_indices = []
    for val_subjects in fold_subject_groups:
        train_mask = ~torch.isin(subjects, torch.tensor(val_subjects))
        val_mask = torch.isin(subjects, torch.tensor(val_subjects))
        fold_indices.append((
            torch.where(train_mask)[0],
            torch.where(val_mask)[0],
        ))

    n_folds = len(fold_indices)
    actual_epochs = 1 if test_mode else epochs

    # ── 逐折训练 ──
    fold_results = []
    summary_file = os.path.join(results_dir, model_name, 'summary.json')
    csv_path = os.path.join(results_dir, model_name, 'all_epochs.csv')
    best_model_path = os.path.join(results_dir, model_name, 'best_model.pt')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    all_epochs_records = []
    best_overall_acc = 0.0
    best_model_state = None

    for fold_idx in range(n_folds):
        train_idx, val_idx = fold_indices[fold_idx]
        train_ds = PreprocessedDataset(data, labels, subjects, indices=train_idx)
        val_ds = PreprocessedDataset(data, labels, subjects, indices=val_idx)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, num_workers=0)

        # 创建模型
        model = create_model(model_name, num_classes=DEAP_NUM_CLASSES)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, scheduler_name, actual_epochs)
        early_stopping = EarlyStopping(
            patience=early_patience, mode='max', verbose=verbose)

        fold_best_acc = 0.0

        if verbose:
            print(f'\n{"="*50}')
            print(f'  Fold {fold_idx+1}/{n_folds}')
            print(f'  Train: {len(train_ds)} | Val: {len(val_ds)}')
            print(f'{"="*50}')

        for epoch in range(actual_epochs):
            t_epoch = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

            if val_acc > fold_best_acc:
                fold_best_acc = val_acc

            if val_acc > best_overall_acc:
                best_overall_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())

            record = {
                'epoch': epoch + 1, 'fold': fold_idx + 1,
                'train_loss': round(train_loss, 4),
                'train_acc': round(train_acc, 2),
                'val_loss': round(val_loss, 4),
                'val_acc': round(val_acc, 2),
                'time': round(time.time() - t_epoch, 2),
            }
            all_epochs_records.append(record)

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == actual_epochs - 1:
                print(f'  Ep {epoch+1:3d}/{actual_epochs} | '
                      f'T_loss:{train_loss:.4f} T_acc:{train_acc:.2f}% | '
                      f'V_loss:{val_loss:.4f} V_acc:{val_acc:.2f}% | '
                      f'{record["time"]:.1f}s')

            if early_stopping(val_acc, epoch):
                break

        fold_results.append(fold_best_acc)
        print(f'  >>> Fold {fold_idx+1} best: {fold_best_acc:.2f}%')

        # 每折保存详细指标
        fold_csv = os.path.join(results_dir, model_name, f'fold_{fold_idx+1}_metrics.csv')
        fold_df = pd.DataFrame([r for r in all_epochs_records if r['fold'] == fold_idx + 1])
        fold_df.to_csv(fold_csv, index=False)

        # 清理 GPU
        if device != 'cpu':
            del model
            torch.cuda.empty_cache()

    # ── 汇总 ──
    mean_acc = float(np.mean(fold_results))
    std_acc = float(np.std(fold_results))
    summary = {
        'model': model_name,
        'chunk_size': chunk_size,
        'cv_strategy': cv_strategy,
        'n_splits': n_splits,
        'batch_size': batch_size,
        'lr': lr,
        'epochs_actual': actual_epochs,
        'early_patience': early_patience,
        'scheduler': scheduler_name,
        'category': MODEL_CATEGORIES.get(model_name, ''),
        'num_params': count_params(create_model(model_name, DEAP_NUM_CLASSES)),
        'fold_results': [{'fold': i+1, 'best_val_acc': acc,
                           'epochs_trained': len([r for r in all_epochs_records
                                                   if r['fold'] == i+1])}
                          for i, acc in enumerate(fold_results)],
        'mean_val_acc': round(mean_acc, 2),
        'std_val_acc': round(std_acc, 2),
        'best_val_acc': round(best_overall_acc, 2),
        'total_epochs': len(all_epochs_records),
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[SAVE] Summary: {summary_file}')

    # 保存 all_epochs.csv
    pd.DataFrame(all_epochs_records).to_csv(csv_path, index=False)
    print(f'[SAVE] Epochs:  {csv_path}')

    # 保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, best_model_path)
        print(f'[SAVE] Model:   {best_model_path}')

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='DEAP 多类型模型训练')
    parser.add_argument('--models', type=str, nargs='+',
                        default=AVAILABLE_MODELS,
                        choices=AVAILABLE_MODELS + ['all'],
                        help='模型列表')
    parser.add_argument('--preproc-dir', type=str, default='',
                        help='预处理数据目录')
    parser.add_argument('--results-dir', type=str, default='',
                        help='结果输出目录')
    parser.add_argument('--chunk-size', type=int, default=128,
                        choices=[128, 256])
    parser.add_argument('--cv', type=str, default='kfold_groupby_trial',
                        choices=['kfold_groupby_trial', 'kfold',
                                 'leave_one_subject_out'])
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early-patience', type=int, default=15)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'step'])
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--test', action='store_true',
                        help='测试模式 (1 epoch, 1 fold)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    models = args.models
    if 'all' in models:
        models = AVAILABLE_MODELS

    # 结果目录
    preproc_dir = args.preproc_dir or DEFAULT_PREPROC_DIR
    results_base = args.results_dir or DEFAULT_RESULTS_DIR
    window_label = f'{args.chunk_size}pt_{args.chunk_size//128}s'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'MM_{window_label}_{args.cv}_{timestamp}'
    results_dir = os.path.join(results_base, run_name)
    os.makedirs(results_dir, exist_ok=True)

    # 保存配置
    config = vars(args)
    config['device'] = device
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f'[DEVICE] {device}')
    if device == 'cuda':
        print(f'         {torch.cuda.get_device_name(0)}')
    print(f'[PREPROC] {preproc_dir}')
    print(f'[RESULTS] {results_dir}')
    print(f'[MODELS]  {models}')

    # ── 逐个模型运行 ──
    all_summaries = []
    for model_name in models:
        print(f'\n{"#"*65}')
        print(f'#  {model_name:30s} | {args.cv} ({args.n_splits}-fold)')
        print(f'#  Window: {args.chunk_size}pt ({args.chunk_size/128:.0f}s)')
        print(f'#  Device: {device}')
        print(f'{"#"*65}')

        summary = run_experiment(
            model_name=model_name,
            preproc_dir=preproc_dir,
            results_dir=results_dir,
            chunk_size=args.chunk_size,
            cv_strategy=args.cv,
            n_splits=args.n_splits,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=1 if args.test else args.epochs,
            scheduler_name=args.scheduler,
            early_patience=args.early_patience,
            device=device,
            test_mode=args.test,
            verbose=True,
        )

        all_summaries.append(summary)

        # 跨 GPU 清理
        if device != 'cpu':
            torch.cuda.empty_cache()

    # ── 汇总表 ──
    print(f'\n{"="*65}')
    print(f'  实验汇总')
    print(f'{"="*65}')

    rows = []
    for s in all_summaries:
        if 'error' in s:
            print(f'  {s["model"]:25s} | ERROR: {s["error"]}')
            continue
        print(f'  {s["model"]:25s} | '
              f'Mean: {s["mean_val_acc"]:.2f}% ± {s["std_val_acc"]:.2f}% | '
              f'Best: {s["best_val_acc"]:.2f}% | '
              f'Params: {s["num_params"]:,}')
        rows.append({
            'Model': s['model'],
            'Category': s.get('category', ''),
            'CV_Mean_Acc': s['mean_val_acc'],
            'CV_Std': s['std_val_acc'],
            'CV_Best_Acc': s['best_val_acc'],
            'Num_Params': s['num_params'],
        })

    if rows:
        summary_csv = os.path.join(results_dir, 'experiment_summary.csv')
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
        print(f'\n[SAVE] Summary: {summary_csv}')

    print(f'\n完成! 结果目录: {results_dir}')


if __name__ == '__main__':
    main()
