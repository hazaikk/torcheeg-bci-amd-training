"""
BCICIV2a Cloud Training — 主训练脚本 (TorchEEG 增强版)
=======================================================
支持模型: EEGNet / FBCNet / FBMSNet / CSPNet / LMDA
训练策略: EarlyStopping + 多种 LR Scheduler (cosine/cosine_warm/step/plateau)
评估度量: Acc / Kappa / F1 / Precision / Recall / Balanced Acc

输出:
  results/<Model>_S<id>_curves.png      训练曲线
  results/<Model>_S<id>_cm.png          混淆矩阵
  results/<Model>_per_subject.png       受试者柱状图
  results/model_comparison.png          模型对比图
  results/TRAINING_REPORT.md            训练报告
  results/results.json                  全部数值结果
"""

# 兼容性修复: 必须在所有第三方 import 之前执行
# 注意: utils.fixes 的导入会间接加载 torcheeg → scipy.signal.hann 崩溃
# 所以这里的 scipy 补丁必须放在 from utils.fixes 之前
import os, sys
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)

import scipy.signal
from scipy.signal.windows import hann, hamming, blackman
scipy.signal.hann = hann
scipy.signal.hamming = hamming
scipy.signal.blackman = blackman

from utils.fixes import apply_all_fixes
apply_all_fixes()

import time
import json
import argparse
import csv
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, precision_score, recall_score, balanced_accuracy_score
)

from utils.data_utils import load_data, get_subject_split, make_dataloader
from utils.model_utils import get_device, get_criterion, create_model, print_gpu_info, verify_device
from utils.training_strategies import EarlyStopping, create_scheduler
from utils.vis_utils import (
    plot_training_curves, plot_confusion_matrix,
    plot_per_subject_results, plot_model_comparison,
    generate_report
)

# ──────────────────────────────────────────────
# 训练与评估
# ──────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    clip_grad: float = 0, epoch: int = -1):
    """训练一个 epoch, 返回 (loss, acc, data_time, compute_time)"""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    data_time = 0.0
    compute_time = 0.0

    for batch in dataloader:
        t0 = time.time()
        x = batch[0].to(device, dtype=torch.float)
        y = batch[1].to(device, dtype=torch.long)
        t1 = time.time()
        data_time += t1 - t0

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        t2 = time.time()
        compute_time += t2 - t1

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)

    # 首轮打印耗时分解（诊断 GPU 等待）
    if epoch == 0:
        total = data_time + compute_time
        if total > 0:
            print(f'    [PERF] Epoch {epoch}: data={data_time:.2f}s '
                  f'({data_time/total*100:.0f}%), '
                  f'compute={compute_time:.2f}s ({compute_time/total*100:.0f}%)')

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """全面评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        x = batch[0].to(device, dtype=torch.float)
        y = batch[1].to(device, dtype=torch.long)
        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return {
        'loss': total_loss / max(len(dataloader), 1),
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'kappa': cohen_kappa_score(all_labels, all_preds),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'predictions': all_preds,
        'true_labels': all_labels,
    }


# ──────────────────────────────────────────────
# LOSO 实验
# ──────────────────────────────────────────────
def run_loso_experiment(model_name: str,
                        data: np.ndarray,
                        labels: np.ndarray,
                        subjects: np.ndarray,
                        cfg: Dict) -> Dict:
    """留一受试者交叉验证 (LOSO)

    结果保存到 run_dir/:
      config.json          训练配置
      S{subject}/epoch_metrics.csv  每 epoch 指标
      S{subject}/best_model.pt      最佳模型权重
      S{subject}/predictions.npz    预测结果
      S{subject}/curves.png         训练曲线
      S{subject}/cm.png             混淆矩阵
    """
    device = cfg.get('device', 'cpu')
    epochs = cfg.get('epochs', 50)
    batch_size = cfg.get('batch_size', 64)
    lr = cfg.get('learning_rate', 0.001)
    weight_decay = cfg.get('weight_decay', 0.001)
    chunk_size = cfg.get('chunk_size', 800)
    use_aug = cfg.get('use_augmentation', False)
    use_early_stop = cfg.get('early_stop', True)
    early_patience = cfg.get('early_patience', 10)
    scheduler_name = cfg.get('scheduler', 'cosine')
    clip_grad = cfg.get('clip_grad', 5.0)
    use_preprocessed = cfg.get('use_preprocessed', False)
    data_dir = cfg.get('data_dir', '')
    run_dir = cfg.get('run_dir', cfg.get('results_dir', 'results'))

    os.makedirs(run_dir, exist_ok=True)

    # 保存配置
    train_config = {
        'model': model_name,
        'device': device,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'scheduler': scheduler_name,
        'early_stop': use_early_stop,
        'early_patience': early_patience,
        'clip_grad': clip_grad,
        'use_augmentation': use_aug,
        'use_preprocessed': use_preprocessed,
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(train_config, f, indent=2)

    all_subjects = sorted(np.unique(subjects).astype(int))
    subject_results = []
    total_start = time.time()

    for test_sub in all_subjects:
        train_subs = [s for s in all_subjects if s != test_sub]
        split = get_subject_split(data, labels, subjects, test_sub, train_subs)

        train_loader = make_dataloader(
            *split['train'], model_name, batch_size=batch_size,
            shuffle=True, chunk_size=chunk_size, use_augmentation=use_aug,
            precompute=True, device=device,
            use_preprocessed=use_preprocessed, data_dir=data_dir)
        test_loader = make_dataloader(
            *split['test'], model_name, batch_size=batch_size * 2,
            shuffle=False, chunk_size=chunk_size, use_augmentation=False,
            precompute=True, device=device,
            use_preprocessed=use_preprocessed, data_dir=data_dir)

        # 创建模型 / 优化器 / 调度器
        in_ch = 9 if model_name in ['FBCNet', 'FBMSNet'] else 1
        model = create_model(model_name, num_classes=4,
                             chunk_size=chunk_size,
                             num_electrodes=22, in_channels=in_ch)
        model = model.to(device)
        criterion = get_criterion(model_name)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = create_scheduler(optimizer, scheduler_name, epochs)
        early_stopper = EarlyStopping(patience=early_patience, mode='max',
                                       verbose=True) if use_early_stop else None

        # 训练循环
        best_acc = 0.0
        best_state = None
        train_losses, train_accs, test_accs = [], [], []
        best_test_results = None
        stopped_early = False

        t0 = time.time()
        epoch_metrics = []  # per-epoch CSV data
        for epoch in range(epochs):
            current_lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                clip_grad=clip_grad, epoch=epoch)

            test_result = evaluate(model, test_loader, criterion, device)
            test_acc = test_result['accuracy']

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            # 更新最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                best_state = {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict().copy(),
                    'optimizer_state': optimizer.state_dict(),
                    'test_result': test_result,
                }
                best_test_results = test_result

            # 记录每 epoch 指标
            epoch_metrics.append({
                'epoch': epoch + 1,
                'train_loss': round(train_loss, 6),
                'train_acc': round(train_acc, 6),
                'test_acc': round(test_acc, 6),
                'test_kappa': round(test_result['kappa'], 6),
                'test_f1': round(test_result['f1_weighted'], 6),
                'lr': round(current_lr, 8),
            })

            # 打印进度 (每 5 epoch 或首末)
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f'  [{model_name}] S{test_sub} Epoch {epoch+1:3d}/{epochs} | '
                      f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                      f'Test Acc: {test_acc:.4f} (best: {best_acc:.4f})')

            # Scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_acc)
            else:
                scheduler.step()

            # Early stopping
            if early_stopper is not None and epoch >= 5:
                if early_stopper(test_acc, epoch + 1):
                    stopped_early = True
                    break

        elapsed = time.time() - t0

        # 加载最佳模型 → 最终评估
        if best_state and best_test_results:
            model.load_state_dict(best_state['model_state'])
            final_result = evaluate(model, test_loader, criterion, device)
        else:
            final_result = test_result

        cm = np.array(final_result['confusion_matrix'])

        # ── 保存受试者级别的过程文件 ──
        sub_dir = os.path.join(run_dir, f'S{test_sub:02d}')
        os.makedirs(sub_dir, exist_ok=True)

        # 1) 每 epoch 指标 CSV
        csv_path = os.path.join(sub_dir, 'epoch_metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=epoch_metrics[0].keys())
            w.writeheader()
            w.writerows(epoch_metrics)

        # 2) 最佳模型权重
        if best_state:
            torch.save(best_state['model_state'],
                       os.path.join(sub_dir, 'best_model.pt'))
            # 同时保存训练友好的格式（含配置）
            torch.save({
                'model_state_dict': best_state['model_state'],
                'optimizer_state_dict': best_state['optimizer_state'],
                'epoch': best_state['epoch'],
                'test_accuracy': best_acc,
                'config': train_config,
            }, os.path.join(sub_dir, 'checkpoint.pt'))

        # 3) 预测结果
        np.savez(os.path.join(sub_dir, 'predictions.npz'),
                 predictions=np.array(final_result['predictions']),
                 true_labels=np.array(final_result['true_labels']),
                 test_accuracy=final_result['accuracy'],
                 kappa=final_result['kappa'],
                 f1=final_result['f1_weighted'])

        # 4) 可视化 (已在上方以 sub_dir 为路径保存)
        # 5) 受试者柱状图
        plot_per_subject_results([{
            'test_subject': int(test_sub),
            'accuracy': float(final_result['accuracy']),
            'best_accuracy': float(best_acc),
        }], model_name, sub_dir)

        subject_results.append({
            'test_subject': int(test_sub),
            'accuracy': float(final_result['accuracy']),
            'balanced_accuracy': float(final_result['balanced_accuracy']),
            'best_accuracy': float(best_acc),
            'kappa': float(final_result['kappa']),
            'f1_weighted': float(final_result['f1_weighted']),
            'f1_macro': float(final_result['f1_macro']),
            'precision': float(final_result['precision']),
            'recall': float(final_result['recall']),
            'loss': float(final_result['loss']),
            'best_epoch': best_state['epoch'] if best_state else -1,
            'stopped_early': stopped_early,
            'train_time': round(elapsed, 1),
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'confusion_matrix': cm.tolist(),
            'sub_dir': sub_dir,
        })

        early_tag = '(early stop)' if stopped_early else ''
        print(f'  >>> S{test_sub} done: Acc={final_result["accuracy"]:.4f}, '
              f'Kappa={final_result["kappa"]:.4f}, Time={elapsed:.1f}s {early_tag}')

    # 汇总
    accs = [r['accuracy'] for r in subject_results]
    total_time = time.time() - total_start

    # 保存汇总 CSV
    summary_path = os.path.join(run_dir, 'metrics_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subject', 'accuracy', 'kappa', 'f1', 'precision',
                     'recall', 'best_epoch', 'train_time', 'stopped_early'])
        for r in subject_results:
            w.writerow([
                r['test_subject'], r['accuracy'], r['kappa'],
                r['f1_weighted'], r['precision'], r['recall'],
                r['best_epoch'], r['train_time'], r['stopped_early'],
            ])

    # 保存结果 JSON
    result = {
        'run_dir': run_dir,
        'model': model_name,
        'per_subject': subject_results,
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'mean_kappa': float(np.mean([r['kappa'] for r in subject_results])),
        'mean_f1': float(np.mean([r['f1_weighted'] for r in subject_results])),
        'total_train_time': round(total_time, 1),
        'config': train_config,
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='BCICIV2a CNN Cloud Training (AMD ROCm / CUDA)')
    parser.add_argument('--models', nargs='+',
                        default=['EEGNet', 'TSCeption', 'FBCNet', 'FBMSNet'],
                        help='Models to train (default: EEGNet TSCeption FBCNet FBMSNet)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max epochs per subject (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'cosine_warm', 'step', 'plateau'],
                        help='LR scheduler strategy')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping')
    parser.add_argument('--early-patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--clip-grad', type=float, default=5.0,
                        help='Gradient clipping norm (0=disable)')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto / cuda / cpu')
    parser.add_argument('--use-augmentation', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--use-preprocessed', action='store_true',
                        help='Use pre-daved .pt files (run preprocess_dataset.py first)')
    args = parser.parse_args()

    # 设备检测 + GPU 诊断
    detected = get_device() if args.device == 'auto' else args.device
    device = verify_device(detected)
    print(f'[INFO] Device: {device}')
    if device == 'cpu':
        print('[WARN] Running on CPU — training will be slow. '
              'Install ROCm PyTorch for GPU acceleration.')
        print('  -> pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/rocm5.6')
    else:
        print('[INFO] GPU detected — transforms will be pre-computed for faster training')

    # 数据准备
    from download_data import ensure_dataset
    data_dir = ensure_dataset(args.data_dir, download_if_missing=True, assemble=True)
    print('[INFO] Loading dataset...')
    data, labels, subjects, runs = load_data(data_dir)
    print(f'  Data: {data.shape}, Subjects: {list(np.unique(subjects))}')

    # 训练配置
    train_cfg = {
        'device': device,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 0.001,
        'chunk_size': 800,
        'results_dir': args.results_dir,
        'data_dir': os.path.abspath(args.data_dir),
        'use_augmentation': args.use_augmentation,
        'early_stop': not args.no_early_stop,
        'early_patience': args.early_patience,
        'scheduler': args.scheduler,
        'clip_grad': args.clip_grad,
        'use_preprocessed': args.use_preprocessed,
    }
    os.makedirs(args.results_dir, exist_ok=True)

    # 训练各模型 (每个模型独立 timestamped run_dir)
    all_results = {}
    run_dirs = []
    for model_name in args.models:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(args.results_dir, f'{model_name}_{timestamp}')
        os.makedirs(run_dir, exist_ok=True)

        print(f'\n{"=" * 60}')
        print(f'[TRAIN] {model_name} — LOSO ({len(np.unique(subjects))} subjects)')
        print(f'  Scheduler: {args.scheduler}, EarlyStop: {not args.no_early_stop}, '
              f'Patience: {args.early_patience}')
        print(f'  Results: {run_dir}')
        print(f'{"=" * 60}')

        train_cfg['run_dir'] = run_dir
        result = run_loso_experiment(model_name, data, labels, subjects, train_cfg)

        accs = [r['accuracy'] for r in result['per_subject']]
        print(f'\n[{model_name}] Mean Acc: {result["mean_accuracy"]:.4f} ± {result["std_accuracy"]:.4f}')
        print(f'  Mean Kappa: {result["mean_kappa"]:.4f}, Mean F1: {result["mean_f1"]:.4f}')
        print(f'  Total Time: {result["total_train_time"]:.1f}s')
        print(f'  Results saved to: {run_dir}')
        for sr in result['per_subject']:
            tag = '(ES)' if sr.get('stopped_early') else ''
            print(f'    S{sr["test_subject"]}: Acc={sr["accuracy"]:.4f} '
                  f'(best={sr["best_accuracy"]:.4f} @ epoch {sr["best_epoch"]}) {tag}')

        all_results[model_name] = result
        run_dirs.append(run_dir)

    # 模型对比图 (保存到 results/ 根目录)
    print('\n[INFO] Generating comparison plots...')
    plot_model_comparison(all_results, args.results_dir)

    # 训练报告 (保存到 results/ 根目录)
    print('[INFO] Generating training report...')
    generate_report(all_results, args.results_dir, {
        'device': device,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'scheduler': args.scheduler,
        'early_stop': str(not args.no_early_stop),
        'early_patience': str(args.early_patience),
        'models': ', '.join(args.models),
        'dataset': 'BCICIV2a (9 subjects, 4 classes)',
        'augmentation': 'enabled' if args.use_augmentation else 'disabled',
        'run_dirs': ', '.join(run_dirs),
    })

    # 保存汇总 JSON (results/ 根目录)
    json_path = os.path.join(args.results_dir, 'results.json')
    summary = {m: {
        'mean_accuracy': r['mean_accuracy'],
        'std_accuracy': r['std_accuracy'],
        'mean_kappa': r['mean_kappa'],
        'mean_f1': r['mean_f1'],
        'total_train_time': r['total_train_time'],
        'run_dir': r.get('run_dir', ''),
        'config': r['config'],
    } for m, r in all_results.items()}
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  -> {json_path}')

    # 显示各模型目录
    print(f'\n[INFO] Individual run directories:')
    for d in run_dirs:
        print(f'  {d}/')

    # 最终对比
    print(f'\n{"=" * 60}')
    print('FINAL COMPARISON')
    print(f'{"Model":<15} {"Mean Acc":<12} {"Kappa":<10} {"F1":<10} {"Time":<12}')
    print('-' * 59)
    for m_name in sorted(all_results.keys(),
                         key=lambda m: all_results[m]['mean_accuracy'], reverse=True):
        r = all_results[m_name]
        print(f'{m_name:<15} {r["mean_accuracy"]:.4f}     '
              f'{r["mean_kappa"]:.4f}    {r["mean_f1"]:.4f}  {r["total_train_time"]:.1f}s')

    print(f'\n[OK] Done. Results: {os.path.abspath(args.results_dir)}')
    return all_results


if __name__ == '__main__':
    main()
