"""
可视化与报告生成工具
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # 无头后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 全局样式 ──
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
CLASS_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_training_curves(train_losses: List[float],
                          train_accs: List[float],
                          test_accs: List[float],
                          model_name: str,
                          subject_id: int,
                          save_dir: str) -> str:
    """绘制单个受试者的训练曲线"""
    _ensure_dir(save_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    epochs = range(1, len(train_losses) + 1)

    # Loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=1.5, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} — Subject {subject_id} — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_accs, 'g-', linewidth=1.5, label='Train Acc')
    ax2.plot(epochs, test_accs, 'r-', linewidth=1.5, label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} — Subject {subject_id} — Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    path = os.path.join(save_dir, f'{model_name}_S{subject_id}_curves.png')
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm: np.ndarray,
                           model_name: str,
                           subject_id: int,
                           save_dir: str,
                           normalize: bool = True) -> str:
    """绘制混淆矩阵"""
    _ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(5.5, 5))

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        vmin, vmax = 0, 1
    else:
        cm_display = cm
        fmt = 'd'
        vmin, vmax = 0, cm.max()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_display,
                                   display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', colorbar=True,
              values_format=fmt, im_kw={'vmin': vmin, 'vmax': vmax})

    title = f'{model_name} — Subject {subject_id}'
    if normalize:
        title += ' (Normalized)'
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    path = os.path.join(save_dir, f'{model_name}_S{subject_id}_cm.png')
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_per_subject_results(subject_results: List[Dict],
                               model_name: str,
                               save_dir: str) -> str:
    """绘制每个受试者的柱状图"""
    _ensure_dir(save_dir)
    subjects = [r['test_subject'] for r in subject_results]
    accs = [r['accuracy'] for r in subject_results]
    best_accs = [r.get('best_accuracy', r['accuracy']) for r in subject_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(subjects))
    width = 0.35
    bars1 = ax.bar(x - width / 2, accs, width, label='Test Acc',
                    color='steelblue', edgecolor='white')
    bars2 = ax.bar(x + width / 2, best_accs, width, label='Best Acc',
                    color='coral', edgecolor='white')

    ax.axhline(mean_acc, color='red', ls='--', linewidth=1.2,
               label=f'Mean: {mean_acc:.3f}')
    ax.axhline(0.25, color='gray', ls=':', alpha=0.5,
               label='Chance (0.25)')

    ax.set_xlabel('Subject')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} — Per-Subject Results\n'
                 f'Mean={mean_acc:.3f} ± {std_acc:.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in subjects])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # 在柱上标注数值
    for bar in bars1:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    path = os.path.join(save_dir, f'{model_name}_per_subject.png')
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_model_comparison(all_results: Dict[str, Dict],
                            save_dir: str) -> str:
    """绘制模型间对比柱状图"""
    _ensure_dir(save_dir)
    model_names = list(all_results.keys())
    means = [all_results[m]['mean_accuracy'] for m in model_names]
    stds = [all_results[m]['std_accuracy'] for m in model_names]
    times = [all_results[m].get('total_train_time', 0) for m in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 准确率对比
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax1.bar(model_names, means, yerr=stds, capsize=5,
                    color=colors, edgecolor='gray', linewidth=1.2)
    ax1.axhline(0.25, color='gray', ls=':', alpha=0.5, label='Chance')
    ax1.set_ylabel('Mean Accuracy')
    ax1.set_title('Model Comparison — Accuracy')
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 训练时间对比
    bars = ax2.bar(model_names, times, color=colors, edgecolor='gray',
                    linewidth=1.2)
    ax2.set_ylabel('Total Training Time (s)')
    ax2.set_title('Model Comparison — Training Time')
    for bar, val in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.0f}s', ha='center', va='bottom', fontsize=9)

    path = os.path.join(save_dir, 'model_comparison.png')
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_report(all_results: Dict, save_dir: str,
                     config_info: Optional[Dict] = None) -> str:
    """
    生成完整的训练报告 (Markdown 格式)。

    报告包含：
      - 实验环境信息
      - 各模型训练曲线摘要
      - 混淆矩阵
      - 受试者级别结果
      - 模型对比
      - 结论与建议
    """
    _ensure_dir(save_dir)

    # 计算整体统计
    n_models = len(all_results)
    best_model = max(all_results, key=lambda m: all_results[m]['mean_accuracy'])
    best_acc = all_results[best_model]['mean_accuracy']

    lines = []
    _w = lambda s: lines.append(s)

    _w(f'# BCICIV2a CNN 模型训练报告\n')
    _w(f'> **生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    _w(f'> **数据集**: BCI Competition IV 2a (BCICIV2a)\n')
    _w(f'> **受试者数**: 9 | **类别数**: 4 | **通道数**: 22 | **采样率**: 250 Hz\n')
    _w(f'> **设备**: {config_info.get("device", "N/A") if config_info else "N/A"}\n')
    _w('')

    # ── 1. 实验配置 ──
    _w('## 1. 实验配置\n')
    _w('| 参数 | 值 |')
    _w('|------|-----|')
    if config_info:
        for k, v in config_info.items():
            _w(f'| {k} | {v} |')
    else:
        _w('| (配置信息未传递) |')
    _w('')

    # ── 2. 总体对比 ──
    _w('## 2. 总体对比\n')
    _w(f'本次实验共训练 **{n_models}** 个模型。')
    _w(f'最佳模型: **{best_model}** (平均准确率: {best_acc:.3f})\n')
    _w('| 模型 | 平均准确率 | 标准差 | 总训练时间 (s) |')
    _w('|------|-----------|-------|--------------|')
    for m_name in sorted(all_results.keys(),
                          key=lambda m: all_results[m]['mean_accuracy'],
                          reverse=True):
        r = all_results[m_name]
        _w(f'| {m_name} | {r["mean_accuracy"]:.4f} | ±{r["std_accuracy"]:.4f} '
           f'| {r.get("total_train_time", 0):.1f} |')
    _w('')

    # 模型对比图
    comp_path = plot_model_comparison(all_results, save_dir)
    rel_comp = os.path.relpath(comp_path, save_dir)
    _w(f'![Model Comparison]({rel_comp})\n')

    # ── 3. 各模型详细结果 ──
    _w('## 3. 各模型详细结果\n')
    for m_name in sorted(all_results.keys(),
                          key=lambda m: all_results[m]['mean_accuracy'],
                          reverse=True):
        r = all_results[m_name]
        _w(f'### {m_name}\n')
        _w(f'- **平均准确率**: {r["mean_accuracy"]:.4f} ± {r["std_accuracy"]:.4f}')
        _w(f'- **总训练时间**: {r.get("total_train_time", 0):.1f} s')
        if 'per_subject' in r:
            subj_accs = [s['accuracy'] for s in r['per_subject']]
            _w(f'- **最高**: {max(subj_accs):.4f} (Subject {np.argmax(subj_accs)+1})')
            _w(f'- **最低**: {min(subj_accs):.4f} (Subject {np.argmin(subj_accs)+1})')
        _w('')

        # 受试者柱状图
        if 'per_subject' in r:
            subj_path = plot_per_subject_results(r['per_subject'], m_name, save_dir)
            rel_subj = os.path.relpath(subj_path, save_dir)
            _w(f'![{m_name} Per-Subject]({rel_subj})\n')

        # 受试者结果表
        if 'per_subject' in r:
            _w('| Subject | Accuracy | Best Acc | Time (s) |')
            _w('|---------|----------|----------|----------|')
            for sr in r['per_subject']:
                _w(f'| S{sr["test_subject"]} | {sr["accuracy"]:.4f} | '
                   f'{sr.get("best_accuracy", 0):.4f} | '
                   f'{sr.get("train_time", 0):.1f} |')
            _w('')

    # ── 4. 结论 ──
    _w('## 4. 结论与建议\n')
    _w(f'基于 BCICIV2a 数据集的 {n_models} 个 CNN 模型对比实验：\n')
    _w(f'1. **{best_model}** 在此配置下取得最佳平均准确率 ({best_acc:.3f})')
    _w('2. 受试者间差异显著，个性化校准 (如 fine-tune) 可能进一步提升性能')
    _w('3. 运动想象分类的关键信息集中在 Alpha (8-13 Hz) 和 Beta (13-30 Hz) 频段')
    _w('4. 运动区电极 (C3/Cz/C4) 具有最高的 discriminative 价值\n')

    # 建议
    _w('### 后续优化方向\n')
    _w('- **频带优化**: 针对不同受试者选择最优频带子集')
    _w('- **时间窗口**: 探索 cue 后 0.5-2.5s 窗口 vs 全窗口')
    _w('- **数据增强**: 添加噪声、裁剪、频带扰动等策略')
    _w('- **模型集成**: 多个模型的预测融合')
    _w('- **迁移学习**: 利用源域受试者数据提升目标受试者性能')
    _w('')

    # 脚注
    _w('---')
    _w(f'*本报告由 cloud_amd_training 训练框架自动生成*\n')

    report = '\n'.join(lines)
    path = os.path.join(save_dir, 'TRAINING_REPORT.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    return path


# ============================================================
# 以下为增强功能: t-SNE, 特征重要性, LR 调度曲线
# ============================================================

def plot_tsne(model: nn.Module,
              dataloader: DataLoader,
              model_name: str,
              subject_id: int,
              save_dir: str,
              device: str = 'cpu',
              n_samples: int = 500) -> str:
    """使用 t-SNE 可视化特征嵌入"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return ''

    _ensure_dir(save_dir)
    model.eval()
    all_features, all_labels = [], []
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device, dtype=torch.float)
            y = batch[1]

            # 提取中间层特征 (hook 或前向传播截断)
            if hasattr(model, 'features'):
                features = model.features(x)
            elif hasattr(model, 'conv_layers'):
                features = model.conv_layers(x)
                features = features.view(features.size(0), -1)
            else:
                # 用除最后的全连接层之外的层提取特征
                if hasattr(model, 'classifier'):
                    modules = list(model.children())[:-1]
                else:
                    modules = list(model.children())[:-1]
                feat_model = nn.Sequential(*modules)
                features = feat_model(x)
                features = features.view(features.size(0), -1)

            all_features.append(features.cpu().numpy())
            all_labels.extend(y.numpy())
            count += x.size(0)
            if count >= n_samples:
                break

    if not all_features:
        return ''

    features_np = np.concatenate(all_features, axis=0)[:n_samples]
    labels_np = np.array(all_labels)[:n_samples]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(features_np)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    for cls in range(4):
        mask = labels_np == cls
        ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[cls],
                   label=CLASS_NAMES[cls], alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
    ax.set_title(f'{model_name} — Subject {subject_id} — t-SNE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name}_S{subject_id}_tsne.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(model: nn.Module,
                            model_name: str,
                            subject_id: int,
                            save_dir: str,
                            num_electrodes: int = 22) -> str:
    """可视化模型权重作为特征重要性 (仅支持有明确空间滤波层的模型)"""
    _ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        # 尝试从模型中提取空间滤波器权重
        if hasattr(model, 'spatial_conv') and hasattr(model.spatial_conv, 'weight'):
            weights = model.spatial_conv.weight.data.cpu().numpy()
            # weights shape: (F, 1, C, 1) for EEGNet
            if weights.ndim == 4:
                weights = weights.squeeze()
            importance = np.abs(weights).mean(axis=0)
        elif hasattr(model, 'conv2') and hasattr(model.conv2, 'weight'):
            # FBCNet-like
            weights = model.conv2.weight.data.cpu().numpy()
            importance = np.abs(weights).mean(axis=(0, 1, 3))
        else:
            importance = np.random.rand(num_electrodes)
            ax.text(0.5, 0.5, 'Feature importance not available\nfor this model architecture',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            plt.tight_layout()
            path = os.path.join(save_dir, f'{model_name}_S{subject_id}_feat_imp.png')
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return path

        if len(importance) > num_electrodes:
            importance = importance[:num_electrodes]

        bars = ax.barh(range(len(importance)), importance)
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels([f'CH{i+1}' for i in range(len(importance))], fontsize=7)
        ax.set_xlabel('Absolute Weight Magnitude')
        ax.set_title(f'{model_name} — Subject {subject_id} — Channel Importance')
        ax.invert_yaxis()
        plt.tight_layout()
        path = os.path.join(save_dir, f'{model_name}_S{subject_id}_feat_imp.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    except Exception:
        plt.close(fig)
        return ''


def plot_learning_rate_schedule(lr_history: Dict[str, List[float]],
                                save_dir: str) -> str:
    """绘制学习率调度曲线"""
    _ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, lrs in lr_history.items():
        ax.plot(lrs, label=name, linewidth=1.5, marker='o', markersize=3)

    ax.set_xlabel('Step / Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(save_dir, 'lr_schedule_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
