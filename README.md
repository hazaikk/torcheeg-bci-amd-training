# BCICIV2a + DEAP TorchEEG Training — AMD Developer Cloud

> 在 [AMD Radeon™ Cloud (anruicloud.com)](https://radeon.anruicloud.com/) 上使用免费 GPU 额度，
> 训练 TorchEEG CNN 模型，支持 **BCICIV2a (运动想象)** 和 **DEAP (情感识别)** 双数据集。

## 📁 项目结构

```
cloud_amd_training/
│
├── BCICIV2a 运动想象
│   ├── BCICIV2a_TorchEEG_Training.ipynb  # ★ 主 Notebook (训练 + 可视化)
│   ├── train.py                       # ★ 主训练脚本 (LOSO, 全部模型)
│   ├── preprocess_dataset.py          # ★ 预处理 (持久化.pt)
│   ├── download_data.py               # 自动下载 (BNCI Horizon 2020)
│   └── train_torcheeg_native.py       # TorchEEG 原生 API 示例
│
├── DEAP 情感识别 (新增)
│   ├── train_deap.py                  # ★ DEAP 训练脚本 (Table 1 复现)
│   ├── preprocess_deap.py             # ★ DEAP 预处理 (持久化.pt)
│   └── download_deap.py               # 自动下载 (KaggleHub)
│
├── 通用
│   ├── config.py                      # 集中配置 (超参数, 路径)
│   ├── test_torcheeg_features.py      # TorchEEG 功能测试 (7 项)
│   ├── quick_start.ipynb              # 快速验证 (3 步)
│   ├── TorchEEG_Feature_Test.ipynb    # TorchEEG 功能测试 Notebook
│   ├── requirements.txt               # Python 依赖
│   ├── run_training.sh                # AMD Cloud 一键运行脚本
│   ├── check_gpu.py                   # GPU 诊断
│   ├── utils/
│   │   ├── data_utils.py              # 数据加载
│   │   ├── model_utils.py             # 模型工厂 + 设备检测
│   │   ├── training_strategies.py     # 早停 + LR 调度
│   │   ├── fixes.py                   # 兼容性修复
│   │   ├── vis_utils.py               # 可视化
│   │   └── __init__.py
│   └── results/                       # 输出目录 (自动生成)
├── data/                              # 数据集 (自动下载)
│   ├── BCICIV2a.mat                  # BCIC 预组装
│   ├── A01T.mat ~ A09E.mat           # BCIC 单受试者
│   └── deap/                          # DEAP (s01.dat ~ s32.dat)
└── README.md
```

## 🚀 AMD Developer Cloud 快速开始

### 1. 登录并创建模型

1. 访问 https://developer.amd.com.cn/radeon/profile?bind=1
2. 填写:
   - **GitHub Repo URL**: `https://github.com/<你的用户名>/<你的仓库>`
   - **Notebook Path**: `cloud_amd_training/BCICIV2a_TorchEEG_Training.ipynb`
3. 平台会自动克隆仓库并启动 Jupyter Notebook

### 2. 在 Notebook 中运行

打开 `BCICIV2a_TorchEEG_Training.ipynb`，依次执行各 cell:

| Cell | 操作 | 耗时 |
|------|------|------|
| 0. 环境检测 | 确认 GPU 可用 | ~1s |
| 1. 安装依赖 | `pip install -r requirements.txt` | ~2min |
| 2. 下载数据 | 从 BNCI 自动下载 (~700 MB) | ~5-15min |
| 3. 数据探索 | 查看数据集信息 | ~1s |
| 4. 模型训练 | 配置参数 → 启动训练 | ~20-60min |
| 5-7. 结果展示 | 曲线 / 矩阵 / 报告 | 自动 |

**可选 Notebook:**
- `quick_start.ipynb` — 3 步快速验证 (5 epoch)
- `TorchEEG_Feature_Test.ipynb` — 7 项 TorchEEG 功能全面测试

### 3. 或者通过 Terminal 运行 (传统方式)

启动 Notebook 后，打开 Terminal：

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 下载数据 (~700 MB, 自动从 BNCI 下载)
python download_data.py

# 3) 【可选】数据集预处理 (只需跑一次)
#    将 BandSignal FFT 等运算结果存为 .pt 文件，后续训练加 --use-preprocessed 跳过重计算
python preprocess_dataset.py --models EEGNet TSCeption FBCNet FBMSNet

# 4) 训练 (4 个模型, 每模型 30 epoch)
python train.py \
    --models EEGNet TSCeption FBCNet FBMSNet \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.001 \
    --device auto

# 使用预处理数据加速 (跳过 BandSignal FFT):
python train.py \
    --models EEGNet TSCeption FBCNet FBMSNet \
    --epochs 30 \
    --use-preprocessed

# 5) 运行 TorchEEG 全面功能测试 (7 项测试)
python test_torcheeg_features.py
```

### 5. 下载结果

训练完成后, 在 Notebook 中打包结果:
```bash
# 打包结果目录
tar czf results.tar.gz results/
# 或直接在 Jupyter 中通过右键下载
```

---

## 🧠 DEAP 情感识别 (Table 1 复现)

在 DEAP 数据集上复现 [TorchEEG EMO 论文](https://arxiv.org/abs/2401.14571) Table 1 的 Valence 二分类准确率。

### 数据集: DEAP

- 32 名受试者, 40 段 63s 视频刺激
- 32 EEG 通道 + 8 外周生理通道, 128Hz 采样率
- 评分维度: valence / arousal / dominance / liking (1-9 连续值)
- 任务: **Valence 二分类** (high > 5 vs low ≤ 5)

### 支持模型

| 模型 | 参数量 | 输入格式 | 备注 |
|------|--------|---------|------|
| **EEGNet** | 1.7K | (1, 32, T) | 深度可分离卷积 |
| **TSCeption** | 13.9K | (1, 32, T) | Inception 多尺度时域 |
| **FBCNet** | 12.4K | (9, 32, T) | 9频带并行 + LogVar |
| **FBMSNet** | 16.8K | (9, 32, T) | 多尺度频带分解 |
| **CCNN** | 6.2M | (T, 9, 9) 网格 | 紧凑型 2D CNN |

### 快速开始

```bash
# 1. 安装依赖 (KaggleHub)
pip install kagglehub

# 2. 下载 DEAP 数据集
python download_deap.py
# 或指定目录:
python download_deap.py --data-dir ./data/deap

# 3. 【推荐】预处理 (只需跑一次, 将 transforms 结果存为 .pt)
#    支持 1 秒 (128pt) 和 2 秒 (256pt) 窗口
python preprocess_deap.py --models all --chunk-size 128
python preprocess_deap.py --models all --chunk-size 256

# 4. 训练 (预处理模式, 最快 — 跳过所有 transforms)
python train_deap.py --models EEGNet TSCeption --use-preprocessed

# 5. 原生模式 (TorchEEG DEAPDataset + 自动 LMDB 缓存)
python train_deap.py --models FBCNet FBMSNet --chunk-size 128 --cv kfold_groupby_trial

# 6. 快速测试 (1 epoch, 2 folds)
python train_deap.py --models EEGNet --test

# 7. 多模型 + 不同交叉验证策略
python train_deap.py --models all --cv leave_one_subject_out
python train_deap.py --models EEGNet FBCNet --cv kfold --n-splits 10
python train_deap.py --models all --chunk-size 256 --cv kfold_per_subject_groupby_trial
```

### 复现论文 Table 1

```bash
# 1 秒窗口 + KFoldGroupbyTrial (5折)
python preprocess_deap.py --models all --chunk-size 128
python train_deap.py --models all --chunk-size 128 --cv kfold_groupby_trial --use-preprocessed
```

### 交叉验证策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `kfold_groupby_trial` (默认) | 按 trial 分组 KFold | **论文 Table 1** |
| `kfold` | 常规 KFold | 通用 |
| `kfold_per_subject_groupby_trial` | 每受试者内按 trial 分组 | 受试者独立评估 |
| `leave_one_subject_out` | 留一受试者交叉验证 | 跨受试者泛化 |

### 输出格式

训练结果保存在 `results/DEAP_<窗口>_<策略>_<时间戳>/` 下:
- `config.json` — 训练配置
- `<模型>/summary.json` — 指标汇总 (mean/std/best acc)
- `<模型>/best_model.pt` — 最佳模型权重
- `<模型>/all_epochs.csv` — 全部 epoch 的 train/val 指标
- `<模型>/fold_N_metrics.csv` — 每折详细指标
- `table1_summary.csv` — 所有模型汇总表

---

## 🧪 TorchEEG 功能测试矩阵

| 测试编号 | 测试内容 | 涵盖模块 |
|---------|---------|---------|
| Test 1 | 版本与运行环境 | `torcheeg.__version__`, 可用模型列表 |
| Test 2 | 原生数据集加载 | `BCICIV2aDataset`, `io_mode='pickle'` |
| Test 3 | Transforms 流水线 | `To2d`, `MeanStdNormalize`, `BandSignal`, `TimeMask`, `GaussianNoise`, `ChannelDropout`, `ToGrid` |
| Test 4 | 模型快速对比 | EEGNet, FBCNet, FBMSNet, CSPNet, LMDA (参数量/准确率/时间) |
| Test 5 | 优化器与 LR 调度 | AdamW+SGD × CosineAnnealing+StepLR+ReduceLROnPlateau |
| Test 6 | 评价度量 | Acc, Balanced Acc, Kappa, F1, Precision, Recall, AUC-OVR |
| Test 7 | 完整 LOSO + 报告 | 9 受试者 LOSO, 生成完整 Markdown 报告 |

## 🧠 支持模型 (BCICIV2a)

| 模型 | 类型 | 参数量 | 关键特性 |
|------|------|--------|---------|
| **EEGNet** | 紧凑型 CNN | ~3K | 深度可分离卷积, 适合小样本 |
| **TSCeption** | 时序CNN | ~12K | Inception 多尺度时域卷积 |
| **FBCNet** | 频带 CNN | ~12K | 多频带并行 + LogVar 时域聚合 |
| **FBMSNet** | 多尺度 CNN | ~16K | 多尺度频带分解 |
| **CSPNet** | 混合 CNN | ~10K | 时空卷积组合 |
| **LMDA** | 轻量 CNN | ~15K | 通道+深度注意力 |

## 📊 输出说明

### 训练报告 (`results/TRAINING_REPORT.md`)
- 模型对比表格 (准确率 / 标准差 / 时间)
- 各模型 9 受试者详细结果表
- 混淆矩阵图引用 (归一化)
- 训练曲线图引用
- t-SNE 特征可视化引用
- 结论与优化建议

### 图表
| 文件 | 说明 |
|------|------|
| `model_comparison.png` | 模型间准确率 + 时间对比 |
| `*_curves.png` | 各受试者训练曲线 (Loss + Acc) |
| `*_cm.png` | 各受试者混淆矩阵 (归一化) |
| `*_per_subject.png` | 各模型 9 受试者柱状图 |
| `*_tsne.png` | t-SNE 特征嵌入可视化 |
| `transform_comparison.png` | Transforms 效果对比 |
| `optimizer_comparison.png` | 优化器 + LR 调度对比 |

### 数值结果 (`results/results.json`)
- 每受试者: accuracy, kappa, f1, precision, recall, confusion_matrix
- 汇总: mean/std accuracy, mean kappa, mean f1

## ⚙️ 参数说明

```bash
python train.py --help

# 关键参数:
--models         模型列表 (默认: EEGNet FBCNet FBMSNet)
--epochs         训练轮数 (默认: 50, 建议 AMD Cloud: 30)
--batch-size     批次大小 (默认: 64, AMD 16GiB GPU 可用 128)
--lr             学习率 (默认: 0.001)
--results-dir    输出目录 (默认: results)
--data-dir       数据目录 (默认: data)
--device         设备: auto / cuda / cpu (默认: auto)
--use-augmentation 启用数据增强
--use-preprocessed 使用预处理 .pt 文件 (跳过 BandSignal FFT, 加速训练)
```

## 📦 数据集: BCI Competition IV 2a

- 9 名受试者 (A01–A09)
- 4 类运动想象: 左手(1) / 右手(2) / 双脚(3) / 舌头(4)
- 22 EEG 通道, 250 Hz 采样率
- 每受试者 2 sessions × 6 runs × 48 trials = 576 trials
- 总计: 5184 trials, 每 trial 800 时间点 (3.2s)

数据自动从 [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets/001-2014/) 下载。
源数据为单受试者 `.mat` 文件 (A01T.mat–A09E.mat), 首次运行自动下载并组装为 `BCICIV2a.mat`。

## 🏗 AMD ROCm 兼容性

本代码在 AMD ROCm 环境下测试通过:

| 组件 | 状态 |
|------|------|
| `torch.cuda.is_available()` | ✅ 在 ROCm 下返回 `True` |
| AMD GPU 检测 | ✅ 自动识别 AMD/Instinct/Radeon |
| PyTorch ROCm 5.6+ | ✅ `torch>=2.0.0` 兼容 |
| 训练速度 (16 GiB GPU) | ✅ 30 epoch × 3 models ≈ 20 min |
| 内存使用 | ✅ < 8 GiB for batch_size=64 |

## 🐍 Python 3.12 兼容: scipy 版本冲突

AMD Cloud 默认 Python 3.12, 而 torcheeg 官方要求 `scipy<=1.10.1`。
Python 3.12 需要 `scipy>=1.12.0`, 产生冲突。

**解决方法:** 先装 scipy, 再装 torcheeg 时跳过依赖检查:

```bash
# ❌ 这样会失败:
pip install torcheeg  # 因为 scipy 版本约束冲突

# ✅ 正确方法:
pip install scipy --upgrade          # 先装最新 scipy (>=1.12)
pip install torcheeg --no-deps       # 跳过依赖检查
pip install torchmetrics lmdb pytorch-lightning  # 手动装 torcheeg 的其他依赖

# ✅ 或直接用脚本:
bash run_training.sh deap           # run_training.sh 已自动处理 Python 3.12 兼容
```

`run_training.sh` 和 `requirements.txt` 均已内置 Python 3.12 兼容逻辑。

---

## 📥 DEAP 数据集下载慢的解决方案

DEAP 数据集 ~1.4 GB (32 个 .dat 文件), 从 Kaggle 下载在 AMD Cloud 可能较慢。

### 方案 A: 本地下载后上传 (推荐)

```bash
# 1. 在本地 (Windows/Mac) 下载
#    方法 1: 使用 kagglehub
pip install kagglehub
python download_deap.py --data-dir ./deap_data

#    方法 2: 从官方直接下载
#    http://www.eecs.qmul.ac.uk/mmv/datasets/deap/data/data_preprocessed_python.zip
#    解压得到 data_preprocessed_python/ 目录 (含 s01.dat ~ s32.dat)

# 2. 在 AMD Cloud Jupyter 中上传
#    - 打开 Jupyter File Browser
#    - Upload → 选择 s01.dat ~ s32.dat
#    - 或者打包上传: tar czf deap_data.tar.gz deap_data/
#    - 然后在 Jupyter 中解压: tar xzf deap_data.tar.gz
```

### 方案 B: wget 直接从官方下载 (AMD Cloud 命令行)

```bash
# 在 AMD Cloud Terminal 中执行:
cd /workspace/repo
mkdir -p data/deap && cd data/deap

# 官方 DEAP 数据集 (约 1.4 GB)
wget -c http://www.eecs.qmul.ac.uk/mmv/datasets/deap/data/data_preprocessed_python.zip
unzip data_preprocessed_python.zip
mv data_preprocessed_python/*.dat ./
rm -rf data_preprocessed_python data_preprocessed_python.zip
cd ../..
python preprocess_deap.py --models all --data-dir data/deap
```

### 方案 C: KaggleHub + 代理

```bash
# 设置国内代理 (如适用)
export KAGGLEHUB_CACHE=/workspace/repo/kagglehub_cache
python download_deap.py --data-dir data/deap
```

> **提示:** DEAP 只需下载一次, preprocess_deap.py 预处理后即可删除原始 .dat 文件。

---

## 🔧 常见问题

### Q: pip install torcheeg 失败?
```bash
# 尝试从源码安装
pip install git+https://github.com/torcheeg/torcheeg.git

# 或指定版本
pip install torcheeg==1.1.0
```

### Q: 下载数据慢?
```bash
# 手动下载后放置到 data/ 目录
# 单个文件下载地址:
# http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat
# http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01E.mat
# ... 到 A09E.mat
```

### Q: 内存不足 (OOM)?
```bash
# 减小 batch size
python train.py --batch-size 32
# 或减少同时训练的模型数
python train.py --models EEGNet
```

### Q: 如何在 GitHub 上分享?
```bash
# 使用打包脚本
python create_github_upload.py
# 然后上传 cloud_amd_training_for_github/ 目录到 GitHub
```

## 📚 参考文献

- [TorchEEG: A Deep Learning Toolbox for EEG](https://github.com/torcheeg/torcheeg)
- [EEGNet: A Compact CNN for EEG](https://arxiv.org/abs/1611.08024)
- [FBCNet: Filter Bank CNN for MI](https://ieeexplore.ieee.org/document/9233841)
- [FBMSNet: Multi-Scale CNN for MI](https://ieeexplore.ieee.org/document/9837422)
- [BCI Competition IV 2a](https://www.bbci.de/competition/iv/)
