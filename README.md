# BCICIV2a TorchEEG Training — AMD Developer Cloud

> 在 [AMD Radeon™ Cloud (anruicloud.com)](https://radeon.anruicloud.com/) 上使用免费 GPU 额度，
> 训练 TorchEEG CNN 模型，自动下载 BCICIV2a 数据集，生成完整训练报告。

## 📁 项目结构

```
cloud_amd_training/
├── BCICIV2a_TorchEEG_Training.ipynb  # ★ 主 Notebook (训练 + 可视化, AMD Cloud 入口)
├── TorchEEG_Feature_Test.ipynb       # ★ TorchEEG 功能测试 (7 项)
├── quick_start.ipynb                 # 快速验证 (3 步)
├── preprocess_dataset.py          # ★ 数据集预处理 (持久化.pt, 加速训练)
├── train.py                       # ★ 主训练脚本 (LOSO 交叉验证, 全部模型)
├── test_torcheeg_features.py      # ★ TorchEEG 功能全面测试 (7 项)
├── train_torcheeg_native.py       # TorchEEG 原生 API 使用示例
├── download_data.py               # 数据自动下载 (BNCI Horizon 2020)
├── config.py                      # 集中配置 (超参数, 路径)
├── requirements.txt               # Python 依赖
├── run_training.sh                # AMD Cloud 一键运行脚本
├── create_github_upload.py        # GitHub 上传打包工具
├── preprocess_dataset.py          # 数据集预处理脚本
├── utils/
│   ├── data_utils.py              # 数据加载与预处理
│   ├── model_utils.py             # 模型工厂 + 设备检测
│   └── vis_utils.py               # 可视化 + 报告生成 (含 t-SNE)
├── results/                       # 输出目录 (自动生成)
│   ├── TRAINING_REPORT.md         # ★ 完整训练报告
│   ├── model_comparison.png       # 模型对比图
│   ├── results.json               # 全部数值结果
│   ├── *_curves.png               # 训练曲线 (各受试者)
│   ├── *_cm.png                   # 混淆矩阵 (各受试者)
│   ├── *_tsne.png                 # t-SNE 特征可视化
│   ├── transform_comparison.png   # Transforms 对比
│   ├── optimizer_comparison.png   # 优化器对比
│   └── feature_tests/             # test_torcheeg_features.py 输出
├── data/                          # 数据集 (自动下载)
│   ├── BCICIV2a.mat              # 预组装文件 (5184 样本)
│   ├── A01T.mat ~ A09E.mat       # 单受试者文件
│   └── torcheeg_cache/            # TorchEEG 原生缓存
└── README.md                      # ★ 本文档
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

## 🧠 支持模型

| 模型 | 类型 | 参数量 | 关键特性 |
|------|------|--------|---------|
| **EEGNet** | 紧凑型 CNN | ~3K | 深度可分离卷积, 适合小样本 |
| **TSCeption** | 时序CNN | ~12K | Inception 多尺度时域卷积 |
| **FBCNet** | 频带 CNN | ~12K | 多频带并行 + LogVar 时域聚合 |
| **FBMSNet** | 多尺度 CNN | ~16K | 多尺度频带分解 |

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
