# DEAP Multi-Model Research

> 在 DEAP 情感识别数据集上训练并对比 TorchEEG 各类模型 (CNN + 非 CNN)。

## 支持模型

| 类别 | 模型 | 输入形状 | 论文 |
|------|------|---------|------|
| Transformer | **Conformer** | `(1, 32, T)` | Conformer (2021) |
| Transformer | **VanillaTransformer** | `(32, T)` | Attention Is All You Need (2017) |
| RNN | **LSTM** | `(32, T)` | Long Short-Term Memory (1997) |
| RNN | **GRU** | `(32, T)` | GRU (2014) |
| GNN | **DGCNN** | `(32, 9)` 节点特征 | Dynamic Graph CNN (2018) |
| GNN | **LGGNet** | `(1, 32, T)` | LGGNet (2021) |
| GNN | **STNet** | `(T, 9, 9)` 电极网格 | 时空网络 |
| Lightweight | **LMDA** | `(1, 32, T)` | LMDA (2022) |
| Lightweight | **CSPNet** | `(1, 32, T)` | CSP+CNN |
| 🆕 CNN | **MTCNN** | `(8, 8, 9)` 特征网格 | Rudakov et al. (2021) |
| 🆕 CNN | **SSTEmotionNet** | `(36, 16, 16)` 特征网格 | Jia et al. (2020, ACM MM) |
| 🆕 CNN | **TSLANet** | `(32, T)` | Eldele et al. (2024, ICML) |

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt

# Python 3.12 兼容
pip install scipy --upgrade
pip install torcheeg --no-deps
pip install torchmetrics lmdb pytorch-lightning
```

### 2. 下载 DEAP 数据

```bash
# 选项 A: 使用 download_deap.py (需要 kagglehub)
python ../cloud_amd_training_for_github/download_deap.py --data-dir ./data/deap

# 选项 B: 手动下载
# wget http://www.eecs.qmul.ac.uk/mmv/datasets/deap/data/data_preprocessed_python.zip
# 解压后将 s01.dat~s32.dat 放入 ./data/deap/
```

### 3. 预处理 (生成 .pt 文件)

```bash
# 所有模型 (含新增 MTCNN, SSTEmotionNet, TSLANet)
python preprocess.py --models all --gpu

# 仅新增模型
python preprocess.py --models MTCNN SSTEmotionNet TSLANet --gpu

# 指定模型
python preprocess.py --models Conformer LSTM --gpu
```

### 4. 训练

```bash
# 所有模型
python train.py --models all --gpu

# 仅训练新增的 3 个模型
python train.py --models MTCNN SSTEmotionNet TSLANet --gpu

# 指定模型
python train.py --models Conformer LMDA --gpu --epochs 100

# 调整超参数
python train.py --models LSTM --lr 0.0005 --batch-size 64 --epochs 150
```

## 新增模型详细说明

### MTCNN — Multi-Task CNN

| 项目 | 说明 |
|------|------|
| 论文 | Rudakov et al., "Multi-Task CNN model for emotion recognition from EEG Brain maps", BioSMART 2021 |
| 输入 | 4 DE bands + 4 PSD bands → 8×9 电极网格 |
| 参数量 | ~1.5M |
| 特点 | 多任务输出 (valence + arousal), 使用脑地形图 |

### SSTEmotionNet — Spatial-Spectral-Temporal Emotion Network

| 项目 | 说明 |
|------|------|
| 论文 | Jia et al., "SST-EmotionNet: Spatial-Spectral-Temporal based Attention 3D Dense Network", ACM MM 2020 |
| 输入 | 4 DE spectral bands + 32 temporal points → 16×16 插值网格 |
| 参数量 | ~3M |
| 特点 | 双流 3D DenseNet + 空谱时注意力, 在 SEED 达 96.02% |

### TSLANet — Time Series Lightweight Adaptive Network

| 项目 | 说明 |
|------|------|
| 论文 | Eldele et al., "TSLANet: Rethinking Transformers for Time Series Representation Learning", ICML 2024 |
| 输入 | (32, 128) — 标准 2D EEG 输入 |
| 参数量 | ~62K |
| 特点 | 频域自适应滤波 + 轻量卷积, 快速高效 |

## 输出结构

```
results/MM_128pt_1s_kfold_groupby_trial_<timestamp>/
├── config.json                    # 训练配置
├── experiment_summary.csv         # 所有模型汇总表
├── Conformer/
│   ├── summary.json               # 指标汇总
│   ├── all_epochs.csv             # 全部 epoch 指标
│   ├── best_model.pt              # 最佳模型权重
│   ├── fold_1_metrics.csv         # 每折详细指标
│   └── fold_2_metrics.csv
├── MTCNN/
│   └── ...
├── SSTEmotionNet/
│   └── ...
├── TSLANet/
│   └── ...
└── ...
```

## 项目结构

```
deap_multi_model/
├── preprocess.py          # 数据预处理 → .pt
├── train.py               # 训练 + 评估
├── config.py              # 配置 (模型参数, 路径)
├── utils.py               # 工具 (EarlyStopping, scheduler)
├── requirements.txt       # 依赖
└── README.md              # 本文件
```
