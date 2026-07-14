# DEAP Multi-Model Research

> 在 DEAP 情感识别数据集上训练并对比 TorchEEG 各类非 CNN 模型。

## 支持模型

| 类别 | 模型 | 输入形状 |
|------|------|---------|
| Transformer | **Conformer** | `(1, 32, T)` |
| Transformer | **VanillaTransformer** | `(32, T)` |
| RNN | **LSTM** | `(32, T)` |
| RNN | **GRU** | `(32, T)` |
| GNN | **DGCNN** | `(32, 9)` 节点特征 |
| GNN | **LGGNet** | `(1, 32, T)` |
| GNN | **STNet** | `(T, 9, 9)` 电极网格 |
| Lightweight | **LMDA** | `(1, 32, T)` |
| Lightweight | **CSPNet** | `(1, 32, T)` |

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
# 所有模型
python preprocess.py --models all --gpu

# 指定模型
python preprocess.py --models Conformer LSTM --gpu
```

### 4. 训练

```bash
# 所有模型
python train.py --models all --gpu

# 指定模型
python train.py --models Conformer LMDA --gpu --epochs 100

# 调整超参数
python train.py --models LSTM --lr 0.0005 --batch-size 64 --epochs 150

# 2s 窗口
python train.py --models all --chunk-size 256 --gpu
```

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
├── LSTM/
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
