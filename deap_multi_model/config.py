"""
DEAP 多类型模型训练 - 配置常量

支持模型:
  - Transformer: Conformer, VanillaTransformer
  - RNN: LSTM, GRU
  - GNN: DGCNN, LGGNet, STNet
  - Lightweight: LMDA, CSPNet
  - CNN: MTCNN, SSTEmotionNet, TSLANet
"""

import os

# ── 路径 ──
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'deap')
DEFAULT_PREPROC_DIR = os.path.join(PROJECT_DIR, 'deap_preprocessed')
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# ── DEAP 常量 ──
DEAP_NUM_CHANNELS = 32
DEAP_SAMPLING_RATE = 128
DEAP_NUM_CLASSES = 2
DEAP_CHUNK_SIZE = 128
DEAP_OVERLAP = 0

# ── 频带定义 ──
FBC_BANDS = {f'band{i}': [4 * i, 4 * (i + 1)] for i in range(1, 10)}

# DE 频带 (用于 MTCNN、CCNN)
DE_BANDS = {
    'theta': [4, 8],
    'alpha': [8, 14],
    'beta': [14, 31],
    'gamma': [31, 49],
}

# ── 可用模型 ──
AVAILABLE_MODELS = [
    # 原有 9 个模型
    'Conformer',
    'VanillaTransformer',
    'LSTM',
    'GRU',
    'DGCNN',
    'LGGNet',
    'STNet',
    'LMDA',
    'CSPNet',
    # 新增 3 个模型
    'MTCNN',
    'SSTEmotionNet',
    'TSLANet',
]

# 模型所属类别
MODEL_CATEGORIES = {
    'Conformer': 'Transformer',
    'VanillaTransformer': 'Transformer',
    'LSTM': 'RNN',
    'GRU': 'RNN',
    'DGCNN': 'GNN',
    'LGGNet': 'GNN',
    'STNet': 'GNN',
    'LMDA': 'Lightweight',
    'CSPNet': 'Lightweight',
    'MTCNN': 'CNN',
    'SSTEmotionNet': 'CNN',
    'TSLANet': 'CNN',
}

# ── LGGNet 脑区划分 ──
DEAP_REGION_LIST = [
    [0, 1, 2, 3, 4, 5],      # 前额区 Fp1, AF3, F7, F3, FC1, FC5
    [6, 7, 8, 9, 10],         # 中央区 C3, CP1, CP5, P7, P3
    [11, 12, 13, 14],         # 顶枕区 Pz, PO3, O1, Oz
    [15, 16, 17, 18, 19],     # 右前额 Fp2, AF4, F8, F4, FC2
    [20, 21, 22, 23, 24],     # 右中央 FC6, C4, CP2, CP6, P8
    [25, 26, 27, 28, 29],     # 右顶枕 P4, PO4, O2, Fz, Cz
    [30, 31],                  # 中线/其他
]

# ── MTCNN 专用 8×9 电极布局 (DEAP 32通道映射) ──
# 按照 Rudakov et al. 2021 论文推荐的布局
MTCNN_GRID_8x9 = [
    ['-', '-', 'AF3', 'FP1', '-', 'FP2', 'AF4', '-', '-'],
    ['F7', '-', 'F3', '-', 'FZ', '-', 'F4', '-', 'F8'],
    ['-', 'FC5', '-', 'FC1', '-', 'FC2', '-', 'FC6', '-'],
    ['T7', '-', 'C3', '-', 'CZ', '-', 'C4', '-', 'T8'],
    ['-', 'CP5', '-', 'CP1', '-', 'CP2', '-', 'CP6', '-'],
    ['P7', '-', 'P3', '-', 'PZ', '-', 'P4', '-', 'P8'],
    ['-', '-', '-', 'PO3', '-', 'PO4', '-', '-', '-'],
    ['-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-'],
]

# ── 训练默认参数 ──
TRAIN_DEFAULTS = {
    'epochs': 200,
    'batch_size': 128,
    'lr': 0.001,
    'weight_decay': 0.0,
    'early_patience': 15,
    'n_splits': 5,
    'scheduler': 'cosine',
}

# ── 各模型特有参数 ──
# 参数来源: 各模型原始论文推荐配置 + TorchEEG 默认实现
MODEL_PARAMS = {
    # ═══════ 原有 9 个模型 ═══════
    'Conformer': dict(
        num_electrodes=32, sampling_rate=128,
        hid_channels=32, depth=3, heads=4,
    ),
    'VanillaTransformer': dict(
        num_electrodes=32, chunk_size=128,
        t_patch_size=16, hid_channels=32, depth=3, heads=4,
    ),
    'LSTM': dict(
        num_electrodes=32, hid_channels=64,
    ),
    'GRU': dict(
        num_electrodes=32, hid_channels=64,
    ),
    'DGCNN': dict(
        in_channels=9, num_electrodes=32, hid_channels=32,
    ),
    'LGGNet': dict(
        region_list=DEAP_REGION_LIST, in_channels=1,
        num_electrodes=32, chunk_size=128, sampling_rate=128,
        num_T=15, hid_channels=32, dropout=0.5,
    ),
    'STNet': dict(
        chunk_size=128, grid_size=(9, 9),
    ),
    'LMDA': dict(
        num_electrodes=32, chunk_size=128,
    ),
    'CSPNet': dict(
        chunk_size=128, num_electrodes=32,
    ),

    # ═══════ 新增 3 个模型 ═══════

    # MTCNN (Rudakov et al. 2021)
    # 论文: Multi-Task CNN model for emotion recognition from EEG Brain maps
    # 输入: (batch, 8, 8, 9) — 4 DE bands + 4 PSD bands, 8×9 grid
    # 预处: BandDifferentialEntropy + BandPowerSpectralDensity → Concatenate → ToGrid(8×9)
    # 训练: lr=0.001, batch=128, CosineAnnealing, early_stop=15
    # 注意: 多任务输出 (valence, arousal), 我们只取 valence
    'MTCNN': dict(
        in_channels=8,          # 4 DE bands + 4 PSD bands
        grid_size=(8, 9),       # 8×9 电极网格
        dropout=0.2,
    ),

    # SSTEmotionNet (Jia et al. 2020, ACM MM)
    # 论文: SST-EmotionNet: Spatial-Spectral-Temporal based Attention 3D Dense Network
    # 输入: (batch, 36, 16, 16) — 4 spectral + 32 temporal channels, 16×16 grid
    # 预处: BaselineRemoval → Concatenate([BandDifferentialEntropy, Downsample(32)]) → ToInterpolatedGrid → Resize(16,16)
    # 训练: lr=0.001, batch=64 (显存较大), CosineAnnealing, early_stop=15
    'SSTEmotionNet': dict(
        grid_size=(16, 16),
        spectral_in_channels=4,      # 4 DE 频带 (theta, alpha, beta, gamma)
        temporal_in_channels=32,     # 32 下采样时间点 (128→32)
        spectral_depth=16,           # 光谱流深度 (论文默认)
        temporal_depth=22,           # 时间流深度 (论文默认)
        spectral_growth_rate=12,     # 光谱流增长率 (论文默认)
        temporal_growth_rate=24,     # 时间流增长率 (论文默认)
        num_dense_block=3,           # A3DB 数量
        hid_channels=50,             # 分类头隐藏层
        densenet_dropout=0.0,
        task_dropout=0.0,
    ),

    # TSLANet (Eldele et al. 2024, ICML)
    # 论文: TSLANet: Rethinking Transformers for Time Series Representation Learning
    # 输入: (batch, 32, 128) — 32 electrodes × 128 timepoints (1s)
    # 预处: To2d → ToTensor (与 EEGNet 相同)
    # 训练: lr=0.001, batch=128, CosineAnnealing, early_stop=15
    'TSLANet': dict(
        chunk_size=128,
        patch_size=16,              # 每 patch 16 时间点
        num_electrodes=32,          # 32 通道
        emb_dim=64,                 # 嵌入维度 (减小以防过拟合)
        dropout_rate=0.15,
        depth=2,                    # TSLANet block 数
    ),
}
