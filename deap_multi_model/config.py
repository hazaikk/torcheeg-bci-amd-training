"""
DEAP 多类型模型训练 - 配置常量
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

# ── 可用模型 ──
AVAILABLE_MODELS = [
    'Conformer',
    'VanillaTransformer',
    'LSTM',
    'GRU',
    'DGCNN',
    'LGGNet',
    'STNet',
    'LMDA',
    'CSPNet',
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
}

# ── LGGNet 脑区划分 ──
# DEAP 32 通道的粗略脑区 (按标准 10-20 系统大致分组)
DEAP_REGION_LIST = [
    [0, 1, 2, 3, 4, 5],      # 前额区 Fp1, AF3, F7, F3, FC1, FC5
    [6, 7, 8, 9, 10],         # 中央区 C3, CP1, CP5, P7, P3
    [11, 12, 13, 14],         # 顶枕区 Pz, PO3, O1, Oz
    [15, 16, 17, 18, 19],     # 右前额 Fp2, AF4, F8, F4, FC2
    [20, 21, 22, 23, 24],     # 右中央 FC6, C4, CP2, CP6, P8
    [25, 26, 27, 28, 29],     # 右顶枕 P4, PO4, O2, Fz, Cz
    [30, 31],                  # 中线/其他
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
MODEL_PARAMS = {
    'Conformer': dict(num_electrodes=32, sampling_rate=128, hid_channels=32, depth=3, heads=4),
    'VanillaTransformer': dict(num_electrodes=32, chunk_size=128, t_patch_size=16, hid_channels=32, depth=3, heads=4),
    'LSTM': dict(num_electrodes=32, hid_channels=64),
    'GRU': dict(num_electrodes=32, hid_channels=64),
    'DGCNN': dict(in_channels=9, num_electrodes=32, hid_channels=32),
    'LGGNet': dict(region_list=DEAP_REGION_LIST, in_channels=1, num_electrodes=32, chunk_size=128, sampling_rate=128, num_T=15, hid_channels=32, dropout=0.5),
    'STNet': dict(chunk_size=128, grid_size=(9, 9)),
    'LMDA': dict(num_electrodes=32, chunk_size=128),
    'CSPNet': dict(chunk_size=128, num_electrodes=32),
}
