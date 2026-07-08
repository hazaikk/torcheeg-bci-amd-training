# ⚠️ 兼容性修复：必须在任何可能加载 torcheeg 的 import 之前执行
# torcheeg 的 band.py 直接 from scipy.signal import hann（新 scipy 已移除）
import scipy.signal
from scipy.signal.windows import hann, hamming, blackman
scipy.signal.hann = hann
scipy.signal.hamming = hamming
scipy.signal.blackman = blackman

from .fixes import apply_all_fixes
apply_all_fixes()

from .data_utils import (
    load_data, get_subject_split, get_transform, make_dataloader,
    CombinedBCIDataset, precompute_transforms,
    check_preprocessed, load_preprocessed, get_preprocessed_path
)
from .model_utils import create_model, get_criterion, get_device, print_gpu_info, verify_device
from .training_strategies import EarlyStopping, create_scheduler
from .vis_utils import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_subject_results,
    plot_model_comparison,
    generate_report,
    plot_tsne,
    plot_feature_importance,
    plot_learning_rate_schedule,
)
