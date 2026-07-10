#!/bin/bash
# ============================================
# AMD Cloud Training — BCICIV2a + DEAP
# 专为 AMD Radeon Cloud (radeon.anruicloud.com) 设计
#
# Python 3.12 兼容:
#   torcheeg 官方要求 scipy<=1.10.1, 但 Python 3.12 需 scipy>=1.12.0。
#   解决方法: 先装 scipy (最新版), 再装 torcheeg 时跳过依赖检查。
#
# 用法:
#   bash run_training.sh                    (完整流程: BCIC)
#   bash run_training.sh quick              (快速验证: 5 epoch)
#   bash run_training.sh deap               (DEAP 情感识别)
#   bash run_training.sh test               (仅功能测试)
#   bash run_training.sh report             (仅生成报告)
# ============================================
set -e

MODE="${1:-full}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " TorchEEG Training — AMD ROCm"
echo " Date: $(date)"
echo " Mode: ${MODE}"
echo "============================================"
echo ""

# ── 0. 环境检测 ──
detect_env() {
    echo "[0/5] Environment detection..."
    echo "  Python: $(python --version 2>&1)"
    echo "  Working dir: $(pwd)"

    python -c "
import torch, torcheeg
print(f'  PyTorch: {torch.__version__}')
print(f'  TorchEEG: {torcheeg.__version__}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  GPU: {props.name}')
    print(f'  Memory: {props.total_mem / 1024**3:.1f} GiB')
    print(f'  CUDA/ROCm: {torch.version.cuda}')
else:
    print('  GPU: None (CPU mode)')
"
    echo ""
}

# ── 1. 安装依赖 (兼容 Python 3.12) ──
install_deps() {
    echo "[1/5] Installing dependencies..."

    # Python 3.12 兼容: scipy 版本冲突处理
    # torcheeg 约束 scipy<=1.10.1 但 Python 3.12 需要 scipy>=1.12.0
    PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "  Python version: ${PY_VER}"

    if [ "$PY_VER" = "3.12" ] || [ "$PY_VER" = "3.13" ]; then
        echo "  Python 3.12+ detected — applying scipy workaround for torcheeg compatibility..."

        # 1. 先装基础依赖 (不加 scipy, 避免版本冲突)
        pip install numpy>=1.24.0 torch>=2.0.0 -q

        # 2. 装最新 scipy (Python 3.12 需要 scipy>=1.12)
        pip install scipy --upgrade -q

        # 3. 装 torcheeg 其他依赖 (跳过 scipy 检查)
        pip install torcheeg --no-deps -q

        # 4. 手动装 torcheeg 的运行时依赖 (排除 scipy 版本约束)
        pip install torchmetrics>=0.11.0 \
                    pytorch-lightning>=1.9.0 \
                    sleepingnet-pytorch>=0.0.1 \
                    lmdb>=1.4.0 \
                    -q 2>/dev/null || true

        # 5. 装其余工具
        pip install scikit-learn>=1.2.0 \
                    matplotlib>=3.7.0 \
                    seaborn>=0.12.0 \
                    tqdm>=4.65.0 \
                    pandas>=2.0.0 \
                    kagglehub>=0.3.0 \
                    ipykernel>=6.0.0 \
                    -q

        echo "  Python 3.12 workaround applied."
    else
        # 标准安装
        pip install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt
    fi

    # 验证 torcheeg 能正常导入
    python -c "
import scipy.signal
from scipy.signal.windows import hann, hamming, blackman
scipy.signal.hann = hann
scipy.signal.hamming = hamming
scipy.signal.blackman = blackman

import torcheeg
print(f'  TorchEEG {torcheeg.__version__} — OK')

import scipy
print(f'  SciPy {scipy.__version__} — OK')
"
    echo ""
}

# ── 2. 准备数据 ──
prepare_data() {
    echo "[2/5] Preparing dataset..."
    python download_data.py
    echo ""
}

# ── 3. 训练模型 (BCICIV2a) ──
run_training() {
    local EPOCHS=$1
    echo "[3/5] Training models ($EPOCHS epochs each)..."
    python train.py \
        --models EEGNet TSCeption FBCNet FBMSNet \
        --epochs "$EPOCHS" \
        --batch-size 64 \
        --lr 0.001 \
        --results-dir results \
        --data-dir data \
        --device auto
    echo ""
}

# ── 3b. DEAP 训练 ──
run_deap_training() {
    echo "[3/5] Training DEAP models..."

    # 下载 DEAP 数据集 (如果尚未下载)
    if [ ! -f "data/deap/s01.dat" ]; then
        echo "  Downloading DEAP dataset..."
        python download_deap.py --data-dir data/deap
    fi

    # 预处理 (只需一次)
    echo "  Preprocessing DEAP data..."
    python preprocess_deap.py \
        --models EEGNet TSCeption FBCNet FBMSNet CCNN \
        --chunk-size 128 \
        --data-dir data/deap \
        --output-dir data/deap_preprocessed

    # 训练
    echo "  Training DEAP models (Table 1 reproduction)..."
    python train_deap.py \
        --models EEGNet TSCeption FBCNet FBMSNet CCNN \
        --chunk-size 128 \
        --cv kfold_groupby_trial \
        --use-preprocessed \
        --preproc-dir data/deap_preprocessed \
        --epochs 100
    echo ""
}

# ── 4. TorchEEG 功能测试 ──
run_feature_tests() {
    echo "[4/5] Running TorchEEG feature tests..."
    python test_torcheeg_features.py
    echo ""
}

# ── 5. 汇总结果 ──
show_results() {
    echo "[5/5] Results summary:"
    echo ""
    echo "  Generated files:"
    ls -lh results/*.png results/*.md results/*.json 2>/dev/null | awk '{print "    " $NF}' || echo "    (no results yet)"
    echo ""
    echo "  Results directory: $(pwd)/results/"
    echo ""

    # 如 DEAP 结果存在, 也展示
    ls -d results/DEAP_*/ 2>/dev/null | while read d; do
        echo "  DEAP results: $d"
    done

    echo ""
    echo "  To download results from AMD Cloud:"
    echo "    tar czf results.tar.gz results/"
    echo "    # then download results.tar.gz via Jupyter UI"
    echo ""
}

# ── 快速验证 (BCIC 5 epoch) ──
run_quick() {
    detect_env
    install_deps
    prepare_data
    echo "[QUICK] Training with 5 epochs..."
    python train.py \
        --models EEGNet FBCNet \
        --epochs 5 \
        --batch-size 64 \
        --lr 0.001 \
        --results-dir results \
        --data-dir data \
        --device auto
    show_results
}

# ── 仅测试 ──
run_test_only() {
    detect_env
    install_deps
    prepare_data
    python test_torcheeg_features.py
}

# ── 主流程 ──
case "$MODE" in
    quick)
        run_quick
        ;;
    deap)
        detect_env
        install_deps
        run_deap_training
        show_results
        ;;
    test)
        run_test_only
        ;;
    report)
        show_results
        ;;
    full|*)
        detect_env
        install_deps
        prepare_data
        run_training 30
        run_feature_tests
        show_results
        ;;
esac

echo ""
echo "============================================"
echo " Done! Mode: ${MODE}"
echo "============================================"
