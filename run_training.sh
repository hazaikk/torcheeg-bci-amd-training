#!/bin/bash
# ============================================
# AMD Cloud Training — BCICIV2a CNN
# 专为 AMD Radeon Cloud (radeon.anruicloud.com) 设计
#
# 用法:
#   bash run_training.sh              (完整流程)
#   bash run_training.sh quick        (快速验证: 5 epoch)
#   bash run_training.sh test         (仅运行功能测试)
#   bash run_training.sh report       (仅生成报告)
# ============================================
set -e

MODE="${1:-full}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " BCICIV2a TorchEEG Training — AMD ROCm"
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

# ── 1. 安装依赖 ──
install_deps() {
    echo "[1/5] Installing dependencies..."
    pip install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt
    echo ""
}

# ── 2. 准备数据 ──
prepare_data() {
    echo "[2/5] Preparing dataset..."
    python download_data.py
    echo ""
}

# ── 3. 训练模型 ──
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
    ls -lh results/*.png results/*.md results/*.json 2>/dev/null | awk '{print "    " $NF}'
    echo ""
    echo "  Results directory: $(pwd)/results/"
    echo "  Training report: results/TRAINING_REPORT.md"
    echo "  JSON results: results/results.json"
    echo ""
    echo "  To download results from AMD Cloud:"
    echo "    tar czf results.tar.gz results/"
    echo "    # then download results.tar.gz via Jupyter UI"
    echo ""
}

# ── 快速验证 (5 epoch) ──
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
