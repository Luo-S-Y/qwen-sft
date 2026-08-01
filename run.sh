#!/bin/bash
# AutoDL 4090: 训练 + 评估 一键串联脚本
# 用法: bash run.sh   (新实例直接跑即可, 会自动安装 vllm 环境 + 检查数据/模型)
# 前置: 已跑过 bash set_data.sh (数据/模型下载, 模型不在时本脚本会报错提示)
# 功能:
#   [0]   自动准备 vllm 环境 (缺失/过旧则运行 setup_vllm.sh 安装)
#   [1/2] 训练 9 组 (LoRA r16后8层 + Full后8/16层 x 100/500/1000 步)
#   [2/2] 评估 10 版本 (vLLM 批量, 若环境无 vllm 自动回退 transformers)
# 日志: logs/train.log, logs/eval.log (终端同时实时显示)
set -euo pipefail
cd "$(dirname "$0")"

# ---------- 0. 确保 vllm 环境可用 (缺失/过旧则自动安装) ----------
load_conda() {
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        for p in /root/miniconda3 /opt/conda "$HOME/miniconda3"; do
            [ -f "$p/etc/profile.d/conda.sh" ] && { source "$p/etc/profile.d/conda.sh"; return 0; }
        done
        echo "错误: 未找到 conda, 请确认 AutoDL 环境已装 miniconda"; exit 1
    fi
}
if [ "${CONDA_DEFAULT_ENV:-}" != "vllm" ]; then
    load_conda
    if conda env list | grep -qE '^vllm[[:space:]]'; then
        echo "激活环境: vllm"
        conda activate vllm
    fi
fi
# 验证 vllm>=0.8.5 (Qwen3 支持), 缺失/过旧则自动安装
if ! python -c "import vllm; assert tuple(map(int, vllm.__version__.split('.')[:2])) >= (0, 8)" 2>/dev/null; then
    echo "vllm 环境缺失或版本过旧, 自动安装 (bash setup_vllm.sh)..."
    [ -f setup_vllm.sh ] || { echo "错误: 缺少 setup_vllm.sh"; exit 1; }
    bash setup_vllm.sh
    load_conda
    conda activate vllm
fi
echo "环境: $CONDA_DEFAULT_ENV ($(python --version 2>&1))"

# ---------- 前置检查: 数据/模型 ----------
if [ ! -f data/lora_short/train.jsonl ] || [ ! -f data/lora_short/valid.jsonl ]; then
    echo "错误: 数据集缺失 (data/lora_short/), 请先运行: bash set_data.sh"; exit 1
fi
if [ ! -f models/Qwen3-0.6B/config.json ]; then
    echo "错误: 模型缺失 (models/Qwen3-0.6B), 请先运行: bash set_data.sh"; exit 1
fi

mkdir -p logs

echo "==================== ReasonLite SFT (AutoDL 4090) ===================="
echo "Python: $(python --version 2>&1)"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))" 2>/dev/null || echo "警告: 未检测到 torch/GPU"

echo ""
echo "[1/2] 训练 9 组实验..."
python pipeline_autodl.py --train-only 2>&1 | tee logs/train.log

echo ""
echo "[2/2] 评估 10 版本..."
python pipeline_autodl.py --eval-only 2>&1 | tee logs/eval.log

echo ""
echo "==================== 全部完成 ===================="
echo "对比报告: result/autodl/comparison.md"
echo "评测日志: result/autodl/*.log"
echo "训练日志: logs/train.log"
