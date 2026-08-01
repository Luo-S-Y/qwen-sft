#!/bin/bash
# AutoDL 4090: 训练 + 评估 一键串联脚本
# 用法: bash run.sh   (直接用当前已激活的环境, 不自动装 conda 环境)
# 前置: 1) 当前环境已装 torch/transformers/peft (评测加速可选 vllm>=0.8.5)
#       2) 已跑过 bash set_data.sh (数据/模型下载)
# 功能:
#   [1/2] 训练 9 组 (LoRA r16后8层 + Full后8/16层 x 100/500/1000 步)
#   [2/2] 评估 10 版本 (环境有 vllm 用 vLLM 批量, 否则 transformers generate)
# 日志: logs/train.log, logs/eval.log (终端同时实时显示)
set -euo pipefail
cd "$(dirname "$0")"

# ---------- 0. 环境检查 (不管理 conda 环境) ----------
echo "环境: ${CONDA_DEFAULT_ENV:-系统} ($(python --version 2>&1))"
python -c "import vllm; print('vllm:', vllm.__version__, '(评测加速)')" 2>/dev/null \
    || echo "vllm: 未安装, 评测将使用 transformers generate (装 vllm>=0.8.5 可加速)"

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
