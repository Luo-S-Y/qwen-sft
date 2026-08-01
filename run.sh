#!/bin/bash
# AutoDL 4090: 训练 + 评估 一键串联脚本
# 用法: bash run.sh
# 前置: conda activate vllm  (torch>=2.6 + vllm>=0.8.5 + transformers/peft/datasets/sympy)
# 功能:
#   [1/2] 训练 9 组 (LoRA r16后8层 + Full后8/16层 x 100/500/1000 步)
#   [2/2] 评估 10 版本 (有 vllm 用 vLLM 批量, 否则 transformers), 输出对比报告
# 日志: logs/train.log, logs/eval.log (终端同时实时显示)
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "==================== ReasonLite SFT (AutoDL 4090) ===================="
echo "Python: $(python --version 2>&1)"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))" 2>/dev/null || echo "警告: 未检测到 torch/GPU"

echo ""
echo "[1/2] 训练 9 组实验..."
python pipeline_autodl.py --train-only | tee logs/train.log

echo ""
echo "[2/2] 评估 10 版本..."
python pipeline_autodl.py --eval-only | tee logs/eval.log

echo ""
echo "==================== 全部完成 ===================="
echo "对比报告: result/autodl/comparison.md"
echo "评测日志: result/autodl/*.log"
echo "训练日志: logs/train.log"
