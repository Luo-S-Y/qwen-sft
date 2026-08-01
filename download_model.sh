#!/bin/bash
# AutoDL 模型下载脚本（hf-mirror 加速，下载到本地 models/ 目录）
# 用法: bash download_model.sh [模型名] [目标目录]
#   bash download_model.sh                     # 默认 Qwen/Qwen3-0.6B -> models/Qwen3-0.6B
#   bash download_model.sh Qwen/Qwen3-0.6B     # 指定模型
set -e

MODEL=${1:-Qwen/Qwen3-0.6B}
DEST=${2:-models/Qwen3-0.6B}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "==================== 下载模型 ===================="
echo "模型: $MODEL"
echo "目标: $DEST"
echo "镜像: $HF_ENDPOINT"
echo ""

mkdir -p "$(dirname "$DEST")"
python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL', local_dir='$DEST')
print('下载完成:', p)
"

echo ""
echo "==================== 完成 ===================="
echo "运行 pipeline 使用本地模型:"
echo "  AUTODL_MODEL=$DEST python pipeline_autodl.py"
