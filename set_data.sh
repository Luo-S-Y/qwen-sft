#!/bin/bash
# AutoDL 数据/模型存储设置 + 下载
# 用法: bash set_data.sh
# 功能:
#   1. models 链接到数据盘 /root/autodl-tmp (避免大文件占系统盘)
#   2. 下载模型 Qwen/Qwen3-0.6B (已下载则跳过)
#   3. 检查数据集 data/lora_short
set -e

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "==================== 数据/模型存储设置 ===================="

# ---------- 1. models 目录链接到数据盘 ----------
if [ -d /root/autodl-tmp ]; then
    echo "[1/3] models 目录链接到数据盘..."
    mkdir -p /root/autodl-tmp/models
    if [ -L models ]; then
        echo "  models 已是符号链接 -> $(readlink models), 跳过"
    elif [ -d models ] && [ -n "$(ls -A models 2>/dev/null)" ]; then
        echo "  迁移现有 models/ 到数据盘..."
        mv models/* /root/autodl-tmp/models/
        rm -rf models
        ln -s /root/autodl-tmp/models models
        echo "  已迁移并链接 models -> /root/autodl-tmp/models"
    else
        rm -rf models 2>/dev/null
        ln -s /root/autodl-tmp/models models
        echo "  已链接 models -> /root/autodl-tmp/models"
    fi
else
    echo "[1/3] 未检测到数据盘 /root/autodl-tmp, models 目录保持默认位置"
fi

# ---------- 2. 下载模型 (幂等: 已下载则跳过) ----------
echo ""
echo "[2/3] 检查模型 models/Qwen3-0.6B..."
if [ -f models/Qwen3-0.6B/config.json ]; then
    echo "  模型已存在 (models/Qwen3-0.6B), 跳过下载"
else
    echo "  下载 Qwen/Qwen3-0.6B -> models/Qwen3-0.6B ..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/Qwen3-0.6B')
print('  模型下载完成')
"
fi

# ---------- 3. 检查数据集 ----------
echo ""
echo "[3/3] 检查数据集 data/lora_short..."
if [ -f data/lora_short/train.jsonl ] && [ -f data/lora_short/valid.jsonl ]; then
    echo "  数据集已存在 (4000 训练 + 50 验证)"
else
    echo "  数据集缺失! 请确认已 clone 仓库 (数据在仓库 data/lora_short/ 中)"
    echo "  或运行 pipeline 时自动从 HF 下载划分 (amd/ReasonLite-Dataset)"
fi

echo ""
echo "==================== 完成 ===================="
echo "之后运行: python pipeline_autodl.py"
