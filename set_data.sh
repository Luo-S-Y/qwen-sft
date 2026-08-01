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

# ---------- 1. models + adapters 目录链接到数据盘 ----------
if [ -d /root/autodl-tmp ]; then
    echo "[1/4] 链接 models / adapters 到数据盘..."
    for dir in models adapters; do
        if [ -L "$dir" ]; then
            echo "  $dir 已是符号链接 -> $(readlink "$dir"), 跳过"
        elif [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
            mkdir -p "/root/autodl-tmp/$dir"
            echo "  迁移现有 $dir/ 到数据盘..."
            mv "$dir"/* "/root/autodl-tmp/$dir/"
            rm -rf "$dir"
            ln -s "/root/autodl-tmp/$dir" "$dir"
            echo "  已迁移并链接 $dir -> /root/autodl-tmp/$dir"
        else
            mkdir -p "/root/autodl-tmp/$dir"
            rm -rf "$dir" 2>/dev/null
            ln -s "/root/autodl-tmp/$dir" "$dir"
            echo "  已链接 $dir -> /root/autodl-tmp/$dir"
        fi
    done
else
    echo "[1/4] 未检测到数据盘 /root/autodl-tmp, 目录保持默认位置"
fi

# 预创建训练保存目录 (新实例数据盘为空, 避免训练时 FileNotFoundError)
echo "  预创建训练保存目录..."
for d in lora_100 lora_500 lora_1000 full8_100 full8_500 full8_1000 full16_100 full16_500 full16_1000; do
    mkdir -p "adapters/$d"
done
mkdir -p "models/full/full8_100" "models/full/full8_500" "models/full/full8_1000" \
         "models/full/full16_100" "models/full/full16_500" "models/full/full16_1000"
echo "  adapters/: $(ls adapters/ | tr '\n' ' ')"
echo "  models/full/: $(ls models/full/ 2>/dev/null | tr '\n' ' ')"

# ---------- 2. 下载模型 (幂等: 已下载则跳过) ----------
echo ""
echo "[2/4] 检查模型 models/Qwen3-0.6B..."
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
echo "[3/4] 检查数据集 data/lora_short..."
if [ -f data/lora_short/train.jsonl ] && [ -f data/lora_short/valid.jsonl ]; then
    echo "  数据集已存在 (4000 训练 + 50 验证)"
else
    echo "  数据集缺失! 请确认已 clone 仓库 (数据在仓库 data/lora_short/ 中)"
    echo "  或运行 pipeline 时自动从 HF 下载划分 (amd/ReasonLite-Dataset)"
fi

echo ""
echo "[4/4] 存储位置总览..."
ls -l models adapters 2>/dev/null | grep -E "^l|^d" || true
echo "  models  -> $(readlink models 2>/dev/null || echo '系统盘')"
echo "  adapters-> $(readlink adapters 2>/dev/null || echo '系统盘')"

echo ""
echo "==================== 完成 ===================="
echo "之后运行: python pipeline_autodl.py"
