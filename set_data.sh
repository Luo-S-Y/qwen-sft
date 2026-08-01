#!/bin/bash
# AutoDL 数据/模型存储目录设置
# 用法: bash set_data.sh
# 功能: 将 models (基座模型 + Full 训练产物) 链接到数据盘 /root/autodl-tmp,
#       避免 ~8GB+ 的大文件占满系统盘
set -e

echo "==================== 数据/模型存储设置 ===================="

if [ ! -d /root/autodl-tmp ]; then
    echo "未检测到数据盘 /root/autodl-tmp, 目录保持默认位置, 无需设置"
    exit 0
fi

echo "[1/2] models 目录链接到数据盘..."
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

echo "[2/2] 检查 data 目录..."
if [ -d data ]; then
    echo "  data 当前体积: $(du -sh data 2>/dev/null | cut -f1) (很小, 保持系统盘即可)"
else
    echo "  未找到 data 目录 (clone 仓库后应包含 data/lora_short)"
fi

echo ""
echo "==================== 完成 ===================="
echo "之后运行: bash download_model.sh && python pipeline_autodl.py"
