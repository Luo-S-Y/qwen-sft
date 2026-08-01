#!/bin/bash
# AutoDL 一键环境安装脚本
# 用法: bash setup.sh   (或 ./setup.sh)
# 功能: 配置清华 pip 镜像 + HF 镜像 + 模型目录链接到数据盘 + 安装依赖 + 验证
set -e

PY=${PY:-python}   # 可用 PY=/path/to/python bash setup.sh 覆盖

echo "==================== 环境安装 ===================="
echo "Python: $($PY --version)"

echo ""
echo "[1/5] 配置清华 pip 镜像 (~/.pip/pip.conf)..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
echo "已写入 ~/.pip/pip.conf"

echo ""
echo "[2/5] 配置 HF 镜像 (写入 ~/.bashrc)..."
grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
export HF_ENDPOINT=https://hf-mirror.com
echo "已设置 HF_ENDPOINT=https://hf-mirror.com"

echo ""
echo "[3/5] 模型目录链接到数据盘 (存在 /root/autodl-tmp 时)..."
if [ -d /root/autodl-tmp ]; then
    mkdir -p /root/autodl-tmp/models
    if [ -L models ]; then
        echo "models 已是符号链接 -> $(readlink models), 跳过"
    elif [ -d models ] && [ -n "$(ls -A models 2>/dev/null)" ]; then
        echo "迁移现有 models/ 到数据盘..."
        mv models/* /root/autodl-tmp/models/
        rm -rf models
        ln -s /root/autodl-tmp/models models
        echo "已迁移并链接 models -> /root/autodl-tmp/models"
    else
        rm -rf models 2>/dev/null
        ln -s /root/autodl-tmp/models models
        echo "已链接 models -> /root/autodl-tmp/models"
    fi
else
    echo "未检测到数据盘 /root/autodl-tmp, 模型目录保持默认位置"
fi

echo ""
echo "[4/5] 安装依赖 (torch/transformers/peft/datasets/vllm/sympy)..."
$PY -m pip install torch transformers peft datasets vllm sympy sentencepiece

echo ""
echo "[5/5] 验证安装..."
$PY -c "
import torch, transformers, peft, datasets, sympy
import vllm
print(f'torch        {torch.__version__}')
print(f'transformers {transformers.__version__}')
print(f'peft         {peft.__version__}')
print(f'datasets     {datasets.__version__}')
print(f'vllm         {vllm.__version__}')
print('全部依赖 OK')
"

echo ""
echo "==================== 完成 ===================="
echo "接下来运行:  bash download_model.sh  # 下载模型(自动存数据盘)"
echo "            python pipeline_autodl.py"
echo "(数据集已含在仓库 data/lora_short/, 模型从 HF 镜像下载到 models/)"
