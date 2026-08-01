#!/bin/bash
# AutoDL 一键环境安装脚本
# 用法: bash setup.sh   (或 ./setup.sh)
# 功能: 配置清华 pip 镜像 + HF 镜像 + 安装全部依赖 + 验证
set -e

PY=${PY:-python}   # 可用 PY=/path/to/python bash setup.sh 覆盖

echo "==================== 环境安装 ===================="
echo "Python: $($PY --version)"

echo ""
echo "[1/4] 配置清华 pip 镜像 (~/.pip/pip.conf)..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
echo "已写入 ~/.pip/pip.conf"

echo ""
echo "[2/4] 配置 HF 镜像 (写入 ~/.bashrc)..."
grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
export HF_ENDPOINT=https://hf-mirror.com
echo "已设置 HF_ENDPOINT=https://hf-mirror.com"

echo ""
echo "[3/4] 安装依赖 (torch/transformers/peft/datasets/vllm/sympy)..."
$PY -m pip install torch transformers peft datasets vllm sympy sentencepiece

echo ""
echo "[4/4] 验证安装..."
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
echo "接下来运行:  python pipeline_autodl.py"
echo "(模型 Qwen/Qwen3-0.6B 与数据集将自动从 HF 镜像下载)"
