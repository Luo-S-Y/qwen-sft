#!/bin/bash
# AutoDL 4090: vLLM 评测加速环境安装 (独立 conda 环境, 不碰训练环境 torch 2.5.1)
# 用法: bash setup_vllm.sh
# 功能: 建 vllm conda 环境 + 清华镜像 + 安装 vllm(>=0.8.5, 支持 Qwen3) + 训练/评测依赖
# 之后: conda activate vllm && bash run.sh  (训练 + vLLM 批量评估)
set -e

echo "==================== vLLM 环境安装 (4090 评测加速) ===================="

echo ""
echo "[1/4] 配置清华 pip 镜像..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
echo "已写入 ~/.pip/pip.conf"

echo ""
echo "[2/4] 创建 conda 环境 vllm (python 3.12)..."
conda create -n vllm python=3.12 -y || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
echo "已激活环境: $(python --version)"

echo ""
echo "[3/4] 安装 vllm + 训练/评测依赖..."
# Qwen3 需要 vllm>=0.8.5; pip 自动解析匹配的 torch (2.7 需驱动>=570, 4090 新实例满足)
pip install vllm==0.8.5 transformers peft datasets sympy sentencepiece

echo ""
echo "[4/4] 验证安装..."
python -c "
import torch, transformers, vllm, peft, datasets, sympy
print(f'torch        {torch.__version__}  CUDA: {torch.cuda.is_available()}')
print(f'transformers {transformers.__version__}')
print(f'vllm         {vllm.__version__}')
print(f'peft         {peft.__version__}')
print(f'datasets     {datasets.__version__}')
print('全部依赖 OK')
"

echo ""
echo "==================== 完成 ===================="
echo "接下来运行:"
echo "  conda activate vllm"
echo "  bash run.sh                  # 训练 + vLLM 评估一键串联"
echo ""
echo "若 pip 报 torch 版本冲突, 回退: pip install torch==2.6.0 vllm==0.8.5"
