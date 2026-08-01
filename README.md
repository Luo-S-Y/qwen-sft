# qwen-sft

ReasonLite SFT 复现：Qwen3-0.6B 上对比 LoRA 与 Full SFT 的数学推理效果。

## 数据
- 来源：`amd/ReasonLite-Dataset` (medium, short CoT)
- 已划分：4000 训练 + 50 验证（token<500 短样本，`data/lora_short/`）

## 实验矩阵（10 组）
| 方法 | 可训层 | 步数 |
|---|---|---|
| baseline | — | — |
| LoRA (r=16) | 后8层 | 100 / 500 / 1000 |
| Full SFT | 后8层 (layer20-27) | 100 / 500 / 1000 |
| Full SFT | 后16层 (layer12-27) | 100 / 500 / 1000 |

## 运行
- **Mac (Apple Silicon, MLX)**: `python pipeline_all.py`（训练 MLX，评估 llama.cpp GGUF）
- **AutoDL (CUDA)**: `python pipeline_autodl.py`（训练 transformers+peft，评估 vLLM）
  ```bash
  pip install torch transformers peft datasets vllm sympy sentencepiece
  python pipeline_autodl.py
  ```
- 模型 `Qwen/Qwen3-0.6B` 自动从 HF 下载（`HF_ENDPOINT=https://hf-mirror.com`）
- 已含断点续跑（产物存在即跳过）

## 输出
- 每版本评测日志：`result/gguf_eval/{version}.log` (Mac) / `result/autodl/{version}.log` (AutoDL)
- 对比报告：`comparison.md`
