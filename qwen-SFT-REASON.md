# Qwen3-0.6B SFT + ReasonLite 复现实验记录

## 实验目标

复现 ReasonLite 蒸馏方法，在小规模上跑通 Qwen3-0.6B SFT 训练 + AIME24 评测 pipeline。

### ReasonLite 背景 (AMD, 2025.12)

| 阶段 | 模型 | 数据 | AIME24 |
|------|------|------|--------|
| 基座 | Qwen3-0.6B | - | 11.0 |
| Stage 1 (Turbo) | Qwen3-0.6B + Short CoT SFT | 4.3M | 57.1 |
| Stage 2 | ReasonLite-0.6B + Long CoT SFT | 1.8M | 75.2 |

---

## 环境配置

- **硬件**: Apple M3 (24GB), macOS
- **框架**: MLX (Apple Silicon GPU)
- **Python**: 3.9.6 (venv)
- **依赖**: `mlx-lm`, `transformers`, `datasets`, `sympy`

### 镜像配置

```bash
# HF 镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com
# pip 镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...
```

### 模型下载

Qwen3-0.6B 通过 `mlx_lm.convert` 下载并转为 MLX 格式：

```bash
# BF16 版本 (1.4GB)
mlx_lm.convert --hf-path Qwen/Qwen3-0.6B --mlx-path models/Qwen3-0.6B

# 4-bit 量化版本 (320MB)
mlx_lm.convert --hf-path models/Qwen3-0.6B --mlx-path models/Qwen3-0.6B-4bit -q --q-bits 4
```

**注意**: `mlx_lm.convert -q` 到输出目录才有效，in-place 转换不生效。

---

## 数据准备

### 数据集

- **来源**: `amd/ReasonLite-Dataset` (medium split, 4.3M 条 short CoT)
- **格式**: `prompt` (数学题) + `answer` (模型生成的解答)
- **提取**: 使用 streaming 模式取前 1100 条，shuffle 后分 800/100/100

### 处理脚本

`train/prepare_data.py` — 下载并格式化为 prompt+completion JSONL。

### MLX LoRA 数据格式

使用 `text` 字段（TextDataset），预拼好完整 chat template：

```
<|im_start|>system
You are Qwen, a helpful AI assistant.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

---

## 训练脚本

- `train/train_lora.py` — 封装 `mlx_lm.lora.run()` 的训练入口
- `train/benchmark.py` — 速度对比测试

## 评测脚本

- `eval/eval_aime24.py` — 基础 AIME24 评测（支持 LoRA adapter 加载）
- `eval/eval_stream.py` — 单进程流式评测，实时显示 token
- `eval/eval_parallel.py` — 多 worker 并行评测，独立日志文件

---

## 实验记录

### Exp 1: Q-LoRA 100 iters 训练验证

**目的**: 验证训练 pipeline 是否跑通

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-0.6B (BF16, 1.4GB) |
| 方法 | LoRA (rank=8, alpha=16, 8 layers) |
| 数据 | 800 条 (ReasonLite medium) |
| 训练 | 100 iters, batch=4, accum=2 |
| 耗时 | ~20 min |
| 峰值内存 | 11.7 GB |

**Loss 曲线**:

```
Iter  1: Val loss 1.410
Iter 10: Train loss 1.430
Iter 20: Train loss 1.234
Iter 60: Train loss 1.099
Iter 90: Train loss 0.957
Iter100: Val loss 0.960 / Train loss 1.147
```

**结论**: 训练正常收敛，但 100 iters 远不足以学到有效推理。

### Exp 2: AIME24 评测（训练后）

**目的**: 验证评测 pipeline

**配置**: greedy (temp=0), 1 sample, max_tokens=2048

| 问题 | 耗时 | Token数 | 预期 | 抽取得 | 结果 |
|:----:|:----:|:-------:|:----:|:------:|:----:|
| 1 | 46.2s | 649 | 204 | `for` | ✗ |
| 2 | 44.2s | 465 | 113 | `\[` | ✗ |
| 3 | 45.2s | 886 | 371 | `the` | ✗ |
| 4 | 45.2s | 552 | 385 | `(2\pi` | ✗ |
| 5 | 42.6s | 363 | 110 | `\pmod` | ✗ |

**结论**: 答案均错误（训练量不够），但 pipeline 完整跑通。

### Exp 3: 并行评测（2 workers）

**目的**: 2 个 worker 并行处理 AIME30 题

- Worker 1: 题 1-15 → `worker_1.log`
- Worker 2: 题 16-30 → `worker_2.log`
- 各写独立日志文件，终端交错显示 token 流

**日志格式**:
```
######第1个问题######
问题原文：Every morning Aya goes for a ...
---以上是问题，以下是模型输出---
<think>We need to interpret...
[streaming tokens...]
##第1个问题的答案##：2
```

**结果（2 题）**:

| Worker | 耗时 | Token | 预期→预测 | 结果 |
|:------:|:----:|:-----:|:----------:|:----:|
| W1 | 69s | 1282 | 204→2 | ✗ |
| W2 | 69s | 1101 | 113→D | ✗ |

### Exp 4a: 16-bit vs 4-bit LoRA 速度对比

**配置**: batch=2, accum=2, 10 iters, 800 条数据

| 方法 | It/sec | 10it耗时 | s/it | 训练参数量 | 峰值内存 |
|:----:|:------:|:--------:|:----:|:---------:|:--------:|
| 16-bit LoRA | **0.201** | **~50s** | 5.0s | 1.44M | 11.7 GB |
| **4-bit LoRA** | 0.130 | ~77s | 7.7s | 1.44M | 10.6 GB |

**结论**: 4-bit 反而慢 35%！量化层每次前向都要解量化，小模型上开销占比大。

### Exp 4b: 16-bit LoRA vs Full FT 速度对比

| 方法 | It/sec | 10it耗时 | s/it | 训练参数量 |
|:----:|:------:|:--------:|:----:|:---------:|
| 16-bit LoRA | **0.201** | ~50s | 5.0s | 1.44M (0.24%) |
| 16-bit Full FT (21%) | 0.141 | ~71s | 7.1s | 125.8M (21%) |

**注意**: mlx-lm 的 `full` 模式只解冻最后 `num_layers` 层，非全参。

### Exp 4c: batch × grad_accum 组合对比

**固定每步 4 条样本**:

| batch | accum | It/sec | 10it耗时 | 峰值内存 |
|:----:|:-----:|:------:|:--------:|:--------:|
| **1** | **4** | **0.485** | **~21s** | **6.5 GB** |
| 2 | 2 | 0.186 | ~54s | 11.7 GB |
| 4 | 1 | ❌ OOM | ❌ | ❌ |

**关键发现**: `batch=1, accum=4` 比 `batch=2, accum=2` **快 2.6 倍**，内存省一半。

原因: Apple M3 统一内存架构下，GPU 内存带宽是瓶颈。小 batch 减少每次加载数据量，GPU 利用率更高。

---

## 模型量化对比

| 格式 | 文件大小 | 加权类型 | 是否量化 |
|:----:|:--------:|:--------:|:--------:|
| BF16 (16-bit) | 1.4 GB | `Linear` | ❌ |
| 4-bit | **320 MB** | **`QuantizedLinear`** | ✅ (bits=4) |

### 模型结构

- 架构: Qwen3ForCausalLM
- 参数: ~600M
- 层数: 28
- 隐藏维度: 1024
- 词表: 151,936

---

## 关键教训

1. **Q-LoRA 对小模型（0.6B）是负优化**: 4-bit 比 BF16 更慢，内存节省有限 (~1GB)。量化适合大模型（7B+）。
2. **batch=1 + 梯度累积 最优**: Apple M3 上小 batch 最快最省内存。
3. **mlx-lm 的 "full" 不是全参**: 只解冻最后 `num_layers` 层，注意配置。
4. **模型重复输出**: 100 iters 训练不够，模型仍在重复生成相同内容。需要更多数据。
5. **流式并行评测**: 2 worker 并行 + 独立日志文件 + 实时 token 显示的 pipeline 已跑通。

---

## 脚本结构

```
0726-deepseek-r1/
├── models/
│   ├── Qwen3-0.6B/          # BF16 模型 (1.4 GB)
│   └── Qwen3-0.6B-4bit/     # 4-bit 量化模型 (320 MB)
├── data/
│   └── lora_subset/          # ReasonLite 子集 800+100+100
├── train/
│   ├── prepare_data.py       # 数据下载与格式化
│   ├── train_lora.py         # LoRA 训练入口
│   └── benchmark.py          # 速度对比
├── eval/
│   ├── eval_aime24.py        # 基础 AIME24 评测
│   ├── eval_stream.py        # 单进程流式评测
│   └── eval_parallel.py      # 多 worker 并行评测
├── adapters/                 # 训练好的 adapters
└── result/
    └── stream_logs/          # 评测日志
```
