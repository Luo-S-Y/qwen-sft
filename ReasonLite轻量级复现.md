# ReasonLite 轻量级复现指南

> 在小规模上复现 AMD ReasonLite 的 SFT 蒸馏思路：`Qwen3-0.6B` 上对比 **LoRA 与 Full SFT** 的数学推理效果。
> 覆盖 **AutoDL (NVIDIA CUDA)** 主路径 + **Mac (MLX)** 备选路径。

## 0. 背景：ReasonLite 论文怎么做的

ReasonLite（AMD, 2025.12）是 0.6B 参数的轻量数学推理模型，AIME 2024 达 **75.2%**（与 Qwen3-8B 相当，参数少约 13x），核心方法是**课程蒸馏 (Curriculum Distillation)**：

1. **数据收集**：从 Polaris、NVIDIA OpenMathReasoning 等收集 343K 道数学竞赛题
2. **教师蒸馏**：用强教师 **GPT-OSS-120B** 以 medium / high-depth 两种推理深度为每题生成解答，产出约 **9.1M** 个 AI 解
3. **伪标签筛选**：教师多次作答 + **多数投票 (majority-voting)** 生成可靠伪标签，与原标签一致性过滤，最终 **6.1M** 高质量题-解对
4. **两阶段 SFT 课程**：
   - Stage 1：**Short CoT（4.3M 条）** → 高效 Turbo 版（AIME24 **57.1**）
   - Stage 2：**Long CoT（1.8M 条）** → 最终版（AIME24 **75.2**）
   - 基座 Qwen3-0.6B 仅 **11.0**
5. **开源**：权重 `amd/ReasonLite-0.6B` + 数据 `amd/ReasonLite-Dataset` + 代码全公开

> 本项目轻量复现：Qwen3-0.6B + 4000 条 token<500 短样本（来自 ReasonLite-Dataset medium 子集），对比 LoRA/Full SFT 各 100/500/1000 步，验证集 50 题。

## 1. 实验设计

| 项 | 值 |
|---|---|
| 基座模型 | Qwen/Qwen3-0.6B（28 层，596M 参数） |
| 数据 | `amd/ReasonLite-Dataset` (medium, short CoT)，**token<500 短样本** |
| 划分 | **4000 训练 + 50 验证**（`data/lora_short/train.jsonl` / `valid.jsonl`） |
| 精度 | bf16（Ampere+ 原生；旧卡 Turing 需 fp16） |

### 10 组实验矩阵

| # | 方法 | 可训层 | 步数 |
|---|---|---|---|
| 1 | baseline | —（原始模型不训练） | — |
| 2-4 | LoRA r=16 | 后8层 (layers 20-27) | 100 / 500 / 1000 |
| 5-7 | Full SFT | 后8层 (layers 20-27) | 100 / 500 / 1000 |
| 8-10 | Full SFT | 后16层 (layers 12-27) | 100 / 500 / 1000 |

## 2. 快速复现（AutoDL 4090 推荐）

```bash
# ① 拉代码
git clone git@github.com:Luo-S-Y/qwen-sft.git && cd qwen-sft

# ② 装环境（清华 pip + HF 镜像 + torch/transformers/peft/datasets/sympy）
bash setup.sh

# ③ 数据盘链接 + 预创建保存目录 + 下载模型
bash set_data.sh

# ④ 一键串联：训练 9 组 → 评测 10 版本 → 对比报告
#    （自动检测：环境有 vllm 用 vLLM 批量评测，无则 transformers generate）
bash run.sh
```

后台运行并实时看进度：

```bash
nohup bash run.sh > logs/run.log 2>&1 &
tail -f logs/run.log
```

也可以分步/单独跑：

```bash
python pipeline_autodl.py --train-only   # 只训练
python pipeline_autodl.py --eval-only    # 只评测（已训练完时）
FORCE=1 python pipeline_autodl.py --eval-only   # 强制重评
```

## 3. 脚本说明

| 脚本 | 作用 |
|---|---|
| `setup.sh` | 清华 pip 镜像 + HF 镜像 + 训练依赖安装 + 验证 |
| `set_data.sh` | models/adapters 链接数据盘、预创建 9+6 个实验子目录、幂等下载模型、检查数据集 |
| `setup_vllm.sh` | （可选）独立 conda 环境装 vllm 0.8.5 评测加速 |
| `pipeline_autodl.py` | 主流程：数据 → 训练 9 组 → 评测 10 版本 → comparison.md；`--train-only/--eval-only` |
| `run.sh` | 训练 + 评测串联，日志落 `logs/`（用当前环境，不装 conda） |
| `download_model.sh` | 单独下载模型 |

## 4. 关键实现细节

### 训练（transformers + peft Trainer）

- **LoRA**：`r=16, alpha=16, target=[q,k,v,o]_proj`，`layers_to_transform=[20..27]`（只注入后8层）
- **Full**：冻结 embed + lm_head + 前 N 层，只解冻后 8/16 层
- **梯度检查点默认关闭**：与"冻结前N层"组合会触发 checkpoint 重算报错
  （`element 0 of tensors does not require grad`）；OOM 时用 `AUTODL_CKPT=1` 开启
- batch/累积可配：`AUTODL_BATCH=4 AUTODL_ACCUM=1`（环境变量覆盖）
- 断点续跑：产物存在即跳过（`adapters/`、`models/full/`）

### 评测

- 后端自动选择：有 `vllm`（≥0.8.5）→ 批量评测；否则 → transformers generate
- **必须禁用 Qwen3 thinking**：走 vllm chat 接口 + `chat_template_kwargs={"enable_thinking": false}`，
  否则模型输出全在 `<think>` 里、`\boxed` 答案不生成，准确率虚低
- 答案提取：嵌套 `\boxed` 正则 + `sympy` 语义匹配（`\dfrac` vs `\frac`、`0.5` vs `\dfrac12` 等）
- 每版本日志独立：`result/autodl/{version}.log` + `.json`，汇总 `comparison.md`

## 5. 环境版本（AutoDL）

| 组件 | 版本 | 说明 |
|---|---|---|
| torch | 2.5.1+cu124 | 训练；匹配 CUDA 12.4 驱动 |
| transformers | 4.51.3 | 支持 Qwen3 |
| vllm（可选） | **0.8.5 + torch 2.6.0** | 评测加速；**vllm 与 torch 强绑定**，版本必须配套 |

> vllm 版本- torch 对应：0.6.6↔2.5.x（不支持 Qwen3）、0.8.5↔2.6.0、0.9.x↔2.7.0。
> 切换 vllm 版本时务必 `pip uninstall vllm torch-c-dlpack-ext` 清残留，否则报 undefined symbol。

## 6. 当前结果（50 题验证集）

| 版本 | 正确率 | 备注 |
|---|---|---|
| lora_100 / lora_1000 | 52% | 并列最高 |
| full8_500 / full8_1000 | 50% | |
| full16_500 / full16_1000 | 48% | |
| lora_500 | 40% | 反常低于 lora_100 |
| baseline | 14% | **该轮未禁用 thinking，虚低**；禁用后预计 40%+ |

> 注意：上表为 thinking 未禁用时的 vLLM 结果，修复后需 `FORCE=1` 重评取最终数字。

## 7. 踩坑速查

| 现象 | 原因 / 解决 |
|---|---|
| `torch.OutOfMemoryError` | Full 训练 OOM → bf16 + `AUTODL_CKPT=1` 或降 batch |
| `element 0 of tensors does not require grad` | 梯度检查点 + 冻结层冲突 → 默认关，OOM 再开 |
| `FileNotFoundError: adapters/lora_100` | 数据盘符号链接空目录 → `set_data.sh` 已预创建 |
| baseline 输出全 `<think>`、acc 极低 | Qwen3 thinking 未禁用 → chat 接口 + `enable_thinking:false` |
| `Missing required argument 'lora_int_id'` | vllm 0.8.x LoRARequest 需显式传 `lora_int_id` |
| `undefined symbol ... dlpack` | vllm/torch 版本切换残留 → 清 `torch-c-dlpack-ext` 重装 |
| `torch.compile takes 55s` | vllm 启动编译（一次性）→ 已默认 `VLLM_TORCH_COMPILE_LEVEL=0` |
| 驱动太旧 | AutoDL 旧实例报 driver 12040 → 降 torch 2.5.1（cu124）；4090 新实例随意 |

## 8. Mac (Apple Silicon, MLX) 备选路径

```bash
# 数据准备（token<500 短样本，4000+50）
python train/prepare_data.py

# 训练 10 组矩阵（MLX LoRA）
python pipeline_all.py

# GGUF 评测（llama.cpp + llama-server，需禁用 thinking）
python eval/eval_gguf.py --dataset data/lora_short/valid.jsonl
```

- Mac 版结论（小模型经验）：主题 ≤5 效果更好、200 干净样本 > 1000 脏样本、val loss 平台即停训
- 完整 Mac 实验记录见 `qwen-SFT-REASON.md`
