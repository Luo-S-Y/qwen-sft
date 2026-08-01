#!/usr/bin/env python3
"""
ReasonLite SFT 复现 pipeline — AutoDL (NVIDIA CUDA) 版
框架: transformers + peft 训练 | vLLM 评估
数据: ReasonLite 短样本(token<500): 4000 训练 + 50 验证
实验: 10 组矩阵 (baseline + LoRA r16后8层 100/500/1000 + Full后8层 100/500/1000 + Full后16层 100/500/1000)

安装依赖:
  # 训练环境: pip install torch transformers peft datasets sympy sentencepiece
  # 评测加速(可选, 4090 推荐): 装好 vllm>=0.8.5 后自动用 vLLM 批量评测
  #   conda create -n vllm python=3.12 -y && conda activate vllm
  #   pip install vllm==0.8.5 transformers peft datasets sympy sentencepiece
  #   (vllm 环境可同时训练+评测, 一次性跑通全流程)

运行:
  # 模型/数据从 HF 下载(国内走 hf-mirror, 脚本已默认设置 HF_ENDPOINT)
  # 也可设置 AUTODL_MODEL=/path/to/Qwen3-0.6B 使用本地模型
  python pipeline_autodl.py

评测后端自动选择: 环境有 vllm -> vLLM 批量评测(5-10x 加速); 否则回退 transformers generate
环境变量:
  AUTODL_BATCH / AUTODL_ACCUM  训练 batch / 梯度累积 (默认 4/1)
  AUTODL_CKPT=1                开启梯度检查点(Full OOM 时用)
  MAX_NEW=512                  vLLM 生成最大 token 数
  FORCE=1                      强制重评(覆盖已有评测结果)

进度: 全程 stdout 实时打印(时间戳 + tqdm), 每版本评测日志独立存 result/autodl/{version}.log/.json
"""
import os, sys, re, json, time, argparse
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "lora_short")
ADAPTER_DIR = os.path.join(BASE, "adapters")
FULL_DIR = os.path.join(BASE, "models", "full")
EVAL_DIR = os.path.join(BASE, "result", "autodl")
# 训练 batch 配置 (每步样本数 = BATCH_SIZE × GRAD_ACCUM)
# 默认 batch=4 + accum=1 (每步 4 样本)
# 可环境变量覆盖: AUTODL_BATCH=16 AUTODL_ACCUM=1
BATCH_SIZE = int(os.environ.get("AUTODL_BATCH", "4"))
GRAD_ACCUM = int(os.environ.get("AUTODL_ACCUM", "1"))
# 优先用本地已下载的模型 (set_data.sh 负责下载), 否则回退 HF id
LOCAL_MODEL = os.path.join(BASE, "models", "Qwen3-0.6B")
MODEL = os.environ.get("AUTODL_MODEL") or (
    LOCAL_MODEL if os.path.isfile(os.path.join(LOCAL_MODEL, "config.json")) else "Qwen/Qwen3-0.6B"
)
CHAT_TEMPLATE = "<|im_start|>system\nYou are Qwen, a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"
EVAL_SYSTEM = "You are a helpful math assistant. Solve the problem and provide the final answer directly."
EVAL_PROMPT = "<|im_start|>system\n" + EVAL_SYSTEM + "<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"

# (method, 可训层, iters, 保存名, LoRA?)
MATRIX = [
    ("baseline", "—", None, "baseline", False),
    ("LoRA", "r16 后8层", 100, "lora_100", True),
    ("LoRA", "r16 后8层", 500, "lora_500", True),
    ("LoRA", "r16 后8层", 1000, "lora_1000", True),
    ("Full", "后8层", 100, "full8_100", False),
    ("Full", "后8层", 500, "full8_500", False),
    ("Full", "后8层", 1000, "full8_1000", False),
    ("Full", "后16层", 100, "full16_100", False),
    ("Full", "后16层", 500, "full16_500", False),
    ("Full", "后16层", 1000, "full16_1000", False),
]
QWEN_LAYERS = 28

# 评测后端: 环境有 vllm 则用 vLLM 批量评测(4090 加速 5-10x), 否则 transformers generate
try:
    import vllm  # noqa: F401
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
MAX_NEW = int(os.environ.get("MAX_NEW", "512"))
FORCE = os.environ.get("FORCE", "0") == "1"


def log(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_env():
    missing = []
    for m in ("torch", "transformers", "datasets", "peft", "sympy"):
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        print(f"缺少依赖: {', '.join(missing)}", flush=True)
        print("请先安装: pip install torch transformers peft datasets sympy sentencepiece", flush=True)
        sys.exit(1)


def extract_answer(text):
    boxed = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    if boxed: return boxed[-1].strip()
    lines = text.strip().split('\n')
    for line in reversed(lines):
        dollars = re.findall(r'\$([^$]+)\$', line)
        if dollars: return dollars[-1].strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums: return nums[-1]
    return ""


def answers_match(pred, expected, tolerance=1e-2):
    p, e = pred.strip(), expected.strip()
    if not p or not e:
        return False
    if p == e:
        return True
    try:
        return abs(float(p) - float(e)) < tolerance
    except ValueError:
        pass
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        diff = simplify(parse_latex(p) - parse_latex(e))
        if diff == 0:
            return True
        try:
            return abs(float(diff.evalf())) < tolerance
        except Exception:
            return False
    except Exception:
        return False


# ==================== 数据准备 ====================
def prepare_data(num_train=4000, num_valid=50, max_tokens=500, max_scan=300000, seed=42):
    import random
    from transformers import AutoTokenizer
    from datasets import load_dataset
    os.makedirs(DATA_DIR, exist_ok=True)

    # 本地已有划分好的数据则直接使用
    train_p, valid_p = os.path.join(DATA_DIR, "train.jsonl"), os.path.join(DATA_DIR, "valid.jsonl")
    if os.path.exists(train_p) and os.path.exists(valid_p):
        n_tr = sum(1 for _ in open(train_p))
        n_va = sum(1 for _ in open(valid_p))
        log(f"本地数据已存在: {n_tr} 训练 + {n_va} 验证 (跳过下载/划分)")
        return

    random.seed(seed)
    t0 = time.time()
    log(f"Step 0: 筛选 token<{max_tokens} 短样本 ({num_train} 训练 + {num_valid} 验证)")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ds = load_dataset("amd/ReasonLite-Dataset", split="medium", streaming=True, trust_remote_code=True)

    BATCH, target = 5000, num_train + num_valid
    buf, short = [], []
    i = 0
    for i, row in enumerate(ds):
        if i >= max_scan:
            break
        buf.append(CHAT_TEMPLATE.format(prompt=row["prompt"], completion=row["answer"]))
        if len(buf) >= BATCH:
            lens = [len(e) for e in tok(buf, add_special_tokens=True, padding=False)["input_ids"]]
            short.extend(t for t, l in zip(buf, lens) if l <= max_tokens)
            buf = []
            print(f"  scanned {i+1}, kept {len(short)} ({time.time()-t0:.0f}s)", flush=True)
            if len(short) >= target:
                break
    if buf:
        lens = [len(e) for e in tok(buf, add_special_tokens=True, padding=False)["input_ids"]]
        short.extend(t for t, l in zip(buf, lens) if l <= max_tokens)
    print(f"  scanned {i+1}, kept {len(short)} short samples ({time.time()-t0:.0f}s)", flush=True)
    if len(short) < target:
        print(f"WARNING: 仅 {len(short)} 条短样本, 不足 {target}, 需增大 max_scan!", flush=True)

    random.shuffle(short)
    for split_name, n in [("train", num_train), ("valid", num_valid)]:
        rows = short[:n]
        del short[:n]
        out = os.path.join(DATA_DIR, f"{split_name}.jsonl")
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps({"text": r}, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(rows)} -> {out}", flush=True)
    log(f"数据就绪 -> {DATA_DIR}")


# ==================== 训练 ====================
def train(name, method, iters, is_lora, num_layers=None):
    """num_layers: Full 时训练的后 N 层; LoRA 时注入 LoRA 的后 N 层"""
    import torch
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from datasets import load_dataset

    save_dir = os.path.join(ADAPTER_DIR, name) if is_lora else os.path.join(FULL_DIR, name)
    if os.path.exists(os.path.join(save_dir, "adapter_config.json" if is_lora else "config.json")):
        log(f"{name} 已存在, 跳过训练")
        return

    log(f"Step: {method} {iters} iters -> {save_dir}")
    # 统一 bf16 (模型 bf16 + bf16 AMP, 无 scaler)
    use_bf16 = True
    model_dtype = torch.bfloat16
    print(f"混合精度: bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=model_dtype, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 冻结 embed + lm_head
    for p in model.model.embed_tokens.parameters():
        p.requires_grad_(False)
    for p in model.lm_head.parameters():
        p.requires_grad_(False)

    if is_lora:
        start = QWEN_LAYERS - num_layers
        for i in range(start):
            for p in model.model.layers[i].parameters():
                p.requires_grad_(False)
        lora_cfg = peft.LoraConfig(
            r=16, lora_alpha=16, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            layers_to_transform=list(range(start, QWEN_LAYERS)),
        )
        model = peft.get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        start = QWEN_LAYERS - num_layers
        for p in model.parameters():
            p.requires_grad_(False)
        for i in range(start, QWEN_LAYERS):
            for p in model.model.layers[i].parameters():
                p.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable/1e6:.2f}M / {sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)

    # 数据
    ds = load_dataset("json", data_files={"train": os.path.join(DATA_DIR, "train.jsonl"),
                                          "valid": os.path.join(DATA_DIR, "valid.jsonl")})

    def tok_fn(examples):
        return tok(examples["text"], truncation=True, max_length=512)

    train_ds = ds["train"].map(tok_fn, batched=True, remove_columns=["text"])
    eval_ds = ds["valid"].map(tok_fn, batched=True, remove_columns=["text"])

    # 默认关闭梯度检查点: 与"冻结前N层"组合会触发 checkpoint 重算报错
    # (element 0 of tensors does not require grad...)
    # 若 Full 训练 OOM, 用 AUTODL_CKPT=1 开启(省显存, 代价约 20% 速度)
    use_grad_ckpt = os.environ.get("AUTODL_CKPT", "0") == "1"

    # 确保保存目录存在 (adapters/models 可能是指向数据盘的符号链接, 新实例为空目录)
    os.makedirs(save_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=os.path.join(save_dir, "_ckpt"), max_steps=iters,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1e-4, bf16=True, fp16=False,
        logging_steps=5, save_strategy="no",
        eval_strategy="steps", eval_steps=100, report_to=[], seed=42,
        dataloader_num_workers=4,   # CPU 并行加载数据(避免 GPU 等数据)
        dataloader_pin_memory=True,
        gradient_checkpointing=use_grad_ckpt,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))
    t0 = time.time()
    trainer.train()
    print(f"  训练耗时 {time.time()-t0:.0f}s", flush=True)

    os.makedirs(save_dir, exist_ok=True)
    if is_lora:
        model.save_pretrained(save_dir)  # adapter_config.json + adapter_model.safetensors
    else:
        model.save_pretrained(save_dir)  # 完整模型
    tok.save_pretrained(save_dir)
    log(f"已保存 -> {save_dir}")


# ==================== 评测 ====================
def load_problems():
    pat = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.S)
    problems = []
    with open(os.path.join(DATA_DIR, "valid.jsonl")) as f:
        for line in f:
            m = pat.search(json.loads(line)["text"])
            if m:
                problems.append((m.group(1).strip(), extract_answer(m.group(2).strip())))
    return problems


def evaluate(name, method, is_lora):
    json_path = os.path.join(EVAL_DIR, f"{name}.json")
    log_path = os.path.join(EVAL_DIR, f"{name}.log")
    if os.path.exists(json_path):
        log(f"{name} 评测已存在, 跳过")
        return
    os.makedirs(EVAL_DIR, exist_ok=True)

    log(f"Eval: {name} ({method})")
    problems = load_problems()
    total = len(problems)

    # 评测后端: transformers generate (兼容 2080 Ti sm_75, 不依赖 vLLM)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if name == "baseline":
        model_path, adapter_dir = MODEL, None
    elif is_lora:
        model_path, adapter_dir = MODEL, os.path.join(ADAPTER_DIR, name)
    else:
        model_path, adapter_dir = os.path.join(FULL_DIR, name), None
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                 trust_remote_code=True).to("cuda")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir).to("cuda")
    model.eval()

    results = []
    log_f = open(log_path, "w", buffering=1)
    t_start = time.time()
    for i, (problem, expected) in enumerate(problems):
        prompt = EVAL_PROMPT.format(problem=problem)
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        t1 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        elapsed = time.time() - t1
        text = tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ans = extract_answer(text)
        correct = answers_match(ans, expected)
        tok_cnt = gen.shape[1] - inputs.input_ids.shape[1]
        results.append({"idx": i, "problem": problem, "expected": expected, "output": text,
                        "extracted": ans, "correct": correct,
                        "completion_tokens": tok_cnt, "output_chars": len(text),
                        "gen_time_s": round(elapsed, 2), "has_boxed": "\\boxed" in text})
        log_f.write(f"######第{i+1}个问题######\n{problem}\n\n---模型输出---\n{text}\n\n")
        log_f.write(f"##第{i+1}个问题的答案##：{ans} | 预期: {expected} | {'✓' if correct else '✗'}\n{'='*40}\n\n")
        print(f"  [#{i+1:02d}] {elapsed:.1f}s {tok_cnt}tok | Expect={expected} | Got={ans} | {'✓' if correct else '✗'}", flush=True)
    log_f.close()

    correct = sum(1 for r in results if r["correct"])
    toks = [r["completion_tokens"] for r in results]
    summary = {"version": name, "accuracy": correct / total, "correct": correct, "total": total,
               "avg_completion_tokens": round(sum(toks) / total, 1),
               "extract_failed": sum(1 for r in results if not r["extracted"]),
               "boxed_ratio": round(sum(1 for r in results if r["has_boxed"]) / total, 3),
               "results": results}
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Accuracy: {correct}/{total} = {correct/total*100:.1f}% | 总耗时 {time.time()-t_start:.0f}s", flush=True)
    log(f"已保存 -> {json_path}")


# ==================== 评测 (vLLM 批量, 需环境装 vllm>=0.8.5) ====================
def _eval_vllm_run(llm, sampling, name, method, prompts, problems, lora_path=None):
    json_path = os.path.join(EVAL_DIR, f"{name}.json")
    if os.path.exists(json_path) and not FORCE:
        log(f"{name} 评测已存在, 跳过 (FORCE=1 强制重评)")
        return
    from vllm.lora.request import LoRARequest
    lora_req = LoRARequest(lora_name=name, lora_path=lora_path) if lora_path else None
    log(f"Eval(vLLM): {name} ({method})" + (" + LoRA" if lora_req else ""))
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)
    gen_time = time.time() - t0

    results = []
    log_f = open(os.path.join(EVAL_DIR, f"{name}.log"), "w", buffering=1)
    for i, (out, (problem, expected)) in enumerate(zip(outputs, problems)):
        text = out.outputs[0].text
        ans = extract_answer(text)
        correct = answers_match(ans, expected)
        tok_cnt = len(out.outputs[0].token_ids)
        results.append({"idx": i, "problem": problem, "expected": expected, "output": text,
                        "extracted": ans, "correct": correct,
                        "completion_tokens": tok_cnt, "output_chars": len(text),
                        "gen_time_s": 0.0, "has_boxed": "\\boxed" in text})
        log_f.write(f"######第{i+1}个问题######\n{problem}\n\n---模型输出---\n{text}\n\n")
        log_f.write(f"##第{i+1}个问题的答案##：{ans} | 预期: {expected} | {'✓' if correct else '✗'}\n{'='*40}\n\n")
    log_f.close()

    total = len(problems)
    correct = sum(1 for r in results if r["correct"])
    toks = [r["completion_tokens"] for r in results]
    summary = {"version": name, "method": method, "accuracy": correct / total, "correct": correct, "total": total,
               "avg_completion_tokens": round(sum(toks) / total, 1),
               "extract_failed": sum(1 for r in results if not r["extracted"]),
               "boxed_ratio": round(sum(1 for r in results if r["has_boxed"]) / total, 3),
               "total_gen_time_s": round(gen_time, 1), "results": results}
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Accuracy: {correct}/{total} = {correct/total*100:.1f}% | 50条生成 {gen_time:.0f}s ({total/gen_time:.1f} 条/s)", flush=True)


def evaluate_all_vllm():
    from vllm import LLM, SamplingParams
    problems = load_problems()
    prompts = [EVAL_PROMPT.format(problem=p) for p, _ in problems]
    os.makedirs(EVAL_DIR, exist_ok=True)
    sampling = SamplingParams(max_tokens=MAX_NEW, temperature=0.0)

    # base 实例: baseline + 3 个 LoRA (换 lora_request, 免重复加载模型)
    log(f"评测后端 vLLM: 加载 base 模型 {MODEL} (enable_lora)...")
    llm = LLM(model=MODEL, trust_remote_code=True, enable_lora=True, max_lora_rank=16,
              dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.8, tensor_parallel_size=1)
    for method, layers, iters, name, is_lora in MATRIX:
        if name == "baseline":
            _eval_vllm_run(llm, sampling, "baseline", "baseline", prompts, problems)
        elif is_lora:
            _eval_vllm_run(llm, sampling, name, "LoRA", prompts, problems, lora_path=os.path.join(ADAPTER_DIR, name))
    del llm

    # Full: 每版本独立模型路径 (0.6B 加载几秒, 可接受)
    for method, layers, iters, name, is_lora in MATRIX:
        if is_lora or name == "baseline":
            continue
        mp = os.path.join(FULL_DIR, name)
        if not os.path.isfile(os.path.join(mp, "config.json")):
            log(f"{name} 模型不存在 {mp}, 跳过")
            continue
        log(f"加载 Full 模型 {mp}...")
        fllm = LLM(model=mp, trust_remote_code=True, enable_lora=False,
                   dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.8, tensor_parallel_size=1)
        _eval_vllm_run(fllm, sampling, name, "Full", prompts, problems)
        del fllm
        import gc; gc.collect()


# ==================== 对比报告 ====================
def compare_results():
    log("生成对比报告")
    lines = ["# ReasonLite SFT 10 组实验对比报告 (AutoDL)", f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             f"数据: 4000 训练 + 50 验证 (token<500) | 评测: transformers generate, 验证集 50 条\n",
             "| 版本 | 方法 | 可训层 | 步数 | 正确率 | 平均token | 提取失败 |", "|------|------|-------|------|-------|-----------|---------|"]
    rows = []
    for method, layers, iters, name, _ in MATRIX:
        p = os.path.join(EVAL_DIR, f"{name}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            rows.append((d["accuracy"], name, method, layers, iters, d))
    rows.sort(key=lambda x: -x[0])
    for acc, name, method, layers, iters, d in rows:
        lines.append(f"| {name} | {method} | {layers} | {iters or '—'} | {acc*100:.1f}% | {d['avg_completion_tokens']} | {d['extract_failed']} |")

    lines.append("\n## 逐题对错矩阵\n")
    evaled = [n for _, _, _, n, _ in MATRIX if os.path.exists(os.path.join(EVAL_DIR, f"{n}.json"))]
    lines.append("| # | 预期 | " + " | ".join(evaled) + " |")
    lines.append("|:-:|:-:|" + ":|" * len(evaled))
    if evaled:
        base = json.load(open(os.path.join(EVAL_DIR, f"{evaled[0]}.json")))
        for i in range(base["total"]):
            row = f"| {i+1} | {base['results'][i]['expected']} "
            for n in evaled:
                r = json.load(open(os.path.join(EVAL_DIR, f"{n}.json")))["results"][i]
                row += f" | {'✓' if r['correct'] else '✗'}"
            lines.append(row + " |")
    report = "\n".join(lines)
    path = os.path.join(EVAL_DIR, "comparison.md")
    open(path, "w").write(report)
    log(f"报告 -> {path}")
    print(report, flush=True)


def main(train_only=False, eval_only=False):
    log("=" * 60)
    log("ReasonLite SFT PIPELINE (AutoDL) START")
    log(f"模型: {MODEL}")
    log("模式: " + ("仅训练" if train_only else "仅评估" if eval_only else "训练 + 评估"))
    log("=" * 60)
    t_start = time.time()
    if not eval_only:
        ensure_env()
        prepare_data()
        # LoRA 后8层 100/500/1000
        for iters in [100, 500, 1000]:
            train(f"lora_{iters}", f"LoRA r16 后8层 {iters}", iters, is_lora=True, num_layers=8)
        # Full 后8层
        for iters in [100, 500, 1000]:
            train(f"full8_{iters}", f"Full 后8层 {iters}", iters, is_lora=False, num_layers=8)
        # Full 后16层
        for iters in [100, 500, 1000]:
            train(f"full16_{iters}", f"Full 后16层 {iters}", iters, is_lora=False, num_layers=16)
    # 评估全部 10 版本 (优先 vLLM 批量, 否则 transformers 逐条)
    if not train_only:
        if VLLM_AVAILABLE:
            log("评测后端: vLLM (批量, 5-10x 加速)")
            try:
                evaluate_all_vllm()
            except Exception as e:
                print(f"  vLLM 评测失败: {e}, 回退 transformers", flush=True)
                for method, _, _, name, is_lora in MATRIX:
                    try:
                        evaluate(name, method, is_lora)
                    except Exception as e2:
                        print(f"  {name} 评测失败: {e2}", flush=True)
        else:
            log("评测后端: transformers generate (装 vllm>=0.8.5 可加速)")
            for method, _, _, name, is_lora in MATRIX:
                try:
                    evaluate(name, method, is_lora)
                except Exception as e:
                    print(f"  {name} 评测失败: {e}", flush=True)
        compare_results()
    log(f"PIPELINE COMPLETE! 总耗时 {(time.time()-t_start)/3600:.1f} 小时")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ReasonLite SFT pipeline (AutoDL)")
    ap.add_argument("--train-only", action="store_true", help="只训练 (9 组), 跳过评估")
    ap.add_argument("--eval-only", action="store_true", help="只评估 (10 版本, 需已有训练产物), 跳过训练")
    args = ap.parse_args()
    main(train_only=args.train_only, eval_only=args.eval_only)
