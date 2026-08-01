#!/usr/bin/env python3
"""
vLLM 批量评测 — AutoDL (4090, sm_89)
对比 transformers 逐条 generate, vLLM 连续批处理加速约 5-10 倍.

用法 (独立 conda 环境, 不碰训练环境 torch 2.5.1):
  conda create -n vllm python=3.12 -y && conda activate vllm
  pip install vllm==0.8.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
  python eval_autodl_vllm.py            # 输出 result/autodl/{name}.json+.log
  FORCE=1 python eval_autodl_vllm.py    # 强制重评(覆盖旧 transformers 结果)

环境变量:
  MAX_NEW=512   生成最大 token 数 (vLLM 快, 数学题答案不需要 2048)
  FORCE=1       忽略已有结果强制重评
"""
import os, re, json, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "lora_short")
ADAPTER_DIR = os.path.join(BASE, "adapters")
FULL_DIR = os.path.join(BASE, "models", "full")
EVAL_DIR = os.path.join(BASE, "result", "autodl")
LOCAL_MODEL = os.path.join(BASE, "models", "Qwen3-0.6B")
MODEL = LOCAL_MODEL if os.path.isfile(os.path.join(LOCAL_MODEL, "config.json")) else "Qwen/Qwen3-0.6B"
MAX_NEW = int(os.environ.get("MAX_NEW", "512"))
FORCE = os.environ.get("FORCE", "0") == "1"
EVAL_SYSTEM = "You are a helpful math assistant. Solve the problem and provide the final answer directly."
EVAL_PROMPT = ("<|im_start|>system\n" + EVAL_SYSTEM + "<|im_end|>\n"
               "<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n")

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


def log(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def load_problems():
    pat = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.S)
    problems = []
    with open(os.path.join(DATA_DIR, "valid.jsonl")) as f:
        for line in f:
            m = pat.search(json.loads(line)["text"])
            if m:
                problems.append((m.group(1).strip(), extract_answer(m.group(2).strip())))
    return problems


def run_version(llm, sampling, name, method, prompts, problems, lora_path=None):
    json_path = os.path.join(EVAL_DIR, f"{name}.json")
    if os.path.exists(json_path) and not FORCE:
        log(f"{name} 评测已存在, 跳过 (FORCE=1 强制重评)")
        return
    from vllm.lora.request import LoRARequest
    lora_req = LoRARequest(lora_name=name, lora_path=lora_path) if lora_path else None
    log(f"Eval: {name} ({method})" + (" + LoRA" if lora_req else ""))
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


def generate_report():
    lines = ["# ReasonLite SFT 10 组实验对比报告 (AutoDL vLLM)", f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             f"数据: 4000 训练 + 50 验证 (token<500) | 评测: vLLM, 验证集 50 条\n",
             "| 版本 | 方法 | 可训层 | 步数 | 正确率 | 平均token | 提取失败 | 50条耗时(s) |",
             "|------|------|-------|------|-------|-----------|---------|-----------|"]
    rows = []
    for method, layers, iters, name, _ in MATRIX:
        p = os.path.join(EVAL_DIR, f"{name}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            rows.append((d["accuracy"], name, method, layers, iters, d))
    rows.sort(key=lambda x: -x[0])
    for acc, name, method, layers, iters, d in rows:
        lines.append(f"| {name} | {method} | {layers} | {iters or '—'} | {acc*100:.1f}% | {d['avg_completion_tokens']} | {d['extract_failed']} | {d.get('total_gen_time_s', '-')} |")

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
    path = os.path.join(EVAL_DIR, "comparison_vllm.md")
    open(path, "w").write(report)
    log(f"报告 -> {path}")
    print(report, flush=True)


def main():
    from vllm import LLM, SamplingParams
    problems = load_problems()
    prompts = [EVAL_PROMPT.format(problem=p) for p, _ in problems]
    total = len(problems)
    os.makedirs(EVAL_DIR, exist_ok=True)
    sampling = SamplingParams(max_tokens=MAX_NEW, temperature=0.0)

    # base 实例: baseline + 3 个 LoRA (换 lora_request, 免重复加载模型)
    log(f"加载 base 模型 {MODEL} (enable_lora)...")
    llm = LLM(model=MODEL, trust_remote_code=True, enable_lora=True, max_lora_rank=16,
              dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.8, tensor_parallel_size=1)
    for method, layers, iters, name, is_lora in MATRIX:
        if name == "baseline":
            run_version(llm, sampling, "baseline", "baseline", prompts, problems)
        elif is_lora:
            run_version(llm, sampling, name, "LoRA", prompts, problems, lora_path=os.path.join(ADAPTER_DIR, name))

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
        run_version(fllm, sampling, name, "Full", prompts, problems)
        del fllm
        import gc; gc.collect()

    generate_report()
    log("vLLM 评测全部完成")


if __name__ == "__main__":
    main()
