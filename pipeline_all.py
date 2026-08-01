"""
ReasonLite SFT 复现流程 (llama.cpp GGUF 评估)
1. 数据: token<500 短样本, 4000 train + 50 valid
2. 训练 9 组:
   - LoRA (r=16, 后8层): 100 / 500 / 1000 iters
   - Full 后8层 (layer20-27): 100 / 500 / 1000
   - Full 后16层 (layer12-27): 100 / 500 / 1000
3. 每版本 fuse(合并adapter) -> 转 GGUF (Q8_0)
4. llama.cpp GGUF AIME24 评测 (baseline + 9 训练版), 分版本日志
5. 对比报告
"""
import os, sys, time, json, subprocess
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTHONUNBUFFERED'] = '1'

BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)
LOG_DIR = BASE_DIR / "result" / "pipeline"
GGUF_EVAL_DIR = BASE_DIR / "result" / "gguf_eval"
ADAPTER_DIR = BASE_DIR / "adapters"
DATA_DIR = BASE_DIR / "data" / "lora_short"
GGUF_DIR = BASE_DIR / "models" / "gguf"
FUSED_DIR = BASE_DIR / "models" / "fused"
VENV_PYTHON = str(BASE_DIR / "venv" / "bin" / "python3")
CONVERT = str(BASE_DIR / "llama.cpp" / "convert_hf_to_gguf.py")
LLAMA_CLI = str(BASE_DIR / "llama.cpp" / "build" / "bin")

LOG_DIR.mkdir(parents=True, exist_ok=True)
GGUF_EVAL_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)
FUSED_DIR.mkdir(parents=True, exist_ok=True)

# (method, 可训层, iters, adapter, version)
MATRIX = [
    ("baseline", "—", None, None, "baseline"),
    ("LoRA", "r16 后8层", 100, "lora_100", "lora_100"),
    ("LoRA", "r16 后8层", 500, "lora_500", "lora_500"),
    ("LoRA", "r16 后8层", 1000, "lora_1000", "lora_1000"),
    ("Full", "后8层", 100, "full8_100", "full8_100"),
    ("Full", "后8层", 500, "full8_500", "full8_500"),
    ("Full", "后8层", 1000, "full8_1000", "full8_1000"),
    ("Full", "后16层", 100, "full16_100", "full16_100"),
    ("Full", "后16层", 500, "full16_500", "full16_500"),
    ("Full", "后16层", 1000, "full16_1000", "full16_1000"),
]

def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"\n[{t}] {msg}", flush=True)

def run_cmd(cmd, step_name, filter_keywords=True):
    log(f"Starting: {step_name}")
    log(f"  {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        if not filter_keywords or ("Iter " in line or "Saved" in line or "Val loss" in line
                                   or "FAILED" in line or "Accuracy" in line or "kept" in line
                                   or "Scanned" in line or "Eval:" in line or "  [" in line):
            print(f"  {line}", flush=True)
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log(f"  FAILED (exit={proc.returncode}) after {elapsed:.0f}s")
        return False
    log(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return True

def prepare_data():
    log("Step 0: Prepare short samples (token<500): 4000 train + 50 valid")
    return run_cmd([
        VENV_PYTHON, "train/prepare_data.py",
        "--num_train", "4000", "--num_valid", "50",
        "--max_tokens", "500", "--output_dir", str(DATA_DIR),
    ], "Data preparation")

def train_lora(iters, adapter_name):
    log(f"Step: LoRA r16 后8层 {iters} iters -> {adapter_name}")
    return run_cmd([
        VENV_PYTHON, "train/train_lora.py",
        "--model_path", "models/Qwen3-0.6B",
        "--iters", str(iters),
        "--batch_size", "1", "--grad_accumulation_steps", "4",
        "--num_layers", "8", "--lora_rank", "16",
        "--train_file", str(DATA_DIR / "train.jsonl"),
        "--valid_file", str(DATA_DIR / "valid.jsonl"),
        "--adapter_path", str(ADAPTER_DIR / adapter_name),
    ], f"LoRA-{iters}")

def train_full(iters, adapter_name, num_layers):
    log(f"Step: Full FT 后{num_layers}层 {iters} iters -> {adapter_name}")
    return run_cmd([
        VENV_PYTHON, "train/train_lora.py",
        "--model_path", "models/Qwen3-0.6B",
        "--iters", str(iters),
        "--batch_size", "2", "--grad_accumulation_steps", "2",
        "--fine_tune_type", "full", "--num_layers", str(num_layers),
        "--train_file", str(DATA_DIR / "train.jsonl"),
        "--valid_file", str(DATA_DIR / "valid.jsonl"),
        "--adapter_path", str(ADAPTER_DIR / adapter_name),
    ], f"Full{num_layers}-{iters}")

def fuse_and_convert(adapter_name, version):
    """合并 adapter -> HF 格式 -> GGUF (Q8_0)"""
    fused = FUSED_DIR / adapter_name
    gguf = GGUF_DIR / f"{version}.gguf"
    if gguf.exists():
        log(f"GGUF exists, skip fuse/convert: {gguf}")
        return True
    log(f"Step: fuse {adapter_name} + convert to GGUF")
    ok = run_cmd([
        VENV_PYTHON, "-m", "mlx_lm.fuse",
        "--model", "models/Qwen3-0.6B",
        "--adapter-path", str(ADAPTER_DIR / adapter_name),
        "--save-path", str(fused),
    ], f"fuse {adapter_name}", filter_keywords=False)
    if not ok:
        return False
    return run_cmd([
        VENV_PYTHON, CONVERT, str(fused),
        "--outfile", str(gguf), "--outtype", "q8_0",
    ], f"convert {version}", filter_keywords=False)

def evaluate_gguf(version):
    log(f"Eval (GGUF): {version}")
    if (GGUF_EVAL_DIR / f"{version}.json").exists():
        log(f"  Eval result exists, skip: {version}.json")
        return True
    gguf = GGUF_DIR / f"{version}.gguf"
    return run_cmd([
        VENV_PYTHON, "eval/eval_gguf.py",
        "--gguf", str(gguf), "--version", version,
        "--outdir", str(GGUF_EVAL_DIR),
    ], f"GGUF eval {version}", filter_keywords=False)

def compare_results():
    log("Generating comparison report")
    versions = [v[4] for v in MATRIX]
    all_results = {}
    for v in versions:
        f = GGUF_EVAL_DIR / f"{v}.json"
        if f.exists():
            with open(f) as fh:
                all_results[v] = json.load(fh)

    report_path = LOG_DIR / "comparison.md"
    lines = []
    lines.append("# ReasonLite SFT 10 组实验对比报告\n")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"数据: 4000 训练 + 50 验证 (token<500 短样本) | 评测: llama.cpp GGUF Q8_0, AIME24 30题\n")

    lines.append("## 总体表现\n")
    lines.append("| 版本 | 方法 | 可训层 | 步数 | 正确率 | 平均耗时(s) | 平均token | 提取失败 | boxed占比 |")
    lines.append("|------|------|-------|------|-------|------------|-----------|---------|-----------|")
    rows = []
    for method, layers, iters, _, v in MATRIX:
        if v in all_results:
            d = all_results[v]
            rows.append((d["accuracy"], v, method, layers, iters, d))
    rows.sort(key=lambda x: -x[0])
    for acc, v, method, layers, iters, d in rows:
        lines.append(f"| {v} | {method} | {layers} | {iters or '—'} | {acc*100:.1f}% "
                     f"| {d['avg_gen_time_s']} | {d['avg_completion_tokens']} "
                     f"| {d['extract_failed']} | {d['boxed_ratio']*100:.0f}% |")

    lines.append("\n## 逐题对错矩阵 (✓=对, ✗(答案)=错)\n")
    lines.append("| # | 预期 | " + " | ".join(v[4] for v in MATRIX if v[4] in all_results) + " |")
    ncols = len([v[4] for v in MATRIX if v[4] in all_results])
    lines.append("|:-:|:-:|" + ":|" * ncols)
    if all_results:
        n = list(all_results.values())[0]["total"]
        for i in range(n):
            expected = list(all_results.values())[0]["results"][i]["expected"]
            row = f"| {i+1} | {expected} "
            for _, _, _, _, v in MATRIX:
                if v in all_results:
                    r = all_results[v]["results"][i]
                    mark = "✓" if r["correct"] else f"✗({r['extracted'][:8]})"
                    row += f" | {mark}"
            lines.append(row + " |")

    lines.append("\n## 说明\n")
    lines.append("- 每版本完整日志: `result/gguf_eval/{version}.log` (问题原文+完整输出+答案)")
    lines.append("- 结构化结果: `result/gguf_eval/{version}.json` (含耗时/token/格式维度)")
    lines.append("- 提取失败: 模型输出中无法解析出数字答案的题数")

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    log(f"Report saved to {report_path}")
    print(f"\n{report}", flush=True)

# ==================== MAIN ====================
if __name__ == "__main__":
    log("=" * 60)
    log("ReasonLite SFT PIPELINE START")
    log("LoRA(r16后8层) 100/500/1000 | Full后8层 100/500/1000 | Full后16层 100/500/1000")
    log("评估: llama.cpp GGUF Q8_0, AIME24 30题, 分版本日志")
    log("=" * 60)

    t_start = time.time()

    if not prepare_data():
        log("Data prep failed, aborting")
        sys.exit(1)

    # LoRA 100/500/1000
    for iters in [100, 500, 1000]:
        train_lora(iters, f"lora_{iters}")

    # Full 后8层 100/500/1000
    for iters in [100, 500, 1000]:
        train_full(iters, f"full8_{iters}", num_layers=8)

    # Full 后16层 100/500/1000
    for iters in [100, 500, 1000]:
        train_full(iters, f"full16_{iters}", num_layers=16)

    # fuse + convert + evaluate 全部版本
    for method, _, _, adapter, version in MATRIX:
        if method == "baseline":
            # baseline: 直接转换原模型
            gguf = GGUF_DIR / "baseline.gguf"
            if not gguf.exists():
                ok = run_cmd([VENV_PYTHON, CONVERT, "models/Qwen3-0.6B",
                              "--outfile", str(gguf), "--outtype", "q8_0"],
                             "convert baseline", filter_keywords=False)
                if not ok:
                    log("baseline convert failed, aborting")
                    sys.exit(1)
        else:
            if not fuse_and_convert(adapter, version):
                log(f"fuse/convert {version} failed, skipping eval")
                continue
        evaluate_gguf(version)

    compare_results()

    total = time.time() - t_start
    log(f"\n{'='*60}")
    log(f"PIPELINE COMPLETE! Total: {total/3600:.1f} hours")
    log(f"{'='*60}")
