"""
GGUF (llama.cpp) AIME24 评测：llama-server + HTTP 逐题请求。
输出: 每个版本独立 .json(结构化+多维度) 和 .log(问题+完整输出) 到 result/gguf_eval/
用法: venv/bin/python3 eval/eval_gguf.py --gguf models/gguf/lora_100.gguf --version lora_100
"""
import os, re, json, time, subprocess, urllib.request, sys
from pathlib import Path

BASE = Path(__file__).parent.parent
LLAMA_SERVER = BASE / "llama.cpp" / "build" / "bin" / "llama-server"
PORT = 8085
BASE_URL = f"http://127.0.0.1:{PORT}"
SYSTEM = "You are a helpful math assistant. Solve the problem and provide the final answer directly."

def extract_answer(text):
    # 支持嵌套花括号: \boxed{-\dfrac{1}{2}} -> -\dfrac{1}{2}
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
    """语义匹配: 字符串相等 / 数值近似 / LaTeX 表达式等价"""
    p, e = pred.strip(), expected.strip()
    if not p or not e:
        return False
    if p == e:
        return True
    # 数值近似（容忍 90 vs 90^\circ 等尾部修饰）
    try:
        pf, ef = float(p), float(e)
        return abs(pf - ef) < tolerance
    except ValueError:
        pass
    # LaTeX 表达式等价
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        pe, ee = parse_latex(p), parse_latex(e)
        diff = simplify(pe - ee)
        if diff == 0:
            return True
        try:
            return abs(float(diff.evalf())) < tolerance
        except Exception:
            return False
    except Exception:
        return False

def wait_server(url, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False

def chat_completion(problem, max_tokens=2048):
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem},
        ],
        "temperature": 0, "max_tokens": max_tokens, "n": 1,
        "chat_template_kwargs": {"enable_thinking": False},  # 禁用 Qwen3 thinking
    }).encode()
    req = urllib.request.Request(BASE_URL + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    msg = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    return msg, usage

def load_problems(dataset):
    """加载评测问题: dataset="aime" 或本地 jsonl 路径(从 text 解析 user/completion)"""
    if dataset.lower() in ("aime", "aime24"):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        from datasets import load_dataset
        ds = load_dataset("zwhe99/aime90", split="2024", trust_remote_code=True)
        return [(row["problem"], str(row["expected_answer"]), False) for row in ds]  # (问题, 预期, 需提取?)
    import json as _json
    problems = []
    pat = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.S)
    for line in open(dataset):
        if not line.strip():
            continue
        m = pat.search(_json.loads(line)["text"])
        if m:
            problems.append((m.group(1).strip(), m.group(2).strip(), True))  # 预期需从 completion 提取
    return problems

def main(gguf="models/gguf/baseline.gguf", version="baseline", outdir="result/gguf_eval",
         dataset="aime"):
    outdir = BASE / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{version}.json"
    log_path = outdir / f"{version}.log"

    problems = load_problems(dataset)
    total = len(problems)
    print(f"评测集: {total} 题 | 来源: {dataset} | GGUF: {gguf}", flush=True)

    # 启动 llama-server
    server = subprocess.Popen(
        [str(LLAMA_SERVER), "-m", gguf, "--port", str(PORT), "--n-gpu-layers", "99",
         "-c", "4096", "--threads", "8"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_server(BASE_URL):
            print("ERROR: llama-server 启动失败", flush=True)
            return 1

        results = []
        log_f = open(log_path, "w", buffering=1)
        t_start = time.time()
        for i in range(total):
            problem, expected_raw, need_extract = problems[i]
            expected = extract_answer(expected_raw) if need_extract else expected_raw
            t1 = time.time()
            out, usage = chat_completion(problem)
            elapsed = time.time() - t1
            ans = extract_answer(out)
            correct = answers_match(ans, expected)

            # 记录维度
            results.append({
                "idx": i, "problem": problem, "expected": expected,
                "output": out, "extracted": ans, "correct": correct,
                "gen_time_s": round(elapsed, 2),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "output_chars": len(out), "has_boxed": "\\boxed" in out,
            })
            log_f.write(f"######第{i+1}个问题######\n{problem}\n\n---模型输出---\n{out}\n\n")
            log_f.write(f"##第{i+1}个问题的答案##：{ans} | 预期: {expected} | {'✓' if correct else '✗'}\n")
            log_f.write(f"[{elapsed:.1f}s, {usage.get('completion_tokens',0)}tok] {'='*40}\n\n")
            print(f"  [#{i+1:02d}] {elapsed:.1f}s | Expect={expected} | Got={ans} | {'✓' if correct else '✗'}", flush=True)
        log_f.close()

        correct = sum(1 for r in results if r["correct"])
        times = [r["gen_time_s"] for r in results]
        toks = [r["completion_tokens"] for r in results]
        failed_extract = sum(1 for r in results if not r["extracted"])
        boxed_ratio = sum(1 for r in results if r["has_boxed"]) / total
        summary = {
            "version": version, "gguf": str(gguf),
            "accuracy": correct / total, "correct": correct, "total": total,
            "avg_gen_time_s": round(sum(times) / total, 2),
            "median_gen_time_s": round(sorted(times)[total // 2], 2),
            "avg_completion_tokens": round(sum(toks) / total, 1),
            "extract_failed": failed_extract, "boxed_ratio": round(boxed_ratio, 3),
            "results": results,
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  Accuracy: {correct}/{total} = {correct/total*100:.1f}%", flush=True)
        print(f"  Avg {summary['avg_gen_time_s']}s/题, {summary['avg_completion_tokens']}tok/题, 提取失败{failed_extract}题, boxed占比{boxed_ratio:.0%}", flush=True)
        print(f"  Saved: {json_path}", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()
    return 0

if __name__ == "__main__":
    import fire
    fire.Fire(main)
