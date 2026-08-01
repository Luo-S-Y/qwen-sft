"""
单进程 AIME24 评测：实时显示 token + 日志文件，格式：
######第N个问题######
[streaming tokens...]
##第N个问题的答案##：extracted_answer
"""
import os, re, json, time, sys
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

SYSTEM_PROMPT = "You are a helpful math assistant. Solve the problem and provide the final answer directly."
PROMPT_TEMPLATE = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"

def extract_answer(text):
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed: return boxed[-1].strip()
    lines = text.strip().split('\n')
    for line in reversed(lines):
        dollars = re.findall(r'\$([^$]+)\$', line)
        if dollars: return dollars[-1].strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums: return nums[-1]
    return text.strip().split()[-1][:30] if text.strip() else ""

def main(adapter_path="adapters/stage1_test", max_tokens=2048, log_dir="result/stream_logs"):
    import warnings
    warnings.filterwarnings("ignore")
    
    from datasets import load_dataset
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    
    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    model, tok = load("models/Qwen3-0.6B", adapter_path=adapter_path if adapter_path != "none" else None)
    print(f"Loaded in {time.time()-t0:.1f}s\n", flush=True)
    
    # Load dataset
    ds = load_dataset("zwhe99/aime90", split="2024", trust_remote_code=True)
    total = len(ds)
    print(f"AIME24: {total} problems\n", flush=True)
    
    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / "eval_full.log"
    log_f = open(log_path, "w", buffering=1)
    
    sampler = make_sampler(temp=0)
    
    results = []
    t_start = time.time()
    
    for i in range(total):
        problem = ds[i]["problem"]
        expected = ds[i]["expected_answer"]
        
        header = f"######第{i+1}个问题######"
        print(f"\n{header}", flush=True)
        print(f"{problem[:200]}...", flush=True)
        log_f.write(f"{header}\n")
        log_f.write(f"{problem}\n\n")
        log_f.flush()
        
        prompt = PROMPT_TEMPLATE.format(problem=problem)
        tokens = []
        t1 = time.time()
        
        for response in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler):
            text = response.text
            tokens.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
            log_f.write(text)
            log_f.flush()
        
        elapsed = time.time() - t1
        full_text = "".join(tokens)
        if prompt in full_text:
            full_text = full_text[full_text.index(prompt) + len(prompt):]
        
        ans = extract_answer(full_text)
        tok_cnt = len(full_text.split())
        correct = ans.strip() == expected.strip()
        
        ans_line = f"\n##第{i+1}个问题的答案##：{ans}"
        print(ans_line, flush=True)
        log_f.write(f"\n{ans_line}\n")
        
        summary = f"[#{i+1:02d}] {elapsed:.1f}s | {tok_cnt} tok | Expected={expected} | Got={ans} | {'✓' if correct else '✗'}"
        print(summary, flush=True)
        log_f.write(f"{summary}\n{'='*60}\n\n")
        log_f.flush()
        
        results.append((i, expected, ans, elapsed, tok_cnt, correct))
    
    log_f.close()
    total_time = time.time() - t_start
    correct_count = sum(1 for r in results if r[5])
    
    print(f"\n{'='*60}", flush=True)
    print(f"Done! {total_time:.1f}s total | Correct: {correct_count}/{total} ({correct_count/total*100:.1f}%)", flush=True)
    print(f"Log: {log_path}", flush=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"result/eval_{timestamp}.json"
    os.makedirs("result", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "total_time": total_time, "correct": correct_count, "total": total,
            "accuracy": correct_count / total,
            "results": [{"idx": r[0], "expected": r[1], "predicted": r[2], "time": r[3], "tokens": r[4]} for r in results],
            "config": {"model": "models/Qwen3-0.6B", "adapter": adapter_path, "temp": 0, "max_tokens": max_tokens}
        }, f, indent=2)
    print(f"Results: {out_path}", flush=True)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
