"""
2 workers, 2 problems (1 each). Log format:
######第n个问题######
[streaming tokens]
##第n个问题的答案##：answer
"""
import os, re, json, time, sys
from multiprocessing import Process, Queue
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

def worker(worker_id, idx, problem, expected, result_queue, log_dir="result/stream_logs"):
    """Single worker: processes one problem, streams tokens to its log file."""
    import warnings
    warnings.filterwarnings("ignore")
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    adapter = "adapters/stage1_test"
    print(f"[W{worker_id}] Loading model...", flush=True)
    t0 = time.time()
    model, tok = load("models/Qwen3-0.6B", adapter_path=adapter)
    print(f"[W{worker_id}] Loaded in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / f"worker_{worker_id}.log"
    log_f = open(log_path, "w", buffering=1)
    print(f"[W{worker_id}] Log: {log_path}", flush=True)

    sampler = make_sampler(temp=0)

    # Write problem header + problem text
    log_f.write(f"######第{idx+1}个问题######\n")
    log_f.write(f"问题原文：{problem}\n")
    log_f.write(f"---以上是问题，以下是模型输出---\n")
    log_f.flush()

    prompt = PROMPT_TEMPLATE.format(problem=problem)
    t1 = time.time()
    tokens = []

    # Stream tokens to log
    for response in stream_generate(model, tok, prompt=prompt, max_tokens=2048, sampler=sampler):
        text = response.text
        tokens.append(text)
        log_f.write(text)
        log_f.flush()

    elapsed = time.time() - t1
    full_text = "".join(tokens)
    if prompt in full_text:
        full_text = full_text[full_text.index(prompt) + len(prompt):]

    ans = extract_answer(full_text)
    tok_cnt = len(full_text.split())

    # Write answer line
    log_f.write(f"\n##第{idx+1}个问题的答案##：{ans}\n")
    log_f.flush()

    correct = ans.strip() == expected.strip()
    print(f"[W{worker_id} #{idx+1}] {elapsed:.1f}s | {tok_cnt} tok | Expected={expected} | Got={ans} | {'✓' if correct else '✗'}", flush=True)

    log_f.close()
    result_queue.put((worker_id, idx, expected, ans, elapsed, tok_cnt, correct))

def main(log_dir="result/stream_logs"):
    from datasets import load_dataset

    print("Loading AIME24...", flush=True)
    ds = load_dataset("zwhe99/aime90", split="2024", trust_remote_code=True)

    # Only 2 problems
    prob0 = (0, ds[0]["problem"], ds[0]["expected_answer"])
    prob1 = (1, ds[1]["problem"], ds[1]["expected_answer"])

    print(f"Starting 2 workers (1 problem each)...\n", flush=True)
    t_start = time.time()

    # Run workers as separate processes
    q = Queue()
    proc1 = Process(target=worker, args=(1, prob0[0], prob0[1], prob0[2], q, log_dir))
    proc2 = Process(target=worker, args=(2, prob1[0], prob1[1], prob1[2], q, log_dir))

    proc1.start()
    proc2.start()

    results = [None, None]
    for _ in range(2):
        wid, idx, expected, ans, elapsed, tok_cnt, correct = q.get()
        results[idx] = (idx, expected, ans, elapsed, tok_cnt, correct)

    proc1.join()
    proc2.join()

    total_time = time.time() - t_start
    correct_count = sum(1 for r in results if r[5])

    print(f"\n{'='*60}", flush=True)
    print(f"Done! {total_time:.1f}s | Correct: {correct_count}/2", flush=True)
    print(f"Logs: {Path(log_dir).resolve()}/worker_1.log, worker_2.log", flush=True)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
