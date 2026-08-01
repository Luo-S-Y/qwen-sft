"""
Prepare ReasonLite short-CoT data: 只保留 token<500 的短样本。
输出: 4000 训练 + 50 验证（其余丢弃）。完整模板 text 的 token 数 < max_tokens。
"""
import os, json, random, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
from transformers import AutoTokenizer

CHAT_TEMPLATE = "<|im_start|>system\nYou are Qwen, a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"

def main(num_train=4000, num_valid=50, max_tokens=500, max_scan=300000,
         output_dir="data/lora_short", seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    print(f"Loading tokenizer (models/Qwen3-0.6B)...")
    tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    print(f"Streaming ReasonLite-Dataset (medium), filter text<= {max_tokens} tokens...")
    ds = load_dataset("amd/ReasonLite-Dataset", split="medium", streaming=True, trust_remote_code=True)

    # 批量 tokenize 提速（transformers 逐条 encode 太慢）
    BATCH = 5000
    target = num_train + num_valid
    buf, short = [], []
    i = 0
    for i, row in enumerate(ds):
        if i >= max_scan:
            break
        buf.append(CHAT_TEMPLATE.format(prompt=row["prompt"], completion=row["answer"]))
        if len(buf) >= BATCH:
            encs = tok(buf, add_special_tokens=True, padding=False)
            lens = [len(e) for e in encs["input_ids"]]
            short.extend(t for t, l in zip(buf, lens) if l <= max_tokens)
            buf = []
            print(f"  scanned {i+1}, kept {len(short)} ({time.time()-t0:.0f}s)", flush=True)
            if len(short) >= target:
                break
    if buf:
        encs = tok(buf, add_special_tokens=True, padding=False)
        lens = [len(e) for e in encs["input_ids"]]
        short.extend(t for t, l in zip(buf, lens) if l <= max_tokens)
    print(f"Scanned {i+1} rows, kept {len(short)} short samples ({time.time()-t0:.0f}s)", flush=True)

    short = [{"text": t} for t in short]
    if len(short) < num_train + num_valid:
        print(f"WARNING: 仅 {len(short)} 条短样本, 不足 {num_train+num_valid}, 需增大 max_scan!", flush=True)
    random.shuffle(short)
    for split_name, n in [("train", num_train), ("valid", num_valid)]:
        rows = short[:n]
        del short[:n]
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for r in rows:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")
        print(f"  {split_name}: {len(rows)} -> {out_path}")

    # token 长度分布（用于报告）
    with open(os.path.join(output_dir, "train.jsonl")) as f:
        toks = sorted(len(tok.encode(json.loads(line)["text"])) for line in f if line.strip())
    print(f"Train token len: min={toks[0]} p50={toks[len(toks)//2]} max={toks[-1]}")
    print(f"Done -> {output_dir}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
