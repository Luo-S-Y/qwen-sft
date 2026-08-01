"""Test AIME24 evaluation with 1 problem."""
import os, sys, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.stdout.reconfigure(line_buffering=True)

from datasets import load_dataset
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

print("Loading model...")
model, tok = load('models/Qwen3-0.6B')

print("Loading dataset...")
ds = load_dataset('zwhe99/aime90', split='2024', trust_remote_code=True)

p, e = ds[0]['problem'], ds[0]['expected_answer']
print(f"Problem 1 - Expected: {e}")

prompt = f"<|im_start|>system\nYou are Qwen, a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
sampler = make_sampler(temp=0.7, top_p=0.95)
max_new_tokens = 4096*2

def extract_answer(text):
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed: return boxed[-1].strip()
    lines = text.strip().split('\n')
    for line in reversed(lines):
        dollars = re.findall(r'\$([^$]+)\$', line)
        if dollars: return dollars[-1].strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums: return nums[-1]
    return text.strip().split()[-1] if text.strip() else ""

for s in range(2):
    print(f"Sample {s+1}...")
    out = generate(model, tok, prompt=prompt, max_tokens=max_new_tokens, sampler=sampler, verbose=False)
    if prompt in out:
        out = out[out.index(prompt) + len(prompt):]
    ans = extract_answer(out)
    print(f"  Output[-300:] = ...{out[-300:]}")
    print(f"  Extracted answer: {ans}")
    print(f"  Correct: {ans == e}")
    print()

print("DONE")
