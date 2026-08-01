"""
AIME24 evaluation for Qwen3-0.6B using MLX (Apple Silicon).
Based on ReasonLite evaluation pipeline.
"""
import os, re, math, json, time, random
from tqdm import tqdm
from datasets import load_dataset
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from sympy import simplify, Eq
from sympy.parsing.latex import parse_latex

# Qwen3 chat template
QWEN3_CHAT_TEMPLATE = "<|im_start|>system\nYou are Qwen, a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"

def extract_answer(text):
    """Extract final answer from model output."""
    # Try \boxed{} first
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        return boxed[-1].strip()
    # Try $...$ in last line
    lines = text.strip().split('\n')
    for line in reversed(lines):
        dollars = re.findall(r'\$([^$]+)\$', line)
        if dollars:
            return dollars[-1].strip()
    # Try last number in text
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        return nums[-1]
    return text.strip().split()[-1] if text.strip() else ""

def normalize_answer(ans):
    """Normalize answer string for comparison."""
    if not ans:
        return ""
    ans = ans.strip()
    # Remove \\boxed wrapper
    ans = re.sub(r'\\boxed\{([^}]*)\}', r'\1', ans)
    # Remove $ signs
    ans = ans.replace('$', '')
    # Remove commas in numbers
    ans = re.sub(r'(?<=\d),(?=\d)', '', ans)
    return ans.strip()

def answers_match(pred, expected, tolerance=1e-6):
    """Check if predicted answer matches expected using sympy."""
    pred = normalize_answer(pred)
    expected = normalize_answer(expected)
    if not pred or not expected:
        return False
    if pred == expected:
        return True
    # Try numeric comparison
    try:
        p_float = float(pred)
        e_float = float(expected)
        return abs(p_float - e_float) < tolerance
    except ValueError:
        pass
    # Try sympy expression comparison
    try:
        p_expr = parse_latex(pred)
        e_expr = parse_latex(expected)
        diff = simplify(p_expr - e_expr)
        if diff == 0:
            return True
        try:
            return abs(float(diff.evalf())) < tolerance
        except:
            return False
    except:
        return False

def pass_at_k(correct_list, k):
    """Compute pass@k metric."""
    n = len(correct_list)
    c = sum(correct_list)
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    import math
    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log((n - c - k + 1 + i) / (n - k + 1 + i))
    return 1.0 - math.exp(log_ratio)

def evaluate_aime24(model_path, adapter_path=None, num_samples=16, max_tokens=2048, temperature=0.7, top_p=0.95, output_dir="result"):
    """Evaluate model on AIME24."""
    temperature = float(temperature)
    top_p = float(top_p)
    num_samples = int(num_samples)
    max_tokens = int(max_tokens)
    print(f"Loading model from {model_path}...")
    if adapter_path:
        print(f"  With LoRA adapter: {adapter_path}")
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_path)
    
    print("Loading AIME24 dataset...")
    ds = load_dataset("zwhe99/aime90", split="2024", trust_remote_code=True)
    problems = [(row["problem"], row["expected_answer"]) for row in ds]
    print(f"Loaded {len(problems)} AIME24 problems.")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    all_results = []
    problem_scores = []
    
    for idx, (problem, expected) in enumerate(tqdm(problems, desc="AIME24")):
        prompt = QWEN3_CHAT_TEMPLATE.format(problem=problem)
        corrects = []
        responses = []
        
        sampler = make_sampler(temp=temperature, top_p=top_p) if temperature > 0 else None
        for s in range(num_samples):
            out = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            # Remove prompt prefix if present
            if prompt in out:
                out = out[out.index(prompt) + len(prompt):]
            responses.append(out)
            pred = extract_answer(out)
            correct = answers_match(pred, expected)
            corrects.append(correct)
        
        avg_correct = sum(corrects) / num_samples
        pk8 = pass_at_k(corrects, 8)
        problem_scores.append(avg_correct)
        
        result = {
            "problem_idx": idx,
            "problem": problem[:100],
            "expected": expected,
            "avg@16": avg_correct,
            "pass@8": pk8,
            "correct_count": sum(corrects),
            "total": num_samples,
            "predictions": responses[:3],  # save first 3 responses
            "extracted": [extract_answer(r) for r in responses[:3]],
        }
        all_results.append(result)
        
        print(f"\n  Problem {idx+1}: avg@{num_samples}={avg_correct:.3f}, pass@8={pk8:.3f} (expected: {expected})")
    
    # Summary
    overall_avg = sum(problem_scores) / len(problem_scores)
    print(f"\n{'='*60}")
    print(f"OVERALL avg@{num_samples}: {overall_avg*100:.1f}")
    print(f"Problems correct (avg@{num_samples}): {sum(1 for s in problem_scores if s > 0.5)}/{len(problem_scores)}")
    
    summary = {
        "model": model_path,
        "dataset": "zwhe99/aime90 (2024)",
        "num_samples": num_samples,
        "overall_avg": overall_avg,
        "problem_scores": problem_scores,
        "results": all_results,
        "config": {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
    }
    
    out_path = os.path.join(output_dir, f"aime24_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")
    
    return summary

if __name__ == "__main__":
    import fire
    fire.Fire(evaluate_aime24)
