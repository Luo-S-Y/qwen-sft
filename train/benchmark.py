"""
Benchmark: compare training speed. Run each test separately.
"""
import os, time, json, warnings, sys
from pathlib import Path
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
warnings.filterwarnings("ignore")

from mlx_lm.lora import run

DATA_PATH = "data/lora_subset"

def make_args(model_path, fine_tune_type, adapter_dir):
    class Args: pass
    args = Args()
    args.model = model_path
    args.data = DATA_PATH
    args.adapter_path = adapter_dir
    args.seed = 42
    args.train = True
    args.test = False
    args.fine_tune_type = fine_tune_type
    args.num_layers = 8
    args.batch_size = 2
    args.iters = 10
    args.val_batches = 5
    args.steps_per_report = 2
    args.steps_per_eval = 10
    args.save_every = 10
    args.max_seq_length = 2048
    args.learning_rate = 1e-4
    args.optimizer = "adamw"
    args.optimizer_config = {}
    args.grad_checkpoint = False
    args.grad_accumulation_steps = 2
    args.lora_parameters = {
        "rank": 8, "alpha": 16, "dropout": 0.0, "scale": 16/8
    } if fine_tune_type == "lora" else None
    args.lr_schedule = None
    args.resume_adapter_file = None
    args.hf_dataset = False
    args.prompt_feature = "prompt"
    args.text_feature = "text"
    args.completion_feature = "completion"
    args.chat_feature = "messages"
    args.mask_prompt = False
    args.report_to = ""
    args.project_name = None
    return args

def run_benchmark(name, model_path, fine_tune_type):
    adapter_dir = f"adapters/bench_{fine_tune_type}_{Path(model_path).name}"
    os.makedirs(adapter_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"  Model: {model_path}")
    print(f"  Type: {fine_tune_type}")
    print(f"  Adapter: {adapter_dir}")
    print(f"{'='*60}")

    args = make_args(model_path, fine_tune_type, adapter_dir)

    t1 = time.time()
    run(args)
    total = time.time() - t1

    return {
        "name": name,
        "type": fine_tune_type,
        "train_time_10it": round(total, 1),
        "avg_s_per_it": round(total / 10, 2),
        "it_per_s": round(10 / total, 3) if total > 0 else 0,
        "trainable_m": 600 if fine_tune_type == "full" else 1.44,
        "total_m": 600,
    }

if __name__ == "__main__":
    test = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = []

    if test in ("all", "16lora"):
        results.append(run_benchmark("16-bit LoRA", "models/Qwen3-0.6B", "lora"))
    if test in ("all", "4lora"):
        results.append(run_benchmark("4-bit LoRA", "models/Qwen3-0.6B-4bit", "lora"))
    if test in ("all", "16full"):
        results.append(run_benchmark("16-bit Full FT", "models/Qwen3-0.6B", "full"))

    print(f"\n\n{'='*80}")
    print(f"{'BENCHMARK SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'10it(s)':<12} {'s/it':<12} {'it/s':<12} {'Trainable':<15} {'Total':<10}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['name']:<20} {r['train_time_10it']:<12} {r['avg_s_per_it']:<12} {r['it_per_s']:<12} {r['trainable_m']:<15} {r['total_m']:<10}")

    out_path = "result/benchmark.json"
    os.makedirs("result", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
