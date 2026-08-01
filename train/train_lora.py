"""
Q-LoRA training for Qwen3-0.6B on ReasonLite-Dataset (short CoT).
Uses mlx-lm's built-in LoRA training with 4-bit quantized base model.
"""
import os, sys
from pathlib import Path
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
from mlx_lm.lora import run

def train_lora(
    model_path="models/Qwen3-0.6B",
    train_file="data/lora_subset/train.jsonl",
    valid_file="data/lora_subset/valid.jsonl",
    test_file="data/lora_subset/test.jsonl",
    adapter_path="adapters/stage1_lora",
    num_layers=8,
    lora_rank=16,
    lora_alpha=16,
    lora_dropout=0.0,
    fine_tune_type="lora",
    batch_size=4,
    iters=500,
    val_batches=25,
    steps_per_report=10,
    steps_per_eval=50,
    save_every=100,
    max_seq_length=2048,
    learning_rate=1e-4,
    optimizer="adamw",
    grad_checkpoint=False,
    grad_accumulation_steps=2,
    seed=42,
):
    """
    Train Q-LoRA on ReasonLite short CoT data.
    """
    class Args:
        pass
    
    args = Args()
    args.model = model_path
    args.data = str(Path(train_file).parent)
    args.adapter_path = adapter_path
    args.seed = seed
    args.train = True
    args.test = False
    args.fine_tune_type = fine_tune_type
    args.num_layers = num_layers
    args.batch_size = batch_size
    args.iters = iters
    args.val_batches = val_batches
    args.steps_per_report = steps_per_report
    args.steps_per_eval = steps_per_eval
    args.save_every = save_every
    args.max_seq_length = max_seq_length
    args.learning_rate = learning_rate
    args.optimizer = optimizer
    args.optimizer_config = {}
    args.grad_checkpoint = grad_checkpoint
    args.grad_accumulation_steps = grad_accumulation_steps
    args.lora_parameters = {
        "rank": lora_rank,
        "alpha": lora_alpha,
        "dropout": lora_dropout,
        "scale": lora_alpha / lora_rank,
    }
    args.lr_schedule = None
    args.resume_adapter_file = None
    args.hf_dataset = False
    args.prompt_feature = "prompt"
    args.text_feature = "text"
    args.completion_feature = "completion"
    args.chat_feature = "messages"
    args.mask_prompt = False  # not supported for text dataset
    args.report_to = ""
    args.project_name = None
    
    print("=" * 60)
    print("Q-LoRA Training: Qwen3-0.6B + ReasonLite Short CoT")
    print("=" * 60)
    print(f"  Base model: {model_path}")
    print(f"  Data: {train_file}")
    print(f"  LoRA rank={lora_rank}, alpha={lora_alpha}, layers={num_layers}")
    print(f"  Batch={batch_size}, iters={iters}, lr={learning_rate}")
    print(f"  Adapter: {adapter_path}")
    print()
    
    run(args)
    
    print(f"\nDone! Adapter saved to {adapter_path}/adapters.safetensors")

if __name__ == "__main__":
    import fire
    fire.Fire(train_lora)
