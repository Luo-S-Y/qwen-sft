#!/bin/bash
# Pipeline: 训练 + 评测全自动执行
# 16-bit LoRA: 100/500/1000/2000 iters
# 16-bit Full: 100/500/1000 iters
# 然后 AIME24 评测 + 结果对比

set -e
cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com

BASE="models/Qwen3-0.6B"
DATA="data/lora_4k"
BEST_BATCH="--batch_size 1 --grad_accumulation_steps 4"

echo "========================================="
echo "Step 0: Prepare 4000-sample dataset"
echo "========================================="
python3 train/prepare_data.py --num_train 4000 --num_valid 100 --num_test 100 --output_dir "$DATA"

TRAIN_CMD="python3 train/train_lora.py --model_path $BASE --data $DATA --val_batches 25"

# ==================== LoRA training ====================
for ITERS in 100 500 1000 2000; do
    echo ""
    echo "========================================="
    echo "LoRA ${ITERS} iters"
    echo "========================================="
    $TRAIN_CMD --iters $ITERS $BEST_BATCH --adapter_path "adapters/lora_${ITERS}"
done

# ==================== Full FT (last 8 layers) ====================
for ITERS in 100 500 1000; do
    echo ""
    echo "========================================="
    echo "Full FT ${ITERS} iters"
    echo "========================================="
    $TRAIN_CMD --iters $ITERS --batch_size 2 --grad_accumulation_steps 2 \
        --adapter_path "adapters/full_${ITERS}" --fine_tune_type full
done

echo ""
echo "========================================="
echo "All training done! Starting AIME24 eval..."
echo "========================================="

# ==================== Eval all models ====================
RESULT_DIR="result/pipeline"
mkdir -p "$RESULT_DIR"

for CONFIG in "lora_100" "lora_500" "lora_1000" "lora_2000" "full_100" "full_500" "full_1000"; do
    echo ""
    echo "---------- Evaluating $CONFIG ----------"
    python3 -c "
import os, sys
sys.path.insert(0, '.')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from eval.eval_aime24 import evaluate_aime24
adapter = f'adapters/{CONFIG}'
result = evaluate_aime24(
    model_path='$BASE',
    adapter_path=adapter,
    num_samples=1,
    max_tokens=2048,
    temperature=0,
    output_dir='$RESULT_DIR'
)
# Rename result file
import shutil
src = f'$RESULT_DIR/aime24_' + os.path.basename(result.get('results_file', ''))
" 2>&1
done

echo ""
echo "========================================="
echo "Done! Results in $RESULT_DIR"
echo "========================================="
