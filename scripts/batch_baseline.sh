#!/bin/bash
# Baseline Training Pipeline (without Phase 0 enhancements)
# Matches synergy training architectures for fair comparison

cd /home/ubuntu/storage1/protein-lm-distill

echo "=== Starting Baseline Training Pipeline ==="
echo "Start time: $(date)"

echo "=== [1/6] Training Baseline Tiny (4L/4H/512E) ==="
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 4 --n_head 4 --n_embd 512 \
    --train_size_prop 0.1 --learning_rate 1e-3 \
    --output_dir ./models/baseline-tiny

echo "=== [2/6] Evaluating Baseline Tiny ==="
python scripts/evaluate.py \
    --student_model ./models/baseline-tiny \
    --num_samples 100 --compute_ece \
    --output results/eval_baseline_tiny.json

echo "=== [3/6] Training Baseline Small (6L/8H/768E) ==="
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 6 --n_head 8 --n_embd 768 \
    --train_size_prop 0.1 --learning_rate 5e-4 \
    --output_dir ./models/baseline-small

echo "=== [4/6] Evaluating Baseline Small ==="
python scripts/evaluate.py \
    --student_model ./models/baseline-small \
    --num_samples 100 --compute_ece \
    --output results/eval_baseline_small.json

echo "=== [5/6] Training Baseline Medium (12L/16H/1024E) ==="
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 12 --n_head 16 --n_embd 1024 \
    --train_size_prop 0.1 --learning_rate 1e-4 \
    --output_dir ./models/baseline-medium

echo "=== [6/6] Evaluating Baseline Medium ==="
python scripts/evaluate.py \
    --student_model ./models/baseline-medium \
    --num_samples 100 --compute_ece \
    --output results/eval_baseline_medium.json

echo "=== Pipeline Complete ==="
echo "End time: $(date)"
