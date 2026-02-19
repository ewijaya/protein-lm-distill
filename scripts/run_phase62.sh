#!/usr/bin/env bash
# Phase 6.2: Few-shot sample efficiency experiment
# 5 models x 5 subset sizes x 3 families = 75 fine-tuning + evaluation runs
# Skips runs whose result JSON already exists (safe to restart after interruption)
set -euo pipefail

source /home/ubuntu/storage1/anaconda3/etc/profile.d/conda.sh
conda activate pepmlm

cd /home/ubuntu/storage1/protein-lm-distill

# Log path can be overridden, e.g. PHASE62_LOG=/tmp/phase62.log bash scripts/run_phase62.sh
LOG_FILE="${PHASE62_LOG:-/home/ubuntu/storage1/protein-lm-distill/phase62.log}"
if ! touch "$LOG_FILE" 2>/dev/null; then
    LOG_FILE="/tmp/phase62.log"
    touch "$LOG_FILE"
    echo "WARN: Could not write requested log path; falling back to $LOG_FILE"
fi
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

FAMILIES=(amp conotoxin lysozyme)
SUBSETS=(50 100 200 500 1000)
MODELS=(teacher medium small tiny baseline-tiny)

# Model paths
declare -A PATHS
PATHS[teacher]="nferruz/ProtGPT2"
PATHS[medium]="littleworth/protgpt2-distilled-medium"
PATHS[small]="littleworth/protgpt2-distilled-small"
PATHS[tiny]="littleworth/protgpt2-distilled-tiny"
PATHS[baseline-tiny]="models/baseline-tiny"

# Learning rates (middle of PRD grid per model size)
declare -A LRS
LRS[teacher]="2e-5"
LRS[medium]="5e-5"
LRS[small]="1e-4"
LRS[tiny]="2e-4"
LRS[baseline-tiny]="2e-4"

# Batch sizes (smaller for teacher to fit T4 16GB)
declare -A BS GA
BS[teacher]=4;          GA[teacher]=2
BS[medium]=8;           GA[medium]=1
BS[small]=8;            GA[small]=1
BS[tiny]=8;             GA[tiny]=1
BS[baseline-tiny]=8;    GA[baseline-tiny]=1

# HMM profiles (AMPs: none - use AA/length KL instead)
declare -A HMMS
HMMS[amp]=""
HMMS[conotoxin]="data/hmm/PF02950.hmm"
HMMS[lysozyme]="data/hmm/PF00959.hmm"

mkdir -p results/finetune/seqs

TOTAL=$(( ${#FAMILIES[@]} * ${#MODELS[@]} * ${#SUBSETS[@]} ))
DONE=0
SKIPPED=0

echo "===== Phase 6.2 starting: $TOTAL runs ====="
echo "Start time: $(date)"

for family in "${FAMILIES[@]}"; do
    for model in "${MODELS[@]}"; do
        for subset in "${SUBSETS[@]}"; do
            train_file="train_${subset}.fasta"
            out_dir="models/finetune/${family}-${model}-${subset}"
            result_file="results/finetune/${family}-${model}-${subset}.json"

            # Skip if evaluation result already exists
            if [[ -f "$result_file" ]]; then
                SKIPPED=$((SKIPPED + 1))
                echo "SKIP ($SKIPPED): $result_file exists"
                continue
            fi

            DONE=$((DONE + 1))
            echo ""
            echo "===== [$DONE/$TOTAL] FINETUNE: ${family} / ${model} / ${subset} ====="
            echo "Time: $(date)"

            python scripts/finetune.py \
                --model "${PATHS[$model]}" \
                --data_dir "data/finetune/${family}" \
                --train_file "$train_file" \
                --val_file val.fasta \
                --output_dir "$out_dir" \
                --epochs 20 \
                --batch_size "${BS[$model]}" \
                --gradient_accumulation_steps "${GA[$model]}" \
                --learning_rate "${LRS[$model]}" \
                --early_stopping_patience 3 \
                --warmup_steps 100 \
                --overwrite_output_dir \
                --wandb_project PROTGPT2_FINETUNE \
                --wandb_run_name "ft-${family}-${model}-${subset}"

            echo "===== [$DONE/$TOTAL] EVALUATE: ${family} / ${model} / ${subset} ====="

            hmm_args=""
            if [[ -n "${HMMS[$family]}" ]]; then
                hmm_args="--hmm_profile ${HMMS[$family]}"
            fi

            python scripts/evaluate_finetune.py \
                --model "$out_dir" \
                --train_file "data/finetune/${family}/${train_file}" \
                --test_file "data/finetune/${family}/test.fasta" \
                --family "$family" \
                --num_generate 200 \
                $hmm_args \
                --output "$result_file" \
                --save_sequences "results/finetune/seqs/${family}-${model}-${subset}.fasta"
        done
    done
done

echo ""
echo "===== Phase 6.2 complete ====="
echo "Finished: $DONE runs, skipped: $SKIPPED"
echo "End time: $(date)"

# Auto-stop instance when done
/home/ubuntu/bin/stopinstance
