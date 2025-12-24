# Project TODO List

## Phase 1: Initial GPU Validation (Before Full-Scale Training)

These minimal tests verify the reorganized codebase works correctly on a GPU instance.

### 1.1 Environment Setup

```bash
# Switch to GPU instance (g4dn.xlarge minimum, g5.xlarge recommended)
# Then run:

# Activate conda environment
conda activate pepmlm

# Install/update dependencies
pip install -r requirements.txt

# Verify GPU access
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Verify imports work
python -c "from src.distillation import DistillationTrainer; import config; print('Imports OK')"
```

**Checklist:**
- [ ] GPU instance running
- [ ] Conda environment activated
- [ ] Dependencies installed
- [ ] GPU detected by PyTorch
- [ ] Project imports work

### 1.2 Quick Smoke Test (~5-10 min)

Run minimal training to verify everything works:

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Minimal training run (very small model, tiny dataset fraction)
python scripts/train.py \
    --temperature 2.0 \
    --alpha 0.5 \
    --n_layer 2 \
    --n_head 2 \
    --n_embd 128 \
    --train_size_prop 0.001 \
    --learning_rate 1e-3
```

**Checklist:**
- [ ] Teacher model loads successfully
- [ ] Student model initializes
- [ ] Dataset loads from parquet files
- [ ] Training loop starts without errors
- [ ] W&B logging works (check https://wandb.ai)
- [ ] Model saves to `./models/` directory
- [ ] No CUDA out-of-memory errors

### 1.3 Evaluate Existing Models (~2-3 min)

Test evaluation script with pre-trained HuggingFace model:

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Evaluate a published model
python scripts/evaluate.py \
    --student_model littleworth/protgpt2-distilled-tiny \
    --num_samples 20 \
    --output results_test.json

# View results
cat results_test.json
```

**Checklist:**
- [ ] Both teacher and student models load
- [ ] Perplexity computed for both
- [ ] KL divergence computed
- [ ] Sequences generated from both models
- [ ] Results saved to `results_test.json`
- [ ] Assessment printed (Good/Acceptable/Marginal/Poor)

### 1.4 Generation Test (~1 min)

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Generate sequences
python scripts/generate.py \
    --model littleworth/protgpt2-distilled-tiny \
    --num_sequences 5 \
    --max_length 50

# Save to file
python scripts/generate.py \
    --model littleworth/protgpt2-distilled-tiny \
    --num_sequences 10 \
    --max_length 100 \
    --output generated_sequences.fasta

# View output
cat generated_sequences.fasta
```

**Checklist:**
- [ ] Model loads without errors
- [ ] Sequences generated in FASTA format
- [ ] Output contains valid amino acid characters (A-Z)
- [ ] File saved correctly

### 1.5 Legacy Script Test (Optional, ~5 min)

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Test legacy training with text file
python scripts/train_legacy.py \
    --temperature 2.0 \
    --alpha 0.5 \
    --data_file data/train_small.txt
```

**Checklist:**
- [ ] Text data loads and tokenizes
- [ ] Training runs without errors

### 1.6 ESMFold Test (Optional, requires 16GB+ GPU)

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Test ESMFold pLDDT prediction
python -c "
from src.esmfold import predict_plddt
seq = 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'
score = predict_plddt(seq)
print(f'pLDDT score: {score:.2f}')
"
```

**Checklist:**
- [ ] ESMFold model loads
- [ ] pLDDT score computed (should be 0-100)

---

## Phase 2: Full-Scale Training

After Phase 1 passes, proceed with production training.

### 2.1 Single Model Training

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Tiny model (fastest, ~30 min with 10% data)
python scripts/train.py \
    --temperature 2.0 \
    --alpha 0.5 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 256 \
    --train_size_prop 0.1 \
    --learning_rate 1e-3

# Small model (~1-2 hours with 10% data)
python scripts/train.py \
    --temperature 2.0 \
    --alpha 0.5 \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --train_size_prop 0.1 \
    --learning_rate 5e-4

# Medium model (~3-4 hours with 10% data)
python scripts/train.py \
    --temperature 2.0 \
    --alpha 0.5 \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --train_size_prop 0.1 \
    --learning_rate 1e-4
```

**Checklist:**
- [ ] Tiny model trained
- [ ] Small model trained
- [ ] Medium model trained

### 2.2 Background Training with nohup

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Run training in background (survives SSH disconnect)
nohup python scripts/train.py \
    --temperature 2.0 \
    --alpha 0.5 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 256 \
    --train_size_prop 0.1 \
    --learning_rate 1e-3 \
    > training.log 2>&1 &

# Check progress
tail -f training.log

# Check if still running
ps aux | grep train.py
```

### 2.3 Batch Training (Multiple Configurations)

First, edit the parameter sets in `scripts/batch_train.sh`:

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Edit batch_train.sh to add your configurations
nano scripts/batch_train.sh

# Example configurations to add:
# parameter_sets=(
#    "2.0 0.5 4 4 256 0.1 0.001"
#    "2.0 0.3 4 4 256 0.1 0.001"
#    "5.0 0.5 4 4 256 0.1 0.001"
#    "2.0 0.5 6 6 512 0.1 0.0005"
# )

# Run batch training in background
nohup ./scripts/batch_train.sh > batch_training.log 2>&1 &

# Monitor progress
tail -f batch_training.log
```

**Checklist:**
- [ ] Parameter sets configured
- [ ] Batch training started
- [ ] All configurations completed

### 2.4 Hyperparameter Sweep Examples

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Temperature sweep
for temp in 1.0 2.0 5.0 10.0; do
    python scripts/train.py \
        --temperature $temp \
        --alpha 0.5 \
        --n_layer 4 --n_head 4 --n_embd 256 \
        --train_size_prop 0.1 \
        --learning_rate 1e-3
done

# Alpha sweep
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    python scripts/train.py \
        --temperature 2.0 \
        --alpha $alpha \
        --n_layer 4 --n_head 4 --n_embd 256 \
        --train_size_prop 0.1 \
        --learning_rate 1e-3
done

# Learning rate sweep
for lr in 1e-4 5e-4 1e-3; do
    python scripts/train.py \
        --temperature 2.0 \
        --alpha 0.5 \
        --n_layer 4 --n_head 4 --n_embd 256 \
        --train_size_prop 0.1 \
        --learning_rate $lr
done
```

---

## Phase 3: Evaluation & Analysis

### 3.1 Evaluate All Trained Models

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# List all trained models
ls -la models/

# Evaluate each model (replace with actual model name)
python scripts/evaluate.py \
    --student_model ./models/protgpt2-distilled-t2.0-a0.5-l4-h4-e256-p0.1-lr1e-03.uniprot \
    --num_samples 100 \
    --output results/eval_tiny_t2.0_a0.5.json

# Batch evaluate all models
mkdir -p results
for model_dir in models/protgpt2-distilled-*; do
    model_name=$(basename $model_dir)
    echo "Evaluating $model_name..."
    python scripts/evaluate.py \
        --student_model "$model_dir" \
        --num_samples 100 \
        --output "results/eval_${model_name}.json"
done

# View all results
cat results/*.json | grep -E "(student_model|perplexity_ratio|ASSESSMENT)"
```

**Checklist:**
- [ ] All models evaluated
- [ ] Results saved to `results/` directory
- [ ] Best performing models identified

### 3.2 Compare Perplexity Across Models

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Quick comparison script
python -c "
import json
import glob

results = []
for f in glob.glob('results/eval_*.json'):
    with open(f) as fp:
        data = json.load(fp)
        results.append({
            'model': data['student_model'].split('/')[-1],
            'compression': data['compression_ratio'],
            'ppl_ratio': data['perplexity_ratio'],
            'kl_div': data['kl_divergence']
        })

# Sort by perplexity ratio
results.sort(key=lambda x: x['ppl_ratio'])
print('Ranked by perplexity ratio (lower is better):')
for r in results:
    print(f\"  {r['model'][:50]}: {r['ppl_ratio']:.2f}x (KL: {r['kl_div']:.4f})\")
"
```

### 3.3 Generate Sequences for Analysis

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Generate from teacher
python scripts/generate.py \
    --model nferruz/ProtGPT2 \
    --num_sequences 100 \
    --max_length 200 \
    --output results/teacher_sequences.fasta

# Generate from best student
python scripts/generate.py \
    --model ./models/YOUR_BEST_MODEL \
    --num_sequences 100 \
    --max_length 200 \
    --output results/student_sequences.fasta

# Compare sequence lengths
wc -l results/teacher_sequences.fasta results/student_sequences.fasta
```

### 3.4 ESMFold Structural Evaluation (Optional)

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Evaluate pLDDT scores for generated sequences
python -c "
from src.esmfold import predict_plddt
import numpy as np

# Read sequences from FASTA
sequences = []
with open('results/student_sequences.fasta') as f:
    seq = ''
    for line in f:
        if line.startswith('>'):
            if seq:
                sequences.append(seq)
            seq = ''
        else:
            seq += line.strip()
    if seq:
        sequences.append(seq)

# Evaluate first 10 (ESMFold is slow)
scores = []
for i, seq in enumerate(sequences[:10]):
    score = predict_plddt(seq[:100])  # Truncate for speed
    scores.append(score)
    print(f'Seq {i}: pLDDT = {score:.2f}')

print(f'Mean pLDDT: {np.mean(scores):.2f} +/- {np.std(scores):.2f}')
"
```

---

## Phase 4: Model Release

### 4.1 Upload Best Model to HuggingFace

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Login to HuggingFace (if not using HF_TOKEN env var)
huggingface-cli login

# Upload model
python tools/upload_to_hf.py \
    --model_dir ./models/YOUR_BEST_MODEL \
    --repo_id YOUR_USERNAME/protgpt2-distilled-tiny

# Upload medium model
python tools/upload_to_hf.py \
    --model_dir ./models/YOUR_MEDIUM_MODEL \
    --repo_id YOUR_USERNAME/protgpt2-distilled-medium
```

**Checklist:**
- [ ] Best tiny model uploaded
- [ ] Best medium model uploaded
- [ ] Models accessible on HuggingFace

### 4.2 Test Uploaded Model

```bash
# Verify uploaded model works
python scripts/generate.py \
    --model YOUR_USERNAME/protgpt2-distilled-tiny \
    --num_sequences 5

python scripts/evaluate.py \
    --student_model YOUR_USERNAME/protgpt2-distilled-tiny \
    --num_samples 50
```

---

## Phase 5: Documentation & Cleanup

### 5.1 Update Documentation

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Update README with final results
nano README.md

# Add evaluation results, best configurations, etc.
```

### 5.2 Cleanup

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

# Remove test outputs
rm -f results_test.json generated_sequences.fasta

# List large model directories
du -sh models/*

# Remove unwanted model checkpoints (be careful!)
# rm -rf models/protgpt2-distilled-UNWANTED-MODEL

# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Check disk usage
df -h
```

### 5.3 Final Git Push

```bash
cd /home/ubuntu/storage1/distilling_protgpt2

git add -A
git status
git commit -m "docs: add final evaluation results and cleanup"
git push origin master
git push github master
```

---

## Quick Reference

### AWS Instance Recommendations

| Instance | GPU | VRAM | Hourly Cost | Use Case |
|----------|-----|------|-------------|----------|
| g4dn.xlarge | T4 | 16GB | ~$0.53 | Development, tiny models |
| g5.xlarge | A10G | 24GB | ~$1.01 | Training, evaluation |
| g5.2xlarge | A10G | 24GB | ~$1.21 | Faster training |
| p3.2xlarge | V100 | 16GB | ~$3.06 | Large models |

### Typical Training Times (10% data, 3 epochs)

| Model Size | g4dn.xlarge | g5.xlarge |
|------------|-------------|-----------|
| Tiny (4L/4H/256E) | ~45 min | ~30 min |
| Small (6L/8H/512E) | ~2 hours | ~1.5 hours |
| Medium (12L/12H/768E) | ~5 hours | ~3.5 hours |

### Model Naming Convention

```
protgpt2-distilled-t{temp}-a{alpha}-l{layers}-h{heads}-e{embed}-p{prop}-lr{lr}.uniprot
```

Example: `protgpt2-distilled-t2.0-a0.5-l4-h4-e256-p0.1-lr1e-03.uniprot`
