# Project TODO List

## Phase 1: Baseline Training (IN PROGRESS)

**Status**: Currently running via nohup (`training_baseline.log`)

**Note**: Current training uses smaller architectures for initial baseline. After completion, we will retrain with architectures matching existing HuggingFace models.

### Current Training Progress

| Model | Config | Est. Params | Est. Time | Status |
|-------|--------|-------------|-----------|--------|
| Tiny | 4L/4H/256E, T=2.0, α=0.5 | ~10M | ~45 min | ⏳ In Progress |
| Small | 6L/8H/512E, T=2.0, α=0.5 | ~40M | ~2 hours | ⏸️ Pending |
| Medium | 12L/12H/768E, T=2.0, α=0.5 | ~125M | ~5 hours | ⏸️ Pending |

**Monitoring**:
```bash
tail -f training_baseline.log
```

**Checklist**:
- [x] Training started via nohup
- [ ] Tiny model completed
- [ ] Small model completed
- [ ] Medium model completed
- [ ] Instance auto-stops after completion

---

## Phase 2: Hyperparameter Sweeps

### 2.1 Temperature Sweep (on Tiny model - fastest iteration)

**Rationale**: Temperature controls softness of probability distributions. Current HF models use T=10, new baseline uses T=2.0.

```bash
for temp in 1.0 2.0 4.0 6.0 8.0 10.0; do
    python scripts/train.py \
        --temperature $temp \
        --alpha 0.5 \
        --n_layer 4 --n_head 4 --n_embd 512 \
        --train_size_prop 0.1 \
        --learning_rate 1e-3
done
```

**Checklist**:
- [ ] T=1.0 completed
- [ ] T=2.0 completed
- [ ] T=4.0 completed
- [ ] T=6.0 completed
- [ ] T=8.0 completed
- [ ] T=10.0 completed
- [ ] Best temperature identified

### 2.2 Alpha Sweep (with best temperature)

**Rationale**: Alpha balances hard loss (ground truth) vs soft loss (teacher mimicking).

```bash
BEST_TEMP=X.X  # From 2.1 results

for alpha in 0.1 0.3 0.5 0.7 0.9; do
    python scripts/train.py \
        --temperature $BEST_TEMP \
        --alpha $alpha \
        --n_layer 4 --n_head 4 --n_embd 512 \
        --train_size_prop 0.1 \
        --learning_rate 1e-3
done
```

**Checklist**:
- [ ] α=0.1 completed
- [ ] α=0.3 completed
- [ ] α=0.5 completed
- [ ] α=0.7 completed
- [ ] α=0.9 completed
- [ ] Best alpha identified

### 2.3 Final Training with Optimal Hyperparameters

Apply best T and α to all model sizes (matching HuggingFace architectures):

| Model | Config | Est. Params | Matches HF |
|-------|--------|-------------|------------|
| Tiny | 4L/4H/512E, T=best, α=best | ~39M | littleworth/protgpt2-distilled-tiny |
| Small | 6L/8H/768E, T=best, α=best | ~82M | littleworth/protgpt2-distilled-small |
| Medium | 12L/16H/1024E, T=best, α=best | ~200M | littleworth/protgpt2-distilled-medium |

```bash
nohup bash -c '
# Tiny (matches littleworth/protgpt2-distilled-tiny)
python scripts/train.py --temperature $BEST_T --alpha $BEST_A --n_layer 4 --n_head 4 --n_embd 512 --train_size_prop 0.1 --learning_rate 1e-3 && \

# Small (matches littleworth/protgpt2-distilled-small)
python scripts/train.py --temperature $BEST_T --alpha $BEST_A --n_layer 6 --n_head 8 --n_embd 768 --train_size_prop 0.1 --learning_rate 5e-4 && \

# Medium (matches littleworth/protgpt2-distilled-medium)
python scripts/train.py --temperature $BEST_T --alpha $BEST_A --n_layer 12 --n_head 16 --n_embd 1024 --train_size_prop 0.1 --learning_rate 1e-4 && \

# Stop instance when done
aws ec2 stop-instances --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id)
' > training_optimized.log 2>&1 &
```

**Checklist**:
- [ ] Best hyperparameters determined (T=___, α=___)
- [ ] Tiny (HF-matching) trained
- [ ] Small (HF-matching) trained
- [ ] Medium (HF-matching) trained

---

## Phase 3: Comprehensive Evaluation

### 3.1 Core Metrics to Collect

| Metric | Description | Target |
|--------|-------------|--------|
| Perplexity | Next-token prediction | Lower = better |
| Perplexity Ratio | Student/Teacher PPL | Size-dependent (see below) |
| KL Divergence | Output distribution similarity | Lower = better |
| Compression Ratio | Teacher/Student params | Report actual |
| AA Distribution KL | vs natural protein distribution | < 0.05 |

**Size-dependent perplexity ratio thresholds:**

| Model | Params | % of Teacher | Threshold |
|-------|--------|--------------|-----------|
| Tiny | ~39M | 5% | < 3.0 |
| Small | ~82M | 11% | < 2.5 |
| Medium | ~200M | 27% | < 2.0 |

### 3.2 Evaluation Commands

```bash
mkdir -p results

# Evaluate new models
for model_dir in models/protgpt2-distilled-*-BEST*; do
    name=$(basename $model_dir)
    python scripts/evaluate.py \
        --student_model "$model_dir" \
        --num_samples 200 \
        --output "results/eval_${name}.json"
done

# Evaluate existing HF models for comparison
python scripts/evaluate.py --student_model littleworth/protgpt2-distilled-tiny --num_samples 200 --output results/eval_hf_tiny_old.json
python scripts/evaluate.py --student_model littleworth/protgpt2-distilled-small --num_samples 200 --output results/eval_hf_small_old.json
python scripts/evaluate.py --student_model littleworth/protgpt2-distilled-medium --num_samples 200 --output results/eval_hf_medium_old.json
```

**Checklist**:
- [ ] Baseline models evaluated
- [ ] Existing HF models evaluated for comparison
- [ ] Optimized models evaluated
- [ ] Comparison summary generated
- [ ] Best models identified for each size

### 3.3 pLDDT Structural Evaluation (Optional, requires 16GB+ GPU)

```bash
python -c "
from src.esmfold import predict_plddt
import numpy as np

# Test on generated sequences
sequences = ['MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG']
scores = [predict_plddt(seq) for seq in sequences]
print(f'Mean pLDDT: {np.mean(scores):.2f}')
"
```

**Checklist**:
- [ ] pLDDT evaluation completed (if GPU supports it)

---

## Phase 4: HuggingFace Update

### 4.1 Repositories to Update

| Repo | Current Config | New Config (matching architecture) |
|------|----------------|-----------------------------------|
| `littleworth/protgpt2-distilled-tiny` | 4L/4H/512E, T=10, α=0.1 | 4L/4H/512E, T=best, α=best |
| `littleworth/protgpt2-distilled-small` | 6L/8H/768E, T=10, α=0.1 | 6L/8H/768E, T=best, α=best |
| `littleworth/protgpt2-distilled-medium` | 12L/16H/1024E, T=10, α=0.1 | 12L/16H/1024E, T=best, α=best |

### 4.2 Pre-Upload Checklist

**Decision**: Always update HuggingFace models with new training approach (user preference)

- [ ] README.md prepared with updated metrics
- [ ] training_hyperparameters.json included
- [ ] Comparison table showing old vs new metrics (for transparency)
- [ ] Version history documented in README

### 4.3 Upload Commands

```bash
python tools/upload_to_hf.py \
    --model_dir ./models/BEST_TINY_MODEL \
    --repo_id littleworth/protgpt2-distilled-tiny

python tools/upload_to_hf.py \
    --model_dir ./models/BEST_SMALL_MODEL \
    --repo_id littleworth/protgpt2-distilled-small

python tools/upload_to_hf.py \
    --model_dir ./models/BEST_MEDIUM_MODEL \
    --repo_id littleworth/protgpt2-distilled-medium
```

**Checklist**:
- [ ] Tiny model uploaded
- [ ] Small model uploaded
- [ ] Medium model uploaded

### 4.4 Post-Upload Verification

```bash
# Verify each model loads from HF
for size in tiny small medium; do
    python scripts/generate.py \
        --model littleworth/protgpt2-distilled-${size} \
        --num_sequences 3 \
        --max_length 50
done
```

**Checklist**:
- [ ] All models load from HuggingFace
- [ ] Generation works correctly
- [ ] Model cards updated

---

## Phase 5: Publication

### 5.1 Paper Structure

**Title**: "Efficient Knowledge Distillation for Protein Language Models: Compressing ProtGPT2 for Rapid Protein Sequence Generation"

1. **Abstract** (~250 words)
2. **Introduction** (1-2 pages) - Motivation, gap, contribution
3. **Methods** (2-3 pages) - Distillation framework, architecture, training protocol
4. **Results** (2-3 pages) - Hyperparameter analysis, model comparison, generation quality
5. **Discussion** (1 page) - Comparison, limitations, future work
6. **Conclusion** (~0.5 page)

### 5.2 Key Figures

| Figure | Content |
|--------|---------|
| Fig 1 | Temperature vs. Perplexity Ratio |
| Fig 2 | Alpha vs. Perplexity Ratio |
| Fig 3 | Compression Ratio vs. Quality (Pareto frontier) |
| Fig 4 | AA Distribution: Teacher vs Student vs Natural |
| Fig 5 | Inference Speed Benchmark |

### 5.3 Key Tables

| Table | Content |
|-------|---------|
| Table 1 | Model configurations (params, layers, heads, embed) |
| Table 2 | Evaluation metrics for all models |
| Table 3 | Comparison with existing HF models |

### 5.4 Publication Strategy

**Step 1: bioRxiv Preprint** (Immediate upon completion)
- Submit to establish priority
- Get DOI for HuggingFace model cards
- Enable community feedback

**Step 2: Journal Submission** (After preprint)

| Venue | Type | Notes |
|-------|------|-------|
| Bioinformatics | Journal | High visibility in computational biology |
| PLOS Comp Bio | Journal | Open access, good for methods papers |
| Briefings in Bioinformatics | Journal | Review-style, high impact |

**Checklist**:
- [ ] Paper draft complete
- [ ] Figures generated
- [ ] Tables completed
- [ ] Internal review done
- [ ] bioRxiv submitted
- [ ] Journal submitted

### 5.5 Existing Resources

- **docs/METHODS.md** - Complete mathematical framework with LaTeX, ready for paper Methods section
- **W&B Dashboard** - https://wandb.ai/ewijaya/PROTGPT2_DISTILLATION - Training curves and metrics

---

## Success Criteria

### Must-Have (MVP)
- [ ] Three baseline models trained (Tiny, Small, Medium)
- [ ] Perplexity ratio meets size-dependent thresholds (Tiny < 3.0, Small < 2.5, Medium < 2.0)
- [ ] Evaluation metrics documented
- [ ] HuggingFace repos updated with new models and documentation

### Should-Have
- [ ] Optimal hyperparameters from sweeps
- [ ] pLDDT evaluation completed
- [ ] Paper draft complete

### Nice-to-Have
- [ ] Paper submitted to preprint
- [ ] Comparison with other distillation methods

---

## Decision Checkpoints

### After Phase 3 Evaluation: Data Size Decision

Check if each model meets its size-dependent perplexity ratio threshold:

| Condition | Action |
|-----------|--------|
| Model meets its threshold | Proceed to Phase 4 (HuggingFace Update) |
| Model exceeds its threshold | Retrain that model with `--train_size_prop 0.2` |
| After 20% retrain, still exceeds | Retrain with `--train_size_prop 0.5` |
| After 50% retrain, still exceeds | Keep existing HF model, document in paper |

---

## Risk Mitigation

### If New Models Underperform Existing HF Models

1. Try higher temperature (match T=10 from existing)
2. Lower alpha toward 0.1
3. Match embedding dimensions (512, 768, 1024)
4. Increase training data to 20% or 50% (see Decision Checkpoints above)
5. **Fallback**: Keep existing HF models, report comparison in paper

### If Hyperparameter Sweeps Take Too Long

1. Use 5% data for sweeps, validate best on 10%
2. Reduce to 2 epochs for sweeps
3. Run multiple sweeps in parallel on different instances

### If pLDDT Scores Are Low

1. Use longer sequences (100+ AA)
2. Compare fairly with teacher under same conditions
3. Emphasize AA distribution metrics instead

---

## Quick Reference

### Model Naming Convention

```
protgpt2-distilled-t{temp}-a{alpha}-l{layers}-h{heads}-e{embed}-p{prop}-lr{lr}.uniprot
```

Example: `protgpt2-distilled-t2.0-a0.5-l4-h4-e256-p0.1-lr1e-03.uniprot`

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
