# ProtGPT2 Distillation Project - Comprehensive PRD

## Executive Summary

Complete the ProtGPT2 distillation project through systematic training, evaluation, HuggingFace deployment, and paper publication. The goal is to produce three production-quality distilled protein language models (Tiny, Small, Medium) with rigorous hyperparameter optimization and comprehensive evaluation.

---

## Project Timeline Overview

| Phase | Milestone | Duration | Dependencies |
|-------|-----------|----------|--------------|
| **1** | Baseline Training | ~8 hours | None (In Progress) |
| **2** | Hyperparameter Sweeps | 2-3 days | Phase 1 |
| **3** | Comprehensive Evaluation | 1 day | Phase 2 |
| **4** | HuggingFace Update | 1 day | Phase 3 |
| **5** | Publication Writing | 3-5 days | Phase 3 |

---

## Phase 1: Baseline Training (IN PROGRESS)

**Status**: Currently running via nohup

**Note**: Current training uses smaller architectures. After completion, we will retrain with architectures matching existing HuggingFace models.

**Current Training** (will complete for reference):
| Model | Config | Est. Params | Est. Time |
|-------|--------|-------------|-----------|
| Tiny | 4L/4H/256E, T=2.0, α=0.5 | ~10M | ~45 min |
| Small | 6L/8H/512E, T=2.0, α=0.5 | ~40M | ~2 hours |
| Medium | 12L/12H/768E, T=2.0, α=0.5 | ~125M | ~5 hours |

**Final Training** (matching HuggingFace architectures):
| Model | Config | Est. Params | Matches HF |
|-------|--------|-------------|------------|
| Tiny | 4L/4H/512E, T=best, α=best | ~39M | littleworth/protgpt2-distilled-tiny |
| Small | 6L/8H/768E, T=best, α=best | ~82M | littleworth/protgpt2-distilled-small |
| Medium | 12L/16H/1024E, T=best, α=best | ~200M | littleworth/protgpt2-distilled-medium |

**Output Location**: `./models/protgpt2-distilled-t2.0-a0.5-l{L}-h{H}-e{E}-p0.1-lr*.uniprot/`

**Monitoring**:
```bash
tail -f training_baseline.log
```

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

### 2.3 Final Training with Optimal Hyperparameters

Apply best T and α to all model sizes (matching HuggingFace architectures):

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

---

## Phase 3: Comprehensive Evaluation

### 3.1 Core Metrics to Collect

| Metric | Description | Target |
|--------|-------------|--------|
| Perplexity | Next-token prediction | Lower = better |
| Perplexity Ratio | Student/Teacher PPL | < 1.5 (Excellent), < 2.0 (Good) |
| KL Divergence | Output distribution similarity | Lower = better |
| Compression Ratio | Teacher/Student params | Report actual |
| AA Distribution KL | vs natural protein distribution | < 0.05 |

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

### 3.4 Comparison Summary Table

Generate after all evaluations:
```bash
python -c "
import json, glob
results = []
for f in glob.glob('results/eval_*.json'):
    with open(f) as fp:
        data = json.load(fp)
        results.append({
            'model': data['student_model'].split('/')[-1][:40],
            'params': data.get('student_params', 'N/A'),
            'compression': data.get('compression_ratio', 'N/A'),
            'ppl_ratio': data.get('perplexity_ratio', 'N/A'),
            'kl_div': data.get('kl_divergence', 'N/A')
        })
results.sort(key=lambda x: x.get('ppl_ratio', 999) if isinstance(x.get('ppl_ratio'), (int, float)) else 999)
for r in results:
    print(f\"{r['model']}: PPL ratio={r['ppl_ratio']}, KL={r['kl_div']}\")
"
```

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
# Only upload if new model is better than existing
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

### 4.4 Model Card Template (README.md for each repo)

```markdown
# ProtGPT2-Distilled-{Size}

A knowledge-distilled version of [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2) for efficient protein sequence generation.

## Model Details

| Property | Value |
|----------|-------|
| Parameters | XX M |
| Layers | X |
| Attention Heads | X |
| Embedding Size | XXX |
| Compression Ratio | XXx vs ProtGPT2 |

## Training

- **Teacher Model**: nferruz/ProtGPT2 (738M params)
- **Temperature**: X.X
- **Alpha**: X.X
- **Dataset**: UniRef50 (XX% subset)
- **Training Time**: X hours on Tesla T4

## Evaluation

| Metric | This Model | Teacher |
|--------|------------|---------|
| Perplexity | XX.X | XX.X |
| Perplexity Ratio | X.Xx | 1.0x |
| KL Divergence | X.XXX | - |

## Usage

\```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("littleworth/protgpt2-distilled-{size}")
tokenizer = GPT2Tokenizer.from_pretrained("littleworth/protgpt2-distilled-{size}")

# Generate protein sequence
input_ids = tokenizer.encode("<|endoftext|>", return_tensors="pt")
output = model.generate(input_ids, max_length=100, do_sample=True, top_k=950)
sequence = tokenizer.decode(output[0])
\```

## Citation

If you use this model, please cite:
\```bibtex
@misc{protgpt2distilled2024,
  author = {Your Name},
  title = {Efficient Knowledge Distillation for Protein Language Models},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/littleworth/protgpt2-distilled-{size}}
}
\```
```

### 4.5 Post-Upload Verification

```bash
# Verify each model loads from HF
for size in tiny small medium; do
    python scripts/generate.py \
        --model littleworth/protgpt2-distilled-${size} \
        --num_sequences 3 \
        --max_length 50
done
```

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

### 5.4 Publication Strategy (User Preference: Preprint + Journal)

**Step 1: bioRxiv Preprint** (Immediate upon completion)
- Submit to establish priority
- Get DOI for HuggingFace model cards
- Enable community feedback

**Step 2: Journal Submission** (After preprint)

| Venue | Type | Timeline | Notes |
|-------|------|----------|-------|
| Bioinformatics | Journal | 2-3 months | High visibility in computational biology |
| PLOS Comp Bio | Journal | 2-3 months | Open access, good for methods papers |
| Briefings in Bioinformatics | Journal | 2-3 months | Review-style, high impact |

**Timeline**:
1. Week 1: Complete paper draft
2. Week 2: Internal review, polish
3. Week 2-3: Submit to bioRxiv
4. Week 3: Submit to journal

### 5.5 Existing Resources

- **docs/METHODS.md** - Complete mathematical framework with LaTeX, ready for paper Methods section
- **W&B Dashboard** - https://wandb.ai/ewijaya/PROTGPT2_DISTILLATION - Training curves and metrics

---

## Risk Mitigation

### If New Models Underperform Existing HF Models

1. Try higher temperature (match T=10 from existing)
2. Lower alpha toward 0.1
3. Match embedding dimensions (512, 768, 1024)
4. Increase training data to 20% or 50%
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

## Critical Files

| File | Purpose |
|------|---------|
| `scripts/train.py` | Main training script |
| `scripts/evaluate.py` | Evaluation metrics |
| `scripts/generate.py` | Sequence generation |
| `tools/upload_to_hf.py` | HuggingFace upload |
| `docs/METHODS.md` | Paper Methods content |
| `src/distillation.py` | DistillationTrainer class |

---

## Success Criteria

### Must-Have (MVP)
- [ ] Three baseline models trained (Tiny, Small, Medium) matching HF architectures
- [ ] Perplexity ratio < 2.0x for all models
- [ ] Evaluation metrics documented
- [ ] HuggingFace repos updated with new models and documentation

### Should-Have
- [ ] Optimal hyperparameters from sweeps
- [ ] pLDDT evaluation completed
- [ ] Paper draft complete

### Nice-to-Have
- [ ] Paper submitted to preprint
- [ ] Comparison with other distillation methods
