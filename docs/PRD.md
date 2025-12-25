# ProtGPT2 Distillation Project - Comprehensive PRD

## Executive Summary

Complete the ProtGPT2 distillation project through systematic training, evaluation, HuggingFace deployment, and paper publication. The goal is to produce three production-quality distilled protein language models (Tiny, Small, Medium) with rigorous hyperparameter optimization, **protein-specific enhancements**, and comprehensive evaluation.

**Key Innovation**: This project introduces **uncertainty-aware position weighting** and **calibration-aware distillation** specifically tailored for protein sequence generation, making it the first comprehensive study of knowledge distillation for autoregressive protein language models with domain-specific optimizations.

---

## Project Timeline Overview

| Phase | Milestone | Duration | Dependencies |
|-------|-----------|----------|--------------|
| **0** | **Methodological Enhancements** | **2-3 days** | Phase 1 baseline (in progress) |
| **1** | Baseline Training | ~8 hours | None (In Progress) |
| **2** | Hyperparameter Sweeps | 2-3 days | Phase 0, Phase 1 |
| **3** | Comprehensive Evaluation | 1 day | Phase 2 |
| **4** | HuggingFace Update | 1 day | Phase 3 |
| **5** | Publication Writing | 3-5 days | Phase 3 |

---

## Phase 0: Methodological Enhancements (NEW)

**Status**: To be implemented before hyperparameter sweeps

**Objective**: Implement two protein-specific distillation enhancements that improve upon standard Hinton-style knowledge distillation.

### 0.1 Enhancement #1: Uncertainty-Aware Position Weighting

**Rationale**: Protein sequences have variable-difficulty positions. Conserved regions (buried hydrophobic cores) are easier to predict, while functional sites (active sites, binding regions, loops) are harder. Standard distillation treats all positions equally.

**Innovation**: Weight the distillation loss by position-specific uncertainty derived from teacher entropy. Focus student learning on challenging regions where the teacher has higher uncertainty.

**Mathematical Formulation**:
```
uncertainty(t) = -Σ p_teacher(t) * log(p_teacher(t))
weight(t) = 0.5 + 0.5 * normalize(uncertainty(t))
L_soft_weighted = mean(weight(t) * L_soft(t))
```

**Implementation Location**: `src/distillation.py`, in `compute_loss()` method

**Expected Impact**: 15-25% better perplexity on difficult sequences (multi-domain, low-homology proteins)

**References**:
- Kim et al. (2025). "U-Know-DiffPAN: An Uncertainty-aware Knowledge Distillation Diffusion Framework with Details Enhancement for Pansharpening". CVPR 2025. [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_U-Know-DiffPAN_An_Uncertainty-aware_Knowledge_Distillation_Diffusion_Framework_with_Details_Enhancement_CVPR_2025_paper.html)
- Yuan et al. (2025). "Uncertainty-Aware and Decoupled Distillation". International Journal of Computer Vision, 133(2), 523-541. [Paper](https://link.springer.com/article/10.1007/s11263-025-02585-2)

**Implementation Timeline**: 1-2 days
- Day 1: Implement entropy computation and position weighting
- Day 2: Test on Tiny baseline, validate improvements

### 0.2 Enhancement #2: Calibration-Aware Distillation

**Rationale**: For experimental protein design, we need well-calibrated confidence estimates to prioritize sequences for wet-lab synthesis. Standard distillation can produce overconfident predictions.

**Innovation**: Apply dynamic label smoothing to teacher distributions based on prediction confidence. Low-confidence predictions receive more smoothing, ensuring the student inherits realistic uncertainty.

**Mathematical Formulation**:
```
max_prob = max(p_teacher)
adaptive_smoothing = smoothing_factor * (1 - max_prob)
p_smoothed = (1 - adaptive_smoothing) * p_teacher + adaptive_smoothing / vocab_size
L_soft = KL(p_student || p_smoothed)
```

**Implementation Location**: `src/distillation.py`, in `compute_loss()` method

**Expected Impact**: 20-30% better calibration (lower ECE score), more reliable confidence for experimental validation

**References**:
- Song et al. (2025). "Calibration Transfer via Knowledge Distillation". Proceedings of ACCV 2024, Springer LNCS 15478, pp. 195-211. [Paper](https://link.springer.com/chapter/10.1007/978-981-96-0966-6_13)
- Label smoothing foundation: Szegedy et al. (2016). "Rethinking the Inception Architecture for Computer Vision". CVPR 2016.

**Implementation Timeline**: 1 day
- Implement dynamic label smoothing function
- Integrate with existing distillation loss
- Validate calibration improvements

### 0.3 Combined Implementation Strategy

**Week 1 Plan**:
1. **Days 1-2**: Implement Enhancement #1 (Uncertainty-Aware)
   - Add entropy computation
   - Add position weighting
   - Test on current Tiny baseline

2. **Day 3**: Implement Enhancement #2 (Calibration-Aware)
   - Add dynamic label smoothing
   - Integrate with weighted loss
   - Verify no performance regression

3. **Day 4**: Validation and Testing
   - Compare: Baseline vs. +Enhancement#1 vs. +Enhancement#1+2
   - Document improvements
   - Update training scripts

**Deliverables**:
- [ ] Enhanced `src/distillation.py` with both innovations
- [ ] Validation notebook comparing baseline vs. enhanced models
- [ ] Updated METHODS.md with mathematical formulations
- [ ] Ready for Phase 2 hyperparameter sweeps with enhancements

### 0.4 Publication Impact

**Before (Current)**:
> "We present the first systematic study of knowledge distillation for autoregressive protein language models."

**After (With Enhancements)**:
> "We present the first systematic study of knowledge distillation for autoregressive protein language models, **introducing uncertainty-aware position weighting and calibration-conscious techniques specifically tailored for protein sequence generation**."

**Upgraded Positioning**:
- From: First application of existing method
- To: First application + Novel protein-specific enhancements
- Opens doors to: Nature Communications, PNAS (vs. just Bioinformatics)

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

**Title**: "Uncertainty-Aware Knowledge Distillation for Autoregressive Protein Language Models"

**Alternative Title**: "Protein-Specific Knowledge Distillation: Uncertainty-Aware and Calibration-Conscious Compression of ProtGPT2"

1. **Abstract** (~250 words)
   - First systematic study of distillation for causal protein LMs
   - Introduction of two protein-specific enhancements
   - Key results: compression ratios, quality metrics, calibration improvements

2. **Introduction** (1-2 pages)
   - Motivation: Large protein LMs are powerful but resource-intensive
   - Gap: No prior work on distilling autoregressive protein LMs
   - Gap: Standard distillation doesn't account for protein-specific properties
   - Contributions:
     1. First comprehensive distillation study for causal protein LMs
     2. Uncertainty-aware position weighting for variable-difficulty sequences
     3. Calibration-aware distillation for reliable experimental predictions
     4. Systematic evaluation across model sizes and hyperparameters

3. **Methods** (3-4 pages)
   - 3.1: Standard knowledge distillation framework (Hinton et al. 2015)
   - 3.2: Enhancement #1 - Uncertainty-aware position weighting
   - 3.3: Enhancement #2 - Calibration-aware distillation
   - 3.4: Model architectures (Tiny, Small, Medium)
   - 3.5: Training protocol and hyperparameters
   - 3.6: Evaluation metrics (perplexity, KL divergence, ECE, pLDDT)

4. **Results** (3-4 pages)
   - 4.1: Hyperparameter analysis (temperature, alpha sweeps)
   - 4.2: Ablation study (baseline vs. +uncertainty vs. +uncertainty+calibration)
   - 4.3: Model comparison across sizes
   - 4.4: Calibration analysis (ECE scores, reliability diagrams)
   - 4.5: Generation quality (AA distribution, structural plausibility)
   - 4.6: Inference speed benchmarks

5. **Discussion** (1-2 pages)
   - Comparison with related work (DistilProtBERT, MTDP, SpiderGPT)
   - Why causal LM distillation differs from masked LM distillation
   - Importance of protein-specific enhancements
   - Limitations and future work
   - Applications to protein engineering

6. **Conclusion** (~0.5 page)

### 5.2 Key Figures

| Figure | Content |
|--------|---------|
| Fig 1 | Ablation Study: Baseline vs. +Uncertainty vs. +Uncertainty+Calibration |
| Fig 2 | Temperature vs. Perplexity Ratio (with/without enhancements) |
| Fig 3 | Alpha vs. Perplexity Ratio (with/without enhancements) |
| Fig 4 | Compression Ratio vs. Quality (Pareto frontier) |
| Fig 5 | Calibration Analysis: ECE scores and reliability diagrams |
| Fig 6 | AA Distribution: Teacher vs Student vs Natural |
| Fig 7 | Inference Speed Benchmark |
| Fig 8 | Position-wise uncertainty heatmap and weight visualization |

### 5.3 Key Tables

| Table | Content |
|-------|---------|
| Table 1 | Model configurations (params, layers, heads, embed) |
| Table 2 | Ablation study metrics (baseline vs. enhancements) |
| Table 3 | Evaluation metrics for all models (with size-dependent thresholds) |
| Table 4 | Calibration metrics (ECE, MCE, Brier score) |
| Table 5 | Comparison with related work (DistilProtBERT, MTDP, SpiderGPT) |
| Table 6 | Inference speed and memory benchmarks |

### 5.4 Publication Strategy (Enhanced with Innovations)

**Step 1: bioRxiv Preprint** (Immediate upon completion)
- Submit to establish priority
- Get DOI for HuggingFace model cards
- Enable community feedback

**Step 2: Journal Submission** (After preprint)

**With enhancements, we can target higher-tier venues:**

| Tier | Venue | Type | Timeline | Notes |
|------|-------|------|----------|-------|
| **Top-Tier** | Nature Communications | Journal | 3-4 months | Methodological novelty + protein-specific innovations |
| **Top-Tier** | PNAS | Journal | 2-3 months | Computational biology, broad impact |
| **Top-Tier** | Cell Systems | Journal | 3-4 months | Focus on computational methods |
| **Second-Tier** | Bioinformatics | Journal | 2-3 months | High visibility in computational biology |
| **Second-Tier** | PLOS Comp Bio | Journal | 2-3 months | Open access, good for methods papers |
| **Alternative** | Briefings in Bioinformatics | Journal | 2-3 months | Review-style, high impact |
| **ML Venues** | ICML/NeurIPS | Conference | 6-8 months | Computational biology workshop/track |

**Recommended Primary Target**: **Nature Communications**
- Rationale: Methodological novelty (protein-specific enhancements) + comprehensive study
- Original ProtGPT2 was published in Nature Communications
- Precedent for protein LM distillation methods papers

**Backup Targets** (in order):
1. PNAS (faster turnaround)
2. Bioinformatics (safe choice, highly cited in field)
3. PLOS Computational Biology (open access benefits)

**Timeline**:
1. Week 1: Complete paper draft with ablation studies
2. Week 2: Internal review, polish figures
3. Week 2-3: Submit to bioRxiv
4. Week 3: Submit to Nature Communications (or PNAS)
5. Months 1-3: Review process, revisions
6. Month 4: Acceptance (optimistic timeline)

### 5.5 Existing Resources

- **docs/METHODS.md** - Complete mathematical framework with LaTeX, ready for paper Methods section
- **W&B Dashboard** - https://wandb.ai/ewijaya/PROTGPT2_DISTILLATION - Training curves and metrics

---

## Decision Checkpoints

### After Phase 3 Evaluation: Data Size Decision

After completing evaluation, check perplexity ratios using **size-dependent thresholds**:

| Model | Params | % of Teacher | Perplexity Ratio Threshold |
|-------|--------|--------------|----------------------------|
| Tiny | ~39M | 5% | < 3.0 |
| Small | ~82M | 11% | < 2.5 |
| Medium | ~200M | 27% | < 2.0 |

**Rationale**: Smaller models have more aggressive compression, so expecting teacher-level performance is unrealistic. These thresholds balance compression vs. quality.

**Decision flow:**

| Condition | Action |
|-----------|--------|
| Model meets its threshold | Proceed to Phase 4 (HuggingFace Update) |
| Model exceeds its threshold | Retrain that model with `--train_size_prop 0.2` |
| After 20% retrain, still exceeds | Retrain with `--train_size_prop 0.5` |
| After 50% retrain, still exceeds | Keep existing HF model, document in paper |

**Commands for retraining with more data:**
```bash
# 20% data
python scripts/train.py --temperature $BEST_T --alpha $BEST_A --n_layer X --n_head X --n_embd X --train_size_prop 0.2

# 50% data (if 20% insufficient)
python scripts/train.py --temperature $BEST_T --alpha $BEST_A --n_layer X --n_head X --n_embd X --train_size_prop 0.5
```

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
- [ ] **Phase 0: Methodological enhancements implemented**
  - [ ] Uncertainty-aware position weighting functional
  - [ ] Calibration-aware distillation functional
  - [ ] Ablation study completed (baseline vs. enhanced)
- [ ] Three models trained (Tiny, Small, Medium) with enhancements
- [ ] Models match HF architectures (512E, 768E, 1024E)
- [ ] Perplexity ratio meets size-dependent thresholds
- [ ] Calibration improvements demonstrated (ECE scores)
- [ ] Evaluation metrics documented
- [ ] HuggingFace repos updated with enhanced models

### Should-Have
- [ ] Optimal hyperparameters from sweeps (with enhancements)
- [ ] pLDDT evaluation completed
- [ ] Position-wise uncertainty visualization
- [ ] Reliability diagrams for calibration
- [ ] Paper draft complete with ablation studies

### Nice-to-Have
- [ ] Paper submitted to Nature Communications/PNAS
- [ ] Comparison with SpiderGPT and other distillation methods
- [ ] Interactive demo on HuggingFace Spaces
