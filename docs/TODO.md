# Project TODO List

**Updated**: December 25, 2025 - Phase 0 implementation complete

## Phase 0: Methodological Enhancements (COMPLETE ✓)

**Status**: ✅ Implemented and validated

**Objective**: Implement protein-specific distillation enhancements to upgrade publication from "first application" to "first application + novel techniques"

### 0.1 Enhancement #1: Uncertainty-Aware Position Weighting

**Timeline**: 1-2 days

**Rationale**: Weight distillation loss by position-specific uncertainty (entropy) to focus learning on difficult regions (active sites, loops) vs. easy regions (conserved cores).

**Implementation Steps**:
- [x] Add entropy computation function in `src/distillation.py`
  - Implemented as `compute_teacher_entropy()` method (lines 77-103)
- [x] Add position weighting in `compute_loss()` method
  - Implemented as `compute_position_weights()` method (lines 105-165)
  - Integrated in `compute_loss()` (lines 281-286)
- [x] Test on current Tiny baseline model
  - Functional tests passed
- [ ] Compare: baseline vs. uncertainty-weighted (expect 15-25% improvement on difficult sequences)
  - Ablation notebook created, awaiting training runs
- [x] Visualize position-wise uncertainty and weights
  - Implemented in `notebooks/phase_0_ablation.ipynb`
- [x] Document in `docs/METHODS.md` with mathematical formulation
  - Section 7.2 updated with implementation references

**References**:
- CVPR 2025: [U-Know-DiffPAN](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_U-Know-DiffPAN_An_Uncertainty-aware_Knowledge_Distillation_Diffusion_Framework_with_Details_Enhancement_CVPR_2025_paper.html)
- IJCV 2025: [Uncertainty-Aware Distillation](https://link.springer.com/article/10.1007/s11263-025-02585-2)

### 0.2 Enhancement #2: Calibration-Aware Distillation

**Timeline**: 1 day

**Rationale**: Apply dynamic label smoothing based on teacher confidence to ensure well-calibrated predictions for experimental prioritization.

**Implementation Steps**:
- [x] Add dynamic label smoothing function
  - Implemented as `apply_calibration_smoothing()` method (lines 167-203)
- [x] Integrate with distillation loss
  - Integrated in `compute_loss()` (lines 267-269)
- [x] Implement ECE (Expected Calibration Error) metric for evaluation
  - Implemented in `scripts/evaluate.py` as `compute_ece()` function (lines 30-137)
  - Added `--compute_ece` CLI flag
- [x] Test calibration improvements
  - Functional tests passed, ECE computation verified
- [x] Generate reliability diagrams
  - Implemented in `notebooks/phase_0_ablation.ipynb`
- [x] Document in `docs/METHODS.md`
  - Section 7.2 updated with implementation references

**References**:
- ACCV 2024/Springer 2025: [Calibration Transfer via KD](https://link.springer.com/chapter/10.1007/978-981-96-0966-6_13)

### 0.3 Ablation Study

**Timeline**: 1 day (after implementing both enhancements)

**Status**: Infrastructure ready, awaiting training runs

**Comparison Matrix** (CLI flags implemented):
- [x] Baseline (current Hinton-style distillation) - `scripts/train.py` (no enhancement flags)
- [x] Baseline + Uncertainty-Aware - `--use_uncertainty_weighting`
- [x] Baseline + Calibration-Aware - `--use_calibration_smoothing`
- [x] Baseline + Both Enhancements - `--use_uncertainty_weighting --use_calibration_smoothing`

**Metrics to Compare** (evaluation ready):
- [x] Perplexity on test set - `scripts/evaluate.py`
- [ ] Perplexity on difficult sequences (multi-domain proteins) - pending training
- [x] KL divergence - `scripts/evaluate.py`
- [x] ECE (Expected Calibration Error) - `--compute_ece` flag
- [ ] Inference time (ensure no regression) - pending training

**Deliverables**:
- [x] Ablation notebook created - `notebooks/phase_0_ablation.ipynb`
- [ ] Ablation table for paper - pending training runs
- [ ] Visualization of improvements - pending training runs
- [x] `docs/METHODS.md` updated with formulations

### 0.4 Commands to Complete Phase 0 (Run After Phase 1 Finishes)

**Prerequisites**: Phase 1 baseline training must be complete.

**Step 1: Train ablation variants** (~2 hours total on g4dn.xlarge)

```bash
# Activate environment
conda activate pepmlm
cd /home/ubuntu/storage1/protein-lm-distill

# Create results directory
mkdir -p results

# Train +Uncertainty only variant
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 4 --n_head 4 --n_embd 512 \
    --train_size_prop 0.1 --num_train_epochs 3 \
    --learning_rate 1e-3 \
    --use_uncertainty_weighting \
    --output_dir ./models/ablation-uncertainty

# Train +Calibration only variant
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 4 --n_head 4 --n_embd 512 \
    --train_size_prop 0.1 --num_train_epochs 3 \
    --learning_rate 1e-3 \
    --use_calibration_smoothing \
    --smoothing_factor 0.1 \
    --output_dir ./models/ablation-calibration

# Train +Both enhancements variant
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 4 --n_head 4 --n_embd 512 \
    --train_size_prop 0.1 --num_train_epochs 3 \
    --learning_rate 1e-3 \
    --use_uncertainty_weighting \
    --use_calibration_smoothing \
    --smoothing_factor 0.1 \
    --output_dir ./models/ablation-both
```

**Step 2: Evaluate all variants** (~10 min)

```bash
# Find the Phase 1 baseline model directory
BASELINE_MODEL=$(ls -td models/protgpt2-distilled-t2.0-a0.5-l4-h4-e512* 2>/dev/null | head -1)
echo "Baseline model: $BASELINE_MODEL"

# Evaluate baseline (from Phase 1)
python scripts/evaluate.py \
    --student_model "$BASELINE_MODEL" \
    --num_samples 100 \
    --compute_ece \
    --output results/ablation_baseline.json

# Evaluate +Uncertainty
python scripts/evaluate.py \
    --student_model ./models/ablation-uncertainty \
    --num_samples 100 \
    --compute_ece \
    --output results/ablation_uncertainty.json

# Evaluate +Calibration
python scripts/evaluate.py \
    --student_model ./models/ablation-calibration \
    --num_samples 100 \
    --compute_ece \
    --output results/ablation_calibration.json

# Evaluate +Both
python scripts/evaluate.py \
    --student_model ./models/ablation-both \
    --num_samples 100 \
    --compute_ece \
    --output results/ablation_both.json
```

**Step 3: Generate comparison summary**

```bash
python -c "
import json
from pathlib import Path

results_dir = Path('results')
variants = ['baseline', 'uncertainty', 'calibration', 'both']

print('=' * 70)
print('PHASE 0 ABLATION STUDY RESULTS')
print('=' * 70)
print(f'{\"Configuration\":<25} {\"PPL Ratio\":>12} {\"KL Div\":>12} {\"ECE\":>12}')
print('-' * 70)

for variant in variants:
    filepath = results_dir / f'ablation_{variant}.json'
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
        ppl = data.get('perplexity_ratio', 'N/A')
        kl = data.get('kl_divergence', 'N/A')
        ece = data.get('student_ece', {}).get('ece', 'N/A') if 'student_ece' in data else 'N/A'

        ppl_str = f'{ppl:.4f}' if isinstance(ppl, (int, float)) else str(ppl)
        kl_str = f'{kl:.4f}' if isinstance(kl, (int, float)) else str(kl)
        ece_str = f'{ece:.4f}' if isinstance(ece, (int, float)) else str(ece)

        label = {'baseline': 'Baseline', 'uncertainty': '+Uncertainty',
                 'calibration': '+Calibration', 'both': '+Both'}[variant]
        print(f'{label:<25} {ppl_str:>12} {kl_str:>12} {ece_str:>12}')
    else:
        print(f'{variant:<25} FILE NOT FOUND')

print('=' * 70)
print('Lower values are better for all metrics.')
print()
"
```

**Step 4: Run ablation notebook for visualizations**

```bash
jupyter nbconvert --execute notebooks/phase_0_ablation.ipynb --to html --output phase_0_ablation_results.html
```

**One-liner to run all steps** (for nohup):

```bash
nohup bash -c '
conda activate pepmlm
cd /home/ubuntu/storage1/protein-lm-distill
mkdir -p results

# Train variants
python scripts/train.py --temperature 2.0 --alpha 0.5 --n_layer 4 --n_head 4 --n_embd 512 --train_size_prop 0.1 --num_train_epochs 3 --learning_rate 1e-3 --use_uncertainty_weighting --output_dir ./models/ablation-uncertainty && \
python scripts/train.py --temperature 2.0 --alpha 0.5 --n_layer 4 --n_head 4 --n_embd 512 --train_size_prop 0.1 --num_train_epochs 3 --learning_rate 1e-3 --use_calibration_smoothing --smoothing_factor 0.1 --output_dir ./models/ablation-calibration && \
python scripts/train.py --temperature 2.0 --alpha 0.5 --n_layer 4 --n_head 4 --n_embd 512 --train_size_prop 0.1 --num_train_epochs 3 --learning_rate 1e-3 --use_uncertainty_weighting --use_calibration_smoothing --smoothing_factor 0.1 --output_dir ./models/ablation-both && \

# Evaluate all
BASELINE_MODEL=$(ls -td models/protgpt2-distilled-t2.0-a0.5-l4-h4-e512* 2>/dev/null | head -1)
python scripts/evaluate.py --student_model "$BASELINE_MODEL" --num_samples 100 --compute_ece --output results/ablation_baseline.json && \
python scripts/evaluate.py --student_model ./models/ablation-uncertainty --num_samples 100 --compute_ece --output results/ablation_uncertainty.json && \
python scripts/evaluate.py --student_model ./models/ablation-calibration --num_samples 100 --compute_ece --output results/ablation_calibration.json && \
python scripts/evaluate.py --student_model ./models/ablation-both --num_samples 100 --compute_ece --output results/ablation_both.json && \

echo "Phase 0 ablation study complete!"
' > phase0_ablation.log 2>&1 &
```

**Monitor progress**:
```bash
tail -f phase0_ablation.log
```

---

## Phase 1: Baseline Training (IN PROGRESS)

**Status**: Currently running via nohup (`training_baseline.log`)

**Note**: Current training uses smaller architectures for initial baseline. Will serve as comparison for enhanced models.

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
- [ ] Tiny model completed (will serve as baseline for ablation)
- [ ] Small model completed
- [ ] Medium model completed
- [ ] Instance auto-stops after completion

**Note**: After Phase 0 enhancements are implemented, we'll proceed directly to Phase 2 with the enhanced approach.

---

## Phase 2: Hyperparameter Sweeps (WITH ENHANCEMENTS)

**Important**: All hyperparameter sweeps will use the enhanced distillation method (uncertainty-aware + calibration-aware) from Phase 0.

### 2.1 Temperature Sweep (on Tiny model - fastest iteration)

**Rationale**: Temperature controls softness of probability distributions. Current HF models use T=10, new baseline uses T=2.0. Optimal T may differ with enhancements.

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

## Phase 5: Publication (ENHANCED)

### 5.1 Paper Structure

**Title**: "Uncertainty-Aware Knowledge Distillation for Autoregressive Protein Language Models"

**Alternative**: "Protein-Specific Knowledge Distillation: Uncertainty-Aware and Calibration-Conscious Compression of ProtGPT2"

1. **Abstract** (~250 words)
   - First systematic study + protein-specific enhancements
   - Key innovations: uncertainty weighting, calibration-aware distillation
   - Results: compression ratios, quality, calibration improvements

2. **Introduction** (1-2 pages)
   - Motivation, gap, contributions
   - Contribution 1: First causal protein LM distillation
   - Contribution 2: Uncertainty-aware position weighting
   - Contribution 3: Calibration-aware distillation
   - Contribution 4: Systematic evaluation framework

3. **Methods** (3-4 pages)
   - Standard KD framework (Hinton et al. 2015)
   - Enhancement #1: Uncertainty-aware weighting
   - Enhancement #2: Calibration-aware distillation
   - Model architectures, training protocol
   - Evaluation metrics (including ECE, reliability)

4. **Results** (3-4 pages)
   - Ablation study (key figure)
   - Hyperparameter analysis
   - Model comparison
   - Calibration analysis (ECE scores, reliability diagrams)
   - Generation quality
   - Inference speed

5. **Discussion** (1-2 pages)
   - Comparison with related work
   - Importance of protein-specific enhancements
   - Limitations and future work

6. **Conclusion** (~0.5 page)

### 5.2 Key Figures

| Figure | Content | Priority |
|--------|---------|----------|
| Fig 1 | **Ablation Study: Baseline vs. +Uncertainty vs. +Both** | ⭐⭐⭐ Must-have |
| Fig 2 | Temperature vs. Perplexity Ratio (with/without enhancements) | ⭐⭐⭐ Must-have |
| Fig 3 | Alpha vs. Perplexity Ratio (with/without enhancements) | ⭐⭐ Should-have |
| Fig 4 | Compression Ratio vs. Quality (Pareto frontier) | ⭐⭐⭐ Must-have |
| Fig 5 | **Calibration Analysis: ECE scores and reliability diagrams** | ⭐⭐⭐ Must-have |
| Fig 6 | AA Distribution: Teacher vs Student vs Natural | ⭐⭐ Should-have |
| Fig 7 | Inference Speed Benchmark | ⭐⭐ Should-have |
| Fig 8 | **Position-wise uncertainty heatmap and weight visualization** | ⭐⭐ Should-have |

### 5.3 Key Tables

| Table | Content | Priority |
|-------|---------|----------|
| Table 1 | Model configurations (params, layers, heads, embed) | ⭐⭐⭐ Must-have |
| Table 2 | **Ablation study metrics (baseline vs. enhancements)** | ⭐⭐⭐ Must-have |
| Table 3 | Evaluation metrics for all models (with size-dependent thresholds) | ⭐⭐⭐ Must-have |
| Table 4 | **Calibration metrics (ECE, MCE, Brier score)** | ⭐⭐⭐ Must-have |
| Table 5 | **Comparison with related work (DistilProtBERT, MTDP, SpiderGPT)** | ⭐⭐ Should-have |
| Table 6 | Inference speed and memory benchmarks | ⭐⭐ Should-have |

### 5.4 Publication Strategy (ENHANCED - Higher-Tier Venues)

**With protein-specific enhancements, we can target top-tier venues:**

**Step 1: bioRxiv Preprint** (Immediate upon completion)
- Submit to establish priority
- Get DOI for HuggingFace model cards
- Enable community feedback

**Step 2: Journal Submission** (After preprint)

**Primary Target: Nature Communications**
- Rationale: Methodological novelty + protein-specific innovations
- Original ProtGPT2 was published in Nature Communications
- Timeline: 3-4 months review

**Backup Targets** (in priority order):
1. **PNAS** (2-3 months, computational biology focus)
2. **Cell Systems** (3-4 months, computational methods focus)
3. **Bioinformatics** (2-3 months, safe choice, highly cited)
4. **PLOS Computational Biology** (2-3 months, open access)

**Alternative: ML Venues**
- ICML/NeurIPS computational biology track (6-8 months)

**Checklist**:
- [ ] Paper draft complete with ablation studies
- [ ] All figures generated (especially Fig 1: ablation, Fig 5: calibration)
- [ ] All tables completed (especially Table 2: ablation, Table 4: calibration)
- [ ] Internal review done
- [ ] bioRxiv submitted
- [ ] **Nature Communications** submitted (primary choice)
- [ ] Backup venue ready if rejected

### 5.5 Existing Resources

- **docs/METHODS.md** - Complete mathematical framework with LaTeX, ready for paper Methods section
- **W&B Dashboard** - https://wandb.ai/ewijaya/PROTGPT2_DISTILLATION - Training curves and metrics

---

## Success Criteria

### Must-Have (MVP)
- [x] **Phase 0: Enhancements implemented and validated**
  - [x] Uncertainty-aware position weighting functional
  - [x] Calibration-aware distillation functional
  - [ ] Ablation study completed (baseline vs. enhanced) - infrastructure ready, pending training
  - [x] ECE metric implemented and tested
- [ ] Three models trained (Tiny, Small, Medium) **with enhancements**
- [ ] Models match HF architectures (512E, 768E, 1024E)
- [ ] Perplexity ratio meets size-dependent thresholds
- [ ] **Calibration improvements demonstrated** (ECE scores)
- [ ] Evaluation metrics documented
- [ ] HuggingFace repos updated with enhanced models

### Should-Have
- [ ] Optimal hyperparameters from sweeps (with enhancements)
- [ ] pLDDT evaluation completed
- [x] Position-wise uncertainty visualization - implemented in `notebooks/phase_0_ablation.ipynb`
- [x] Reliability diagrams for calibration - implemented in `notebooks/phase_0_ablation.ipynb`
- [ ] Paper draft complete with ablation studies

### Nice-to-Have
- [ ] Paper submitted to **Nature Communications** (upgraded from Bioinformatics)
- [ ] Comparison with SpiderGPT and other distillation methods
- [ ] Interactive demo on HuggingFace Spaces

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
