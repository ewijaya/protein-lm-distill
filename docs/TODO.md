# Project TODO List

**Updated**: February 9, 2026

---

## Overview

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ✅ Complete | Methodological enhancements + ablation study |
| **Phase 1** | ✅ Complete | Baseline training (4 model sizes) |
| **Phase 2** | ⏭️ Skip | Hyperparameter sweeps (ablation validates method) |
| **Phase 3** | ⏳ In Progress | Comprehensive evaluation |
| **Phase 4** | ⏸️ Pending | HuggingFace update |
| **Phase 5** | ⏸️ Pending | Publication |

---

## Phase 0: Methodological Enhancements

### 0.1 Implementation (COMPLETE ✅)

- [x] Uncertainty-aware position weighting (`src/distillation.py`)
- [x] Calibration-aware distillation (`src/distillation.py`)
- [x] ECE metric implementation (`scripts/evaluate.py`)
- [x] Ablation notebook (`notebooks/phase_0_ablation.ipynb`)
- [x] Documentation (`docs/METHODS.md`)

### 0.2 Ablation Training (COMPLETE ✅)

**Completed**: January 16, 2026

**Training variants**:
| Variant | Config | Output Dir | Status |
|---------|--------|------------|--------|
| +Uncertainty | `--use_uncertainty_weighting` | `./models/ablation-uncertainty` | ✅ Complete |
| +Calibration | `--use_calibration_smoothing` | `./models/ablation-calibration` | ✅ Complete |
| +Both | Both flags | `./models/ablation-both` | ✅ Complete |

### 0.3 Ablation Evaluation (COMPLETE ✅)

**Completed**: January 16, 2026

**Results Summary**:

| Configuration | PPL Ratio | KL Divergence | Student ECE | KL from Natural |
|--------------|-----------|---------------|-------------|-----------------|
| Baseline | 18.95 | 2.23 | 0.274 | 0.030 |
| +Uncertainty | 36.89 | 2.87 | 0.325 | 0.020 |
| +Calibration | 39.64 | 3.00 | 0.319 | 0.040 |
| **+Both** | **8.93** | **1.62** | **0.216** | 0.024 |

**Key Finding: Complementary Effect**

Individual enhancements HURT performance, but together they dramatically improve:
- **53% reduction in PPL ratio** (18.95 → 8.93)
- **27% reduction in KL divergence** (2.23 → 1.62)
- **21% improvement in calibration (ECE)** (0.274 → 0.216)

**Interpretation**: Uncertainty weighting and calibration smoothing compensate for each other's weaknesses. This complementary effect is the core novel finding for publication.

**Checklist**:
- [x] Ablation training complete
- [x] All 4 variants evaluated
- [x] Ablation results table generated
- [x] Best enhancement configuration identified: **+Both**

---

## Phase 1: Baseline Training (COMPLETE ✅)

**Completed**: January 7, 2026

### Trained Models

| Model | Config | Directory | Date |
|-------|--------|-----------|------|
| XS | 2L/2H/128E | `protgpt2-distilled-t2.0-a0.5-l2-h2-e128-p0.001-lr1e-03.uniprot` | Dec 25 |
| Tiny | 4L/4H/256E | `protgpt2-distilled-t2.0-a0.5-l4-h4-e256-p0.1-lr1e-03.uniprot` | Dec 29 |
| Small | 6L/8H/512E | `protgpt2-distilled-t2.0-a0.5-l6-h8-e512-p0.1-lr5e-04.uniprot` | Jan 1 |
| Medium | 12L/12H/768E | `protgpt2-distilled-t2.0-a0.5-l12-h12-e768-p0.1-lr1e-04.uniprot` | Jan 7 |

### Note: Architecture Mismatch with HuggingFace

Current models don't match HF architectures. Final training will use:

| Size | Current | HF Target |
|------|---------|-----------|
| Tiny | l4-h4-**e256** | l4-h4-**e512** |
| Small | l6-h8-**e512** | l6-h8-**e768** |
| Medium | l12-h12-**e768** | l12-h16-**e1024** |

---

## Phase 2: Hyperparameter Sweeps (OPTIONAL ⏭️)

**Status**: Can be skipped if Phase 0 ablation validates the method

**Rationale for skipping**:
- Core novelty is uncertainty-aware + calibration-aware distillation (validated by ablation)
- T=2.0 and α=0.5 are well-established defaults from Hinton et al. (2015)
- Can cite prior work: *"We use T=2.0 and α=0.5 following standard practice [Hinton 2015]"*
- If reviewers request sensitivity analysis, add minimal sweep (T ∈ {1,2,4}) during revision

**If you choose to run sweeps**:

### 2.1 Temperature Sweep

```bash
for temp in 1.0 2.0 4.0 6.0 8.0 10.0; do
    python scripts/train.py \
        --temperature $temp --alpha 0.5 \
        --n_layer 4 --n_head 4 --n_embd 256 \
        --train_size_prop 0.1 --learning_rate 1e-3 \
        --use_uncertainty_weighting --use_calibration_smoothing
done
```

- [ ] T=1.0
- [ ] T=2.0
- [ ] T=4.0
- [ ] T=6.0
- [ ] T=8.0
- [ ] T=10.0
- [ ] Best temperature identified: T=___

### 2.2 Alpha Sweep

```bash
BEST_TEMP=X.X
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    python scripts/train.py \
        --temperature $BEST_TEMP --alpha $alpha \
        --n_layer 4 --n_head 4 --n_embd 256 \
        --train_size_prop 0.1 --learning_rate 1e-3 \
        --use_uncertainty_weighting --use_calibration_smoothing
done
```

- [ ] α=0.1
- [ ] α=0.3
- [ ] α=0.5
- [ ] α=0.7
- [ ] α=0.9
- [ ] Best alpha identified: α=___

### 2.3 Final Training (HF-Matching Architectures)

```bash
nohup bash -c '
python scripts/train.py --temperature $BEST_T --alpha $BEST_A --n_layer 4 --n_head 4 --n_embd 512 --train_size_prop 0.1 --learning_rate 1e-3 --use_uncertainty_weighting --use_calibration_smoothing && \
python scripts/train.py --temperature $BEST_T --alpha $BEST_A --n_layer 6 --n_head 8 --n_embd 768 --train_size_prop 0.1 --learning_rate 5e-4 --use_uncertainty_weighting --use_calibration_smoothing && \
python scripts/train.py --temperature $BEST_T --alpha $BEST_A --n_layer 12 --n_head 16 --n_embd 1024 --train_size_prop 0.1 --learning_rate 1e-4 --use_uncertainty_weighting --use_calibration_smoothing && \
/home/ubuntu/bin/stopinstance
' > training_final.log 2>&1 &
```

- [ ] Tiny (4L/4H/512E) trained
- [ ] Small (6L/8H/768E) trained
- [ ] Medium (12L/16H/1024E) trained

---

## Phase 3: Comprehensive Evaluation (IN PROGRESS ⏳)

### 3.1 Ablation Study Results (COMPLETE ✅) - Core Paper Finding

**This is the main publication result.** Comparing combined method vs standard KD baseline (Hinton 2015).

| Configuration | PPL Ratio | KL Divergence | Student ECE | vs Baseline |
|--------------|-----------|---------------|-------------|-------------|
| Baseline (standard KD) | 18.95 | 2.23 | 0.274 | — |
| +Uncertainty only | 36.89 | 2.87 | 0.325 | worse |
| +Calibration only | 39.64 | 3.00 | 0.319 | worse |
| **+Both (combined)** | **8.93** | **1.62** | **0.216** | **53% better** |

**Key Finding**: Individual enhancements hurt, but together they dramatically improve. This complementary effect is the core novel contribution.

### 3.2 Publication-Quality Model Training (COMPLETE ✅)

**Status**: All three models trained and evaluated.

**Objective**: Train publication-quality models using combined method (+Both) to replace old HuggingFace models.

**Note**: Old HF models (`littleworth/protgpt2-distilled-*`) were suboptimally trained. These new models will replace them after paper publication.

| Size | Architecture | Output Dir | Replaces | Status |
|------|--------------|------------|----------|--------|
| Tiny | 4L/4H/512E | `./models/synergy-tiny` | `littleworth/protgpt2-distilled-tiny` | ✅ Complete (Jan 20) |
| Small | 6L/8H/768E | `./models/synergy-small` | `littleworth/protgpt2-distilled-small` | ✅ Complete (Jan 23) |
| Medium | 12L/16H/1024E | `./models/synergy-medium` | `littleworth/protgpt2-distilled-medium` | ✅ Complete (Jan 28) |

**Results Summary**:

| Model | PPL Ratio | KL Div | ECE | Notes |
|-------|-----------|--------|-----|-------|
| Synergy-tiny | 129.78 | 4.17 | 0.349 | Poor results - potential training issue |
| Synergy-small | 7.05 | 1.69 | 0.259 | Good improvement over baseline |
| Synergy-medium | 5.16 | 1.34 | 0.189 | Best results - clear scaling benefit |

**Note on Medium model**: Required `--batch_size 4 --gradient_accumulation 8` to fit in 22GB GPU memory.

**Monitor progress**:
```bash
tail -f nohup_synergy_training.out
```

<details>
<summary><strong>Training Command (for reference)</strong></summary>

```bash
nohup bash -c '
cd /home/ubuntu/storage1/protein-lm-distill && \
echo "=== Starting Synergy Training Pipeline ===" && \
echo "Start time: $(date)" && \
\
echo "=== [1/6] Training Tiny (4L/4H/512E) ===" && \
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 4 --n_head 4 --n_embd 512 \
    --train_size_prop 0.1 --learning_rate 1e-3 \
    --use_uncertainty_weighting --use_calibration_smoothing \
    --output_dir ./models/synergy-tiny && \
\
echo "=== [2/6] Evaluating Tiny ===" && \
python scripts/evaluate.py \
    --student_model ./models/synergy-tiny \
    --num_samples 100 --compute_ece \
    --output results/eval_synergy_tiny.json && \
\
echo "=== [3/6] Training Small (6L/8H/768E) ===" && \
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 6 --n_head 8 --n_embd 768 \
    --train_size_prop 0.1 --learning_rate 5e-4 \
    --use_uncertainty_weighting --use_calibration_smoothing \
    --output_dir ./models/synergy-small && \
\
echo "=== [4/6] Evaluating Small ===" && \
python scripts/evaluate.py \
    --student_model ./models/synergy-small \
    --num_samples 100 --compute_ece \
    --output results/eval_synergy_small.json && \
\
echo "=== [5/6] Training Medium (12L/16H/1024E) ===" && \
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 12 --n_head 16 --n_embd 1024 \
    --train_size_prop 0.1 --learning_rate 1e-4 \
    --batch_size 4 --gradient_accumulation 8 \
    --use_uncertainty_weighting --use_calibration_smoothing \
    --output_dir ./models/synergy-medium && \
\
echo "=== [6/6] Evaluating Medium ===" && \
python scripts/evaluate.py \
    --student_model ./models/synergy-medium \
    --num_samples 100 --compute_ece \
    --output results/eval_synergy_medium.json && \
\
echo "=== Pipeline Complete ===" && \
echo "End time: $(date)" && \
/home/ubuntu/bin/stopinstance
' > nohup_synergy_training.out 2>&1 &
```

</details>

#### When Training Completes

**Step 1: Verify Training Success**
```bash
# Check all models exist
ls -la models/synergy-tiny models/synergy-small models/synergy-medium

# Check all evaluation results exist
ls -la results/eval_synergy_*.json

# Check training log for errors
grep -i "error\|exception\|failed" nohup_synergy_training.out || echo "No errors found"

# View training completion time
tail -20 nohup_synergy_training.out
```

**Step 2: View Model Results**
```bash
python -c "
import json
models = {
    'Ablation baseline (256E)': 'results/ablation_baseline.json',
    'Ablation +Both (256E)': 'results/ablation_both.json',
    'Synergy-tiny (512E)': 'results/eval_synergy_tiny.json',
    'Synergy-small (768E)': 'results/eval_synergy_small.json',
    'Synergy-medium (1024E)': 'results/eval_synergy_medium.json',
}
print(f'{\"Model\":<26} {\"PPL Ratio\":>10} {\"KL Div\":>10} {\"ECE\":>10}')
print('-' * 60)
for name, path in models.items():
    try:
        d = json.load(open(path))
        ppl = d.get('perplexity_ratio', float('nan'))
        kl = d.get('kl_divergence', float('nan'))
        ece = d.get('student_ece', {}).get('ece', float('nan'))
        print(f'{name:<26} {ppl:>10.2f} {kl:>10.4f} {ece:>10.4f}')
    except FileNotFoundError:
        print(f'{name:<26} {\"---\":>10} {\"---\":>10} {\"---\":>10}')
"
```

**Step 3: Check Quality Thresholds**

Models should improve as size increases (more capacity):

| Model | Target PPL Ratio | Target ECE | Ready for HF? |
|-------|------------------|------------|---------------|
| Synergy-tiny | < 5.0 | < 0.30 | Fill in after eval |
| Synergy-small | < 4.0 | < 0.25 | Fill in after eval |
| Synergy-medium | < 3.0 | < 0.20 | Fill in after eval |

**Step 4: Proceed to Next Phase**

If models meet quality thresholds:
1. Update this TODO.md - Mark Phase 3 complete, fill in results
2. Proceed to Phase 4 - Upload new models to HuggingFace (replaces old models)
3. Proceed to Phase 5 - Write paper (ablation study is core finding)

If results need investigation:
1. Check W&B dashboard: https://wandb.ai/ewijaya/PROTGPT2_DISTILLATION
2. Check for training issues (loss curves, gradients)
3. Consider longer training or learning rate adjustments

**Step 5: Git Commit Results**
```bash
cd /home/ubuntu/storage1/protein-lm-distill
git add results/eval_synergy_*.json nohup_synergy_training.out
git commit -m "feat: add synergy model evaluation results (tiny/small/medium)"
git push origin HEAD && git push github HEAD
```

### 3.3 Matching Baseline Training (COMPLETE ✅)

**Completed**: February 9, 2026

**Objective**: Train baselines with matching architectures (no enhancements) for fair comparison against synergy models.

| Size | Architecture | Synergy Model | Baseline Model | Status |
|------|--------------|---------------|----------------|--------|
| Tiny | 4L/4H/512E | `synergy-tiny` | `baseline-tiny` | ✅ Complete (Jan 31) |
| Small | 6L/8H/768E | `synergy-small` | `baseline-small` | ✅ Complete (Feb 3) |
| Medium | 12L/16H/1024E | `synergy-medium` | `baseline-medium` | ✅ Complete (Feb 7) |

**Results (Baseline vs Synergy)**:

| Size | Baseline PPL Ratio | Synergy PPL Ratio | Baseline KL | Synergy KL | Baseline ECE | Synergy ECE |
|------|-------------------|-------------------|-------------|------------|--------------|-------------|
| Tiny | 39.91 | 129.78 | 3.16 | 4.17 | 0.345 | 0.349 |
| Small | 15.19 | 7.05 | 2.03 | 1.69 | 0.235 | 0.259 |
| Medium | 3.72 | 5.16 | 1.34 | 1.34 | 0.169 | 0.189 |

**Key Finding**: Synergy enhancements improve small-scale models (Small: 53% PPL improvement) but not at all scales. At medium scale, baseline outperforms synergy on PPL ratio (3.72 vs 5.16). This scale-dependent effect warrants investigation for the paper.

### 3.4 Size-Dependent Thresholds

| Model | Params | % of Teacher | PPL Ratio Threshold |
|-------|--------|--------------|---------------------|
| Tiny | ~39M | 5% | < 5.0 |
| Small | ~82M | 11% | < 4.0 |
| Medium | ~200M | 27% | < 3.0 |

### 3.5 Checklist

- [x] Ablation study complete (core paper finding: complementary effect)
- [x] Publication viability assessed: **Strong** (complementary effect is novel)
- [x] Lesson-learned document created (`docs/Lesson-Learned-Phase0-Ablation-Synergy-2026-01-16-2337.md`)
- [x] Synergy-tiny trained and evaluated (Jan 20)
- [x] Synergy-small trained and evaluated (Jan 23)
- [x] Synergy-medium trained and evaluated (Jan 28)
- [x] Matching baselines trained (`scripts/batch_baseline.sh`)
- [ ] Mechanistic explanation drafted for paper

---

## Phase 4: HuggingFace Update (PENDING)

**Objective**: Replace old suboptimal models with new models trained using the combined uncertainty + calibration method from the paper.

### Upload Commands

```bash
# Upload new models to replace old HF models
python tools/upload_to_hf.py --model_dir ./models/synergy-tiny --repo_id littleworth/protgpt2-distilled-tiny
python tools/upload_to_hf.py --model_dir ./models/synergy-small --repo_id littleworth/protgpt2-distilled-small
python tools/upload_to_hf.py --model_dir ./models/synergy-medium --repo_id littleworth/protgpt2-distilled-medium
```

### Checklist

- [ ] Synergy-tiny uploaded (replaces old tiny)
- [ ] Synergy-small uploaded (replaces old small)
- [ ] Synergy-medium uploaded (replaces old medium)
- [ ] Model cards updated with paper citation and training details
- [ ] Add note: "Trained with uncertainty-aware calibration-conscious distillation"
- [ ] Post-upload verification

---

## Phase 5: Publication (PENDING)

### Core Paper Story: Complementary Distillation

**Central Finding**: Uncertainty-aware and calibration-aware distillation exhibit a **complementary effect** — each method individually degrades performance, but their combination yields dramatic improvement.

**Why This Is Publishable**:
1. Counter-intuitive finding (individual components hurt, together they help)
2. Large effect size (53% PPL improvement, 27% KL improvement)
3. Novel contribution to protein LM compression literature
4. Mechanistic implications for understanding distillation dynamics

**Proposed Mechanistic Explanation** (to develop in Discussion):
- Uncertainty-only: Upweights high-entropy positions but amplifies teacher miscalibration
- Calibration-only: Smooths overconfident predictions but loses signal in uncertain regions
- Together: Calibration prevents noise amplification; uncertainty focuses smoothing where needed

### Paper Title Options

**Short and punchy (preferred):**
1. "Uncertainty Meets Calibration: Improved Distillation for Protein Language Models"
2. "Calibrated Distillation for Protein Language Models"
3. "When Uncertainty Meets Calibration in Protein LM Distillation"

**Method-focused:**
4. "Uncertainty-Aware Calibration-Conscious Distillation for Protein Language Models"
5. "Improved Protein Language Model Distillation via Joint Uncertainty and Calibration Awareness"

**Finding-focused:**
6. "Complementary Effects in Protein LM Distillation: Uncertainty Weighting Meets Calibration Smoothing"

### Key Deliverables

- [ ] Paper draft complete
- [ ] Ablation study figure (Fig 1)
- [ ] Calibration analysis figure (Fig 5)
- [ ] All tables completed
- [ ] bioRxiv submitted
- [ ] Nature Communications submitted

### Target Venues

| Priority | Venue | Timeline |
|----------|-------|----------|
| 1 | Nature Communications | 3-4 months |
| 2 | PNAS | 2-3 months |
| 3 | Bioinformatics | 2-3 months |

---

## Quick Reference

### Monitoring Current Training

```bash
tail -f nohup_synergy_training.out
```

### Model Naming Convention

```
protgpt2-distilled-t{temp}-a{alpha}-l{layers}-h{heads}-e{embed}-p{prop}-lr{lr}.uniprot
```

New models use temp names: `synergy-tiny`, `synergy-small`, `synergy-medium` (will be uploaded as `protgpt2-distilled-*`)

### Synergy Model Sizes & Aliases

| Size | Architecture | Directory | Alias/Symlink |
|------|--------------|-----------|---------------|
| **Nano** | 4L/4H/256E | `ablation-both` | `synergy-nano` → `ablation-both` |
| Tiny | 4L/4H/512E | `synergy-tiny` | — |
| Small | 6L/8H/768E | `synergy-small` | — |
| Medium | 12L/16H/1024E | `synergy-medium` | — |

> **Note**: `synergy-nano` is a symlink to `ablation-both` (the +Both ablation model with 256E architecture). This preserves ablation study naming while providing consistent synergy naming.

### W&B Dashboard

https://wandb.ai/ewijaya/PROTGPT2_DISTILLATION

### Key Results Files

| File | Description |
|------|-------------|
| `results/ablation_baseline.json` | Baseline - standard KD (paper comparison) |
| `results/ablation_uncertainty.json` | +Uncertainty only (paper comparison) |
| `results/ablation_calibration.json` | +Calibration only (paper comparison) |
| `results/ablation_both.json` | +Both combined = synergy-nano (paper comparison) |
| `results/eval_synergy_tiny.json` | Synergy-tiny for HF upload - complete |
| `results/eval_synergy_small.json` | Synergy-small for HF upload - complete |
| `results/eval_synergy_medium.json` | Synergy-medium for HF upload - complete |
