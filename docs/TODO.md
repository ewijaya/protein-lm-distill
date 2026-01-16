# Project TODO List

**Updated**: January 17, 2026 (training running)

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

**Key Finding: Synergistic Effect**

Individual enhancements HURT performance, but together they dramatically improve:
- **53% reduction in PPL ratio** (18.95 → 8.93)
- **27% reduction in KL divergence** (2.23 → 1.62)
- **21% improvement in calibration (ECE)** (0.274 → 0.216)

**Interpretation**: Uncertainty weighting and calibration smoothing compensate for each other's weaknesses. This synergistic effect is the core novel finding for publication.

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

### 3.1 HuggingFace Baseline Comparison (COMPLETE ✅)

**Completed**: January 16, 2026

| Model | Architecture | Output | Status |
|-------|--------------|--------|--------|
| `littleworth/protgpt2-distilled-tiny` | 4L/4H/512E | `results/eval_hf_tiny_old.json` | ✅ Complete |

### 3.2 Ablation vs HF-tiny Comparison

**Key Finding**: +Both (256E) outperforms HF-tiny (512E) on calibration despite smaller size.

| Metric | +Both (4L/4H/256E) | HF-tiny (4L/4H/512E) | Winner |
|--------|-------------------|----------------------|--------|
| PPL Ratio | 8.93 | 5.35 | HF-tiny (larger model) |
| KL Divergence | **1.62** | 2.92 | **+Both (44% better)** |
| Student ECE | **0.216** | 0.398 | **+Both (46% better)** |
| KL from Natural | **0.024** | 0.042 | **+Both (43% better)** |
| Compression | 47.5x | 19.9x | +Both (2.4x smaller) |

**Interpretation**: HF-tiny's lower PPL ratio is due to 2x larger embedding (512 vs 256). On calibration and distributional fidelity, +Both is substantially better. To make a fair comparison, we need to train +Both on HF-matching architectures.

### 3.3 HF-Matching Architecture Training (RUNNING ⏳)

**Status**: Training pipeline launched January 16, 2026.

**Objective**: Train +Both on same architectures as published HF models for direct comparison.

| Size | Architecture | Output Dir | Status |
|------|--------------|------------|--------|
| Tiny | 4L/4H/512E | `./models/synergy-tiny` | ⏳ Running |
| Small | 6L/8H/768E | `./models/synergy-small` | ⏳ Queued |
| Medium | 12L/16H/1024E | `./models/synergy-medium` | ⏳ Queued |

### 3.4 Size-Dependent Thresholds

| Model | Params | % of Teacher | PPL Ratio Threshold |
|-------|--------|--------------|---------------------|
| Tiny | ~39M | 5% | < 3.0 |
| Small | ~82M | 11% | < 2.5 |
| Medium | ~200M | 27% | < 2.0 |

### 3.5 Checklist

- [x] Ablation variants evaluated (4/4 complete)
- [x] HF-tiny baseline evaluated
- [x] Comparison table generated (ablation vs HF-tiny)
- [x] Publication viability assessed: **Strong** (synergistic effect is novel)
- [ ] +Both trained on HF-matching architectures (Tiny/Small/Medium)
- [ ] +Both HF-matching models evaluated
- [ ] Final comparison table (new +Both vs old HF models)

---

## Phase 4: HuggingFace Update (PENDING)

### Upload Commands

```bash
python tools/upload_to_hf.py --model_dir ./models/BEST_TINY --repo_id littleworth/protgpt2-distilled-tiny
python tools/upload_to_hf.py --model_dir ./models/BEST_SMALL --repo_id littleworth/protgpt2-distilled-small
python tools/upload_to_hf.py --model_dir ./models/BEST_MEDIUM --repo_id littleworth/protgpt2-distilled-medium
```

- [ ] Tiny uploaded
- [ ] Small uploaded
- [ ] Medium uploaded
- [ ] Model cards updated
- [ ] Post-upload verification

---

## Phase 5: Publication (PENDING)

### Core Paper Story: Synergistic Distillation

**Central Finding**: Uncertainty-aware and calibration-aware distillation exhibit a **synergistic effect** — each method individually degrades performance, but their combination yields dramatic improvement.

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

1. "Synergistic Effects of Uncertainty-Aware and Calibration-Conscious Distillation for Protein Language Models"
2. "Uncertainty-Aware Knowledge Distillation for Autoregressive Protein Language Models"
3. "Protein-Specific Knowledge Distillation: Uncertainty-Aware and Calibration-Conscious Compression of ProtGPT2"

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

## Immediate Next Actions

### Train +Both on HF-Matching Architectures (RUNNING ⏳)

**Status**: Training pipeline launched January 16, 2026. Instance will auto-shutdown on completion.

**Objective**: Enable direct comparison with published HF models by training +Both at same sizes.

**Pastable command** (sequential, with auto-shutdown):

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

**Monitor progress**:
```bash
tail -f nohup_synergy_training.out
```

### After Training Completes (WHEN NOHUP FINISHES)

#### Step 1: Verify Training Success

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

#### Step 2: Compare All Results

```bash
python -c "
import json
models = {
    'Ablation +Both (256E)': 'results/ablation_both.json',
    'HF-tiny old (512E)': 'results/eval_hf_tiny_old.json',
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

#### Step 3: Evaluate Results and Decide Next Steps

**Expected outcomes** (synergy-tiny vs HF-tiny at same 512E size):

| Scenario | PPL Ratio | KL Div | ECE | Action |
|----------|-----------|--------|-----|--------|
| **Best case** | Synergy < HF | Synergy < HF | Synergy < HF | Strong paper: new method beats old at same size |
| **Good case** | Synergy ≈ HF | Synergy < HF | Synergy < HF | Good paper: comparable PPL but better calibration |
| **Okay case** | Synergy > HF | Synergy < HF | Synergy < HF | Paper focuses on calibration benefits |
| **Investigate** | Synergy > HF | Synergy > HF | Synergy > HF | Check training logs, may need hyperparameter tuning |

#### Step 4: Update Documentation

```bash
# Update TODO.md with results (replace placeholders with actual values)
# Update Phase 3.3 table with completion status
# Update checklist items
```

#### Step 5: Proceed to Next Phase

**If results are good (Scenario 1-3):**

1. **Update TODO.md** - Mark Phase 3 complete
2. **Create results summary** - Add final comparison table to lesson-learned doc
3. **Proceed to Phase 4** - Upload to HuggingFace:
   ```bash
   python tools/upload_to_hf.py --model_dir ./models/synergy-tiny --repo_id littleworth/protgpt2-distilled-tiny
   python tools/upload_to_hf.py --model_dir ./models/synergy-small --repo_id littleworth/protgpt2-distilled-small
   python tools/upload_to_hf.py --model_dir ./models/synergy-medium --repo_id littleworth/protgpt2-distilled-medium
   ```
4. **Proceed to Phase 5** - Start paper draft

**If results need investigation (Scenario 4):**

1. Check W&B dashboard for training curves: https://wandb.ai/ewijaya/PROTGPT2_DISTILLATION
2. Compare loss curves between synergy models and old HF models
3. Consider re-running with different learning rates or longer training

#### Step 6: Git Commit Results

```bash
cd /home/ubuntu/storage1/protein-lm-distill
git add results/eval_synergy_*.json nohup_synergy_training.out
git commit -m "feat: add synergy model evaluation results (tiny/small/medium)"
git push origin HEAD && git push github HEAD
```

### Publication Readiness Checklist

- [x] Ablation study complete with synergistic effect finding
- [x] HF-tiny comparison complete (shows +Both wins on calibration/KL)
- [x] Lesson-learned document created (`docs/Lesson-Learned-Phase0-Ablation-Synergy-2026-01-16-2337.md`)
- [ ] +Both trained on HF-matching architectures
- [ ] Final comparison showing +Both beats HF models at same size
- [ ] Mechanistic explanation drafted

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

New synergy models use simplified names: `synergy-tiny`, `synergy-small`, `synergy-medium`

### W&B Dashboard

https://wandb.ai/ewijaya/PROTGPT2_DISTILLATION

### Key Results Files

| File | Description |
|------|-------------|
| `results/ablation_baseline.json` | Baseline (no enhancements) |
| `results/ablation_both.json` | +Both (synergistic, 256E) |
| `results/eval_hf_tiny_old.json` | Published HF-tiny (512E) |
| `results/eval_synergy_tiny.json` | New +Both (512E) - pending |
| `results/eval_synergy_small.json` | New +Both (768E) - pending |
| `results/eval_synergy_medium.json` | New +Both (1024E) - pending |
