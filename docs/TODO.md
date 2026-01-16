# Project TODO List

**Updated**: January 16, 2026

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

### 3.1 HuggingFace Baseline Comparison

**Status**: Running

| Model | Architecture | Output | Status |
|-------|--------------|--------|--------|
| `littleworth/protgpt2-distilled-tiny` | 4L/4H/512E | `results/eval_hf_tiny_old.json` | ⏳ Running |
| `littleworth/protgpt2-distilled-small` | 6L/8H/768E | — | ⏸️ Not needed (different arch) |
| `littleworth/protgpt2-distilled-medium` | 12L/16H/1024E | — | ⏸️ Not needed (different arch) |

### 3.2 Final Comparison Table (To Be Generated)

Will compare:
- Baseline (T=2.0, α=0.5, no enhancements)
- +Both (T=2.0, α=0.5, uncertainty + calibration)
- HF-tiny (T=10, α=0.1, old training)

### 3.3 Size-Dependent Thresholds

### 3.4 Size-Dependent Thresholds

| Model | Params | % of Teacher | PPL Ratio Threshold |
|-------|--------|--------------|---------------------|
| Tiny | ~39M | 5% | < 3.0 |
| Small | ~82M | 11% | < 2.5 |
| Medium | ~200M | 27% | < 2.0 |

**Note**: Current +Both achieves PPL ratio 8.93 on tiny architecture (4L/4H/256E). This is above threshold but represents 53% improvement over baseline. The comparison with HF-tiny (4L/4H/512E, T=10) will contextualize this result.

### 3.5 Checklist

- [x] Ablation variants evaluated (4/4 complete)
- [ ] HF-tiny baseline evaluated (running)
- [ ] Comparison table generated
- [ ] Publication viability assessed

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

### Phase 3: HuggingFace Model Comparison (IN PROGRESS ⏳)

**Objective**: Compare +Both results against existing published HF models to validate improvement.

**Currently running**:
```bash
python scripts/evaluate.py \
    --student_model littleworth/protgpt2-distilled-tiny \
    --num_samples 100 --compute_ece \
    --output results/eval_hf_tiny_old.json
```

**Why HF-tiny only**: Ablation models are 4L/4H/256E, closest to HF-tiny (4L/4H/512E). Small/medium have different architectures.

### After HF Comparison Completes

1. **Compare results**:
   ```bash
   python -c "
   import json
   models = {
       '+Both (new)': 'results/ablation_both.json',
       'HF-tiny (old)': 'results/eval_hf_tiny_old.json'
   }
   print('Model                  PPL Ratio    KL Div       ECE')
   print('-' * 60)
   for name, path in models.items():
       try:
           d = json.load(open(path))
           ppl = d.get('perplexity_ratio', 'N/A')
           kl = d.get('kl_divergence', 'N/A')
           ece = d.get('student_ece', {}).get('ece', 'N/A') if 'student_ece' in d else 'N/A'
           print(f'{name:<22} {ppl:>10.4f} {kl:>10.4f} {ece:>10.4f}')
       except: pass
   "
   ```

2. **Decision point**:
   - If +Both < HF-tiny → Strong paper story (new method beats published model)
   - If +Both > HF-tiny → Need to investigate (architecture difference: 256E vs 512E)

### Publication Readiness Checklist

- [x] Ablation study complete with synergistic effect finding
- [ ] HF model comparison complete
- [ ] Mechanistic explanation drafted
- [ ] Consider: Replicate ablation on larger architectures (optional for stronger paper)

---

## Quick Reference

### Monitoring Current Evaluation

```bash
tail -f nohup_eval.out
```

### Model Naming Convention

```
protgpt2-distilled-t{temp}-a{alpha}-l{layers}-h{heads}-e{embed}-p{prop}-lr{lr}.uniprot
```

### W&B Dashboard

https://wandb.ai/ewijaya/PROTGPT2_DISTILLATION
