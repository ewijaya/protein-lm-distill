# Project TODO List

**Updated**: January 16, 2026

---

## Overview

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ⏳ Evaluation Running | Methodological enhancements + ablation study |
| **Phase 1** | ✅ Complete | Baseline training (4 model sizes) |
| **Phase 2** | ⏸️ Pending | Hyperparameter sweeps |
| **Phase 3** | ⏸️ Pending | Comprehensive evaluation |
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

### 0.3 Ablation Evaluation (IN PROGRESS ⏳)

**Started**: January 16, 2026

**Running command**:
```bash
nohup bash -c 'mkdir -p results && \
python scripts/evaluate.py --student_model ./models/protgpt2-distilled-t2.0-a0.5-l4-h4-e256-p0.1-lr1e-03.uniprot --num_samples 100 --compute_ece --output results/ablation_baseline.json && \
python scripts/evaluate.py --student_model ./models/ablation-uncertainty --num_samples 100 --compute_ece --output results/ablation_uncertainty.json && \
python scripts/evaluate.py --student_model ./models/ablation-calibration --num_samples 100 --compute_ece --output results/ablation_calibration.json && \
python scripts/evaluate.py --student_model ./models/ablation-both --num_samples 100 --compute_ece --output results/ablation_both.json && \
/home/ubuntu/bin/stopinstance' > nohup_eval.out 2>&1 &
```

**Monitor**:
```bash
tail -f nohup_eval.out
```

**Evaluation variants**:
| Variant | Model | Output | Status |
|---------|-------|--------|--------|
| Baseline | `protgpt2-distilled-t2.0-a0.5-l4-h4-e256-p0.1-lr1e-03.uniprot` | `results/ablation_baseline.json` | ⏳ Running |
| +Uncertainty | `ablation-uncertainty` | `results/ablation_uncertainty.json` | ⏸️ Queued |
| +Calibration | `ablation-calibration` | `results/ablation_calibration.json` | ⏸️ Queued |
| +Both | `ablation-both` | `results/ablation_both.json` | ⏸️ Queued |

**Instance auto-stops after completion.**

**Checklist**:
- [x] Ablation training complete
- [ ] All 4 variants evaluated
- [ ] Ablation results table generated
- [ ] Best enhancement configuration identified

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

## Phase 2: Hyperparameter Sweeps (PENDING)

**Prerequisites**: Phase 0 ablation results

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

## Phase 3: Comprehensive Evaluation (PENDING)

### Size-Dependent Thresholds

| Model | Params | % of Teacher | PPL Ratio Threshold |
|-------|--------|--------------|---------------------|
| Tiny | ~39M | 5% | < 3.0 |
| Small | ~82M | 11% | < 2.5 |
| Medium | ~200M | 27% | < 2.0 |

### Evaluation Commands

```bash
# Evaluate final models
for model_dir in models/protgpt2-distilled-*-FINAL*; do
    python scripts/evaluate.py \
        --student_model "$model_dir" \
        --num_samples 200 --compute_ece \
        --output "results/eval_$(basename $model_dir).json"
done

# Compare with existing HF models
python scripts/evaluate.py --student_model littleworth/protgpt2-distilled-tiny --num_samples 200 --output results/eval_hf_tiny_old.json
python scripts/evaluate.py --student_model littleworth/protgpt2-distilled-small --num_samples 200 --output results/eval_hf_small_old.json
python scripts/evaluate.py --student_model littleworth/protgpt2-distilled-medium --num_samples 200 --output results/eval_hf_medium_old.json
```

- [ ] All new models evaluated
- [ ] Existing HF models evaluated
- [ ] Comparison table generated
- [ ] Best models identified

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

### Paper Title Options

1. "Uncertainty-Aware Knowledge Distillation for Autoregressive Protein Language Models"
2. "Protein-Specific Knowledge Distillation: Uncertainty-Aware and Calibration-Conscious Compression of ProtGPT2"

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

### When Instance Restarts (After Evaluation Completes)

1. **Check evaluation completed**:
   ```bash
   ls -la results/ablation_*.json
   tail nohup_eval.out
   ```

2. **Generate ablation results table**:
   ```bash
   python -c "
   import json
   from pathlib import Path

   variants = ['baseline', 'uncertainty', 'calibration', 'both']
   print('Configuration          PPL Ratio    KL Div       ECE')
   print('-' * 60)
   for v in variants:
       f = Path(f'results/ablation_{v}.json')
       if f.exists():
           d = json.load(open(f))
           ppl = d.get('perplexity_ratio', 'N/A')
           kl = d.get('kl_divergence', 'N/A')
           ece = d.get('student_ece', {}).get('ece', 'N/A') if 'student_ece' in d else 'N/A'
           print(f'{v:<22} {ppl:>10.4f} {kl:>10.4f} {ece:>10.4f}')
   "
   ```

3. **Decide on enhancements** based on results, then proceed to Phase 2

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
