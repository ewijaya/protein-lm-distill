# Investigation Summary: Synergy Enhancement Regression Analysis

## Overview

This document summarizes the investigation into why the Phase 0 "synergy" enhancements (uncertainty-aware position weighting + calibration-aware label smoothing) produce inconsistent results across model scales. Two specific anomalies were investigated:

1. **Synergy-tiny regression**: synergy-tiny (512E) has PPL ratio 129.78, which is 3.25x worse than baseline-tiny (PPL ratio 39.91) despite using enhancements that should improve quality.
2. **Non-monotonic scaling**: synergy enhancements help dramatically at small scale (768E) but hurt at medium scale (1024E), suggesting the enhancements interact with model capacity in unexpected ways.

## Experimental Setup

All models use the same training data, tokenizer (ProtGPT2), optimizer (AdamW), linear LR schedule with no warmup, weight_decay=0.01, fp16=True, 3 epochs, and seed=42. The only variables are model architecture, learning rate, and whether Phase 0 enhancements are enabled.

### Model Configurations

| Model | n_embd | n_layer | n_head | LR | Batch (eff.) | Enhancements |
|-------|--------|---------|--------|----|-------------|-------------|
| baseline-tiny | 512 | 4 | 4 | 1e-3 | 32 | OFF |
| synergy-tiny | 512 | 4 | 4 | 1e-3 | 32 | ON |
| baseline-small | 768 | 6 | 8 | 5e-4 | 32 | OFF |
| synergy-small | 768 | 6 | 8 | 5e-4 | 32 | ON |
| baseline-medium | 1024 | 12 | 16 | 1e-4 | 32 | OFF |
| synergy-medium | 1024 | 12 | 16 | 1e-4 | 32 | ON |
| ablation-both | 256 | 4 | 4 | 1e-3 | 32 | ON |

### Ablation Study (all at 256E)

| Model | Enhancements | PPL Ratio | KL Div |
|-------|-------------|-----------|--------|
| ablation-baseline | None | 18.95 | 2.227 |
| ablation-uncertainty | Uncertainty only | 36.89 | 2.869 |
| ablation-calibration | Calibration only | 39.64 | 3.002 |
| ablation-both | Both | **8.93** | **1.623** |

## Key Metrics Summary

| Model | PPL Ratio | KL Div | Student ECE | Train Loss | Student kl_from_natural |
|-------|-----------|--------|-------------|------------|------------------------|
| ablation-both (256E) | **8.93** | **1.623** | 0.216 | 4.634 | 0.024 |
| synergy-small (768E) | **7.05** | **1.686** | 0.259 | 4.476 | 0.012 |
| baseline-medium (1024E) | **3.72** | **1.340** | 0.169 | 4.926 | 0.023 |
| synergy-medium (1024E) | 5.16 | 1.344 | 0.189 | 4.479 | 0.017 |
| baseline-small (768E) | 15.19 | 2.026 | 0.235 | 4.898 | 0.018 |
| baseline-tiny (512E) | 39.91 | 3.160 | 0.345 | 5.007 | 0.031 |
| synergy-tiny (512E) | **129.78** | **4.165** | 0.349 | 4.557 | 0.022 |

## Finding 1: Synergy-Tiny Regression (512E)

### Root Cause: Overfitting to the Modified Distillation Objective

The smoking gun is a **train-eval divergence**: synergy-tiny achieves a LOWER training loss (4.557) than baseline-tiny (5.007), yet has 3.25x WORSE evaluation perplexity ratio (129.78 vs 39.91). The model successfully minimized the modified loss function but failed to learn generalizable protein language patterns.

### Evidence

**Identical configurations except enhancements.** Both synergy-tiny and baseline-tiny use the exact same architecture (512E/4L/4H), hyperparameters (LR=1e-3, 3 epochs, no warmup), and training data. The model config.json files are byte-for-byte identical. The ONLY difference is `use_uncertainty_weighting=True` and `use_calibration_smoothing=True` in synergy-tiny.

**No training instability.** Grep found zero NaN/Inf values in synergy-tiny training logs. Gradient norms are stable (0.1-0.5 range with occasional ~1.1 spikes). Loss monotonically decreases with no divergence, spikes, or plateaus. The model trained without any issues -- it simply converged to a degenerate minimum.

**The enhancements make the training objective easier.** Synergy-tiny starts at lower loss (6.62 vs 7.94 at step 1) and maintains lower loss throughout. This is expected because:
- Calibration smoothing redistributes teacher probability mass toward uniform, making the KL divergence target inherently easier to match
- Uncertainty weighting de-emphasizes positions where the student already performs well, reducing the effective difficulty of the loss

**Capacity-dependent failure.** The same enhancements at 256E (ablation-both) produce the best PPL ratio of 8.93. At 256E (~3.5M params), limited model capacity acts as natural regularization, forcing the model to learn genuine structure. At 512E (~6.6M params), the model has enough capacity to exploit the easier objective without learning real protein patterns.

**KL divergence confirms generalization failure.** Synergy-tiny has the worst KL divergence (4.165) of all models, indicating the student has diverged maximally from the teacher's actual output distribution despite achieving low training loss on the modified objective.

**Calibration enhancement paradoxically worsens calibration.** Despite using calibration-aware smoothing, synergy-tiny student ECE (0.349) is slightly worse than baseline-tiny (0.345). Both have 0% accuracy with ~35% confidence.

### Mechanism

At 512E with LR=1e-3 and no warmup, the model rapidly converges in early training. The enhancements make the soft loss target easier (smoothed teacher distribution + down-weighted easy positions), so the model finds a shortcut: match the smoothed, re-weighted distribution without encoding actual protein language structure. This shortcut minimizes training loss effectively but produces a representation that fails at evaluation time when tested against real (unsmoothed, unweighted) next-token prediction.

## Finding 2: Non-Monotonic Scaling Pattern

### The Pattern

| Scale | Synergy PPL | Baseline PPL | Synergy Effect | Improvement |
|-------|-------------|--------------|----------------|-------------|
| Tiny (512E) | 129.78 | 39.91 | Harmful | -225% |
| Small (768E) | 7.05 | 15.19 | **Helpful** | **+54%** |
| Medium (1024E) | 5.16 | 3.72 | Harmful | -39% |

### Root Cause: Regularization vs. Capacity Tradeoff

The enhancements function as implicit regularization. Their effect depends on whether the model is underfitting or overfitting:

**At small scale (768E, LR=5e-4):** The baseline model underfits -- it has moderate capacity and a moderate learning rate. The enhancements provide beneficial regularization and richer training signal (position weighting focuses attention on variable regions, smoothing prevents overconfidence). The synergy effect is strongly positive: PPL ratio improves from 15.19 to 7.05 (54% improvement), KL divergence drops from 2.026 to 1.686.

**At medium scale (1024E, LR=1e-4):** The baseline model already fits well -- it has high capacity (approaching teacher size at compression ratio 3.8x) and a conservative learning rate. The enhancements add unnecessary regularization that impedes the model from fully matching the teacher distribution. The synergy effect is mildly negative: PPL ratio worsens from 3.72 to 5.16 (39% degradation). Notably, the KL divergences are nearly identical (1.340 vs 1.344), suggesting the medium model partially compensates through its large capacity but the smoothed target distribution prevents it from achieving the same precision.

**At tiny scale (512E, LR=1e-3):** This is the pathological case described in Finding 1. The combination of high learning rate, moderate capacity, and easier objective creates conditions for catastrophic overfitting to the modified loss. The synergy effect is severely negative: PPL ratio worsens from 39.91 to 129.78 (225% degradation).

### The Learning Rate Confound

A critical observation: the learning rates differ across scales (1e-3 for tiny, 5e-4 for small, 1e-4 for medium). This is not an independent variable -- it was presumably tuned for each scale. However, it means the synergy enhancements interact with different LR regimes:

- At LR=1e-3 (tiny): High LR + easier objective = rapid convergence to degenerate solution
- At LR=5e-4 (small): Moderate LR + easier objective = beneficial regularization
- At LR=1e-4 (medium): Low LR + easier objective = mild undershoot (can't fully exploit teacher signal)

The enhancements likely need their own LR tuning at each scale, which was not done.

### Training Loss vs. Eval Perplexity Pattern

Across all scales, synergy models achieve LOWER training loss than their baseline counterparts:

| Scale | Synergy Train Loss | Baseline Train Loss | Synergy Better? | Eval Better? |
|-------|-------------------|--------------------|-----------------| ------------|
| Tiny | 4.557 | 5.007 | Yes (-9.0%) | **No** (+225%) |
| Small | 4.476 | 4.898 | Yes (-8.6%) | **Yes** (-54%) |
| Medium | 4.479 | 4.926 | Yes (-9.1%) | **No** (+39%) |

The training loss improvement is remarkably consistent (~9% lower for synergy across all scales), confirming the enhancements make the training objective uniformly easier. However, lower training loss translates to better eval perplexity ONLY at small scale. This underscores that the modified training objective is not well-aligned with the evaluation metric at all scales.

## Conclusions

1. **The Phase 0 enhancements are not scale-agnostic.** They function as implicit regularization that benefits models in the underfitting regime (small/768E) but harms models that are either overfitting (tiny/512E with high LR) or well-fit (medium/1024E).

2. **The training loss is a misleading signal.** All synergy models achieve ~9% lower training loss, but this does not predict eval performance. The modified distillation objective (smoothed targets + weighted positions) diverges from the true evaluation metric (standard perplexity).

3. **The synergy-tiny regression is the most severe case** of a general phenomenon: the enhancements make the training objective easier to minimize without guaranteeing the learned representation is useful. At 512E with LR=1e-3, the model has enough capacity and learning speed to fully exploit this shortcut.

4. **The ablation study (256E) was misleading.** The enhancements showed strong results at 256E (PPL ratio 8.93 vs 18.95 baseline) because extreme capacity constraints forced genuine learning. This success did not generalize to 512E.

## Recommendations

### Short-term Fixes

1. **Reduce learning rate for synergy models at 512E.** Try 5e-4 or 1e-4 to slow convergence and prevent degenerate solutions. The success at small scale (768E, LR=5e-4) suggests LR=5e-4 may also work for 512E.

2. **Add warmup steps.** All models currently use warmup_steps=0. Adding 100-500 warmup steps would slow early convergence, which is where the degenerate solution is likely selected.

3. **Reduce smoothing_factor from 0.1.** Try 0.01-0.05 for larger models. Lower smoothing keeps the target distribution closer to the teacher's actual output, reducing the gap between training and evaluation objectives.

4. **Disable enhancements for medium scale.** At 1024E (compression ratio 3.8x), baseline distillation already works well. The enhancements provide no benefit and introduce mild degradation.

### Longer-term Improvements

5. **Add validation-based early stopping.** Monitor held-out perplexity, not training loss, to detect train-eval divergence early. The current setup has no eval during training (`evaluation_strategy: "no"`).

6. **Capacity-aware enhancement scheduling.** Scale the smoothing_factor and uncertainty weight magnitude based on model size. Smaller smoothing and weaker weighting for larger models.

7. **Separate LR sweeps with enhancements.** The optimal learning rate likely differs between baseline and synergy training. The current LRs were presumably tuned for baseline distillation and applied unchanged to synergy.

8. **Investigate individual enhancement contributions at 512E.** The ablation study was done at 256E only. Running `ablation-uncertainty` and `ablation-calibration` at 512E would clarify which enhancement (or their interaction) causes the regression at this scale.
