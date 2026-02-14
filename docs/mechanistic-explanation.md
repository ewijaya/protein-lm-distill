# Mechanistic Explanation: Complementary Effect of Uncertainty-Aware and Calibration-Aware Distillation

**Date**: February 14, 2026
**Status**: Working document for paper Discussion section

---

## 1. Summary of the Phenomenon

The core finding of this work is a **complementary effect**: uncertainty-aware position weighting and calibration-aware label smoothing each *degrade* distillation quality when applied independently, but *dramatically improve* quality when applied together. This document proposes a mechanistic explanation grounded in the mathematical formulations and supported by empirical training dynamics.

### 1.1 Ablation Evidence (256E Architecture)

| Configuration | PPL Ratio | KL Div | ECE | vs Baseline |
|---------------|-----------|--------|-----|-------------|
| Baseline (standard KD) | 18.95 | 2.23 | 0.274 | — |
| +Uncertainty only | 36.89 | 2.87 | 0.325 | +95% worse |
| +Calibration only | 39.64 | 3.00 | 0.319 | +109% worse |
| **+Both (combined)** | **8.93** | **1.62** | **0.216** | **53% better** |

### 1.2 Scaling Evidence (v2 models with LR + warmup fix)

| Scale | Compression | Baseline PPL | Synergy PPL | Improvement | Synergy ECE | Baseline ECE |
|-------|-------------|-------------|-------------|-------------|-------------|--------------|
| Tiny (512E) | 20x | 39.91 | **5.06** | **87%** | **0.183** | 0.345 |
| Small (768E) | 9.4x | 15.19 | **7.05** | **54%** | 0.259 | 0.235 |
| Medium (1024E) | 3.8x | 3.72 | **2.58** | **31%** | **0.135** | 0.169 |

---

## 2. What Each Enhancement Does Individually

### 2.1 Uncertainty-Aware Position Weighting (Alone)

**Mechanism**: Computes Shannon entropy of teacher predictions at each position, then upweights high-entropy positions in the soft loss:

$$w_t = 0.5 + 0.5 \cdot \text{normalize}(H(p_T(\cdot | x_{<t})))$$

**Intended effect**: Focus the student on learning the teacher's behavior at biologically variable positions (loops, linkers, surface residues) rather than constrained positions where the prediction is trivial.

**Why it hurts alone**: High-entropy positions are precisely the positions where the teacher's probability distribution is most **spread out and potentially miscalibrated**. The teacher assigns non-trivial probability to many amino acids at these positions, but the relative ranking of these probabilities may not reflect genuine biological preferences — it reflects the teacher's own modeling uncertainty, including its errors.

By upweighting these positions, uncertainty-only training forces the student to invest disproportionate capacity in matching a noisy signal. The student faithfully reproduces the teacher's uncertainty patterns, including artifacts, while underlearning the reliable, low-entropy positions where the teacher's predictions are accurate.

**Empirical signature**: PPL ratio nearly doubles (18.95 → 36.89), KL divergence increases by 29% (2.23 → 2.87), and ECE worsens (0.274 → 0.325). The student becomes worse at predicting correct tokens *and* worse-calibrated.

### 2.2 Calibration-Aware Label Smoothing (Alone)

**Mechanism**: Applies adaptive smoothing to teacher distributions, with smoothing inversely proportional to teacher confidence:

$$\epsilon_t = \lambda \cdot (1 - \max_v p_T(v | x_{<t}))$$
$$\bar{p}_T(v) = (1 - \epsilon_t) \cdot p_T(v) + \epsilon_t / |\mathcal{V}|$$

**Intended effect**: Reduce teacher overconfidence by blending high-confidence predictions with a uniform distribution, producing more conservative soft targets that improve student calibration.

**Why it hurts alone**: Smoothing is strongest at low-confidence positions (where $\max p_T$ is low), which pushes those distributions closer to uniform. This *destroys discriminative signal* at exactly the positions where the teacher has the most to teach — positions where multiple amino acids are plausible but in specific proportions reflecting biological constraints. The student receives targets that increasingly resemble "any amino acid is equally likely" at variable positions, losing the nuanced probability structure that distinguishes functional from non-functional substitutions.

At high-confidence positions, smoothing is minimal ($\epsilon_t \approx 0$), so those positions are essentially unchanged from standard KD. The net effect is that calibration-only training selectively degrades the most informative part of the teacher's knowledge.

**Empirical signature**: PPL ratio more than doubles (18.95 → 39.64), the worst of all configurations. KL divergence is highest (3.00). The student has maximally diverged from the teacher's true distribution.

---

## 3. The Complementary Mechanism

### 3.1 Why Together They Help: Mutual Error Correction

When both enhancements are applied simultaneously, they compensate for each other's failure modes:

**Calibration smoothing neutralizes uncertainty weighting's noise amplification problem.**

Without smoothing, uncertainty weighting upweights high-entropy positions where the teacher's distribution is noisy and potentially miscalibrated. With smoothing applied *before* weighting, the teacher's distribution at high-entropy positions is regularized — extreme probability spikes from miscalibration are dampened, and the remaining distribution better reflects genuine amino acid preferences. The student now matches a *cleaned* version of the teacher's uncertainty rather than raw noise.

Mathematically, at high-entropy positions where both effects are active:
- Smoothing reduces $\max p_T$, pulling the distribution toward uniform (noise reduction)
- Weighting increases the loss contribution at these positions (signal amplification)

The combined effect is **amplified but regularized attention to variable positions**: the student is told "pay extra attention here" (uncertainty weighting) but what it's asked to match is a more conservative, less noisy target (calibration smoothing).

**Uncertainty weighting focuses calibration smoothing where it's needed most.**

Without weighting, calibration smoothing degrades signal at *all* uncertain positions equally, treating them as noise to be suppressed. With uncertainty weighting, the student allocates more capacity to these positions, partially compensating for the signal loss from smoothing. The weighting creates a gradient that says "these positions matter more," counteracting the tendency of smoothing to flatten them into irrelevance.

At low-entropy positions (confident predictions), neither enhancement has much effect:
- Smoothing is minimal ($\epsilon_t \approx 0$ because $\max p_T \approx 1$)
- Weighting is low ($w_t \approx 0.5$, the floor)

This means both enhancements are effectively inactive at easy positions, preserving standard KD behavior where the teacher is most reliable. The modifications are concentrated at the positions that need them — high-entropy, potentially miscalibrated regions.

### 3.2 The Information-Theoretic View

Consider the distillation process as information transfer from teacher to student. At each position, the teacher provides:

1. **Signal**: genuine amino acid preferences reflecting protein biology
2. **Noise**: artifacts of the teacher's own training (miscalibration, overfitting to training distribution)

| Position Type | Teacher Entropy | Signal Content | Noise Content |
|---------------|----------------|----------------|---------------|
| Constrained (helices, strands) | Low | High, concentrated | Low |
| Variable (loops, termini) | High | Moderate, distributed | High |

Standard KD treats all positions equally, resulting in a mix of signal and noise. The two enhancements create a modified information channel:

- **Uncertainty weighting** increases bandwidth to high-entropy positions (more signal *and* more noise)
- **Calibration smoothing** acts as a low-pass filter on the teacher distribution (reduces noise *and* signal)

Applied independently, each degrades the signal-to-noise ratio:
- Weighting alone: increases noise more than signal (because noise dominates at high-entropy positions)
- Smoothing alone: decreases signal more than noise (because smoothing toward uniform destroys distributional structure)

Applied together, they achieve the optimal combination:
- Increased bandwidth to important positions (weighting)
- Filtered through a noise-reducing channel (smoothing)
- Net effect: higher signal-to-noise ratio than any single configuration

### 3.3 Analogy: Radio Signal Processing

The complementary effect is analogous to radio signal processing:

- **Uncertainty weighting** is like turning up the volume (amplification) on a noisy channel
- **Calibration smoothing** is like a noise filter

Amplification alone makes both signal and noise louder — overall quality degrades. Filtering alone attenuates everything — you lose faint but important signals. But **amplification followed by filtering** (boost weak signals, then remove noise) is a standard signal processing technique that improves reception quality.

In our framework, the operations are applied in this beneficial order:
1. First, teacher probabilities are computed
2. Then, calibration smoothing filters the distribution (noise reduction)
3. Then, uncertainty weighting amplifies important positions (signal boost on clean input)
4. The student optimizes against this filtered-and-amplified target

---

## 4. Training Dynamics Evidence

### 4.1 Early Training: The Critical Window

Analysis of training logs reveals that the first ~500 steps are the critical period. Comparing v1 (no warmup) vs v2 (warmup=500) models:

| Model | Initial Loss | Loss at ~step 50 | Loss at ~step 200 | Final Loss |
|-------|-------------|-------------------|--------------------| -----------|
| synergy-tiny-v1 (LR=1e-3, no warmup) | 6.62 | 5.37 | 4.99 | 4.41 |
| **synergy-tiny-v2** (LR=5e-4, warmup=500) | **7.93** | **5.39** | **4.99** | **4.40** |
| baseline-tiny (LR=1e-3, no warmup) | 7.94 | 6.18 | 5.61 | 4.79 |
| synergy-medium-v1 (LR=1e-4, no warmup) | 7.62 | 5.39 | 5.00 | 4.30 |
| **synergy-medium-v2** (LR=5e-5, warmup=500) | **8.01** | **5.60** | **5.12** | **4.35** |
| baseline-medium (LR=1e-4, no warmup) | 9.07 | 6.14 | 5.61 | 4.67 |

**Key observation**: The v1 synergy models start with an anomalously low initial loss (synergy-tiny-v1: 6.62 vs baseline: 7.94). This indicates the modified objective (smoothed targets + weighted positions) is inherently easier to minimize from random initialization. With full learning rate from step 0, the model rapidly converges toward a solution that exploits this easier objective — a **degenerate minimum** that achieves low training loss but poor generalization.

The v2 models with warmup start at the same initial loss as baselines (synergy-tiny-v2: 7.93 ≈ baseline: 7.94) because the learning rate is near-zero during warmup. This prevents the model from committing to a degenerate solution early. By the time the learning rate reaches full value (~step 500), the model has already formed preliminary representations that constrain it to a more generalizable region of the loss landscape.

### 4.2 Final Training Loss vs Evaluation Performance

All synergy models achieve lower final training loss than baselines (consistently ~8-9% lower), but this translates to better evaluation performance only with the LR + warmup fix:

| Model | Final Train Loss | Eval PPL Ratio | Train-Eval Alignment |
|-------|-----------------|----------------|---------------------|
| synergy-tiny-v1 | 4.41 | 129.78 | Severe misalignment |
| **synergy-tiny-v2** | **4.40** | **5.06** | **Good alignment** |
| baseline-tiny | 4.79 | 39.91 | Moderate |
| synergy-medium-v1 | 4.30 | 5.16 | Mild misalignment |
| **synergy-medium-v2** | **4.35** | **2.58** | **Good alignment** |
| baseline-medium | 4.67 | 3.72 | Moderate |

The v2 models have nearly identical final training loss to their v1 counterparts (tiny: 4.40 vs 4.41; medium: 4.35 vs 4.30) but dramatically better evaluation performance. This confirms that the improvement is not about reaching a lower loss — it is about reaching a **different minimum** that generalizes better.

### 4.3 The Warmup Interaction

Warmup is particularly important for synergy training because the combined enhancements create a modified loss landscape with additional local minima. The complementary mechanism depends on the student learning *genuine protein structure* before being asked to match the enhanced targets. Without warmup:

1. The student sees the full modified objective from step 0
2. The easier objective (smoothed targets, weighted positions) creates a gradient path toward a shortcut solution
3. With full learning rate, the model takes large steps along this shortcut path
4. Once committed to the shortcut minimum, the model cannot escape

With warmup:

1. Small learning rate forces the student to make incremental updates
2. The student learns basic token frequency patterns first (a generalizable foundation)
3. By the time LR reaches full value, the model is already in a basin that corresponds to meaningful protein representations
4. The enhanced objective then *refines* this good representation rather than corrupting it

---

## 5. Scale-Dependent Effects

### 5.1 Why the Improvement Decreases with Scale

The synergy improvement over baseline decreases as model scale increases (87% → 54% → 31%). This is expected because:

1. **Larger models have less to gain from regularization.** The medium student (1024E, 200M params, 3.8x compression) is only 3.8x smaller than the teacher and already achieves good distillation with standard KD (PPL ratio 3.72). The enhancements provide diminishing marginal benefit as the student approaches teacher capacity.

2. **The baseline gets better faster with scale.** Baseline PPL ratio improves roughly exponentially with model size (39.91 → 15.19 → 3.72). The synergy method also improves (5.06 → 7.05 → 2.58), but the gap between them narrows because standard KD becomes increasingly effective.

3. **The noise-to-signal ratio at uncertain positions decreases with capacity.** Larger students can better model the true distribution at variable positions, reducing the benefit of the noise-filtering mechanism.

### 5.2 The Learning Rate Scaling Pattern

A practical finding is that synergy training requires a lower learning rate than standard KD at the same scale:

| Scale | Baseline LR | Synergy v1 LR | Synergy v2 LR | LR Ratio (v2/baseline) |
|-------|------------|---------------|---------------|----------------------|
| Tiny (512E) | 1e-3 | 1e-3 | **5e-4** | 0.5x |
| Small (768E) | 5e-4 | 5e-4 | 5e-4 | 1.0x |
| Medium (1024E) | 1e-4 | 1e-4 | **5e-5** | 0.5x |

The successful pattern is roughly **half the baseline LR** for synergy training, plus warmup. This makes sense: the modified objective has a smoother loss landscape (due to smoothed targets), so the same learning rate produces effectively larger functional steps. Halving the LR compensates for this.

Synergy-small already used LR=5e-4 (which happened to be appropriate) and showed improvement without needing a v2 re-run. The failures at tiny (LR=1e-3) and medium (LR=1e-4) scales were both cases where the LR was too high for the synergy objective.

---

## 6. Mathematical Formalization

### 6.1 The Modified Soft Loss

When both enhancements are active, the soft loss at position $t$ is:

$$L_{\text{soft}}^t = w_t \cdot D_{KL}(\bar{p}_T^{(\tau)}(\cdot | x_{<t}) \| p_S^{(\tau)}(\cdot | x_{<t}))$$

where $w_t$ is the entropy-based weight and $\bar{p}_T$ is the smoothed teacher distribution.

### 6.2 Decomposing the Interaction

Define the teacher distribution at position $t$ as $p_T = s_t + n_t$, where $s_t$ represents the "true signal" (genuine amino acid preferences) and $n_t$ represents "noise" (miscalibration artifacts).

**Uncertainty weighting** scales the loss by $w_t \propto H(p_T)$. At high-entropy positions, $H(p_T)$ is large, so:

$$\text{Weighted loss} \propto H(p_T) \cdot D_{KL}(p_T \| p_S) \approx H(p_T) \cdot [D_{KL}(s_t \| p_S) + \text{noise terms}]$$

The noise terms are amplified proportionally to entropy.

**Calibration smoothing** modifies the target: $\bar{p}_T = (1-\epsilon_t) p_T + \epsilon_t / |\mathcal{V}|$. At high-entropy positions, $\epsilon_t$ is large, effectively replacing noisy components with uniform:

$$\bar{p}_T \approx (1-\epsilon_t)(s_t + n_t) + \epsilon_t / |\mathcal{V}| \approx (1-\epsilon_t) s_t + \text{attenuated noise}$$

**Combined**, the loss at high-entropy positions becomes:

$$L_{\text{combined}}^t \propto H(p_T) \cdot D_{KL}(\bar{p}_T \| p_S) \approx H(p_T) \cdot D_{KL}((1-\epsilon_t) s_t + \text{small noise} \| p_S)$$

The student sees an **amplified but cleaned signal**: the position is marked as important (high weight) but the target distribution has been denoised (smoothing). This is strictly better than either:
- Amplified raw signal (uncertainty-only): $H(p_T) \cdot D_{KL}(s_t + n_t \| p_S)$
- Attenuated everything (calibration-only): $D_{KL}((1-\epsilon_t)(s_t + n_t) + \text{uniform} \| p_S)$

### 6.3 Why the Effect Is Non-Additive

The key insight is that the interaction is **multiplicative, not additive**. The two enhancements modify different aspects of the loss:
- Weighting modifies the **per-position contribution** (outer multiplier)
- Smoothing modifies the **target distribution** (inner KL argument)

Because they operate on different components of the loss, their effects compose multiplicatively. A multiplicative interaction can be synergistic (as observed) when the two modifications address complementary failure modes:

$$\text{Synergy} = \text{Combined effect} - (\text{Effect}_1 + \text{Effect}_2)$$

Individual effects are negative (degradation). But the combined effect removes the *reason* each individual effect is negative:
- Weighting's degradation comes from amplifying noise → smoothing removes the noise
- Smoothing's degradation comes from destroying signal → weighting compensates by increasing attention

---

## 7. Implications for Publication

### 7.1 Core Narrative

The complementary effect provides a compelling and counter-intuitive finding for the paper:

> *Individual modifications to the distillation objective that independently degrade performance can combine to produce dramatic improvement. This occurs because each modification addresses the other's failure mode: calibration smoothing removes the noise that uncertainty weighting would amplify, while uncertainty weighting compensates for the signal loss introduced by smoothing.*

### 7.2 Connection to Broader Literature

This type of complementary interaction is known in other domains:

- **Dropout + batch normalization**: Individually beneficial, sometimes harmful together, but the combination can be tuned to be synergistic
- **Data augmentation + regularization**: Aggressive augmentation alone can hurt; combined with appropriate regularization, it improves generalization
- **Ensemble diversity + accuracy**: Individual diverse predictors may be weaker, but their combination outperforms any single model

The protein distillation setting provides a clean, well-controlled demonstration of this principle because:
1. Only two binary modifications (on/off for each enhancement)
2. Large effect sizes (>50% improvement, not marginal)
3. Consistent across model scales (with appropriate LR)
4. Clear mechanistic explanation grounded in information theory

### 7.3 Practical Implications

1. **Protein LM distillation benefits from joint uncertainty + calibration awareness.** Neither enhancement should be used alone.
2. **Learning rate must be adjusted for synergy training.** Roughly 0.5x the baseline LR with warmup.
3. **The benefit is largest at high compression ratios** (87% improvement at 20x compression) and diminishes but remains significant at low compression (31% at 3.8x).
4. **Warmup is essential** to prevent the student from exploiting the easier modified objective during early training.

---

## 8. Open Questions

1. **Is the 0.5x LR rule universal?** We confirmed it at tiny and medium scales but only tested one LR per scale. A systematic LR sweep with synergy enhancements could refine this.

2. **Would synergy-small benefit from warmup?** Small already uses the "correct" LR (5e-4) but was trained without warmup. Adding warmup might improve its ECE (currently 0.259, worse than baseline 0.235).

3. **What is the optimal smoothing factor ($\lambda$)?** All experiments used $\lambda = 0.1$. The complementary mechanism suggests $\lambda$ should scale with teacher miscalibration, which may vary by dataset.

4. **Does this complementary effect generalize beyond ProtGPT2?** Testing on other protein LMs (ESM, ProGen) or non-protein domains would establish generality.

---

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network.
- See `docs/METHODS.md` for complete mathematical framework and implementation references.
- See `docs/investigation-summary.md` for the regression analysis that motivated the v2 re-runs.
