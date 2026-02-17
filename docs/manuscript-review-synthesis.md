# Manuscript Review Synthesis — Panel of 5 Reviewers

**Date:** 2026-02-17
**Verdict:** NOT READY — Major Revisions Required

The panel identified **7 critical issues** (consensus across 3+ reviewers), **10 major issues**, and **~15 minor issues**. Three critical issues are independently fatal for submission. The core finding (constructive interference) is potentially interesting but cannot be trusted in the current form due to confounded experiments, a code-paper mismatch, and an invalid evaluation set.

---

## Critical Issues (Unanimous or Near-Unanimous — Must Fix)

### C1. Evaluation on 5 Hardcoded Sequences (5/5 reviewers flagged)

**The single most damaging finding.** `scripts/evaluate.py:get_test_sequences()` returns exactly 5 hardcoded protein sequences, repeated cyclically to reach the desired count. ALL perplexity and ECE numbers in Tables 1-2 may be computed on these 5 sequences. The paper claims "held-out protein sequences" — this is materially misleading.

- **stats-reviewer**: "evaluating on at most 5 distinct sequences (possibly memorized from training)"
- **bio-reviewer**: "a fundamental methodological flaw"
- **devil-advocate**: "this invalidates the quantitative claims entirely"

**Fix**: Recompute all metrics on a genuine held-out set of 1,000+ diverse UniProt sequences.

### C2. Confounded Ablation — LR/Warmup Not Controlled (4/5 reviewers)

The synergy models use half the baseline learning rate + 500-step warmup. The baseline models do NOT receive this treatment. The central comparison is therefore:

**(Synergy + halved LR + warmup)** vs **(Baseline + original LR + no warmup)**

No experiment tests **Baseline + halved LR + warmup**. The 87% improvement could be entirely attributable to the learning rate schedule, not the proposed method.

- **devil-advocate**: "This is not a minor confound. It is the central experimental claim of the paper."
- **stats-reviewer**: "A 2x2 ablation where the winning combination requires special hyperparameter treatment that was not applied in the ablation is not a valid isolation of component effects."

**Fix**: Train baseline models with the same halved LR + warmup schedule. If they match synergy performance, the method contribution vanishes.

### C3. Code-Paper Mismatch — Smoothing on Temperature-Scaled Logits (3/5 reviewers, verified by team lead)

**Paper** (Eq. 6): `epsilon_t = lambda * (1 - max_v p_T(v|x_{<t}))` — uses raw (T=1) teacher confidence.
**Code** (distillation.py:265-269): Smoothing is applied to `F.softmax(logits / T)` — temperature-scaled (T=2.0) probabilities, which artificially deflates `max_prob` and systematically over-smooths at every position.

This means the "calibration-aware" mechanism as implemented is **not** what the paper describes. The mechanistic explanation for why smoothing works is invalidated for the actual code.

- **devil-advocate**: "The 'calibration-aware' behavior is therefore not actually calibration-aware"
- **bio-reviewer**: "the paper should explicitly justify why using unscaled entropy for weighting while using temperature-scaled distributions for the KL target is the right design choice"

**Fix**: Either (a) fix the code to use raw logits for confidence estimation and re-run all experiments, or (b) update the paper equations to match the actual implementation and provide justification.

### C4. Two Different Architectures Both Called "Tiny" (4/5 reviewers)

- Ablation (Table 1): Tiny = 4L/4H/**256E**, baseline PPL = 18.95
- Scaling (Table 2): Tiny = 4L/4H/**512E**, baseline PPL = 39.91

The "53% improvement" headline comes from the 256E ablation. The "87% improvement" comes from the 512E scaling. These are different models. The paper never explains the discrepancy. The ablation doesn't prove constructive interference for the architecture actually presented as the main contribution.

**Fix**: Use consistent naming. Add 256E to the architectures table. Explain why the ablation used a different architecture than the scaling experiments.

### C5. No Statistical Significance Testing (4/5 reviewers)

Zero confidence intervals, zero standard deviations, zero p-values, zero replicate runs across the entire paper. All results are single training runs. Given the documented >2x variance in Tiny perplexity between configurations, single-point estimates are scientifically indefensible.

**Fix**: Run each configuration with 3-5 random seeds. Report mean +/- std for all metrics.

### C6. "Constructive Interference" Terminology Is Scientifically Incorrect (3/5 reviewers)

The term borrows from wave physics where it has precise mathematical meaning (superposition with phase alignment). The phenomenon here is a regularization synergy — a hyperparameter interaction effect. The signal processing analogy (amplification + filtering) is formally backwards: the code applies filtering (smoothing) BEFORE amplification (weighting), not after.

- **claims-reviewer**: "The term 'interference' is scientifically incorrect here"
- **devil-advocate**: "'constructive interference' is a marketing term, not a mechanistic claim"

**Fix**: Rename to "regularization synergy" or "complementary regularization." Remove or explicitly disclaim the signal processing analogies.

### C7. Biological Validity Claims Rest on Superficial Metrics (3/5 reviewers)

- KL < 0.015 threshold is arbitrary with no biological justification
- No sequence diversity metrics (pairwise identity, clustering)
- No functional assessments (secondary structure, motif analysis)
- No per-residue analysis of rare but critical amino acids (C, W, H, M)
- pLDDT assessment on 50 sequences with no error bars; teacher pLDDT of 51.2 is itself in the "very low confidence" range

**Fix**: Add per-residue frequency deviations, diversity metrics, increase pLDDT sample size with error bars, acknowledge that pLDDT < 50 means unreliable structural predictions.

---

## Major Issues (Should Fix)

| # | Issue | Flagged By |
|---|-------|------------|
| M1 | Post-hoc "mechanistic explanation" presented as formal information theory but contains no derivations, no falsifiable predictions | stats, claims, devil |
| M2 | Biopharma deployment claims (antibody screening, data governance) are speculative and unsubstantiated — no regulatory citations, no actual deployment | claims, bio |
| M3 | "First systematic study" claim needs "to our knowledge" hedge and more thorough literature search (ESM2, ProGen2, ProtTrans not cited) | claims, writing |
| M4 | Small model ECE regression (0.259 vs 0.235) undermines calibration claims; LR explanation is hand-waving | devil, stats, claims |
| M5 | Token-level ECE is wrong metric for protein engineering (sequence-level calibration matters) | stats, bio |
| M6 | Training data ("10% of UniProt") not described: which release, Swiss-Prot vs TrEMBL, sampling strategy, sequence count | bio, writing |
| M7 | Inference speedup conflates generation throughput with scoring throughput — different operations for different use cases | bio, claims |
| M8 | pLDDT 38-40 for students is "very low confidence" — claiming utility for protein engineering at these scores is questionable | bio, devil |
| M9 | Notation inconsistency: `x_{<t}` vs `x_{<=t}` conditioning in equations | writing, stats |
| M10 | Hard loss equation missing normalization factor (paper uses sum, code uses mean) | writing, stats |

---

## Minor Issues (Nice to Fix)

- Fig 9 referenced only in Discussion, not Results (writing)
- Naeini 2015 citation wrong for equal-width bins ECE — should cite Guo 2017 only (writing)
- "Consumer-grade GPU" claim — benchmarks on L40S ($10K professional card), not consumer hardware (writing, devil, bio)
- "Pareto frontier" from 3 data points is not a Pareto frontier (claims, devil)
- Two different KL divergence quantities both called "KL divergence" without disambiguation (claims, bio)
- Code docstring cites Kim 2025 and Song 2025 not in references.bib (writing)
- Speedup benchmarks lack batch size, sequence length details (stats)
- No validation loss monitoring during training — no convergence evidence (stats, devil)
- pLDDT section reports only Medium scale comparison, not Tiny/Small (writing)
- Missing key references: ESM2, ProGen2, ProtTrans (writing)
- GPT-2 citation lists "OpenAI Blog" as venue (writing)
- Sequence length distribution of generated sequences not reported (bio)

---

## Strengths (Consensus Across Reviewers)

1. **The core empirical finding is genuinely interesting** — if the confounds are resolved and the effect holds, a regularization synergy in protein LM distillation is a real contribution
2. **Honest limitations section** — all 5 reviewers praised the transparency about single teacher, token-level ECE, no wet-lab validation
3. **ECE alongside perplexity** is underutilized in protein LM literature; the calibration analysis adds real value
4. **Practical deployment analysis** (memory footprint, GPU throughput) is concrete and useful
5. **Open-source model release** on HuggingFace is a genuine contribution regardless of paper issues
6. **2x2 ablation design** is conceptually the right approach to detect interaction effects
7. **Code-paper alignment** on the core distillation algorithm (T^2 scaling, loss combination) is solid

---

## Required Experiments Before Resubmission

| Priority | Experiment | Addresses |
|----------|-----------|-----------|
| 1 | Recompute all metrics on proper held-out set (1,000+ UniProt sequences) | C1 |
| 2 | Train baseline with halved LR + warmup, compare against synergy | C2 |
| 3 | Fix smoothing to use raw logits for confidence OR update equations, re-run | C3 |
| 4 | Run all configs with 3-5 random seeds, report mean +/- std | C5 |
| 5 | Run 2x2 ablation at Small and Medium scales | C4, M4 |
| 6 | Add per-residue amino acid analysis, diversity metrics, increase pLDDT n | C7 |
| 7 | Benchmark inference on actual consumer GPU (RTX 3090/4060) | Minor |

---

## Venue Recommendation

- **NeurIPS/ICML**: Not ready. Insufficient novelty, narrow scope (one teacher, one architecture family), no statistical rigor. Would be desk-rejected or receive 3-4/10 scores.
- **Bioinformatics / PLoS Comp Bio / Proteins: Structure, Function, and Bioinformatics**: Appropriately scoped IF critical issues are resolved. The practical contribution (compressed protein LMs) is genuine.
- **Workshop paper (NeurIPS MLCompBio, ICML CompBio)**: Could be accepted as-is after fixing C1-C3. The finding is interesting enough for a workshop even with limited scale.
