# bioRxiv Revision Notes: v1 → v2

> **Preprint**: Wijaya, E. "Distilling Protein Language Models with Complementary Regularizers"
> **DOI**: 10.64898/2026.02.17.706304
> **v1 URL**: https://www.biorxiv.org/content/10.64898/2026.02.17.706304v1
> **Revision date**: 2026-02-21

---

## Summary of Changes

Version 2 adds domain-specific fine-tuning experiments demonstrating that synergy-distilled student models are superior starting points for adaptation on scarce protein family data. This is the major addition — 75 fine-tuning runs across 3 protein families, 5 model types, and 5 training set sizes (50–1,000 sequences). All other sections from v1 are preserved with minor updates.

---

## New Content

### New Results Subsection: "Domain-specific fine-tuning" (Section 2.8)

Added after Section 2.7 (Structural quality). Reports fine-tuning experiments on conotoxin (PF02950) and lysozyme (PF00959) protein families at subset sizes of 50–1,000 sequences. Key findings:

- **Sample efficiency (Conotoxin)**: All synergy-distilled students achieve lower test perplexity than the 738M-parameter teacher at every training set size, with a 4.5x gap at N=50. The Synergy-Medium model at N=50 (PPL 372) already outperforms the teacher at N=100 (PPL 1,153), demonstrating 2x sample efficiency.

- **Generation quality (Lysozyme)**: Despite higher test perplexity, distilled students generate sequences that more frequently match the lysozyme Pfam HMM profile (PF00959). Synergy-Small achieves 94% HMMER hit rate at N=1,000 versus the teacher's 69% (+25 percentage points). This perplexity-vs-hit-rate decoupling demonstrates that lower perplexity does not imply better domain-specific generation.

- **Cross-scale synergy advantage**: All three synergy students (Medium, Small, Tiny) outperform the teacher on conotoxin PPL and lysozyme HMMER hit rate. The controlled comparison — Synergy-Tiny vs. Baseline-Tiny (same 37M architecture, standard KD) — confirms the distillation method drives the advantage: Baseline-Tiny performs near teacher level while Synergy-Tiny far exceeds both. Synergy-Tiny wins 15/15 perplexity comparisons across all families.

- **Training cost**: Students fine-tune 20–162x faster in wall-clock time. Synergy-Tiny completes fine-tuning on 1,000 AMP sequences in 25 seconds versus 66 minutes for the teacher, without requiring gradient checkpointing.

### New Figures

| Figure | Description |
|--------|-------------|
| **Figure 10** | Fine-tuning sample efficiency. Two panels showing test perplexity (log scale) vs. training set size for all five models. (a) Conotoxins: all students outperform the teacher at every N. (b) Lysozymes: teacher achieves lowest PPL but students converge, setting up the hit-rate inversion in Figure 11. |
| **Figure 11** | Family-specific generation quality. Two panels showing HMMER hit rate (%) vs. training set size. (a) Lysozyme (PF00959): Synergy-Small reaches 94% vs. teacher 69% at N=1,000. (b) Conotoxin (PF02950): Synergy-Medium reaches 42.5% vs. teacher 8%. |

### New Table (Main Text)

- Synergy-Tiny vs. Baseline-Tiny perplexity comparison across conotoxin and lysozyme at all subset sizes, showing 15/15 wins for synergy distillation.

### New Appendix

Added appendix with fine-tuning experimental details:

| Table | Content |
|-------|---------|
| **Table A1** | Fine-tuning hyperparameters per model (learning rate, batch size, gradient checkpointing, etc.) |
| **Table A2** | Dataset summary for conotoxin, lysozyme, and AMP families (sizes, splits, Pfam IDs) |
| **Table A3** | Generation and evaluation configuration (sampling parameters, HMMER thresholds) |
| **Table A4** | Complete AMP fine-tuning results (perplexity and amino acid KL; included for completeness — teacher dominates on this family) |

---

## Updated Sections

### Abstract
Added two sentences describing the fine-tuning results: student superiority on conotoxin perplexity and lysozyme HMMER hit rate. Total word count remains under 250 words.

### Introduction
- Added motivation sentence connecting distillation to the fine-tuning use case in biopharma (domain adaptation on 50–1,000 proprietary sequences).
- Added contribution item 6: demonstration that distilled students are superior fine-tuning starting points on scarce protein family data.

### Methods
- Added subsection "Fine-tuning evaluation" with brief prose describing the experimental setup (families, subset sizes, optimization, evaluation metrics). Full hyperparameter tables are in the appendix.

### Discussion
- **New paragraph**: "Perplexity-vs-hit-rate decoupling" — interprets the lysozyme finding where students with higher PPL generate better family-matching sequences. Discusses the bias-variance advantage of smaller models on scarce data and notes the advantage holds across all three synergy scales.
- **Updated "Practical implications"**: Added fine-tuning cost advantage (domain-adapted models in minutes vs. hours, fewer sequences needed).
- **Updated "Limitations"**: Added fine-tuning caveats (single learning rate per model size, generation length mismatch that depresses HMMER hit rates — making reported results conservative).
- **Updated "Future directions"**: Added length-constrained generation and fine-tuning on additional protein families.

---

## Unchanged Sections

The following v1 content is preserved without modification:

- Section 2.1: Ablation reveals complementary regularizers (Table 1, Figure 1)
- Section 2.2: Scaling across model sizes (Table 2, Figure 2)
- Section 2.3: Calibration analysis (Figure 3)
- Section 2.4: Biological validity (Figure 4)
- Section 2.5: Compression-quality tradeoff (Figure 5)
- Section 2.6: Practical deployment (Figures 6, 8)
- Section 2.7: Structural quality (Figure 7)
- Section 3 Discussion: Mechanistic explanation, training dynamics, scale-dependent effects, learning rate scaling, Small model ECE regression (Figure 9)
- Section 4 Methods: Sections 4.1–4.6 (standard distillation, uncertainty weighting, calibration-aware distillation, model architectures, training details, data availability)
- All 19 references from v1

---

## Version Comparison

| Metric | v1 | v2 |
|--------|----|----|
| Pages (est.) | 15 | ~18–19 |
| Main-text figures | 9 (Fig. 1–9) | 11 (Fig. 1–11) |
| Main-text tables | 3 (Tables 1–3) | 4 (Tables 1–4) |
| Appendix tables | 0 | 4 (Tables A1–A4) |
| Results subsections | 7 | 8 |
| Fine-tuning runs reported | 0 | 75 |
| Protein families evaluated | 0 | 3 (conotoxin, lysozyme, AMP) |
