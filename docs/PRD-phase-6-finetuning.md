# Phase 6: Fine-Tuning Advantage Experiments

> Demonstrate that distilled student models are superior to the ProtGPT2 teacher as starting points for domain-specific fine-tuning — the primary real-world use case in biopharma.

## Executive Summary

The current paper (Phases 0–5) establishes that distilled models are faster, smaller, and preserve biological quality for **general** protein generation. Phase 6 addresses a critical gap: showing that smaller distilled models are **better starting points for domain adaptation** on scarce, real-world datasets. This is the strongest practical argument for adoption, since biopharma users don't just run models — they fine-tune them on proprietary protein families.

**Core hypothesis**: Distilled students will match or exceed the teacher's fine-tuned performance with fewer examples, less compute, and lower overfitting risk — precisely because they have fewer parameters and better-calibrated representations from synergy distillation.

---

## Sub-Phase Checklist

- [ ] **6.0** Dataset acquisition & preparation (AMPs, conotoxins, lysozymes)
- [ ] **6.1** Fine-tuning infrastructure (`scripts/finetune.py`, `scripts/evaluate_finetune.py`)
- [ ] **6.2** Few-shot sample efficiency experiment (5 models × 6 sizes × 3 families)
- [ ] **6.3** Overfitting & catastrophic forgetting analysis
- [ ] **6.4** Conditional generation quality (HMMER hit rate, novelty, diversity)
- [ ] **6.5** Fine-tuning cost benchmark (wall-clock, GPU memory)
- [ ] **6.6** Analysis, figures & paper section

---

## Motivation

### Why fine-tuning matters more than generation

In real biopharma workflows, the model is rarely used out-of-the-box. Instead:

1. **Antibody discovery**: Fine-tune on a proprietary antibody library (~500–5,000 sequences), then generate variants for affinity maturation
2. **Enzyme engineering**: Fine-tune on a specific enzyme family (~100–1,000 sequences), then generate candidates for directed evolution
3. **Peptide therapeutics**: Fine-tune on antimicrobial peptides (~2,000 sequences), then generate novel candidates

In all cases, labeled data is **scarce** and **proprietary**. A model that adapts well from 100–500 examples is more valuable than one requiring 10,000+.

### Why students should win at fine-tuning

| Factor | Teacher (738M) | Student (~37M Tiny) | Advantage |
|--------|---------------|---------------------|-----------|
| Parameters to update | 738M | 37M | 20x fewer = less overfitting |
| Fine-tuning GPU memory | ~6 GB | ~0.3 GB | Fits on any GPU |
| Fine-tuning time (1 epoch, 1K seqs) | ~15 min | ~1 min | Faster iteration cycles |
| Effective regularization | None | Built-in from distillation | Better generalization |
| Calibration | Standard | Synergy-enhanced (lower ECE) | More reliable confidence |

---

## Project Timeline

| Sub-phase | Description | Duration | Dependencies |
|-----------|-------------|----------|--------------|
| 6.0 | Dataset acquisition & preparation | 1 day | None |
| 6.1 | Fine-tuning infrastructure | 1 day | 6.0 |
| 6.2 | Few-shot sample efficiency experiment | 2 days | 6.1 |
| 6.3 | Overfitting & forgetting analysis | 1 day | 6.2 |
| 6.4 | Conditional generation quality | 1 day | 6.2 |
| 6.5 | Fine-tuning cost benchmark | 0.5 day | 6.1 |
| 6.6 | Analysis, figures, paper section | 2 days | 6.2–6.5 |
| **Total** | | **~8 days** | |

---

## Phase 6.0: Dataset Acquisition & Preparation

### Target protein families

We select three families with varying dataset sizes and biological complexity:

| Family | Source | Dataset Size | Why |
|--------|--------|-------------|-----|
| **Antimicrobial peptides (AMPs)** | DBAASP v3 / APD3 | ~3,000 sequences | Small, well-characterized, direct therapeutic relevance |
| **Conotoxins** | ConoServer / UniProt | ~1,500 sequences | Very small, disulfide-rich, challenging for LMs |
| **Lysozymes** | UniProt (PF00959) | ~5,000 sequences | Medium-size, structurally well-studied, clear Pfam HMM |

**Minimum viable experiment**: AMPs alone are sufficient for a compelling result. Conotoxins and lysozymes provide robustness across protein types.

### Data preparation

```bash
# Directory structure
data/
├── finetune/
│   ├── amp/
│   │   ├── train_full.fasta      # All training sequences
│   │   ├── train_50.fasta        # Subsets for sample-efficiency curves
│   │   ├── train_100.fasta
│   │   ├── train_200.fasta
│   │   ├── train_500.fasta
│   │   ├── train_1000.fasta
│   │   ├── val.fasta             # 10% held out for validation
│   │   └── test.fasta            # 10% held out for final eval
│   ├── conotoxin/
│   │   └── ...                   # Same structure
│   └── lysozyme/
│       └── ...                   # Same structure
```

### Data acquisition steps

1. Download AMP sequences from DBAASP (https://dbaasp.org/download) — filter for experimentally validated, length 10–100 AA
2. Download conotoxin sequences from UniProt (taxonomy:6490 + keyword:conotoxin)
3. Download lysozyme sequences from UniProt (Pfam:PF00959)
4. Remove duplicates (CD-HIT at 100% identity)
5. Random split: 80% train / 10% val / 10% test
6. Create size-stratified training subsets (50, 100, 200, 500, 1000)

### Deliverables

| Deliverable | Location | Validation |
|-------------|----------|------------|
| AMP dataset | `data/finetune/amp/` | ≥2,000 unique sequences |
| Conotoxin dataset | `data/finetune/conotoxin/` | ≥1,000 unique sequences |
| Lysozyme dataset | `data/finetune/lysozyme/` | ≥3,000 unique sequences |
| Data prep script | `scripts/prepare_finetune_data.py` | Reproducible from raw downloads |

---

## Phase 6.1: Fine-Tuning Infrastructure

### New script: `scripts/finetune.py`

A lightweight fine-tuning script (standard causal LM, **no** distillation) that works for both teacher and student models.

```bash
python scripts/finetune.py \
    --model littleworth/protgpt2-distilled-tiny \
    --data_dir data/finetune/amp/ \
    --train_file train_500.fasta \
    --val_file val.fasta \
    --output_dir models/finetune/amp-tiny-500 \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --early_stopping_patience 3 \
    --wandb_project PROTGPT2_FINETUNE
```

### Key design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Optimizer | AdamW | Consistent with distillation training |
| Learning rate | Grid search per model size | Fair comparison requires per-model tuning |
| Early stopping | Patience=3 on val loss | Prevents overfitting, fair across model sizes |
| Max epochs | 20 | Enough for convergence on small data |
| LR schedule | Linear warmup (100 steps) + cosine decay | Standard practice |
| Gradient accumulation | Match effective batch size across models | Fair comparison |
| Weight decay | 0.01 | Standard regularization |

### Learning rate grid (per model)

| Model | LR candidates |
|-------|--------------|
| Teacher (738M) | 1e-5, 2e-5, 5e-5 |
| Medium (194M) | 2e-5, 5e-5, 1e-4 |
| Small (78M) | 5e-5, 1e-4, 2e-4 |
| Tiny (37M) | 1e-4, 2e-4, 5e-4 |

Select best LR per model on the full training set using val loss, then use that LR for all subset sizes.

### Deliverables

| Deliverable | Location | Validation |
|-------------|----------|------------|
| Fine-tune script | `scripts/finetune.py` | Runs on both teacher and student models |
| Evaluation additions | `scripts/evaluate_finetune.py` | Reports all Phase 6 metrics |

---

## Phase 6.2: Few-Shot Sample Efficiency (Primary Experiment)

### Design

Fine-tune each model on subsets of increasing size and measure how quickly each adapts to the target family.

**Models under comparison**:
- Teacher: `nferruz/ProtGPT2` (738M)
- Synergy-Medium: `littleworth/protgpt2-distilled-medium` (194M)
- Synergy-Small: `littleworth/protgpt2-distilled-small` (78M)
- Synergy-Tiny: `littleworth/protgpt2-distilled-tiny` (37M)
- Baseline-Tiny: `models/baseline-tiny` (37M, standard KD — no synergy)

Including Baseline-Tiny isolates the effect of synergy distillation from model size.

**Training subset sizes**: 50, 100, 200, 500, 1000, full

**Metrics evaluated on held-out test set**:

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| **Test perplexity** | Language modeling quality on target family | Standard next-token PPL on test.fasta |
| **Family hit rate** | % of generated sequences classified as target family | HMMER `hmmsearch` against family's Pfam HMM |
| **Sequence novelty** | % of generated sequences not in training set | Min edit distance to nearest training sequence > 0.3 |
| **Sequence diversity** | Diversity of generated pool | Mean pairwise edit distance among generated sequences |
| **pLDDT (if feasible)** | Structural plausibility | ESMFold pLDDT on generated sequences |

### Execution plan

```bash
# For each family × model × subset size
for family in amp conotoxin lysozyme; do
    for model in teacher medium small tiny baseline-tiny; do
        for n in 50 100 200 500 1000 full; do
            python scripts/finetune.py \
                --model $MODEL_PATH \
                --data_dir data/finetune/$family/ \
                --train_file train_${n}.fasta \
                --val_file val.fasta \
                --output_dir models/finetune/${family}-${model}-${n} \
                --epochs 20 \
                --early_stopping_patience 3

            python scripts/evaluate_finetune.py \
                --model models/finetune/${family}-${model}-${n} \
                --test_file data/finetune/$family/test.fasta \
                --family $family \
                --num_generate 200 \
                --output results/finetune/${family}-${model}-${n}.json
        done
    done
done
```

**Total runs**: 3 families × 5 models × 6 sizes = 90 fine-tuning runs

### Key figure: Sample Efficiency Curves

**Figure concept** (one panel per family):
- X-axis: Number of fine-tuning sequences (log scale)
- Y-axis: Test perplexity (or family hit rate)
- Lines: One per model, color-coded
- Expected result: Student curves plateau at lower N than teacher

### Success criteria for Phase 6.2

| Criterion | Threshold | Priority |
|-----------|-----------|----------|
| **S1**: Tiny synergy matches teacher test PPL with ≤50% of the data | Tiny at N=500 ≤ Teacher at N=1000 | Must-have |
| **S2**: Tiny synergy achieves ≥80% family hit rate at N=200 | hmmsearch hit rate ≥ 0.80 | Must-have |
| **S3**: Synergy-Tiny outperforms Baseline-Tiny at all subset sizes | Lower PPL at every N | Should-have |
| **S4**: Students show higher sequence diversity than teacher | Mean pairwise distance > teacher's | Should-have |
| **S5**: At least one student achieves teacher-level PPL at any N | PPL_student(N_s) ≤ PPL_teacher(N_full) for some N_s | Must-have |

---

## Phase 6.3: Overfitting & Catastrophic Forgetting Analysis

### 6.3.1 Overfitting analysis

For each fine-tuning run, record train loss and val loss at every epoch. Plot the **generalization gap** (train loss − val loss) over training.

**Expected result**: Teacher shows larger generalization gap (more overfitting) than students, especially at small N.

**Metric**: Max generalization gap across training epochs.

### 6.3.2 Catastrophic forgetting

After fine-tuning on a specific family, evaluate on the **original general protein test set** (same set used in the main paper).

| Metric | What it measures |
|--------|-----------------|
| General test PPL (post-fine-tune) | How much general capability is lost |
| AA distribution KL vs UniProt | Whether AA usage shifts away from natural |
| Forgetting ratio | PPL_after / PPL_before (>1 = forgetting) |

**Expected result**: Students may forget more (due to fewer parameters) or less (due to regularized representations from distillation). Either outcome is interesting and publishable.

### Success criteria for Phase 6.3

| Criterion | Threshold | Priority |
|-----------|-----------|----------|
| **S6**: Teacher shows ≥2x larger generalization gap than Tiny at N=100 | gap_teacher / gap_tiny ≥ 2.0 | Should-have |
| **S7**: Forgetting analysis completed for all models at N=500 | All forgetting ratios reported | Must-have |

---

## Phase 6.4: Conditional Generation Quality

After fine-tuning on each family (at N=full), generate 500 sequences per model and evaluate:

### Metrics

| Metric | Method | Tool |
|--------|--------|------|
| **Family hit rate** | `hmmsearch` against Pfam HMM | HMMER3 |
| **Novelty** | % sequences with min edit distance > 30% from any training sequence | Custom script |
| **Diversity** | Mean pairwise normalized edit distance among 500 generated sequences | Custom script |
| **Length distribution** | KL divergence of generated length distribution vs. training set | Custom script |
| **AA distribution** | KL divergence of AA frequencies vs. training family | Existing `evaluate.py` |
| **pLDDT** | Mean ESMFold pLDDT of top-50 by hit rate | `src/esmfold.py` |

### Key figure: Generation Quality Radar Chart

Spider/radar chart comparing teacher vs. Tiny synergy across the 5 metrics above (normalized to [0, 1]).

### Success criteria for Phase 6.4

| Criterion | Threshold | Priority |
|-----------|-----------|----------|
| **S8**: Tiny achieves ≥70% of teacher's family hit rate | hit_tiny / hit_teacher ≥ 0.70 | Must-have |
| **S9**: Tiny generates ≥50% novel sequences | novelty ≥ 0.50 | Must-have |
| **S10**: Tiny diversity within 20% of teacher's diversity | diversity_tiny / diversity_teacher ≥ 0.80 | Should-have |

---

## Phase 6.5: Fine-Tuning Cost Benchmark

### Measurements

For each model, fine-tune on AMP (N=1000) and measure:

| Metric | How |
|--------|-----|
| Wall-clock time per epoch | `time` wrapper |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` |
| Total fine-tuning time to convergence | Epochs × time/epoch |
| Minimum viable GPU | Can it fine-tune on 8 GB / 12 GB / 16 GB? |

### Key table

| Model | Params | Time/epoch | Peak GPU | Total time | Min GPU |
|-------|--------|-----------|----------|------------|---------|
| Teacher | 738M | — | — | — | — |
| Medium | 194M | — | — | — | — |
| Small | 78M | — | — | — | — |
| Tiny | 37M | — | — | — | — |

### Success criteria for Phase 6.5

| Criterion | Threshold | Priority |
|-----------|-----------|----------|
| **S11**: Tiny fine-tunes ≥5x faster per epoch than teacher | speedup ≥ 5.0 | Must-have |
| **S12**: Tiny fine-tunes within 8 GB GPU memory | peak_memory ≤ 8 GB | Must-have |
| **S13**: Teacher requires ≥16 GB for fine-tuning | peak_memory_teacher ≥ 16 GB | Should-have |

---

## Phase 6.6: Analysis, Figures & Paper Section

### Figures to produce

| Figure | Type | Key message |
|--------|------|-------------|
| **Fig A** | Line plot (sample efficiency curves) | Students match teacher with fewer examples |
| **Fig B** | Bar chart (overfitting gap) | Students overfit less |
| **Fig C** | Grouped bar chart (generation quality) | Fine-tuned students produce valid family-specific sequences |
| **Fig D** | Table/bar chart (fine-tuning cost) | Students are dramatically cheaper to fine-tune |
| **Fig E** | Radar chart (multi-metric comparison) | Overall advantage profile |

### Paper section outline

New section for paper: **"Domain adaptation advantage"** (insert after Practical deployment, before Structural quality or in Discussion)

```
\subsection{Domain adaptation advantage}

Paragraph 1: Motivation — real-world use requires fine-tuning on small proprietary datasets
Paragraph 2: Experimental setup — AMP dataset, subset sizes, metrics
Paragraph 3: Sample efficiency results — students match teacher with Nx fewer examples [Fig A]
Paragraph 4: Overfitting analysis — students show smaller generalization gap [Fig B]
Paragraph 5: Generation quality — fine-tuned students produce valid family-specific proteins [Fig C]
Paragraph 6: Cost analysis — Nx faster, fits on consumer GPU [Fig D]
```

### Deliverables

| Deliverable | Location |
|-------------|----------|
| Sample efficiency figure | `figures/pdf/fig_finetune_efficiency.pdf` |
| Overfitting figure | `figures/pdf/fig_finetune_overfitting.pdf` |
| Generation quality figure | `figures/pdf/fig_finetune_generation.pdf` |
| Cost benchmark figure/table | `figures/pdf/fig_finetune_cost.pdf` |
| Paper section LaTeX | `paper/sections/results.tex` (appended) |
| Raw results | `results/finetune/*.json` |

---

## Consolidated Success Criteria

### Must-Have (experiment is publishable)

| ID | Criterion | Metric |
|----|-----------|--------|
| S1 | Student matches teacher PPL with ≤50% of data | PPL at N vs. 2N |
| S2 | ≥80% family hit rate at N=200 | HMMER hit rate |
| S5 | At least one student reaches teacher-level PPL | PPL comparison at any N |
| S7 | Forgetting analysis completed | All ratios reported |
| S8 | Tiny achieves ≥70% of teacher's hit rate | Hit rate ratio |
| S9 | Tiny generates ≥50% novel sequences | Novelty fraction |
| S11 | Tiny fine-tunes ≥5x faster than teacher | Wall-clock speedup |
| S12 | Tiny fine-tunes within 8 GB GPU memory | Peak memory |

### Should-Have (strengthens the story)

| ID | Criterion | Metric |
|----|-----------|--------|
| S3 | Synergy-Tiny beats Baseline-Tiny at all N | PPL comparison |
| S4 | Students show higher diversity | Pairwise edit distance |
| S6 | Teacher overfits ≥2x more than Tiny | Generalization gap ratio |
| S10 | Tiny diversity within 20% of teacher | Diversity ratio |
| S13 | Teacher requires ≥16 GB for fine-tuning | Peak memory |

### Nice-to-Have (compelling extras)

| ID | Criterion | Metric |
|----|-----------|--------|
| S14 | Results replicate on ≥2 of 3 protein families | Cross-family consistency |
| S15 | pLDDT of fine-tuned student sequences ≥ teacher's | ESMFold pLDDT |
| S16 | Synergy calibration advantage persists after fine-tuning | Post-fine-tune ECE |

---

## Risk Mitigation

### If students don't outperform teacher at fine-tuning

1. **Verify LR tuning is fair** — re-run with expanded LR grid for teacher
2. **Try LoRA/adapter fine-tuning for teacher** — this is a weaker baseline since it adds complexity, but worth comparing
3. **Pivot the narrative**: Even if PPL is similar, the cost/memory advantage alone is a strong result ("same quality, 5x cheaper")
4. **Report honestly** — negative results about fine-tuning transferability are still publishable

### If family hit rates are low for all models

1. Verify HMMER profiles are correct (test on known family members first)
2. Increase fine-tuning data or epochs
3. Use sequence similarity (BLAST) instead of HMM as a softer metric
4. Consider that ProtGPT2 was trained on general UniProt, not optimized for specific families — low hit rate may be expected

### If overfitting patterns are unclear

1. Add explicit regularization (dropout sweep) to teacher fine-tuning
2. Use weight decay sweep
3. Report train/val curves directly — visual evidence is compelling even without crisp numerical thresholds

---

## Dependencies & Prerequisites

| Dependency | Status | Required for |
|------------|--------|-------------|
| Trained synergy models (Tiny, Small, Medium) | Done | All phases |
| Trained baseline-tiny model | Done | Phase 6.2 (S3) |
| HMMER3 installed | To verify | Phase 6.2, 6.4 |
| Pfam HMM profiles for target families | To download | Phase 6.2, 6.4 |
| ESMFold available | Done (`src/esmfold.py`) | Phase 6.4 (pLDDT) |
| W&B project | Exists | All phases (tracking) |

---

## Compute Budget Estimate

| Component | Runs | Est. time/run | Total GPU-hours |
|-----------|------|--------------|-----------------|
| LR grid search (4 models × 3 LRs) | 12 | ~30 min | 6 |
| Sample efficiency (5 models × 6 sizes × 3 families) | 90 | ~15 min avg | 22.5 |
| Generation + evaluation | 90 | ~10 min | 15 |
| Cost benchmarking | 4 | ~30 min | 2 |
| **Total** | | | **~46 GPU-hours** |

Fits within 2–3 days on a single L40S GPU.

---

## Critical Files

| File | Purpose |
|------|---------|
| `scripts/finetune.py` | New: domain-specific fine-tuning (no distillation) |
| `scripts/evaluate_finetune.py` | New: family-specific evaluation metrics |
| `scripts/prepare_finetune_data.py` | New: dataset download and preparation |
| `data/finetune/` | New: family-specific datasets |
| `results/finetune/` | New: fine-tuning experiment results |
| `src/distillation.py` | Existing: reference for training setup |
| `scripts/evaluate.py` | Existing: general evaluation (for forgetting analysis) |
| `src/esmfold.py` | Existing: pLDDT scoring |
