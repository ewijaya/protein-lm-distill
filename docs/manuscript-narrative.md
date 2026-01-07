# Manuscript Narrative: Uncertainty-Aware Knowledge Distillation for Protein Language Models

**Document Purpose**: Comprehensive reference for manuscript writing
**Last Updated**: January 7, 2026
**Target Venue**: Nature Communications (primary), PNAS/Bioinformatics (backup)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Manuscript Story](#2-the-manuscript-story)
3. [Scientific Contributions](#3-scientific-contributions)
4. [Experiment Plan](#4-experiment-plan)
5. [Current Progress](#5-current-progress)
6. [Planned Results & Figures](#6-planned-results--figures)
7. [Success Criteria](#7-success-criteria)
8. [Paper Structure](#8-paper-structure)
9. [Related Work & Positioning](#9-related-work--positioning)
10. [Timeline & Milestones](#10-timeline--milestones)
11. [Risk Mitigation](#11-risk-mitigation)
12. [Supplementary Considerations](#12-supplementary-considerations)

---

## 1. Executive Summary

### One-Paragraph Story

Large protein language models like ProtGPT2 (738M parameters) have revolutionized computational protein design but remain computationally prohibitive for practical deployment. We present the **first systematic study of knowledge distillation for autoregressive protein language models**, introducing two protein-specific innovations beyond standard Hinton-style distillation: (1) **uncertainty-aware position weighting** that emphasizes biologically variable regions during training, and (2) **calibration-aware distillation** that produces well-calibrated confidence estimates essential for wet-lab prioritization. Our compressed models (39M-200M parameters) achieve 6-18× compression with <2× perplexity increase while maintaining biological validity and improving calibration, enabling practical protein design on consumer hardware.

### Key Numbers (Target)

| Metric | Target |
|--------|--------|
| Compression ratio | 6-18× |
| Perplexity ratio | <2.0 (size-dependent) |
| Inference speedup | 2-6× |
| ECE improvement | 20-30% vs baseline |
| pLDDT preservation | >70 mean score |

---

## 2. The Manuscript Story

### 2.1 The Problem (Introduction Hook)

**Opening narrative**: Protein language models have emerged as powerful tools for *de novo* protein design, with ProtGPT2 demonstrating the ability to generate sequences with natural-like properties. However, their computational demands create a practical barrier:

- **738M parameters** require high-end GPUs
- **Slow inference** (~3 seconds per sequence) limits throughput
- **Deployment impossible** on edge devices or resource-constrained environments
- **Cost-prohibitive** for high-throughput screening

**The tension**: How do we democratize access to powerful protein generation while preserving the capabilities that make these models valuable?

### 2.2 The Gap (Why This Matters)

**Literature landscape**:
- **Masked LM distillation exists**: DistilProtBERT (2022), MTDP (2024)
- **Domain-specific causal LM**: SpiderGPT (2025) - spider silk only
- **General causal protein LM distillation**: **NONE**

**Critical insight**: Autoregressive (causal) protein LMs are fundamentally different from masked LMs:
- Generate complete sequences from scratch
- Enable *de novo* design, not just representation learning
- Require different distillation considerations (sequential dependencies)

### 2.3 Our Solution (The Innovation)

**Base method**: Standard Hinton-style knowledge distillation (established, well-understood)

**Two protein-specific innovations**:

#### Innovation #1: Uncertainty-Aware Position Weighting
```
Insight: Protein sequences have heterogeneous difficulty
- Conserved positions (buried cores) → Low teacher entropy → Easier
- Variable positions (loops, active sites) → High teacher entropy → Harder

Method: Weight distillation loss by teacher uncertainty
- w(t) = 0.5 + 0.5 × normalize(entropy(t))
- Focus student learning on biologically meaningful variability
```

#### Innovation #2: Calibration-Aware Distillation
```
Insight: Wet-lab requires reliable confidence estimates
- Overconfident predictions waste experimental resources
- Need accurate confidence for sequence prioritization

Method: Dynamic label smoothing based on teacher confidence
- ε(t) = λ × (1 - max(p_teacher))
- Low-confidence predictions → more smoothing
- Produces well-calibrated student models
```

### 2.4 The Resolution (What We Achieve)

**Compression without compromise**:
- 6-18× smaller models
- 2-6× faster inference
- Biological validity preserved (AA distributions, structural plausibility)

**Better than baseline**:
- Uncertainty weighting improves perplexity on difficult sequences
- Calibration smoothing produces reliable confidence for experiments

**Practical impact**:
- Protein design on consumer GPUs
- High-throughput virtual screening enabled
- Democratized access to protein language models

---

## 3. Scientific Contributions

### 3.1 Primary Contributions

| # | Contribution | Novelty Level |
|---|--------------|---------------|
| 1 | First systematic study of distillation for autoregressive protein LMs | Application novelty |
| 2 | Uncertainty-aware position weighting for protein sequences | Methodological innovation |
| 3 | Calibration-aware distillation for experimental prioritization | Methodological innovation |
| 4 | Comprehensive evaluation framework (perplexity, KL, ECE, pLDDT, AA dist) | Practical contribution |
| 5 | Open-source models and code | Community contribution |

### 3.2 What We Claim vs. What We Don't

**We claim**:
- ✅ First comprehensive general-purpose study
- ✅ Novel protein-specific enhancements
- ✅ Rigorous evaluation methodology
- ✅ Practical deployment guidance

**We don't claim**:
- ❌ Novel distillation algorithm (base method is Hinton 2015)
- ❌ State-of-the-art protein generation (we match, not exceed, teacher)
- ❌ Structural innovation (we use established transformer architectures)

---

## 4. Experiment Plan

### 4.1 Phase Overview

| Phase | Description | Status | Duration |
|-------|-------------|--------|----------|
| **Phase 0** | Methodological enhancements implementation + ablation | ⏳ Ablation running | 2-3 days |
| **Phase 1** | Baseline training (4 model sizes) | ✅ Complete | ~8 hours |
| **Phase 2** | Hyperparameter sweeps (T, α) | ⏸️ Pending | 2-3 days |
| **Phase 3** | Comprehensive evaluation | ⏸️ Pending | 1 day |
| **Phase 4** | HuggingFace model update | ⏸️ Pending | 1 day |
| **Phase 5** | Manuscript writing | ⏸️ Pending | 3-5 days |

### 4.2 Phase 0: Ablation Study (Critical for Paper)

**Objective**: Demonstrate that each enhancement provides measurable improvement

**Experimental Design**:
| Variant | Uncertainty Weighting | Calibration Smoothing | Output Dir |
|---------|----------------------|----------------------|------------|
| Baseline | ✗ | ✗ | `models/ablation-baseline` |
| +Uncertainty | ✓ | ✗ | `models/ablation-uncertainty` |
| +Calibration | ✗ | ✓ | `models/ablation-calibration` |
| +Both (Full) | ✓ | ✓ | `models/ablation-both` |

**Architecture for ablation**: Tiny (4L/4H/256E) - fastest iteration

**Metrics to compare**:
- Perplexity ratio (student/teacher)
- KL divergence from teacher
- ECE (Expected Calibration Error)
- AA distribution KL from natural

### 4.3 Phase 2: Hyperparameter Sweeps

**Temperature sweep** (with best enhancement config):
```
T ∈ {1.0, 2.0, 4.0, 6.0, 8.0, 10.0}
```
- SpiderGPT uses T=10, our current default is T=2.0
- Need to find optimal for general protein generation

**Alpha sweep** (with best T):
```
α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
```
- α=0.1: Heavy soft loss (teacher-focused)
- α=0.9: Heavy hard loss (ground-truth focused)
- SpiderGPT uses α=0.1, our default is α=0.5

### 4.4 Phase 3: Final Model Training

**Target architectures** (matching HuggingFace):
| Model | Layers | Heads | Embed | Est. Params | HF Repo |
|-------|--------|-------|-------|-------------|---------|
| Tiny | 4 | 4 | 512 | ~39M | littleworth/protgpt2-distilled-tiny |
| Small | 6 | 8 | 768 | ~82M | littleworth/protgpt2-distilled-small |
| Medium | 12 | 16 | 1024 | ~200M | littleworth/protgpt2-distilled-medium |

**Note**: Current trained models have smaller embeddings - final training will use HF-matching dimensions.

---

## 5. Current Progress

### 5.1 Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| DistillationTrainer | ✅ Complete | `src/distillation.py` |
| Uncertainty weighting | ✅ Implemented | `src/distillation.py:77-165` |
| Calibration smoothing | ✅ Implemented | `src/distillation.py:167-203` |
| ECE computation | ✅ Implemented | `scripts/evaluate.py:30-137` |
| Training script | ✅ Complete | `scripts/train.py` |
| Evaluation script | ✅ Complete | `scripts/evaluate.py` |
| ESMFold pLDDT | ⚠️ Partial | `src/esmfold.py` (not in eval pipeline) |
| METHODS.md | ✅ Complete | `docs/METHODS.md` |

### 5.2 Training Status

**Phase 1 Models Trained**:
| Model | Config | Date | Directory |
|-------|--------|------|-----------|
| XS | 2L/2H/128E | Dec 25 | `protgpt2-distilled-t2.0-a0.5-l2-h2-e128-p0.001-lr1e-03.uniprot` |
| Tiny | 4L/4H/256E | Dec 29 | `protgpt2-distilled-t2.0-a0.5-l4-h4-e256-p0.1-lr1e-03.uniprot` |
| Small | 6L/8H/512E | Jan 1 | `protgpt2-distilled-t2.0-a0.5-l6-h8-e512-p0.1-lr5e-04.uniprot` |
| Medium | 12L/12H/768E | Jan 7 | `protgpt2-distilled-t2.0-a0.5-l12-h12-e768-p0.1-lr1e-04.uniprot` |

**Phase 0 Ablation Status**:
- ⏳ `+Uncertainty` variant: Currently running
- ⏸️ `+Calibration` variant: Queued
- ⏸️ `+Both` variant: Queued

### 5.3 Preliminary Results (from notebooks)

**Speed benchmarks** (from `compare_student_teacher.ipynb`):
| Model | Inference Time | Speedup |
|-------|---------------|---------|
| Teacher (ProtGPT2) | ~2.85s | 1.0× |
| Tiny (l4-h4-e512) | ~0.47s | **6.1×** |
| Small (l6-h8-e768) | ~0.64s | **4.5×** |
| Medium (l12-h16-e1024) | ~1.08s | **2.6×** |

**Perplexity observations**:
- High variance across sequences noted
- "Student beats teacher" effect observed in some cases
- Need more rigorous evaluation with proper test set

---

## 6. Planned Results & Figures

### 6.1 Main Figures

| Fig # | Title | Content | Status |
|-------|-------|---------|--------|
| **Fig 1** | Ablation Study | Bar chart: Baseline vs +Uncertainty vs +Calibration vs +Both | ⏸️ Awaiting ablation results |
| **Fig 2** | Temperature Sweep | Line plot: T vs Perplexity Ratio (with/without enhancements) | ⏸️ Awaiting Phase 2 |
| **Fig 3** | Alpha Sweep | Line plot: α vs Perplexity Ratio | ⏸️ Awaiting Phase 2 |
| **Fig 4** | Compression-Quality Tradeoff | Pareto frontier: Compression ratio vs PPL ratio | ⏸️ Awaiting Phase 3 |
| **Fig 5** | Calibration Analysis | Reliability diagram + ECE comparison | ⏸️ Awaiting ablation |
| **Fig 6** | AA Distribution | Heatmap: Teacher vs Students vs Natural | ⏸️ Awaiting Phase 3 |
| **Fig 7** | Inference Speed | Bar chart: Time per sequence by model size | ✅ Preliminary data exists |
| **Fig 8** | Uncertainty Visualization | Heatmap: Position-wise entropy and weights | ⏸️ Need to generate |

### 6.2 Main Tables

| Table # | Title | Content | Status |
|---------|-------|---------|--------|
| **Table 1** | Model Configurations | Architecture details (L, H, E, params) | ✅ Can generate |
| **Table 2** | Ablation Results | PPL ratio, KL div, ECE for 4 variants | ⏸️ Awaiting ablation |
| **Table 3** | Final Model Evaluation | Comprehensive metrics for Tiny/Small/Medium | ⏸️ Awaiting Phase 3 |
| **Table 4** | Calibration Metrics | ECE, MCE, Brier score comparison | ⏸️ Awaiting ablation |
| **Table 5** | Comparison with Prior Work | vs DistilProtBERT, MTDP, SpiderGPT | ⏸️ Need literature metrics |
| **Table 6** | Inference Benchmarks | Speed, memory, GPU requirements | ✅ Preliminary data exists |

### 6.3 Expected Result Patterns

**Ablation study (Fig 1 / Table 2)**:
```
Expected pattern:
- Baseline: PPL ratio ~1.8, ECE ~0.08
- +Uncertainty: PPL ratio ~1.6 (10% improvement), ECE ~0.07
- +Calibration: PPL ratio ~1.75, ECE ~0.05 (30% improvement)
- +Both: PPL ratio ~1.55 (15% improvement), ECE ~0.045 (40% improvement)
```

**Temperature analysis (Fig 2)**:
```
Expected pattern:
- T=1.0: Highest PPL ratio (too sharp)
- T=2.0-4.0: Optimal range
- T=10.0: Good but potentially over-soft
```

**Calibration reliability diagram (Fig 5)**:
```
Expected pattern:
- Baseline: Significant overconfidence (curve above diagonal)
- +Calibration: Closer to diagonal
- +Both: Best calibration
```

---

## 7. Success Criteria

### 7.1 Must-Have (MVP for Publication)

| Criterion | Metric | Target | Priority |
|-----------|--------|--------|----------|
| Ablation validates enhancements | PPL improvement | >10% for at least one enhancement | Critical |
| Calibration improvement | ECE reduction | >20% vs baseline | Critical |
| Compression maintained | Size ratio | 6-18× smaller than teacher | Critical |
| Speed improvement | Inference time | >2× faster | Critical |
| Biological validity | AA distribution KL | <0.05 from natural | Critical |

### 7.2 Size-Dependent Perplexity Thresholds

| Model | Params | % of Teacher | PPL Ratio Target |
|-------|--------|--------------|------------------|
| Tiny | ~39M | 5% | < 3.0 |
| Small | ~82M | 11% | < 2.5 |
| Medium | ~200M | 27% | < 2.0 |

**Rationale**: Smaller models have more aggressive compression; expecting teacher-level performance is unrealistic.

### 7.3 Should-Have (Strengthens Paper)

- [ ] Optimal hyperparameters identified (T, α sweeps)
- [ ] pLDDT structural evaluation
- [ ] Position-wise uncertainty visualization
- [ ] Reliability diagrams for calibration
- [ ] All three model sizes updated on HuggingFace

### 7.4 Nice-to-Have (Bonus)

- [ ] Comparison with SpiderGPT on general proteins
- [ ] Interactive demo on HuggingFace Spaces
- [ ] Multi-teacher distillation experiments
- [ ] On-policy distillation comparison

---

## 8. Paper Structure

### 8.1 Proposed Title Options

1. **"Uncertainty-Aware Knowledge Distillation for Autoregressive Protein Language Models"** (Recommended)
2. "Protein-Specific Knowledge Distillation: Uncertainty-Aware and Calibration-Conscious Compression of ProtGPT2"
3. "Efficient Protein Generation via Knowledge Distillation with Biological Priors"

### 8.2 Section Outline

**Abstract** (~250 words)
- Gap: No general-purpose causal protein LM distillation
- Method: Hinton-style + two protein-specific innovations
- Results: Key numbers (compression, speedup, calibration improvement)
- Impact: Democratized protein design

**1. Introduction** (1.5-2 pages)
- 1.1 Protein LMs for sequence generation
- 1.2 Computational challenges of large models
- 1.3 Knowledge distillation as solution
- 1.4 Gap: No systematic study for causal protein LMs
- 1.5 Our contributions (numbered list)

**2. Related Work** (1 page)
- 2.1 Protein language models (ProtGPT2, ESM, ProGen)
- 2.2 Knowledge distillation paradigms (response, feature, relation)
- 2.3 Protein model compression (DistilProtBERT, MTDP, SpiderGPT)
- 2.4 Positioning: First general-purpose causal protein LM distillation

**3. Methods** (3-4 pages)
- 3.1 Standard distillation framework (Hinton 2015)
- 3.2 Temperature-scaled soft targets
- 3.3 Combined loss: L = αL_hard + (1-α)T²L_soft
- 3.4 **Enhancement #1: Uncertainty-aware position weighting**
- 3.5 **Enhancement #2: Calibration-aware distillation**
- 3.6 Model architectures (Tiny, Small, Medium)
- 3.7 Evaluation metrics (PPL, KL, ECE, AA dist, pLDDT)

**4. Experiments** (3-4 pages)
- 4.1 Experimental setup (data, hardware, hyperparameters)
- 4.2 **Ablation study: Validating enhancements**
- 4.3 Hyperparameter analysis (T, α sweeps)
- 4.4 Final model comparison
- 4.5 Calibration analysis
- 4.6 Generation quality (AA distribution, structural plausibility)
- 4.7 Inference benchmarks

**5. Discussion** (1.5 pages)
- 5.1 Why protein-specific enhancements matter
- 5.2 Comparison with prior work
- 5.3 Limitations
- 5.4 Future directions

**6. Conclusion** (0.5 page)

**Supplementary Material**
- Extended ablation results
- Per-bin ECE statistics
- Additional generation examples
- Full hyperparameter tables

---

## 9. Related Work & Positioning

### 9.1 Direct Competitors

| Work | Type | Domain | Published | Our Advantage |
|------|------|--------|-----------|---------------|
| DistilProtBERT | Masked LM distillation | General | 2022 | We do causal (generative) LM |
| MTDP | Multi-teacher masked LM | General | 2024 | We do causal LM, single teacher |
| SpiderGPT | Causal LM distillation | Spider silk only | 2025 | We are general-purpose |
| AMPLIFY | Efficient training | General | 2024 | Not distillation (different approach) |

### 9.2 Key Differentiators

**vs. SpiderGPT**:
- SpiderGPT: Domain-specific (592 spider silk sequences)
- Ours: General-purpose (UniProt, millions of sequences)
- SpiderGPT: T=10, α=0.1 (fixed)
- Ours: Systematic hyperparameter study + protein-specific enhancements

**vs. Masked LM work**:
- Masked LMs: Good for embeddings/property prediction
- Causal LMs: Required for *de novo* generation
- Different distillation challenges (sequential dependencies)

### 9.3 Positioning Statement

> "We present the first systematic study of knowledge distillation for autoregressive protein language models, introducing uncertainty-aware position weighting and calibration-conscious techniques specifically tailored for protein sequence generation. Unlike domain-specific approaches (SpiderGPT) or masked language model distillation (DistilProtBERT, MTDP), our method enables general-purpose protein design with well-calibrated confidence estimates for experimental prioritization."

---

## 10. Timeline & Milestones

### 10.1 Remaining Work

| Week | Tasks | Deliverables |
|------|-------|--------------|
| **Week 1** | Complete Phase 0 ablation, run Phase 2 sweeps | Ablation results table, optimal T and α |
| **Week 2** | Phase 3 evaluation, final model training | Comprehensive metrics, final models |
| **Week 3** | HuggingFace update, figure generation | Updated HF repos, all figures |
| **Week 4-5** | Manuscript writing | Complete draft |
| **Week 6** | Internal review, bioRxiv submission | Preprint live |
| **Week 7+** | Journal submission, revisions | Under review |

### 10.2 Critical Path

```
Ablation training → Ablation evaluation → Identify best config
                                              ↓
                              Hyperparameter sweeps with best config
                                              ↓
                              Final model training (HF architectures)
                                              ↓
                              Comprehensive evaluation + figures
                                              ↓
                              Manuscript writing
```

---

## 11. Risk Mitigation

### 11.1 Technical Risks

| Risk | Mitigation |
|------|------------|
| Enhancements don't improve over baseline | Show negative results honestly; focus on systematic study contribution |
| pLDDT scores are low | Use longer sequences; compare fairly with teacher under same conditions |
| New models underperform existing HF models | Try matching SpiderGPT hyperparameters (T=10, α=0.1); increase training data |
| Hyperparameter sweeps take too long | Use 5% data for sweeps, validate best on 10% |

### 11.2 Decision Points

**After ablation completes**:
- If neither enhancement helps → Pivot to "systematic study" framing
- If only one helps → Focus paper on that enhancement
- If both help → Full paper as planned

**After hyperparameter sweeps**:
- If current T=2.0 is optimal → Proceed
- If T=10 is better → Retrain with higher temperature
- If none reach PPL threshold → Increase training data proportion

---

## 12. Supplementary Considerations

### 12.1 Items to Add to Paper

**Reproducibility**:
- [ ] Code availability statement
- [ ] Data availability (UniProt subset description)
- [ ] Computing resources used
- [ ] Random seeds and variance

**Broader Impact**:
- [ ] Positive: Democratized protein design
- [ ] Positive: Reduced computational carbon footprint
- [ ] Potential concern: Misuse for harmful proteins (but low risk given public teacher model)

**Limitations section**:
- Single teacher model (ProtGPT2)
- No comparison with other protein generation methods (ProGen, etc.)
- ECE computed on next-token, not sequence-level
- pLDDT is predicted, not experimental validation

### 12.2 Author Contributions (CRediT)

- Conceptualization: [Author]
- Methodology: [Author]
- Software: [Author]
- Validation: [Author]
- Formal analysis: [Author]
- Writing - Original Draft: [Author]
- Writing - Review & Editing: [Author]

### 12.3 Data & Code Availability

**GitHub Repository**: To be made public upon publication
- Training scripts
- Evaluation code
- Trained model checkpoints
- Jupyter notebooks for reproduction

**HuggingFace Models**:
- littleworth/protgpt2-distilled-tiny
- littleworth/protgpt2-distilled-small
- littleworth/protgpt2-distilled-medium

### 12.4 Acknowledgments Template

> This work was supported by [funding]. Computational resources were provided by [AWS/cloud]. We thank the authors of ProtGPT2 for releasing their model and the UniProt consortium for protein sequence data.

---

## Appendix: Quick Reference Commands

### Monitor Current Training
```bash
tail -f phase0_ablation.log
```

### Run Ablation Evaluation (After Training)
```bash
mkdir -p results
python scripts/evaluate.py --student_model ./models/ablation-baseline --num_samples 100 --compute_ece --output results/ablation_baseline.json
python scripts/evaluate.py --student_model ./models/ablation-uncertainty --num_samples 100 --compute_ece --output results/ablation_uncertainty.json
python scripts/evaluate.py --student_model ./models/ablation-calibration --num_samples 100 --compute_ece --output results/ablation_calibration.json
python scripts/evaluate.py --student_model ./models/ablation-both --num_samples 100 --compute_ece --output results/ablation_both.json
```

### Generate Ablation Summary
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
        ece = d.get('student_ece', {}).get('ece', 'N/A')
        print(f'{v:<22} {ppl:>10.4f} {kl:>10.4f} {ece:>10.4f}')
"
```

---

**Document End**

*This narrative serves as the comprehensive reference for manuscript writing. Update sections as experiments complete and results become available.*
