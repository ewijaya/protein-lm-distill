# Citation Audit Report

**Manuscript:** protein-lm-distill (ProtGPT2 knowledge distillation)
**Date:** 2026-02-17
**Scope:** All 19 entries in `paper/references.bib` cross-referenced against all `.tex` files in `paper/sections/`

---

## Executive Summary

| Severity | Count | Description |
|----------|-------|-------------|
| ERROR    | 8     | Factually wrong metadata or claims |
| WARNING  | 7     | Incomplete metadata, suspicious entries, potential misattribution |
| INFO     | 2     | Minor style issues |

**Critical findings:**
- 1 citation references a paper that does not exist as described (`spidergpt2025`)
- 1 citation points to the wrong paper entirely (`brandes2022proteinbert` → ProteinBERT instead of DistilProtBert)
- 1 claim misattributes hyperparameter recommendations to Hinton et al.
- 4 bib entries are never cited in the manuscript (orphan references)
- 3 bib entries have misspelled author names
- 0 broken `\cite{}` keys (all cite keys resolve to bib entries)

---

## Part 1: Bibliography Metadata Verification

### ERROR-level findings

#### E1. `spidergpt2025` — Paper does not exist as cited

- **Bib entry:** "SpiderGPT: Knowledge distillation of a spider silk protein language model" by "SpiderGPT Consortium", bioRxiv 2025
- **Reality:** No such paper exists on bioRxiv. The actual paper is:
  - **Title:** "Customizing Spider Silk: Generative Models with Mechanical Property Conditioning for Protein Engineering"
  - **Authors:** Neeru Dubey, Elin Karlsson, Miguel Angel Redondo, Johan Reimegard, Anna Rising, Hedvig Kjellstrom
  - **Venue:** arXiv (not bioRxiv), April 2025, arXiv:2504.08437
  - "SpiderGPT" is the name of the distilled model within the paper, not the paper title
- **Fix:** Rewrite the entire bib entry with correct title, authors, venue, and arXiv ID

#### E2. `wang2024mtdp` — Wrong authors and title

- **Bib entry:** author={Wang, Yijia and others}, title="MTDP: Multi-Teacher Knowledge Distillation for Protein Representations"
- **Reality:** The actual paper is:
  - **Title:** "Accurate and efficient protein embedding using multi-teacher distillation learning"
  - **Authors:** Jiayu Shang, Cheng Peng, Yongxin Ji, Jiaojiao Guan, Dehan Cai, Xubo Tang, Yanni Sun
  - **Journal:** Bioinformatics, Volume 40, Issue 9, article btae567, September 2024
  - There is no "Wang, Yijia" among the authors
- **Fix:** Replace authors entirely, correct title, add volume=40, number=9, pages=btae567

#### E3. `lin2023esmfold` — Misspelled author names

- **Bib entry:** contains "Smeber, Nikita" and "Candber, Sal"
- **Reality:**
  - "Smeber, Nikita" → **"Smetanin, Nikita"**
  - "Candber, Sal" → **"Candido, Salvatore"**
- All other metadata (Science, vol 379, issue 6637, pages 1123–1130) is correct
- **Fix:** Correct the two author names

#### E4. `naeini2015ece` — Wrong word in title

- **Bib entry:** "Obtaining Well Calibrated **Predictions** Using Bayesian Binning into Quantiles"
- **Reality:** "Obtaining Well Calibrated **Probabilities** Using Bayesian Binning into Quantiles"
- Pages (2901–2907) and other metadata are correct
- **Fix:** Change "Predictions" to "Probabilities"

#### E5. `madani2023progen` — Misspelled author name

- **Bib entry:** "Sun, Zhz Z"
- **Reality:** **"Sun, Zachary Z"**
- All other metadata (Nature Biotechnology, vol 41, pages 1099–1106) is correct
- **Fix:** Correct "Zhz Z" to "Zachary Z"

#### E6. `uniprot2023` — Wrong page numbers

- **Bib entry:** pages={D483--D489}
- **Reality:** pages=**{D523--D531}** (Nucleic Acids Research, Volume 51, Issue D1, 2023)
- **Fix:** Change pages to D523–D531

### Entries verified correct (13 of 19)

| Key | Title | Verdict |
|-----|-------|---------|
| `hinton2015distilling` | Distilling the Knowledge in a Neural Network | Correct |
| `ferruz2022protgpt2` | ProtGPT2 is a deep unsupervised language model for protein design | Correct |
| `rives2021biological` | Biological structure and function emerge from scaling... | Correct |
| `sanh2019distilbert` | DistilBERT, a distilled version of BERT | Correct |
| `brandes2022proteinbert` | ProteinBERT: a universal deep-learning model... | Correct (but see E7 below — wrong paper is being cited) |
| `guo2017calibration` | On Calibration of Modern Neural Networks | Correct |
| `muller2019label` | When Does Label Smoothing Help? | Correct |
| `romero2015fitnets` | FitNets: Hints for Thin Deep Nets | Correct |
| `park2019relational` | Relational Knowledge Distillation | Correct |
| `jiao2020tinybert` | TinyBERT: Distilling BERT for NLU | Correct |
| `vaswani2017attention` | Attention Is All You Need | Correct |
| `radford2019language` | Language Models are Unsupervised Multitask Learners | Correct |
| `hie2024antibodies` | Efficient evolution of human antibodies... | Correct |

---

## Part 2: Claim Verification

### ERROR-level findings

#### E7. `brandes2022proteinbert` — Wrong paper cited for "DistilProtBERT"

- **Manuscript text** (introduction.tex:32): "DistilProtBERT~\cite{brandes2022proteinbert} compressed ESM-style models"
- **Problem:** The bib entry `brandes2022proteinbert` points to **ProteinBERT** (Brandes et al. 2022, Bioinformatics), which is a universal deep-learning model — **not** a distillation paper.
- **The actual DistilProtBert paper** is: Geffen, Ofran, and Unger, "DistilProtBert: A distilled protein language model used to distinguish between real proteins and randomly generated amino acid sequences", Bioinformatics 2022 (Supplement 2, ISMB proceedings).
- **Additional error:** The manuscript claims DistilProtBERT "compressed ESM-style models", but DistilProtBert actually distills **ProtBert** (from the ProtTrans family), not ESM.
- **Fix:** (1) Replace the bib entry with the correct Geffen et al. 2022 citation. (2) Change "compressed ESM-style models" to "compressed ProtBert" or "compressed protein BERT-style models."

#### E8. `hinton2015distilling` — Misattributed hyperparameter defaults

- **Manuscript text** (methods.tex:158–159): "We use Hinton et al.'s recommended defaults: temperature τ = 2.0 and balancing coefficient α = 0.5"
- **Problem:** Hinton et al. do NOT recommend T=2.0 and α=0.5 as defaults. The original paper:
  - Uses temperatures ranging from 1 to 20 (T=20 for MNIST, varied for speech recognition)
  - Explicitly states the best results came with α "much smaller" than the soft-loss weight (i.e., α << 0.5)
- **Fix:** Remove "Hinton et al.'s recommended defaults" attribution. Reword to: "We set temperature τ = 2.0 and balancing coefficient α = 0.5" or "Following common practice in knowledge distillation, we set..."

### WARNING-level findings

#### W1. `madani2023progen` — Reductive characterization as "enzyme engineering"

- **Manuscript text** (introduction.tex:18–19): "enzyme engineering campaigns~\cite{madani2023progen}"
- **Issue:** ProGen is a general-purpose protein language model trained on 280M sequences across >19,000 families. While it demonstrates functional enzyme generation (lysozymes), describing it solely as an "enzyme engineering" tool is reductive.
- **Suggested fix:** "protein engineering campaigns~\cite{madani2023progen}" or acknowledge it demonstrates enzyme engineering among other capabilities

#### W2. `spidergpt2025` — Wrong sequence count in manuscript text

- **Manuscript text** (introduction.tex:36): "592 spider silk sequences"
- **Reality:** The actual paper uses **572** MaSp repeats with mechanical properties (plus 6,000 MaSp repeats in earlier training)
- **Fix:** Change "592" to "572" (or verify against the actual paper — may be 572 or a related number)

#### W3. `spidergpt2025` — "SpiderGPT" name not used in original paper

- **Issue:** The manuscript refers to "SpiderGPT~\cite{spidergpt2025}" but the actual paper (Dubey et al. 2025) does not use the name "SpiderGPT" — it describes "a lightweight GPT-based generative model." Using an informal name not coined by the original authors may cause confusion.
- **Suggested fix:** Use the model description from the actual paper, or note that "SpiderGPT" is an informal shorthand

---

## Part 3: Cross-Reference Analysis

### Broken references: 0
All 15 `\cite{}` keys used in the manuscript resolve to entries in `references.bib`.

### WARNING: Orphan references (4 entries never cited)

| Key | Paper | Potential use |
|-----|-------|--------------|
| `sanh2019distilbert` | Sanh et al., "DistilBERT" (2019) | Cite in intro alongside DistilProtBERT — it's the direct inspiration |
| `romero2015fitnets` | Romero et al., "FitNets" (2015) | Cite in methods to justify choosing response-based over feature-based KD |
| `jiao2020tinybert` | Jiao et al., "TinyBERT" (2020) | Cite in intro to establish NLP distillation context |
| `park2019relational` | Park et al., "Relational Knowledge Distillation" (2019) | Cite in methods to acknowledge alternative KD approaches |

**Recommendation:** Either cite these in the introduction/methods to strengthen related work coverage, or remove them from `references.bib` to keep the bibliography clean.

### INFO: Suggested additional citations

#### I1. Introduction could cite NLP distillation predecessors

The claim "no systematic study has addressed distillation for general-purpose autoregressive protein language models" (introduction.tex:37–39) would be strengthened by first establishing that NLP distillation has been successful (DistilBERT, TinyBERT), making the gap protein-specific.

#### I2. `radford2019language` venue style

The venue "OpenAI Blog" is unconventional. Some citation styles use "OpenAI Technical Report" instead. This is purely cosmetic.

---

## Consolidated Fix List (by priority)

| # | Severity | Entry | Action Required |
|---|----------|-------|-----------------|
| 1 | ERROR | `spidergpt2025` | Rewrite entire bib entry: correct title (Dubey et al.), authors, venue (arXiv:2504.08437). Fix "592" → "572" in manuscript text. |
| 2 | ERROR | `brandes2022proteinbert` | Replace with correct DistilProtBert citation (Geffen, Ofran, Unger 2022). Fix "ESM-style models" → "ProtBert" in introduction.tex:32. |
| 3 | ERROR | `hinton2015distilling` | Remove "recommended defaults" attribution in methods.tex:158–159. Reword to "We set" or "Following common practice." |
| 4 | ERROR | `wang2024mtdp` | Replace authors (Shang et al.), fix title, add volume=40, number=9, pages=btae567. |
| 5 | ERROR | `lin2023esmfold` | Fix "Smeber, Nikita" → "Smetanin, Nikita" and "Candber, Sal" → "Candido, Salvatore". |
| 6 | ERROR | `naeini2015ece` | Fix title: "Predictions" → "Probabilities". |
| 7 | ERROR | `madani2023progen` | Fix author: "Sun, Zhz Z" → "Sun, Zachary Z". |
| 8 | ERROR | `uniprot2023` | Fix pages: D483–D489 → D523–D531. |
| 9 | WARNING | `madani2023progen` | Consider changing "enzyme engineering campaigns" to "protein engineering campaigns" in text. |
| 10 | WARNING | `spidergpt2025` | Note that "SpiderGPT" is not the paper's own terminology. |
| 11 | WARNING | `spidergpt2025` | Manuscript says "592 sequences" — actual paper says 572. |
| 12 | WARNING | `sanh2019distilbert` | Orphan reference — cite or remove. |
| 13 | WARNING | `romero2015fitnets` | Orphan reference — cite or remove. |
| 14 | WARNING | `jiao2020tinybert` | Orphan reference — cite or remove. |
| 15 | WARNING | `park2019relational` | Orphan reference — cite or remove. |
| 16 | INFO | `radford2019language` | "OpenAI Blog" venue is unconventional (cosmetic). |
| 17 | INFO | Introduction | Could cite NLP distillation predecessors for stronger gap framing. |
