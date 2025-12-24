# Knowledge Distillation for Protein Language Model Compression

## Abstract

We present a knowledge distillation framework for compressing ProtGPT2, a transformer-based protein language model, into smaller, efficient student models that retain generative capabilities for protein sequence design. Our approach combines temperature-scaled soft targets with hard label supervision, enabling student models to capture both the teacher's learned probability distributions and ground-truth sequence patterns. We provide rigorous mathematical formulations with corresponding implementation references, establishing a reproducible methodology for protein language model compression.

---

## 1. Introduction

### 1.1 Motivation

Large protein language models (pLMs) such as ProtGPT2 have demonstrated remarkable capabilities in generating novel protein sequences with natural-like properties. However, their computational requirements—approximately 738 million parameters for ProtGPT2—limit deployment in resource-constrained environments and impede rapid prototyping in protein engineering workflows.

Knowledge distillation offers a principled approach to model compression by training a smaller "student" model to mimic the behavior of a larger "teacher" model. Unlike pruning or quantization, distillation can achieve substantial compression ratios (10-50×) while preserving the essential learned representations that enable high-quality sequence generation.

### 1.2 Problem Statement

Given a pre-trained teacher model $T$ with parameters $\theta_T$ and a student model $S$ with parameters $\theta_S$ where $|\theta_S| \ll |\theta_T|$, our objective is to find optimal student parameters:

$$\theta_S^* = \arg\min_{\theta_S} \mathcal{L}(\theta_S; \theta_T, \mathcal{D})$$

where $\mathcal{L}$ is the distillation loss and $\mathcal{D}$ is the training corpus of protein sequences.

### 1.3 Contributions

1. A complete knowledge distillation framework for autoregressive protein language models
2. Rigorous mathematical formulation with gradient analysis for temperature scaling
3. Evaluation methodology encompassing perplexity, distributional similarity, and structural plausibility
4. Open-source implementation with explicit formula-to-code mappings

---

## 2. Related Work

### 2.1 Knowledge Distillation Paradigms

Knowledge distillation methods can be categorized into three paradigms based on the type of knowledge transferred:

#### 2.1.1 Response-Based Distillation

Introduced by Hinton et al. (2015), response-based distillation transfers knowledge through the teacher's output probability distributions. The key insight is that "soft" probability distributions contain richer information than one-hot labels—they encode inter-class relationships and uncertainty. For protein sequences, this means learning that certain amino acid substitutions are more probable than others at each position.

**Our method adopts this paradigm**, using temperature-scaled softmax distributions as soft targets.

#### 2.1.2 Feature-Based Distillation

FitNets (Romero et al., 2015) and subsequent work match intermediate layer representations between student and teacher. This approach requires architectural alignment and is less suitable for our setting where student and teacher have different layer counts.

#### 2.1.3 Relation-Based Distillation

Methods like RKD (Park et al., 2019) transfer structural relationships between data samples. While promising for contrastive learning, this is less applicable to autoregressive generation where per-token predictions are the primary objective.

### 2.2 Language Model Distillation

DistilBERT (Sanh et al., 2019) demonstrated successful BERT compression using soft labels and optional layer initialization from teacher weights. TinyBERT (Jiao et al., 2020) extended this with attention transfer and two-stage distillation. For autoregressive models, GPT-based distillation typically focuses on output logits due to the causal attention structure.

### 2.3 Protein Language Models

ProtGPT2 (Ferruz et al., 2022) is a GPT-2 architecture model trained on UniRef50 sequences, capable of generating protein sequences with natural amino acid distributions. Other notable pLMs include ESM (Rives et al., 2021) for representation learning and ProGen (Madani et al., 2023) for conditional generation. To our knowledge, systematic distillation of protein language models remains underexplored.

---

## 3. Mathematical Framework

### 3.1 Notation

| Symbol | Description |
|--------|-------------|
| $x = (x_1, \ldots, x_n)$ | Protein sequence of length $n$ |
| $\mathcal{V}$ | Vocabulary of amino acid tokens (including special tokens) |
| $z_t \in \mathbb{R}^{|\mathcal{V}|}$ | Logit vector at position $t$ |
| $T$ | Temperature hyperparameter |
| $\alpha$ | Loss balancing coefficient |
| $p^{(T)}(\cdot)$ | Temperature-scaled probability distribution |

### 3.2 Autoregressive Language Modeling

An autoregressive language model factorizes the sequence probability as:

$$P(x) = \prod_{t=1}^{n} P(x_t | x_{<t})$$

Each conditional probability is computed via softmax over logits:

$$P(x_t = v | x_{<t}) = \frac{\exp(z_{t,v})}{\sum_{v' \in \mathcal{V}} \exp(z_{t,v'})}$$

For causal language models, the logits at position $t$ predict the token at position $t+1$. This requires careful alignment during loss computation.

### 3.3 Temperature-Scaled Softmax

Temperature scaling modifies the softmax function to control the "sharpness" of probability distributions:

$$p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_{j=1}^{|\mathcal{V}|} \exp(z_j / T)}$$

**Properties:**
- $T = 1$: Standard softmax (original distribution)
- $T > 1$: Softer distribution, higher entropy, reveals more inter-class relationships
- $T \to \infty$: Uniform distribution
- $T \to 0^+$: One-hot distribution (argmax)

**Implementation** (`src/distillation.py:94-95`):
```python
F.log_softmax(shift_student_logits / self.temperature, dim=-1)
F.softmax(shift_teacher_logits / self.temperature, dim=-1)
```

### 3.4 Soft Loss: Kullback-Leibler Divergence

The soft loss measures the divergence between temperature-scaled student and teacher distributions:

$$L_{\text{soft}} = D_{KL}\left(p_T^{(\tau)} \| p_S^{(\tau)}\right) = \sum_{i=1}^{|\mathcal{V}|} p_T^{(\tau)}(z_i) \log \frac{p_T^{(\tau)}(z_i)}{p_S^{(\tau)}(z_i)}$$

Equivalently, using the definition of KL divergence:

$$L_{\text{soft}} = \sum_{i=1}^{|\mathcal{V}|} p_T^{(\tau)}(z_i) \left[ \log p_T^{(\tau)}(z_i) - \log p_S^{(\tau)}(z_i) \right]$$

Since the first term (teacher entropy) is constant with respect to student parameters, optimizing this is equivalent to minimizing the cross-entropy between teacher and student distributions.

**Implementation** (`src/distillation.py:92-96`):
```python
loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
soft_loss = loss_fct(
    F.log_softmax(shift_student_logits / self.temperature, dim=-1),
    F.softmax(shift_teacher_logits / self.temperature, dim=-1),
)
```

**Note:** PyTorch's `KLDivLoss` expects log-probabilities for the first argument and probabilities for the second, matching the mathematical definition.

### 3.5 Hard Loss: Cross-Entropy

The hard loss is the standard cross-entropy between student predictions and ground-truth labels:

$$L_{\text{hard}} = -\sum_{t=1}^{n-1} \log P_S(x_{t+1} | x_{\leq t}) = -\sum_{t=1}^{n-1} \log \text{softmax}(z_t^S)_{x_{t+1}}$$

This reduces to:

$$L_{\text{hard}} = \text{CrossEntropy}(\hat{y}, y) = -\sum_{i} y_i \log \hat{y}_i$$

where $y$ is the one-hot ground-truth and $\hat{y}$ is the predicted distribution.

**Implementation** (`src/distillation.py:99-102`):
```python
hard_loss = torch.nn.CrossEntropyLoss()(
    shift_student_logits.view(-1, shift_student_logits.size(-1)),
    shift_labels.view(-1),
)
```

### 3.6 Combined Distillation Loss

The total loss is a weighted combination of hard and soft losses:

$$\boxed{L = \alpha \cdot L_{\text{hard}} + (1 - \alpha) \cdot T^2 \cdot L_{\text{soft}}}$$

where:
- $\alpha \in [0, 1]$ balances the contribution of each term
- $T^2$ scales the soft loss to maintain proper gradient magnitudes

**Implementation** (`src/distillation.py:105-108`):
```python
loss = (
    self.alpha * hard_loss
    + (1.0 - self.alpha) * (self.temperature**2) * soft_loss
)
```

### 3.7 Gradient Analysis: The $T^2$ Scaling Factor

The $T^2$ factor is essential for maintaining proper gradient magnitudes. We derive this analytically.

#### 3.7.1 Gradient of Temperature-Scaled Softmax

For softmax with temperature:
$$p_i^{(T)} = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

The gradient with respect to logit $z_k$ is:

$$\frac{\partial p_i^{(T)}}{\partial z_k} = \frac{1}{T} \cdot p_i^{(T)} (\delta_{ik} - p_k^{(T)})$$

where $\delta_{ik}$ is the Kronecker delta. Note the factor of $\frac{1}{T}$ compared to standard softmax.

#### 3.7.2 Gradient of KL Divergence

The gradient of $L_{\text{soft}} = D_{KL}(p_T \| p_S)$ with respect to student logits $z^S$ involves:

$$\frac{\partial L_{\text{soft}}}{\partial z_k^S} = \frac{1}{T} \left( p_S^{(T)}(z_k) - p_T^{(T)}(z_k) \right)$$

Combined with the softmax gradient, the total scaling is $\frac{1}{T^2}$.

#### 3.7.3 Gradient Magnitude Compensation

Without compensation, the gradient magnitude of soft loss would be:

$$\left\| \nabla_{z^S} L_{\text{soft}} \right\| \propto \frac{1}{T^2}$$

For $T = 2$, gradients would be $4\times$ smaller than at $T = 1$. By multiplying $L_{\text{soft}}$ by $T^2$, we restore:

$$\left\| \nabla_{z^S} (T^2 \cdot L_{\text{soft}}) \right\| \propto 1$$

This ensures consistent learning dynamics regardless of temperature choice and allows meaningful comparison of loss values across different temperature settings.

### 3.8 Causal Language Model Alignment

In autoregressive models, the logits at position $t$ predict the token at position $t+1$:

$$\text{logits}[t] \to \text{predicts} \to \text{token}[t+1]$$

To properly compute the loss, we must align predictions with targets:

**Shifting Operation:**
- Logits: Remove last position (predicts beyond sequence)
- Labels: Remove first position (no prediction target)

Mathematically, for sequence $x = (x_0, x_1, \ldots, x_{n-1})$:
- Shifted logits: $(z_0, z_1, \ldots, z_{n-2})$
- Shifted labels: $(x_1, x_2, \ldots, x_{n-1})$

Now $z_t$ correctly predicts $x_{t+1}$.

**Implementation** (`src/distillation.py:86-88`):
```python
shift_student_logits = student_logits[..., :-1, :].contiguous()
shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
```

---

## 4. Model Architecture

### 4.1 Teacher Model: ProtGPT2

ProtGPT2 (Ferruz et al., 2022) is based on the GPT-2 architecture:

| Component | Specification |
|-----------|---------------|
| Architecture | Decoder-only Transformer |
| Parameters | ~738 million |
| Layers | 36 |
| Hidden dimension | 1280 |
| Attention heads | 20 |
| Context length | 1024 |
| Vocabulary | BPE tokenizer on amino acids |
| Pre-training | UniRef50 (45M sequences) |

The model was trained using standard causal language modeling on protein sequences, learning to predict the next amino acid given preceding context.

### 4.2 Student Model Configuration

Student models use the same GPT-2 architecture family with reduced dimensions:

| Parameter | Symbol | Typical Values | Purpose |
|-----------|--------|----------------|---------|
| Layers | $L$ | 2, 4, 6, 12 | Depth of representation |
| Heads | $H$ | 2, 4, 8, 12 | Multi-head attention capacity |
| Embedding | $d$ | 128, 256, 512, 768 | Hidden dimension |

**Constraints:** Number of heads must divide embedding dimension ($d \mod H = 0$).

**Inherited from teacher:**
- Vocabulary size and tokenizer
- Maximum position embeddings
- Special tokens (BOS, EOS, PAD)

**Implementation** (`scripts/train.py:41-53`):
```python
def create_student_config(teacher_model, n_embd, n_layer, n_head):
    """Create a smaller student model configuration based on teacher."""
    return GPT2Config(
        vocab_size=teacher_model.config.vocab_size,
        n_positions=teacher_model.config.n_positions,
        n_ctx=teacher_model.config.n_ctx,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        activation_function="gelu_new",
        bos_token_id=teacher_model.config.bos_token_id,
        eos_token_id=teacher_model.config.eos_token_id,
    )
```

### 4.3 Compression Analysis

For a transformer with $L$ layers, $d$ hidden dimension, $H$ heads, and vocabulary $V$:

$$\text{Parameters} \approx V \cdot d + L \cdot (4d^2 + 4d^2) + d = V \cdot d + 8Ld^2 + d$$

The dominant term is $8Ld^2$, making compression roughly proportional to:

$$\text{Compression Ratio} \approx \frac{L_T \cdot d_T^2}{L_S \cdot d_S^2}$$

| Student Config | Parameters | Compression vs Teacher |
|----------------|------------|------------------------|
| L=4, H=4, d=256 | ~10M | ~74× |
| L=6, H=8, d=512 | ~40M | ~18× |
| L=12, H=12, d=768 | ~125M | ~6× |

---

## 5. Evaluation Metrics

### 5.1 Perplexity

Perplexity measures how well a model predicts a held-out test set:

$$\text{PPL} = \exp\left(\frac{1}{N} \sum_{i=1}^{N} L_{\text{CE}}(x_i)\right) = \exp(\bar{L}_{\text{CE}})$$

where $N$ is the total number of tokens and $L_{\text{CE}}(x_i)$ is the cross-entropy loss for sequence $i$.

**Interpretation:**
- Lower perplexity indicates better prediction
- Perplexity of $k$ means the model is as uncertain as choosing uniformly among $k$ options
- Natural protein sequences typically have perplexity 10-20 for well-trained pLMs

**Implementation** (`scripts/evaluate.py:66-67`):
```python
avg_loss = total_loss / total_tokens
perplexity = np.exp(avg_loss)
```

### 5.2 Perplexity Ratio

To assess distillation quality, we compute the ratio of student to teacher perplexity:

$$\rho_{\text{PPL}} = \frac{\text{PPL}_S}{\text{PPL}_T}$$

**Assessment thresholds:**
| Ratio | Quality |
|-------|---------|
| $< 1.5$ | Excellent |
| $1.5 - 2.0$ | Good |
| $2.0 - 3.0$ | Acceptable |
| $> 3.0$ | Poor |

### 5.3 Output KL Divergence

We measure how closely student output distributions match teacher distributions on held-out sequences:

$$D_{KL}^{\text{eval}} = \frac{1}{|\mathcal{X}|} \sum_{x \in \mathcal{X}} \frac{1}{|x|} \sum_{t=1}^{|x|} D_{KL}\left(p_T(\cdot|x_{<t}) \| p_S(\cdot|x_{<t})\right)$$

Unlike training (which uses temperature scaling), evaluation uses raw logits ($T=1$) to measure true distributional similarity.

**Implementation** (`scripts/evaluate.py:163-166`):
```python
student_probs = F.log_softmax(student_logits, dim=-1)
teacher_probs = F.softmax(teacher_logits, dim=-1)
kl = F.kl_div(student_probs, teacher_probs, reduction="sum")
```

### 5.4 Amino Acid Distribution Analysis

Generated sequences should exhibit amino acid frequencies similar to natural proteins. We compute:

$$D_{KL}^{\text{AA}} = \sum_{a \in \mathcal{A}} p_{\text{gen}}(a) \log \frac{p_{\text{gen}}(a)}{p_{\text{natural}}(a)}$$

where $\mathcal{A}$ is the set of 20 standard amino acids, $p_{\text{gen}}$ is the empirical distribution from generated sequences, and $p_{\text{natural}}$ is the UniProt reference distribution.

**Natural amino acid frequencies (UniProt):**

| AA | Freq | AA | Freq | AA | Freq | AA | Freq |
|----|------|----|------|----|------|----|------|
| A | 8.25% | F | 3.86% | L | 9.66% | S | 6.56% |
| C | 1.37% | G | 7.07% | M | 2.42% | T | 5.34% |
| D | 5.45% | H | 2.27% | N | 4.06% | V | 6.87% |
| E | 6.75% | I | 5.96% | P | 4.70% | W | 1.08% |
| K | 5.84% | Q | 3.93% | R | 5.53% | Y | 2.92% |

**Implementation** (`scripts/evaluate.py:121-126`):
```python
kl_div = 0
for aa in AMINO_ACIDS:
    p = aa_dist.get(aa, 1e-10)
    q = natural_dist.get(aa, 1e-10)
    if p > 0:
        kl_div += p * np.log(p / q)
```

### 5.5 Structural Plausibility (Optional)

For generated sequences, structural plausibility can be assessed using ESMFold's predicted local distance difference test (pLDDT):

$$\text{pLDDT} \in [0, 100]$$

| Score Range | Interpretation |
|-------------|----------------|
| > 90 | Very high confidence, reliable structure |
| 70-90 | Confident prediction |
| 50-70 | Low confidence |
| < 50 | Likely disordered or incorrect |

---

## 6. Training Protocol

### 6.1 Data Preparation

Training data consists of UniProt protein sequences stored in Parquet format. Each sequence is tokenized using the teacher's BPE tokenizer, with `input_ids` serving as both inputs and labels for causal language modeling.

### 6.2 Hyperparameters

| Hyperparameter | Default | Range | Notes |
|----------------|---------|-------|-------|
| Temperature ($T$) | 2.0 | 1.0-10.0 | Higher = softer distributions |
| Alpha ($\alpha$) | 0.5 | 0.0-1.0 | 0 = soft only, 1 = hard only |
| Learning rate | 1e-3 | 1e-5 to 1e-2 | Adam optimizer |
| Batch size | 1 | - | Per-device |
| Gradient accumulation | 32 | - | Effective batch = 32 |
| Epochs | 3 | 1-10 | Early stopping optional |
| Weight decay | 0.01 | - | L2 regularization |
| Precision | FP16 | - | Mixed precision |

### 6.3 Training Procedure

1. **Load teacher model** and freeze parameters
2. **Initialize student model** with random weights
3. **For each batch:**
   - Forward pass through both models
   - Compute shifted logits and labels
   - Calculate soft loss (KL divergence)
   - Calculate hard loss (cross-entropy)
   - Combine with $\alpha$ and $T^2$ scaling
   - Backpropagate through student only
4. **Save model** and training logs

---

## 7. Summary: Formula-to-Code Mapping

| Mathematical Formula | Description | File | Lines |
|---------------------|-------------|------|-------|
| $p_i^{(T)} = \text{softmax}(z_i/T)$ | Temperature-scaled softmax | `src/distillation.py` | 94-95 |
| $L_{\text{soft}} = D_{KL}(p_T^{(\tau)} \| p_S^{(\tau)})$ | Soft loss (KL divergence) | `src/distillation.py` | 92-96 |
| $L_{\text{hard}} = \text{CE}(y, \hat{y})$ | Hard loss (cross-entropy) | `src/distillation.py` | 99-102 |
| $L = \alpha L_{\text{hard}} + (1-\alpha) T^2 L_{\text{soft}}$ | Combined loss | `src/distillation.py` | 105-108 |
| Logit/label shifting for causal LM | Alignment for next-token prediction | `src/distillation.py` | 86-88 |
| $\text{PPL} = \exp(\bar{L}_{\text{CE}})$ | Perplexity | `scripts/evaluate.py` | 66-67 |
| $D_{KL}(p_T \| p_S)$ at $T=1$ | Evaluation KL divergence | `scripts/evaluate.py` | 163-166 |
| $D_{KL}^{\text{AA}}(p_{\text{gen}} \| p_{\text{natural}})$ | Amino acid distribution KL | `scripts/evaluate.py` | 121-126 |

---

## 8. Implementation Reference

### 8.1 Core Files

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `src/distillation.py` | Distillation training | `DistillationTrainer.compute_loss()` |
| `scripts/train.py` | Training pipeline | `create_student_config()`, main loop |
| `scripts/evaluate.py` | Model evaluation | `compute_perplexity()`, `compute_output_kl_divergence()` |
| `config.py` | Hyperparameter defaults | Configuration constants |

### 8.2 Critical Code Sections

**Distillation Loss (`src/distillation.py:44-110`):**
The `compute_loss()` method implements the complete distillation objective, including device handling, forward passes, label shifting, and loss combination.

**Student Configuration (`scripts/train.py:41-53`):**
The `create_student_config()` function ensures proper inheritance of vocabulary and tokenizer settings from teacher.

**Perplexity Computation (`scripts/evaluate.py:45-68`):**
Implements token-weighted average cross-entropy with proper handling of variable-length sequences.

---

## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *arXiv preprint arXiv:1503.02531*.

2. Ferruz, N., Schmidt, S., & Höcker, B. (2022). ProtGPT2 is a deep unsupervised language model for protein design. *Nature Communications*, 13(1), 4348.

3. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*.

4. Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2015). FitNets: Hints for Thin Deep Nets. *ICLR*.

5. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

6. Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., ... & Liu, Q. (2020). TinyBERT: Distilling BERT for Natural Language Understanding. *EMNLP*.

7. Park, W., Kim, D., Lu, Y., & Cho, M. (2019). Relational Knowledge Distillation. *CVPR*.

8. Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., ... & Fergus, R. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*, 118(15).

9. Madani, A., Krause, B., Greene, E. R., Subramanian, S., Mohr, B. P., Holton, J. M., ... & Naik, N. (2023). Large language models generate functional protein sequences across diverse families. *Nature Biotechnology*, 41(8), 1099-1106.

---

*Document generated for distilling_protgpt2 project. For implementation details, see the source code repository.*
