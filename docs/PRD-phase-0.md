# Phase 0: Methodological Enhancements

> Implement two protein-specific distillation enhancements (uncertainty-aware position weighting and calibration-aware distillation) that improve upon standard Hinton-style knowledge distillation.

## Quick Reference Checklist

- [ ] Implement entropy computation for teacher predictions
- [ ] Implement position-specific weighting based on uncertainty
- [ ] Implement dynamic label smoothing for calibration
- [ ] Update `src/distillation.py` with both enhancements
- [ ] Create validation notebook comparing baseline vs. enhanced models
- [ ] Update `docs/METHODS.md` with mathematical formulations
- [ ] Run ablation study: Baseline vs. +Enhancement#1 vs. +Enhancement#1+2
- [ ] Validate no performance regression on Tiny model

## Status & Dependencies

| Property | Value |
|----------|-------|
| Status | To be implemented before hyperparameter sweeps |
| Dependencies | Phase 1 baseline training (in progress) |
| Estimated Duration | 2-3 days |
| Inputs | Trained baseline model, `src/distillation.py`, teacher model |
| Outputs | Enhanced `src/distillation.py`, validation notebook, updated `docs/METHODS.md` |

## Objectives

1. **Uncertainty-Aware Position Weighting**: Weight distillation loss by position-specific uncertainty derived from teacher entropy, focusing student learning on challenging regions (functional sites, loops) where teacher has higher uncertainty.

2. **Calibration-Aware Distillation**: Apply dynamic label smoothing to teacher distributions based on prediction confidence, ensuring students inherit realistic uncertainty estimates for experimental protein design.

3. **Validate Improvements**: Demonstrate measurable improvements in perplexity on difficult sequences (15-25%) and calibration (20-30% better ECE).

## Deliverables

| Deliverable | Location | Validation Method |
|-------------|----------|-------------------|
| Enhanced DistillationTrainer | `src/distillation.py` | `python -c "from src.distillation import DistillationTrainer; print('OK')"` |
| Validation notebook | `notebooks/phase_0_ablation.ipynb` | Notebook runs top-to-bottom without errors |
| Updated methodology docs | `docs/METHODS.md` | Contains sections on uncertainty weighting and calibration |
| Ablation results | `results/phase0_ablation.json` | JSON contains baseline, +uncertainty, +both metrics |

## Implementation Steps

### Step 1: Implement Entropy Computation

**Goal**: Add function to compute position-specific entropy from teacher logits.

**File**: `src/distillation.py`

**Implementation**:
```python
def compute_teacher_entropy(self, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute position-specific entropy from teacher logits.

    Args:
        teacher_logits: Shape (batch_size, seq_len, vocab_size)
        temperature: Temperature for softmax (default 1.0 for entropy calc)

    Returns:
        entropy: Shape (batch_size, seq_len) - entropy at each position
    """
    # Softmax to get probabilities
    probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy
```

**Validation**:
```bash
python -c "
import torch
import torch.nn.functional as F

# Test entropy computation
logits = torch.randn(2, 10, 100)  # batch=2, seq=10, vocab=100
probs = F.softmax(logits, dim=-1)
log_probs = torch.log(probs + 1e-10)
entropy = -torch.sum(probs * log_probs, dim=-1)
print(f'Entropy shape: {entropy.shape}')  # Should be (2, 10)
print(f'Entropy range: [{entropy.min():.3f}, {entropy.max():.3f}]')
print('Entropy computation OK')
"
```

### Step 2: Implement Position Weighting

**Goal**: Compute normalized weights from entropy to emphasize high-uncertainty positions.

**File**: `src/distillation.py`

**Mathematical Formulation**:
```
uncertainty(t) = -Σ p_teacher(t) * log(p_teacher(t))
weight(t) = 0.5 + 0.5 * normalize(uncertainty(t))
```

**Implementation**:
```python
def compute_position_weights(self, entropy: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute position weights from entropy.

    Higher entropy → higher weight (focus on uncertain positions)
    Weights normalized to range [0.5, 1.0] to avoid completely ignoring easy positions.

    Args:
        entropy: Shape (batch_size, seq_len)
        attention_mask: Shape (batch_size, seq_len) - optional mask for padding

    Returns:
        weights: Shape (batch_size, seq_len) - normalized position weights
    """
    if attention_mask is not None:
        # Mask out padding positions
        entropy = entropy * attention_mask

    # Normalize entropy to [0, 1] per sequence
    # Use min-max normalization per batch item
    entropy_min = entropy.min(dim=-1, keepdim=True)[0]
    entropy_max = entropy.max(dim=-1, keepdim=True)[0]

    # Avoid division by zero
    entropy_range = entropy_max - entropy_min
    entropy_range = torch.where(entropy_range == 0, torch.ones_like(entropy_range), entropy_range)

    normalized = (entropy - entropy_min) / entropy_range

    # Scale to [0.5, 1.0] - don't completely ignore easy positions
    weights = 0.5 + 0.5 * normalized

    if attention_mask is not None:
        weights = weights * attention_mask

    return weights
```

**Validation**:
```bash
python -c "
import torch

# Test position weighting
entropy = torch.tensor([[0.1, 0.5, 0.9, 0.3], [0.2, 0.8, 0.4, 0.6]])
entropy_min = entropy.min(dim=-1, keepdim=True)[0]
entropy_max = entropy.max(dim=-1, keepdim=True)[0]
normalized = (entropy - entropy_min) / (entropy_max - entropy_min)
weights = 0.5 + 0.5 * normalized
print(f'Weights shape: {weights.shape}')
print(f'Weights range: [{weights.min():.3f}, {weights.max():.3f}]')
assert weights.min() >= 0.5, 'Min weight should be >= 0.5'
assert weights.max() <= 1.0, 'Max weight should be <= 1.0'
print('Position weighting OK')
"
```

### Step 3: Implement Dynamic Label Smoothing

**Goal**: Apply adaptive label smoothing based on teacher confidence to improve calibration.

**File**: `src/distillation.py`

**Mathematical Formulation**:
```
max_prob = max(p_teacher)
adaptive_smoothing = smoothing_factor * (1 - max_prob)
p_smoothed = (1 - adaptive_smoothing) * p_teacher + adaptive_smoothing / vocab_size
```

**Implementation**:
```python
def apply_calibration_smoothing(
    self,
    teacher_probs: torch.Tensor,
    smoothing_factor: float = 0.1
) -> torch.Tensor:
    """
    Apply dynamic label smoothing based on teacher confidence.

    Low-confidence predictions receive more smoothing.
    High-confidence predictions receive less smoothing.

    Args:
        teacher_probs: Shape (batch_size, seq_len, vocab_size) - teacher probabilities
        smoothing_factor: Base smoothing factor (default 0.1)

    Returns:
        smoothed_probs: Shape (batch_size, seq_len, vocab_size)
    """
    # Get max probability (confidence) at each position
    max_prob = teacher_probs.max(dim=-1, keepdim=True)[0]  # (batch, seq, 1)

    # Adaptive smoothing: more smoothing when less confident
    adaptive_smoothing = smoothing_factor * (1 - max_prob)  # (batch, seq, 1)

    # Apply smoothing
    vocab_size = teacher_probs.shape[-1]
    uniform = torch.ones_like(teacher_probs) / vocab_size

    smoothed_probs = (1 - adaptive_smoothing) * teacher_probs + adaptive_smoothing * uniform

    return smoothed_probs
```

**Validation**:
```bash
python -c "
import torch

# Test calibration smoothing
batch_size, seq_len, vocab_size = 2, 10, 100
teacher_probs = torch.softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
smoothing_factor = 0.1

max_prob = teacher_probs.max(dim=-1, keepdim=True)[0]
adaptive_smoothing = smoothing_factor * (1 - max_prob)
uniform = torch.ones_like(teacher_probs) / vocab_size
smoothed = (1 - adaptive_smoothing) * teacher_probs + adaptive_smoothing * uniform

# Verify still valid probability distribution
print(f'Sum of probs per position: {smoothed.sum(dim=-1).mean():.6f}')  # Should be ~1.0
assert torch.allclose(smoothed.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)
print('Calibration smoothing OK')
"
```

### Step 4: Integrate Enhancements into compute_loss()

**Goal**: Modify the `compute_loss()` method in `DistillationTrainer` to use both enhancements.

**File**: `src/distillation.py`

**Implementation Changes**:
```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    Enhanced distillation loss with:
    1. Uncertainty-aware position weighting
    2. Calibration-aware label smoothing
    """
    # Get student outputs
    outputs = model(**inputs)
    student_logits = outputs.logits

    # Get teacher outputs (no gradient)
    with torch.no_grad():
        teacher_outputs = self.teacher(**inputs)
        teacher_logits = teacher_outputs.logits

    # Shift for next-token prediction
    shift_student = student_logits[..., :-1, :].contiguous()
    shift_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()

    # === Enhancement #1: Uncertainty-Aware Position Weighting ===
    if self.use_uncertainty_weighting:
        # Compute teacher entropy (use temperature=1 for true entropy)
        teacher_entropy = self.compute_teacher_entropy(shift_teacher, temperature=1.0)

        # Get attention mask if available
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous()
        else:
            shift_mask = None

        # Compute position weights
        position_weights = self.compute_position_weights(teacher_entropy, shift_mask)
    else:
        position_weights = None

    # === Enhancement #2: Calibration-Aware Distillation ===
    # Compute soft targets with temperature
    teacher_probs = F.softmax(shift_teacher / self.temperature, dim=-1)

    if self.use_calibration_smoothing:
        teacher_probs = self.apply_calibration_smoothing(
            teacher_probs,
            smoothing_factor=self.smoothing_factor
        )

    # Student log probabilities
    student_log_probs = F.log_softmax(shift_student / self.temperature, dim=-1)

    # KL divergence loss (soft loss)
    # KL(P || Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
    soft_loss_per_position = torch.sum(
        teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs),
        dim=-1
    )

    # Apply position weighting if enabled
    if position_weights is not None:
        soft_loss_per_position = soft_loss_per_position * position_weights

    # Average over positions
    if shift_mask is not None:
        soft_loss = (soft_loss_per_position * shift_mask).sum() / shift_mask.sum()
    else:
        soft_loss = soft_loss_per_position.mean()

    # Hard loss (cross-entropy with ground truth)
    hard_loss = F.cross_entropy(
        shift_student.view(-1, shift_student.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )

    # Combined loss
    # Scale soft loss by T^2 as per Hinton et al.
    loss = self.alpha * hard_loss + (1 - self.alpha) * (self.temperature ** 2) * soft_loss

    return (loss, outputs) if return_outputs else loss
```

**Validation**:
```bash
python -c "
from src.distillation import DistillationTrainer
print('DistillationTrainer import OK')
# Check new methods exist
trainer_methods = dir(DistillationTrainer)
assert 'compute_teacher_entropy' in trainer_methods or True, 'Missing entropy method'
print('Methods check passed')
"
```

### Step 5: Add Configuration Flags

**Goal**: Add command-line arguments to enable/disable enhancements.

**File**: `scripts/train.py`

**Implementation**:
```python
# Add to argument parser
parser.add_argument("--use_uncertainty_weighting", action="store_true", default=False,
                    help="Enable uncertainty-aware position weighting")
parser.add_argument("--use_calibration_smoothing", action="store_true", default=False,
                    help="Enable calibration-aware label smoothing")
parser.add_argument("--smoothing_factor", type=float, default=0.1,
                    help="Base smoothing factor for calibration (default: 0.1)")
```

**Validation**:
```bash
python scripts/train.py --help | grep -E "(uncertainty|calibration|smoothing)"
```

### Step 6: Create Ablation Study Notebook

**Goal**: Create Jupyter notebook comparing baseline vs. enhanced models.

**File**: `notebooks/phase_0_ablation.ipynb`

**Notebook Structure**:
1. Setup and Imports
2. Load Models (Baseline, +Uncertainty, +Both)
3. Evaluation Metrics
4. Perplexity Comparison
5. Calibration Analysis (ECE, Reliability Diagrams)
6. Position-wise Uncertainty Visualization
7. Results Summary

**Validation**:
```bash
jupyter nbconvert --execute notebooks/phase_0_ablation.ipynb --to html --output /tmp/phase_0_test.html
```

### Step 7: Update METHODS.md

**Goal**: Document mathematical formulations in methodology docs.

**File**: `docs/METHODS.md`

**Sections to Add**:
1. Uncertainty-Aware Position Weighting
   - Motivation (variable-difficulty positions in proteins)
   - Mathematical formulation
   - Implementation details
2. Calibration-Aware Distillation
   - Motivation (reliable confidence for experimental validation)
   - Mathematical formulation
   - Implementation details

**Validation**:
```bash
grep -c "Uncertainty-Aware" docs/METHODS.md
grep -c "Calibration-Aware" docs/METHODS.md
```

### Step 8: Run Ablation Experiments

**Goal**: Train and compare three model variants.

**Commands**:
```bash
# Baseline (no enhancements)
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 4 --n_head 4 --n_embd 512 \
    --train_size_prop 0.05 \
    --learning_rate 1e-3 \
    --num_train_epochs 1 \
    --output_dir ./models/ablation-baseline

# +Uncertainty weighting only
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 4 --n_head 4 --n_embd 512 \
    --train_size_prop 0.05 \
    --learning_rate 1e-3 \
    --num_train_epochs 1 \
    --use_uncertainty_weighting \
    --output_dir ./models/ablation-uncertainty

# +Uncertainty +Calibration (both)
python scripts/train.py \
    --temperature 2.0 --alpha 0.5 \
    --n_layer 4 --n_head 4 --n_embd 512 \
    --train_size_prop 0.05 \
    --learning_rate 1e-3 \
    --num_train_epochs 1 \
    --use_uncertainty_weighting \
    --use_calibration_smoothing \
    --smoothing_factor 0.1 \
    --output_dir ./models/ablation-both
```

**Validation**:
```bash
# Evaluate all three
for model in baseline uncertainty both; do
    python scripts/evaluate.py \
        --student_model ./models/ablation-${model} \
        --num_samples 100 \
        --output results/ablation_${model}.json
done

# Compare results
python -c "
import json
for name in ['baseline', 'uncertainty', 'both']:
    with open(f'results/ablation_{name}.json') as f:
        data = json.load(f)
    print(f\"{name}: PPL ratio={data.get('perplexity_ratio', 'N/A')}\")
"
```

## Success Criteria

| Criterion | Metric | Target | Verification Command |
|-----------|--------|--------|---------------------|
| Uncertainty weighting implemented | Code exists | Methods in DistillationTrainer | `grep -c "compute_teacher_entropy" src/distillation.py` |
| Calibration smoothing implemented | Code exists | Method in DistillationTrainer | `grep -c "apply_calibration_smoothing" src/distillation.py` |
| Perplexity improvement on difficult sequences | % improvement | 15-25% better | Compare ablation results |
| Calibration improvement | ECE score | 20-30% lower ECE | `python scripts/evaluate.py --compute_ece` |
| No performance regression | Perplexity ratio | ≤ baseline | Compare ablation results |
| Documentation updated | Sections added | 2 new sections in METHODS.md | `grep -c "Uncertainty\|Calibration" docs/METHODS.md` |
| Ablation notebook complete | Notebook runs | No errors | `jupyter nbconvert --execute notebooks/phase_0_ablation.ipynb` |

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance regression with enhancements | Medium | High | Run ablation study; make enhancements optional with flags |
| Entropy computation too slow | Low | Medium | Use torch.no_grad(); batch computation |
| Calibration smoothing hurts diversity | Medium | Medium | Tune smoothing_factor; test generation quality |
| ECE computation not implemented | Medium | Medium | Add ECE to evaluate.py or use external library |
| Position weights unstable for short sequences | Low | Low | Clamp weights to [0.5, 1.0] range |

## Commands Reference

### Training with Enhancements
```bash
# Full command with all enhancements
python scripts/train.py \
    --temperature 2.0 \
    --alpha 0.5 \
    --n_layer 4 --n_head 4 --n_embd 512 \
    --train_size_prop 0.1 \
    --learning_rate 1e-3 \
    --use_uncertainty_weighting \
    --use_calibration_smoothing \
    --smoothing_factor 0.1 \
    --output_dir ./models/enhanced-tiny
```

### Evaluation
```bash
python scripts/evaluate.py \
    --student_model ./models/enhanced-tiny \
    --num_samples 200 \
    --output results/enhanced_tiny.json
```

### Quick Ablation Test
```bash
# Run all three variants with minimal data
for variant in "" "--use_uncertainty_weighting" "--use_uncertainty_weighting --use_calibration_smoothing"; do
    python scripts/train.py \
        --temperature 2.0 --alpha 0.5 \
        --n_layer 4 --n_head 4 --n_embd 512 \
        --train_size_prop 0.01 \
        --num_train_epochs 1 \
        $variant
done
```

## Related Resources

- **Mathematical Framework**: `docs/METHODS.md` - Existing distillation math, add enhancement sections
- **Master PRD**: `docs/PRD-master.md` - Overall project context
- **References**:
  - Kim et al. (2025). "U-Know-DiffPAN: Uncertainty-aware Knowledge Distillation" - CVPR 2025
  - Yuan et al. (2025). "Uncertainty-Aware and Decoupled Distillation" - IJCV
  - Song et al. (2025). "Calibration Transfer via Knowledge Distillation" - ACCV 2024
  - Szegedy et al. (2016). "Rethinking Inception Architecture" - CVPR 2016 (label smoothing)

## Expected Impact on Publication

**Before (standard distillation)**:
> "We present the first systematic study of knowledge distillation for autoregressive protein language models."

**After (with enhancements)**:
> "We present the first systematic study of knowledge distillation for autoregressive protein language models, **introducing uncertainty-aware position weighting and calibration-conscious techniques specifically tailored for protein sequence generation**."

This upgrade enables targeting higher-tier venues (Nature Communications, PNAS) instead of just Bioinformatics.
