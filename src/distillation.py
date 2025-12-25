"""
DistillationTrainer class for knowledge distillation of protein language models.

This module provides a custom Trainer that implements knowledge distillation
using a combination of soft (KL divergence) and hard (cross-entropy) losses.

Includes two protein-specific enhancements:
1. Uncertainty-aware position weighting: Emphasizes positions where teacher
   has higher entropy (uncertainty), focusing learning on variable regions.
2. Calibration-aware distillation: Applies dynamic label smoothing based on
   teacher confidence to produce well-calibrated student models.

References:
- Hinton et al. (2015). Distilling the Knowledge in a Neural Network.
- Kim et al. (2025). U-Know-DiffPAN: Uncertainty-aware Knowledge Distillation. CVPR.
- Song et al. (2025). Calibration Transfer via Knowledge Distillation. ACCV/Springer.
"""

import json
import torch
from torch.nn import functional as F
from transformers import Trainer


class DistillationTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation with protein-specific enhancements.

    Combines soft loss (KL divergence on temperature-scaled logits) with
    hard loss (cross-entropy on ground truth labels) as per Hinton et al. (2015).

    Protein-specific enhancements:
    - Uncertainty-aware position weighting: Weights distillation loss by
      position-specific entropy from teacher predictions.
    - Calibration-aware distillation: Applies adaptive label smoothing to
      teacher distributions based on prediction confidence.

    Args:
        temperature: Temperature for softening probability distributions.
        alpha: Weight for hard loss (1-alpha for soft loss).
        teacher_model: The teacher model to distill from.
        use_uncertainty_weighting: Enable uncertainty-aware position weighting.
        use_calibration_smoothing: Enable calibration-aware label smoothing.
        smoothing_factor: Base smoothing factor for calibration (default: 0.1).
        *args, **kwargs: Additional arguments passed to Trainer.
    """

    def __init__(
        self,
        temperature,
        alpha,
        teacher_model,
        use_uncertainty_weighting=False,
        use_calibration_smoothing=False,
        smoothing_factor=0.1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model = teacher_model
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_calibration_smoothing = use_calibration_smoothing
        self.smoothing_factor = smoothing_factor
        self.training_logs = []

    def log(self, logs: dict):
        super().log(logs)
        self.training_logs.append(logs)

    def save_logs(self, save_path):
        """Save collected training logs to a JSON file."""
        with open(save_path, "w") as f:
            json.dump(self.training_logs, f, indent=4)

    def compute_teacher_entropy(self, teacher_logits, temperature=1.0):
        """
        Compute position-specific entropy from teacher logits.

        Entropy measures the uncertainty in teacher predictions at each position.
        Higher entropy indicates the teacher is less certain about the prediction,
        which often corresponds to biologically variable regions in proteins.

        Mathematical formulation:
            H(p) = -Σ p(v) * log(p(v))

        Args:
            teacher_logits: Shape (batch_size, seq_len, vocab_size)
            temperature: Temperature for softmax (default 1.0 for true entropy)

        Returns:
            entropy: Shape (batch_size, seq_len) - entropy at each position
        """
        # Compute probabilities with temperature scaling
        probs = F.softmax(teacher_logits / temperature, dim=-1)

        # Compute entropy: H = -Σ p * log(p)
        # Add small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        return entropy

    def compute_position_weights(self, entropy, attention_mask=None):
        """
        Compute position weights from entropy for uncertainty-aware distillation.

        Higher entropy positions receive higher weights, focusing the student
        on learning the teacher's behavior at uncertain/variable positions.

        The weights are normalized to [0.5, 1.0] to ensure all positions
        contribute but emphasizing high-uncertainty regions.

        Mathematical formulation:
            w(t) = 0.5 + 0.5 * normalize(entropy(t))

        Args:
            entropy: Shape (batch_size, seq_len) - entropy at each position
            attention_mask: Shape (batch_size, seq_len) - optional mask for padding

        Returns:
            weights: Shape (batch_size, seq_len) - normalized position weights in [0.5, 1.0]
        """
        if attention_mask is not None:
            # Mask out padding positions for normalization
            masked_entropy = entropy.clone()
            masked_entropy[attention_mask == 0] = float('-inf')
        else:
            masked_entropy = entropy

        # Min-max normalization per batch item
        # Compute min/max only over valid (non-padded) positions
        if attention_mask is not None:
            # Replace -inf with a large positive for min computation
            entropy_for_min = entropy.clone()
            entropy_for_min[attention_mask == 0] = float('inf')
            entropy_min = entropy_for_min.min(dim=-1, keepdim=True)[0]

            entropy_for_max = entropy.clone()
            entropy_for_max[attention_mask == 0] = float('-inf')
            entropy_max = entropy_for_max.max(dim=-1, keepdim=True)[0]
        else:
            entropy_min = entropy.min(dim=-1, keepdim=True)[0]
            entropy_max = entropy.max(dim=-1, keepdim=True)[0]

        # Avoid division by zero when all entropies are equal
        entropy_range = entropy_max - entropy_min
        entropy_range = torch.where(
            entropy_range < 1e-8,
            torch.ones_like(entropy_range),
            entropy_range
        )

        # Normalize to [0, 1]
        normalized = (entropy - entropy_min) / entropy_range

        # Scale to [0.5, 1.0] - don't completely ignore easy positions
        weights = 0.5 + 0.5 * normalized

        # Apply attention mask to zero out padding positions
        if attention_mask is not None:
            weights = weights * attention_mask

        return weights

    def apply_calibration_smoothing(self, teacher_probs, smoothing_factor=None):
        """
        Apply dynamic label smoothing based on teacher confidence.

        Low-confidence predictions receive more smoothing, high-confidence
        predictions receive less. This helps the student inherit realistic
        uncertainty estimates rather than overconfident predictions.

        Mathematical formulation:
            ε(t) = λ * (1 - max(p_teacher))
            p_smoothed = (1 - ε) * p_teacher + ε / |V|

        Args:
            teacher_probs: Shape (batch_size, seq_len, vocab_size) - teacher probabilities
            smoothing_factor: Base smoothing factor λ (default: self.smoothing_factor)

        Returns:
            smoothed_probs: Shape (batch_size, seq_len, vocab_size)
        """
        if smoothing_factor is None:
            smoothing_factor = self.smoothing_factor

        # Get max probability (confidence) at each position
        max_prob = teacher_probs.max(dim=-1, keepdim=True)[0]  # (batch, seq, 1)

        # Adaptive smoothing: more smoothing when less confident
        # When max_prob ≈ 1 (confident): adaptive_smoothing ≈ 0
        # When max_prob is low (uncertain): adaptive_smoothing ≈ smoothing_factor
        adaptive_smoothing = smoothing_factor * (1.0 - max_prob)  # (batch, seq, 1)

        # Apply smoothing: blend teacher probs with uniform distribution
        vocab_size = teacher_probs.size(-1)
        uniform = torch.ones_like(teacher_probs) / vocab_size

        smoothed_probs = (1.0 - adaptive_smoothing) * teacher_probs + adaptive_smoothing * uniform

        return smoothed_probs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the combined knowledge distillation loss with optional enhancements.

        This method calculates the distillation loss by combining:
        1. Soft Loss: KL divergence between temperature-softened student and teacher logits
        2. Hard Loss: Cross-entropy on ground truth labels

        Optional protein-specific enhancements:
        - Uncertainty-aware position weighting: Weight soft loss by teacher entropy
        - Calibration-aware distillation: Apply adaptive label smoothing

        The combined loss is:
            L = alpha * L_hard + (1 - alpha) * T^2 * L_soft

        The T^2 scaling maintains proper gradient magnitude as per Hinton et al. (2015).

        Args:
            model: The student model being trained.
            inputs: Dictionary containing input tensors (input_ids, attention_mask, labels).
            return_outputs: Whether to return model outputs with the loss.

        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs=True.
        """
        # Handle DataParallel wrapped models
        if isinstance(model, torch.nn.DataParallel):
            device = model.module.device
        else:
            device = model.device

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.get("labels")

        # Forward pass through student
        outputs = model(**inputs)
        student_logits = outputs.logits

        # Forward pass through teacher (no gradients)
        self.teacher_model = self.teacher_model.to(device)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Shift logits and labels for causal LM (predict next token)
        # logits[i] predicts token[i+1], so we align them properly
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Get attention mask if available, shifted to match
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:].contiguous()
        else:
            shift_attention_mask = None

        # === Soft Loss Computation with Enhancements ===

        # Compute teacher probabilities with temperature scaling
        teacher_probs = F.softmax(shift_teacher_logits / self.temperature, dim=-1)

        # Enhancement #2: Calibration-aware label smoothing
        if self.use_calibration_smoothing:
            teacher_probs = self.apply_calibration_smoothing(teacher_probs)

        # Student log probabilities
        student_log_probs = F.log_softmax(shift_student_logits / self.temperature, dim=-1)

        # Compute per-position KL divergence
        # KL(P || Q) = Σ P * (log P - log Q)
        kl_per_position = torch.sum(
            teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs),
            dim=-1
        )  # Shape: (batch_size, seq_len)

        # Enhancement #1: Uncertainty-aware position weighting
        if self.use_uncertainty_weighting:
            # Compute entropy using raw teacher logits (temperature=1 for true entropy)
            teacher_entropy = self.compute_teacher_entropy(shift_teacher_logits, temperature=1.0)
            position_weights = self.compute_position_weights(teacher_entropy, shift_attention_mask)
            kl_per_position = kl_per_position * position_weights

        # Compute mean soft loss, handling attention mask
        if shift_attention_mask is not None:
            # Weighted mean over non-padded positions
            soft_loss = (kl_per_position * shift_attention_mask).sum() / shift_attention_mask.sum().clamp(min=1)
        else:
            soft_loss = kl_per_position.mean()

        # === Hard Loss Computation ===
        hard_loss = torch.nn.CrossEntropyLoss()(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Combined loss with T^2 scaling for soft loss
        loss = (
            self.alpha * hard_loss
            + (1.0 - self.alpha) * (self.temperature**2) * soft_loss
        )

        return (loss, outputs) if return_outputs else loss
