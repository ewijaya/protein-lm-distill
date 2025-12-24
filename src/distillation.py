"""
DistillationTrainer class for knowledge distillation of protein language models.

This module provides a custom Trainer that implements knowledge distillation
using a combination of soft (KL divergence) and hard (cross-entropy) losses.
"""

import json
import torch
from torch.nn import functional as F
from transformers import Trainer


class DistillationTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation.

    Combines soft loss (KL divergence on temperature-scaled logits) with
    hard loss (cross-entropy on ground truth labels) as per Hinton et al. (2015).

    Args:
        temperature: Temperature for softening probability distributions.
        alpha: Weight for hard loss (1-alpha for soft loss).
        teacher_model: The teacher model to distill from.
        *args, **kwargs: Additional arguments passed to Trainer.
    """

    def __init__(self, temperature, alpha, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model = teacher_model
        self.training_logs = []

    def log(self, logs: dict):
        super().log(logs)
        self.training_logs.append(logs)

    def save_logs(self, save_path):
        """Save collected training logs to a JSON file."""
        with open(save_path, "w") as f:
            json.dump(self.training_logs, f, indent=4)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the combined knowledge distillation loss.

        This method calculates the distillation loss by combining:
        1. Soft Loss: KL divergence between temperature-softened student and teacher logits
        2. Hard Loss: Cross-entropy on ground truth labels

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

        # Soft loss: KL divergence on temperature-scaled distributions
        # Scale by T^2 as per Hinton et al. (2015) to maintain gradient magnitude
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        soft_loss = loss_fct(
            F.log_softmax(shift_student_logits / self.temperature, dim=-1),
            F.softmax(shift_teacher_logits / self.temperature, dim=-1),
        )

        # Hard loss: cross-entropy on ground truth
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
