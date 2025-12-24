"""
Source modules for ProtGPT2 distillation.
"""

from .distillation import DistillationTrainer
from .esmfold import predict_plddt, predict_plddt_batch

__all__ = ["DistillationTrainer", "predict_plddt", "predict_plddt_batch"]
