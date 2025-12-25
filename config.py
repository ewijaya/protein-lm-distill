"""
Centralized configuration for ProtGPT2 distillation project.

Uses environment variables where available (HF_HOME, HF_DATASETS_CACHE from ~/.zshrc).
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data paths
# Uses HF_DATASETS_CACHE if set, otherwise falls back to default location
DATA_DIR = Path(
    os.environ.get(
        "HF_DATASETS_CACHE",
        "/home/ubuntu/storage2/various_hugging_face_data_and_models",
    )
) / "data"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
TEACHER_MODEL = "nferruz/ProtGPT2"

# Default training hyperparameters
DEFAULT_TEMPERATURE = 2.0
DEFAULT_ALPHA = 0.5
DEFAULT_N_LAYER = 4
DEFAULT_N_HEAD = 4
DEFAULT_N_EMBD = 256
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_TRAIN_SIZE_PROP = 0.1
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_NUM_EPOCHS = 3

# W&B configuration
WANDB_PROJECT = "PROTGPT2_DISTILLATION"
