#!/usr/bin/env python
"""
Train a distilled ProtGPT2 model using knowledge distillation.

Usage:
    python scripts/train.py --temperature 2.0 --alpha 0.5 --n_layer 4 --n_head 4 --n_embd 256

For long-running training:
    nohup python scripts/train.py --temperature 2.0 --alpha 0.5 > nohup.out &
"""

import gc
import os
import sys
import shutil
import glob
import argparse
import logging
from pathlib import Path

import torch
import wandb
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
)
from datasets import load_dataset, DatasetDict, disable_caching

# Enable progress bars for dataset loading
from datasets.utils.logging import set_verbosity_info
set_verbosity_info()

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distillation import DistillationTrainer
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def main():
    parser = argparse.ArgumentParser(description="ProtGPT2 Distillation Training")
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help="Temperature for distillation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=config.DEFAULT_ALPHA,
        help="Weight for hard loss (1-alpha for soft loss)",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=config.DEFAULT_N_EMBD,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=config.DEFAULT_N_LAYER,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=config.DEFAULT_N_HEAD,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--train_size_prop",
        type=float,
        default=config.DEFAULT_TRAIN_SIZE_PROP,
        help="Proportion of dataset to use for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config.DEFAULT_LEARNING_RATE,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=config.DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=config.DEFAULT_NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for model (default: auto-generated based on config)",
    )
    # Phase 0 enhancements
    parser.add_argument(
        "--use_uncertainty_weighting",
        action="store_true",
        default=False,
        help="Enable uncertainty-aware position weighting (Phase 0 enhancement)",
    )
    parser.add_argument(
        "--use_calibration_smoothing",
        action="store_true",
        default=False,
        help="Enable calibration-aware label smoothing (Phase 0 enhancement)",
    )
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.1,
        help="Base smoothing factor for calibration (default: 0.1)",
    )
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Use fast gp3 storage for HuggingFace cache
    os.environ["HF_HOME"] = str(config.HF_CACHE_DIR)
    os.environ["HF_DATASETS_CACHE"] = str(config.FAST_STORAGE / "datasets_cache")
    print(f"Using HF cache: {config.HF_CACHE_DIR}", flush=True)
    print(f"Using datasets cache: {config.FAST_STORAGE / 'datasets_cache'}", flush=True)

    # Load teacher model
    print(f"Loading teacher model: {config.TEACHER_MODEL}", flush=True)
    teacher_model = GPT2LMHeadModel.from_pretrained(config.TEACHER_MODEL).to(device)
    teacher_tokenizer = GPT2TokenizerFast.from_pretrained(config.TEACHER_MODEL)

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "left"

    # Load dataset from fast gp3 storage (pre-copied random subset with seed=42)
    local_parquet_dir = Path("/home/ubuntu/storage3/data/parquet_subset")
    all_parquet_files = sorted(glob.glob(str(local_parquet_dir / "train*.parquet")))
    print(f"Loading {len(all_parquet_files)} parquet files from {local_parquet_dir}", flush=True)

    data_files = {"train": all_parquet_files}
    dataset = load_dataset("parquet", data_files=data_files, trust_remote_code=True)
    print(f"Training dataset size: {len(dataset['train'])}", flush=True)
    tokenized_dataset = DatasetDict({"train": dataset["train"]})

    # Create student model
    student_config = create_student_config(
        teacher_model, args.n_embd, args.n_layer, args.n_head
    )
    student_model = GPT2LMHeadModel(student_config).to(device)

    # Model name encodes configuration
    enhancement_suffix = ""
    if args.use_uncertainty_weighting:
        enhancement_suffix += "-uw"
    if args.use_calibration_smoothing:
        enhancement_suffix += f"-cs{args.smoothing_factor}"
    model_name = (
        f"protgpt2-distilled-t{args.temperature}-a{args.alpha}"
        f"-l{args.n_layer}-h{args.n_head}-e{args.n_embd}"
        f"-p{args.train_size_prop}-lr{args.learning_rate:.0e}{enhancement_suffix}.uniprot"
    )

    # Initialize W&B
    wandb.init(
        project=config.WANDB_PROJECT,
        name=model_name,
        settings=wandb.Settings(_service_wait=120),
    )

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.MODELS_DIR / model_name
    if output_dir.exists():
        print(f"Output directory {output_dir} exists. Deleting...")
        shutil.rmtree(output_dir)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        save_total_limit=1,
        fp16=True,
    )

    # Log enhancement status
    if args.use_uncertainty_weighting or args.use_calibration_smoothing:
        print("Phase 0 Enhancements enabled:")
        if args.use_uncertainty_weighting:
            print("  - Uncertainty-aware position weighting")
        if args.use_calibration_smoothing:
            print(f"  - Calibration-aware smoothing (factor={args.smoothing_factor})")

    # Initialize trainer
    trainer = DistillationTrainer(
        temperature=args.temperature,
        alpha=args.alpha,
        teacher_model=teacher_model,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
        use_calibration_smoothing=args.use_calibration_smoothing,
        smoothing_factor=args.smoothing_factor,
        model=student_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=teacher_tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model and logs
    print(f"Saving model to {output_dir}")
    trainer.save_logs(str(output_dir / "training_logs.json"))
    student_model.save_pretrained(str(output_dir))
    teacher_tokenizer.save_pretrained(str(output_dir))

    # Save hyperparameters
    import json
    hyperparams = {
        "training_arguments": training_args.to_dict(),
        "distillation_temperature": trainer.temperature,
        "distillation_alpha": trainer.alpha,
        "model_architecture": {
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
        },
        "phase0_enhancements": {
            "use_uncertainty_weighting": args.use_uncertainty_weighting,
            "use_calibration_smoothing": args.use_calibration_smoothing,
            "smoothing_factor": args.smoothing_factor,
        },
    }
    with open(output_dir / "training_hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=4)

    wandb.finish()

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    print("Training complete!")


if __name__ == "__main__":
    main()
