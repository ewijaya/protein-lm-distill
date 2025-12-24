#!/usr/bin/env python
"""
Legacy training script using text-based data files.

This script uses local text files instead of pre-tokenized parquet files.
For most use cases, prefer scripts/train.py which uses the larger UniProt dataset.

Usage:
    python scripts/train_legacy.py --temperature 2.0 --alpha 0.5
"""

import gc
import os
import sys
import multiprocessing
import shutil
import argparse
import logging
import json

import torch
import wandb
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
)
from datasets import load_dataset

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distillation import DistillationTrainer
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Legacy ProtGPT2 Distillation")
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
        help="Weight for hard loss",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/all_natural_train_data.txt",
        help="Path to training data file",
    )
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load teacher model
    print(f"Loading teacher model: {config.TEACHER_MODEL}")
    teacher_model = GPT2LMHeadModel.from_pretrained(config.TEACHER_MODEL).to(device)
    teacher_tokenizer = GPT2Tokenizer.from_pretrained(config.TEACHER_MODEL)

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "left"

    # Create student model (fixed small architecture for legacy script)
    student_config = GPT2Config(
        vocab_size=teacher_model.config.vocab_size,
        n_positions=teacher_model.config.n_positions,
        n_ctx=teacher_model.config.n_ctx,
        n_embd=256,
        n_layer=4,
        n_head=4,
        activation_function="gelu_new",
        bos_token_id=teacher_model.config.bos_token_id,
        eos_token_id=teacher_model.config.eos_token_id,
    )
    student_model = GPT2LMHeadModel(student_config).to(device)

    # Load and tokenize text dataset
    data_path = config.PROJECT_ROOT / args.data_file
    dataset = load_dataset("text", data_files=str(data_path))

    def tokenize_function(examples):
        tokenized = teacher_tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
    )

    # Model name
    model_name = (
        f"protgpt2-distilled-t{args.temperature}-a{args.alpha}"
        f"-l{student_config.n_layer}-h{student_config.n_head}-e{student_config.n_embd}"
    )

    # Initialize W&B
    wandb.init(project=config.WANDB_PROJECT, name=model_name)

    # Setup output directory
    output_dir = config.MODELS_DIR / model_name
    if output_dir.exists():
        print(f"Output directory {output_dir} exists. Deleting...")
        shutil.rmtree(output_dir)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=0.0001,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        logging_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        fp16=True,
    )

    # Initialize trainer
    trainer = DistillationTrainer(
        temperature=args.temperature,
        alpha=args.alpha,
        teacher_model=teacher_model,
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
    hyperparams = {
        "training_arguments": training_args.to_dict(),
        "distillation_temperature": trainer.temperature,
        "distillation_alpha": trainer.alpha,
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
