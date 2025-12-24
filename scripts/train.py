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

import torch
import wandb
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
)
from datasets import load_dataset, DatasetDict

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
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load teacher model
    print(f"Loading teacher model: {config.TEACHER_MODEL}")
    teacher_model = GPT2LMHeadModel.from_pretrained(config.TEACHER_MODEL).to(device)
    teacher_tokenizer = GPT2TokenizerFast.from_pretrained(config.TEACHER_MODEL)

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "left"

    # Load dataset
    data_files = {"train": glob.glob(f"{config.DATA_DIR}/train*.parquet")}
    dataset = load_dataset("parquet", data_files=data_files, trust_remote_code=True)

    # Subsample dataset
    train_subset = dataset["train"].train_test_split(
        train_size=args.train_size_prop
    )["train"]
    print(f"Training subset size: {len(train_subset)}")
    tokenized_dataset = DatasetDict({"train": train_subset})

    # Create student model
    student_config = create_student_config(
        teacher_model, args.n_embd, args.n_layer, args.n_head
    )
    student_model = GPT2LMHeadModel(student_config).to(device)

    # Model name encodes configuration
    model_name = (
        f"protgpt2-distilled-t{args.temperature}-a{args.alpha}"
        f"-l{args.n_layer}-h{args.n_head}-e{args.n_embd}"
        f"-p{args.train_size_prop}-lr{args.learning_rate:.0e}.uniprot"
    )

    # Initialize W&B
    wandb.init(
        project=config.WANDB_PROJECT,
        name=model_name,
        settings=wandb.Settings(_service_wait=120),
    )

    # Setup output directory
    output_dir = config.MODELS_DIR / model_name
    if output_dir.exists():
        print(f"Output directory {output_dir} exists. Deleting...")
        shutil.rmtree(output_dir)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.DEFAULT_NUM_EPOCHS,
        per_device_train_batch_size=config.DEFAULT_BATCH_SIZE,
        gradient_accumulation_steps=config.DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        logging_strategy="steps",
        logging_steps=10,
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
