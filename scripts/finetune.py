#!/usr/bin/env python
"""
Fine-tune a ProtGPT2 teacher or distilled student model on a small dataset.

Example:
    python scripts/finetune.py \
        --model littleworth/protgpt2-distilled-tiny \
        --data_dir data/finetune/amp \
        --train_file train_500.fasta \
        --val_file val.fasta \
        --output_dir models/finetune/amp-tiny-500 \
        --epochs 20 \
        --batch_size 8 \
        --learning_rate 5e-5 \
        --early_stopping_patience 3 \
        --wandb_project PROTGPT2_FINETUNE
"""

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

# Set HF cache paths BEFORE importing datasets/transformers
os.environ.setdefault("HF_HOME", "/home/ubuntu/storage3/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/ubuntu/storage3/datasets_cache")

import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def read_fasta(path: Path) -> list[str]:
    sequences = []
    current = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            sequences.append("".join(current))
    return sequences


def resolve_path(path_str: str, data_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = data_dir / path
    return path


def load_tokenizer(model_name: str):
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    except Exception:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_dataset(seqs: list[str], tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_dict({"sequence": seqs})

    def tokenize_fn(batch):
        tokens = tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return tokens

    return dataset.map(tokenize_fn, batched=True, remove_columns=["sequence"])


class LogCollector(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = dict(logs)
            entry["step"] = state.global_step
            entry["epoch"] = state.epoch
            self.logs.append(entry)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ProtGPT2 models")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF name")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with FASTA files")
    parser.add_argument("--train_file", type=str, default="train_full.fasta")
    parser.add_argument("--val_file", type=str, default="val.fasta")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="PROTGPT2_FINETUNE")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = config.PROJECT_ROOT / data_dir

    train_path = resolve_path(args.train_file, data_dir)
    val_path = resolve_path(args.val_file, data_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = config.PROJECT_ROOT / output_dir
    else:
        model_slug = args.model.replace("/", "-")
        output_dir = config.PROJECT_ROOT / "models" / "finetune" / f"{model_slug}-{train_path.stem}"

    if output_dir.exists() and not args.resume_from_checkpoint:
        if args.overwrite_output_dir:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output dir exists: {output_dir} (use --overwrite_output_dir or --resume_from_checkpoint)"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.fp16 is None:
        fp16 = device == "cuda"
    else:
        fp16 = args.fp16

    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")
    tokenizer = load_tokenizer(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    train_sequences = read_fasta(train_path)
    val_sequences = read_fasta(val_path)
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences:   {len(val_sequences)}")

    train_dataset = build_dataset(train_sequences, tokenizer, args.max_length)
    val_dataset = build_dataset(val_sequences, tokenizer, args.max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    eval_batch_size = args.eval_batch_size or args.batch_size

    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=args.wandb_run_name,
        seed=args.seed,
    )

    log_collector = LogCollector()
    callbacks = []
    if args.early_stopping_patience and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    callbacks.append(log_collector)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    print("Starting fine-tuning...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving model and tokenizer...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log_path = output_dir / "training_logs.json"
    with open(log_path, "w") as f:
        json.dump(log_collector.logs, f, indent=2)

    hyperparams = {
        "model": args.model,
        "train_file": str(train_path),
        "val_file": str(val_path),
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "training_arguments": training_args.to_dict(),
        "max_length": args.max_length,
    }
    with open(output_dir / "training_hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=2)

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Fine-tuning complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
