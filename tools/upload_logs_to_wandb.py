#!/usr/bin/env python3
"""
Upload training logs to W&B retroactively.

Usage:
    python tools/upload_logs_to_wandb.py --model_dir ./models/protgpt2-distilled-...
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import wandb


def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def extract_run_name(model_dir: Path) -> str:
    """Extract a clean run name from the model directory name."""
    return model_dir.name


def build_config(hyperparams: dict) -> dict:
    """Build W&B config from hyperparameters."""
    training_args = hyperparams.get("training_arguments", {})
    model_arch = hyperparams.get("model_architecture", {})

    return {
        # Distillation params
        "temperature": hyperparams.get("distillation_temperature"),
        "alpha": hyperparams.get("distillation_alpha"),
        # Model architecture
        "n_embd": model_arch.get("n_embd"),
        "n_layer": model_arch.get("n_layer"),
        "n_head": model_arch.get("n_head"),
        # Training params
        "learning_rate": training_args.get("learning_rate"),
        "num_train_epochs": training_args.get("num_train_epochs"),
        "per_device_train_batch_size": training_args.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": training_args.get("gradient_accumulation_steps"),
        "weight_decay": training_args.get("weight_decay"),
        "warmup_steps": training_args.get("warmup_steps"),
        "fp16": training_args.get("fp16"),
        "seed": training_args.get("seed"),
        "lr_scheduler_type": training_args.get("lr_scheduler_type"),
        "logging_steps": training_args.get("logging_steps"),
    }


def upload_to_wandb(
    model_dir: Path,
    project: str = "PROTGPT2_DISTILLATION",
    dry_run: bool = False,
):
    """Upload training logs to W&B."""
    logs_path = model_dir / "training_logs.json"
    hyperparams_path = model_dir / "training_hyperparameters.json"

    if not logs_path.exists():
        raise FileNotFoundError(f"Training logs not found: {logs_path}")
    if not hyperparams_path.exists():
        raise FileNotFoundError(f"Hyperparameters not found: {hyperparams_path}")

    # Load data
    logs = load_json(logs_path)
    hyperparams = load_json(hyperparams_path)

    # Build config and run name
    config = build_config(hyperparams)
    run_name = extract_run_name(model_dir)

    print(f"Run name: {run_name}")
    print(f"Project: {project}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"Total log entries: {len(logs)}")

    if dry_run:
        print("\n[DRY RUN] Would upload the following logs:")
        for i, entry in enumerate(logs[:5]):
            print(f"  Step {i}: {entry}")
        if len(logs) > 5:
            print(f"  ... and {len(logs) - 5} more entries")
        return

    # Initialize W&B run
    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        notes=f"Retroactively uploaded from {model_dir}",
        tags=["retroactive", "distillation"],
    )

    # Extract actual training runtime from final log entry
    actual_runtime = None
    for entry in logs:
        if "train_runtime" in entry:
            actual_runtime = entry.get("train_runtime")
            break

    # Log each training step
    step = 0
    for entry in logs:
        # Skip the final summary entry (has train_runtime instead of loss)
        if "train_runtime" in entry:
            # Log final summary metrics
            wandb.log({
                "train/runtime": entry.get("train_runtime"),
                "train/samples_per_second": entry.get("train_samples_per_second"),
                "train/steps_per_second": entry.get("train_steps_per_second"),
                "train/final_loss": entry.get("train_loss"),
                "train/final_epoch": entry.get("epoch"),
            }, step=step)
        else:
            # Log training step metrics
            wandb.log({
                "train/loss": entry.get("loss"),
                "train/grad_norm": entry.get("grad_norm"),
                "train/learning_rate": entry.get("learning_rate"),
                "train/epoch": entry.get("epoch"),
            }, step=step)
            step += 1

    # Set actual training runtime in summary (overrides W&B's auto-calculated runtime)
    if actual_runtime:
        wandb.run.summary["_runtime"] = actual_runtime
        wandb.run.summary["duration_seconds"] = actual_runtime
        wandb.run.summary["duration_hours"] = actual_runtime / 3600
        print(f"\nActual training runtime: {actual_runtime:.1f}s ({actual_runtime/3600:.2f} hours)")

    # Mark run as finished
    wandb.finish()

    print(f"\nSuccessfully uploaded {step} training steps to W&B!")
    print(f"View run at: {run.url}")


def main():
    parser = argparse.ArgumentParser(description="Upload training logs to W&B retroactively")
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to the model directory containing training_logs.json and training_hyperparameters.json",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="PROTGPT2_DISTILLATION",
        help="W&B project name (default: PROTGPT2_DISTILLATION)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )

    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    upload_to_wandb(
        model_dir=args.model_dir,
        project=args.project,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
