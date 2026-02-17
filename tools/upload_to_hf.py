#!/usr/bin/env python
"""
Upload a trained model to Hugging Face Hub.

Usage:
    python tools/upload_to_hf.py --model_dir ./models/your-model --repo_id username/model-name

Requires HF_TOKEN environment variable to be set (configured in ~/.zshrc).
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def upload_model(model_dir: str, repo_id: str, private: bool = False):
    """
    Upload a model directory to Hugging Face Hub.

    Args:
        model_dir: Path to the model directory.
        repo_id: HuggingFace repository ID (e.g., "username/model-name").
        private: Whether to make the repository private.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")

    api = HfApi()

    print(f"Creating/updating repository: {repo_id}")
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    ignore_patterns = ["checkpoint-*", "runs/**", "runs/"]

    print(f"Uploading model from: {model_dir}")
    print(f"Ignoring patterns: {ignore_patterns}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_path),
        revision="main",
        ignore_patterns=ignore_patterns,
    )

    print(f"Upload complete! Model available at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    args = parser.parse_args()

    # Check for HF token
    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN environment variable not set.")
        print("Make sure you're logged in with `huggingface-cli login`")

    upload_model(args.model_dir, args.repo_id, args.private)


if __name__ == "__main__":
    main()
