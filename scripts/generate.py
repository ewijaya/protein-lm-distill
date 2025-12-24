#!/usr/bin/env python
"""
Generate protein sequences using a trained distilled ProtGPT2 model.

Usage:
    python scripts/generate.py --model littleworth/protgpt2-distilled-tiny --num_sequences 10
    python scripts/generate.py --model ./models/your-model --max_length 200
"""

import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline


def clean_sequence(text: str) -> str:
    """Clean generated text to extract only amino acid sequence."""
    # Remove special tokens
    text = text.replace("<|endoftext|>", "")
    # Keep only alphabetic characters (amino acids)
    return "".join(char for char in text if char.isalpha()).upper()


def generate_sequences(
    model_name: str,
    num_sequences: int = 10,
    max_length: int = 100,
    top_k: int = 950,
    repetition_penalty: float = 1.2,
    device: str = None,
) -> list[str]:
    """
    Generate protein sequences using a trained model.

    Args:
        model_name: Path to model or HuggingFace model name.
        num_sequences: Number of sequences to generate.
        max_length: Maximum sequence length.
        top_k: Top-k sampling parameter.
        repetition_penalty: Penalty for repeated tokens.
        device: Device to use (cuda/cpu). Auto-detected if None.

    Returns:
        List of generated protein sequences.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # Create pipeline
    generator = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device_id,
    )

    # Generate
    outputs = generator(
        "<|endoftext|>",
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_sequences,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=0,
        truncation=True,
    )

    # Clean and return sequences
    sequences = [clean_sequence(out["generated_text"]) for out in outputs]
    return [seq for seq in sequences if seq]  # Filter empty sequences


def main():
    parser = argparse.ArgumentParser(description="Generate protein sequences")
    parser.add_argument(
        "--model",
        type=str,
        default="littleworth/protgpt2-distilled-tiny",
        help="Model path or HuggingFace name",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=10,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=950,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (FASTA format). If not specified, prints to stdout.",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    sequences = generate_sequences(
        model_name=args.model,
        num_sequences=args.num_sequences,
        max_length=args.max_length,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    # Output in FASTA format
    output_lines = []
    for i, seq in enumerate(sequences):
        output_lines.append(f">Seq_{i}")
        output_lines.append(seq)

    output_text = "\n".join(output_lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text + "\n")
        print(f"Saved {len(sequences)} sequences to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
