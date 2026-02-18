#!/usr/bin/env python
"""
Prepare fine-tuning datasets for Phase 6 experiments.

This script:
1) Loads raw sequences (FASTA/TSV/CSV/TXT)
2) Cleans + filters sequences
3) De-duplicates (100% identity)
4) Splits into train/val/test
5) Creates size-stratified training subsets

Example:
    python scripts/prepare_finetune_data.py \
        --family amp \
        --input data/raw/amp.fasta \
        --min_len 10 --max_len 100 \
        --output_dir data/finetune/amp
"""

import argparse
import csv
import json
import random
from pathlib import Path
import textwrap
from typing import Optional, Set, Tuple, List, Dict

# Add project root to path for imports
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


DEFAULT_ALLOWED = "ACDEFGHIKLMNPQRSTVWYBXZJUO"


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


def read_txt(path: Path) -> list[str]:
    sequences = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sequences.append(line)
    return sequences


def read_table(path: Path, seq_col: str, delimiter: str) -> list[str]:
    sequences = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if seq_col not in reader.fieldnames:
            raise ValueError(
                f"Column '{seq_col}' not found. Available columns: {reader.fieldnames}"
            )
        for row in reader:
            seq = row.get(seq_col, "")
            if seq:
                sequences.append(seq)
    return sequences


def infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".fa", ".fasta", ".faa", ".fas"]:
        return "fasta"
    if suffix in [".csv"]:
        return "csv"
    if suffix in [".tsv", ".tab"]:
        return "tsv"
    return "txt"


def clean_sequence(seq: str) -> str:
    return "".join(c for c in seq.upper() if c.isalpha())


def filter_sequences(
    sequences: list[str],
    min_len: Optional[int],
    max_len: Optional[int],
    allowed_chars: Optional[Set[str]],
    dedupe: bool,
) -> Tuple[List[str], Dict]:
    stats = {
        "input_total": len(sequences),
        "filtered_empty": 0,
        "filtered_length": 0,
        "filtered_chars": 0,
        "deduped": 0,
    }

    cleaned = []
    for s in sequences:
        seq = clean_sequence(s)
        if not seq:
            stats["filtered_empty"] += 1
            continue
        if min_len is not None and len(seq) < min_len:
            stats["filtered_length"] += 1
            continue
        if max_len is not None and len(seq) > max_len:
            stats["filtered_length"] += 1
            continue
        if allowed_chars is not None and any(c not in allowed_chars for c in seq):
            stats["filtered_chars"] += 1
            continue
        cleaned.append(seq)

    if dedupe:
        seen = set()
        deduped = []
        for s in cleaned:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        stats["deduped"] = len(cleaned) - len(deduped)
        cleaned = deduped

    stats["output_total"] = len(cleaned)
    return cleaned, stats


def write_fasta(sequences: list[str], path: Path, prefix: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">{prefix}_{i}\n")
            f.write(textwrap.fill(seq, width=80))
            f.write("\n")


def parse_subset_sizes(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    return [int(v) for v in value.split(",") if v]


def split_train_val_test(
    sequences: list[str],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    sequences = list(sequences)
    rng.shuffle(sequences)

    n_total = len(sequences)
    if n_total < 3:
        return sequences, [], []

    val_n = max(1, int(round(val_fraction * n_total)))
    test_n = max(1, int(round(test_fraction * n_total)))
    if val_n + test_n >= n_total:
        test_n = max(1, min(test_n, n_total - 2))
        val_n = max(1, min(val_n, n_total - 1 - test_n))
    train_n = n_total - val_n - test_n

    train = sequences[:train_n]
    val = sequences[train_n:train_n + val_n]
    test = sequences[train_n + val_n:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning datasets")
    parser.add_argument("--family", type=str, required=True, help="Family name (amp, conotoxin, lysozyme)")
    parser.add_argument("--input", type=str, required=True, help="Input file path (FASTA/CSV/TSV/TXT)")
    parser.add_argument("--input_format", type=str, default=None, choices=["fasta", "csv", "tsv", "txt"],
                        help="Optional input format override")
    parser.add_argument("--seq_col", type=str, default="sequence",
                        help="Column name for CSV/TSV inputs")
    parser.add_argument("--min_len", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_len", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--strict_aa", action="store_true",
                        help="Only allow 20 canonical amino acids")
    parser.add_argument("--allowed_chars", type=str, default=DEFAULT_ALLOWED,
                        help=f"Allowed characters (default: {DEFAULT_ALLOWED})")
    parser.add_argument("--no_dedupe", action="store_true", help="Disable exact de-duplication")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--test_fraction", type=float, default=0.1, help="Test split fraction")
    parser.add_argument("--subset_sizes", type=str, default="50,100,200,500,1000",
                        help="Comma-separated training subset sizes")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data/finetune/<family>)")
    parser.add_argument("--dry_run", action="store_true", help="Run without writing files")
    args = parser.parse_args()

    if args.family.lower() in {"amp", "amps"}:
        if args.min_len is None:
            args.min_len = 10
        if args.max_len is None:
            args.max_len = 100

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = config.PROJECT_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.input_format:
        input_format = args.input_format
    else:
        input_format = infer_format(input_path)

    if input_format == "fasta":
        sequences = read_fasta(input_path)
    elif input_format == "txt":
        sequences = read_txt(input_path)
    elif input_format == "csv":
        sequences = read_table(input_path, args.seq_col, delimiter=",")
    elif input_format == "tsv":
        sequences = read_table(input_path, args.seq_col, delimiter="\t")
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    allowed_chars = set("ACDEFGHIKLMNPQRSTVWY") if args.strict_aa else set(args.allowed_chars)

    filtered, stats = filter_sequences(
        sequences,
        min_len=args.min_len,
        max_len=args.max_len,
        allowed_chars=allowed_chars,
        dedupe=not args.no_dedupe,
    )

    train, val, test = split_train_val_test(
        filtered,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    subset_sizes = parse_subset_sizes(args.subset_sizes)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = config.PROJECT_ROOT / output_dir
    else:
        output_dir = config.PROJECT_ROOT / "data" / "finetune" / args.family

    meta = {
        "family": args.family,
        "input_path": str(input_path),
        "input_format": input_format,
        "seed": args.seed,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "strict_aa": args.strict_aa,
        "allowed_chars": "".join(sorted(allowed_chars)),
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "subset_sizes": subset_sizes,
        "counts": {
            "total": len(filtered),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "filter_stats": stats,
    }

    print(json.dumps(meta, indent=2))

    if args.dry_run:
        print("Dry run enabled; no files written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(train, output_dir / "train_full.fasta", prefix=f"{args.family}_train")
    write_fasta(val, output_dir / "val.fasta", prefix=f"{args.family}_val")
    write_fasta(test, output_dir / "test.fasta", prefix=f"{args.family}_test")

    for n in subset_sizes:
        n_eff = min(n, len(train))
        subset = train[:n_eff]
        write_fasta(subset, output_dir / f"train_{n}.fasta", prefix=f"{args.family}_train{n}")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote dataset to {output_dir}")


if __name__ == "__main__":
    main()
