#!/usr/bin/env python
"""
Evaluate a fine-tuned ProtGPT2 model on a domain-specific family.

Metrics:
- Test perplexity
- Generated sequence quality (AA dist, length dist)
- Novelty vs. training set
- Diversity within generated set
- Optional HMMER hit rate
- Optional ESMFold pLDDT

Example:
    python scripts/evaluate_finetune.py \
        --model models/finetune/amp-tiny-500 \
        --train_file data/finetune/amp/train_500.fasta \
        --test_file data/finetune/amp/test.fasta \
        --family amp \
        --num_generate 200 \
        --output results/finetune/amp-tiny-500.json
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Set HF cache paths BEFORE importing transformers
os.environ.setdefault("HF_HOME", "/home/ubuntu/storage3/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/ubuntu/storage3/datasets_cache")

from typing import Optional

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


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


def clean_sequence(text: str) -> str:
    return "".join(c for c in text.upper() if c.isalpha())


def load_model_and_tokenizer(model_path, device):
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    except Exception:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, sequences, device, max_length=512):
    total_loss = 0.0
    total_tokens = 0
    for seq in sequences:
        inputs = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        seq_len = inputs["input_ids"].size(1)
        total_loss += outputs.loss.item() * seq_len
        total_tokens += seq_len
    avg_loss = total_loss / max(1, total_tokens)
    return float(np.exp(avg_loss))


def generate_sequences(
    model,
    tokenizer,
    num_sequences: int,
    device: str,
    max_length: int,
    top_k: int,
    repetition_penalty: float,
    temperature: float,
    seed: int,
) -> list[str]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sequences = []
    input_ids = torch.tensor([[tokenizer.eos_token_id]]).to(device)
    attempts = 0
    max_attempts = max(10, num_sequences * 3)
    while len(sequences) < num_sequences and attempts < max_attempts:
        attempts += 1
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        cleaned = clean_sequence(generated)
        if cleaned:
            sequences.append(cleaned)
    return sequences


def aa_distribution(sequences: list[str]) -> dict[str, float]:
    counts = {aa: 0 for aa in AMINO_ACIDS}
    other = 0
    for seq in sequences:
        for c in seq:
            if c in counts:
                counts[c] += 1
            else:
                other += 1
    total = sum(counts.values())
    if total == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}
    return {aa: counts[aa] / total for aa in AMINO_ACIDS}


def kl_divergence(p: dict, q: dict, epsilon: float = 1e-8) -> float:
    keys = set(p.keys()) | set(q.keys())
    kl = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, 0.0)
        if pk > 0:
            kl += pk * math.log((pk + epsilon) / (qk + epsilon))
    return float(kl)


def length_distribution(sequences: list[str]) -> dict[int, float]:
    counts = {}
    for seq in sequences:
        l = len(seq)
        counts[l] = counts.get(l, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def levenshtein_distance(a: str, b: str, max_dist: Optional[int] = None) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        min_row = i
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            val = min(insert, delete, replace)
            current.append(val)
            if val < min_row:
                min_row = val
        if max_dist is not None and min_row > max_dist:
            return max_dist + 1
        previous = current
    return previous[-1]


def normalized_edit_distance(a: str, b: str, max_norm: Optional[float] = None) -> float:
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    max_dist = None
    if max_norm is not None:
        max_dist = int(math.floor(max_norm * denom))
    dist = levenshtein_distance(a, b, max_dist=max_dist)
    return dist / denom


def novelty_metrics(
    generated: list[str],
    train: list[str],
    threshold: float,
    train_cap: int,
    seed: int,
) -> dict:
    if not train or not generated:
        return {"novelty_fraction": None, "mean_min_distance": None}

    rng = random.Random(seed)
    if len(train) > train_cap:
        train = rng.sample(train, train_cap)

    min_dists = []
    for seq in generated:
        min_dist = 1.0
        for t in train:
            dist = normalized_edit_distance(seq, t, max_norm=threshold)
            if dist < min_dist:
                min_dist = dist
                if min_dist == 0.0:
                    break
        min_dists.append(min_dist)

    novelty_fraction = sum(d > threshold for d in min_dists) / len(min_dists)
    mean_min_dist = float(np.mean(min_dists)) if min_dists else None
    return {
        "novelty_fraction": novelty_fraction,
        "mean_min_distance": mean_min_dist,
    }


def diversity_metrics(sequences: list[str], max_pairs: int, seed: int) -> dict:
    n = len(sequences)
    if n < 2:
        return {"mean_pairwise_distance": None, "pairs": 0}

    total_pairs = n * (n - 1) // 2
    rng = random.Random(seed)
    pairs = []
    if total_pairs <= max_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
    else:
        for _ in range(max_pairs):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pairs.append((i, j))

    dists = []
    for i, j in pairs:
        dists.append(normalized_edit_distance(sequences[i], sequences[j], max_norm=None))
    return {
        "mean_pairwise_distance": float(np.mean(dists)) if dists else None,
        "pairs": len(dists),
    }


def write_fasta(sequences: list[str], path: Path):
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n")
            f.write(seq + "\n")


def run_hmmsearch(hmm_profile: Path, fasta_path: Path, hmmer_path: str, evalue: float):
    if shutil.which(hmmer_path) is None and not Path(hmmer_path).exists():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tblout = Path(tmpdir) / "hits.tbl"
        cmd = [
            hmmer_path,
            "--noali",
            "--tblout",
            str(tblout),
            "-E",
            str(evalue),
            str(hmm_profile),
            str(fasta_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            return None

        hits = {}
        with open(tblout, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                target_name = parts[0]
                hit_evalue = float(parts[4])
                score = float(parts[5])
                if hit_evalue <= evalue:
                    prev = hits.get(target_name)
                    if prev is None or score > prev["score"]:
                        hits[target_name] = {"evalue": hit_evalue, "score": score}

        return hits


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--num_generate", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--eval_max_length", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=950)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--novelty_threshold", type=float, default=0.3)
    parser.add_argument("--novelty_train_cap", type=int, default=5000)
    parser.add_argument("--diversity_max_pairs", type=int, default=20000)
    parser.add_argument("--hmm_profile", type=str, default=None)
    parser.add_argument("--hmmer_path", type=str, default="hmmsearch")
    parser.add_argument("--hmmer_evalue", type=float, default=1e-5)
    parser.add_argument("--compute_plddt", action="store_true")
    parser.add_argument("--plddt_top_k", type=int, default=50)
    parser.add_argument("--save_sequences", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    train_path = Path(args.train_file)
    if not train_path.is_absolute():
        train_path = config.PROJECT_ROOT / train_path
    test_path = Path(args.test_file)
    if not test_path.is_absolute():
        test_path = config.PROJECT_ROOT / test_path

    train_sequences = [s for s in (clean_sequence(s) for s in read_fasta(train_path)) if s]
    test_sequences = [s for s in (clean_sequence(s) for s in read_fasta(test_path)) if s]

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    print(f"Computing test perplexity on {len(test_sequences)} sequences...")
    test_ppl = compute_perplexity(
        model, tokenizer, test_sequences, args.device, max_length=args.eval_max_length
    )

    print(f"Generating {args.num_generate} sequences...")
    generated = generate_sequences(
        model,
        tokenizer,
        num_sequences=args.num_generate,
        device=args.device,
        max_length=args.max_length,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        seed=args.seed,
    )

    if args.save_sequences:
        save_path = Path(args.save_sequences)
        if not save_path.is_absolute():
            save_path = config.PROJECT_ROOT / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        write_fasta(generated, save_path)

    gen_lengths = [len(s) for s in generated]
    train_lengths = [len(s) for s in train_sequences]

    gen_aa_dist = aa_distribution(generated)
    train_aa_dist = aa_distribution(train_sequences)

    length_kl = kl_divergence(
        length_distribution(generated),
        length_distribution(train_sequences),
    )
    aa_kl = kl_divergence(gen_aa_dist, train_aa_dist)

    novelty = novelty_metrics(
        generated,
        train_sequences,
        threshold=args.novelty_threshold,
        train_cap=args.novelty_train_cap,
        seed=args.seed,
    )
    diversity = diversity_metrics(generated, args.diversity_max_pairs, args.seed)

    hmm_hits = None
    hit_rate = None
    if args.hmm_profile:
        hmm_profile = Path(args.hmm_profile)
        if not hmm_profile.is_absolute():
            hmm_profile = config.PROJECT_ROOT / hmm_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "generated.fasta"
            write_fasta(generated, fasta_path)
            hmm_hits = run_hmmsearch(
                hmm_profile,
                fasta_path,
                hmmer_path=args.hmmer_path,
                evalue=args.hmmer_evalue,
            )
        if hmm_hits is not None:
            hit_rate = len(hmm_hits) / max(1, len(generated))

    plddt_stats = None
    if args.compute_plddt:
        try:
            from src.esmfold import predict_plddt_batch
        except Exception as exc:
            print(f"ESMFold unavailable: {exc}")
        else:
            if hmm_hits:
                # Rank by HMMER score
                ranked = sorted(hmm_hits.items(), key=lambda x: x[1]["score"], reverse=True)
                indices = [int(k.split("_")[-1]) for k, _ in ranked]
                selected = [generated[i] for i in indices[: args.plddt_top_k] if i < len(generated)]
            else:
                selected = generated[: args.plddt_top_k]
            if selected:
                scores = predict_plddt_batch(selected)
                plddt_stats = {
                    "n": len(scores),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                }

    results = {
        "model": args.model,
        "family": args.family,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "num_generated": len(generated),
        "test_perplexity": test_ppl,
        "generation": {
            "avg_length": float(np.mean(gen_lengths)) if gen_lengths else None,
            "std_length": float(np.std(gen_lengths)) if gen_lengths else None,
            "length_kl": length_kl,
            "aa_kl": aa_kl,
            "novelty": novelty,
            "diversity": diversity,
        },
        "hmmer": {
            "hmm_profile": args.hmm_profile,
            "evalue_threshold": args.hmmer_evalue,
            "hit_rate": hit_rate,
            "hits": len(hmm_hits) if hmm_hits is not None else None,
        },
        "plddt": plddt_stats,
    }

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = config.PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
