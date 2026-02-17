#!/usr/bin/env python
"""
Benchmark throughput and GPU memory usage for protein sequence generation.

Measures wall-clock time, sequences/minute, and peak GPU memory for each model.

Usage:
    python scripts/benchmark_throughput.py --num_sequences 100 --output results/throughput_benchmark.json
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline


def benchmark_model(model_path, num_sequences=100, max_length=200, warmup=5):
    """Benchmark a single model's generation throughput."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1

    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    num_params = sum(p.numel() for p in model.parameters())

    generator = TextGenerationPipeline(
        model=model, tokenizer=tokenizer, device=device_id,
    )

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup runs
    for _ in range(warmup):
        generator(
            "<|endoftext|>", max_length=50, do_sample=True, top_k=950,
            repetition_penalty=1.2, num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=0, truncation=True,
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Timed generation
    times = []
    seq_lengths = []
    for i in range(num_sequences):
        t0 = time.time()
        outputs = generator(
            "<|endoftext|>", max_length=max_length, do_sample=True, top_k=950,
            repetition_penalty=1.2, num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=0, truncation=True,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        seq = "".join(c for c in outputs[0]["generated_text"] if c.isalpha())
        seq_lengths.append(len(seq))

    # Collect memory stats
    peak_memory_mb = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Cleanup
    del model, generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import numpy as np
    total_time = sum(times)
    return {
        "model_path": model_path,
        "num_params": num_params,
        "num_sequences": num_sequences,
        "total_time_s": total_time,
        "avg_time_per_seq_s": float(np.mean(times)),
        "std_time_per_seq_s": float(np.std(times)),
        "sequences_per_min": num_sequences / total_time * 60,
        "avg_seq_length": float(np.mean(seq_lengths)),
        "peak_gpu_memory_mb": peak_memory_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark generation throughput")
    parser.add_argument("--num_sequences", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/throughput_benchmark.json")
    args = parser.parse_args()

    models = {
        "teacher": "nferruz/ProtGPT2",
        "synergy-tiny": "./models/synergy-tiny-v2",
        "synergy-small": "./models/synergy-small",
        "synergy-medium": "./models/synergy-medium-v2",
    }

    results = {}
    for name, path in models.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name} ({path})")
        print(f"{'='*60}")
        results[name] = benchmark_model(path, args.num_sequences, args.max_length)
        print(f"  Avg time/seq: {results[name]['avg_time_per_seq_s']:.3f}s")
        print(f"  Sequences/min: {results[name]['sequences_per_min']:.1f}")
        if results[name]['peak_gpu_memory_mb']:
            print(f"  Peak GPU memory: {results[name]['peak_gpu_memory_mb']:.0f} MB")

    # Compute speedups relative to teacher
    if "teacher" in results:
        teacher_time = results["teacher"]["avg_time_per_seq_s"]
        for name in results:
            results[name]["speedup_vs_teacher"] = teacher_time / results[name]["avg_time_per_seq_s"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
