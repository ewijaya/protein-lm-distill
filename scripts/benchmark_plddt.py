#!/usr/bin/env python
"""
Benchmark structural quality of generated sequences using ESMFold pLDDT scores.

Generates sequences from teacher, synergy, and baseline models, then evaluates
structural plausibility via ESMFold pLDDT predictions.

Usage:
    python scripts/benchmark_plddt.py --num_sequences 50 --output results/plddt_benchmark.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import generate_sequences


def run_plddt_benchmark(models, num_sequences=50, max_length=200, output_path=None):
    """Generate sequences from each model and score with ESMFold pLDDT."""
    from src.esmfold import predict_plddt

    results = {}

    for name, model_path in models.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name} ({model_path})")
        print(f"{'='*60}")

        # Generate sequences
        print(f"Generating {num_sequences} sequences...")
        t0 = time.time()
        sequences = generate_sequences(
            model_name=model_path,
            num_sequences=num_sequences,
            max_length=max_length,
        )
        gen_time = time.time() - t0
        print(f"Generated {len(sequences)} sequences in {gen_time:.1f}s")

        # Score with ESMFold
        print(f"Computing pLDDT scores...")
        scores = []
        for i, seq in enumerate(sequences):
            if len(seq) < 10:
                continue
            try:
                score = predict_plddt(seq)
                scores.append(score)
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(sequences)}: pLDDT={score:.1f}")
            except Exception as e:
                print(f"  {i+1}/{len(sequences)}: FAILED ({e})")

        import numpy as np
        results[name] = {
            "model_path": model_path,
            "num_sequences": len(sequences),
            "num_scored": len(scores),
            "plddt_scores": scores,
            "mean_plddt": float(np.mean(scores)) if scores else None,
            "std_plddt": float(np.std(scores)) if scores else None,
            "median_plddt": float(np.median(scores)) if scores else None,
            "min_plddt": float(np.min(scores)) if scores else None,
            "max_plddt": float(np.max(scores)) if scores else None,
            "pct_above_70": float(np.mean([s > 70 for s in scores])) if scores else None,
            "generation_time_s": gen_time,
            "avg_seq_length": float(np.mean([len(s) for s in sequences])),
        }

        print(f"  Mean pLDDT: {results[name]['mean_plddt']:.1f}")
        print(f"  % above 70: {results[name]['pct_above_70']*100:.0f}%")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark pLDDT structural quality")
    parser.add_argument("--num_sequences", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/plddt_benchmark.json")
    args = parser.parse_args()

    models = {
        "teacher": "nferruz/ProtGPT2",
        "synergy-tiny": "./models/synergy-tiny-v2",
        "synergy-small": "./models/synergy-small",
        "synergy-medium": "./models/synergy-medium-v2",
        "baseline-medium": "./models/baseline-medium",
    }

    run_plddt_benchmark(
        models,
        num_sequences=args.num_sequences,
        max_length=args.max_length,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
