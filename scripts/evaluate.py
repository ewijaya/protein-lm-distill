#!/home/ubuntu/storage1/anaconda3/envs/pepmlm/bin/python
"""
Evaluate distilled ProtGPT2 models against the teacher model.

This script compares:
1. Perplexity on held-out sequences
2. Generated sequence quality (amino acid distribution, length)
3. KL divergence between student and teacher output distributions

Usage:
    python evaluate_model.py --student_model ./models/your-model --num_samples 100
    python evaluate_model.py --student_model littleworth/protgpt2-distilled-tiny
"""

import argparse
import torch
import numpy as np
from collections import Counter
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
from torch.nn import functional as F
import json
import warnings

warnings.filterwarnings("ignore")

# Standard amino acids
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def compute_ece(model, tokenizer, sequences, device, n_bins=10, max_length=512):
    """
    Compute Expected Calibration Error (ECE) for a model.

    ECE measures how well-calibrated a model's confidence estimates are.
    A well-calibrated model's confidence should match its accuracy.

    Mathematical formulation:
        ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|

    where predictions are binned by confidence into M bins.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        sequences: List of protein sequences to evaluate
        device: Device to run on
        n_bins: Number of confidence bins (default: 10)
        max_length: Maximum sequence length

    Returns:
        dict with ECE, MCE (max calibration error), and per-bin statistics
    """
    model.eval()

    # Collect all predictions
    all_confidences = []
    all_correct = []

    for seq in sequences:
        inputs = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        # Get probabilities and predictions
        probs = F.softmax(shift_logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)

        # Check correctness
        correct = (predictions == shift_labels).float()

        # Flatten and collect
        all_confidences.extend(confidences.view(-1).cpu().numpy())
        all_correct.extend(correct.view(-1).cpu().numpy())

    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)

    # Compute ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = all_confidences[in_bin].mean()
            avg_accuracy = all_correct[in_bin].mean()
            calibration_error = abs(avg_accuracy - avg_confidence)

            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)

            bin_stats.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": int(in_bin.sum()),
                "avg_confidence": float(avg_confidence),
                "avg_accuracy": float(avg_accuracy),
                "calibration_error": float(calibration_error),
            })
        else:
            bin_stats.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": 0,
                "avg_confidence": None,
                "avg_accuracy": None,
                "calibration_error": None,
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "n_samples": len(all_confidences),
        "overall_accuracy": float(all_correct.mean()),
        "overall_confidence": float(all_confidences.mean()),
        "bin_stats": bin_stats,
    }


def load_model_and_tokenizer(model_path, device):
    """Load model and tokenizer, handling both local and HuggingFace paths."""
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    except Exception:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, sequences, device, max_length=512):
    """Compute perplexity on a list of sequences."""
    total_loss = 0
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
            # Loss is already averaged over tokens
            seq_len = inputs["input_ids"].size(1)
            total_loss += outputs.loss.item() * seq_len
            total_tokens += seq_len

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def generate_sequences(model, tokenizer, num_sequences, device, max_length=100):
    """Generate protein sequences from the model."""
    model.eval()
    sequences = []

    for _ in range(num_sequences):
        input_ids = torch.tensor([[tokenizer.eos_token_id]]).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                top_k=950,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        # Clean sequence - keep only amino acids
        cleaned = "".join(c for c in generated if c.isalpha() and c.upper() in AMINO_ACIDS)
        if cleaned:
            sequences.append(cleaned.upper())

    return sequences


def analyze_sequences(sequences):
    """Analyze generated sequences for quality metrics."""
    if not sequences:
        return {"error": "No valid sequences generated"}

    lengths = [len(s) for s in sequences]
    all_aas = "".join(sequences)
    aa_counts = Counter(all_aas)
    total_aas = len(all_aas)

    # Amino acid distribution
    aa_dist = {aa: aa_counts.get(aa, 0) / total_aas for aa in AMINO_ACIDS}

    # Expected natural distribution (approximate from UniProt)
    natural_dist = {
        "A": 0.0825, "R": 0.0553, "N": 0.0406, "D": 0.0545, "C": 0.0137,
        "Q": 0.0393, "E": 0.0675, "G": 0.0707, "H": 0.0227, "I": 0.0596,
        "L": 0.0966, "K": 0.0584, "M": 0.0242, "F": 0.0386, "P": 0.0470,
        "S": 0.0656, "T": 0.0534, "W": 0.0108, "Y": 0.0292, "V": 0.0687,
    }

    # KL divergence from natural distribution
    kl_div = 0
    for aa in AMINO_ACIDS:
        p = aa_dist.get(aa, 1e-10)
        q = natural_dist.get(aa, 1e-10)
        if p > 0:
            kl_div += p * np.log(p / q)

    return {
        "num_sequences": len(sequences),
        "avg_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "aa_distribution": aa_dist,
        "kl_from_natural": kl_div,
    }


def compute_output_kl_divergence(
    student_model, teacher_model, tokenizer, sequences, device, max_length=512
):
    """Compute KL divergence between student and teacher output distributions."""
    total_kl = 0
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
            student_outputs = student_model(**inputs)
            teacher_outputs = teacher_model(**inputs)

            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits

            # Compute KL divergence
            student_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            kl = F.kl_div(student_probs, teacher_probs, reduction="sum")
            total_kl += kl.item()
            total_tokens += inputs["input_ids"].numel()

    return total_kl / total_tokens


def get_test_sequences(num_sequences=50):
    """Return a set of test protein sequences."""
    # Short representative sequences for testing
    test_seqs = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLFRPGQKNGILFSKGSG",
        "MGLSDGEWQQVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHG",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
    ]
    # Repeat to get desired count
    while len(test_seqs) < num_sequences:
        test_seqs = test_seqs + test_seqs
    return test_seqs[:num_sequences]


def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled ProtGPT2 model")
    parser.add_argument(
        "--student_model",
        type=str,
        required=True,
        help="Path to student model (local or HuggingFace)",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="nferruz/ProtGPT2",
        help="Path to teacher model",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of sequences to generate for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--compute_ece",
        action="store_true",
        default=False,
        help="Compute Expected Calibration Error (ECE)",
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    print()

    # Load models
    print("Loading teacher model...")
    teacher_model, teacher_tokenizer = load_model_and_tokenizer(
        args.teacher_model, args.device
    )
    print(f"  Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")

    print("Loading student model...")
    student_model, student_tokenizer = load_model_and_tokenizer(
        args.student_model, args.device
    )
    print(f"  Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")

    compression_ratio = sum(p.numel() for p in teacher_model.parameters()) / sum(
        p.numel() for p in student_model.parameters()
    )
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    print()

    # Get test sequences
    test_sequences = get_test_sequences(args.num_samples)

    # Compute perplexity
    print("Computing perplexity on test sequences...")
    teacher_ppl = compute_perplexity(
        teacher_model, teacher_tokenizer, test_sequences, args.device
    )
    student_ppl = compute_perplexity(
        student_model, student_tokenizer, test_sequences, args.device
    )
    print(f"  Teacher perplexity: {teacher_ppl:.2f}")
    print(f"  Student perplexity: {student_ppl:.2f}")
    print(f"  Ratio (lower is better): {student_ppl / teacher_ppl:.2f}x")
    print()

    # Compute KL divergence
    print("Computing KL divergence between student and teacher...")
    kl_div = compute_output_kl_divergence(
        student_model, teacher_model, student_tokenizer, test_sequences[:20], args.device
    )
    print(f"  Average KL divergence: {kl_div:.4f}")
    print()

    # Generate sequences
    print(f"Generating {args.num_samples} sequences from each model...")
    print("  Generating from teacher...")
    teacher_seqs = generate_sequences(
        teacher_model, teacher_tokenizer, args.num_samples, args.device
    )
    print("  Generating from student...")
    student_seqs = generate_sequences(
        student_model, student_tokenizer, args.num_samples, args.device
    )
    print()

    # Analyze generated sequences
    print("Analyzing generated sequences...")
    teacher_analysis = analyze_sequences(teacher_seqs)
    student_analysis = analyze_sequences(student_seqs)

    print(f"  Teacher: {teacher_analysis['num_sequences']} seqs, "
          f"avg length {teacher_analysis['avg_length']:.1f}, "
          f"KL from natural: {teacher_analysis['kl_from_natural']:.4f}")
    print(f"  Student: {student_analysis['num_sequences']} seqs, "
          f"avg length {student_analysis['avg_length']:.1f}, "
          f"KL from natural: {student_analysis['kl_from_natural']:.4f}")
    print()

    # Compute ECE if requested
    teacher_ece_results = None
    student_ece_results = None
    if args.compute_ece:
        print("Computing Expected Calibration Error (ECE)...")
        teacher_ece_results = compute_ece(
            teacher_model, teacher_tokenizer, test_sequences[:20], args.device
        )
        student_ece_results = compute_ece(
            student_model, student_tokenizer, test_sequences[:20], args.device
        )
        print(f"  Teacher ECE: {teacher_ece_results['ece']:.4f}")
        print(f"  Student ECE: {student_ece_results['ece']:.4f}")
        ece_improvement = (teacher_ece_results['ece'] - student_ece_results['ece']) / teacher_ece_results['ece'] * 100
        print(f"  ECE improvement: {ece_improvement:+.1f}%")
        print()

    # Summary
    results = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "compression_ratio": compression_ratio,
        "teacher_perplexity": teacher_ppl,
        "student_perplexity": student_ppl,
        "perplexity_ratio": student_ppl / teacher_ppl,
        "kl_divergence": kl_div,
        "teacher_generation": teacher_analysis,
        "student_generation": student_analysis,
    }

    # Add ECE results if computed
    if args.compute_ece:
        results["teacher_ece"] = teacher_ece_results
        results["student_ece"] = student_ece_results

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Compression: {compression_ratio:.1f}x smaller")
    print(f"Perplexity degradation: {student_ppl / teacher_ppl:.2f}x")
    print(f"Output KL divergence: {kl_div:.4f}")
    print()

    if student_ppl / teacher_ppl < 1.5:
        print("ASSESSMENT: Good - Student closely matches teacher quality")
    elif student_ppl / teacher_ppl < 2.0:
        print("ASSESSMENT: Acceptable - Some quality loss but usable")
    elif student_ppl / teacher_ppl < 3.0:
        print("ASSESSMENT: Marginal - Significant quality loss, may need retraining")
    else:
        print("ASSESSMENT: Poor - Consider retraining with fixed code")

    # Save results
    if args.output:
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            return obj

        results = convert_for_json(results)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
