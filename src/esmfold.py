"""
ESMFold pLDDT prediction for evaluating protein sequence structural plausibility.

This module provides functions to predict pLDDT (predicted Local Distance Difference Test)
scores using ESMFold, which indicates how structurally plausible a protein sequence is.

Note: Requires significant GPU memory (~16GB+). Use g5.xlarge or larger AWS instance.
"""

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

# Lazy initialization - models loaded on first use
_tokenizer = None
_model = None


def _load_model():
    """Load ESMFold model and tokenizer (lazy initialization)."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        _model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        if torch.cuda.is_available():
            _model = _model.cuda()
        # Optionally set chunk size for reduced memory usage:
        # _model.set_chunk_size(128)
    return _tokenizer, _model


def predict_plddt(sequence: str) -> float:
    """
    Predict the pLDDT score for a given protein sequence.

    pLDDT (predicted Local Distance Difference Test) scores range from 0-100,
    where higher scores indicate more confident/plausible structures:
    - >90: Very high confidence
    - 70-90: Confident
    - 50-70: Low confidence
    - <50: Very low confidence

    Args:
        sequence: The protein sequence as a string (amino acid letters).

    Returns:
        The mean pLDDT score across all residues.
    """
    _, model = _load_model()

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    plddt = output["plddt"].mean().item()
    return plddt


def predict_plddt_batch(sequences: list[str]) -> list[float]:
    """
    Predict pLDDT scores for multiple protein sequences.

    Args:
        sequences: List of protein sequences.

    Returns:
        List of mean pLDDT scores for each sequence.
    """
    return [predict_plddt(seq) for seq in sequences]


if __name__ == "__main__":
    # Example usage
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    plddt_score = predict_plddt(sequence)
    print(f"pLDDT score: {plddt_score:.2f}")
