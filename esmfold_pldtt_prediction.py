from transformers import AutoTokenizer, EsmForProteinFolding
import torch

# Initialize the model and tokenizer (only done once)
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
model = model.cuda()
# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)


def predict_plddt(sequence):
    """
    Predict the pLDDT score for a given protein sequence.

    Args:
        sequence (str): The protein sequence as a string.

    Returns:
        float: The pLDDT score.
    """
    with torch.no_grad():
        output = model.infer_pdb(sequence)
    struct = output["pdb_bb"]
    plddt = output["plddt"].mean().item()
    return plddt


if __name__ == "__main__":
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    plddt_score = predict_plddt(sequence)
    print(f"pLDDT score: {plddt_score:.2f}")
