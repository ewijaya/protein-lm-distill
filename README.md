# Distilling ProtGPT2

Knowledge distillation for [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2), creating smaller and faster protein language models that retain the generative capabilities of the original model.

## Overview

This project trains compact "student" GPT-2 models to mimic the behavior of the full ProtGPT2 "teacher" model using knowledge distillation. The distillation process combines:

- **Soft loss**: KL divergence between temperature-softened logits from student and teacher
- **Hard loss**: Standard cross-entropy loss on ground truth labels

The combined loss enables student models to learn both the teacher's output distribution and the correct token predictions.

## Requirements

### AWS Instance Requirements

| Task | Minimum Instance | Recommended | Notes |
|------|------------------|-------------|-------|
| **Training** | g4dn.xlarge (T4 16GB) | g5.xlarge (A10G 24GB) | GPU required for distillation |
| **Inference** | g4dn.xlarge | g4dn.xlarge | Single GPU sufficient |
| **Evaluation** | g4dn.xlarge | g5.xlarge | Loads both teacher and student |
| **CPU-only inference** | m5.xlarge | m5.2xlarge | Slow but works for small batches |

**Memory considerations:**
- Teacher model (ProtGPT2): ~500M parameters (~2GB GPU memory)
- Student model: varies by config (tiny ~10M, medium ~125M)
- Training requires both models loaded simultaneously

### Software Requirements

```bash
conda activate pepmlm
```

Required packages:
- transformers
- torch (with CUDA for GPU instances)
- datasets
- wandb
- pyarrow (for parquet data)

## Usage

### Training a Distilled Model

```bash
./distill_using_nferruz_dataset.py \
    --temperature 2.0 \
    --alpha 0.5 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 256 \
    --train_size_prop 0.1 \
    --learning_rate 1e-3
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--temperature` | Softens probability distributions for knowledge transfer | 2.0 |
| `--alpha` | Weight for hard loss (1-alpha for soft loss) | 0.5 |
| `--n_layer` | Number of transformer layers in student | 4 |
| `--n_head` | Number of attention heads | 4 |
| `--n_embd` | Embedding dimension | 256 |
| `--train_size_prop` | Fraction of dataset to use | 0.1 |
| `--learning_rate` | Initial learning rate | 1e-3 |

### Batch Training

Edit `wrap.sh` to define parameter sets, then run:

```bash
./wrap.sh
```

### Generating Protein Sequences

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline

model_name = "littleworth/protgpt2-distilled-tiny"  # or local path
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)
sequences = pipeline(
    "<|endoftext|>",
    max_length=100,
    do_sample=True,
    top_k=950,
    repetition_penalty=1.2,
    num_return_sequences=10,
    eos_token_id=0
)
```

## Model Variants

Pre-trained distilled models available on Hugging Face:

- [littleworth/protgpt2-distilled-tiny](https://huggingface.co/littleworth/protgpt2-distilled-tiny) - Smallest variant
- [littleworth/protgpt2-distilled-medium](https://huggingface.co/littleworth/protgpt2-distilled-medium) - Medium variant

## Output Structure

Trained models are saved to `./models/{model_name}/` containing:

```
models/protgpt2-distilled-t2.0-a0.5-l4-h4-e256-p0.1-lr1e-03.uniprot/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── training_logs.json
└── training_hyperparameters.json
```

## Evaluation

### Model Quality Evaluation

Compare a distilled model against the teacher:

```bash
python evaluate_model.py \
    --student_model ./models/your-model \
    --num_samples 100 \
    --output results.json
```

This evaluates:
- **Perplexity**: How well the model predicts held-out sequences
- **KL divergence**: How closely student matches teacher outputs
- **Generation quality**: Amino acid distribution of generated sequences

Example output:
```
Compression: 36.0x smaller
Perplexity degradation: 1.45x
Output KL divergence: 0.0234

ASSESSMENT: Good - Student closely matches teacher quality
```

### Structural Plausibility (ESMFold)

Use ESMFold to evaluate the structural plausibility of generated sequences:

```python
from esmfold_pldtt_prediction import predict_plddt

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
plddt_score = predict_plddt(sequence)
print(f"pLDDT score: {plddt_score:.2f}")
```

Higher pLDDT scores indicate more structurally plausible sequences.

**Note:** ESMFold requires significant GPU memory (~16GB+). Use g5.xlarge or larger.

## Training Tracking

Training metrics are logged to [Weights & Biases](https://wandb.ai) under the project `PROTGPT2_DISTILLATION`.

## References

- Ferruz, N., et al. (2022). ProtGPT2 is a deep unsupervised language model for protein design. *Nature Communications*.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *arXiv:1503.02531*.

## License

See LICENSE file for details.
