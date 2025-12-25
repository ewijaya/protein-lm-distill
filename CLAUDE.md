# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements knowledge distillation for ProtGPT2, a protein language model. It creates smaller, faster "student" models that learn from the larger "teacher" model (nferruz/ProtGPT2) while retaining similar capabilities for protein sequence generation.

## Project Structure

```
protein-lm-distill/
├── config.py                 # Centralized paths and defaults
├── scripts/                  # Executable scripts
│   ├── train.py             # Main training (parquet data)
│   ├── train_legacy.py      # Legacy training (text files)
│   ├── evaluate.py          # Model evaluation
│   ├── generate.py          # Sequence generation
│   └── batch_train.sh       # Batch training
├── src/                      # Reusable modules
│   ├── distillation.py      # DistillationTrainer class
│   └── esmfold.py           # pLDDT prediction
├── tools/                    # Utilities
│   └── upload_to_hf.py      # HuggingFace upload
├── notebooks/                # Jupyter notebooks
├── data/                     # Training data
└── models/                   # Trained outputs
```

## Commands

### Training

```bash
conda activate pepmlm
python scripts/train.py --temperature 2.0 --alpha 0.5 --n_layer 4 --n_head 4 --n_embd 256
```

For long-running training:
```bash
nohup python scripts/train.py --temperature 2.0 --alpha 0.5 > nohup.out &
```

Batch training with multiple configs:
```bash
./scripts/batch_train.sh
```

### Evaluation

```bash
python scripts/evaluate.py --student_model ./models/your-model --num_samples 100
```

### Generation

```bash
python scripts/generate.py --model littleworth/protgpt2-distilled-tiny --num_sequences 10
```

### Upload to HuggingFace

```bash
python tools/upload_to_hf.py --model_dir ./models/your-model --repo_id username/model-name
```

## Architecture

### DistillationTrainer (`src/distillation.py`)

Custom Trainer extending HuggingFace's Trainer. The distillation loss combines:
- **Soft loss**: KL divergence between student and teacher softened logits (temperature-scaled)
- **Hard loss**: Cross-entropy on ground truth labels
- **Combined**: `loss = alpha * hard_loss + (1 - alpha) * T² * soft_loss`

Key hyperparameters:
- `temperature`: Softens probability distributions (default: 2.0)
- `alpha`: Weight between hard/soft loss (default: 0.5)
- `n_embd`, `n_layer`, `n_head`: Student model architecture

### Configuration (`config.py`)

Centralized paths using environment variables from `~/.zshrc`:
- `HF_DATASETS_CACHE` → `DATA_DIR`
- `WANDB_API_KEY`, `HF_TOKEN` used by respective libraries

### Data

- Training data: Parquet files from UniProt at `$HF_DATASETS_CACHE`
- Legacy text data: `data/all_natural_train_data.txt`

### Model Outputs

Trained models saved to `./models/{model_name}/` with:
- Model weights and tokenizer
- `training_logs.json`: Training metrics
- `training_hyperparameters.json`: Full configuration

## Git Remotes

- `origin`: Bitbucket (git@bitbucket.org:stemrim-bi/protein-lm-distill.git)
- `github`: GitHub (git@github.com:ewijaya/protein-lm-distill.git)

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train.py` | Main training script |
| `src/distillation.py` | DistillationTrainer class |
| `config.py` | Centralized configuration |
| `scripts/evaluate.py` | Model quality evaluation |
| `docs/PRD-master.md` | Comprehensive project requirements and roadmap |
