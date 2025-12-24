# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements knowledge distillation for ProtGPT2, a protein language model. It creates smaller, faster "student" models that learn from the larger "teacher" model (nferruz/ProtGPT2) while retaining similar capabilities for protein sequence generation.

## Commands

### Running Distillation Training

Activate the conda environment and run training:
```bash
conda activate pepmlm
./distill_using_nferruz_dataset.py --temperature 2.0 --alpha 0.5 --n_layer 4 --n_head 4 --n_embd 256 --train_size_prop 0.1 --learning_rate 1e-3
```

Run with nohup for long training sessions:
```bash
nohup sh -c './distill_using_nferruz_dataset.py --temperature 2.0 --alpha 0.5 > nohup.out' &
```

Use wrap.sh to run with multiple parameter configurations:
```bash
./wrap.sh
```

### Inference

Generate protein sequences using a trained model:
```bash
python inference.py
```

### Upload Model to Hugging Face

```bash
python notebooks/submit_to_hf_repo.py
```

## Architecture

### Distillation Training (`distill_using_nferruz_dataset.py`)

The main training script implements a custom `DistillationTrainer` extending HuggingFace's `Trainer`. The distillation loss combines:
- **Soft loss**: KL divergence between student and teacher softened logits (temperature-scaled)
- **Hard loss**: Cross-entropy on ground truth labels
- **Combined**: `loss = alpha * hard_loss + (1 - alpha) * soft_loss`

Key hyperparameters:
- `--temperature`: Softens probability distributions (default: 2.0)
- `--alpha`: Weight between hard/soft loss (default: 0.5)
- `--n_embd`, `--n_layer`, `--n_head`: Student model architecture
- `--train_size_prop`: Proportion of validation set to use for training

### Data

- Training data: Parquet files from UniProt stored in `/home/ubuntu/storage2/various_hugging_face_data_and_models/data/`
- Legacy text data: `data/all_natural_train_data.txt`

### Model Outputs

Trained models are saved to `./models/{model_name}/` with:
- Model weights and tokenizer
- `training_logs.json`: Training metrics
- `training_hyperparameters.json`: Full configuration

### Tracking

Training runs are logged to Weights & Biases under project `PROTGPT2_DISTILLATION`.

## Git Remotes

- `origin`: Bitbucket (git@bitbucket.org:stemrim-bi/distilling_protgpt2.git)
- `github`: GitHub (git@github.com:ewijaya/distilling_protgpt2.git)

## Notes

- The DeepSpeed version (`distill_using_nferruz_dataset.DEEPSPEED.py`) is non-functional; use the standard version which supports multiple GPUs via DataParallel
- ESMFold pLDDT prediction (`esmfold_pldtt_prediction.py`) evaluates generated sequence quality
