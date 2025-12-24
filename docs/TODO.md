# Project TODO List

## Phase 1: Initial GPU Validation (Before Full-Scale Training)

These minimal tests verify the reorganized codebase works correctly on a GPU instance.

### 1.1 Environment Setup
- [ ] Switch to GPU instance (g4dn.xlarge minimum, g5.xlarge recommended)
- [ ] Verify conda environment: `conda activate pepmlm`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify GPU access: `python -c "import torch; print(torch.cuda.is_available())"`

### 1.2 Quick Smoke Test (5-10 min)
Run a minimal training to verify imports and GPU usage work:

```bash
python scripts/train.py \
    --temperature 2.0 \
    --alpha 0.5 \
    --n_layer 2 \
    --n_head 2 \
    --n_embd 128 \
    --train_size_prop 0.001 \
    --learning_rate 1e-3
```

**Expected outcome:**
- [ ] Teacher model loads successfully
- [ ] Student model initializes
- [ ] Dataset loads from parquet files
- [ ] Training loop starts without errors
- [ ] W&B logging works (check wandb.ai dashboard)
- [ ] Model saves to `./models/` directory

### 1.3 Evaluate Existing Models
Test evaluation script with pre-trained HuggingFace model:

```bash
python scripts/evaluate.py \
    --student_model littleworth/protgpt2-distilled-tiny \
    --num_samples 20 \
    --output results_test.json
```

**Expected outcome:**
- [ ] Both models load
- [ ] Perplexity computed
- [ ] KL divergence computed
- [ ] Sequences generated
- [ ] Results saved to JSON

### 1.4 Generation Test
```bash
python scripts/generate.py \
    --model littleworth/protgpt2-distilled-tiny \
    --num_sequences 5 \
    --max_length 50
```

**Expected outcome:**
- [ ] Model loads
- [ ] Sequences generated in FASTA format
- [ ] Output looks like valid protein sequences

### 1.5 Legacy Script Test (Optional)
```bash
python scripts/train_legacy.py \
    --temperature 2.0 \
    --alpha 0.5 \
    --data_file data/train_small.txt
```

---

## Phase 2: Full-Scale Training

After Phase 1 passes, proceed with production training.

### 2.1 Training Configurations to Run
- [ ] Tiny model: `--n_layer 4 --n_head 4 --n_embd 256`
- [ ] Small model: `--n_layer 6 --n_head 6 --n_embd 512`
- [ ] Medium model: `--n_layer 12 --n_head 12 --n_embd 768`

### 2.2 Hyperparameter Experiments
- [ ] Temperature sweep: 1.0, 2.0, 5.0, 10.0
- [ ] Alpha sweep: 0.1, 0.3, 0.5, 0.7, 0.9
- [ ] Learning rate sweep: 1e-4, 5e-4, 1e-3

### 2.3 Training Runs
Edit `scripts/batch_train.sh` with desired configurations:
```bash
nohup ./scripts/batch_train.sh > batch_training.log 2>&1 &
```

---

## Phase 3: Evaluation & Analysis

### 3.1 Evaluate All Trained Models
- [ ] Run `scripts/evaluate.py` on each model in `./models/`
- [ ] Compare perplexity ratios across configurations
- [ ] Identify best performing models

### 3.2 Structural Plausibility (ESMFold)
For top models:
```python
from src.esmfold import predict_plddt
# Generate sequences and evaluate pLDDT scores
```

### 3.3 Compare with Teacher
- [ ] Generate sequences from teacher and students
- [ ] Compare amino acid distributions
- [ ] Compare sequence length distributions

---

## Phase 4: Model Release

### 4.1 Select Best Models
- [ ] Choose best tiny/small/medium variants
- [ ] Document hyperparameters used

### 4.2 Upload to HuggingFace
```bash
python tools/upload_to_hf.py \
    --model_dir ./models/best-model \
    --repo_id username/protgpt2-distilled-xxx
```

### 4.3 Update Model Cards
- [ ] Add model description
- [ ] Document training configuration
- [ ] Add usage examples
- [ ] Include evaluation metrics

---

## Phase 5: Documentation & Cleanup

### 5.1 Documentation
- [ ] Update README with final results
- [ ] Add example notebooks
- [ ] Document best practices learned

### 5.2 Code Cleanup
- [ ] Remove unused model checkpoints
- [ ] Archive old W&B runs
- [ ] Clean up any temporary files

---

## Quick Reference: AWS Instance Costs (as of 2024)

| Instance | GPU | Hourly Cost | Use Case |
|----------|-----|-------------|----------|
| g4dn.xlarge | T4 16GB | ~$0.53 | Development, small models |
| g5.xlarge | A10G 24GB | ~$1.01 | Training, evaluation |
| g5.2xlarge | A10G 24GB | ~$1.21 | Faster training |
| p3.2xlarge | V100 16GB | ~$3.06 | Large models |

---

## Notes

- Always run Phase 1 tests on a new GPU instance before long training
- Monitor W&B dashboard during training for loss curves
- Save evaluation results for each model configuration
- Keep training logs for reproducibility
