# Phase 6.2: Lysozyme Runs (Deferred)

**Created**: February 20, 2026

## Context

Phase 6.2 completed for `amp` and `conotoxin` families (50 runs total).
Lysozyme was skipped because the **teacher model OOMs on L4 (22 GB)** during
fine-tuning — lysozyme sequences (~130 AA) are much longer than conotoxin (~30 AA),
and `batch_size=4` exceeds GPU memory.

The script was updated in commit `38a998f` to exclude lysozyme.

## What Needs to Run

25 runs: 5 models x 5 subset sizes for the `lysozyme` family.

| Model | Subsets |
|-------|---------|
| teacher | 50, 100, 200, 500, 1000 |
| medium | 50, 100, 200, 500, 1000 |
| small | 50, 100, 200, 500, 1000 |
| tiny | 50, 100, 200, 500, 1000 |
| baseline-tiny | 50, 100, 200, 500, 1000 |

## Fix Options for Teacher OOM

1. **Reduce batch size** — change `BS[teacher]=2` and `GA[teacher]=4` (keeps effective batch size = 8)
2. **Add `--max_length 256`** to truncate long sequences during training
3. **Use a larger GPU** (e.g., g5.xlarge with A10G 24 GB, or g6e with L40S 48 GB)

Option 1 is simplest and most likely sufficient.

## Steps to Run

1. Re-add lysozyme to the script:
   ```bash
   # In scripts/run_phase62.sh, change:
   FAMILIES=(amp conotoxin)
   # To:
   FAMILIES=(lysozyme)
   ```

2. Reduce teacher batch size for lysozyme (edit run_phase62.sh):
   ```bash
   BS[teacher]=2;  GA[teacher]=4
   ```

3. Launch:
   ```bash
   nohup bash /home/ubuntu/storage1/protein-lm-distill/scripts/run_phase62.sh </dev/null & disown
   tail -f /home/ubuntu/storage1/protein-lm-distill/phase62.log
   ```

4. After completion, restore original settings and re-add all families if needed.
