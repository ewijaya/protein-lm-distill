# Phase 6.2: Post-Fix Re-run Plan

**Created**: February 19, 2026

## Context

The `EvalLossEarlyStopping` callback in `scripts/finetune.py` was saving the
**last epoch's** weights instead of the **best** weights when early stopping
triggered. Fixed in commit `1d5ff85` by keeping an in-memory copy of the best
`state_dict` and restoring it at `on_train_end`.

## Affected Runs

13 runs completed before the fix (all `amp` family):

| Model | Subsets |
|-------|--------|
| teacher | 50, 100, 200, 500, 1000 |
| medium | 50, 100, 200, 500, 1000 |
| small | 50, 100, 200 |

All remaining runs (small 500/1000, tiny, baseline-tiny, conotoxin, lysozyme)
will use the fixed code automatically.

## Action Plan

### 1. Let the current run finish

Do not interrupt `run_phase62.sh`. All runs from `amp-small-500` onward use the
fixed code (each invocation spawns a fresh Python process).

### 2. Check impact on affected runs

For each affected run, compare best eval loss vs final eval loss:

```bash
for d in models/finetune/amp-{teacher,medium,small}-{50,100,200,500,1000}; do
    [ -f "$d/training_logs.json" ] || continue
    echo "=== $(basename $d) ==="
    python3 -c "
import json, sys
logs = json.load(open('$d/training_logs.json'))
evals = [l for l in logs if 'eval_loss' in l]
if not evals:
    print('  No eval logs found')
    sys.exit()
best = min(evals, key=lambda x: x['eval_loss'])
last = evals[-1]
gap = last['eval_loss'] - best['eval_loss']
print(f'  Best: {best[\"eval_loss\"]:.4f} (epoch {best[\"epoch\"]:.0f})')
print(f'  Last: {last[\"eval_loss\"]:.4f} (epoch {last[\"epoch\"]:.0f})')
print(f'  Gap:  {gap:.4f} {\"<-- RE-RUN\" if gap > 0.05 else \"(OK)\"}')
"
done
```

### 3. Selectively re-run

Only re-run where the gap is significant (> 0.05 eval loss difference).
Delete the corresponding result JSONs and re-launch:

```bash
# Example: re-run only medium runs (most likely to benefit, fast to re-run)
rm results/finetune/amp-medium-{50,100,200,500,1000}.json
bash scripts/run_phase62.sh
```

**Priority for re-runs (by expected impact):**
1. **medium** - largest overfitting (train/eval gap ~5.7), fast (~30 min total)
2. **small** - moderate overfitting, very fast (~1 min total)
3. **teacher** - smallest overfitting (gap ~2.1), but ~20+ hours to re-run

### 4. Clean up

After re-runs, delete this file.
