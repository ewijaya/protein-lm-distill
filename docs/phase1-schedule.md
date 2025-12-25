# Phase 1 Training Schedule

Started: December 26, 2025 ~07:13

## Training Speed
- Current: ~2.88s/iteration
- GPU: NVIDIA A10G (g5.xlarge)
- Cost: ~$1.00/hr

## Estimated Completion Times

| Model | Config | Steps | Est. Duration | Est. Finish |
|-------|--------|-------|---------------|-------------|
| Tiny | 4L/4H/256E | 88,338 | ~70.5 hrs | **Dec 29, ~01:45** |
| Small | 6L/8H/512E | ~162,000 | ~130 hrs | **Jan 3, ~11:45** |
| Medium | 12L/12H/768E | ~324,000 | ~260 hrs | **Jan 14, ~07:45** |

## Total
- Duration: ~19 days
- Estimated cost: ~$460

## Monitoring
- W&B: https://wandb.ai/ewijaya/PROTGPT2_DISTILLATION
- Log: `tail -f training_baseline.log`

## Output
Models saved to: `./models/protgpt2-distilled-t2.0-a0.5-l{L}-h{H}-e{E}-p0.1-lr*.uniprot/`
