---
description: Generate expanded PRD for a specific phase (0-5) from PRD-master.md
argument-hint: <phase-number>
allowed-tools: ["Read", "Write", "Glob", "Grep"]
---

# Phase-Specific PRD Generator

Generate a detailed, rigorous PRD document for a specific project phase.

## Input

Phase number: `$1`

Valid phases: 0, 1, 2, 3, 4, 5

## Validation

First, validate the input:
- If `$1` is empty or not a number 0-5, respond with an error message listing valid phases
- Valid phases:
  - 0: Methodological Enhancements
  - 1: Baseline Training
  - 2: Hyperparameter Sweeps
  - 3: Comprehensive Evaluation
  - 4: HuggingFace Update
  - 5: Publication

## Task

1. **Read the master PRD**: @docs/PRD-master.md

2. **Extract the phase section**: Find `## Phase $1:` and extract all content until the next `## Phase` or `---` section break.

3. **Also extract**:
   - Relevant success criteria from the "Success Criteria" section
   - Relevant decision checkpoints
   - Any risk mitigation items for this phase
   - Timeline information from the overview table

4. **Generate an expanded PRD** with this exact structure:

```markdown
# Phase $1: [Phase Name]

> [One-sentence summary of what this phase accomplishes]

## Quick Reference Checklist

- [ ] [Actionable item 1]
- [ ] [Actionable item 2]
- [ ] ...

## Status & Dependencies

| Property | Value |
|----------|-------|
| Status | [From master PRD] |
| Dependencies | [Prior phases required] |
| Inputs | [What artifacts/data are needed] |
| Outputs | [What this phase produces] |

## Objectives

[Clear, measurable goals - not vague statements]

## Deliverables

| Deliverable | Location | Validation Method |
|-------------|----------|-------------------|
| [File/artifact] | [Path] | [How to verify it's correct] |

## Implementation Steps

### Step 1: [Name]
**Goal**: [What this step achieves]
**Commands**:
```bash
[Exact commands to run]
```
**Validation**: [How to verify step succeeded]

### Step 2: [Name]
...

## Success Criteria

| Criterion | Metric | Target | Verification Command |
|-----------|--------|--------|---------------------|
| [What to measure] | [Unit] | [Threshold] | [Command to check] |

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| [What could go wrong] | Low/Med/High | Low/Med/High | [How to handle] |

## Commands Reference

[All relevant commands from the master PRD, formatted for copy-paste]

## Related Resources

- [Links to METHODS.md sections if relevant]
- [Links to other phase dependencies]
- [External references]
```

5. **Key requirements for the expanded PRD**:

   - **Success criteria must be specific and measurable**
     - BAD: "Good perplexity"
     - GOOD: "Perplexity ratio < 2.0 for Medium model"

   - **Each deliverable must have a validation method**
     - BAD: "Model trained"
     - GOOD: "Model saved to ./models/X, verify with `ls -la ./models/X/pytorch_model.bin`"

   - **Commands must be copy-pasteable**
     - Include full paths, environment variables, all flags
     - No placeholders like "X.X" unless the user must fill them in

   - **Implementation steps must be atomic and ordered**
     - Each step should be completable independently
     - Steps should have clear completion criteria

   - **Include dependencies explicitly**
     - What files/artifacts from prior phases are needed
     - What environment setup is required

   - **Code standards**:
     - **Python scripts (`.py`)** for:
       - Training code (long-running, nohup-able) → `scripts/`
       - Evaluation code (batch-run) → `scripts/`
       - Generation/inference code → `scripts/`
       - Upload/deployment tools → `tools/`
       - Reusable modules/libraries → `src/`
     - **Jupyter notebooks (`.ipynb`)** for:
       - Exploratory analysis and visualization
       - Comparison/ablation studies
       - One-off investigations and debugging
       - Documentation with executable code
       - Location: `notebooks/`

   - **Jupyter notebook requirements** (when creating notebooks):
     - Organize with clear numbered sections using markdown headers (e.g., `## 1. Data Loading`)
     - Include a minimal, self-sufficient description before each code cell explaining what that code block does
     - Configure all plots to display inline (`%matplotlib inline`), not saved as external files
     - Ensure notebook is fully executable from top to bottom
     - Example naming: `notebooks/phase_{N}_analysis.ipynb`

6. **Save the output** to `docs/PRD-phase-$1.md`

7. **Confirm completion** by displaying:
   - The output file path
   - A summary of sections generated
   - Any warnings about missing information in the master PRD

## Example

For `/prd 0`, generate `docs/PRD-phase-0.md` containing the expanded Phase 0: Methodological Enhancements PRD with all uncertainty-aware and calibration-aware distillation implementation details, specific success metrics like "ECE score < 0.05", and exact implementation locations like "src/distillation.py:compute_loss()".
