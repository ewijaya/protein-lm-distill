---
description: Compile the paper (figures, pdf, or clean build artifacts)
argument-hint: [paper|figures|all|clean]
allowed-tools: ["Bash"]
---

# Paper Compilation

Compile the LaTeX manuscript by running Makefile targets in the `paper/` directory.

## Input

Target: `$1`

## Instructions

1. Determine the make target from `$1`:
   - Empty or `paper` → `make paper` (pdflatex + bibtex + pdflatex x2)
   - `figures` → `make figures` (regenerate all figure PDFs)
   - `all` → `make all` (figures + paper)
   - `clean` → `make clean` (remove build artifacts)
   - Anything else → print error: "Invalid target. Use: paper, figures, all, or clean"

2. Run the command:
   ```
   make -C paper <target>
   ```

3. Report result:
   - On success: confirm which target ran and mention output file (`paper/main.pdf`) if applicable
   - On failure: show the error output so the user can fix LaTeX/figure issues
