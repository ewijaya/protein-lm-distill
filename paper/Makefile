LATEX = pdflatex
BIBTEX = bibtex
PYTHON = python

MAIN = main
SECTIONS = $(wildcard sections/*.tex)
FIGSCRIPTS = $(wildcard figures/scripts/fig*.py)

.PHONY: all clean figures paper

all: figures paper

figures:
	mkdir -p figures/pdf
	cd .. && for script in paper/figures/scripts/fig*.py; do \
		echo "=== Generating $$(basename $$script .py) ===" && \
		$(PYTHON) $$script || exit 1; \
	done

paper: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(SECTIONS) references.bib
	$(LATEX) -interaction=nonstopmode $(MAIN) || true
	$(BIBTEX) $(MAIN) || true
	$(LATEX) -interaction=nonstopmode $(MAIN) || true
	$(LATEX) -interaction=nonstopmode $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf $(MAIN).synctex.gz
	rm -f figures/pdf/*.pdf
