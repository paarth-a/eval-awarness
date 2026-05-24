# Paper artifact

This directory contains the LaTeX source of the EMNLP 2026 submission
*"Test Smells: A Causal Decomposition of Eval-Identifiability in Modern LLM
Benchmarks"* that this release accompanies.

- `paper_v2.tex` — main paper source. Compiles to ~8 main pages + appendix
  with the EMNLP 2026 `acl.sty` template. The TikZ Figure 1 is inline (no
  external image files needed).
- `refs.bib` — bibliography (34 entries, all cited).

## Compile

```bash
cd paper/
# Drop acl.sty + acl_natbib.bst from the EMNLP 2026 LaTeX bundle here, then:
pdflatex paper_v2.tex
bibtex paper_v2
pdflatex paper_v2.tex
pdflatex paper_v2.tex
```

## Linking paper claims to release artifacts

Every numerical claim in `paper_v2.tex` traces to a JSON file in the
release at `../results/` or `../tells/data/`. The mapping is documented
two ways:

1. **`../README.md`** — paper-section → script → output-JSON ledger.
2. **`../METHODOLOGY.md`** — per-construct procedural description with
   explicit JSON references.

Reviewers verifying a number from the paper should:
1. Find the paper section / table / cited number.
2. Look up the relevant row in `../README.md`'s ledger.
3. Open the named JSON in `../results/` to confirm the value.
4. Confirm SHA256 against `../SHA256SUMS` if data integrity matters.

The release is anonymous. Author identifiers in the paper are deliberately
placeholders (`Anonymous EMNLP submission`); replacement happens at
camera-ready.
