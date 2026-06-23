# arXiv Submission Package - Paper 3

This directory contains the arXiv-ready source package for:

**Policy-Relevant Block Abstraction for Sequential Farmland Consolidation Scenario Screening**

## Package Layout

- `source/main.tex`: arXiv compilation entry point.
- `source/paper3_*.png`: manuscript figures required by `main.tex`.
- `source/paper3_*.tex`: table fragments required by `main.tex`.

Upload the contents of `source/` as the arXiv source package. The source folder is intentionally flat so that arXiv can compile from the upload root without needing nested output directories.

## Local Compile Check

From `submission/arxiv/source/`:

```bash
pdflatex main.tex
pdflatex main.tex
```

The manuscript uses an inline `thebibliography` block, so BibTeX is not required for this arXiv package.

## Excluded Files

Do not upload generated files such as `main.pdf`, `main.aux`, `main.log`, `main.out`, or SyncTeX files. Do not upload the repository's older journal-specific submission packages.

## Data Boundary

The public repository includes code, derived result artifacts, figure-generation scripts, and table/figure inputs used by the preprint. Restricted Third National Land Survey parcel geometries are not redistributed.
