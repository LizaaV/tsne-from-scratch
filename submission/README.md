# Submission: Group 35, t-SNE

COMP3314 group project. Paper: L. van der Maaten and G. Hinton, *Visualizing Data using t-SNE*, JMLR 9(86):2579–2605, 2008.

This folder holds what we hand in and the runs behind it. Paper details, setup and all results are in the [README at the repository root](../README.md).

## What is here

| Path | What it is |
|---|---|
| `report/report.tex` | The written report (NeurIPS 2026 template, preprint option) |
| `report/report.pdf` | The compiled report |
| `report/extra/neurips_2026.sty` | The official style file |
| `report/figures/` | The report's figures: ours (`B4_*`, `ours_*`; TikZ sources in `figures/source/`) and the paper's Figures 2–5 and 7 (`paper_fig*`, reproduced from the paper) |
| `code/tsne.py` | Unmodified copy of `resources/tsne.py` (our from-scratch t-SNE, Sammon mapping and PCA) |
| `code/experiments.py` | Copy of `resources/experiments.py` that writes to `outputs/`; used for E8 (MNIST-6000) |
| `code/compare3.py` | The paper's Section 4 comparison on MNIST, Olivetti and COIL-20 |
| `outputs/figures/`, `outputs/results/` | E8 figure and numbers |
| `outputs/compare3/` | The 12 maps, `results.json`, `run.log`, and `REPORT.md` (full write-up) |
| `data/` | Downloaded data sets; not tracked in git |

## Running the code

From `submission/code/`, with the environment from the root README:

```bash
pip install "pandas<3"
python experiments.py mnist       # E8, about 2 min
python compare3.py                # about 30 min; needs COIL-20 unzipped into ../data/coil-20/
```

## Building the report

From `submission/report/`:

```bash
tectonic -X compile -Z search-path=extra report.tex
# or: TEXINPUTS=extra: pdflatex report.tex
```

The report reads its figures from `report/figures/` and our maps from `outputs/compare3/` (see `\graphicspath` in `report.tex`).

## Hand-in checklist

- [ ] Main text is 4 pages or fewer (references and appendix excluded)
- [x] All four names and affiliations are on the first page
- [x] The repository link is in the report
- [x] Every figure and number is labelled as the paper's or ours
- [ ] Exported as `Group_35.pdf` and uploaded to Moodle by Sun 29 Nov 2026, 23:59
