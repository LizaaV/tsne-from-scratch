# t-SNE from scratch — a partial reproduction of van der Maaten & Hinton (2008)

**Paper:** L. van der Maaten and G. Hinton. *Visualizing Data using t-SNE.* Journal of Machine Learning Research 9(86):2579–2605, 2008. [jmlr.org/papers/v9/vandermaaten08a.html](https://jmlr.org/papers/v9/vandermaaten08a.html)

COMP3314 (HKU, 2026–27) group project, Group 35.

Repository: https://github.com/LizaaV/tsne-from-scratch

For the group, a plain-language walkthrough of the code and of every experiment is in `03 Demo/TSNE_03_Demo_Explained` in our shared project folder.

## What is here

| File | What it is | Ours or reused |
|---|---|---|
| `tsne.py` | t-SNE in NumPy: perplexity binary search (Eq. 1), symmetrised $p_{ij}$ (§3.1), Student-t $q_{ij}$ (Eq. 4), gradient (Eq. 5), optimiser of §3.4 (momentum, Jacobs gains, early exaggeration); symmetric SNE as an option; Sammon mapping (§6.1) | **ours**, written from the paper's equations |
| `check_gradient.py` | finite-difference check of the gradient; perplexity check | ours |
| `worked_example.py` | the 3-point hand example of `TSNE_01_Paper_Explained.md`, Part 7 | ours |
| `experiments.py` | experiments E1–E8 below | ours; uses scikit-learn for Isomap, LLE, its own t-SNE (cross-check only), 1-NN classifier, trustworthiness |
| `demo_live.py` | animated run for the presentation | ours |
| `figures/`, `results/` | outputs of `experiments.py` | generated |

No code was copied from the authors' Matlab implementation or from scikit-learn's t-SNE.

## Setup and running

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python check_gradient.py          # about 5 s
python experiments.py all         # E1-E7, about 15-20 min on a laptop
python experiments.py mnist       # E8: the paper's MNIST-6000 setting (downloads MNIST, ~1 h)
# if the download fails with a certificate error: MNIST_NPZ=/path/to/mnist.npz python experiments.py mnist
python demo_live.py --perplexity 30
```

Tested with Python 3.7, NumPy 1.21, scikit-learn 1.0, matplotlib 3.5.

## Data

The paper uses MNIST (6,000 of 60,000 images, 28 × 28), Olivetti faces and COIL-20. We use scikit-learn's bundled **digits** set (1,797 images, 8 × 8 = 64 pixels, 10 classes): same kind of data, no download, and small enough to run every experiment several times. `experiments.py mnist` runs the paper's own MNIST setting.

## Settings (as in the paper)

PCA to 30 dimensions → perplexity 40 → $Y^{(0)} \sim \mathcal N(0, 10^{-4}I)$ → 1,000 iterations, learning rate 100 with Jacobs (1988) gains (+0.2 / ×0.8, minimum 0.01), momentum 0.5 for $t < 250$ and 0.8 after, early exaggeration ×4 for 50 iterations. Isomap and LLE: $k = 12$. Sammon: 500 iterations of Sammon's diagonal Newton method.

## Results

### E1 — t-SNE vs Sammon, Isomap, LLE, PCA (paper Figures 2–3 analogue)

| Method (same PCA-30 input) | 1-NN error, 10-fold CV | Trustworthiness ($k = 12$) | Time (s) |
|---|---|---|---|
| t-SNE (ours) | 1.9% | 0.992 | 93.4 |
| t-SNE (scikit-learn) | 2.1% | 0.993 | 10.6 |
| LLE k=12 | 23.3% | 0.879 | 0.4 |
| Isomap k=12 | 30.0% | 0.857 | 3.2 |
| Sammon (ours) | 41.7% | 0.858 | 172.8 |
| PCA | 42.5% | 0.832 | 0.0 |
| original 64-D pixels (no map) | 2.5% | — | — |

Paper (MNIST, random-walk t-SNE, §5): 1-NN error 5.75% in 784-D vs 5.13% on the 2-D map. **Same pattern here.** Figure: `figures/E1_compare_methods.png`.

### Other experiments

| # | Question | Result | Figure |
|---|---|---|---|
| E2 | Does Figure 1 come out of the formulas? | yes in shape; SNE max attraction 20.0 vs max repulsion 0.86 (paper ≈ 19 vs 1) | `E2_gradients_fig1.png` |
| E3 | Is the Student-t really what fixes crowding? | yes: same everything, Gaussian map kernel → 1-NN 8.2%, Student-t → 1.9% | `E3_crowding_sne_vs_tsne.png` |
| E4 | Perplexity 5 / 15 / 40 / 100 | 1-NN 2.3%, 2.2%, 1.9%, 2.2%; the picture changes a lot | `E4_perplexity.png` |
| E5 | Does early exaggeration help? | slightly on digits: KL 0.638 vs 0.663, 1-NN 1.9% vs 2.2% | `E5_*.png`, `E5_tsne_animation.gif` |
| E6 | Run-to-run variation (5 seeds) | KL 0.637 ± 0.004, 1-NN 2.1 ± 0.1%; layout changes | `E6_seeds.png` |
| E7 | Are cluster sizes and gaps preserved? (our extension) | no: true spread ratio 5.1 → 1.4; gap ratio 5.2 → 3.4 | `E7_sizes_and_gaps.png` |
| E8 | MNIST-6000 as in the paper | scripted, not yet run | `E8_mnist6000.png` |

## Correctness checks

- **Gradient:** analytic gradient (Eq. 5) against central finite differences on a random 40-point problem: relative difference $6.8 \times 10^{-9}$ (Student-t) and $2.1 \times 10^{-9}$ (Gaussian / symmetric SNE).
- **Perplexity search:** every row reaches the target perplexity 10 to within $10^{-4}$.
- **Outlier guarantee (§3.1):** $\sum_j p_{ij} > 1/2n$ for every point.
- **Against scikit-learn:** same input and settings give the same 1-NN error and trustworthiness to within about 0.1 percentage points and 0.001 (E1).

## Discrepancies with the paper, and challenges

- **Sign of the update.** Algorithm 1 (and the SNE update in §2) prints $Y^{(t)} = Y^{(t-1)} + \eta\,\partial C/\partial Y + \ldots$, which is gradient ascent. We use $-\eta\,\partial C/\partial Y$.
- **Figure 1 normalisation** is not given in the paper. With unnormalised unit-width kernels our SNE panel has max attraction 20 and max repulsion 0.86 (paper: about 19 and 1). Our t-SNE panel spans ±0.65 where the paper's colour bar shows ±1. The shapes agree.
- **Appendix A** drops a factor $d_{ij}$ twice in its intermediate steps. The final gradient is right, as our finite-difference check confirms.
- **Jacobs gains:** the paper names the scheme but not its constants. We use the common +0.2 / ×0.8 / 0.01.
- **Run time:** our $O(n^2)$ NumPy version needs about 95 s for 1,797 points (scikit-learn's Barnes–Hut version: about 11 s). MNIST-6000 needs roughly 10× more memory and time.
- **Not reproduced:** random-walk t-SNE (§5), Olivetti faces, COIL-20, the supplementary comparisons (CCA, MVU, Laplacian Eigenmaps).
