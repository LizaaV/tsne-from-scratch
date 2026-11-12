# Reproduction of the paper's Section 4 comparison (Figures 2–5), with numbers

van der Maaten & Hinton (2008), *Visualizing Data using t-SNE*, JMLR 9:2579–2605.
Run on 12 Nov 2026 with `submission/code/compare3.py`. Total time 1,711 s (28.5 min) on an Apple Silicon laptop.
All numbers below are **ours**. The paper reports no numbers for Figures 2–5.

**Setup (as in the paper, §4.2 and Table 1)**
- Every method gets the same input: PCA to 30 dimensions.
- t-SNE: perplexity 40, 1,000 iterations, the paper's optimiser.
- Sammon: up to 500 Newton iterations.
- Isomap and LLE: k = 12.
- One run per method, seed 0.

**Measures**
- 1-NN error: 10-fold cross-validation, as in E1 and E8.
- Trustworthiness: k = 12, measured against the 30-D PCA input, as in E1.
- Labels only colour the points.

**Reused code (unmodified copies)**
- `tsne()`, `sammon()` and `pca()` from `tsne.py`.
- `one_nn_error()` and `scatter()` from `experiments.py`.
- The Isomap, LLE and trustworthiness calls, with the same arguments as `e1_compare()`.

## a) 1-NN error on the 2-D map (lower is better)

| Method | MNIST (6,000) | Olivetti (400) | COIL-20 (1,440) |
|---|---|---|---|
| **t-SNE** | **5.92%** | **6.25%** | **3.61%** |
| Sammon | 57.97% | 64.00% | 30.35% |
| Isomap | 53.75% | 64.25% | 20.92%† |
| LLE | 61.68% | 60.75% | 18.75%† |
| *Raw pixels (no map)* | *6.03%* | *5.00%* | *4.79%* (5.99% on the 1,152-point subset†) |

† Isomap and LLE on COIL-20 map only the largest connected part of the k = 12 neighbour graph: 1,152 of 1,440 points, 16 of 20 objects. Compare them with the 5.99% raw-pixel error on the same subset.

## b) Trustworthiness, k = 12 (higher is better; 1 = no false neighbours)

| Method | MNIST | Olivetti | COIL-20 |
|---|---|---|---|
| **t-SNE** | **0.986** | **0.945** | **0.997** |
| Sammon | 0.788 | 0.869 | 0.915 |
| Isomap | 0.788 | 0.852 | 0.943† |
| LLE | 0.700 | 0.716 | 0.880† |

## Run time (seconds)

| Method | MNIST | Olivetti | COIL-20 |
|---|---|---|---|
| t-SNE (ours, O(n²) NumPy) | 90.7 | 0.5 | 5.3 |
| Sammon (ours, O(n²) NumPy) | 1,587.5 | 0.5 | 10.4 |
| Isomap (scikit-learn) | 4.3 | <0.1 | 0.1 |
| LLE (scikit-learn) | 2.1 | <0.1 | <0.1 |

t-SNE final KL: MNIST 1.374 (identical to E8), Olivetti 0.417, COIL-20 0.291.

## Visual comparison with the paper (Figures 2–5)
- **MNIST** (Figures 2–3):
  - t-SNE gives 10 separate clusters, with some 4/9 mixing.
  - Sammon gives a "ball" in which only 0 and 1 stand somewhat apart. The paper says 0, 1 and 7; our 7s mix with 4 and 9.
  - Isomap and LLE overlap heavily. LLE puts most points at the centre, with a long spike of 1s.
  - All of this matches the paper's description.
- **Olivetti** (Figure 4):
  - t-SNE puts most people's 10 photos in a small tight group.
  - Sammon, Isomap and LLE scatter each person's photos. LLE again crowds most points in the centre.
  - This matches the paper.
- **COIL-20** (Figure 5):
  - t-SNE draws most objects as closed loops, one point per viewing angle around the full 360° turn.
  - Objects 3, 5, 6, 9 and 19 form long parallel "sausages" that touch.
  - The paper describes the same thing for its toy cars, which look alike at the same angle.
  - Sammon, Isomap and LLE mix the objects.
  - All of this matches the paper.

## c) Limitations
1. **MNIST sample differs from the paper.** Ours is 6,000 images drawn with seed 0 from all 70,000 OpenML images (training plus test), the same sample as E8. The paper used 6,000 of the 60,000 training images and gives no seed.
2. **Olivetti resolution differs.** scikit-learn's copy is 64 × 64 = 4,096 pixels; the paper used 92 × 112 = 10,304.
3. **COIL-20 resize is ours.** The processed set (`coil-20-proc.zip`, Columbia CAVE) is 128 × 128. We averaged each 4 × 4 block to get 32 × 32 = 1,024 pixels, which equals area / BOX resampling, then divided by 255. The paper doesn't say how it got to 32 × 32.
4. **Connected parts of the k = 12 neighbour graph** (made symmetric):
   - MNIST: 1 part (6,000 of 6,000 points mapped).
   - Olivetti: 1 part (400 of 400).
   - COIL-20: **5 parts**, sizes 1,152, 72, 72, 72 and 72. Isomap and LLE map only the 1,152 points. **Objects 13, 16, 17 and 20 are dropped**: each forms its own separate ring of 72 views.
   - t-SNE and Sammon always use all points.
5. **One run per method** (seed 0), like the paper. E6 found a run-to-run spread of about ±0.1 points in t-SNE's 1-NN error on digits. Small differences between methods should not be over-read.
6. **Settings are not tuned.** We used the paper's settings, exactly as the paper did.
7. **The paper reports no numbers for Figures 2–5**, so our comparison with the paper is **visual only**. All 1-NN and trustworthiness values are ours.
8. **Sammon:** our version uses Sammon's diagonal Newton step with step halving. The paper doesn't say exactly which Newton variant it used.

## d) What went wrong or looked odd
- **Nothing failed.** All 12 runs finished, and none raised a warning.
- **Sammon stopped early on Olivetti and COIL-20:**
  - Our `sammon()` stops when 20 step halvings can't lower the stress, i.e. it has reached a minimum.
  - Olivetti stopped after **304** of 500 iterations, and COIL-20 after **496**.
  - We didn't measure this for MNIST, because a re-run takes about 26 min.
  - This is why Olivetti's Sammon took only 0.5 s.
- **t-SNE's map is slightly worse than the raw pixels on Olivetti** (6.25% vs 5.00%; on 400 points that's 25 vs 20 errors). On MNIST (5.92% vs 6.03%) and COIL-20 (3.61% vs 4.79%) the map is slightly better.
  - With only 10 photos per person and perplexity 40, each point's neighbourhood spans about four people. That probably blurs some identities.
  - This is our guess; we didn't test it.
- **COIL-20 Isomap and LLE errors look lower than Sammon's, but they're scored on an easier subset** (16 objects, not 20). The fair baseline for them is 5.99%, not 4.79%.
- **Isomap and Sammon have almost the same trustworthiness on MNIST** (0.7879 vs 0.7877), though their maps look very different. That's a coincidence; don't read anything into it.
- **The MNIST t-SNE result is bit-for-bit identical to E8** (KL 1.3742, 1-NN 5.92%), which confirms the pipeline is deterministic.

## Files
All of these are in this folder, `submission/outputs/compare3/`.
- `<method>_<dataset>.png`: 12 maps, titled "<method> on <data set> (ours)".
- `_overview_all12.png`: all 12 maps in one grid. Rows: MNIST, Olivetti, COIL-20. Columns: t-SNE, Sammon, Isomap, LLE.
- `results.json`: every number above, plus the component sizes.
- `run.log`: the console output from the run.

**To reproduce:**
1. Unzip `coil-20-proc.zip` into `submission/data/coil-20/`. The `data/` folder is not tracked in git.
2. From `submission/code/`, run `python compare3.py`.
3. Use pandas below version 3.
