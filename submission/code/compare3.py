"""Reproduce the paper's Section 4 comparison (Figures 2-5) on MNIST, Olivetti and COIL-20,
with numbers added (1-NN error, trustworthiness, run time).

Reuses, unmodified: tsne(), sammon(), pca() from tsne.py; one_nn_error(), scatter() and the
E1 method calls (same arguments) from experiments.py. Only loaders and the loop are new.

Run from submission/code/:  python compare3.py      (writes ../outputs/compare3/)
COIL-20: download https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip
and unzip it into submission/data/coil-20/ first. MNIST and Olivetti download automatically.
Needs pandas<3 (pandas 3 breaks scikit-learn's MNIST loader).
"""
import json
import os
import sys
import time
import warnings

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import fetch_openml, fetch_olivetti_faces
from sklearn.manifold import Isomap, LocallyLinearEmbedding, trustworthiness
from sklearn.neighbors import kneighbors_graph

import experiments                      # unmodified copy; reuse its helpers
from experiments import one_nn_error, scatter
from tsne import tsne, sammon, pca

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "compare3")
COIL_DIR = os.path.join(HERE, "..", "data", "coil-20", "coil-20-proc")   # unzip coil-20-proc.zip here (not in git)
os.makedirs(OUT, exist_ok=True)
K = 12                                   # Isomap / LLE neighbours and trustworthiness k (paper Table 1)


# --------------------------------------------------------------------------- data
def load_mnist():
    """Same sample as E8: 6,000 of all 70,000 OpenML images, seed 0, pixels /255."""
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    idx = np.random.RandomState(0).choice(len(X), 6000, replace=False)
    return X[idx] / 255.0, y[idx].astype(int)


def load_olivetti():
    """scikit-learn copy: 400 images of 64x64 (paper: 92x112). Already scaled to [0, 1]."""
    d = fetch_olivetti_faces()
    return d.data.astype(float), d.target.astype(int)


def load_coil20():
    """COIL-20 processed set, 128x128 -> 32x32 by averaging each 4x4 block (= area / BOX resize)."""
    X, y = [], []
    for obj in range(1, 21):
        for angle in range(72):
            im = np.asarray(Image.open(os.path.join(COIL_DIR, "obj%d__%d.png" % (obj, angle))),
                            dtype=float)
            assert im.shape == (128, 128)
            small = im.reshape(32, 4, 32, 4).mean(axis=(1, 3))
            X.append(small.ravel() / 255.0)
            y.append(obj)
    return np.array(X), np.array(y)


def palette(n):
    """Enough distinct colours for n classes (scatter() indexes experiments.CMAP by label)."""
    cols = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) \
        + list(plt.get_cmap("tab20c").colors)
    return ListedColormap(cols[:n] if n > 10 else plt.get_cmap("tab10").colors)


DATASETS = [  # fast ones first, so problems show up early
    ("Olivetti", load_olivetti),
    ("COIL-20", load_coil20),
    ("MNIST", load_mnist),
]


# --------------------------------------------------------------------------- helpers
def largest_component(X30):
    """Symmetrised k=12 neighbour graph; the paper plots only its largest connected part."""
    A = kneighbors_graph(X30, K, mode="connectivity")
    A = A + A.T
    n_comp, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    keep = np.where(lab == np.argmax(sizes))[0]
    return n_comp, sorted(sizes.tolist(), reverse=True), keep


def safe(f, *a, **kw):
    try:
        return round(float(f(*a, **kw)), 4)
    except Exception as e:                       # e.g. too few points for 10-fold CV
        return "error: %s" % e


def main():
    results = {}
    t_all = time.time()
    for dname, loader in DATASETS:
        print("\n=== %s ===" % dname, flush=True)
        X, y = loader()
        X30 = pca(X, 30)
        n_cls = len(np.unique(y))
        experiments.CMAP = palette(int(y.max()) + 1)
        R = {"n_points": int(len(X)), "n_pixels": int(X.shape[1]), "n_classes": int(n_cls)}
        t = time.time()
        R["raw_pixels_1nn_error"] = safe(one_nn_error, X, y)
        print("  raw pixels 1-NN error: %s (%.0f s)" % (R["raw_pixels_1nn_error"], time.time() - t),
              flush=True)

        n_comp, sizes, keep = largest_component(X30)
        R["knn_graph_k12"] = {"connected": n_comp == 1, "n_components": int(n_comp),
                              "component_sizes": sizes, "points_in_largest": int(len(keep))}
        print("  k=12 graph: %d component(s), sizes %s" % (n_comp, sizes[:10]), flush=True)

        methods = [
            ("t-SNE", None, lambda: tsne(X, perplexity=40, verbose=False)),
            ("Sammon", None, lambda: sammon(X30, n_iter=500)),
            ("Isomap", keep, lambda: Isomap(n_neighbors=K).fit_transform(X30[keep])),
            ("LLE", keep, lambda: LocallyLinearEmbedding(n_neighbors=K, random_state=0)
             .fit_transform(X30[keep])),
        ]
        R["methods"] = {}
        for mname, idx, f in methods:
            print("  %s ..." % mname, flush=True)
            sub = np.arange(len(X)) if idx is None else idx
            M = {"n_points_in_map": int(len(sub))}
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                t = time.time()
                try:
                    out = f()
                except Exception as e:
                    M["error"] = repr(e)
                    R["methods"][mname] = M
                    print("    FAILED: %r" % e, flush=True)
                    continue
                M["seconds"] = round(time.time() - t, 1)
                if mname == "t-SNE":
                    Y, h = out
                    M["final_kl"] = round(float(h["kl"][-1]), 4)
                else:
                    Y = out
                M["one_nn_error_map"] = safe(one_nn_error, Y, y[sub])
                M["trustworthiness_k12"] = safe(trustworthiness, X30[sub], Y, n_neighbors=K)
                if idx is not None and len(sub) < len(X):
                    M["raw_pixels_1nn_error_same_subset"] = safe(one_nn_error, X[sub], y[sub])
                    M["n_classes_in_map"] = int(len(np.unique(y[sub])))
                M["warnings"] = sorted({str(x.message)[:200] for x in w})
            print("    %s" % {k: v for k, v in M.items() if k != "warnings"}, flush=True)
            R["methods"][mname] = M

            fig, ax = plt.subplots(figsize=(7, 7))
            scatter(ax, Y, y[sub], "%s on %s (ours)" % (mname, dname))
            fig.tight_layout()
            fn = "%s_%s.png" % (mname.replace("-", "").lower(), dname.replace("-", "").lower())
            fig.savefig(os.path.join(OUT, fn), dpi=140)
            plt.close(fig)
            M["figure"] = fn
        results[dname] = R
        json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    results["_total_seconds"] = round(time.time() - t_all, 1)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print("\nDONE in %.0f s" % (time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
