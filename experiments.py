"""
All experiments of our partial reproduction of van der Maaten & Hinton (2008).

    python experiments.py all          # everything (about 15 minutes on a laptop)
    python experiments.py compare      # E1  Figures 2-3 analogue: t-SNE vs Sammon, Isomap, LLE, PCA
    python experiments.py gradients    # E2  Figure 1: the three gradients
    python experiments.py crowding     # E3  symmetric SNE (Gaussian) vs t-SNE (Student-t), same optimiser
    python experiments.py perplexity   # E4  perplexity 5 / 15 / 40 / 100
    python experiments.py optimisation # E5  snapshots + KL curve + GIF; early exaggeration on/off
    python experiments.py seeds        # E6  run-to-run variation (the paper's claim in Section 6.2)
    python experiments.py distances    # E7  our extension: what the map does NOT preserve
    python experiments.py mnist        # E8  the paper's own MNIST-6000 setting (downloads MNIST)

Data: scikit-learn's bundled "digits" (1,797 images of 8x8 = 64 pixels, 10 classes), so no
download is needed. It is a small cousin of the paper's MNIST (28x28 = 784 pixels).
Figures go to figures/, numbers to results/.
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE as SkTSNE, Isomap, LocallyLinearEmbedding, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from tsne import tsne, sammon, pca

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
CMAP = plt.get_cmap("tab10")


def load():
    X, y = load_digits(return_X_y=True)
    return X / 16.0, y                      # pixel values 0..16 -> 0..1


def one_nn_error(Y, y):
    """Generalisation error of a 1-nearest-neighbour classifier, 10-fold cross-validation
    (the paper's Section 5 measure; Method 14 in the COMP3314 L02b note)."""
    return 1.0 - cross_val_score(KNeighborsClassifier(1), Y, y, cv=10).mean()


def scatter(ax, Y, y, title):
    """Class colours only colour the points; they are never used to place them (Section 4.2)."""
    ax.scatter(Y[:, 0], Y[:, 1], c=[CMAP(int(c)) for c in y], s=4, linewidths=0)
    for c in np.unique(y):                  # write the digit on its cluster instead of a legend
        m = np.median(Y[y == c], axis=0)
        ax.text(m[0], m[1], str(c), fontsize=11, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=160)
    plt.close(fig)
    print("  wrote figures/" + name)


# ---------------------------------------------------------------------------
def e1_compare():
    """Figures 2-3 analogue. Paper settings (Table 1): t-SNE perplexity 40; Isomap and LLE k = 12;
    Sammon 500 Newton iterations; every method gets the same PCA-30 input (Section 4.2)."""
    print("E1 compare")
    X, y = load()
    X30 = pca(X, 30)
    out, maps = {}, {}

    def run(name, f):
        t = time.time(); Y = f(); dt = time.time() - t
        maps[name] = Y
        out[name] = {"seconds": round(dt, 1), "one_nn_error": round(one_nn_error(Y, y), 4),
                     "trustworthiness_k12": round(trustworthiness(X30, Y, n_neighbors=12), 4)}
        print("  %-22s %s" % (name, out[name]))

    run("t-SNE (ours)", lambda: tsne(X, perplexity=40, verbose=False)[0])
    run("t-SNE (scikit-learn)", lambda: SkTSNE(perplexity=40, init="random", learning_rate=100,
                                               random_state=0).fit_transform(X30))
    run("Sammon (ours)", lambda: sammon(X30, n_iter=500))
    run("Isomap k=12", lambda: Isomap(n_neighbors=12).fit_transform(X30))
    run("LLE k=12", lambda: LocallyLinearEmbedding(n_neighbors=12, random_state=0).fit_transform(X30))
    run("PCA", lambda: pca(X30, 2))
    out["original 64-D pixels"] = {"one_nn_error": round(one_nn_error(X, y), 4)}
    print("  original:", out["original 64-D pixels"])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (name, Y) in zip(axes.ravel(), maps.items()):
        scatter(ax, Y, y, "%s\n1-NN error %.1f%%" % (name, 100 * out[name]["one_nn_error"]))
    save(fig, "E1_compare_methods.png")
    np.save(os.path.join(RES, "tsne_ours_digits.npy"), maps["t-SNE (ours)"])
    json.dump(out, open(os.path.join(RES, "E1_compare.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
def e2_gradients():
    """Figure 1: gradient between two points as a function of their high-D distance (x axis)
    and low-D distance (y axis). Positive = attraction, negative = repulsion.
    The paper does not give its normalisation; we use unnormalised unit-width affinities
      p = exp(-r_x^2), q_SNE = exp(-r_y^2), q_UNI = (1-rho) exp(-r_y^2) + rho, q_t = 1/(1+r_y^2)
    and force = 2 (p - q) r_y [x (1 + r_y^2)^-1 for t-SNE]. rho = 0.2 is our choice.
    The shapes, not the exact numbers, are what we compare with the paper."""
    print("E2 gradients")
    rx = np.linspace(0, 5, 300)[None, :]
    ry = np.linspace(0, 10, 300)[:, None]
    p = np.exp(-rx ** 2)
    g_sne = 2 * (p - np.exp(-ry ** 2)) * ry
    rho = 0.2
    g_uni = 2 * (p - ((1 - rho) * np.exp(-ry ** 2) + rho)) * ry
    g_t = 2 * (p - 1 / (1 + ry ** 2)) * ry / (1 + ry ** 2)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, g, t in zip(axes, [g_sne, g_uni, g_t], ["(a) SNE", "(b) UNI-SNE", "(c) t-SNE"]):
        lim = np.abs(g).max()
        im = ax.imshow(g, origin="lower", extent=[0, 5, 0, 10], aspect="auto", cmap="RdBu_r",
                       vmin=-lim, vmax=lim)
        ax.contour(rx.ravel(), ry.ravel(), g, levels=[0], colors="k", linewidths=1)
        ax.set_xlabel("high-dimensional distance  ||x_i - x_j||")
        ax.set_ylabel("low-dimensional distance  ||y_i - y_j||")
        ax.set_title("%s   range %.2f to %.2f" % (t, g.min(), g.max()))
        fig.colorbar(im, ax=ax, label="gradient (+ attract, - repel)")
    save(fig, "E2_gradients_fig1.png")
    json.dump({"SNE": [float(g_sne.min()), float(g_sne.max())],
               "UNI-SNE": [float(g_uni.min()), float(g_uni.max())],
               "t-SNE": [float(g_t.min()), float(g_t.max())]},
              open(os.path.join(RES, "E2_gradient_ranges.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
def e3_crowding():
    """Section 3.2: the crowding problem. Same P, same optimiser, only the low-D kernel changes."""
    print("E3 crowding")
    X, y = load()
    Y_sne, h_sne = tsne(X, perplexity=40, kernel="gaussian", verbose=False)
    Y_t = np.load(os.path.join(RES, "tsne_ours_digits.npy")) if os.path.exists(
        os.path.join(RES, "tsne_ours_digits.npy")) else tsne(X, perplexity=40, verbose=False)[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    scatter(axes[0], Y_sne, y, "symmetric SNE (Gaussian in the map)\n1-NN error %.1f%%"
            % (100 * one_nn_error(Y_sne, y)))
    scatter(axes[1], Y_t, y, "t-SNE (Student-t in the map)\n1-NN error %.1f%%"
            % (100 * one_nn_error(Y_t, y)))
    save(fig, "E3_crowding_sne_vs_tsne.png")


# ---------------------------------------------------------------------------
def e4_perplexity():
    print("E4 perplexity")
    X, y = load()
    perps = [5, 15, 40, 100]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    out = {}
    for ax, perp in zip(axes, perps):
        Y, h = tsne(X, perplexity=perp, verbose=False)
        out[perp] = {"one_nn_error": round(one_nn_error(Y, y), 4), "final_kl": round(h["kl"][-1], 4),
                     "median_sigma": round(float(np.median(h["sigma"])), 4)}
        scatter(ax, Y, y, "perplexity %d\n1-NN error %.1f%%, KL %.2f"
                % (perp, 100 * out[perp]["one_nn_error"], out[perp]["final_kl"]))
        print("  perplexity", perp, out[perp])
    save(fig, "E4_perplexity.png")
    json.dump(out, open(os.path.join(RES, "E4_perplexity.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
def e5_optimisation():
    """Section 3.4: early exaggeration. Snapshots, KL curve, a GIF, and an ablation without it."""
    print("E5 optimisation")
    X, y = load()
    Y, h = tsne(X, perplexity=40, verbose=False, snapshot_every=10)
    Y0, h0 = tsne(X, perplexity=40, verbose=False, early_exaggeration=1.0)

    show = [0, 30, 50, 100, 250, 990]
    snaps = dict(h["snapshots"])
    fig, axes = plt.subplots(1, len(show), figsize=(18, 3.4))
    for ax, t in zip(axes, show):
        scatter(ax, snaps[t], y, "iteration %d%s" % (t, " (exaggeration)" if t < 50 else ""))
    save(fig, "E5_snapshots.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(h["iter"], h["kl"], color="k")
    ax.plot(h0["iter"], h0["kl"], color="tab:red", ls="--")
    ax.text(600, h["kl"][-1] - 0.12, "with early exaggeration (final %.3f)" % h["kl"][-1], fontsize=9)
    ax.text(600, h0["kl"][-1] + 0.25, "without (final %.3f)" % h0["kl"][-1], fontsize=9, color="tab:red")
    ax.axvline(50, color="grey", lw=0.8); ax.axvline(250, color="grey", lw=0.8)
    ax.text(52, max(h["kl"]) * 0.95, "exaggeration ends", fontsize=8, color="grey")
    ax.text(252, max(h["kl"]) * 0.85, "momentum 0.5 -> 0.8", fontsize=8, color="grey")
    ax.set_xlabel("iteration t"); ax.set_ylabel("cost KL(P || Q) (nats)")
    ax.set_ylim(min(h["kl"]) - 0.3, max(h["kl"]) * 1.05)
    save(fig, "E5_kl_curve.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    scatter(axes[0], Y, y, "with early exaggeration\n1-NN %.1f%%, KL %.3f" % (100 * one_nn_error(Y, y), h["kl"][-1]))
    scatter(axes[1], Y0, y, "without early exaggeration\n1-NN %.1f%%, KL %.3f" % (100 * one_nn_error(Y0, y), h0["kl"][-1]))
    save(fig, "E5_exaggeration_ablation.png")
    json.dump({"with": h["kl"][-1], "without": h0["kl"][-1]},
              open(os.path.join(RES, "E5_exaggeration.json"), "w"), indent=2)

    try:                                     # animated GIF for the presentation
        from matplotlib.animation import FuncAnimation, PillowWriter
        frames = [s for s in h["snapshots"] if s[0] <= 400 or s[0] % 50 == 0]
        fig, ax = plt.subplots(figsize=(5, 5))
        def draw(k):
            ax.clear(); t, Yt = frames[k]
            scatter(ax, Yt, y, "t-SNE on digits, iteration %d%s" % (t, "  (early exaggeration)" if t < 50 else ""))
        anim = FuncAnimation(fig, draw, frames=len(frames))
        anim.save(os.path.join(FIG, "E5_tsne_animation.gif"), writer=PillowWriter(fps=8))
        plt.close(fig)
        print("  wrote figures/E5_tsne_animation.gif")
    except Exception as e:                   # Pillow missing: skip the GIF only
        print("  GIF skipped:", e)


# ---------------------------------------------------------------------------
def e6_seeds():
    """Section 6.2 claims "the quality of the optima does not vary much from run to run"."""
    print("E6 seeds")
    X, y = load()
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for k, seed in enumerate([0, 1, 2, 3, 4]):
        Y, h = tsne(X, perplexity=40, seed=seed, verbose=False)
        out[seed] = {"final_kl": round(h["kl"][-1], 4), "one_nn_error": round(one_nn_error(Y, y), 4)}
        print("  seed", seed, out[seed])
        if k < 3:
            scatter(axes[k], Y, y, "seed %d: KL %.3f, 1-NN %.1f%%" % (seed, h["kl"][-1], 100 * out[seed]["one_nn_error"]))
    save(fig, "E6_seeds.png")
    kl = [v["final_kl"] for v in out.values()]; er = [v["one_nn_error"] for v in out.values()]
    out["summary"] = {"kl_mean": round(float(np.mean(kl)), 4), "kl_std": round(float(np.std(kl)), 4),
                      "err_mean": round(float(np.mean(er)), 4), "err_std": round(float(np.std(er)), 4)}
    print("  summary", out["summary"])
    json.dump(out, open(os.path.join(RES, "E6_seeds.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
def e7_distances():
    """Our extension (critical analysis): t-SNE keeps neighbours, not sizes or gaps.
    Three Gaussian clusters in 10-D: A and B are close (centres 10 apart) and C is far (50 away);
    A is tight (std 1) and B, C are wide (std 5). Does the map show this?"""
    print("E7 distances")
    rng = np.random.RandomState(0)
    n = 200
    cA = np.zeros(10); cB = np.zeros(10); cB[0] = 10; cC = np.zeros(10); cC[0] = 60
    X = np.r_[cA + 1 * rng.randn(n, 10), cB + 5 * rng.randn(n, 10), cC + 5 * rng.randn(n, 10)]
    y = np.repeat([0, 1, 2], n)
    Yt = tsne(X, perplexity=30, verbose=False, pca_dims=None)[0]
    Yp = pca(X, 2)
    def spread(Y, c): return float(np.mean(np.linalg.norm(Y[y == c] - Y[y == c].mean(0), axis=1)))
    def gap(Y, a, b): return float(np.linalg.norm(Y[y == a].mean(0) - Y[y == b].mean(0)))
    out = {}
    for name, Y in [("input 10-D", X), ("PCA", Yp), ("t-SNE", Yt)]:
        out[name] = {"spread A": spread(Y, 0), "spread B": spread(Y, 1), "spread C": spread(Y, 2),
                     "gap A-B": gap(Y, 0, 1), "gap B-C": gap(Y, 1, 2)}
        out[name]["ratio spread B/A"] = out[name]["spread B"] / out[name]["spread A"]
        out[name]["ratio gap BC/AB"] = out[name]["gap B-C"] / out[name]["gap A-B"]
        print("  %-10s spread B/A = %.2f   gap BC/AB = %.2f" % (name, out[name]["ratio spread B/A"], out[name]["ratio gap BC/AB"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, (name, Y) in zip(axes, [("PCA", Yp), ("t-SNE", Yt)]):
        for c, lab in enumerate("ABC"):
            ax.scatter(Y[y == c, 0], Y[y == c, 1], s=5, color=CMAP(c))
            m = Y[y == c].mean(0); ax.text(m[0], m[1], lab, weight="bold", fontsize=12, ha="center")
        ax.set_title("%s: spread B/A = %.1f (true %.1f), gap BC/AB = %.1f (true %.1f)" % (
            name, out[name]["ratio spread B/A"], out["input 10-D"]["ratio spread B/A"],
            out[name]["ratio gap BC/AB"], out["input 10-D"]["ratio gap BC/AB"]), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "E7_sizes_and_gaps.png")
    json.dump(out, open(os.path.join(RES, "E7_distances.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
def e8_mnist(n=6000):
    """The paper's own setting: 6,000 random MNIST digits, PCA-30, perplexity 40.
    Needs an internet connection the first time (OpenML download, about 15 MB), or set
    MNIST_NPZ=/path/to/mnist.npz to use a local copy. Roughly
    30-60 minutes and 2 GB of memory with this O(n^2) implementation. Good job for the GPU farm."""
    print("E8 MNIST-6000")
    path = os.environ.get("MNIST_NPZ")      # optional local copy (Keras mnist.npz layout)
    if path:
        d = np.load(path)
        X = np.r_[d["x_train"], d["x_test"]].reshape(-1, 784).astype(float)
        y = np.r_[d["y_train"], d["y_test"]]
    else:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), n, replace=False)
    X, y = X[idx] / 255.0, y[idx].astype(int)
    t = time.time(); Y, h = tsne(X, perplexity=40, verbose=True); dt = time.time() - t
    out = {"seconds": dt, "final_kl": h["kl"][-1], "one_nn_error_map": one_nn_error(Y, y),
           "one_nn_error_784D": one_nn_error(X, y)}
    print(out)
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter(ax, Y, y, "t-SNE (ours) on 6,000 MNIST digits — compare paper Figure 2(a)")
    save(fig, "E8_mnist6000.png")
    json.dump(out, open(os.path.join(RES, "E8_mnist.json"), "w"), indent=2)


if __name__ == "__main__":
    jobs = {"compare": e1_compare, "gradients": e2_gradients, "crowding": e3_crowding,
            "perplexity": e4_perplexity, "optimisation": e5_optimisation, "seeds": e6_seeds,
            "distances": e7_distances, "mnist": e8_mnist}
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for k in ["compare", "gradients", "crowding", "perplexity", "optimisation", "seeds", "distances"]:
            jobs[k]()
    else:
        jobs[which]()
