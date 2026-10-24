"""
Live demo for the presentation: our from-scratch t-SNE on a subset of the digits,
drawn while it optimises, so the audience sees early exaggeration and the clusters forming.

    python demo_live.py                    # perplexity 30, 600 digits
    python demo_live.py --perplexity 5     # many small fragments
    python demo_live.py --perplexity 50 --n 900
    python demo_live.py --gaussian         # symmetric SNE: watch the crowding problem

About 30 seconds for 600 points on a laptop. Close the window to exit.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from tsne import pca, joint_probabilities, gradient

ap = argparse.ArgumentParser()
ap.add_argument("--perplexity", type=float, default=30)
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--iters", type=int, default=500)
ap.add_argument("--gaussian", action="store_true", help="use a Gaussian map kernel (SNE) instead of Student-t")
args = ap.parse_args()

X, y = load_digits(return_X_y=True)
rng = np.random.RandomState(0)
idx = rng.choice(len(X), args.n, replace=False)
X, y = pca(X[idx] / 16.0, 30), y[idx]
kernel = "gaussian" if args.gaussian else "student"

P, _ = joint_probabilities(X, args.perplexity)
Y = rng.normal(0, 1e-2, (args.n, 2))
update, gains = np.zeros_like(Y), np.ones_like(Y)

plt.ion()
fig, ax = plt.subplots(figsize=(6.5, 6.5))
cmap = plt.get_cmap("tab10")
for t in range(args.iters):
    grad, Q = gradient(P * (4.0 if t < 50 else 1.0), Y, kernel)       # early exaggeration
    gains = np.maximum(np.where(np.sign(grad) == np.sign(update), gains * 0.8, gains + 0.2), 0.01)
    update = (0.5 if t < 250 else 0.8) * update - 100.0 * gains * grad  # gradient DESCENT
    Y = Y + update
    Y -= Y.mean(0)
    if t % 10 == 0 or t == args.iters - 1:
        ax.clear()
        ax.scatter(Y[:, 0], Y[:, 1], c=[cmap(c) for c in y], s=8)
        for c in range(10):
            m = np.median(Y[y == c], axis=0)
            ax.text(m[0], m[1], str(c), weight="bold", fontsize=12, ha="center")
        kl = np.sum(P * np.log(P / Q))
        ax.set_title("%s  perplexity %g  iteration %d%s  KL %.3f" % (
            "SNE (Gaussian)" if args.gaussian else "t-SNE", args.perplexity, t,
            "  [early exaggeration]" if t < 50 else "", kl))
        ax.set_xticks([]); ax.set_yticks([])
        plt.pause(0.001)
plt.ioff()
plt.show()
