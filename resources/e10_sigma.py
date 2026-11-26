"""E10: does sigma_i keep the cluster widths that the t-SNE map drops? (critical analysis, Extension 3)

Same three clusters, seed and perplexity as E7 in experiments.py. Run from resources/ : python e10_sigma.py
"""
import json
import os
import numpy as np
from tsne import joint_probabilities

n = 200
rng = np.random.RandomState(0)
cA = np.zeros(10); cB = np.zeros(10); cB[0] = 10; cC = np.zeros(10); cC[0] = 60
X = np.r_[cA + 1 * rng.randn(n, 10), cB + 5 * rng.randn(n, 10), cC + 5 * rng.randn(n, 10)]
y = np.repeat([0, 1, 2], n)


def width(c):
    return float(np.mean(np.linalg.norm(X[y == c] - X[y == c].mean(0), axis=1)))


P, sigma = joint_probabilities(X, perplexity=30)
med = [float(np.median(sigma[y == c])) for c in range(3)]
out = {
    "median sigma A, B, C": med,
    "sigma ratio B/A": med[1] / med[0], "true width ratio B/A": width(1) / width(0),
    "sigma ratio C/A": med[2] / med[0], "true width ratio C/A": width(2) / width(0),
    "mass of P on pairs joining B and C": float(2 * P[np.ix_(y == 1, y == 2)].sum()),
}
print(json.dumps(out, indent=2))
os.makedirs("results", exist_ok=True)
json.dump(out, open(os.path.join("results", "E10_sigma.json"), "w"), indent=2)
