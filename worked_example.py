"""
Reproduces the hand-worked example in TSNE_01_Paper_Explained.md, Part 7:
three 1-D data points x = 0, 1, 3 with sigma = 1 for every point (instead of a perplexity
search), a 1-D map y = 0, 0.5, 1, and one plain gradient step with learning rate 1.

    python worked_example.py
"""
import numpy as np
from tsne import squared_distances, q_and_kernel, kl_divergence, gradient

x = np.array([[0.0], [1.0], [3.0]])
Y = np.array([[0.0], [0.5], [1.0]])
n = len(x)

D = squared_distances(x)
E = np.exp(-D / 2.0)                          # Equation 1 numerator with 2 sigma^2 = 2
np.fill_diagonal(E, 0.0)
P_cond = E / E.sum(axis=1, keepdims=True)     # row i = p_{.|i}
P = (P_cond + P_cond.T) / (2 * n)             # Section 3.1
np.fill_diagonal(P, 1e-12)

Q, W = q_and_kernel(Y)
print("p_{j|i} rows:\n", P_cond.round(4))
print("p_12, p_13, p_23 =", P[0, 1].round(4), P[0, 2].round(4), P[1, 2].round(4))
print("w_12, w_13, w_23 =", W[0, 1], W[0, 2], W[1, 2], "  Z =", W.sum())
print("q_12, q_13, q_23 =", Q[0, 1].round(4), Q[0, 2].round(4), Q[1, 2].round(4))
print("cost KL(P||Q) =", round(kl_divergence(P, Q), 4))
g, _ = gradient(P, Y)
print("gradient =", g.ravel().round(4), "  sum =", round(g.sum(), 10))
Y1 = Y - 1.0 * g
print("Y after one step =", Y1.ravel().round(4))
print("new cost =", round(kl_divergence(P, q_and_kernel(Y1)[0]), 4))
print("d_12: 0.5 ->", round(abs(Y1[0, 0] - Y1[1, 0]), 4), "   d_13: 1 ->", round(abs(Y1[0, 0] - Y1[2, 0]), 4))
