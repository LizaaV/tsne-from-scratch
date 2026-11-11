"""
Checks that the gradient in tsne.py (the paper's Equation 5, derived in Appendix A)
is the true derivative of the KL cost, by comparing it with finite differences:
    dC/dy_ik  ~  (C(Y + h e_ik) - C(Y - h e_ik)) / 2h
Also checks the perplexity search: every row of P_cond must reach the target perplexity.

Run:  python check_gradient.py
"""
import numpy as np
from tsne import joint_probabilities, conditional_probabilities, squared_distances, \
    q_and_kernel, kl_divergence, gradient

rng = np.random.RandomState(1)
X = rng.randn(40, 10)                 # 40 points in 10-D
P, _ = joint_probabilities(X, perplexity=10)
Y = 0.5 * rng.randn(40, 2)        # small map, so no q_ij underflows to the 1e-12 floor

for kernel in ["student", "gaussian"]:
    g, _ = gradient(P, Y, kernel)
    num = np.zeros_like(Y)
    h = 1e-6
    for i in range(Y.shape[0]):
        for k in range(Y.shape[1]):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[i, k] += h
            Ym[i, k] -= h
            num[i, k] = (kl_divergence(P, q_and_kernel(Yp, kernel)[0]) -
                         kl_divergence(P, q_and_kernel(Ym, kernel)[0])) / (2 * h)
    rel = np.linalg.norm(g - num) / np.linalg.norm(num)
    print("%-8s kernel: relative difference analytic vs numerical = %.2e  %s"
          % (kernel, rel, "OK" if rel < 1e-5 else "MISMATCH"))

P_cond, sigma = conditional_probabilities(squared_distances(X), perplexity=10)
H = -np.sum(np.where(P_cond > 0, P_cond * np.log2(np.maximum(P_cond, 1e-300)), 0), axis=1)
print("perplexity per row: min %.4f  max %.4f  (target 10)" % ((2 ** H).min(), (2 ** H).max()))
print("sum_j p_ij > 1/2n for every i:", bool(np.all(P.sum(axis=1) > 1 / (2 * len(X)))))
