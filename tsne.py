"""
t-SNE written from scratch in NumPy, following van der Maaten & Hinton (2008),
"Visualizing Data using t-SNE", JMLR 9:2579-2605.

Every function points at the equation or section of the paper it implements.
Nothing here is copied from scikit-learn or the authors' Matlab code; the
optimiser settings are the ones the paper reports in Section 3.4.

Shapes: n = number of datapoints, D = input dimension, d = map dimension (2).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Step 0 (Section 4.2): PCA to 30 dimensions before t-SNE
# ---------------------------------------------------------------------------
def pca(X, n_components=30):
    """Project X (n, D) onto its top principal components -> (n, n_components).
    Same method as the COMP3314 Lecture 3 note: centre, covariance, eigenvectors."""
    Xc = X - X.mean(axis=0)                      # (n, D) centred
    if n_components >= X.shape[1]:
        return Xc
    # SVD of the centred data gives the eigenvectors of the covariance matrix
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T              # (n, n_components)


def squared_distances(X):
    """All pairwise squared Euclidean distances ||x_i - x_j||^2 -> (n, n)."""
    sq = np.sum(X ** 2, axis=1)                  # (n,)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T  # (n, n)
    np.maximum(D, 0, out=D)                      # remove tiny negative round-off
    np.fill_diagonal(D, 0)
    return D


# ---------------------------------------------------------------------------
# Step 1 (Section 2, Equation 1 + perplexity): conditional probabilities p_{j|i}
# ---------------------------------------------------------------------------
def conditional_probabilities(D, perplexity=30.0, tol=1e-5, max_iter=50):
    """For every point i, binary-search the Gaussian precision beta_i = 1 / (2 sigma_i^2)
    so that the row P_i has perplexity 2^{H(P_i)} equal to `perplexity`.

    D : (n, n) squared distances in the high-dimensional space.
    Returns P_cond (n, n) with row i = p_{.|i} (rows sum to 1, p_{i|i} = 0)
    and sigma (n,) the bandwidth found for each point.
    """
    n = D.shape[0]
    target = np.log(perplexity)          # we compare entropies in nats: H_nats = ln(Perp)
    P = np.zeros((n, n))
    beta = np.ones(n)
    for i in range(n):
        lo, hi = -np.inf, np.inf
        Di = np.delete(D[i], i)          # (n-1,) distances to the other points
        for _ in range(max_iter):
            # Equation 1, numerator exp(-||x_i - x_j||^2 / 2 sigma_i^2); shift by min for stability
            e = np.exp(-(Di - Di.min()) * beta[i])
            sum_e = e.sum()
            p = e / sum_e
            # Shannon entropy in nats: H = -sum p ln p = ln(sum e) + beta * sum(p * D) (shifted)
            H = np.log(sum_e) + beta[i] * np.sum(p * (Di - Di.min()))
            diff = H - target
            if abs(diff) < tol:
                break
            if diff > 0:                 # entropy too high -> Gaussian too wide -> raise beta
                lo = beta[i]
                beta[i] = beta[i] * 2 if hi == np.inf else (beta[i] + hi) / 2
            else:                        # entropy too low -> Gaussian too narrow -> lower beta
                hi = beta[i]
                beta[i] = beta[i] / 2 if lo == -np.inf else (beta[i] + lo) / 2
        P[i, np.arange(n) != i] = p
    sigma = np.sqrt(1.0 / (2.0 * beta))
    return P, sigma


def joint_probabilities(X, perplexity=30.0):
    """Section 3.1: p_ij = (p_{j|i} + p_{i|j}) / 2n. Returns P (n, n), symmetric, sums to 1."""
    D = squared_distances(X)
    P_cond, sigma = conditional_probabilities(D, perplexity)
    n = X.shape[0]
    P = (P_cond + P_cond.T) / (2.0 * n)
    return np.maximum(P, 1e-12), sigma


# ---------------------------------------------------------------------------
# Step 2 (Equation 4 / Equation 3): low-dimensional joint probabilities q_ij
# ---------------------------------------------------------------------------
def q_and_kernel(Y, kernel="student"):
    """Returns Q (n, n) and the unnormalised kernel values W (n, n).
    kernel="student": w_ij = (1 + ||y_i - y_j||^2)^-1   (Equation 4, t-SNE)
    kernel="gaussian": w_ij = exp(-||y_i - y_j||^2)     (Equation 3, symmetric SNE)"""
    DY = squared_distances(Y)
    W = 1.0 / (1.0 + DY) if kernel == "student" else np.exp(-DY)
    np.fill_diagonal(W, 0.0)             # q_ii = 0
    Q = W / W.sum()                      # normalise over all pairs k != l
    return np.maximum(Q, 1e-12), W


def kl_divergence(P, Q):
    """Cost C = KL(P || Q) = sum_ij p_ij log(p_ij / q_ij)   (Section 3.1 / Equation 6)."""
    return float(np.sum(P * np.log(P / Q)))


# ---------------------------------------------------------------------------
# Step 3 (Equation 5): the gradient
# ---------------------------------------------------------------------------
def gradient(P, Y, kernel="student"):
    """t-SNE (Equation 5):     dC/dy_i = 4 sum_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i - y_j||^2)^-1
    symmetric SNE (Sec. 3.1): dC/dy_i = 4 sum_j (p_ij - q_ij)(y_i - y_j)
    Vectorised: with M_ij = (p_ij - q_ij) * extra_ij, sum_j M_ij (y_i - y_j) = (diag(M 1) - M) Y."""
    Q, W = q_and_kernel(Y, kernel)
    M = (P - Q) * (W if kernel == "student" else 1.0)   # (n, n)
    np.fill_diagonal(M, 0.0)
    grad = 4.0 * (M.sum(axis=1)[:, None] * Y - M @ Y)  # (n, d)
    return grad, Q


# ---------------------------------------------------------------------------
# Step 4 (Algorithm 1 + Section 3.4): the optimiser
# ---------------------------------------------------------------------------
def tsne(X, n_components=2, perplexity=40.0, n_iter=1000, learning_rate=100.0,
         early_exaggeration=4.0, exaggeration_iters=50, momentum_switch=250,
         initial_momentum=0.5, final_momentum=0.8, pca_dims=30, kernel="student",
         seed=0, snapshot_every=0, verbose=True):
    """Run t-SNE with the settings of Section 3.4:
      - PCA to 30 dimensions first (Section 4.2)
      - Y(0) ~ N(0, 1e-4 I)                                   (Algorithm 1)
      - early exaggeration: P multiplied by 4 for the first 50 iterations
      - momentum 0.5 for t < 250, 0.8 afterwards
      - learning rate 100 with Jacobs (1988) adaptive gains
    Gradient DESCENT: Y(t) = Y(t-1) - eta * dC/dY + alpha(t) (Y(t-1) - Y(t-2)).
    (Algorithm 1 in the paper prints "+ eta dC/dY"; that is a sign typo.)

    Returns Y (n, n_components) and a history dict (KL cost, snapshots, sigmas).
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)
    if pca_dims is not None and X.shape[1] > pca_dims:
        X = pca(X, pca_dims)
    n = X.shape[0]

    P, sigma = joint_probabilities(X, perplexity)
    Y = rng.normal(0.0, 1e-2, size=(n, n_components))   # std 1e-2 -> variance 1e-4
    update = np.zeros_like(Y)                             # Y(t-1) - Y(t-2)
    gains = np.ones_like(Y)
    history = {"iter": [], "kl": [], "snapshots": [], "sigma": sigma}

    for t in range(n_iter):
        exaggerate = t < exaggeration_iters
        P_used = P * early_exaggeration if exaggerate else P
        grad, Q = gradient(P_used, Y, kernel)

        # Jacobs (1988) "delta-bar-delta": grow the step where the gradient keeps its sign
        same_sign = np.sign(grad) == np.sign(update)
        gains = np.where(same_sign, gains * 0.8, gains + 0.2)
        gains = np.maximum(gains, 0.01)

        alpha = initial_momentum if t < momentum_switch else final_momentum
        update = alpha * update - learning_rate * gains * grad
        Y = Y + update
        Y = Y - Y.mean(axis=0)                            # centring does not change the cost

        if t % 50 == 0 or t == n_iter - 1:
            cost = kl_divergence(P, Q)                    # always report the un-exaggerated cost
            history["iter"].append(t)
            history["kl"].append(cost)
            if verbose:
                print("iter %4d  KL(P||Q) = %.4f%s" % (t, cost, "  (exaggeration on)" if exaggerate else ""))
        if snapshot_every and t % snapshot_every == 0:
            history["snapshots"].append((t, Y.copy()))
    return Y, history


# ---------------------------------------------------------------------------
# Baseline from the paper's comparison (Section 6.1): Sammon mapping
# ---------------------------------------------------------------------------
def sammon(X, n_components=2, n_iter=500, seed=0, init=None, verbose=False):
    """Sammon (1969) mapping, the cost on p. 2595 of the paper:
        E = 1 / sum_{i<j} d*_ij * sum_{i<j} (d*_ij - d_ij)^2 / d*_ij
    minimised with Sammon's diagonal Newton step and step halving.
    The paper used 500 Newton iterations (Section 4.2)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    Dstar = np.sqrt(squared_distances(X))
    np.fill_diagonal(Dstar, 1.0)
    Dstar = np.maximum(Dstar, 1e-9)
    scale = 0.5 * Dstar[np.triu_indices(n, 1)].sum() * 2  # sum over i != j  (both triangles)
    Dinv = 1.0 / Dstar
    np.fill_diagonal(Dinv, 0.0)
    rng = np.random.RandomState(seed)
    Y = pca(X, n_components) if init is None else init.copy()
    Y = Y + 1e-6 * rng.randn(*Y.shape)

    def stress(Y):
        d = np.sqrt(squared_distances(Y)); np.fill_diagonal(d, 1.0)
        diff = Dstar - d; np.fill_diagonal(diff, 0.0)
        return np.sum(diff ** 2 * Dinv) / scale, d

    E, d = stress(Y)
    for it in range(n_iter):
        dinv = 1.0 / np.maximum(d, 1e-9); np.fill_diagonal(dinv, 0.0)
        delta = dinv - Dinv                                  # (n, n)
        deltaone = delta.sum(axis=1)                         # (n,)
        g = delta @ Y - Y * deltaone[:, None]                # first derivative (up to constant)
        dinv3 = dinv ** 3
        y2 = Y ** 2
        H = dinv3 @ y2 - deltaone[:, None] - 2 * Y * (dinv3 @ Y) + y2 * dinv3.sum(axis=1)[:, None]
        step = -g.ravel() / np.abs(H.ravel())                # diagonal Newton step
        step = step.reshape(Y.shape)
        for _ in range(20):                                  # step halving until the stress falls
            Ynew = Y + step
            Enew, dnew = stress(Ynew)
            if Enew < E:
                break
            step *= 0.5
        else:
            break
        Y, E, d = Ynew, Enew, dnew
        if verbose and it % 50 == 0:
            print("sammon iter %d  E = %.5f" % (it, E))
    return Y
