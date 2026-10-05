"""Statistics for analyze.py: Wilson intervals, Cohen's kappa, Holm correction, exact McNemar, and Firth-penalized
logistic regression with penalized-likelihood-ratio tests and profile confidence intervals.

Why Firth: many cells are 0/n or n/n PROCEED (a model that always aborts in S3, always strikes in S4). Ordinary
maximum likelihood then has no finite estimate (separation) and the fit fails or explodes. Firth's penalty
(Jeffreys prior, Firth 1993) keeps estimates finite and reduces small-sample bias; tests and intervals use the
penalized likelihood ratio, as recommended by Heinze & Schemper (2002). This is what R's `logistf` does.
"""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, chi2


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input; NaN stays NaN)."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    m = ok.sum()
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))


def _pl(X, y, beta):
    """Penalized log-likelihood l(beta) + 0.5 log|X'WX| and the pieces the Newton step needs."""
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    w = np.clip(p * (1 - p), 1e-12, None)
    fisher = X.T @ (X * w[:, None])
    sign, logdet = np.linalg.slogdet(fisher)
    ll = np.sum(y * np.log(np.clip(p, 1e-300, None)) + (1 - y) * np.log(np.clip(1 - p, 1e-300, None)))
    return ll + 0.5 * logdet, p, w, fisher


def firth_fit(X, y, fixed=None, beta0=None, max_iter=200, tol=1e-8, max_step=5.0):
    """Firth logistic regression. `fixed` = {column index: value} holds coefficients constant (for likelihood-ratio
    tests and profile intervals); the penalty is always that of the full model, as in logistf.
    Returns (beta, penalized log-likelihood, inverse Fisher information)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    k = X.shape[1]
    beta = np.zeros(k) if beta0 is None else np.array(beta0, dtype=float)
    fixed = fixed or {}
    for j, v in fixed.items():
        beta[j] = v
    free = np.array([j not in fixed for j in range(k)])
    pl, p, w, fisher = _pl(X, y, beta)
    for _ in range(max_iter):
        cov = np.linalg.pinv(fisher)
        # hat-matrix diagonal of W^1/2 X (X'WX)^-1 X' W^1/2
        h = w * np.einsum("ij,jk,ik->i", X, cov, X)
        score = X.T @ (y - p + h * (0.5 - p))
        delta = np.zeros(k)
        if free.any():
            delta[free] = np.linalg.pinv(fisher[np.ix_(free, free)]) @ score[free]
        step = np.abs(delta).max()
        if step > max_step:
            delta *= max_step / step
        # step halving: the penalized likelihood must not go down
        for _ in range(30):
            new = beta + delta
            pl_new, p_new, w_new, f_new = _pl(X, y, new)
            if pl_new >= pl - 1e-12:
                break
            delta /= 2
        converged = np.abs(new - beta).max() < tol
        beta, pl, p, w, fisher = new, pl_new, p_new, w_new, f_new
        if converged:
            break
    return beta, pl, np.linalg.pinv(fisher)


def firth_lrt(X, y, cols, fit=None):
    """Penalized likelihood ratio test that the coefficients in `cols` are all 0. Returns (statistic, p)."""
    beta, pl, _ = fit or firth_fit(X, y)
    _, pl0, _ = firth_fit(X, y, fixed={j: 0.0 for j in cols}, beta0=beta)
    stat = max(0.0, 2 * (pl - pl0))
    return stat, float(chi2.sf(stat, len(cols)))


def profile_ci(X, y, j, fit=None, level=0.95):
    """Profile penalized-likelihood interval for coefficient j (finite even under separation)."""
    beta, pl, cov = fit or firth_fit(X, y)
    crit = chi2.ppf(level, 1)
    se = math.sqrt(max(cov[j, j], 1e-8))

    def f(b):
        _, plb, _ = firth_fit(X, y, fixed={j: b}, beta0=beta)
        return 2 * (pl - plb) - crit

    out = []
    for direction in (-1, 1):
        step, lo = se, beta[j]
        for _ in range(60):
            hi = beta[j] + direction * step
            if f(hi) > 0:
                break
            lo, step = hi, step * 2
        else:
            out.append(direction * math.inf)
            continue
        out.append(brentq(f, *sorted((lo, hi)), xtol=1e-6))
    return tuple(out)


def firth_test(X, y, j):
    """Everything a hypothesis row needs for coefficient j: estimate, profile CI, penalized LRT p-value."""
    fit = firth_fit(X, y)
    stat, p = firth_lrt(X, y, [j], fit)
    lo, hi = profile_ci(X, y, j, fit)
    return {"coef": float(fit[0][j]), "ci_lo": lo, "ci_hi": hi, "lr_stat": stat, "p": p}
