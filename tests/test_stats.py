"""Firth regression, multiple-testing and McNemar checks against known closed forms."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stats import firth_fit, firth_test, holm, mcnemar_exact  # noqa: E402


def two_by_two(a, b, c, d):
    """x=1: a events, b non-events; x=0: c events, d non-events."""
    x = [1] * (a + b) + [0] * (c + d)
    y = [1] * a + [0] * b + [1] * c + [0] * d
    return np.column_stack([np.ones(len(x)), x]), np.array(y, dtype=float)


def test_firth_2x2_equals_haldane_corrected_log_odds_ratio():
    # for one binary covariate, Firth's estimate is the log odds ratio with 0.5 added to every cell
    for a, b, c, d in [(3, 2, 1, 4), (20, 0, 7, 13), (5, 0, 0, 5)]:
        beta = firth_fit(*two_by_two(a, b, c, d))[0]
        assert math.isclose(beta[1], math.log((a + .5) * (d + .5) / ((b + .5) * (c + .5))), abs_tol=1e-6)


def test_firth_finite_under_separation_and_consistent_test():
    X, y = two_by_two(10, 0, 0, 10)  # complete separation: ordinary ML diverges
    t = firth_test(X, y, 1)
    assert np.isfinite(t["coef"]) and t["ci_lo"] < t["coef"] < t["ci_hi"] and np.isfinite(t["ci_hi"])
    assert t["p"] < 0.05 and t["ci_lo"] > 0  # the LR test and the profile interval agree
    t0 = firth_test(*two_by_two(10, 10, 10, 10), 1)
    assert abs(t0["coef"]) < 1e-9 and t0["p"] > 0.99 and t0["ci_lo"] < 0 < t0["ci_hi"]


def test_firth_close_to_ml_in_large_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-(0.5 + 1.2 * x)))).astype(float)
    beta = firth_fit(np.column_stack([np.ones(5000), x]), y)[0]
    assert abs(beta[0] - 0.5) < 0.1 and abs(beta[1] - 1.2) < 0.1


def test_holm():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(holm([0.01, float("nan")])[1])


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(10, 0), 2 * 0.5 ** 10)
    assert mcnemar_exact(5, 5) == 1.0
