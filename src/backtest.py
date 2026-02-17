import math

import numpy as np
from scipy.stats import chi2


def kupiec_pof_test(breaches: int, n: int, alpha: float) -> float:
    """
    Kupiec Proportion of Failures (POF) test.
    H0: breach probability == alpha
    LR_pof ~ Chi-square(1)
    """
    x = int(breaches)
    n = int(n)
    if n <= 0:
        return float("nan")
    # MLE under alternative
    phat = x / n
    # handle edge cases
    if phat <= 0 or phat >= 1:
        # If x==0 or x==n, LR can be computed with limits; return very small p-value if violates
        # Here we compute safely by nudging
        phat = min(max(phat, 1e-12), 1 - 1e-12)

    alpha = float(alpha)
    alpha = min(max(alpha, 1e-12), 1 - 1e-12)

    logL0 = (n - x) * math.log(1 - alpha) + x * math.log(alpha)
    logL1 = (n - x) * math.log(1 - phat) + x * math.log(phat)
    lr = -2 * (logL0 - logL1)
    return float(1 - chi2.cdf(lr, df=1))


def christoffersen_independence_test(breach_series) -> float:
    """
    Christoffersen independence test (for clustering of VaR breaches).
    Input: iterable of 0/1 (or False/True) for each day.
    H0: breaches are independent over time.
    LR_ind ~ Chi-square(1)
    """
    b = np.asarray(breach_series, dtype=int)
    if b.size < 2:
        return float("nan")

    b0 = b[:-1]
    b1 = b[1:]

    n00 = int(np.sum((b0 == 0) & (b1 == 0)))
    n01 = int(np.sum((b0 == 0) & (b1 == 1)))
    n10 = int(np.sum((b0 == 1) & (b1 == 0)))
    n11 = int(np.sum((b0 == 1) & (b1 == 1)))

    n0 = n00 + n01
    n1 = n10 + n11
    n = n0 + n1

    if n == 0:
        return float("nan")

    # probabilities
    pi = (n01 + n11) / n if n > 0 else 0.0
    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0

    # safe logs
    def slog(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return math.log(p)

    # log-likelihood under independence
    logL0 = (n00 + n10) * slog(1 - pi) + (n01 + n11) * slog(pi)
    # log-likelihood under 2-state Markov
    logL1 = (
        n00 * slog(1 - pi0) + n01 * slog(pi0) + n10 * slog(1 - pi1) + n11 * slog(pi1)
    )

    lr = -2 * (logL0 - logL1)
    return float(1 - chi2.cdf(lr, df=1))
