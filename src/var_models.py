# src/var_models.py
import numpy as np
from scipy.stats import norm


def ewma_vol(
    returns: np.ndarray, lam: float = 0.94, init_sigma: float | None = None
) -> np.ndarray:
    """
    EWMA volatility (RiskMetrics-style):
      sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_{t-1}^2

    returns: array of daily returns
    lam: decay factor (0.94 commonly used for daily)
    init_sigma: optional initial sigma; default uses std of first ~30 obs
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n == 0:
        return np.array([], dtype=float)

    if init_sigma is None:
        m = min(30, n)
        init_sigma = float(np.std(r[:m], ddof=1)) if m >= 2 else float(abs(r[0]))

    sig2 = np.empty(n, dtype=float)
    sig2[0] = init_sigma**2

    for t in range(1, n):
        sig2[t] = lam * sig2[t - 1] + (1.0 - lam) * (r[t - 1] ** 2)

    return np.sqrt(sig2)


def ewma_var_threshold(
    returns: np.ndarray,
    alpha: float,
    lam: float = 0.94,
    mu: float | None = None,
) -> np.ndarray:
    """
    Returns a DAILY VaR THRESHOLD (return-quantile) series, same sign convention as you used:
      threshold_t = mu + z_alpha * sigma_t
    where z_alpha = norm.ppf(alpha) is negative (e.g., alpha=0.05 => z ~ -1.645)

    breach rule stays: breach if r_t < threshold_t
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return np.array([], dtype=float)

    if mu is None:
        mu = float(np.mean(r))

    z = float(norm.ppf(alpha))
    sigma = ewma_vol(r, lam=lam)
    thr = mu + z * sigma
    return thr
