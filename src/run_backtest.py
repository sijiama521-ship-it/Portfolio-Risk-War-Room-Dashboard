import os
import math
import pandas as pd
import numpy as np
from scipy.stats import chi2
from src.backtest import kupiec_pof_test, christoffersen_independence_test



def kupiec_pof_test(breaches: int, n: int, alpha: float) -> float:
    """Kupiec POF test p-value."""
    if n <= 0:
        return float("nan")

    x = int(breaches)
    eps = 1e-12
    phat = max(min(x / n, 1 - eps), eps)
    p = max(min(alpha, 1 - eps), eps)

    lr = -2.0 * (
        (n - x) * math.log((1 - p) / (1 - phat))
        + x * math.log(p / phat)
    )
    return 1 - chi2.cdf(lr, df=1)
def christoffersen_independence_test(breach_series: np.ndarray) -> float:
    """
    Christoffersen (1998) Independence test for VaR exceptions.
    H0: breaches are independent over time.
    Returns: p-value (higher is better; >0.05 usually pass)
    """
    b = np.asarray(breach_series).astype(int)
    if b.size < 2:
        return float("nan")

    # Transition counts
    n00 = np.sum((b[:-1] == 0) & (b[1:] == 0))
    n01 = np.sum((b[:-1] == 0) & (b[1:] == 1))
    n10 = np.sum((b[:-1] == 1) & (b[1:] == 0))
    n11 = np.sum((b[:-1] == 1) & (b[1:] == 1))

    n0 = n00 + n01
    n1 = n10 + n11
    if (n0 == 0) or (n1 == 0):
        # Can't estimate both transition probs -> treat as nan
        return float("nan")

    p01 = n01 / n0
    p11 = n11 / n1
    p = (n01 + n11) / (n0 + n1)

    eps = 1e-12
    p01 = min(max(p01, eps), 1 - eps)
    p11 = min(max(p11, eps), 1 - eps)
    p = min(max(p, eps), 1 - eps)

    ll_ind = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    ll_dep = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)

    lr = -2.0 * (ll_ind - ll_dep)
    pval = 1.0 - chi2.cdf(lr, df=1)
    return float(pval)


def main():
    # 1) load VaR thresholds summary (from your earlier step)
    backtest_path = "data/var_backtest_summary.csv"
    bt = pd.read_csv(backtest_path)

    # 2) load realized portfolio returns
    ret_path = "data/portfolio_returns.csv"
    rets = pd.read_csv(ret_path)
    r = rets["portfolio_return"].astype(float).values
    n = len(r)

    rows = []
    for _, row in bt.iterrows():
        raw_conf = row["confidence"]

        # parse confidence like "95%" or 0.95
        if isinstance(raw_conf, str) and raw_conf.strip().endswith("%"):
            conf = float(raw_conf.strip().replace("%", "")) / 100.0
        else:
            conf = float(raw_conf)
            if conf > 1:
                conf = conf / 100.0

        alpha = 1.0 - conf
        var_thr = float(row["VaR_threshold"])  # constant VaR threshold for this confidence

        breach_mask = (r < var_thr).astype(int)
        breaches = int(breach_mask.sum())
        breach_rate = breaches / n
        
        kupiec_p = kupiec_pof_test(breaches, n, alpha)
        christ_p = christoffersen_independence_test(breach_mask)

        rows.append({
            "method": "hist",
            "confidence": conf,
            "alpha": alpha,
            "N": n,
            "VaR_threshold": var_thr,
            "breaches": breaches,
            "breach_rate": breach_rate,
            "kupiec_pvalue": kupiec_p,
            "christoffersen_pvalue": christ_p,
            "kupiec_pass_0.05": (kupiec_p > 0.05) if not np.isnan(kupiec_p) else False,
            "christ_pass_0.05": (christ_p > 0.05) if not np.isnan(christ_p) else False,
        })

    out = pd.DataFrame(rows).sort_values(by=["confidence"], ascending=False)

    os.makedirs("outputs/tables", exist_ok=True)
    out_path = "outputs/tables/var_backtest_summary.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(out)



if __name__ == "__main__":
    main()
