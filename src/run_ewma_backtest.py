# src/run_ewma_backtest.py
import os

import numpy as np
import pandas as pd

from src.backtest import christoffersen_independence_test, kupiec_pof_test
from src.var_models import ewma_var_threshold, ewma_vol


def main():
    # ---- inputs ----
    ret_path = "data/portfolio_returns.csv"
    if not os.path.exists(ret_path):
        raise FileNotFoundError(
            f"Missing {ret_path}. Make sure your returns file exists."
        )

    rets = pd.read_csv(ret_path)

    # Expect a column named portfolio_return (same as your Step 1 script)
    if "portfolio_return" not in rets.columns:
        raise ValueError("portfolio_returns.csv must contain column: portfolio_return")

    r = rets["portfolio_return"].astype(float).values
    n = len(r)

    # common EWMA lambda for daily
    lam = 0.94

    # You can change this: use sample mean or force 0.0
    mu = float(np.mean(r))

    # We backtest at 95% and 99% confidence (alpha = 0.05, 0.01)
    configs = [
        {"confidence": 0.95, "alpha": 0.05},
        {"confidence": 0.99, "alpha": 0.01},
    ]

    # ---- compute full series once (useful to save) ----
    sigma = ewma_vol(r, lam=lam)
    out_series = pd.DataFrame(
        {
            "t": np.arange(n),
            "portfolio_return": r,
            "ewma_sigma": sigma,
        }
    )

    rows = []
    for cfg in configs:
        conf = float(cfg["confidence"])
        alpha = float(cfg["alpha"])

        thr = ewma_var_threshold(
            r, alpha=alpha, lam=lam, mu=mu
        )  # dynamic threshold series
        breach_mask = (r < thr).astype(int)
        breaches = int(breach_mask.sum())
        breach_rate = breaches / n

        kupiec_p = float(kupiec_pof_test(breaches, n, alpha))
        christ_p = float(christoffersen_independence_test(breach_mask))

        rows.append(
            {
                "method": "ewma",
                "lam": lam,
                "mu_used": mu,
                "confidence": conf,
                "alpha": alpha,
                "N": n,
                "breaches": breaches,
                "breach_rate": breach_rate,
                "kupiec_pvalue": kupiec_p,
                "christoffersen_pvalue": christ_p,
                "kupiec_pass_0.05": (
                    (kupiec_p > 0.05) if not np.isnan(kupiec_p) else False
                ),
                "christ_pass_0.05": (
                    (christ_p > 0.05) if not np.isnan(christ_p) else False
                ),
            }
        )

        # also store VaR threshold series in the series output
        out_series[f"ewma_var_thr_{int(conf*100)}"] = thr

    # ---- outputs ----
    os.makedirs("outputs/tables", exist_ok=True)

    # 1) daily EWMA series
    series_path = "outputs/tables/ewma_var_series.csv"
    out_series.to_csv(series_path, index=False)

    # 2) ewma backtest summary
    summary = pd.DataFrame(rows).sort_values(by=["confidence"], ascending=False)
    summary_path = "outputs/tables/var_ewma_backtest_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {series_path}")
    print(f"Saved: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
