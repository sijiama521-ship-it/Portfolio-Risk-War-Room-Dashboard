import os

import numpy as np
import pandas as pd

from src.backtest import christoffersen_independence_test, kupiec_pof_test


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
        var_thr = float(
            row["VaR_threshold"]
        )  # constant VaR threshold for this confidence

        breach_mask = (r < var_thr).astype(int)
        breaches = int(breach_mask.sum())
        breach_rate = breaches / n

        kupiec_p = kupiec_pof_test(breaches, n, alpha)
        christ_p = christoffersen_independence_test(breach_mask)

        rows.append(
            {
                "method": "hist",
                "confidence": conf,
                "alpha": alpha,
                "N": n,
                "VaR_threshold": var_thr,
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

    out = pd.DataFrame(rows).sort_values(by=["confidence"], ascending=False)

    os.makedirs("outputs/tables", exist_ok=True)
    out_path = "outputs/tables/var_backtest_summary.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
