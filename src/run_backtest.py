import os
import math
import pandas as pd
import numpy as np
from scipy.stats import chi2


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


def main():
    returns_path = "data/portfolio_returns.csv"
    backtest_path = "data/var_backtest.csv"

    rets = pd.read_csv(returns_path)
    n = int(rets.shape[0])

    bt = pd.read_csv(backtest_path)

    if "confidence" not in bt.columns or "breaches" not in bt.columns:
        raise ValueError(f"Unexpected columns in {backtest_path}: {list(bt.columns)}")

    rows = []
    for _, r in bt.iterrows():
        raw_conf = r["confidence"]
        if isinstance(raw_conf, str) and raw_conf.strip().endswith("%"):
            conf = float(raw_conf.strip().replace("%","")) / 100.0
        else:
            conf = float(raw_conf)
            if conf > 1:
                conf = conf / 100.0
        alpha = 1.0 - conf
        breaches = int(r["breaches"])
        breach_rate = float(r["breach_rate"]) if "breach_rate" in bt.columns else breaches / n
        expected = float(r["expected"]) if "expected" in bt.columns else alpha * n

        pval = kupiec_pof_test(breaches, n, alpha)

        rows.append({
            "confidence": conf,
            "alpha": alpha,
            "N": n,
            "breaches": breaches,
            "breach_rate": breach_rate,
            "expected_breaches": expected,
            "kupiec_pvalue": pval,
            "pass_0.05": (pval > 0.05) if not np.isnan(pval) else False
        })

    out = pd.DataFrame(rows).sort_values(by="confidence", ascending=False)

    os.makedirs("outputs/tables", exist_ok=True)
    out_path = "outputs/tables/var_backtest_kupiec.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
