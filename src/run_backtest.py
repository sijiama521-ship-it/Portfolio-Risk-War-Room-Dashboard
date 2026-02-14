import os
import math
import pandas as pd
import numpy as np
import yaml
from scipy.stats import chi2


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def kupiec_pof_test(breaches: int, n: int, alpha: float) -> float:
    if n <= 0:
        return float("nan")

    x = breaches
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
    var_path = "data/var_cvar_summary.csv"

    if not os.path.exists(returns_path):
        raise FileNotFoundError(f"Missing {returns_path}. Run your pipeline first.")
    if not os.path.exists(var_path):
        raise FileNotFoundError(f"Missing {var_path}. Run your pipeline first.")

    rets = pd.read_csv(returns_path, parse_dates=["Date"])
    var = pd.read_csv(var_path, parse_dates=["Date"])

    returns_col = [c for c in rets.columns if c.lower() != "date" and "return" in c.lower()]
    if not returns_col:
        raise ValueError("Cannot find return column in data/portfolio_returns.csv")
    returns_col = returns_col[0]

    df = rets[["Date", returns_col]].merge(var, on="Date", how="inner").dropna()

    alphas = [0.05, 0.01]

    rows = []
    for alpha in alphas:
        level = int((1 - alpha) * 100)
        candidates = [c for c in df.columns if "var" in c.lower() and (str(level) in c.lower() or str(alpha) in c.lower())]
        if not candidates:
            raise ValueError(f"Cannot find VaR column for alpha={alpha}. Columns: {list(df.columns)}")
            var_col = candidates[0]


        breaches = int((df[returns_col] < df[var_col]).sum())
        n = int(df.shape[0])
        pval = kupiec_pof_test(breaches, n, alpha)

        rows.append({
            "method": var_col,
            "alpha": alpha,
            "N": n,
            "breaches": breaches,
            "breach_rate": breaches / n if n else float("nan"),
            "kupiec_pvalue": pval,
            "pass_0.05": (pval > 0.05) if not np.isnan(pval) else False
        })

    out = pd.DataFrame(rows)
    os.makedirs("outputs/tables", exist_ok=True)
    out.to_csv("outputs/tables/var_backtest_summary.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
