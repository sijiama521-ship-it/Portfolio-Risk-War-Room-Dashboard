from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
IMG_DIR = Path("images")
IMG_DIR.mkdir(exist_ok=True)

# Load daily returns by asset + portfolio returns
rets = (
    pd.read_csv(DATA_DIR / "returns.csv", parse_dates=["Date"])
    .set_index("Date")
    .sort_index()
)
port = (
    pd.read_csv(DATA_DIR / "portfolio_returns.csv", parse_dates=["Date"])
    .set_index("Date")
    .sort_index()
)

# Load weights
weights = pd.read_csv(DATA_DIR / "weights.csv")
weights["ticker"] = weights["ticker"].astype(str).str.strip()
w = weights.set_index("ticker")["weight"]
w = w / w.sum()

# Align tickers
tickers = [c for c in rets.columns if c in w.index]
rets = rets[tickers]
w = w.loc[tickers]

# ---- Stress scenarios (edit numbers later if you want) ----
# Values are "one-day return shocks"
scenarios = {
    "Equity shock: XIU & VFV -10%": {"XIU": -0.10, "VFV": -0.10},
    "Bond shock: ZAG -3%": {"ZAG": -0.03},
    "Gold rally: GLD +3%": {"GLD": 0.03},
    "International shock: XEF -2%": {"XEF": -0.02},
    "Bank shock: RY -8%": {"RY": -0.08},
    "All risk-off: equities -8%, bonds -2%, gold +2%": {
        "XIU": -0.08,
        "VFV": -0.08,
        "XEF": -0.08,
        "RY": -0.08,
        "ZAG": -0.02,
        "GLD": 0.02,
    },
}


def hist_var(x, alpha=0.05):
    return -np.quantile(x, alpha)


def hist_cvar(x, alpha=0.05):
    q = np.quantile(x, alpha)
    return -(x[x <= q].mean())


base_r = port["portfolio_return"].dropna()

rows = []
for name, shock_dict in scenarios.items():
    shock_vec = pd.Series(0.0, index=tickers)
    for k, v in shock_dict.items():
        if k in shock_vec.index:
            shock_vec.loc[k] = v

    # portfolio one-day hit from shock-only (approx)
    shock_only = float((shock_vec * w).sum())

    # shocked portfolio return series: add shock to each day's asset return (simple what-if)
    shocked_port = (rets.add(shock_vec, axis=1) * w).sum(axis=1).dropna()

    rows.append(
        {
            "scenario": name,
            "shock_only_portfolio_return": shock_only,
            "VaR95_base": hist_var(base_r, 0.05),
            "VaR95_shocked": hist_var(shocked_port, 0.05),
            "CVaR95_base": hist_cvar(base_r, 0.05),
            "CVaR95_shocked": hist_cvar(shocked_port, 0.05),
        }
    )

out = pd.DataFrame(rows).sort_values("VaR95_shocked", ascending=False)
out.to_csv(DATA_DIR / "stress_test_summary.csv", index=False)
print("Saved: data/stress_test_summary.csv")
print(out)

# Plot: compare base distribution vs worst shocked distribution
worst = out.iloc[0]["scenario"]
shock_dict = scenarios[worst]
shock_vec = pd.Series(0.0, index=tickers)
for k, v in shock_dict.items():
    if k in shock_vec.index:
        shock_vec.loc[k] = v
shocked_port = (rets.add(shock_vec, axis=1) * w).sum(axis=1).dropna()

plt.figure()
plt.hist(base_r.values, bins=60, alpha=0.6, label="Base")
plt.hist(shocked_port.values, bins=60, alpha=0.6, label=f"Shocked: {worst}")
plt.title("Portfolio Daily Returns: Base vs Worst Stress Scenario")
plt.xlabel("Daily return")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "stress_test_hist_compare.png", dpi=200)
plt.close()

print("Saved: images/stress_test_hist_compare.png")
