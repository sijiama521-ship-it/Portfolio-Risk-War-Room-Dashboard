# code/06_var_backtest_plot.py
# Plot returns + VaR line + breach points (VaR backtest visualization)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
IMG_DIR = Path("images")
IMG_DIR.mkdir(exist_ok=True)

# ---- Load daily portfolio returns ----
# This file should already exist from your pipeline (02_make_returns_from_prices.py or 03_risk_metrics.py)
# If your filename differs, change it here.
RETURNS_FILE_CANDIDATES = [
    DATA_DIR / "portfolio_returns.csv",
    DATA_DIR / "returns.csv",
]

returns_path = None
for p in RETURNS_FILE_CANDIDATES:
    if p.exists():
        returns_path = p
        break

if returns_path is None:
    raise FileNotFoundError(
        "Could not find portfolio returns file. Expected one of: "
        + ", ".join(str(x) for x in RETURNS_FILE_CANDIDATES)
    )

df = pd.read_csv(returns_path)

# Try to find the return column (common names)
possible_return_cols = ["portfolio_return", "return", "daily_return"]
ret_col = None
for c in possible_return_cols:
    if c in df.columns:
        ret_col = c
        break

if ret_col is None:
    raise ValueError(
        f"Could not find a return column in {returns_path}. "
        f"Found columns: {list(df.columns)}"
    )

# Parse dates if present
date_col = None
for c in ["date", "Date"]:
    if c in df.columns:
        date_col = c
        break

if date_col is not None:
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    x = df[date_col]
else:
    # Fallback to index if no date column
    df = df.reset_index(drop=True)
    x = df.index

r = df[ret_col].astype(float)

# ---- Compute VaR thresholds (historical) ----
# VaR95 = 5% quantile of returns (loss tail), VaR99 = 1% quantile
q95 = np.quantile(r, 0.05)
q99 = np.quantile(r, 0.01)

# Identify breaches (returns <= quantile)
breach95 = r <= q95
breach99 = r <= q99

# ---- Plot VaR95 backtest ----
plt.figure()
plt.plot(x, r, linewidth=1)
plt.axhline(q95, linestyle="--", linewidth=1, label=f"VaR95 threshold ({q95:.4f})")
plt.scatter(x[breach95], r[breach95], s=18, label=f"Breaches: {breach95.sum()}")

plt.title("VaR Backtest (95%): Daily Portfolio Returns with Breaches")
plt.xlabel("Date" if date_col else "Index")
plt.ylabel("Daily return")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "var_backtest_95.png", dpi=200)
plt.close()
print("Saved: images/var_backtest_95.png")

# ---- Plot VaR99 backtest ----
plt.figure()
plt.plot(x, r, linewidth=1)
plt.axhline(q99, linestyle="--", linewidth=1, label=f"VaR99 threshold ({q99:.4f})")
plt.scatter(x[breach99], r[breach99], s=18, label=f"Breaches: {breach99.sum()}")

plt.title("VaR Backtest (99%): Daily Portfolio Returns with Breaches")
plt.xlabel("Date" if date_col else "Index")
plt.ylabel("Daily return")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "var_backtest_99.png", dpi=200)
plt.close()
print("Saved: images/var_backtest_99.png")

# ---- Save summary CSV (optional) ----
summary = pd.DataFrame(
    [
        {
            "confidence": "95%",
            "VaR_threshold": q95,
            "breaches": int(breach95.sum()),
            "breach_rate": float(breach95.mean()),
            "expected": 0.05,
        },
        {
            "confidence": "99%",
            "VaR_threshold": q99,
            "breaches": int(breach99.sum()),
            "breach_rate": float(breach99.mean()),
            "expected": 0.01,
        },
    ]
)
summary.to_csv(DATA_DIR / "var_backtest_summary.csv", index=False)
print("Saved: data/var_backtest_summary.csv")
