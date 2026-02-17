from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
IMG_DIR = Path("images")
IMG_DIR.mkdir(exist_ok=True)

TRADING_DAYS = 252
ALPHAS = [0.05, 0.01]  # 95% and 99%

# ---------- Load portfolio returns ----------
port = pd.read_csv(DATA_DIR / "portfolio_returns.csv", parse_dates=["Date"])
port = port.set_index("Date").sort_index()
r = port["portfolio_return"].dropna()


# ---------- Helper functions ----------
def hist_var(x, alpha):
    # VaR as positive number (loss)
    return -np.quantile(x, alpha)


def hist_cvar(x, alpha):
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return -tail.mean()


def param_var(x, alpha):
    mu = x.mean()
    sigma = x.std(ddof=1)
    z = pd.Series(x).quantile(alpha)  # fallback if you don't want scipy
    # better normal z without scipy: approximate using numpy
    # we'll use inverse CDF via numpy if available:
    try:
        from statistics import NormalDist

        z = NormalDist().inv_cdf(alpha)
    except Exception:
        # rough fallback
        z = -1.645 if alpha == 0.05 else -2.326
    return -(mu + sigma * z)


def param_cvar(x, alpha):
    mu = x.mean()
    sigma = x.std(ddof=1)
    try:
        from statistics import NormalDist

        z = NormalDist().inv_cdf(alpha)
        pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z * z)
        return -(mu - sigma * pdf / alpha)
    except Exception:
        return np.nan


# ---------- Compute ----------
rows = []
for a in ALPHAS:
    rows.append(
        {
            "alpha": a,
            "Hist_VaR": hist_var(r, a),
            "Hist_CVaR": hist_cvar(r, a),
            "Param_VaR": param_var(r, a),
            "Param_CVaR": param_cvar(r, a),
        }
    )

out = pd.DataFrame(rows)
out["confidence"] = (1 - out["alpha"]).map(lambda x: f"{int(x*100)}%")
out = out[["confidence", "Hist_VaR", "Hist_CVaR", "Param_VaR", "Param_CVaR"]]

out.to_csv(DATA_DIR / "var_cvar_summary.csv", index=False)
print("Saved: data/var_cvar_summary.csv")
print(out)

# ---------- Plot: returns distribution + VaR lines ----------
plt.figure()
plt.hist(r.values, bins=60)
for a in ALPHAS:
    v = np.quantile(r, a)
    plt.axvline(v, linestyle="--", label=f"Hist VaR {(1-a):.0%}")
plt.title("Portfolio Daily Returns (Histogram) + VaR Lines")
plt.xlabel("Daily return")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "portfolio_returns_hist_var.png", dpi=200)
plt.close()
print("Saved: images/portfolio_returns_hist_var.png")

# ---------- Optional: breaches check (quick backtest) ----------
bt = []
for a in ALPHAS:
    q = np.quantile(r, a)
    breaches = (r <= q).sum()
    rate = breaches / len(r)
    bt.append(
        {
            "confidence": f"{int((1-a)*100)}%",
            "breaches": int(breaches),
            "breach_rate": rate,
            "expected": a,
        }
    )

pd.DataFrame(bt).to_csv(DATA_DIR / "var_backtest.csv", index=False)
print("Saved: data/var_backtest.csv")
