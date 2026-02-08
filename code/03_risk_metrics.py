import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

# ---------- Load ----------
returns = pd.read_csv(f"{DATA_DIR}/returns.csv", parse_dates=["Date"]).set_index("Date")
port = pd.read_csv(f"{DATA_DIR}/portfolio_returns.csv", parse_dates=["Date"]).set_index("Date")

# Ensure numeric
returns = returns.apply(pd.to_numeric, errors="coerce")
port["portfolio_return"] = pd.to_numeric(port["portfolio_return"], errors="coerce")

# Align dates (inner join)
df = returns.join(port, how="inner")
returns = df[returns.columns].dropna(how="any")
port = df[["portfolio_return"]].loc[returns.index]

TRADING_DAYS = 252

def max_drawdown(r: pd.Series) -> float:
    nav = (1 + r).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1
    return float(dd.min())

def hist_var_cvar(r: pd.Series, alpha: float = 0.05):
    r = r.dropna()
    if len(r) == 0:
        return np.nan, np.nan
    var = np.quantile(r, alpha)              # e.g., 5% quantile (usually negative)
    cvar = r[r <= var].mean() if (r <= var).any() else np.nan
    return float(var), float(cvar)

def summarize_series(r: pd.Series, name: str) -> dict:
    mu_d = r.mean()
    vol_d = r.std(ddof=1)
    ann_ret = mu_d * TRADING_DAYS
    ann_vol = vol_d * np.sqrt(TRADING_DAYS)
    var95, cvar95 = hist_var_cvar(r, alpha=0.05)
    mdd = max_drawdown(r)
    return {
        "asset": name,
        "n_obs": int(r.dropna().shape[0]),
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "VaR_95_hist": var95,
        "CVaR_95_hist": cvar95,
        "max_drawdown": mdd,
    }

# ---------- Risk summary ----------
rows = []
for col in returns.columns:
    rows.append(summarize_series(returns[col], col))
rows.append(summarize_series(port["portfolio_return"], "PORTFOLIO"))

risk_summary = pd.DataFrame(rows).set_index("asset")
risk_summary.to_csv(f"{DATA_DIR}/risk_summary.csv")
print("Saved: data/risk_summary.csv")

# ---------- Correlation ----------
corr = returns.corr()
corr.to_csv(f"{DATA_DIR}/corr.csv")
print("Saved: data/corr.csv")

# ---------- Plots ----------
# 1) Portfolio NAV
nav = (1 + port["portfolio_return"]).cumprod()
plt.figure()
plt.plot(nav.index, nav.values)

plt.title("Portfolio NAV (Cumulative Growth)")
plt.xlabel("Date")
plt.ylabel("NAV")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/portfolio_nav.png", dpi=200)
plt.close()
print("Saved: images/portfolio_nav.png")

# 2) Portfolio drawdown
peak = nav.cummax()
dd = nav / peak - 1
plt.figure()
plt.plot(dd.index, dd.values)
plt.title("Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/portfolio_drawdown.png", dpi=200)
plt.close()
print("Saved: images/portfolio_drawdown.png")

# 3) Rolling 20d vol (portfolio)
roll_vol = port["portfolio_return"].rolling(20).std(ddof=1) * np.sqrt(TRADING_DAYS)
plt.figure()
plt.plot(roll_vol.index, roll_vol.values)
plt.title("Rolling 20D Annualized Vol (Portfolio)")
plt.xlabel("Date")
plt.ylabel("Vol")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/portfolio_rolling_vol_20d.png", dpi=200)
plt.close()
print("Saved: images/portfolio_rolling_vol_20d.png")

print("Done. Date range:", returns.index.min(), "to", returns.index.max(), " | rows:", len(returns))
