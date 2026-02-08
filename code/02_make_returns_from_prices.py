import pandas as pd

# --- Load ---
prices = pd.read_csv("data/prices.csv")

# --- Clean date ---
prices["Date"] = pd.to_datetime(prices["Date"])
prices = prices.sort_values("Date").set_index("Date")

# --- Clean numeric columns: remove $, €, commas, spaces ---
for c in prices.columns:
    prices[c] = (
        prices[c].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    prices[c] = pd.to_numeric(prices[c], errors="coerce")

# drop rows where all tickers are missing
prices = prices.dropna(how="all")

# --- Returns ---
returns = prices.pct_change().dropna(how="all")

# --- Weights ---
weights = pd.read_csv("data/weights.csv")
weights["ticker"] = weights["ticker"].astype(str).str.strip()
w_all = weights.set_index("ticker")["weight"]

# align common tickers
common = [c for c in returns.columns if c in w_all.index]
missing = [c for c in returns.columns if c not in w_all.index]
if missing:
    print("WARNING: missing tickers in weights.csv:", missing)

returns = returns[common]
w = w_all.loc[common]
w = w / w.sum()  # normalize just in case

portfolio_returns = (returns * w).sum(axis=1).to_frame("portfolio_return")

# --- Save ---
prices.to_csv("data/prices_clean.csv")
returns.to_csv("data/returns.csv")
portfolio_returns.to_csv("data/portfolio_returns.csv")

print("Saved: data/prices_clean.csv, data/returns.csv, data/portfolio_returns.csv")
print("Rows of daily returns:", len(returns))
print("Date range:", returns.index.min(), "to", returns.index.max())
