import time

import pandas as pd

weights = pd.read_csv("data/weights.csv")
tickers = weights["ticker"].tolist()

# Convert date to unix timestamps for Yahoo (UTC)
start = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp())
end = int(pd.Timestamp.utcnow().timestamp())


def fetch_adj_close(ticker):
    url = (
        "https://query1.finance.yahoo.com/v7/finance/download/"
        f"{ticker}"
        f"?period1={start}&period2={end}"
        "&interval=1d&events=history&includeAdjustedClose=true"
    )
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # use Adj Close when available, otherwise Close
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].rename(ticker)
    return s


frames = {}
for t in tickers:
    try:
        frames[t] = fetch_adj_close(t)
        time.sleep(0.5)  # be polite; reduce rate limits
    except Exception as e:
        raise RuntimeError(f"Failed for {t}: {e}") from e

prices = pd.concat(frames.values(), axis=1)
prices = prices.dropna(how="all")

returns = prices.pct_change().dropna()

w = weights.set_index("ticker").loc[returns.columns]["weight"]
portfolio_returns = (returns * w).sum(axis=1).to_frame("portfolio_return")

prices.to_csv("data/prices.csv")
returns.to_csv("data/returns.csv")
portfolio_returns.to_csv("data/portfolio_returns.csv")

print("Saved: data/prices.csv, data/returns.csv, data/portfolio_returns.csv")
print("Rows of daily returns:", len(returns))
print("Date range:", returns.index.min(), "to", returns.index.max())
