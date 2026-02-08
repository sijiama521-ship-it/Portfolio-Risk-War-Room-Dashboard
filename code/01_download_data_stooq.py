import pandas as pd
from pandas_datareader import data as pdr

weights = pd.read_csv("data/weights.csv")
tickers = weights["ticker"].tolist()

def to_stooq(t):
    t = t.strip()
    if t.upper().endswith(".TO"):
        return t.lower()
    else:
        return t.lower() + ".us"

stooq_tickers = [to_stooq(t) for t in tickers]

frames = {}
for orig, stq in zip(tickers, stooq_tickers):
    df = pdr.DataReader(stq, "stooq")
    df = df.sort_index()
    frames[orig] = df["Close"].rename(orig)

prices = pd.concat(frames.values(), axis=1)
prices = prices.loc[prices.index >= "2021-01-01"]
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
