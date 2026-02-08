import pandas as pd
import yfinance as yf

weights = pd.read_csv("data/weights.csv")
tickers = weights["ticker"].tolist()

prices = yf.download(tickers, start="2021-01-01", progress=False)["Adj Close"]

if isinstance(prices, pd.Series):
    prices = prices.to_frame()

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
