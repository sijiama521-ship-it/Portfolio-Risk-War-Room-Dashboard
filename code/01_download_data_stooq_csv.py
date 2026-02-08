import pandas as pd

weights = pd.read_csv("data/weights.csv")
tickers = weights["ticker"].tolist()

def stooq_symbol(t):
    t = t.strip()
    if t.upper().endswith(".TO"):
        return t[:-3].lower() + ".to"   # XIU.TO -> xiu.to
    else:
        return t.lower() + ".us"        # GLD -> gld.us

def fetch_close(sym, orig_ticker):
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    df = pd.read_csv(url)

    # If stooq returns an error page or empty, df may have weird columns
    if df.empty:
        raise RuntimeError(f"{orig_ticker} ({sym}): Empty response from Stooq.")

    # Identify date column (usually 'Date', but be robust)
    cols = list(df.columns)
    date_col = None
    for c in cols:
        if str(c).strip().lower() == "date":
            date_col = c
            break
    if date_col is None:
        # fallback: assume first column is date
        date_col = cols[0]

    # Identify close column
    close_col = None
    for c in cols:
        if str(c).strip().lower() == "close":
            close_col = c
            break
    if close_col is None:
        # Print debug info then fail
        preview = df.head(5).to_string(index=False)
        raise RuntimeError(
            f"{orig_ticker} ({sym}): Could not find 'Close' column. "
            f"Columns={cols}. Preview:\n{preview}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()
    return df[close_col]

frames = {}
for t in tickers:
    sym = stooq_symbol(t)
    close = fetch_close(sym, t)
    frames[t] = close.rename(t)

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
