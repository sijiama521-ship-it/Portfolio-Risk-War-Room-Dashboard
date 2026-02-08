Last login: Sat Feb  7 22:41:47 on ttys000
(base) sijiama@eduroam-campus-10-36-178-108 ~ % conda activate warroom

(warroom) sijiama@eduroam-campus-10-36-178-108 ~ % ps aux | grep -i python

sijiama          33557   0.0  0.0 435300432   1344 s000  S+   11:03PM   0:00.00 grep -i python
(warroom) sijiama@eduroam-campus-10-36-178-108 ~ % pip install pandas_datareader

Collecting pandas_datareader
  Downloading pandas_datareader-0.10.0-py3-none-any.whl.metadata (2.9 kB)
Collecting lxml (from pandas_datareader)
  Downloading lxml-6.0.2-cp311-cp311-macosx_10_9_universal2.whl.metadata (3.6 kB)
Requirement already satisfied: pandas>=0.23 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from pandas_datareader) (3.0.0)
Requirement already satisfied: requests>=2.19.0 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from pandas_datareader) (2.32.5)
Requirement already satisfied: numpy>=1.26.0 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from pandas>=0.23->pandas_datareader) (2.4.2)
Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from pandas>=0.23->pandas_datareader) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas>=0.23->pandas_datareader) (1.17.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from requests>=2.19.0->pandas_datareader) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from requests>=2.19.0->pandas_datareader) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from requests>=2.19.0->pandas_datareader) (2.6.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from requests>=2.19.0->pandas_datareader) (2026.1.4)
Downloading pandas_datareader-0.10.0-py3-none-any.whl (109 kB)
Downloading lxml-6.0.2-cp311-cp311-macosx_10_9_universal2.whl (8.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.6/8.6 MB 13.4 MB/s  0:00:00
Installing collected packages: lxml, pandas_datareader
Successfully installed lxml-6.0.2 pandas_datareader-0.10.0
(warroom) sijiama@eduroam-campus-10-36-178-108 ~ % >....                                                                
for orig, stq in zip(tickers, stooq_tickers):
    df = pdr.DataReader(stq, "stooq")
    # Stooq returns newest -> oldest; sort ascending by date
    df = df.sort_index()
    frames[orig] = df["Close"].rename(orig)

prices = pd.concat(frames.values(), axis=1)

# Keep a reasonable window
prices = prices.loc[prices.index >= "2021-01-01"]
prices = prices.dropna(how="all")

# Returns
returns = prices.pct_change().dropna()

# Portfolio return
w = weights.set_index("ticker").loc[returns.columns]["weight"]
portfolio_returns = (returns * w).sum(axis=1).to_frame("portfolio_return")

# Save
prices.to_csv("data/prices.csv")
returns.to_csv("data/returns.csv")
portfolio_returns.to_csv("data/portfolio_returns.csv")

print("Saved: data/prices.csv, data/returns.csv, data/portfolio_returns.csv")
print("Rows of daily returns:", len(returns))
print("Date range:", returns.index.min(), "to", returns.index.max())
EOF

zsh: no such file or directory: code/01_download_data_stooq.py
(warroom) sijiama@eduroam-campus-10-36-178-108 ~ % cd ~/Desktop/portfolio-risk-war-room
pwd
ls

cd: no such file or directory: /Users/sijiama/Desktop/portfolio-risk-war-room
/Users/sijiama
Desktop			Library			Pictures		stylecs116 (1).pdf	stylecs116.pdf
Documents		Movies			Public			stylecs116 (2).pdf	Tutorial 1.pdf
Downloads		Music			sql cheatsheet.pdf	stylecs116 (3).pdf
(warroom) sijiama@eduroam-campus-10-36-178-108 ~ % ls ~/Desktop

6031f2e55e37ae76235338a83f11a509 2.png		Screenshot 2026-01-31 at 8.02.19 PM.png
6031f2e55e37ae76235338a83f11a509.png		Screenshot 2026-02-01 at 6.21.10 PM.png
92f41b37da97e6d0175a1038b7d7fbf3 2.png		Screenshot 2026-02-02 at 12.19.52 AM.png
92f41b37da97e6d0175a1038b7d7fbf3.png		Screenshot 2026-02-02 at 2.56.06 AM.png
bba54386c050f065feddc1295e0e0164 2.jpg		Screenshot 2026-02-03 at 2.51.02 PM.png
bba54386c050f065feddc1295e0e0164.jpg		Screenshot 2026-02-03 at 3.28.33 PM.png
data (3).csv					Screenshot 2026-02-03 at 5.46.04 PM.png
ec6e86b89b485bcfd761fe3a5d41521f 2.jpg		Screenshot 2026-02-04 at 1.14.38 AM.png
ec6e86b89b485bcfd761fe3a5d41521f.jpg		Screenshot 2026-02-04 at 1.15.05 AM.png
ed9388ecd7fc8fca4408d529c6e940f9 2.png		Screenshot 2026-02-07 at 10.59.52 PM.png
ed9388ecd7fc8fca4408d529c6e940f9.png		Sijia Ma coop resume 1.pdf
portfolio-risk-war-room:
(warroom) sijiama@eduroam-campus-10-36-178-108 ~ % cd ~/Desktop/portfolio-risk-war-room
pwd
ls

cd: no such file or directory: /Users/sijiama/Desktop/portfolio-risk-war-room
/Users/sijiama
Desktop			Library			Pictures		stylecs116 (1).pdf	stylecs116.pdf
Documents		Movies			Public			stylecs116 (2).pdf	Tutorial 1.pdf
Downloads		Music			sql cheatsheet.pdf	stylecs116 (3).pdf
(warroom) sijiama@eduroam-campus-10-36-178-108 ~ % cd ~/Desktop/portfolio-risk-war-room
pwd
ls

/Users/sijiama/Desktop/portfolio-risk-war-room
code	data	excel	images	report
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % ls code

01_download_data.py
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % >....                                          
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
EOF

(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % pip install pandas_datareader

Requirement already satisfied: pandas_datareader in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (0.10.0)
Requirement already satisfied: lxml in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from pandas_datareader) (6.0.2)
Requirement already satisfied: pandas>=0.23 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from pandas_datareader) (3.0.0)
Requirement already satisfied: requests>=2.19.0 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from pandas_datareader) (2.32.5)
Requirement already satisfied: numpy>=1.26.0 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from pandas>=0.23->pandas_datareader) (2.4.2)
Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from pandas>=0.23->pandas_datareader) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas>=0.23->pandas_datareader) (1.17.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from requests>=2.19.0->pandas_datareader) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from requests>=2.19.0->pandas_datareader) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from requests>=2.19.0->pandas_datareader) (2.6.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/envs/warroom/lib/python3.11/site-packages (from requests>=2.19.0->pandas_datareader) (2026.1.4)
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % python code/01_download_data_stooq.py

Traceback (most recent call last):
  File "/Users/sijiama/Desktop/portfolio-risk-war-room/code/01_download_data_stooq.py", line 2, in <module>
    from pandas_datareader import data as pdr
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas_datareader/__init__.py", line 5, in <module>
    from .data import (
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas_datareader/data.py", line 273, in <module>
    @deprecate_kwarg("access_key", "api_key")
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: deprecate_kwarg() missing 1 required positional argument: 'new_arg_name'
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % >....                                          
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df["Close"]

frames = {}
for t in tickers:
    sym = stooq_symbol(t)
    frames[t] = fetch_close(sym).rename(t)

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
EOF

(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % python code/01_download_data_stooq_csv.py

Traceback (most recent call last):
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3641, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 168, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 197, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7668, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7676, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'Date'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/sijiama/Desktop/portfolio-risk-war-room/code/01_download_data_stooq_csv.py", line 23, in <module>
    frames[t] = fetch_close(sym).rename(t)
                ^^^^^^^^^^^^^^^^
  File "/Users/sijiama/Desktop/portfolio-risk-war-room/code/01_download_data_stooq_csv.py", line 16, in fetch_close
    df["Date"] = pd.to_datetime(df["Date"])
                                ~~^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/core/frame.py", line 4378, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3648, in get_loc
    raise KeyError(key) from err
KeyError: 'Date'
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % >....                                          
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
EOF

(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % python code/01_download_data_stooq_csv.py

Traceback (most recent call last):
  File "/Users/sijiama/Desktop/portfolio-risk-war-room/code/01_download_data_stooq_csv.py", line 54, in <module>
    close = fetch_close(sym, t)
            ^^^^^^^^^^^^^^^^^^^
  File "/Users/sijiama/Desktop/portfolio-risk-war-room/code/01_download_data_stooq_csv.py", line 19, in fetch_close
    raise RuntimeError(f"{orig_ticker} ({sym}): Empty response from Stooq.")
RuntimeError: XIU.TO (xiu.to): Empty response from Stooq.
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % >....                                          
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].rename(ticker)
    return s

frames = {}
for t in tickers:
    try:
        frames[t] = fetch_adj_close(t)
        time.sleep(0.5)  # be polite; reduce rate limits
    except Exception as e:
        raise RuntimeError(f"Failed for {t}: {e}")

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
EOF

(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % python code/01_download_data_yahoo_csv.py

/Users/sijiama/Desktop/portfolio-risk-war-room/code/01_download_data_yahoo_csv.py:9: Pandas4Warning: Timestamp.utcnow is deprecated and will be removed in a future version. Use Timestamp.now('UTC') instead.
  end = int(pd.Timestamp.utcnow().timestamp())
Traceback (most recent call last):
  File "/Users/sijiama/Desktop/portfolio-risk-war-room/code/01_download_data_yahoo_csv.py", line 29, in <module>
    frames[t] = fetch_adj_close(t)
                ^^^^^^^^^^^^^^^^^^
  File "/Users/sijiama/Desktop/portfolio-risk-war-room/code/01_download_data_yahoo_csv.py", line 18, in fetch_adj_close
    df = pd.read_csv(url)
         ^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 873, in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 300, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 1645, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 1904, in _make_engine
    self.handles = get_handle(
                   ^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/io/common.py", line 772, in get_handle
    ioargs = _get_filepath_or_buffer(
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/io/common.py", line 404, in _get_filepath_or_buffer
    with urlopen(req_info) as req:
         ^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/io/common.py", line 281, in urlopen
    return urllib.request.urlopen(*args, **kwargs)  # noqa: TID251
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/urllib/request.py", line 216, in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/urllib/request.py", line 525, in open
    response = meth(req, response)
               ^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/urllib/request.py", line 634, in http_response
    response = self.parent.error(
               ^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/urllib/request.py", line 563, in error
    return self._call_chain(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/urllib/request.py", line 496, in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/urllib/request.py", line 643, in http_error_default
    raise HTTPError(req.full_url, code, msg, hdrs, fp)
urllib.error.HTTPError: HTTP Error 429: Too Many Requests

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/sijiama/Desktop/portfolio-risk-war-room/code/01_download_data_yahoo_csv.py", line 32, in <module>
    raise RuntimeError(f"Failed for {t}: {e}")
RuntimeError: Failed for XIU.TO: HTTP Error 429: Too Many Requests
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % python code/02_make_returns_from_prices.py

python: can't open file '/Users/sijiama/Desktop/portfolio-risk-war-room/code/02_make_returns_from_prices.py': [Errno 2] No such file or directory
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % pwd
ls
ls code
ls data

/Users/sijiama/Desktop/portfolio-risk-war-room
code	data	excel	images	report
01_download_data_stooq_csv.py	01_download_data_yahoo_csv.py
01_download_data_stooq.py	01_download_data.py
prices.csv	weights.csv
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % head -n 2 data/prices.csv

﻿Date,XIU,VFV,XEF,ZAG,GLD,RY
2024-07-08, $33.40 , $134.71 , $37.55 , € 13.58 , $218.19 , $109.19 
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % cd ~/Desktop/portfolio-risk-war-room

(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % >....                                          
    prices[c] = (
        prices[c]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)  # keep digits . -
        .replace("", pd.NA)
    )
    prices[c] = pd.to_numeric(prices[c], errors="coerce")

# Drop rows where ANY asset price is missing (aligned panel)
prices_clean = prices.dropna(how="any")

# Returns
returns = prices_clean.pct_change().dropna(how="any")

# Portfolio returns (weights from weights.csv)
weights = pd.read_csv("data/weights.csv")
w = weights.set_index("ticker").loc[returns.columns]["weight"]
portfolio_returns = (returns * w).sum(axis=1).to_frame("portfolio_return")

# Save
prices_clean.to_csv("data/prices_clean.csv")
returns.to_csv("data/returns.csv")
portfolio_returns.to_csv("data/portfolio_returns.csv")

print("Saved: data/prices_clean.csv, data/returns.csv, data/portfolio_returns.csv")
print("Rows of daily returns:", len(returns))
print("Date range:", returns.index.min(), "to", returns.index.max())
EOF

(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % python code/02_make_returns_from_prices.py

Traceback (most recent call last):
  File "/Users/sijiama/Desktop/portfolio-risk-war-room/code/02_make_returns_from_prices.py", line 26, in <module>
    w = weights.set_index("ticker").loc[returns.columns]["weight"]
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/core/indexing.py", line 1207, in __getitem__
    return self._getitem_axis(maybe_callable, axis=axis)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/core/indexing.py", line 1438, in _getitem_axis
    return self._getitem_iterable(key, axis=axis)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/core/indexing.py", line 1378, in _getitem_iterable
    keyarr, indexer = self._get_listlike_indexer(key, axis)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/core/indexing.py", line 1576, in _get_listlike_indexer
    keyarr, indexer = ax._get_indexer_strict(key, axis_name)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 6302, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "/opt/anaconda3/envs/warroom/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 6355, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
KeyError: "['XIU', 'VFV', 'XEF', 'ZAG', 'RY'] not in index"
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % >....                                          
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
EOF

(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % python code/02_make_returns_from_prices.py

Saved: data/prices_clean.csv, data/returns.csv, data/portfolio_returns.csv
Rows of daily returns: 390
Date range: 2024-07-09 00:00:00 to 2026-02-06 00:00:00
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % head -n 3 data/returns.csv
head -n 3 data/portfolio_returns.csv

Date,XIU,VFV,XEF,ZAG,GLD,RY
2024-07-09,-0.0029940119760479833,0.000668101848415148,-0.004527296937416603,-0.0022091310751104487,0.0016957697419679452,0.003663339133620269
2024-07-10,0.01501501501501501,0.009421364985163105,0.012306046013911276,0.0014760147601475815,0.0036603221083455484,0.01067615658362997
Date,portfolio_return
2024-07-09,-0.001199892492404725
2024-07-10,0.009212784474100132
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % ls data

portfolio_returns.csv	prices_clean.csv	prices.csv		returns.csv		weights.csv
(warroom) sijiama@eduroam-campus-10-36-178-108 portfolio-risk-war-room % >....                                          
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
pl

heredoc> >....                                                                                                          

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
EOF

zsh: command not found: 3）运行它

The default interactive shell is now zsh.
To update your account to use zsh, please run `chsh -s /bin/zsh`.
For more details, please visit https://support.apple.com/kb/HT208050.
bash-3.2$ cd ~/Desktop/portfolio-risk-war-room
bash-3.2$ 
bash-3.2$ 
bash-3.2$ ls ~/Desktop | grep -i war
portfolio-risk-war-room
bash-3.2$ ls ~/Desktop | grep -i war
portfolio-risk-war-room
bash-3.2$ cd "$HOME/Desktop/portfolio-risk-war-room"
bash-3.2$ cd /Users/sijiama/Desktop/portfolio-risk-war-room ls code
bash-3.2$ cd /Users/sijiama/Desktop/portfolio-risk-war-room ls code
bash-3.2$ cd ~/Desktop/portfolio-risk-war-room
bash-3.2$ pwd
/Users/sijiama/Desktop/portfolio-risk-war-room
bash-3.2$ cd ~/Desktop/portfolio-risk-war-room
bash-3.2$ ls code
01_download_data_stooq_csv.py	01_download_data_yahoo_csv.py	02_make_returns_from_prices.py
01_download_data_stooq.py	01_download_data.py		03_risk_metrics.py
bash-3.2$ python code/03_risk_metrics.py
Matplotlib is building the font cache; this may take a moment.
Saved: data/risk_summary.csv
Saved: data/corr.csv
Traceback (most recent call last):
  File "/Users/sijiama/Desktop/portfolio-risk-war-room/code/03_risk_metrics.py", line 76, in <module>
    pl
NameError: name 'pl' is not defined. Did you mean: 'pd'?
bash-3.2$ open -a "Visual Studio Code" code/03_risk_metrics.py
Unable to find application named 'Visual Studio Code'
bash-3.2$ sed -i '' '76s/^pl$/plt/' code/03_risk_metrics.py
bash-3.2$ nl -ba code/03_risk_metrics.py | sed -n '70,85p'
    70	
    71	# ---------- Plots ----------
    72	# 1) Portfolio NAV
    73	nav = (1 + port["portfolio_return"]).cumprod()
    74	plt.figure()
    75	plt.plot(nav.index, nav.values)
    76	plt
    77	
    78	plt.title("Portfolio NAV (Cumulative Growth)")
    79	plt.xlabel("Date")
    80	plt.ylabel("NAV")
    81	plt.tight_layout()
    82	plt.savefig(f"{IMG_DIR}/portfolio_nav.png", dpi=200)
    83	plt.close()
    84	print("Saved: images/portfolio_nav.png")
    85	
bash-3.2$ sed -i '' '76d' code/03_risk_metrics.py
bash-3.2$ python code/03_risk_metrics.py
Saved: data/risk_summary.csv
Saved: data/corr.csv
Saved: images/portfolio_nav.png
Saved: images/portfolio_drawdown.png
Saved: images/portfolio_rolling_vol_20d.png
Done. Date range: 2024-07-09 00:00:00 to 2026-02-06 00:00:00  | rows: 390
bash-3.2$ ls data | grep -E "returns|portfolio_returns|risk_summary|corr"
corr.csv
portfolio_returns.csv
returns.csv
risk_summary.csv
bash-3.2$ ls images | grep -E "portfolio_nav|drawdown|rolling"
portfolio_drawdown.png
portfolio_nav.png
portfolio_rolling_vol_20d.png
bash-3.2$ cat > code/04_var_cvar.py << 'PY'
> import numpy as np
> import pandas as pd
> import matplotlib.pyplot as plt
> from pathlib import Path
> 
> DATA_DIR = Path("data")
> IMG_DIR = Path("images")
> IMG_DIR.mkdir(exist_ok=True)
> 
> TRADING_DAYS = 252
> ALPHAS = [0.05, 0.01]   # 95% and 99%
> 
> # ---------- Load portfolio returns ----------
> port = pd.read_csv(DATA_DIR / "portfolio_returns.csv", parse_dates=["Date"])
> port = port.set_index("Date").sort_index()
> r = port["portfolio_return"].dropna()
> 
> # ---------- Helper functions ----------
> def hist_var(x, alpha):
>     # VaR as positive number (loss)
>     return -np.quantile(x, alpha)
> 
> def hist_cvar(x, alpha):
>     q = np.quantile(x, alpha)
>     tail = x[x <= q]
>     return -tail.mean()
> 
> def param_var(x, alpha):
>     mu = x.mean()
>     sigma = x.std(ddof=1)
>     z = pd.Series(x).quantile(alpha)  # fallback if you don't want scipy
>     # better normal z without scipy: approximate using numpy
>     # we'll use inverse CDF via numpy if available:
>     try:
>         from statistics import NormalDist
>         z = NormalDist().inv_cdf(alpha)
>     except Exception:
>         # rough fallback
>         z = -1.645 if alpha == 0.05 else -2.326
>     return -(mu + sigma * z)
> 
> def param_cvar(x, alpha):
>     mu = x.mean()
>     sigma = x.std(ddof=1)
>     try:
>         from statistics import NormalDist
>         z = NormalDist().inv_cdf(alpha)
>         pdf = (1/np.sqrt(2*np.pi))*np.exp(-0.5*z*z)
>         return -(mu - sigma * pdf/alpha)
>     except Exception:
>         return np.nan
> 
> # ---------- Compute ----------
> rows = []
> for a in ALPHAS:
>     rows.append({
>         "alpha": a,
>         "Hist_VaR": hist_var(r, a),
>         "Hist_CVaR": hist_cvar(r, a),
>         "Param_VaR": param_var(r, a),
>         "Param_CVaR": param_cvar(r, a),
>     })
> 
> out = pd.DataFrame(rows)
> out["confidence"] = (1 - out["alpha"]).map(lambda x: f"{int(x*100)}%")
> out = out[["confidence", "Hist_VaR", "Hist_CVaR", "Param_VaR", "Param_CVaR"]]
> 
> out.to_csv(DATA_DIR / "var_cvar_summary.csv", index=False)
> print("Saved: data/var_cvar_summary.csv")
> print(out)
> 
> # ---------- Plot: returns distribution + VaR lines ----------
> plt.figure()
> plt.hist(r.values, bins=60)
> for a in ALPHAS:
>     v = np.quantile(r, a)
>     plt.axvline(v, linestyle="--", label=f"Hist VaR {(1-a):.0%}")
> plt.title("Portfolio Daily Returns (Histogram) + VaR Lines")
> plt.xlabel("Daily return")
> plt.ylabel("Count")
> plt.legend()
> plt.tight_layout()
> plt.savefig(IMG_DIR / "portfolio_returns_hist_var.png", dpi=200)
> plt.close()
> print("Saved: images/portfolio_returns_hist_var.png")
> 
> # ---------- Optional: breaches check (quick backtest) ----------
> bt = []
> for a in ALPHAS:
>     q = np.quantile(r, a)
>     breaches = (r <= q).sum()
>     rate = breaches / len(r)
>     bt.append({"confidence": f"{int((1-a)*100)}%", "breaches": int(breaches), "breach_rate": rate, "expected": a})
> 
> pd.DataFrame(bt).to_csv(DATA_DIR / "var_backtest.csv", index=False)
> print("Saved: data/var_backtest.csv")
> PY
bash-3.2$ python code/04_var_cvar.py
Saved: data/var_cvar_summary.csv
  confidence  Hist_VaR  Hist_CVaR  Param_VaR  Param_CVaR
0        95%  0.009738   0.014689   0.009550    0.012180
1        99%  0.017664   0.025534   0.013839    0.015972
Saved: images/portfolio_returns_hist_var.png
Saved: data/var_backtest.csv
bash-3.2$ cd ~/Desktop/portfolio-risk-war-room
bash-3.2$ pwd
/Users/sijiama/Desktop/portfolio-risk-war-room
bash-3.2$ ls data
corr.csv		prices_clean.csv	returns.csv		var_backtest.csv	weights.csv
portfolio_returns.csv	prices.csv		risk_summary.csv	var_cvar_summary.csv
bash-3.2$ cat > code/05_stress_test.py << 'PY'
> import numpy as np
> import pandas as pd
> import matplotlib.pyplot as plt
> from pathlib import Path
> 
> DATA_DIR = Path("data")
> IMG_DIR = Path("images")
> IMG_DIR.mkdir(exist_ok=True)
> 
> # Load daily returns by asset + portfolio returns
> rets = pd.read_csv(DATA_DIR / "returns.csv", parse_dates=["Date"]).set_index("Date").sort_index()
> port = pd.read_csv(DATA_DIR / "portfolio_returns.csv", parse_dates=["Date"]).set_index("Date").sort_index()
> 
> # Load weights
> weights = pd.read_csv(DATA_DIR / "weights.csv")
> weights["ticker"] = weights["ticker"].astype(str).str.strip()
> w = weights.set_index("ticker")["weight"]
> w = w / w.sum()
> 
> # Align tickers
> tickers = [c for c in rets.columns if c in w.index]
> rets = rets[tickers]
> w = w.loc[tickers]
> 
> # ---- Stress scenarios (edit numbers later if you want) ----
> # Values are "one-day return shocks"
> scenarios = {
>     "Equity shock: XIU & VFV -10%": {"XIU": -0.10, "VFV": -0.10},
>     "Bond shock: ZAG -3%": {"ZAG": -0.03},
>     "Gold rally: GLD +3%": {"GLD": 0.03},
>     "International shock: XEF -2%": {"XEF": -0.02},
>     "Bank shock: RY -8%": {"RY": -0.08},
>     "All risk-off: equities -8%, bonds -2%, gold +2%": {"XIU": -0.08, "VFV": -0.08, "XEF": -0.08, "RY": -0.08, "ZAG": -0.02, "GLD": 0.02},
> }
> 
> def hist_var(x, alpha=0.05):
>     return -np.quantile(x, alpha)
> 
> def hist_cvar(x, alpha=0.05):
>     q = np.quantile(x, alpha)
>     return -(x[x <= q].mean())
> 
> base_r = port["portfolio_return"].dropna()
> 
> rows = []
> for name, shock_dict in scenarios.items():
>     shock_vec = pd.Series(0.0, index=tickers)
>     for k, v in shock_dict.items():
>         if k in shock_vec.index:
>             shock_vec.loc[k] = v
> 
>     # portfolio one-day hit from shock-only (approx)
>     shock_only = float((shock_vec * w).sum())
> 
>     # shocked portfolio return series: add shock to each day's asset return (simple what-if)
>     shocked_port = (rets.add(shock_vec, axis=1) * w).sum(axis=1).dropna()
> 
>     rows.append({
>         "scenario": name,
>         "shock_only_portfolio_return": shock_only,
>         "VaR95_base": hist_var(base_r, 0.05),
>         "VaR95_shocked": hist_var(shocked_port, 0.05),
>         "CVaR95_base": hist_cvar(base_r, 0.05),
>         "CVaR95_shocked": hist_cvar(shocked_port, 0.05),
>     })
> 
> out = pd.DataFrame(rows).sort_values("VaR95_shocked", ascending=False)
> out.to_csv(DATA_DIR / "stress_test_summary.csv", index=False)
> print("Saved: data/stress_test_summary.csv")
> print(out)
> 
> # Plot: compare base distribution vs worst shocked distribution
> worst = out.iloc[0]["scenario"]
> shock_dict = scenarios[worst]
> shock_vec = pd.Series(0.0, index=tickers)
> for k, v in shock_dict.items():
>     if k in shock_vec.index:
>         shock_vec.loc[k] = v
> shocked_port = (rets.add(shock_vec, axis=1) * w).sum(axis=1).dropna()
> 
> plt.figure()
> plt.hist(base_r.values, bins=60, alpha=0.6, label="Base")
> plt.hist(shocked_port.values, bins=60, alpha=0.6, label=f"Shocked: {worst}")
> plt.title("Portfolio Daily Returns: Base vs Worst Stress Scenario")
> plt.xlabel("Daily return")
> plt.ylabel("Count")
> plt.legend()
> plt.tight_layout()
> plt.savefig(IMG_DIR / "stress_test_hist_compare.png", dpi=200)
> plt.close()
> 
> print("Saved: images/stress_test_hist_compare.png")
> PY
bash-3.2$ python code/05_stress_test.py
Saved: data/stress_test_summary.csv
                                          scenario  shock_only_portfolio_return  ...  CVaR95_base  CVaR95_shocked
5  All risk-off: equities -8%, bonds -2%, gold +2%                       -0.058  ...     0.014689        0.072689
0                     Equity shock: XIU & VFV -10%                       -0.045  ...     0.014689        0.059689
4                               Bank shock: RY -8%                       -0.008  ...     0.014689        0.022689
1                              Bond shock: ZAG -3%                       -0.006  ...     0.014689        0.020689
3                     International shock: XEF -2%                       -0.003  ...     0.014689        0.017689
2                              Gold rally: GLD +3%                        0.003  ...     0.014689        0.011689

[6 rows x 6 columns]
Saved: images/stress_test_hist_compare.png
bash-3.2$ head -n 10 data/stress_test_summary.csv
scenario,shock_only_portfolio_return,VaR95_base,VaR95_shocked,CVaR95_base,CVaR95_shocked
"All risk-off: equities -8%, bonds -2%, gold +2%",-0.058,0.00973763801953901,0.06773763801953905,0.014688991036429011,0.07268899103642903
Equity shock: XIU & VFV -10%,-0.045000000000000005,0.00973763801953901,0.05473763801953904,0.014688991036429011,0.05968899103642903
Bank shock: RY -8%,-0.008,0.00973763801953901,0.017737638019539037,0.014688991036429011,0.022688991036429025
Bond shock: ZAG -3%,-0.006,0.00973763801953901,0.01573763801953904,0.014688991036429011,0.020688991036429023
International shock: XEF -2%,-0.003,0.00973763801953901,0.012737638019539038,0.014688991036429011,0.017688991036429028
Gold rally: GLD +3%,0.003,0.00973763801953901,0.006737638019539039,0.014688991036429011,0.011688991036429026
bash-3.2$ ls images | grep stress
stress_test_hist_compare.png
bash-3.2$ open images/stress_test_hist_compare.png
bash-3.2$ ls data | grep stress
stress_test_summary.csv
bash-3.2$ ls images | grep stress
stress_test_hist_compare.png
bash-3.2$ head -n 20 data/stress_test_summary.csv
scenario,shock_only_portfolio_return,VaR95_base,VaR95_shocked,CVaR95_base,CVaR95_shocked
"All risk-off: equities -8%, bonds -2%, gold +2%",-0.058,0.00973763801953901,0.06773763801953905,0.014688991036429011,0.07268899103642903
Equity shock: XIU & VFV -10%,-0.045000000000000005,0.00973763801953901,0.05473763801953904,0.014688991036429011,0.05968899103642903
Bank shock: RY -8%,-0.008,0.00973763801953901,0.017737638019539037,0.014688991036429011,0.022688991036429025
Bond shock: ZAG -3%,-0.006,0.00973763801953901,0.01573763801953904,0.014688991036429011,0.020688991036429023
International shock: XEF -2%,-0.003,0.00973763801953901,0.012737638019539038,0.014688991036429011,0.017688991036429028
Gold rally: GLD +3%,0.003,0.00973763801953901,0.006737638019539039,0.014688991036429011,0.011688991036429026
bash-3.2$ cd ~/Desktop/portfolio-risk-war-room
bash-3.2$ git init
xcode-select: note: No developer tools were found, requesting install.
If developer tools are located at a non-default location on disk, use `sudo xcode-select --switch path/to/Xcode.app` to specify the Xcode that you wish to use for command line developer tools, and cancel the installation dialog.
See `man xcode-select` for more details.
bash-3.2$ touch README.md
bash-3.2$ open -a TextEdit README.md
bash-3.2$ cd ~/Desktop/portfolio-risk-war-room
bash-3.2$ git init
xcode-select: note: No developer tools were found, requesting install.
If developer tools are located at a non-default location on disk, use `sudo xcode-select --switch path/to/Xcode.app` to specify the Xcode that you wish to use for command line developer tools, and cancel the installation dialog.
See `man xcode-select` for more details.
bash-3.2$ git --version
git version 2.50.1 (Apple Git-155)
bash-3.2$ git init
Initialized empty Git repository in /Users/sijiama/Desktop/portfolio-risk-war-room/.git/
bash-3.2$ cat > .gitignore << 'EOF'
> # Python
> __pycache__/
> *.pyc
> .ipynb_checkpoints/
> 
> # macOS
> .DS_Store
> 
> # env / secrets
> .env
> .venv/
> venv/
> 
> # editor
> .vscode/
> 
> # raw data (keep results, ignore raw)
> data/prices.csv
> data/prices_clean.csv
> EOF
bash-3.2$ git status
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.gitignore
	README.md
	code/
	data/
	images/

nothing added to commit but untracked files present (use "git add" to track)
bash-3.2$ git check-ignore -v data/prices.csv data/prices_clean.csv
.gitignore:18:data/prices.csv	data/prices.csv
.gitignore:19:data/prices_clean.csv	data/prices_clean.csv
bash-3.2$ git add -n .
add '.gitignore'
add 'README.md'
add 'code/01_download_data.py'
add 'code/01_download_data_stooq.py'
add 'code/01_download_data_stooq_csv.py'
add 'code/01_download_data_yahoo_csv.py'
add 'code/02_make_returns_from_prices.py'
add 'code/03_risk_metrics.py'
add 'code/04_var_cvar.py'
add 'code/05_stress_test.py'
add 'data/corr.csv'
add 'data/portfolio_returns.csv'
add 'data/returns.csv'
add 'data/risk_summary.csv'
add 'data/stress_test_summary.csv'
add 'data/var_backtest.csv'
add 'data/var_cvar_summary.csv'
add 'data/weights.csv'
add 'images/portfolio_drawdown.png'
add 'images/portfolio_nav.png'
add 'images/portfolio_returns_hist_var.png'
add 'images/portfolio_rolling_vol_20d.png'
add 'images/stress_test_hist_compare.png'
bash-3.2$ git add .
bash-3.2$ git status
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   .gitignore
	new file:   README.md
	new file:   code/01_download_data.py
	new file:   code/01_download_data_stooq.py
	new file:   code/01_download_data_stooq_csv.py
	new file:   code/01_download_data_yahoo_csv.py
	new file:   code/02_make_returns_from_prices.py
	new file:   code/03_risk_metrics.py
	new file:   code/04_var_cvar.py
	new file:   code/05_stress_test.py
	new file:   data/corr.csv
	new file:   data/portfolio_returns.csv
	new file:   data/returns.csv
	new file:   data/risk_summary.csv
	new file:   data/stress_test_summary.csv
	new file:   data/var_backtest.csv
	new file:   data/var_cvar_summary.csv
	new file:   data/weights.csv
	new file:   images/portfolio_drawdown.png
	new file:   images/portfolio_nav.png
	new file:   images/portfolio_returns_hist_var.png
	new file:   images/portfolio_rolling_vol_20d.png
	new file:   images/stress_test_hist_compare.png

bash-3.2$ git commit -m "Initial commit: portfolio risk war room"
[main (root-commit) d60f44a] Initial commit: portfolio risk war room
 Committer: sijia ma <sijiama@sijias-MacBook-Air.local>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly. Run the
following command and follow the instructions in your editor to edit
your configuration file:

    git config --global --edit

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 23 files changed, 1446 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 README.md
 create mode 100644 code/01_download_data.py
 create mode 100644 code/01_download_data_stooq.py
 create mode 100644 code/01_download_data_stooq_csv.py
 create mode 100644 code/01_download_data_yahoo_csv.py
 create mode 100644 code/02_make_returns_from_prices.py
 create mode 100644 code/03_risk_metrics.py
 create mode 100644 code/04_var_cvar.py
 create mode 100644 code/05_stress_test.py
 create mode 100644 data/corr.csv
 create mode 100644 data/portfolio_returns.csv
 create mode 100644 data/returns.csv
 create mode 100644 data/risk_summary.csv
 create mode 100644 data/stress_test_summary.csv
 create mode 100644 data/var_backtest.csv
 create mode 100644 data/var_cvar_summary.csv
 create mode 100644 data/weights.csv
 create mode 100644 images/portfolio_drawdown.png
 create mode 100644 images/portfolio_nav.png
 create mode 100644 images/portfolio_returns_hist_var.png
 create mode 100644 images/portfolio_rolling_vol_20d.png
 create mode 100644 images/stress_test_hist_compare.png
bash-3.2$ git remote add origin <https://github.com/sijiama521-ship-it/Portfolio-Risk-War-Room-Dashboard.git>
bash: syntax error near unexpected token `newline'
bash-3.2$ git remote add origin https://github.com/sijiama521-ship-it/Portfolio-Risk-War-Room-Dashboard.git
bash-3.2$ git remote -v
origin	https://github.com/sijiama521-ship-it/Portfolio-Risk-War-Room-Dashboard.git (fetch)
origin	https://github.com/sijiama521-ship-it/Portfolio-Risk-War-Room-Dashboard.git (push)
bash-3.2$ git push -u origin main
Username for 'https://github.com': sijiama521-ship-it
Password for 'https://sijiama521-ship-it@github.com': 
To https://github.com/sijiama521-ship-it/Portfolio-Risk-War-Room-Dashboard.git
 ! [rejected]        main -> main (fetch first)
error: failed to push some refs to 'https://github.com/sijiama521-ship-it/Portfolio-Risk-War-Room-Dashboard.git'
hint: Updates were rejected because the remote contains work that you do not
hint: have locally. This is usually caused by another repository pushing to
hint: the same ref. If you want to integrate the remote changes, use
hint: 'git pull' before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
bash-3.2$ git pull --rebase origin main
remote: Enumerating objects: 6, done.
remote: Counting objects: 100% (6/6), done.
remote: Compressing objects: 100% (4/4), done.
remote: Total 6 (delta 0), reused 0 (delta 0), pack-reused 0 (from 0)
Unpacking objects: 100% (6/6), 2.54 KiB | 370.00 KiB/s, done.
From https://github.com/sijiama521-ship-it/Portfolio-Risk-War-Room-Dashboard
 * branch            main       -> FETCH_HEAD
 * [new branch]      main       -> origin/main
Auto-merging README.md
CONFLICT (add/add): Merge conflict in README.md
error: could not apply d60f44a... Initial commit: portfolio risk war room
hint: Resolve all conflicts manually, mark them as resolved with
hint: "git add/rm <conflicted_files>", then run "git rebase --continue".
hint: You can instead skip this commit: run "git rebase --skip".
hint: To abort and get back to the state before "git rebase", run "git rebase --abort".
hint: Disable this message with "git config set advice.mergeConflict false"
Could not apply d60f44a... # Initial commit: portfolio risk war room
bash-3.2$ open README.md
bash-3.2$ git add README.md
bash-3.2$ git rebase --continue
[detached HEAD 01e7c5e] Initial commit: portfolio risk war room
 Committer: sijia ma <sijiama@sijias-MacBook-Air.local>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly. Run the
following command and follow the instructions in your editor to edit
your configuration file:

    git config --global --edit

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 23 files changed, 1368 insertions(+), 2 deletions(-)
 create mode 100644 .gitignore
 create mode 100644 code/01_download_data.py
 create mode 100644 code/01_download_data_stooq.py
 create mode 100644 code/01_download_data_stooq_csv.py
 create mode 100644 code/01_download_data_yahoo_csv.py
 create mode 100644 code/02_make_returns_from_prices.py
 create mode 100644 code/03_risk_metrics.py
 create mode 100644 code/04_var_cvar.py
 create mode 100644 code/05_stress_test.py
 create mode 100644 data/corr.csv
 create mode 100644 data/portfolio_returns.csv
 create mode 100644 data/returns.csv
 create mode 100644 data/risk_summary.csv
 create mode 100644 data/stress_test_summary.csv
 create mode 100644 data/var_backtest.csv
 create mode 100644 data/var_cvar_summary.csv
 create mode 100644 data/weights.csv
 create mode 100644 images/portfolio_drawdown.png
 create mode 100644 images/portfolio_nav.png
 create mode 100644 images/portfolio_returns_hist_var.png
 create mode 100644 images/portfolio_rolling_vol_20d.png
 create mode 100644 images/stress_test_hist_compare.png
Successfully rebased and updated refs/heads/main.
bash-3.2$ git status
On branch main
nothing to commit, working tree clean
bash-3.2$ git push -u origin main
Enumerating objects: 30, done.
Counting objects: 100% (30/30), done.
Delta compression using up to 10 threads
Compressing objects: 100% (28/28), done.
Writing objects: 100% (28/28), 320.43 KiB | 26.70 MiB/s, done.
Total 28 (delta 3), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (3/3), completed with 1 local object.
To https://github.com/sijiama521-ship-it/Portfolio-Risk-War-Room-Dashboard.git
   93845c1..01e7c5e  main -> main
branch 'main' set up to track 'origin/main'.
bash-3.2$ rm main
bash-3.2$ git add -A
bash-3.2$ git commit -m "remove accidental main file"
[main 5cd4e92] remove accidental main file
 Committer: sijia ma <sijiama@sijias-MacBook-Air.local>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly. Run the
following command and follow the instructions in your editor to edit
your configuration file:

    git config --global --edit

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 1 file changed, 1 deletion(-)
 delete mode 100644 main
bash-3.2$ git push
Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Delta compression using up to 10 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (2/2), 239 bytes | 239.00 KiB/s, done.
Total 2 (delta 1), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (1/1), completed with 1 local object.
To https://github.com/sijiama521-ship-it/Portfolio-Risk-War-Room-Dashboard.git
   01e7c5e..5cd4e92  main -> main
bash-3.2$ touch code/06_var_backtest_plot.py
bash-3.2$ open -a TextEdit code/06_var_backtest_plot.py
bash-3.2$ python code/06_var_backtest_plot.py
Saved: images/var_backtest_95.png
Saved: images/var_backtest_99.png
Saved: data/var_backtest_summary.csv
bash-3.2$ mkdir -p report
bash-3.2$ nano report/REPORT.md

  UW PICO 5.09                                     File: report/REPORT.md                                     Modified  

   
Interpretation:
- If the breach rate is close to expected (e.g., ~5% for VaR95, ~1% for VaR99), the VaR model is directionally reasonab$
- If breach rate is too high, VaR may underestimate tail risk (distribution changed, volatility regime shift, fat tails$
- If breach rate is too low, VaR may be overly conservative.

Limitations:
- Historical VaR assumes the past resembles the future.
- Parametric VaR assumes normality (often violated in real markets).
- Small sample size and regime changes can distort results.

---

## 7) Conclusion (What this dashboard tells us)
This “Risk War Room” provides a complete workflow from price ingestion → return construction → risk metrics → tail r$   

In practice, it helps a portfolio owner:
- understand baseline risk and drawdown behavior,
- quantify tail losses using VaR/CVaR,
- identify the most damaging shock scenarios,
- and validate whether VaR exceedances occur at a reasonable frequency.

Overall, the project demonstrates both **risk measurement** and **risk validation** (stress testing + backtest), which $

   

^G Get Help         ^O WriteOut         ^R Read File        ^Y Prev Pg          ^K Cut Text         ^C Cur Pos          
^X Exit             ^J Justify          ^W Where is         ^V Next Pg          ^U UnCut Text       ^T To Spell        
