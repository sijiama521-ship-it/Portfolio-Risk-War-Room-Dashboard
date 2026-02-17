# app.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf


# =========================
# Paths
# =========================
WEIGHTS_PATH = "data/weights.csv"

OUT_TABLES_DIR = "outputs/tables"
OUT_FIGS_DIR = "outputs/figures"
OUT_REPORT_MD = "outputs/REPORT.md"
ROOT_REPORT_MD = "REPORT.md"
REPORT_DIR_MD = "report/REPORT.md"


# =========================
# Utilities
# =========================
def ensure_dirs() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs(OUT_TABLES_DIR, exist_ok=True)
    os.makedirs(OUT_FIGS_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("report", exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def stable_hash_df(df: pd.DataFrame) -> str:
    # stable-ish cache key
    return str(pd.util.hash_pandas_object(df, index=True).sum())


def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df[df["ticker"] != ""]
    s = df["weight"].sum()
    if s <= 0:
        # fallback equal weights
        if len(df) == 0:
            return pd.DataFrame({"ticker": [], "weight": []})
        df["weight"] = 1.0 / len(df)
    else:
        df["weight"] = df["weight"] / s
    return df.reset_index(drop=True)


def load_weights() -> pd.DataFrame:
    ensure_dirs()
    if not os.path.exists(WEIGHTS_PATH):
        # default example
        df0 = pd.DataFrame(
            {
                "ticker": ["XIU", "VFV", "XEF", "ZAG", "GLD", "RY"],
                "weight": [0.25, 0.20, 0.15, 0.20, 0.10, 0.10],
            }
        )
        df0.to_csv(WEIGHTS_PATH, index=False)
    df = pd.read_csv(WEIGHTS_PATH)
    if "ticker" not in df.columns or "weight" not in df.columns:
        raise ValueError("data/weights.csv must have columns: ticker, weight")
    return normalize_weights(df[["ticker", "weight"]])


def save_weights(df: pd.DataFrame) -> None:
    ensure_dirs()
    df = normalize_weights(df)
    df.to_csv(WEIGHTS_PATH, index=False)


@st.cache_data(show_spinner=False)
def fetch_prices(tickers: Tuple[str, ...], start: str) -> pd.DataFrame:
    # yfinance returns multi-index columns sometimes
    data = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise RuntimeError("Failed to download price data (yfinance returned empty).")
    # prefer Close/Adj Close depending on structure
    if isinstance(data.columns, pd.MultiIndex):
        # (field, ticker)
        if ("Close" in data.columns.get_level_values(0)):
            px = data["Close"]
        else:
            # fallback: first level might contain "Adj Close"
            try:
                px = data["Adj Close"]
            except Exception:
                px = data.xs(data.columns.levels[0][0], axis=1, level=0)
    else:
        # single ticker returns a Series-like DF with columns like 'Close'
        if "Close" in data.columns:
            px = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            # last fallback
            px = data.copy()
            if px.shape[1] == 1:
                px.columns = [tickers[0]]
    px = px.dropna(how="all")
    px = px.ffill().dropna()
    # make sure columns are tickers
    px.columns = [str(c).upper() for c in px.columns]
    return px


def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    r = px.pct_change().dropna()
    return r


def portfolio_returns(r: pd.DataFrame, w: pd.Series) -> pd.Series:
    # align
    cols = [c for c in r.columns if c in w.index]
    r2 = r[cols]
    w2 = w.loc[cols]
    return (r2 * w2.values).sum(axis=1)


def var_normal_series(port_r: pd.Series, z: float = 1.645) -> pd.Series:
    # rolling normal VaR (parametric) using rolling std and mean
    mu = port_r.rolling(60).mean()
    sig = port_r.rolling(60).std(ddof=1)
    # VaR as positive number
    var = -(mu - z * sig)
    return var.dropna()


def var_historical_series(port_r: pd.Series, q: float = 0.05) -> pd.Series:
    # rolling historical VaR (quantile of returns)
    var = -port_r.rolling(60).quantile(q)
    return var.dropna()


def backtest_summary(port_r: pd.Series, var_series: pd.Series) -> pd.DataFrame:
    # Align and count breaches
    df = pd.DataFrame({"ret": port_r, "VaR": var_series}).dropna()
    breaches = (df["ret"] < -df["VaR"]).astype(int)
    out = pd.DataFrame(
        {
            "start": [df.index.min().date()],
            "end": [df.index.max().date()],
            "n_obs": [len(df)],
            "breaches": [int(breaches.sum())],
            "breach_rate": [float(breaches.mean())],
            "avg_VaR": [float(df["VaR"].mean())],
            "avg_ret": [float(df["ret"].mean())],
            "vol_ret": [float(df["ret"].std(ddof=1))],
        }
    )
    return out


def worst_windows(port_r: pd.Series) -> pd.DataFrame:
    # worst day and worst rolling windows (20D/60D)
    df = pd.DataFrame({"port_ret": port_r}).dropna().copy()
    df["roll20"] = df["port_ret"].rolling(20).sum()
    df["roll60"] = df["port_ret"].rolling(60).sum()

    worst_day = df["port_ret"].idxmin()
    worst_20 = df["roll20"].idxmin()
    worst_60 = df["roll60"].idxmin()

    def window_row(name: str, idx: pd.Timestamp, win: int) -> Dict:
        start = idx - pd.Timedelta(days=win * 2)  # loose
        end = idx
        # better: use rolling window indices
        # find actual window slice
        pos = df.index.get_loc(idx)
        start_i = max(0, pos - win + 1)
        win_idx = df.index[start_i : pos + 1]
        wstart = win_idx.min()
        wend = win_idx.max()
        total = float(df.loc[win_idx, "port_ret"].sum())
        return {
            "scenario_name": name,
            "start_date": wstart.date(),
            "end_date": wend.date(),
            "rows_in_window": len(win_idx),
            "worst_day": worst_day.date(),
            "worst_day_return": float(df.loc[worst_day, "port_ret"]),
            "window_total_return": total,
        }

    out = pd.DataFrame(
        [
            window_row("Worst 20D Window (in-sample)", worst_20, 20),
            window_row("Worst 60D Window (in-sample)", worst_60, 60),
            window_row("Worst Day Context (±10d)", worst_day, 21),
        ]
    )
    return out


def risk_contribution(r: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    # align to returns columns
    asset_cols = [c for c in r.columns if c in w.index]
    R = r[asset_cols].dropna()
    wv = w.loc[asset_cols].values.reshape(-1, 1)  # (n,1)

    Sigma = np.cov(R.values, rowvar=False, ddof=1)  # (n,n)

    # scalar variance (fix: ensure scalar)
    port_var = float((wv.T @ Sigma @ wv).item())
    port_vol = float(np.sqrt(port_var))

    # annualize using sqrt(252)
    ann = np.sqrt(252.0)
    asset_vol_ann = np.sqrt(np.diag(Sigma)) * ann
    port_vol_ann = port_vol * ann

    # marginal contribution to VOL: (Sigma w)_i / port_vol
    mrc = (Sigma @ wv) / port_vol  # (n,1)
    crc = wv * mrc  # (n,1)
    vol_contrib_pct = (crc / crc.sum()) * 100.0

    out = pd.DataFrame(
        {
            "asset": asset_cols,
            "weight": w.loc[asset_cols].values,
            "asset_vol_ann": asset_vol_ann,
            "portfolio_vol_ann": [port_vol_ann] * len(asset_cols),
            "vol_contribution_pct": vol_contrib_pct.flatten(),
        }
    )
    out = out.sort_values("vol_contribution_pct", ascending=False).reset_index(drop=True)
    return out


def plot_top_risk_contrib(df_rc: pd.DataFrame, outpath: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(df_rc["asset"], df_rc["vol_contribution_pct"])
    ax.set_title("Top Risk Contributors (Vol Contribution %)")
    ax.set_ylabel("Vol Contribution (%)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


@dataclass
class Artifacts:
    prices: pd.DataFrame
    returns: pd.DataFrame
    weights: pd.DataFrame
    port_returns: pd.Series
    var_normal: pd.Series
    var_hist: pd.Series
    backtest: pd.DataFrame
    scenarios: pd.DataFrame
    risk_contrib: pd.DataFrame
    fig_risk_contrib_path: str
    report_md: str


def build_report_md(a: Artifacts) -> str:
    w = a.weights
    rc = a.risk_contrib
    bt = a.backtest
    sc = a.scenarios

    lines = []
    lines.append(f"# Portfolio Risk War Room Report\n")
    lines.append(f"_Generated: {now_str()}_\n")
    lines.append("## Portfolio Weights\n")
    lines.append(w.to_markdown(index=False))
    lines.append("\n\n## Backtest Summary (Normal VaR)\n")
    lines.append(bt.to_markdown(index=False))
    lines.append("\n\n## Historical Scenarios\n")
    lines.append(sc.to_markdown(index=False))
    lines.append("\n\n## Risk Contribution\n")
    lines.append(rc.to_markdown(index=False))
    lines.append("\n\n## Figures\n")
    lines.append(f"![Top Risk Contributors]({a.fig_risk_contrib_path})\n")
    return "\n".join(lines)


def write_artifacts(a: Artifacts) -> None:
    ensure_dirs()

    # tables
    a.weights.to_csv(f"{OUT_TABLES_DIR}/weights_used.csv", index=False)
    a.backtest.to_csv(f"{OUT_TABLES_DIR}/var_backtest_summary.csv", index=False)
    a.var_normal.rename("var_normal").to_csv(f"{OUT_TABLES_DIR}/var_normal_series.csv")
    a.var_hist.rename("var_hist").to_csv(f"{OUT_TABLES_DIR}/var_hist_series.csv")
    a.scenarios.to_csv(f"{OUT_TABLES_DIR}/historical_scenarios.csv", index=False)
    a.risk_contrib.to_csv(f"{OUT_TABLES_DIR}/risk_contribution.csv", index=False)

    # report markdown (write to multiple places to avoid missing)
    for path in [OUT_REPORT_MD, ROOT_REPORT_MD, REPORT_DIR_MD]:
        with open(path, "w", encoding="utf-8") as f:
            f.write(a.report_md)


def compute_all(weights_df: pd.DataFrame, start: str = "2024-01-01") -> Artifacts:
    ensure_dirs()

    weights_df = normalize_weights(weights_df)
    tickers = tuple(weights_df["ticker"].tolist())
    w = pd.Series(weights_df["weight"].values, index=weights_df["ticker"].values)

    px = fetch_prices(tickers, start=start)
    r = daily_returns(px)

    # portfolio returns
    pr = portfolio_returns(r, w)

    # VaR series
    vn = var_normal_series(pr)
    vh = var_historical_series(pr)

    # backtest normal VaR
    bt = backtest_summary(pr, vn)

    # scenarios
    sc = worst_windows(pr)

    # risk contribution
    rc = risk_contribution(r, w)

    # plot
    fig_path = f"{OUT_FIGS_DIR}/top_risk_contributors.png"
    plot_top_risk_contrib(rc, fig_path)

    # report
    artifacts = Artifacts(
        prices=px,
        returns=r,
        weights=weights_df,
        port_returns=pr,
        var_normal=vn,
        var_hist=vh,
        backtest=bt,
        scenarios=sc,
        risk_contrib=rc,
        fig_risk_contrib_path=fig_path,
        report_md="",
    )
    artifacts.report_md = build_report_md(artifacts)

    # write to disk (so Streamlit Cloud pages read the same outputs)
    write_artifacts(artifacts)

    return artifacts


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Portfolio Risk War Room Dashboard", layout="wide")

ensure_dirs()

st.title("📊 Portfolio Risk War Room Dashboard")
st.caption("A lightweight dashboard to view risk metrics, VaR models, backtests, stress scenarios, and risk contribution outputs.")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Report", "Tables", "Figures"], index=0)

st.sidebar.header("Controls")
if st.sidebar.button("🧹 Clear cache + reload"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("⚖️ Portfolio Weights (edit)")
weights_df = load_weights()

edited = st.sidebar.data_editor(
    weights_df,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    key="weights_editor",
)

edited = normalize_weights(edited)
st.sidebar.caption(f"Current sum of weights = {edited['weight'].sum():.6f}")

colA, colB = st.sidebar.columns(2)
if colA.button("💾 Save\nweights.csv"):
    save_weights(edited)
    st.sidebar.success("Saved to data/weights.csv")

if colB.button("🚀 Save +\nRerun pipeline"):
    save_weights(edited)
    with st.spinner("Recomputing outputs (tables/figures/report)..."):
        a = compute_all(edited)
    st.sidebar.success("Done! Outputs + REPORT.md updated.")
    st.rerun()

# Auto compute on first load / if outputs missing
def outputs_ready() -> bool:
    needed = [
        OUT_REPORT_MD,
        f"{OUT_TABLES_DIR}/risk_contribution.csv",
        f"{OUT_FIGS_DIR}/top_risk_contributors.png",
    ]
    return all(os.path.exists(p) for p in needed)

if "bootstrapped" not in st.session_state:
    st.session_state["bootstrapped"] = True
    if not outputs_ready():
        with st.spinner("Initializing outputs for first run..."):
            compute_all(weights_df)

# Always load the latest artifacts from disk for display (single source of truth)
def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

if page == "Report":
    # Prefer root REPORT.md then outputs/REPORT.md
    md_path_candidates = [ROOT_REPORT_MD, OUT_REPORT_MD, REPORT_DIR_MD]
    md_path = next((p for p in md_path_candidates if os.path.exists(p)), None)

    st.header("🧾 Report")
    if md_path is None:
        st.warning("Missing report file.")
    else:
        with open(md_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        st.caption(f"Loaded: `{md_path}` (mtime: {time.ctime(os.path.getmtime(md_path))})")
        st.markdown(report_text)

elif page == "Tables":
    st.header("📋 Tables")

    tabs = st.tabs(
        [
            "risk_contribution.csv",
            "var_backtest_summary.csv",
            "var_normal_series.csv",
            "var_hist_series.csv",
            "historical_scenarios.csv",
            "weights_used.csv",
        ]
    )

    table_map = {
        "risk_contribution.csv": f"{OUT_TABLES_DIR}/risk_contribution.csv",
        "var_backtest_summary.csv": f"{OUT_TABLES_DIR}/var_backtest_summary.csv",
        "var_normal_series.csv": f"{OUT_TABLES_DIR}/var_normal_series.csv",
        "var_hist_series.csv": f"{OUT_TABLES_DIR}/var_hist_series.csv",
        "historical_scenarios.csv": f"{OUT_TABLES_DIR}/historical_scenarios.csv",
        "weights_used.csv": f"{OUT_TABLES_DIR}/weights_used.csv",
    }

    for tab, name in zip(tabs, table_map.keys()):
        with tab:
            path = table_map[name]
            if not os.path.exists(path):
                st.warning(f"Missing `{path}` — click **Save + Rerun pipeline**")
            else:
                st.caption(f"`{path}` (mtime: {time.ctime(os.path.getmtime(path))})")
                df = read_csv_safe(path)
                st.dataframe(df, use_container_width=True)

elif page == "Figures":
    st.header("🖼️ Figures")

    fig_path = f"{OUT_FIGS_DIR}/top_risk_contributors.png"
    if not os.path.exists(fig_path):
        st.warning(f"Missing `{fig_path}` — click **Save + Rerun pipeline**")
    else:
        st.caption(f"`{fig_path}` (mtime: {time.ctime(os.path.getmtime(fig_path))})")
        st.image(fig_path, use_container_width=True)

