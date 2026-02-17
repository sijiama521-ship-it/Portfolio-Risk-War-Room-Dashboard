# src/run_historical_scenarios.py

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# -----------------------------
# Config: paths + scenarios
# -----------------------------
PORTFOLIO_RETURNS_PATH = "data/portfolio_returns.csv"  # columns: Date, portfolio_return
ASSET_RETURNS_PATH = (
    "data/returns.csv"  # columns: Date, XIU, VFV, XEF, ZAG, GLD, RY ...
)
WEIGHTS_PATH = "data/weights.csv"  # columns: asset, weight (or Asset, Weight)
OUT_PATH = "outputs/tables/historical_scenarios.csv"

MIN_ROWS_IN_WINDOW = 5  # basic sanity check


SCENARIOS: list[tuple[str, str, str]] = [
    ("Worst 20D Window (in-sample)", "2025-03-12", "2025-04-08"),
    ("Worst 60D Window (in-sample)", "2025-01-14", "2025-04-07"),
    ("Worst Day Context (±10d)", "2025-03-25", "2025-04-15"),
]


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_portfolio_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns or "portfolio_return" not in df.columns:
        raise ValueError(
            f"{path} must have columns ['Date','portfolio_return'], got {df.columns.tolist()}"
        )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_asset_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # accept either Date or date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])
    else:
        raise ValueError(
            f"{path} must have a Date/date column, got {df.columns.tolist()}"
        )
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_weights(path: str) -> pd.Series:
    df = pd.read_csv(path)

    # accept common variants
    col_asset = None
    col_weight = None
    for a in ["asset", "Asset", "ticker", "Ticker"]:
        if a in df.columns:
            col_asset = a
            break
    for w in ["weight", "Weight", "w", "W"]:
        if w in df.columns:
            col_weight = w
            break

    if col_asset is None or col_weight is None:
        raise ValueError(
            f"{path} must have asset/weight columns, got {df.columns.tolist()}"
        )

    s = df.set_index(col_asset)[col_weight].astype(float)
    # normalize just in case
    if s.sum() != 0:
        s = s / s.sum()
    return s


def window_slice_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    w = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()
    return w


def cumulative_return(returns: pd.Series) -> float:
    # assuming returns are simple returns
    return float(np.prod(1.0 + returns.values) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def top_contributors_on_day(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    target_date: pd.Timestamp,
    top_k: int = 3,
) -> str:
    # asset_returns columns include Date + tickers
    row = asset_returns[asset_returns["Date"] == target_date]
    if row.empty:
        return "N/A (no asset returns for worst day)"

    tickers = [c for c in asset_returns.columns if c != "Date"]

    # align weights to tickers; missing weights -> 0
    w = weights.reindex(tickers).fillna(0.0)

    r = row.iloc[0][tickers].astype(float)
    contrib = w.values * r.values  # approx contribution to portfolio return that day

    contrib_s = pd.Series(contrib, index=tickers).sort_values()
    worst = contrib_s.head(top_k)

    # format: "XIU:-0.45%, VFV:-0.31%, ..."
    parts = [f"{k}:{v*100:.2f}%" for k, v in worst.items()]
    return ", ".join(parts)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    pr = load_portfolio_returns(PORTFOLIO_RETURNS_PATH)

    # Optional: for top contributors
    have_assets = os.path.exists(ASSET_RETURNS_PATH) and os.path.exists(WEIGHTS_PATH)
    asset_ret = None
    weights = None
    if have_assets:
        asset_ret = load_asset_returns(ASSET_RETURNS_PATH)
        weights = load_weights(WEIGHTS_PATH)

    # global worst day (for reference)
    worst_idx = pr["portfolio_return"].idxmin()
    worst_day = pr.loc[worst_idx, "Date"]
    worst_day_ret = float(pr.loc[worst_idx, "portfolio_return"])

    print(
        f"data_range: {pr['Date'].min().date()} to {pr['Date'].max().date()} n={len(pr)}"
    )
    print(f"worst_day: {worst_day.date()} return={worst_day_ret}")

    rows = []
    for name, start, end in SCENARIOS:
        w = window_slice_by_date(pr, start, end)

        print(f"{name}: rows_in_window={len(w)} start={start} end={end}")
        if len(w) < MIN_ROWS_IN_WINDOW:
            raise ValueError(
                f"{name} window too short or missing data: {start} to {end}"
            )

        win_rets = w["portfolio_return"].astype(float)
        cum_ret = cumulative_return(win_rets)
        mdd = max_drawdown(win_rets)

        # worst day inside window
        widx = win_rets.idxmin()
        wday = pr.loc[widx, "Date"]
        wday_ret = float(pr.loc[widx, "portfolio_return"])

        top3 = ""
        if have_assets and asset_ret is not None and weights is not None:
            top3 = top_contributors_on_day(asset_ret, weights, wday, top_k=3)
        else:
            top3 = "N/A (returns.csv or weights.csv missing)"

        rows.append(
            {
                "scenario_name": name,
                "start_date": pd.to_datetime(start).date().isoformat(),
                "end_date": pd.to_datetime(end).date().isoformat(),
                "n_days_in_window": int(len(w)),
                "portfolio_return_over_window": cum_ret,
                "max_drawdown_over_window": mdd,
                "worst_day_in_window": wday.date().isoformat(),
                "worst_day_return": wday_ret,
                "top_3_contributors_on_worst_day": top3,
            }
        )

    out = pd.DataFrame(rows)

    ensure_dir(OUT_PATH)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print(out)


if __name__ == "__main__":
    main()
