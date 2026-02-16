import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RETURNS_PATH = "data/returns.csv"
WEIGHTS_PATH = "data/weights.csv"

OUT_TABLE = "outputs/tables/risk_contribution.csv"
OUT_FIG = "outputs/figures/top_risk_contributors.png"

ANNUALIZE = 252  # trading days

def load_weights(path: str) -> pd.Series:
    wdf = pd.read_csv(path)
    # support common formats: (asset, weight) or columns with tickers as header
    if {"asset", "weight"}.issubset(set(map(str.lower, wdf.columns))):
        cols = {c.lower(): c for c in wdf.columns}
        w = pd.Series(wdf[cols["weight"]].values, index=wdf[cols["asset"]].values, dtype=float)
    else:
        # assume single-row weights with columns = tickers
        if len(wdf) != 1:
            raise ValueError("weights.csv format not recognized. Use columns asset,weight OR single-row header=tickers.")
        w = wdf.iloc[0].astype(float)
        w.index = w.index.astype(str)
    w = w / w.sum()
    return w

def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    rets = pd.read_csv(RETURNS_PATH)
    # expect first column is Date, rest are assets (your file: Date, XIU, VFV, XEF, ZAG, GLD, RY)
    if "Date" in rets.columns:
        rets["Date"] = pd.to_datetime(rets["Date"])
        rets = rets.sort_values("Date")
        asset_cols = [c for c in rets.columns if c != "Date"]
    else:
        asset_cols = rets.columns.tolist()

    R = rets[asset_cols].astype(float).dropna()

    w = load_weights(WEIGHTS_PATH)

    # align weights to returns columns
    missing = [a for a in asset_cols if a not in w.index]
    extra = [a for a in w.index if a not in asset_cols]
    if missing:
        raise ValueError(f"weights missing these assets: {missing}")
    if extra:
        # ignore any extra weights not in returns
        w = w.loc[[a for a in w.index if a in asset_cols]]

    w = w.loc[asset_cols].values.reshape(-1, 1)  # column vector

    # covariance matrix (daily)
    Sigma = np.cov(R.values, rowvar=False, ddof=1)

    # portfolio vol (daily)
    port_var = float(w.T @ Sigma @ w)
    port_vol = np.sqrt(port_var)

    # marginal contribution to vol: (Sigma w)_i / port_vol
    mrc = (Sigma @ w) / port_vol  # shape (n,1)

    # component contribution to vol: w_i * mrc_i
    crc = w * mrc  # shape (n,1)

    # convert to % contributions
    vol_contrib_pct = (crc / crc.sum()) * 100.0

    # individual asset vols (annualized, for context)
    asset_vol = np.sqrt(np.diag(Sigma)) * np.sqrt(ANNUALIZE)
    port_vol_ann = port_vol * np.sqrt(ANNUALIZE)

    out = pd.DataFrame({
        "asset": asset_cols,
        "weight": w.flatten(),
        "asset_vol_ann": asset_vol,
        "portfolio_vol_ann": [port_vol_ann] * len(asset_cols),
        "vol_contribution_pct": vol_contrib_pct.flatten()
    }).sort_values("vol_contribution_pct", ascending=False)

    out.to_csv(OUT_TABLE, index=False)
    print(f"Saved: {OUT_TABLE}")
    print(out[["asset", "weight", "vol_contribution_pct"]])

    # plot top contributors
    top = out.head(6).copy()
    plt.figure(figsize=(10, 5))
    plt.bar(top["asset"], top["vol_contribution_pct"])
    plt.title("Top Risk Contributors (Vol Contribution %)")
    plt.xlabel("Asset")
    plt.ylabel("Vol Contribution (%)")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    print(f"Saved: {OUT_FIG}")

if __name__ == "__main__":
    main()
