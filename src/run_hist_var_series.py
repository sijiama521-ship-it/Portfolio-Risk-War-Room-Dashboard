import os

import pandas as pd


def main():
    # 1) load realized portfolio returns
    ret_path = "data/portfolio_returns.csv"
    rets = pd.read_csv(ret_path)
    r = rets["portfolio_return"].astype(float)

    # 2) rolling historical VaR (choose a window)
    window = 60  # 约3个月交易日；你也可以改 90/120
    # 你需要的两个置信度（跟你之前一致）
    confidences = [0.95, 0.99]

    out = pd.DataFrame({"portfolio_return": r})

    for conf in confidences:
        alpha = 1.0 - conf
        # rolling quantile: 例如 alpha=0.05 -> 5%分位数（通常是负的）
        q = r.rolling(window=window, min_periods=window).quantile(alpha)
        # 把 VaR 写成正的“损失阈值”也行；但你项目里一直用负阈值比较 r < var_thr
        # 为了和你现有 breach_mask (r < var_thr) 一致，这里保留为“负阈值”
        out[f"VaR_hist_{int(conf*100)}"] = q

    os.makedirs("outputs/tables", exist_ok=True)
    out_path = "outputs/tables/var_hist_series.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(out.tail(3))


if __name__ == "__main__":
    main()
