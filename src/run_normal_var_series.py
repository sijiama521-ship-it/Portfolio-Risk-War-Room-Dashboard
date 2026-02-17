import os

import numpy as np
import pandas as pd
from scipy.stats import norm


# 你项目里如果有专门读returns的函数，就用你自己的。
# 这里写一个“尽量通用”的读取方式：优先找你现有的 portfolio returns 输出文件。
def load_portfolio_returns() -> pd.Series:
    # 你可以按你项目实际情况改这里的路径
    candidates = [
        "outputs/tables/portfolio_returns.csv",
        "outputs/portfolio_returns.csv",
        "data/portfolio_returns.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # 常见列名尝试
            for col in ["portfolio_return", "returns", "return", "ret"]:
                if col in df.columns:
                    s = df[col].astype(float)
                    return s.reset_index(drop=True)

            # 如果只有一列，就当作returns
            if df.shape[1] == 1:
                return df.iloc[:, 0].astype(float).reset_index(drop=True)

    raise FileNotFoundError(
        "找不到 portfolio returns 文件。请把你 Step1 用的 returns 输入文件路径/列名告诉我，"
        "或把你 run_backtest.py 里读returns的那段代码发我，我给你改成完全匹配你项目的版本。"
    )


def main():
    returns = load_portfolio_returns()

    # 用全样本均值、波动率（静态 parametric normal VaR）
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))

    z_95 = norm.ppf(0.05)  # 左尾 5%
    z_99 = norm.ppf(0.01)  # 左尾 1%

    # 你的项目里 VaR 一直用“负号+阈值”那套也行
    # 这里输出和你 hist/ewma 一致：阈值（通常为负数）
    var_95 = mu + z_95 * sigma
    var_99 = mu + z_99 * sigma

    out = pd.DataFrame(
        {
            "portfolio_return": returns,
            "VaR_normal_95": np.full(len(returns), var_95),
            "VaR_normal_99": np.full(len(returns), var_99),
        }
    )

    os.makedirs("outputs/tables", exist_ok=True)
    out_path = "outputs/tables/var_normal_series.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(out.tail(3))


if __name__ == "__main__":
    main()
