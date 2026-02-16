import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    hist_path = "outputs/tables/var_hist_series.csv"
    normal_path = "outputs/tables/var_normal_series.csv"
    ewma_path = "outputs/tables/ewma_var_series.csv"

    hist = pd.read_csv(hist_path)
    normal = pd.read_csv(normal_path)
    ewma = pd.read_csv(ewma_path)

    def pick_var_col(df):
        for c in ["VaR", "var", "VaR_t", "VaR_series", "VaR_threshold"]:
            if c in df.columns:
                return c
        return df.columns[-1]

    hist_col = pick_var_col(hist)
    normal_col = pick_var_col(normal)
    ewma_col = pick_var_col(ewma)

    m = min(len(hist), len(normal), len(ewma))
    hist = hist.iloc[-m:]
    normal = normal.iloc[-m:]
    ewma = ewma.iloc[-m:]

    plt.figure()
    plt.plot(hist[hist_col].values, label="Hist VaR")
    plt.plot(normal[normal_col].values, label="Normal VaR")
    plt.plot(ewma[ewma_col].values, label="EWMA VaR")
    plt.title("VaR Comparison: Hist vs Normal vs EWMA")
    plt.legend()

    os.makedirs("outputs/figures", exist_ok=True)
    out_path = "outputs/figures/var_compare.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
