import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    path = "outputs/tables/var_backtest_kupiec.csv"
    df = pd.read_csv(path)

    os.makedirs("outputs/figures", exist_ok=True)

    # Plot: confidence vs breach_rate and expected_breaches/N
    x = df["confidence"]
    breach_rate = df["breach_rate"]
    expected_rate = df["expected_breaches"] / df["N"]

    plt.figure()
    plt.plot(x, breach_rate, marker="o", label="Actual breach rate")
    plt.plot(x, expected_rate, marker="o", label="Expected breach rate (alpha)")
    plt.xlabel("Confidence level")
    plt.ylabel("Breach rate")
    plt.title("VaR Backtest (Kupiec inputs): Actual vs Expected Breach Rate")
    plt.legend()
    out_path = "outputs/figures/var_backtest_breach_rate.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Plot: confidence vs kupiec_pvalue
    plt.figure()
    plt.plot(x, df["kupiec_pvalue"], marker="o")
    plt.axhline(0.05, linestyle="--")
    plt.xlabel("Confidence level")
    plt.ylabel("Kupiec p-value")
    plt.title("Kupiec POF Test p-values (higher = better)")
    out_path2 = "outputs/figures/kupiec_pvalues.png"
    plt.savefig(out_path2, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path2}")

if __name__ == "__main__":
    main()
