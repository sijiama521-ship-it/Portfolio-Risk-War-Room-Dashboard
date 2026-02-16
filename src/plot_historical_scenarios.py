import pandas as pd
import matplotlib.pyplot as plt

PORTFOLIO_PATH = "data/portfolio_returns.csv"
SCEN_PATH = "outputs/tables/historical_scenarios.csv"
OUT_PATH = "outputs/figures/historical_scenarios.png"

def main():
    df = pd.read_csv(PORTFOLIO_PATH)

    # your file uses "Date"
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # cumulative NAV (start at 1.0)
    df["nav"] = (1 + df["portfolio_return"]).cumprod()

    scen = pd.read_csv(SCEN_PATH)
    scen["start_date"] = pd.to_datetime(scen["start_date"])
    scen["end_date"] = pd.to_datetime(scen["end_date"])

    fig = plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["nav"], label="Portfolio NAV")

    # shade each scenario window
    for _, row in scen.iterrows():
        plt.axvspan(row["start_date"], row["end_date"], alpha=0.2, label=row["scenario_name"])

    plt.title("Historical Scenario Stress Test (In-sample windows)")
    plt.xlabel("Date")
    plt.ylabel("NAV (cumulative)")
    plt.legend(loc="best")
    plt.tight_layout()

    # ensure folder exists
    import os
    os.makedirs("outputs/figures", exist_ok=True)

    plt.savefig(OUT_PATH, dpi=200)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
