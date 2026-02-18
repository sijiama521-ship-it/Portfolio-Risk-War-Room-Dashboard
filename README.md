## Portfolio Risk War Room Dashboard

A lightweight **Streamlit** dashboard for monitoring portfolio risk and performance: **VaR models (Historical / Normal / EWMA)**, **backtesting**, **historical stress scenarios**, and **risk contribution**.

 **Live Demo:** https://portfolio-risk-war-room-dashboard-weaztwbghwyhfwqn8v9f3j.streamlit.app/

---

## What this app does

- **Portfolio overview & performance snapshot**
  - Annualized return / volatility
  - VaR / CVaR
  - Max drawdown
- **VaR model comparison**
  - Compare **Historical vs Normal vs EWMA VaR**
  - View VaR series outputs
- **VaR backtesting**
  - Breach counts / breach rate
  - **Kupiec POF test** and **Christoffersen independence test**
  - Summary tables + plots
- **Historical stress scenarios**
  - Worst window / worst day context (when generated)
  - Scenario summary tables + charts
- **Risk contribution**
  - Volatility contribution by asset
  - “Who drives portfolio risk?” ranking + plot
- **Interactive portfolio weights**
  - Adjust weights from the sidebar
  - Normalize to sum to 1
  - Save weights to `data/weights.csv`
  - (Optional) regenerate outputs locally

---

## Models included 

- **Normal VaR:** assumes returns are normally distributed using mean/vol estimates.
- **Historical VaR:** uses empirical quantiles of historical returns.
- **EWMA VaR:** volatility estimated via exponentially weighted moving average.

Backtests validate whether breaches match expected frequency and whether breaches are independent.

---

## Project structure (key files)

- `app.py` — Streamlit UI + visualization layer  
- `src/` — analytics scripts (VaR series, EWMA, backtest tests, stress, risk contribution, report builder)
- `scripts/run_all.sh` — runs the full pipeline and writes outputs
- `data/weights.csv` — portfolio weights (editable)
- `outputs/`
  - `outputs/tables/` — generated CSV tables
  - `outputs/figures/` — generated PNG charts
- `report/REPORT.md` — auto-generated risk report (rendered inside the app)

---

## Requirements

- Python **3.10+** recommended  
- macOS / Windows / Linux

Install dependencies:

```bash
pip install -r requirements.txt

