# Portfolio Risk War Room — 1-Page Risk Report

**Project:** Portfolio Risk War Room (Python)  
**Portfolio (tickers):** XIU, VFV, XEF, ZAG, GLD, RY  
**Outputs:** CSV tables in `data/` and charts in `images/`  
**Key files:** `data/risk_summary.csv`, `data/var_cvar_summary.csv`, `data/stress_test_summary.csv`, `data/var_backtest_summary.csv`

---

## 1) Objective (What this “War Room” is for)

This project builds a compact risk dashboard for a multi-asset ETF portfolio.  
It converts price history into daily returns, then summarizes:

- **Core risk/return** (annualized return & volatility)
- **Diversification** (correlations)
- **Tail risk** (VaR/CVaR at 95% and 99%)
- **Stress scenarios** (“what-if shocks” to simulate risk-off conditions)
- **VaR backtesting** (how often realized losses exceed the VaR threshold)

The goal is to answer: **“How risky is this portfolio on normal days, and what happens in bad days?”**

---

## 2) Data & Method (How the numbers were produced)

### Data
- Historical daily prices were downloaded and cleaned.
- The cleaned dataset is stored as CSV files under `data/`.

### Returns construction
- Daily returns are computed from prices (percentage change).
- Portfolio daily return is computed as a **weighted sum of asset returns** using `data/weights.csv`.

### Risk metrics
The dashboard computes commonly used risk metrics, including:
- **Annualized volatility** from daily return standard deviation
- **Drawdown** based on portfolio NAV path
- **Correlation matrix** to show diversification relationships

### VaR / CVaR
VaR and CVaR are computed at **95%** and **99%** confidence levels.
- **VaR**: the loss threshold not expected to be exceeded more than (1 − confidence) of the time
- **CVaR (Expected Shortfall)**: the **average loss conditional on exceeding VaR**, capturing tail severity

---

## 3) Tail Risk Results (VaR / CVaR + interpretation)

### What VaR means in this report
- **VaR 95%** is the loss level that should be exceeded on ~**5%** of trading days.
- **VaR 99%** is the loss level that should be exceeded on ~**1%** of trading days.

CVaR is more conservative and better reflects “bad day” magnitude beyond the threshold.

**Files to reference:**
- `data/var_cvar_summary.csv` for the VaR/CVaR table
- `images/portfolio_returns_hist_var.png` for a return distribution view (if included)

---

## 4) Stress Testing (Scenario risk)

Stress testing answers: **“If markets move sharply tomorrow, what happens to the portfolio?”**  
This project includes scenario-style shocks (risk-off style) such as:
- equity down shocks
- rate/bond shocks
- gold rally style shocks
- bank/financial sector stress

**Files to reference:**
- `data/stress_test_summary.csv` (scenario results)
- `images/stress_test_hist_compare.png` (comparison chart, if included)

**Interpretation approach:**
- Compare scenario portfolio return vs normal-day volatility to judge how “extreme” the scenario is.
- Identify which assets contribute most under each shock (equity-heavy vs defensive sleeves).

---

## 5) VaR Backtest (Model validation)

A VaR model is only useful if its “exceedance frequency” is reasonable.

### Backtest definition
A **breach (exceedance)** occurs when:
- daily portfolio return < VaR threshold (a loss larger than VaR prediction)

### Backtest results (from `data/var_backtest_summary.csv`)
| Confidence | Expected breach rate | Observed breach rate | Breaches |
|-----------|-----------------------|----------------------|---------|
| 95%       | 5%                    | ~0.0513              | 20      |
| 99%       | 1%                    | ~0.0103              | 4       |

**Key takeaway:**  
- The **95%** breach rate is close to the expected 5%.  
- The **99%** breach rate is close to the expected 1%.  
Overall, the VaR thresholds appear **well-calibrated** for this sample window.

**Charts to reference:**
- `images/var_backtest_95.png` (95% VaR line + breach points)
- `images/var_backtest_99.png` (99% VaR line + breach points)

---

## 6) Practical Conclusions (What I would tell a PM)

1. **Normal-day risk is summarized well by annualized volatility and drawdown**, and diversification can be inspected via the correlation matrix.
2. **Tail risk is captured using VaR/CVaR at 95% and 99%.** CVaR is the more informative “bad day” metric.
3. **Stress tests provide an intuitive shock framework** (equity down / rate shock / flight-to-safety), making risk communication easier.
4. **VaR backtesting shows the model is behaving reasonably**: breach rates are close to theoretical expectations at both confidence levels.

---

## 7) Limitations & Next Steps

### Limitations
- VaR assumes the future resembles the historical window used.
- Tail events can cluster; risk changes over time (volatility regimes).
- Simple scenario shocks are stylized and may not capture real cross-asset nonlinearities.

### Next steps
- Add rolling-window VaR calibration and regime diagnostics (volatility clustering).
- Add formal VaR tests (Kupiec POF / Christoffersen independence).
- Extend scenarios to historical replay windows (e.g., COVID crash, 2022 rate shock).

---

## Appendix: Where to find outputs
- **Tables (CSV):** `data/`
  - `risk_summary.csv`
  - `var_cvar_summary.csv`
  - `stress_test_summary.csv`
  - `var_backtest_summary.csv`
- **Charts (PNG):** `images/`
  - `portfolio_nav.png`
  - `portfolio_drawdown.png`
  - `portfolio_rolling_vol_20d.png`
  - `stress_test_hist_compare.png`
  - `var_backtest_95.png`
  - `var_backtest_99.png`

