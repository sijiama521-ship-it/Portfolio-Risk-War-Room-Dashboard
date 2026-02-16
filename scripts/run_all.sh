#!/usr/bin/env bash
set -euo pipefail

echo "== Portfolio Risk War Room: Run All =="
echo "Working dir: $(pwd)"

# Ensure output folders exist
mkdir -p outputs/tables outputs/figures report

run_if_exists () {
  local cmd="$1"
  local label="$2"
  echo ""
  echo "---- $label ----"
  echo "$cmd"
  eval "$cmd"
}

# Step 2: VaR series + backtests (try common filenames)
if [[ -f "src/run_normal_var_series.py" ]]; then
  run_if_exists "python src/run_normal_var_series.py" "Normal VaR series"
fi

if [[ -f "src/run_hist_var_series.py" ]]; then
  run_if_exists "python src/run_hist_var_series.py" "Historical VaR series"
fi

# EWMA (some repos use run_ewma_backtest.py or similar)
if [[ -f "src/run_ewma_backtest.py" ]]; then
  run_if_exists "python -m src.run_ewma_backtest" "EWMA VaR backtest"
fi

# Generic backtest runner (some repos have run_backtest.py)
if [[ -f "src/run_backtest.py" ]]; then
  run_if_exists "python -m src.run_backtest" "Backtest summary"
fi

# Step 3: Historical scenarios
if [[ -f "src/run_historical_scenarios.py" ]]; then
  run_if_exists "python -m src.run_historical_scenarios" "Historical stress scenarios"
fi

# Step 4: Risk contribution
if [[ -f "src/risk_contribution.py" ]]; then
  run_if_exists "python src/risk_contribution.py" "Risk contribution"
fi

# Step 5: Report generation (try a few common entrypoints)
if [[ -f "report/generate_report.py" ]]; then
  run_if_exists "python report/generate_report.py" "Generate REPORT.md"
elif [[ -f "src/generate_report.py" ]]; then
  run_if_exists "python src/generate_report.py" "Generate REPORT.md"
elif python -c "import importlib; importlib.import_module('report.generate_report')" >/dev/null 2>&1; then
  run_if_exists "python -m report.generate_report" "Generate REPORT.md"
elif [[ -f "src/build_report.py" ]]; then
  run_if_exists "python -m src.build_report" "Generate REPORT.md"
else
  echo ""
  echo "⚠️  Report generator not found (skipping)."
  echo "   If you have a report script with a different name, tell me its path and I'll plug it in."
fi

echo ""
echo "✅ Done. Outputs should be in outputs/tables, outputs/figures, and report/REPORT.md"
