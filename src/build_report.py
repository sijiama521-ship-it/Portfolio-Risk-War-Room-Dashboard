from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Inputs you already generate
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"

# Output report locations
REPORT_DIR = ROOT / "report"
REPORT_MD = REPORT_DIR / "REPORT.md"
OUTPUTS_MD = ROOT / "outputs" / "report.md"


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def df_to_md_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    """
    Convert DataFrame to a GitHub-friendly markdown table
    without requiring extra dependencies.
    """
    if df is None or df.empty:
        return "_(no data)_"

    df_show = df.head(max_rows).copy()

    # stringify values cleanly
    for c in df_show.columns:
        if pd.api.types.is_float_dtype(df_show[c]):
            df_show[c] = df_show[c].map(lambda x: f"{x:.6g}")
        else:
            df_show[c] = df_show[c].astype(str)

    cols = list(df_show.columns)
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = ""
    for _, row in df_show.iterrows():
        rows += "| " + " | ".join(str(row[c]) for c in cols) + " |\n"

    extra = ""
    if len(df) > max_rows:
        extra = f"\n_(showing first {max_rows} rows of {len(df)})_\n"
    return header + sep + rows + extra


def relpath_from_report(target: Path) -> str:
    # report/REPORT.md -> need paths like ../outputs/figures/xxx.png
    return os.path.relpath(target, REPORT_DIR).replace("\\", "/")


def section(title: str) -> str:
    return f"\n## {title}\n\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tables
    risk_summary = read_csv_if_exists(ROOT / "data" / "risk_summary.csv")
    var_backtest_summary = read_csv_if_exists(TABLES / "var_backtest_summary.csv")
    var_kupiec = read_csv_if_exists(TABLES / "var_backtest_kupiec.csv")
    historical = read_csv_if_exists(TABLES / "historical_scenarios.csv")
    risk_contrib = read_csv_if_exists(TABLES / "risk_contribution.csv")

    # Figure paths
    fig_var_compare = FIGS / "var_compare.png"
    fig_kupiec = FIGS / "kupiec_pvalues.png"
    fig_breach = FIGS / "var_backtest_breach_rate.png"
    fig_top_risk = FIGS / "top_risk_contributors.png"

    # Compose markdown
    md = []
    md.append("# Portfolio Risk War Room — Risk Report\n")
    md.append(
        "This report is auto-generated from `outputs/tables` and `outputs/figures`.\n"
    )

    # 1) Portfolio overview / performance
    md.append(section("1) Portfolio overview & performance snapshot"))
    md.append("Key performance / risk metrics (from `data/risk_summary.csv`):\n\n")
    md.append(df_to_md_table(risk_summary, max_rows=20))
    md.append("\n")

    # 2) VaR comparison
    md.append(section("2) VaR model comparison (Hist vs Normal vs EWMA)"))
    if fig_var_compare.exists():
        md.append(f"![VaR Comparison]({relpath_from_report(fig_var_compare)})\n")
    else:
        md.append("_(missing figure: outputs/figures/var_compare.png)_\n")

    # 3) Backtest
    md.append(section("3) VaR backtest results"))
    md.append("Backtest summary:\n\n")
    md.append(df_to_md_table(var_backtest_summary, max_rows=30))
    md.append("\n\nKupiec test results:\n\n")
    md.append(df_to_md_table(var_kupiec, max_rows=30))
    md.append("\n")

    if fig_kupiec.exists():
        md.append(f"\n![Kupiec p-values]({relpath_from_report(fig_kupiec)})\n")
    else:
        md.append("\n_(missing figure: outputs/figures/kupiec_pvalues.png)_\n")

    if fig_breach.exists():
        md.append(f"\n![Breach rate]({relpath_from_report(fig_breach)})\n")
    else:
        md.append(
            "\n_(missing figure: outputs/figures/var_backtest_breach_rate.png)_\n"
        )

    # 4) Stress scenarios
    md.append(section("4) Historical stress scenarios"))
    md.append("Historical scenario results:\n\n")
    md.append(df_to_md_table(historical, max_rows=20))
    md.append("\n")

    # 5) Risk contribution
    md.append(section("5) Risk contribution (who drives portfolio risk?)"))
    md.append("Risk contribution table:\n\n")
    md.append(df_to_md_table(risk_contrib, max_rows=20))
    md.append("\n")
    if fig_top_risk.exists():
        md.append(f"\n![Top risk contributors]({relpath_from_report(fig_top_risk)})\n")
    else:
        md.append("\n_(missing figure: outputs/figures/top_risk_contributors.png)_\n")

    # Write report
    content = "".join(md)
    REPORT_MD.write_text(content, encoding="utf-8")
    OUTPUTS_MD.write_text(content.replace("../outputs/", "outputs/"), encoding="utf-8")

    print(f"Saved: {REPORT_MD}")
    print(f"Saved: {OUTPUTS_MD}")


if __name__ == "__main__":
    main()
