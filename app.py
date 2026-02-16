# app.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
TAB_DIR = OUTPUTS / "tables"
REPORT_MD = ROOT / "report" / "REPORT.md"

st.set_page_config(page_title="Portfolio Risk War Room", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def file_mtime(p: Path) -> str:
    if not p.exists():
        return "missing"
    ts = datetime.fromtimestamp(p.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def newest_mtime(paths: list[Path]) -> str:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return "missing"
    newest = max(existing, key=lambda x: x.stat().st_mtime)
    return file_mtime(newest)


def list_outputs():
    figs = sorted([p.name for p in FIG_DIR.glob("*.png")]) if FIG_DIR.exists() else []
    tabs = sorted([p.name for p in TAB_DIR.glob("*.csv")]) if TAB_DIR.exists() else []
    return figs, tabs


def show_png(name: str, caption: str | None = None):
    p = FIG_DIR / name
    if not p.exists():
        st.warning(f"Missing figure: {p}")
        return
    st.image(str(p), caption=caption, use_container_width=True)
    with open(p, "rb") as f:
        st.download_button(
            label=f"Download {name}",
            data=f.read(),
            file_name=name,
            mime="image/png",
            use_container_width=True,
        )


def show_csv(name: str):
    p = TAB_DIR / name
    if not p.exists():
        st.warning(f"Missing table: {p}")
        return
    df = pd.read_csv(p)
    st.dataframe(df, use_container_width=True)
    st.caption(f"Source: {p}  |  Updated: {file_mtime(p)}")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {name}",
        data=csv_bytes,
        file_name=name,
        mime="text/csv",
        use_container_width=True,
    )


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Controls")

if st.sidebar.button("🔄 Refresh / Rerun (reload outputs)", use_container_width=True):
    st.rerun()

show_report = st.sidebar.checkbox("Show Report (REPORT.md)", value=True)
show_var_compare = st.sidebar.checkbox("Show VaR Comparison", value=True)
show_var_backtest = st.sidebar.checkbox("Show VaR Backtest", value=True)
show_hist_scen = st.sidebar.checkbox("Show Historical Stress Scenarios", value=True)
show_risk_contrib = st.sidebar.checkbox("Show Risk Contribution", value=True)

st.sidebar.divider()
st.sidebar.caption("Tip: If something is missing, rerun Step 2–5 scripts to regenerate outputs.")


# -----------------------------
# Header
# -----------------------------
st.title("📊 Portfolio Risk War Room Dashboard")
st.write(
    "A lightweight dashboard to view risk metrics, VaR models, backtests, stress scenarios, "
    "and risk contribution outputs."
)

figs, tabs = list_outputs()
all_paths = [FIG_DIR / f for f in figs] + [TAB_DIR / t for t in tabs]
st.info(f"**Last Updated (outputs):** {newest_mtime(all_paths)}")


with st.expander("📁 Output Inventory (click to expand)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Figures")
        st.json(figs)
    with c2:
        st.subheader("Tables")
        st.json(tabs)

st.divider()

# -----------------------------
# Sections
# -----------------------------
if show_report:
    st.header("🧾 Report")
    if REPORT_MD.exists():
        st.caption(f"Updated: {file_mtime(REPORT_MD)}  |  Path: {REPORT_MD}")
        st.markdown(REPORT_MD.read_text(encoding="utf-8"), unsafe_allow_html=False)
        st.download_button(
            "Download REPORT.md",
            data=REPORT_MD.read_text(encoding="utf-8").encode("utf-8"),
            file_name="REPORT.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.warning(f"Missing report file: {REPORT_MD}")

st.divider()

if show_var_compare:
    st.header("📉 VaR Model Comparison")

    # ✅ Show VaR comparison figure (fix broken image in REPORT.md)
    show_png(
        "var_compare.png",
        caption="VaR compare: Hist vs Normal vs EWMA (if available)"
    )

    st.subheader("VaR Series Tables (if present)")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Historical VaR**")
        show_csv("var_hist_series.csv")
    with cols[1]:
        st.markdown("**Normal VaR**")
        show_csv("var_normal_series.csv")
    with cols[2]:
        st.markdown("**EWMA VaR**")
        show_csv("ewma_var_series.csv")

st.divider()

if show_var_backtest:
    st.header("✅ VaR Backtest")
    c1, c2 = st.columns(2)
    with c1:
        show_png("var_backtest_breach_rate.png", caption="Backtest breach rate")
    with c2:
        show_png("kupiec_pvalues.png", caption="Kupiec test p-values")
    st.subheader("Backtest Tables")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Backtest Summary**")
        show_csv("var_backtest_summary.csv")
    with cols[1]:
        st.markdown("**Kupiec Table**")
        show_csv("var_backtest_kupiec.csv")

    st.subheader("EWMA Backtest Summary (if present)")
    show_csv("var_ewma_backtest_summary.csv")

st.divider()

if show_hist_scen:
    st.header("🧨 Historical Stress Scenarios")
    show_png("historical_scenarios.png", caption="Historical scenarios summary (if generated)")
    show_csv("historical_scenarios.csv")

st.divider()

if show_risk_contrib:
    st.header("🧩 Risk Contribution")
    show_png("top_risk_contributors.png", caption="Top risk contributors (vol contribution)")
    show_csv("risk_contribution.csv")
