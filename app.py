from __future__ import annotations

import os
import time
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st


# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGS_DIR = OUTPUTS_DIR / "figures"
REPORT_DIR = ROOT / "report"

WEIGHTS_CSV = DATA_DIR / "weights.csv"
RISK_SUMMARY_CSV = DATA_DIR / "risk_summary.csv"  # used in report sometimes

REPORT_MD = REPORT_DIR / "REPORT.md"

# Some repos also write an outputs/report.md; keep both if present
ALT_REPORT_MD = OUTPUTS_DIR / "report.md"


# -----------------------------
# Helpers
# -----------------------------
def file_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except FileNotFoundError:
        return "N/A"


def safe_read_text(path: Path, encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding, errors="replace")
    except Exception as e:
        return f"⚠️ Failed to read {path}: {e}"


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            return None
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"⚠️ Failed to read CSV: {path}\n\n{e}")
        return None


def show_csv(filename: str, caption: Optional[str] = None) -> None:
    path = TABLES_DIR / filename
    if caption:
        st.caption(caption)
    df = safe_read_csv(path)
    if df is None:
        st.warning(f"Missing table: {path}")
        return
    st.dataframe(df, use_container_width=True)


def show_png(filename: str, caption: Optional[str] = None) -> None:
    path = FIGS_DIR / filename
    if not path.exists():
        st.warning(f"Missing figure: {path}")
        return
    st.image(str(path), caption=caption, use_container_width=True)
    # optional download
    with open(path, "rb") as f:
        st.download_button(
            label=f"Download {filename}",
            data=f,
            file_name=filename,
            mime="image/png",
            use_container_width=True,
        )


def normalize_weights(ws: List[float]) -> List[float]:
    s = sum(ws)
    if s <= 0:
        return ws
    return [w / s for w in ws]


def load_weights() -> pd.DataFrame:
    if WEIGHTS_CSV.exists():
        df = safe_read_csv(WEIGHTS_CSV)
        if df is None:
            return pd.DataFrame(columns=["asset", "weight"])
        # enforce columns
        if "asset" not in df.columns or "weight" not in df.columns:
            return pd.DataFrame(columns=["asset", "weight"])
        df["asset"] = df["asset"].astype(str)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        return df[["asset", "weight"]]
    return pd.DataFrame(columns=["asset", "weight"])


def save_weights(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(WEIGHTS_CSV, index=False)


def run_pipeline() -> Tuple[bool, str]:
    """
    Run scripts/run_all.sh (works on mac/linux + Streamlit Cloud).
    Returns (ok, logs).
    """
    script = ROOT / "scripts" / "run_all.sh"
    if not script.exists():
        return False, f"Missing script: {script}"

    try:
        # Ensure executable
        try:
            script.chmod(script.stat().st_mode | 0o111)
        except Exception:
            pass

        # Run
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        ok = proc.returncode == 0
        return ok, out
    except Exception as e:
        return False, f"Failed to run pipeline: {e}"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Portfolio Risk War Room", layout="wide")

st.title("📊 Portfolio Risk War Room Dashboard")
st.caption(
    "A lightweight dashboard to view risk metrics, VaR models, backtests, stress scenarios, and risk contribution outputs."
)

# Sidebar controls
st.sidebar.header("Controls")

# IMPORTANT: sliders cause reruns; DO NOT auto-run heavy scripts on every rerun.
# So we only update weights, and provide a manual button to regenerate outputs.
with st.sidebar.expander("⚖️ Portfolio Weights (edit)", expanded=False):
    wdf = load_weights()
    if wdf.empty:
        st.info("`data/weights.csv` is missing/empty. Add rows: asset,weight")
    else:
        assets = wdf["asset"].tolist()
        current = wdf["weight"].astype(float).tolist()

        st.write("Adjust weights below. Then click **Save weights**.")
        new_ws = []
        for a, w in zip(assets, current):
            new_ws.append(
                st.slider(
                    label=a,
                    min_value=0.0,
                    max_value=1.0,
                    value=float(w),
                    step=0.01,
                    key=f"w_{a}",
                )
            )

        auto_norm = st.checkbox("Auto-normalize to sum = 1.0", value=True)
        if auto_norm:
            norm_ws = normalize_weights(new_ws)
        else:
            norm_ws = new_ws

        st.write(f"Sum: **{sum(norm_ws):.4f}**")

        if st.button("💾 Save weights", use_container_width=True):
            out = pd.DataFrame({"asset": assets, "weight": norm_ws})
            save_weights(out)
            st.success("Saved `data/weights.csv`. Now click **Regenerate outputs** (below) to update charts/tables.")

# Manual pipeline run
st.sidebar.divider()
if st.sidebar.button("🔄 Regenerate outputs (run_all.sh)", use_container_width=True):
    with st.spinner("Running pipeline... (this may take a bit)"):
        ok, logs = run_pipeline()
    if ok:
        st.sidebar.success("Pipeline finished ✅")
    else:
        st.sidebar.error("Pipeline failed ❌")
    with st.expander("Show pipeline logs"):
        st.code(logs)

# Show last update
st.info(f"Last Updated (outputs): {file_mtime(OUTPUTS_DIR)}", icon="🕒")

# Section toggles
show_report = st.sidebar.checkbox("Show Report (REPORT.md)", value=True)
show_var_compare = st.sidebar.checkbox("Show VaR Comparison", value=True)
show_var_backtest = st.sidebar.checkbox("Show VaR Backtest", value=True)
show_hist_scen = st.sidebar.checkbox("Show Historical Stress Scenarios", value=True)
show_risk_contrib = st.sidebar.checkbox("Show Risk Contribution", value=True)

st.divider()

# -----------------------------
# 1) Report
# -----------------------------
if show_report:
    st.header("🧾 Report")

    # Prefer report/REPORT.md, fallback to outputs/report.md
    report_path = REPORT_MD if REPORT_MD.exists() else ALT_REPORT_MD if ALT_REPORT_MD.exists() else None

    if report_path is None:
        st.warning(f"Missing report file. Expected `{REPORT_MD}` (or `{ALT_REPORT_MD}`).")
    else:
        st.caption(f"Updated: {file_mtime(report_path)} | Path: {report_path}")
        md_text = safe_read_text(report_path)
        st.markdown(md_text, unsafe_allow_html=False)

        st.download_button(
            "Download REPORT.md",
            data=md_text.encode("utf-8"),
            file_name="REPORT.md",
            mime="text/markdown",
            use_container_width=True,
        )

st.divider()

# -----------------------------
# 2) VaR model comparison
# -----------------------------
if show_var_compare:
    st.header("📉 VaR Model Comparison")

    # This is the figure that often fails inside markdown; we render directly.
    show_png("var_compare.png", caption="VaR compare: Hist vs Normal vs EWMA (if available)")

    st.subheader("VaR Series Tables (if present)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Historical VaR**")
        show_csv("var_hist_series.csv")
    with c2:
        st.markdown("**Normal VaR**")
        show_csv("var_normal_series.csv")
    with c3:
        st.markdown("**EWMA VaR**")
        show_csv("ewma_var_series.csv")

st.divider()

# -----------------------------
# 3) VaR backtest
# -----------------------------
if show_var_backtest:
    st.header("✅ VaR Backtest")

    c1, c2 = st.columns(2)
    with c1:
        show_png("var_backtest_breach_rate.png", caption="Backtest breach rate")
    with c2:
        show_png("kupiec_pvalues.png", caption="Kupiec test p-values")

    st.subheader("Backtest Tables")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Backtest Summary**")
        show_csv("var_backtest_summary.csv")
    with c4:
        st.markdown("**Kupiec Table**")
        show_csv("var_backtest_kupiec.csv")

    st.subheader("EWMA Backtest Summary (if present)")
    show_csv("var_ewma_backtest_summary.csv")

st.divider()

# -----------------------------
# 4) Historical stress scenarios
# -----------------------------
if show_hist_scen:
    st.header("📌 Historical Stress Scenarios")
    show_png("historical_scenarios.png", caption="Historical scenarios summary (if generated)")
    show_csv("historical_scenarios.csv")

st.divider()

# -----------------------------
# 5) Risk contribution
# -----------------------------
if show_risk_contrib:
    st.header("🧩 Risk Contribution")
    show_png("top_risk_contributors.png", caption="Top risk contributors (vol contribution)")
    show_csv("risk_contribution.csv")
