from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
TABLE_DIR = OUTPUTS_DIR / "tables"

REPORT_MD_PRIMARY = BASE_DIR / "report" / "REPORT.md"
REPORT_MD_FALLBACK = OUTPUTS_DIR / "report.md"

WEIGHTS_CSV = BASE_DIR / "data" / "weights.csv"


# ============================================================
# Helpers
# ============================================================
def file_mtime(p: Path) -> str:
    if not p.exists():
        return "N/A"
    dt = datetime.fromtimestamp(p.stat().st_mtime)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1", errors="replace")


@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    return pd.read_csv(p)


def show_csv(filename: str, title: str | None = None):
    p = TABLE_DIR / filename
    if title:
        st.markdown(f"**{title}**")
    if not p.exists():
        st.warning(f"Missing table: {p}")
        return
    df = load_csv_cached(str(p))
    st.dataframe(df, use_container_width=True)


def show_png(filename: str, caption: str | None = None):
    p = FIG_DIR / filename
    if not p.exists():
        st.warning(f"Missing figure: {p}")
        return
    st.image(str(p), caption=caption, use_container_width=True)


def list_inventory():
    figs = sorted([f.name for f in FIG_DIR.glob("*.png")]) if FIG_DIR.exists() else []
    tables = sorted([t.name for t in TABLE_DIR.glob("*.csv")]) if TABLE_DIR.exists() else []
    return figs, tables


def resolve_report_path() -> Path | None:
    if REPORT_MD_PRIMARY.exists():
        return REPORT_MD_PRIMARY
    if REPORT_MD_FALLBACK.exists():
        return REPORT_MD_FALLBACK
    return None


def resolve_image_path(img_ref: str, report_file: Path) -> Path | None:
    """
    img_ref could be:
      - "outputs/figures/var_compare.png"
      - "../outputs/figures/var_compare.png"
      - "var_compare.png"
    We try multiple sensible bases.
    """
    # Strip query/anchors if any
    img_ref = img_ref.split("#")[0].split("?")[0].strip()

    candidates = []

    # 1) Relative to report file directory
    candidates.append((report_file.parent / img_ref).resolve())

    # 2) Relative to project root
    candidates.append((BASE_DIR / img_ref).resolve())

    # 3) If it looks like a figures filename, try outputs/figures directly
    candidates.append((FIG_DIR / Path(img_ref).name).resolve())

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def render_report_md_with_images(report_file: Path):
    """
    Render markdown, but convert markdown image lines to st.image
    so images show correctly on Streamlit Cloud.
    """
    md = safe_read_text(report_file)

    # match: ![alt](path)
    img_pat = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")

    buffer_lines: list[str] = []

    def flush_buffer():
        nonlocal buffer_lines
        if buffer_lines:
            st.markdown("\n".join(buffer_lines), unsafe_allow_html=False)
            buffer_lines = []

    for line in md.splitlines():
        m = img_pat.search(line)
        if not m:
            buffer_lines.append(line)
            continue

        # flush text before image
        flush_buffer()

        alt = (m.group("alt") or "").strip()
        img_ref = (m.group("path") or "").strip()

        img_path = resolve_image_path(img_ref, report_file)
        if img_path is None:
            st.warning(f"Missing image referenced in REPORT.md: `{img_ref}`")
        else:
            st.image(str(img_path), caption=alt if alt else None, use_container_width=True)

        # If there is remaining text besides the image markdown in the same line, show it
        # (rare, but just in case)
        remainder = img_pat.sub("", line).strip()
        if remainder:
            st.markdown(remainder, unsafe_allow_html=False)

    flush_buffer()


def load_weights() -> pd.DataFrame:
    if not WEIGHTS_CSV.exists():
        # Create a default file to avoid crashes
        WEIGHTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"asset": [], "weight": []})
        df.to_csv(WEIGHTS_CSV, index=False)
        return df
    return pd.read_csv(WEIGHTS_CSV)


def save_weights(df: pd.DataFrame):
    WEIGHTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WEIGHTS_CSV, index=False)


def try_run_all_scripts() -> tuple[bool, str]:
    """
    Attempt to run scripts/run_all.sh to regenerate outputs.
    Works locally; on Streamlit Cloud it may fail due to permissions/time.
    """
    script = BASE_DIR / "scripts" / "run_all.sh"
    if not script.exists():
        return False, f"Not found: {script}"

    try:
        # Ensure executable (best-effort)
        script.chmod(script.stat().st_mode | 0o111)
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        ok = proc.returncode == 0
        return ok, out[-6000:]  # keep tail so Streamlit doesn't choke
    except Exception as e:
        return False, f"Failed to run scripts: {e}"


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Portfolio Risk War Room", layout="wide")

st.title("📊 Portfolio Risk War Room Dashboard")
st.caption(
    "A lightweight dashboard to view risk metrics, VaR models, backtests, stress scenarios, "
    "and risk contribution outputs."
)

# Sidebar controls
st.sidebar.header("Controls")

# Weights editor
with st.sidebar.expander("⚖️ Portfolio Weights (edit)", expanded=False):
    wdf = load_weights()
    if wdf.empty:
        st.info("`data/weights.csv` is empty. Add assets there first (asset, weight).")
    else:
        # Build sliders
        assets = wdf["asset"].astype(str).tolist()
        weights = wdf["weight"].astype(float).tolist()

        st.write("Adjust weights below. You can auto-normalize to sum to 1.0.")
        new_weights = []
        for a, w in zip(assets, weights):
            new_w = st.slider(a, min_value=0.0, max_value=1.0, value=float(w), step=0.01)
            new_weights.append(new_w)

        total = sum(new_weights)
        st.write(f"Current sum: **{total:.4f}**")

        colA, colB = st.columns(2)
        with colA:
            normalize = st.button("Normalize to sum=1")
        with colB:
            save_btn = st.button("Save to data/weights.csv")

        if normalize and total > 0:
            new_weights = [w / total for w in new_weights]
            total = 1.0
            st.success("Normalized. Now click Save if you want to persist.")

        if save_btn:
            out_df = pd.DataFrame({"asset": assets, "weight": new_weights})
            save_weights(out_df)
            st.success("Saved weights to data/weights.csv ✅")

            st.info(
                "To update charts/tables, you must regenerate outputs. "
                "Locally you can run `./scripts/run_all.sh` (or click below). "
                "On Streamlit Cloud, use **Manage App → Restart** to re-run the preScript."
            )

        # Optional: run scripts from UI (mostly for local use)
        run_now = st.button("Run scripts now (local best)")
        if run_now:
            with st.spinner("Running scripts/run_all.sh ..."):
                ok, log_tail = try_run_all_scripts()
            if ok:
                st.success("Scripts completed. Click 'Refresh / Rerun' on the main page.")
            else:
                st.error("Scripts failed (this is common on Streamlit Cloud).")
            st.code(log_tail)

# Section toggles
show_report = st.sidebar.checkbox("Show Report (REPORT.md)", value=True)
show_var_compare = st.sidebar.checkbox("Show VaR Comparison", value=True)
show_var_backtest = st.sidebar.checkbox("Show VaR Backtest", value=True)
show_hist_scen = st.sidebar.checkbox("Show Historical Stress Scenarios", value=True)
show_risk_contrib = st.sidebar.checkbox("Show Risk Contribution", value=True)

# Refresh button
if st.sidebar.button("Refresh / Rerun (reload outputs)"):
    st.cache_data.clear()
    st.rerun()

# Last updated
st.info(f"Last Updated (outputs): {file_mtime(OUTPUTS_DIR)}")

# Inventory
with st.expander("📁 Output Inventory (click to expand)", expanded=False):
    figs, tables = list_inventory()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Figures:**")
        st.write(figs)
    with c2:
        st.markdown("**Tables:**")
        st.write(tables)

st.divider()

# ============================================================
# Sections
# ============================================================
if show_report:
    st.header("🧾 Report")
    report_file = resolve_report_path()
    if report_file is None:
        st.warning(f"Missing report file. Looked for:\n- {REPORT_MD_PRIMARY}\n- {REPORT_MD_FALLBACK}")
    else:
        st.caption(f"Updated: {file_mtime(report_file)}  |  Path: {report_file}")
        # Render markdown with images fixed
        render_report_md_with_images(report_file)

        # Download
        st.download_button(
            "Download REPORT.md",
            data=safe_read_text(report_file).encode("utf-8"),
            file_name="REPORT.md",
            mime="text/markdown",
            use_container_width=True,
        )

st.divider()

if show_var_compare:
    st.header("📉 VaR Model Comparison")
    # Ensure the comparison figure shows
    show_png("var_compare.png", caption="VaR compare: Hist vs Normal vs EWMA (if available)")

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
