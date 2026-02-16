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
# UI (replace your whole UI section with this)
# ============================================================

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Portfolio Risk War Room", layout="wide")

st.title("📊 Portfolio Risk War Room Dashboard")
st.caption(
    "A lightweight dashboard to view risk metrics, VaR models, backtests, stress scenarios, "
    "and risk contribution outputs."
)

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")

# ---- Weights editor (safe: no rerun on slider drag)
WEIGHTS_PATH = Path("data/weights.csv")

def _load_weights_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["asset", "weight"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["asset", "weight"])
    df["asset"] = df["asset"].astype(str)
    df["weight"] = df["weight"].astype(float)
    return df

def _save_weights_csv(df: pd.DataFrame, path: Path) -> None:
    df = df[["asset", "weight"]].copy()
    df.to_csv(path, index=False)

with st.sidebar.expander("⚖️ Portfolio Weights (edit)", expanded=False):
    wdf = _load_weights_csv(WEIGHTS_PATH)

    if wdf.empty:
        st.info("`data/weights.csv` is empty or missing. Add rows there first (asset, weight).")
    else:
        assets = wdf["asset"].tolist()
        current = dict(zip(wdf["asset"], wdf["weight"]))

        # form prevents rerun spam while dragging sliders
        with st.form("weights_form", clear_on_submit=False):
            st.caption("拖动滑条不会刷新；点 Apply 才保存并更新一次。")
            new_weights = {}
            for a in assets:
                new_weights[a] = st.slider(
                    a,
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current.get(a, 0.0)),
                    step=0.01,
                    key=f"w_{a}",
                )
            normalize = st.checkbox("Auto-normalize to sum=1", value=True)
            apply_btn = st.form_submit_button("✅ Apply weights")

        if apply_btn:
            s = sum(new_weights.values())
            if s <= 0:
                st.error("权重总和不能为 0。")
            else:
                if normalize:
                    new_weights = {k: v / s for k, v in new_weights.items()}

                out_df = pd.DataFrame(
                    {"asset": list(new_weights.keys()), "weight": list(new_weights.values())}
                )
                _save_weights_csv(out_df, WEIGHTS_PATH)
                st.success(f"Saved data/weights.csv (sum={out_df['weight'].sum():.4f})")
                st.rerun()

st.sidebar.divider()

# section toggles
show_report = st.sidebar.checkbox("Show Report (REPORT.md)", value=True)
show_var_compare = st.sidebar.checkbox("Show VaR Comparison", value=True)
show_var_backtest = st.sidebar.checkbox("Show VaR Backtest", value=True)
show_hist_scen = st.sidebar.checkbox("Show Historical Stress Scenarios", value=True)
show_risk_contrib = st.sidebar.checkbox("Show Risk Contribution", value=True)

# optional manual refresh (just rerun page)
if st.sidebar.button("🔄 Refresh / Rerun (reload outputs)"):
    st.rerun()

# ----------------------------
# Main content
# ----------------------------

# timestamp banner (optional but helpful)
try:
    # If you have a helper for outputs folder mtime, keep yours.
    # Otherwise show nothing.
    pass
except Exception:
    pass


# ---- 1) Report
if show_report:
    st.header("🧾 Report")

    # Your project already uses REPORT_MD / file_mtime helpers.
    # This block assumes:
    # - REPORT_MD is a Path to report/REPORT.md
    # - REPORT_MD_exists() returns bool
    if "REPORT_MD" in globals() and callable(globals().get("REPORT_MD_exists", None)):
        if REPORT_MD_exists():
            st.caption(f"Updated: {file_mtime(REPORT_MD)} | Path: {REPORT_MD}")
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
    else:
        st.info("Report helpers not detected in globals; skip rendering REPORT.md.")

st.divider()

# ---- 2) VaR Model Comparison
if show_var_compare:
    st.header("📉 VaR Model Comparison")

    # IMPORTANT: your report.md in Streamlit Cloud often breaks image links,
    # so we explicitly render the png from outputs/figures with show_png().
    # This assumes your show_png() function looks in outputs/figures.
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

# ---- 3) VaR Backtest
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

# ---- 4) Historical stress scenarios
if show_hist_scen:
    st.header("🧨 Historical Stress Scenarios")
    show_png("historical_scenarios.png", caption="Historical scenarios summary (if generated)")
    show_csv("historical_scenarios.csv")

st.divider()

# ---- 5) Risk contribution
if show_risk_contrib:
    st.header("🧩 Risk Contribution")
    show_png("top_risk_contributors.png", caption="Top risk contributors (vol contribution)")
    show_csv("risk_contribution.csv")


