from __future__ import annotations

import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import pandas as pd
import streamlit as st


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

WEIGHTS_PATH = DATA_DIR / "weights.csv"

# Report candidates (some repos use outputs/REPORT.md, some use REPORT.md at root)
REPORT_MD_PRIMARY = OUTPUT_DIR / "REPORT.md"
REPORT_MD_FALLBACK = BASE_DIR / "REPORT.md"


# =========================
# UI Config
# =========================
st.set_page_config(
    page_title="Portfolio Risk War Room Dashboard",
    layout="wide",
)


# =========================
# Helpers
# =========================
def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def file_mtime(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


def latest_mtime_under(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    latest = None
    for p in path.rglob("*"):
        if p.is_file():
            t = p.stat().st_mtime
            latest = t if latest is None else max(latest, t)
    if latest is None:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest))


def list_inventory(fig_dir: Path, table_dir: Path) -> Tuple[List[str], List[str]]:
    figs = []
    tables = []
    if fig_dir.exists():
        figs = sorted([p.name for p in fig_dir.glob("*.png")])
    if table_dir.exists():
        tables = sorted([p.name for p in table_dir.glob("*.csv")])
    return figs, tables


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_weights_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept common formats:
      - ticker,weight
      - asset,weight
      - Asset,Weight
    Return a clean df with columns: ticker, weight (float).
    """
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # map column names
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"ticker", "tickers", "symbol", "symbols"}:
            col_map[c] = "ticker"
        if lc in {"asset", "assets"}:
            col_map[c] = "ticker"
        if lc in {"weight", "weights"}:
            col_map[c] = "weight"

    df = df.rename(columns=col_map)

    if "ticker" not in df.columns or "weight" not in df.columns:
        raise ValueError("weights.csv must have columns: ticker/asset and weight")

    df = df[["ticker", "weight"]].copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).astype(float)

    # drop empty tickers
    df = df[df["ticker"].str.len() > 0].reset_index(drop=True)
    return df


def load_weights(path: Path) -> pd.DataFrame:
    if not path.exists():
        # sensible default
        return pd.DataFrame(
            {"ticker": ["XIU", "VFV", "XEF", "ZAG", "GLD", "RY"], "weight": [0.25, 0.2, 0.15, 0.2, 0.1, 0.1]}
        )
    df = pd.read_csv(path)
    return normalize_weights_df(df)


def save_weights_for_pipeline(df_ticker_weight: pd.DataFrame, path: Path) -> None:
    """
    ROOT FIX:
    - UI uses ticker
    - pipeline scripts expect asset,weight (based on your error message)
    So we write EXACTLY: asset,weight
    """
    out = df_ticker_weight.copy()
    out = normalize_weights_df(out)
    out = out.rename(columns={"ticker": "asset"})
    out.to_csv(path, index=False)


@dataclass
class CmdResult:
    cmd: str
    returncode: int
    stdout: str
    stderr: str


def run_commands(cmds: Iterable[List[str]], cwd: Path) -> Tuple[bool, List[CmdResult]]:
    """
    Run commands with current python env and PYTHONPATH fixed to project root.
    This prevents:
      - No module named 'src'
      - pandas missing due to different interpreter
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)

    results: List[CmdResult] = []
    ok = True

    for cmd in cmds:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
        )
        results.append(
            CmdResult(
                cmd=" ".join(cmd),
                returncode=p.returncode,
                stdout=p.stdout or "",
                stderr=p.stderr or "",
            )
        )
        if p.returncode != 0:
            ok = False
            break

    return ok, results


def discover_pipeline_steps() -> List[Path]:
    """
    Try to run common pipeline scripts if they exist.
    (Your repo has at least risk_contribution.py and run_historical_scenarios.py.)
    """
    candidates = [
        "src/run_hist_var_series.py",
        "src/run_var_normal_series.py",
        "src/run_ewma_var_series.py",
        "src/run_var_backtest.py",
        "src/run_historical_scenarios.py",
        "src/risk_contribution.py",
        "src/build_report.py",
    ]
    steps = []
    for rel in candidates:
        p = BASE_DIR / rel
        if p.exists():
            steps.append(p)

    # If none matched, fall back to any "run_" scripts then build_report/risk_contribution
    if not steps:
        run_scripts = sorted((BASE_DIR / "src").glob("run_*.py")) if (BASE_DIR / "src").exists() else []
        for p in run_scripts:
            steps.append(p)
        for rel in ["src/risk_contribution.py", "src/build_report.py"]:
            p = BASE_DIR / rel
            if p.exists() and p not in steps:
                steps.append(p)

    return steps


def resolve_report_path(primary: Path, fallback: Path) -> Optional[Path]:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return None


def render_report_md_with_images(report_path: Path) -> None:
    """
    Render markdown and inline images.
    Supports image links like:
      ![alt](outputs/figures/xxx.png)
      ![alt](figures/xxx.png)
      ![alt](xxx.png)
    """
    text = report_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    img_pat = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    buf: List[str] = []

    def flush_md():
        nonlocal buf
        if buf:
            st.markdown("\n".join(buf))
            buf = []

    for line in lines:
        m = img_pat.search(line)
        if not m:
            buf.append(line)
            continue

        # markdown before image on same line
        before = line[: m.start()].strip()
        after = line[m.end() :].strip()

        if before:
            buf.append(before)

        flush_md()

        alt = m.group(1)
        raw_path = m.group(2).strip().strip('"').strip("'")

        # resolve image path
        img_candidates = [
            (report_path.parent / raw_path),
            (BASE_DIR / raw_path),
            (OUTPUT_DIR / raw_path),
            (FIG_DIR / Path(raw_path).name),
        ]
        img_path = next((p for p in img_candidates if p.exists()), None)

        if img_path is not None:
            st.image(str(img_path), caption=alt if alt else None, use_container_width=True)
        else:
            st.warning(f"Missing image referenced in report: {raw_path}")

        if after:
            buf.append(after)

    flush_md()


# =========================
# Sidebar Navigation + Controls
# =========================
ensure_dirs()

st.sidebar.title("Navigate")
page = st.sidebar.radio(" ", ["Overview", "Report", "Tables", "Figures"], index=0)

st.sidebar.header("Controls")

if st.sidebar.button("Clear cache + reload"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# =========================
# Portfolio Weights Editor
# =========================
with st.sidebar.expander("Portfolio Weights (edit)", expanded=True):
    try:
        weights_df = load_weights(WEIGHTS_PATH)
    except Exception as e:
        st.error(f"Failed to load weights: {e}")
        weights_df = pd.DataFrame({"ticker": [], "weight": []})

    edited = st.data_editor(
        weights_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ticker": st.column_config.TextColumn("ticker"),
            "weight": st.column_config.NumberColumn("weight", min_value=0.0, max_value=1.0, step=0.01, format="%.6f"),
        },
        key="weights_editor",
    )

    # compute sum
    try:
        edited_norm = normalize_weights_df(pd.DataFrame(edited))
        s = float(edited_norm["weight"].sum())
    except Exception:
        edited_norm = pd.DataFrame({"ticker": [], "weight": []})
        s = float("nan")

    st.caption(f"Current sum of weights = {s:.6f}" if s == s else "Current sum of weights = (invalid)")

    colA, colB = st.columns(2)

    if colA.button("Save weights.csv", use_container_width=True):
        try:
            save_weights_for_pipeline(edited_norm, WEIGHTS_PATH)
            st.success("Saved data/weights.csv (as asset,weight for pipeline compatibility).")
        except Exception as e:
            st.error(f"Save failed: {e}")

    if colB.button("Save + Rerun pipeline", use_container_width=True):
        try:
            save_weights_for_pipeline(edited_norm, WEIGHTS_PATH)
            st.info("Saved weights.csv — running pipeline...")

            steps = discover_pipeline_steps()
            if not steps:
                st.error("No pipeline scripts found under src/.")
            else:
                cmds = [[sys.executable, str(p)] for p in steps]
                ok, results = run_commands(cmds, cwd=BASE_DIR)

                st.subheader("🧾 Pipeline logs")
                for r in results:
                    st.code(
                        f"$ {r.cmd}\n"
                        f"(exit={r.returncode})\n\n"
                        f"{r.stdout}\n"
                        f"{r.stderr}".strip(),
                        language="text",
                    )

                if ok:
                    st.success("Pipeline finished successfully.")
                    # HARD refresh: clear caches and rerun so tables/figures reflect new outputs
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("Pipeline failed. See logs above.")
        except Exception as e:
            st.error(f"Pipeline runner error: {e}")


# =========================
# Main Pages
# =========================
st.title("Portfolio Risk War Room Dashboard")
st.caption("A lightweight dashboard to view risk metrics, VaR models, backtests, stress scenarios, and risk contribution outputs.")

if page == "Overview":
    updated = latest_mtime_under(OUTPUT_DIR)
    st.info(f"Last Updated (outputs): {updated or 'N/A'}")

    with st.expander("Output Inventory (click to expand)", expanded=True):
        figs, tables = list_inventory(FIG_DIR, TABLE_DIR)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Figures**")
            st.json(figs)
        with c2:
            st.write("**Tables**")
            st.json(tables)

    st.divider()
    st.subheader("Quick sanity checks")
    st.write(f"- weights file: `{WEIGHTS_PATH}`")
    st.write(f"- weights last modified: `{file_mtime(WEIGHTS_PATH) or 'missing'}`")
    st.write(f"- outputs dir: `{OUTPUT_DIR}`")
    st.write(f"- outputs last modified: `{updated or 'N/A'}`")

elif page == "Report":
    st.header("Report")
    report_file = resolve_report_path(REPORT_MD_PRIMARY, REPORT_MD_FALLBACK)

    if report_file is None:
        st.warning(f"Missing report file. Looked for:\n- {REPORT_MD_PRIMARY}\n- {REPORT_MD_FALLBACK}")
    else:
        st.caption(f"Updated: {file_mtime(report_file)}   |   Path: {report_file}")
        # Render markdown + inline images
        render_report_md_with_images(report_file)

        st.divider()
        report_text = report_file.read_text(encoding="utf-8", errors="replace")
        st.download_button(
            "Download REPORT.md",
            data=report_text.encode("utf-8"),
            file_name="REPORT.md",
            mime="text/markdown",
        )

elif page == "Tables":
    st.header("Tables")
    if not TABLE_DIR.exists():
        st.warning("outputs/tables not found.")
    else:
        tables = sorted(TABLE_DIR.glob("*.csv"))
        if not tables:
            st.warning("No CSV tables found in outputs/tables.")
        else:
            pick = st.selectbox("Select a table", [p.name for p in tables], index=0)
            p = TABLE_DIR / pick
            st.caption(f"Updated: {file_mtime(p)}")
            df = pd.read_csv(p)
            st.dataframe(df, use_container_width=True)

elif page == "Figures":
    st.header("Figures")
    if not FIG_DIR.exists():
        st.warning("outputs/figures not found.")
    else:
        figs = sorted(FIG_DIR.glob("*.png"))
        if not figs:
            st.warning("No PNG figures found in outputs/figures.")
        else:
            pick = st.selectbox("Select a figure", [p.name for p in figs], index=0)
            p = FIG_DIR / pick
            st.caption(f"Updated: {file_mtime(p)}")
            st.image(str(p), use_container_width=True)

