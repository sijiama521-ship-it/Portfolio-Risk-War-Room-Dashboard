# app.py
from __future__ import annotations

import os
import time
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Portfolio Risk War Room Dashboard",
    page_icon="📊",
    layout="wide",
)

APP_ROOT = Path(__file__).resolve().parent

DATA_DIR = APP_ROOT / "data"
OUTPUTS_DIR = APP_ROOT / "outputs"
OUTPUTS_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUTS_FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORT_DIR = APP_ROOT / "report"

WEIGHTS_PATH = DATA_DIR / "weights.csv"
WEIGHTS_EXAMPLE_PATH = DATA_DIR / "weights.example.csv"

RUN_ALL_SH = APP_ROOT / "scripts" / "run_all.sh"

# Report candidates (your UI earlier searched these)
REPORT_CANDIDATES = [
    OUTPUTS_DIR / "REPORT.md",
    APP_ROOT / "REPORT.md",
    REPORT_DIR / "REPORT.md",
    OUTPUTS_DIR / "report.md",
]

DEFAULT_WEIGHTS = pd.DataFrame(
    {
        "ticker": ["XIU", "VFV", "XEF", "ZAG", "GLD", "RY"],
        "weight": [0.25, 0.20, 0.15, 0.20, 0.10, 0.10],
    }
)


# =========================
# Small utilities
# =========================
def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _dir_mtime(d: Path) -> float:
    """Return latest mtime among files in dir (non-recursive)."""
    if not d.exists():
        return 0.0
    latest = 0.0
    for p in d.iterdir():
        if p.is_file():
            latest = max(latest, _safe_mtime(p))
    return latest


def _normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).astype(float)

    # Drop empty tickers
    df = df[df["ticker"].str.len() > 0]

    # Clip negatives to 0
    df["weight"] = df["weight"].clip(lower=0.0)

    # Combine duplicates by sum
    df = df.groupby("ticker", as_index=False)["weight"].sum()

    s = float(df["weight"].sum())
    if s <= 0:
        # fallback equal weights
        if len(df) == 0:
            df = DEFAULT_WEIGHTS.copy()
            s = float(df["weight"].sum())
        else:
            df["weight"] = 1.0 / len(df)
            s = 1.0

    df["weight"] = df["weight"] / s
    return df[["ticker", "weight"]]


def _sanitize_weights_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert many possible formats into strict schema:
      columns: ticker, weight
    and normalize to sum=1.
    """
    df = raw.copy()

    # If the CSV was saved with an index column, reset it.
    if "Unnamed: 0" in df.columns and ("ticker" not in df.columns):
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]

    # Common alternative column names
    rename_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in {"asset", "symbol", "ticker"}:
            rename_map[c] = "ticker"
        if lc in {"w", "weights", "weight"}:
            rename_map[c] = "weight"
    df = df.rename(columns=rename_map)

    # Sometimes it is two columns but unnamed; try to infer
    if ("ticker" not in df.columns or "weight" not in df.columns) and df.shape[1] >= 2:
        cols = list(df.columns)
        # If first column looks like tickers and second looks numeric -> map
        if "ticker" not in df.columns and "weight" not in df.columns:
            df = df.rename(columns={cols[0]: "ticker", cols[1]: "weight"})
        elif "ticker" not in df.columns and "weight" in df.columns:
            # pick a non-weight col
            other = [c for c in df.columns if c != "weight"][0]
            df = df.rename(columns={other: "ticker"})
        elif "ticker" in df.columns and "weight" not in df.columns:
            other = [c for c in df.columns if c != "ticker"][0]
            df = df.rename(columns={other: "weight"})

    # If ticker is index
    if "ticker" not in df.columns and df.index.name:
        if str(df.index.name).lower() in {"ticker", "asset", "symbol"}:
            df = df.reset_index().rename(columns={df.index.name: "ticker"})

    if "ticker" not in df.columns or "weight" not in df.columns:
        # ultimate fallback
        return _normalize_weights(DEFAULT_WEIGHTS)

    df = df[["ticker", "weight"]].copy()
    return _normalize_weights(df)


def _ensure_weights_file_exists() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if WEIGHTS_PATH.exists():
        return

    # If example exists, use it; else use default
    if WEIGHTS_EXAMPLE_PATH.exists():
        try:
            ex = pd.read_csv(WEIGHTS_EXAMPLE_PATH)
            df = _sanitize_weights_df(ex)
            df.to_csv(WEIGHTS_PATH, index=False)
            return
        except Exception:
            pass

    DEFAULT_WEIGHTS.to_csv(WEIGHTS_PATH, index=False)


@st.cache_data(show_spinner=False)
def load_weights(weights_mtime: float) -> pd.DataFrame:
    """
    Cached by weights file mtime.
    Also self-heals: if file is malformed, it rewrites a correct version.
    """
    _ensure_weights_file_exists()

    try:
        raw = pd.read_csv(WEIGHTS_PATH)
    except Exception:
        df = _normalize_weights(DEFAULT_WEIGHTS)
        df.to_csv(WEIGHTS_PATH, index=False)
        return df

    df = _sanitize_weights_df(raw)

    # If the file on disk is malformed, rewrite it in canonical schema
    try:
        # Compare columns and normalized sum
        needs_rewrite = True
        if list(raw.columns) == ["ticker", "weight"]:
            # check if already normalized
            s = float(pd.to_numeric(raw["weight"], errors="coerce").fillna(0.0).sum())
            if s > 0:
                needs_rewrite = False
        if needs_rewrite:
            df.to_csv(WEIGHTS_PATH, index=False)
    except Exception:
        # ignore rewrite failure
        pass

    return df


def save_weights(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = _sanitize_weights_df(df)
    cleaned.to_csv(WEIGHTS_PATH, index=False)


def run_pipeline() -> Tuple[int, str]:
    """
    Run scripts/run_all.sh via bash (works on Streamlit Cloud).
    Returns (returncode, combined_output).
    """
    if not RUN_ALL_SH.exists():
        return 1, f"❌ Missing pipeline script: {RUN_ALL_SH}"

    cmd = ["bash", str(RUN_ALL_SH)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out.strip()
    except Exception as e:
        return 1, f"❌ Failed to run pipeline: {e}"


@st.cache_data(show_spinner=False)
def read_report(report_mtime: float) -> Tuple[Optional[str], List[str]]:
    """
    Return (content, searched_paths).
    Cached by latest candidate mtime.
    """
    searched = [str(p) for p in REPORT_CANDIDATES]
    for p in REPORT_CANDIDATES:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8"), searched
            except Exception:
                try:
                    return p.read_text(errors="ignore"), searched
                except Exception:
                    return None, searched
    return None, searched


@st.cache_data(show_spinner=False)
def list_tables(dir_mtime: float) -> List[Path]:
    if not OUTPUTS_TABLES_DIR.exists():
        return []
    files = sorted([p for p in OUTPUTS_TABLES_DIR.glob("*.csv") if p.is_file()])
    return files


@st.cache_data(show_spinner=False)
def list_figures(dir_mtime: float) -> List[Path]:
    if not OUTPUTS_FIGURES_DIR.exists():
        return []
    files = sorted([p for p in OUTPUTS_FIGURES_DIR.glob("*.png") if p.is_file()])
    return files


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p)


# =========================
# Sidebar: Navigation & Controls
# =========================
st.sidebar.title("Navigate")
page = st.sidebar.radio(" ", ["Report", "Tables", "Figures"], index=0)

st.sidebar.title("Controls")

if st.sidebar.button("🧹 Clear cache + reload"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.subheader("⚖️ Portfolio Weights (edit)")

# Load weights with cache keyed by mtime
weights_mtime = _safe_mtime(WEIGHTS_PATH)
weights_df = load_weights(weights_mtime)

edited = st.sidebar.data_editor(
    weights_df,
    use_container_width=True,
    num_rows="dynamic",
    key="weights_editor",
)

# Show sum in sidebar (based on edited values)
try:
    tmp = _sanitize_weights_df(pd.DataFrame(edited))
    st.sidebar.caption(f"Current sum of weights = {tmp['weight'].sum():.6f}")
except Exception:
    st.sidebar.caption("Current sum of weights = (invalid)")

col_a, col_b = st.sidebar.columns(2)

with col_a:
    if st.button("💾 Save weights.csv", use_container_width=True):
        try:
            save_weights(pd.DataFrame(edited))
            st.success("Saved weights.csv (auto-normalized).")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Save failed: {e}")

with col_b:
    if st.button("🚀 Save + Rerun pipeline", use_container_width=True):
        try:
            save_weights(pd.DataFrame(edited))
            st.toast("Saved weights.csv. Running pipeline…")
        except Exception as e:
            st.error(f"Save failed: {e}")
            st.stop()

        rc, logs = run_pipeline()
        if rc == 0:
            st.success("Pipeline completed.")
        else:
            st.error("Pipeline failed. See logs below.")
        with st.expander("Pipeline logs"):
            st.code(logs or "(no output)")

        # Clear cached reads so tables/figures/report refresh
        st.cache_data.clear()
        st.rerun()


# =========================
# Main content
# =========================
st.title("📊 Portfolio Risk War Room Dashboard")
st.caption(
    "A lightweight dashboard to view risk metrics, VaR models, backtests, stress scenarios, and risk contribution outputs."
)

# Use mtimes so cached reads always refresh when files change
tables_dir_mtime = _dir_mtime(OUTPUTS_TABLES_DIR)
figures_dir_mtime = _dir_mtime(OUTPUTS_FIGURES_DIR)
report_latest_mtime = max([_safe_mtime(p) for p in REPORT_CANDIDATES] + [0.0])

if page == "Report":
    st.header("🧾 Report")
    content, searched_paths = read_report(report_latest_mtime)
    if content is None:
        st.warning("Missing report file. Looked for:")
        for s in searched_paths:
            st.write(f"- `{s}`")
        st.info(
            "Tip: run **Save + Rerun pipeline** to generate REPORT.md (or check your report builder)."
        )
    else:
        st.markdown(content)

elif page == "Tables":
    st.header("📋 Tables")
    files = list_tables(tables_dir_mtime)
    if not files:
        st.warning("No tables found in `outputs/tables/`.")
        st.info("Run **Save + Rerun pipeline** to generate output tables.")
    else:
        for p in files:
            st.subheader(p.name)
            df = read_csv_cached(str(p), _safe_mtime(p))
            st.dataframe(df, use_container_width=True)

elif page == "Figures":
    st.header("🖼️ Figures")
    files = list_figures(figures_dir_mtime)
    if not files:
        st.warning("No figures found in `outputs/figures/`.")
        st.info("Run **Save + Rerun pipeline** to generate output figures.")
    else:
        for p in files:
            st.subheader(p.name)
            st.image(str(p), use_container_width=True)

