from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


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
    return pd.read_csv(Path(path_str))


def show_csv(table_dir: Path, filename: str, title: str | None = None) -> None:
    p = table_dir / filename
    if title:
        st.markdown(f"**{title}**")
    if not p.exists():
        st.warning(f"Missing table: {p}")
        return
    df = load_csv_cached(str(p))
    st.dataframe(df, use_container_width=True)


def show_png(fig_dir: Path, filename: str, caption: str | None = None) -> None:
    p = fig_dir / filename
    if not p.exists():
        st.warning(f"Missing figure: {p}")
        return
    st.image(str(p), caption=caption, use_container_width=True)


def list_inventory(fig_dir: Path, table_dir: Path) -> tuple[list[str], list[str]]:
    figs = sorted([f.name for f in fig_dir.glob("*.png")]) if fig_dir.exists() else []
    tables = (
        sorted([t.name for t in table_dir.glob("*.csv")]) if table_dir.exists() else []
    )
    return figs, tables


def resolve_report_path(
    report_md_primary: Path, report_md_fallback: Path
) -> Path | None:
    if report_md_primary.exists():
        return report_md_primary
    if report_md_fallback.exists():
        return report_md_fallback
    return None


def resolve_image_path(
    base_dir: Path, fig_dir: Path, img_ref: str, report_file: Path
) -> Path | None:
    """
    img_ref could be:
      - "outputs/figures/var_compare.png"
      - "../outputs/figures/var_compare.png"
      - "var_compare.png"
    We try multiple sensible bases.
    """
    img_ref = img_ref.split("#")[0].split("?")[0].strip()

    candidates: list[Path] = []
    candidates.append((report_file.parent / img_ref).resolve())
    candidates.append((base_dir / img_ref).resolve())
    candidates.append((fig_dir / Path(img_ref).name).resolve())

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def render_report_md_with_images(
    base_dir: Path,
    fig_dir: Path,
    report_file: Path,
) -> None:
    """
    Render markdown, but convert markdown image lines to st.image
    so images show correctly on Streamlit Cloud.
    """
    md = safe_read_text(report_file)

    img_pat = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
    lines = md.splitlines()

    for line in lines:
        m = img_pat.search(line)
        if not m:
            st.markdown(line)
            continue

        alt = m.group("alt").strip() or None
        img_ref = m.group("path").strip()
        img_path = resolve_image_path(base_dir, fig_dir, img_ref, report_file)

        if img_path is None:
            st.markdown(line)  # fallback: show as text
        else:
            st.image(str(img_path), caption=alt, use_container_width=True)
