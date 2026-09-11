#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_app.py -- Streamlit front-end for saxs_autofit.py.

Upload a .zip of .dat files, it is extracted under ./tmp (kept there for
reuse in this and future sessions), then the batch SAXS fitting pipeline
from saxs_autofit.py runs on the selected extracted dataset. Final
per-group CSV tables and parameter-vs-time plots are shown below.
"""

from __future__ import annotations

import os
import zipfile

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_DIR, "tmp")
OUT_ROOT = os.path.join(BASE_DIR, "tmp_output")

import saxs_autofit as sa  # noqa: E402

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUT_ROOT, exist_ok=True)

st.set_page_config(page_title="SAXS Auto-Fit", layout="wide")
st.title("SAXS Autofit")
st.caption("codes/saxs_autofit.py 的 Streamlit 前端")


# ============================================================================
# 1. Upload & extract
# ============================================================================

st.header("1. 上傳資料")

uploaded = st.file_uploader("上傳含 .dat 檔案的 .zip", type="zip")
if uploaded is not None:
    dataset_name = os.path.splitext(uploaded.name)[0]
    dest = os.path.join(TMP_DIR, dataset_name)
    with st.spinner(f"解壓縮至 tmp/{dataset_name} ..."):
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(uploaded) as zf:
            zf.extractall(dest)
    st.success(f"已解壓縮至 tmp/{dataset_name}，之後可重複使用此資料集。")

datasets = sorted(
    d for d in os.listdir(TMP_DIR) if os.path.isdir(os.path.join(TMP_DIR, d))
)
if not datasets:
    st.info("尚無資料集，請先上傳一個 .zip 檔案。")
    st.stop()

st.header("2. 選擇資料集")
dataset = st.selectbox("資料集", datasets,
                        index=len(datasets) - 1)
input_dir = os.path.join(TMP_DIR, dataset)
out_dir = os.path.join(OUT_ROOT, dataset)


# ============================================================================
# 3. Options
# ============================================================================

st.header("3. 選項")

max_cpu = os.cpu_count() or 1
col1, col2 = st.columns(2)

with col1:
    shapes = st.multiselect("候選形狀", sa.SHAPES, default=sa.SHAPES)
    force_shape = st.selectbox("強制指定形狀", ["(自動偵測)"] + sa.SHAPES)
    shape_frame = st.radio("形狀偵測用的參考幀", ["last", "first"],
                            horizontal=True)
    batch_len = st.number_input("每個 batch 的秒數", min_value=1,
                                 value=150, step=1)
    length_unit = st.text_input("長度單位標籤", value="A")

with col2:
    max_npop = st.selectbox("最大群體數", [1, 2], index=1)
    pop2_threshold = st.slider("使用 Population-2 的 chi^2 閾值",
                                min_value=0.5, max_value=1.0, value=0.9, step=0.01)
    nr = st.number_input("Schulz 積分節點數 (nr)", min_value=51, value=201, step=10)
    jobs = st.slider("平行工作程序數", min_value=1,
                      max_value=max_cpu, value=min(50, max_cpu), step=1,
                      help=f"此裝置最大線程數為 {max_cpu}")

if not shapes:
    st.warning("請至少選擇一個候選形狀。")
    st.stop()


# ============================================================================
# 4. Run pipeline
# ============================================================================

st.header("4. 執行擬合")

run_clicked = st.button("開始執行", type="primary")

if run_clicked:
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, ".master_cache")

    groups, skipped = sa.group_files(input_dir, int(batch_len))
    if not groups:
        st.error("找不到符合 [Title]_[batch]_[second].dat 命名規則的檔案。")
        st.stop()
    if skipped:
        st.warning(f"跳過 {len(skipped)} 個不符合命名規則的檔案。")

    master_cache: dict = {}
    log = st.expander("執行紀錄", expanded=True)
    progress = st.progress(0.0, text="準備中 ...")

    with log:
        st.write(f"共 {len(groups)} 個群組: {', '.join(groups.keys())}")

    shapes_needed = [sa.canon_shape(force_shape)] if force_shape != "(自動偵測)" else \
        [sa.canon_shape(s) for s in shapes]
    with log:
        st.write("計算 master form factor 曲線 ...")
    sa.warm_masters(shapes_needed, master_cache, cache_dir, int(jobs), verbose=False)

    group_dfs = {}
    n_groups = len(groups)
    for gi, (title, frames) in enumerate(groups.items()):
        with log:
            st.write(f"=== 群組 '{title}': {len(frames)} 幀 ===")

        if force_shape != "(自動偵測)":
            shape = sa.canon_shape(force_shape)
            with log:
                st.write(f"使用指定形狀: {shape}")
        else:
            ref = frames[-1] if shape_frame == "last" else frames[0]
            q, I, sig, _ = sa.load_frame(ref[3])
            candidate_shapes = [sa.canon_shape(s) for s in shapes]
            shape, scores = sa.auto_select_shape(
                q, I, sig, candidate_shapes, master_cache,
                cache_dir=cache_dir, Nr=int(nr), verbose=False)
            with log:
                st.write(f"自動偵測形狀: {shape}  (chi2red: "
                         f"{', '.join(f'{s}={c:.3g}' for s, c in scores.items())})")

        df = sa.run_group(title, frames, shape, master_cache, out_dir,
                           length_unit, int(max_npop), float(pop2_threshold),
                           cache_dir, Nr=int(nr), verbose=False, jobs=int(jobs))
        group_dfs[title] = (shape, df)
        progress.progress((gi + 1) / n_groups, text=f"完成群組 {title}")

    st.session_state["results"] = {"out_dir": out_dir, "groups": group_dfs}
    progress.progress(1.0, text="完成")
    st.success(f"擬合完成，結果已寫入 tmp_output/{dataset}")


# ============================================================================
# 5. Results: CSV table + charts
# ============================================================================

def _load_existing_results(out_dir: str):
    """Fall back to reading previously-written CSVs if nothing is in session_state
    yet (e.g. after a page reload) for the currently selected dataset."""
    if not os.path.isdir(out_dir):
        return {}
    groups = {}
    for name in sorted(os.listdir(out_dir)):
        grp_dir = os.path.join(out_dir, name)
        csvs = [f for f in os.listdir(grp_dir)] if os.path.isdir(grp_dir) else []
        csv_file = next((f for f in csvs if f.endswith("_fit_results.csv")), None)
        if not csv_file:
            continue
        df = pd.read_csv(os.path.join(grp_dir, csv_file))
        shape = df["shape"].iloc[0] if "shape" in df.columns and len(df) else "?"
        groups[name] = (shape, df)
    return groups


results = st.session_state.get("results")
if results and results["out_dir"] == out_dir:
    group_dfs = results["groups"]
else:
    group_dfs = _load_existing_results(out_dir)

if group_dfs:
    st.header("5. 結果")
    for title, (shape, df) in group_dfs.items():
        st.subheader(f"群組: {title}  (形狀: {shape})")

        st.dataframe(df, use_container_width=True)
        st.download_button(
            f"下載 {title} 的 CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{title}_fit_results.csv",
            mime="text/csv",
            key=f"dl_{title}",
        )

        ok = df[df["error"].fillna("") == ""] if "error" in df.columns else df
        if ok.empty:
            st.warning("此群組沒有成功的擬合結果可繪圖。")
            continue

        chart_df = ok.set_index("time_s")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption(f"R0 ({length_unit}) vs time")
            cols = [c for c in ("R0_1", "R0_2") if chart_df[c].notna().any()]
            st.line_chart(chart_df[cols])
        with c2:
            st.caption("PD vs time")
            cols = [c for c in ("PD_1", "PD_2") if chart_df[c].notna().any()]
            st.line_chart(chart_df[cols])
        with c3:
            st.caption(f"Rg ({length_unit}) vs time")
            cols = [c for c in ("Rg_1", "Rg_2", "RgTot") if chart_df[c].notna().any()]
            st.line_chart(chart_df[cols])

        c4, c5, c6 = st.columns(3)
        with c4:
            st.caption("A (scale) vs time")
            cols = [c for c in ("A_1", "A_2") if chart_df[c].notna().any()]
            st.line_chart(chart_df[cols])
        with c5:
            st.caption("fraction of I(0) (%) vs time")
            cols = [c for c in ("frac_1", "frac_2") if chart_df[c].notna().any()]
            st.line_chart(chart_df[cols])
        with c6:
            st.caption("reduced chi^2 vs time")
            st.line_chart(chart_df[["chi2red"]])

        plots_dir = os.path.join(out_dir, title, "plots")
        if os.path.isdir(plots_dir):
            with st.expander(f"{title} 的擬合圖 (matplotlib)"):
                pngs = sorted(f for f in os.listdir(plots_dir) if f.endswith(".png"))
                img_cols = st.columns(3)
                for i, png in enumerate(pngs):
                    with img_cols[i % 3]:
                        st.image(os.path.join(plots_dir, png))
else:
    st.info("尚未有可顯示的結果，請執行擬合。")
