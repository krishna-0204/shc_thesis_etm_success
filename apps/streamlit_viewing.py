# apps/streamlit_etm_viewer.py
# ETM Clean Features Explorer (Streamlit)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="ETM Clean Features Explorer", layout="wide")
st.title("📊 ETM Clean Features — Explorer")

# ---------- Load data
default_path = Path("data/processed/clean_features.csv")
uploaded = st.file_uploader("Upload a clean_features.csv (optional)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    data_path = "<uploaded file>"
elif default_path.exists():
    df = pd.read_csv(default_path)
    data_path = str(default_path)
else:
    st.error("No data found. Upload a CSV or place it at data/processed/clean_features.csv")
    st.stop()

st.caption(f"Loaded: {data_path} — shape: {df.shape[0]} rows × {df.shape[1]} cols")

# ---------- Column metadata
dtypes = df.dtypes.astype(str)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in num_cols]

with st.expander("ℹ️ Column types & missingness"):
    st.write(
        pd.DataFrame({
            "dtype": dtypes,
            "missing_pct": df.isna().mean().round(3)
        }).sort_index()
    )

# ---------- Sidebar filters
st.sidebar.header("Filters")

# Visible columns selection
visible_cols = st.sidebar.multiselect(
    "Columns to display",
    options=list(df.columns),
    default=list(df.columns)[:20]
)

filtered = df.copy()

# Random ID search (if present)
if "random_id" in df.columns:
    rid_mode = st.sidebar.radio("Random ID search", ["Disabled", "Contains", "Exact"], horizontal=True)
    rid_query = st.sidebar.text_input("Random ID")
    if rid_query and rid_mode != "Disabled":
        s = filtered["random_id"].astype(str)
        if rid_mode == "Contains":
            filtered = filtered[s.str.contains(rid_query, na=False)]
        else:
            filtered = filtered[s == rid_query]

# Numeric range filters (pick specific columns to avoid UI clutter)
choose_num = st.sidebar.multiselect("Numeric filters (range sliders)", options=num_cols, default=[])
for c in choose_num:
    s = filtered[c].dropna()
    if s.empty:
        continue
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    r = st.sidebar.slider(f"{c}", min_value=float(lo), max_value=float(hi), value=(float(lo), float(hi)))
    filtered = filtered[(filtered[c] >= r[0]) & (filtered[c] <= r[1])]

# Categorical multiselect filters
choose_cat = st.sidebar.multiselect("Categorical filters", options=cat_cols, default=[])
for c in choose_cat:
    opts = filtered[c].astype("string").fillna("<NA>").value_counts().index.tolist()
    sel = st.sidebar.multiselect(f"{c}", options=opts, default=[])
    if sel:
        filtered = filtered[filtered[c].astype("string").fillna("<NA>").isin(sel)]

# ---------- KPIs
st.subheader("Key metrics")
kpi_cols = st.columns(4)

def safe_mean(series):
    try:
        return float(np.nanmean(series))
    except Exception:
        return float("nan")

if "graduated_me" in filtered.columns:
    grad_rate = safe_mean(filtered["graduated_me"]) * 100.0
else:
    grad_rate = float("nan")

with kpi_cols[0]:
    st.metric("Graduated ME (%)", f"{grad_rate:.1f}" if not np.isnan(grad_rate) else "n/a")

for i, col in enumerate(["cgpa_at_etm_to_any_campus", "graduating_cgpa", "etm_total_attempts_to_abc"]):
    with kpi_cols[i + 1]:
        val = safe_mean(filtered[col]) if col in filtered.columns else float("nan")
        st.metric(col, f"{val:.3f}" if not np.isnan(val) else "n/a")

# ---------- Data preview
st.subheader("Data preview")
st.dataframe(filtered[visible_cols] if visible_cols else filtered, use_container_width=True, height=450)

# ---------- Quick charts
st.subheader("Quick charts")
tab_hist, tab_bar = st.tabs(["Histogram (numeric)", "Bar (categorical)"])

with tab_hist:
    num_for_hist = st.selectbox("Numeric column", options=(num_cols or ["<none>"]))
    if num_for_hist and num_for_hist in filtered.columns and pd.api.types.is_numeric_dtype(filtered[num_for_hist]):
        vals = filtered[num_for_hist].dropna().values
        if vals.size > 0:
            # Use NumPy histogram; show midpoints on x-axis
            counts, bin_edges = np.histogram(vals, bins=30)
            mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            hist_df = pd.DataFrame({"bin": mids, "count": counts})
            st.bar_chart(hist_df.set_index("bin"))
        else:
            st.info("No data to plot for this selection.")
    else:
        st.info("Pick a numeric column to see a histogram.")

with tab_bar:
    cat_for_bar = st.selectbox("Categorical column", options=(cat_cols or ["<none>"]))
    if cat_for_bar and cat_for_bar in filtered.columns:
        vc = filtered[cat_for_bar].astype("string").fillna("<NA>").value_counts().head(30)
        st.bar_chart(vc)
    else:
        st.info("Pick a categorical column to see top counts.")

# ---------- Export filtered
st.download_button(
    label="⬇️ Download filtered CSV",
    data=filtered.to_csv(index=False),
    file_name="clean_features_filtered.csv",
    mime="text/csv",
)
