# apps/streamlit_etm_viewer.py
# ETM Clean Features Explorer (Streamlit)
# - Loads clean_features.csv (+ optional per-term long CSV)
# - Powerful filtering + KPIs
# - Correlations, cohort gaps, GPA trajectories with CIs
# - Gateway-course mastery and relationship charts
# - Export filtered subset

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import altair as alt

# -------------------- Page setup
st.set_page_config(page_title="ETM Clean Features Explorer", layout="wide")
st.title("ETM Clean Features — Explorer")

# -------------------- Load data
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

# Optional long (per-term) data
long_default_path = Path("data/processed/clean_features_terms_long.csv")
uploaded_long = st.file_uploader("Upload clean_features_terms_long.csv (optional)", type=["csv"], key="long")

if uploaded_long is not None:
    df_long = pd.read_csv(uploaded_long)
    long_path = "<uploaded long file>"
elif long_default_path.exists():
    df_long = pd.read_csv(long_default_path)
    long_path = str(long_default_path)
else:
    df_long = None
    long_path = None

# -------------------- Helpers
def safe_mean(series) -> float:
    try:
        return float(np.nanmean(series))
    except Exception:
        return float("nan")

def join_long_with_master(df_long: pd.DataFrame, df_master: pd.DataFrame, extra_cols=None) -> pd.DataFrame | None:
    """Join per-term long table with master (for cohort variables). Uses merge_id if available, else random_id."""
    if df_long is None:
        return None
    join_key = None
    if "merge_id" in df_long.columns and "merge_id" in df_master.columns:
        join_key = "merge_id"
    elif "random_id" in df_long.columns and "random_id" in df_master.columns:
        join_key = "random_id"
    else:
        return None

    base_cols = [join_key]
    if "graduated_me" in df_master.columns:
        base_cols.append("graduated_me")
    if extra_cols:
        for c in extra_cols:
            if c in df_master.columns and c not in base_cols:
                base_cols.append(c)
    return df_long.merge(df_master[base_cols].drop_duplicates(), on=join_key, how="left")

# Harmonize IDs in long if master has merge_id but long does not
if df_long is not None:
    if "merge_id" in df.columns and "merge_id" not in df_long.columns and "random_id" in df_long.columns and "random_id" in df.columns:
        id_map = df[["random_id", "merge_id"]].dropna().drop_duplicates()
        df_long = df_long.merge(id_map, on="random_id", how="left")

# -------------------- Column metadata
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

# -------------------- Sidebar filters
st.sidebar.header("Filters")

# Visible columns selection
visible_cols = st.sidebar.multiselect(
    "Columns to display",
    options=list(df.columns),
    default=list(df.columns)[:20]
)

filtered = df.copy()

# Random ID search (supports contains/exact)
if "random_id" in df.columns:
    rid_mode = st.sidebar.radio("Random ID search", ["Disabled", "Contains", "Exact"], horizontal=True)
    rid_query = st.sidebar.text_input("Random ID")
    if rid_query and rid_mode != "Disabled":
        s = filtered["random_id"].astype(str)
        if rid_mode == "Contains":
            filtered = filtered[s.str.contains(rid_query, na=False)]
        else:
            filtered = filtered[s == rid_query]

# Numeric range sliders (user picks which)
choose_num = st.sidebar.multiselect("Numeric filters (range sliders)", options=num_cols, default=[])
for c in choose_num:
    s = filtered[c].dropna()
    if s.empty:
        continue
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    r = st.sidebar.slider(f"{c}", min_value=float(lo), max_value=float(hi), value=(float(lo), float(hi)))
    filtered = filtered[(filtered[c] >= r[0]) & (filtered[c] <= r[1])]

# Categorical multiselect filters (user picks which)
choose_cat = st.sidebar.multiselect("Categorical filters", options=cat_cols, default=[])
for c in choose_cat:
    opts = filtered[c].astype("string").fillna("<NA>").value_counts().index.tolist()
    sel = st.sidebar.multiselect(f"{c}", options=opts, default=[])
    if sel:
        filtered = filtered[filtered[c].astype("string").fillna("<NA>").isin(sel)]

# -------------------- KPIs
st.subheader("Key metrics")
kpi_cols = st.columns(4)

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

# -------------------- Cohort KPIs & gaps
st.subheader("Cohort KPIs")
left, right = st.columns(2)

target_ok = "graduated_me" in filtered.columns
with left:
    if target_ok:
        a = filtered[filtered["graduated_me"] == 1]
        b = filtered[filtered["graduated_me"] == 0]
        def m(series): return float(np.nanmean(series)) if series.size else np.nan
        kpi_df = pd.DataFrame({
            "metric": ["count", "ETM cGPA", "Graduating cGPA", "# ETM attempts to ABC"],
            "Graduated=1": [
                len(a), m(a.get("cgpa_at_etm_to_any_campus", pd.Series(dtype=float))),
                m(a.get("graduating_cgpa", pd.Series(dtype=float))),
                m(a.get("etm_total_attempts_to_abc", pd.Series(dtype=float))),
            ],
            "Graduated=0": [
                len(b), m(b.get("cgpa_at_etm_to_any_campus", pd.Series(dtype=float))),
                m(b.get("graduating_cgpa", pd.Series(dtype=float))),
                m(b.get("etm_total_attempts_to_abc", pd.Series(dtype=float))),
            ],
        })
        st.dataframe(kpi_df, use_container_width=True, height=180)
    else:
        st.info("Add a binary outcome column like `graduated_me` to enable cohort KPIs.")

with right:
    if target_ok:
        num = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
        show = []
        for c in num:
            g1 = filtered.loc[filtered["graduated_me"] == 1, c]
            g0 = filtered.loc[filtered["graduated_me"] == 0, c]
            if g1.notna().sum() >= 20 and g0.notna().sum() >= 20:
                mu1, mu0 = np.nanmean(g1), np.nanmean(g0)
                s1, s0 = np.nanstd(g1, ddof=1), np.nanstd(g0, ddof=1)
                sp = np.sqrt(((s1**2) + (s0**2)) / 2.0)
                gap = (mu1 - mu0) / sp if sp > 0 else np.nan
                show.append((c, gap, mu1, mu0))
        if show:
            gap_df = (pd.DataFrame(show, columns=["feature","std_gap","mean_graduated","mean_not"])
                        .assign(abs_gap=lambda d: d["std_gap"].abs())
                        .sort_values("abs_gap", ascending=False).head(12))
            st.dataframe(gap_df.drop(columns=["abs_gap"]), use_container_width=True, height=220)
        else:
            st.info("Not enough data to compute gaps.")

# -------------------- Data preview
st.subheader("Data preview")
st.dataframe(filtered[visible_cols] if visible_cols else filtered, use_container_width=True, height=450)

# -------------------- Quick charts
st.subheader("Quick charts")
tab_hist, tab_bar = st.tabs(["Histogram (numeric)", "Bar (categorical)"])

with tab_hist:
    num_for_hist = st.selectbox("Numeric column", options=(num_cols or ["<none>"]))
    if num_for_hist and num_for_hist in filtered.columns and pd.api.types.is_numeric_dtype(filtered[num_for_hist]):
        vals = filtered[num_for_hist].dropna().values
        if vals.size > 0:
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

# -------------------- Correlation heatmap
st.subheader("Correlation heatmap (selected numeric features)")
corr_defaults = [
    "cgpa_at_etm_to_any_campus","graduating_cgpa","warnings_per_term",
    "low_gpa_term_rate_2_5","low_gpa_term_rate_3_0","peak_minus_etm",
    "grade_forgiveness_used","multi_repeat_flag","terms_with_gpa",
    "mean_term_gpa","gpa_trend_slope","first_two_regular_mean_gpa","summer_term_ratio"
]
corr_pick = st.multiselect(
    "Pick up to ~20 numeric features",
    options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
    default=[c for c in corr_defaults if c in df.columns][:12]
)
if corr_pick:
    cmat = df[corr_pick].corr()
    melt = cmat.reset_index().melt("index", var_name="col2", value_name="corr").rename(columns={"index":"col1"})
    chart = alt.Chart(melt).mark_rect().encode(
        x=alt.X("col1:O", sort=corr_pick, title=""),
        y=alt.Y("col2:O", sort=corr_pick, title=""),
        color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue"), title="r"),
        tooltip=["col1","col2",alt.Tooltip("corr:Q", format=".2f")]
    ).properties(height=300, width=min(700, 30*len(corr_pick)))
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Select some numeric columns to see correlations.")

# -------------------- GPA trajectories (per term)
st.subheader("GPA trajectories (per term)")
if df_long is None:
    st.caption("Tip: drop clean_features_terms_long.csv here to enable per-term charts.")
else:
    st.caption(f"Loaded long data: {long_path} — {df_long.shape[0]} rows")

    # Single-student trajectory (accept merge_id or random_id)
    st.caption("Enter a Random ID (B-… / S-…) or merged ID if present.")
    rid_for_traj = st.text_input("ID for trajectory", value="")
    if rid_for_traj:
        if "merge_id" in df_long.columns:
            key_candidates = ["merge_id", "random_id"]
            key = next((k for k in key_candidates if rid_for_traj in df_long[k].astype(str).unique()), "random_id")
        else:
            key = "random_id"
        traj = df_long[df_long[key].astype(str) == str(rid_for_traj)].sort_values("term_index")
        if traj.empty:
            st.info("No per-term rows for that ID.")
        else:
            st.line_chart(traj.set_index("term_index")["term_gpa"])
            st.dataframe(traj[["term_index", "term_slot", "term_gpa"]], use_container_width=True, height=220)

    # Cohort averages with 95% CI
    st.markdown("#### Cohort GPA trend (mean ± 95% CI)")
    long_join = join_long_with_master(
        df_long, df,
        extra_cols=["cgpa_at_etm_to_any_campus","first_two_regular_mean_gpa","with_math_ap","sat_verb_grouping"]
    )
    if long_join is None or long_join.empty:
        st.info("No long data after join.")
    else:
        cohortable = [c for c in ["graduated_me","with_math_ap","sat_verb_grouping"] if c in long_join.columns]
        if cohortable:
            cohort_var = st.selectbox("Cohort split", options=cohortable, index=0)
            agg = (long_join.groupby(["term_index", cohort_var], dropna=False)["term_gpa"]
                          .agg(["count","mean","std"]).reset_index())
            agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            agg["lo"] = agg["mean"] - 1.96*agg["se"]
            agg["hi"] = agg["mean"] + 1.96*agg["se"]

            base = alt.Chart(agg).encode(x=alt.X("term_index:O", title="Term order"))
            band = base.mark_area(opacity=0.25).encode(
                y="lo:Q", y2="hi:Q", color=alt.Color(f"{cohort_var}:N", title=cohort_var)
            )
            line = base.mark_line(point=True).encode(
                y=alt.Y("mean:Q", title="Mean term GPA"),
                color=alt.Color(f"{cohort_var}:N", title=cohort_var),
                tooltip=[cohort_var,"term_index",alt.Tooltip("mean:Q", format=".2f"),"count"]
            )
            st.altair_chart(band + line, use_container_width=True)
        else:
            st.info("No cohort columns available for trend split.")

# -------------------- Gateway courses — mastery dashboard
st.subheader("Gateway courses — mastery overview")
courses = ["chem_110","edsgn_100","math_140","math_141","phys_211"]
present_courses = [c for c in courses if any(col.startswith(c) for col in df.columns)]
if not present_courses:
    st.info("No gateway-course columns found.")
else:
    rows = []
    for c in present_courses:
        pass1 = f"{c}_pass_by_first_attempt"
        ever  = f"{c}_ever_passed"
        att   = f"{c}_total_attempts"
        outc  = f"{c}_outcome"

        if pass1 in df.columns and pd.api.types.is_numeric_dtype(df[pass1]):
            p1 = float(np.nanmean(df[pass1]))
        elif outc in df.columns:
            p1 = float((df[outc].astype(str) == "Pass on 1st attempt").mean())
        else:
            p1 = np.nan

        if ever in df.columns and pd.api.types.is_numeric_dtype(df[ever]):
            ep = float(np.nanmean(df[ever]))
        elif outc in df.columns:
            ep = float(df[outc].astype(str).str.contains("Pass", na=False).mean())
        else:
            ep = np.nan

        att_mean = float(np.nanmean(df[att])) if att in df.columns else np.nan

        rows.append({"course": c.upper(), "pass_1st": p1, "ever_pass": ep, "mean_attempts": att_mean})

    course_df = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        chart = alt.Chart(course_df).mark_bar().encode(
            x=alt.X("course:N", title="Course"),
            y=alt.Y("pass_1st:Q", axis=alt.Axis(format="%"), title="Pass on 1st attempt"),
            tooltip=["course", alt.Tooltip("pass_1st:Q", format=".1%")]
        )
        st.altair_chart(chart, use_container_width=True)
    with c2:
        chart2 = alt.Chart(course_df).mark_bar().encode(
            x=alt.X("course:N", title="Course"),
            y=alt.Y("ever_pass:Q", axis=alt.Axis(format="%"), title="Ever passed"),
            tooltip=["course", alt.Tooltip("ever_pass:Q", format=".1%")]
        )
        st.altair_chart(chart2, use_container_width=True)

    st.caption("Mean attempts to reach ABC:")
    st.bar_chart(course_df.set_index("course")["mean_attempts"])

# -------------------- Relationships
st.subheader("Relationships")

# Scatter: warnings vs graduating GPA (colored by graduation)
if all(c in filtered.columns for c in ["_warning", "graduating_cgpa"]):
    tmp = filtered[["_warning","graduating_cgpa"]].copy()
    if "graduated_me" in filtered.columns:
        tmp["graduated_me"] = filtered["graduated_me"]
    scat = alt.Chart(tmp).mark_circle(opacity=0.5).encode(
        x=alt.X("_warning:Q", title="# Warnings"),
        y=alt.Y("graduating_cgpa:Q", title="Graduating cGPA"),
        color="graduated_me:N" if "graduated_me" in tmp.columns else alt.value("#1f77b4"),
        tooltip=list(tmp.columns)
    ).properties(height=300)
    st.altair_chart(scat, use_container_width=True)

# Binned relationship: early GPA vs graduating GPA with CI bars
if "first_two_regular_mean_gpa" in filtered.columns and "graduating_cgpa" in filtered.columns:
    d2 = filtered[["first_two_regular_mean_gpa","graduating_cgpa"]].dropna().copy()
    if not d2.empty:
        d2["bin"] = pd.cut(d2["first_two_regular_mean_gpa"],
                           bins=[0,2.0,2.5,3.0,3.25,3.5,3.75,4.1])
        agg = d2.groupby("bin")["graduating_cgpa"].agg(["count","mean","std"]).reset_index()
        agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        agg["lo"] = agg["mean"] - 1.96*agg["se"]
        agg["hi"] = agg["mean"] + 1.96*agg["se"]
        base = alt.Chart(agg).encode(x=alt.X("bin:N", title="First two regular term mean (binned)"))
        st.altair_chart(
            base.mark_bar().encode(y=alt.Y("mean:Q", title="Graduating cGPA")) +
            base.mark_rule(color="black").encode(y="lo:Q", y2="hi:Q"),
            use_container_width=True
        )

# -------------------- Export filtered
st.download_button(
    label="⬇️ Download filtered CSV",
    data=filtered.to_csv(index=False),
    file_name="clean_features_filtered.csv",
    mime="text/csv",
)
