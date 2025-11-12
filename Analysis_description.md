# Analysis_ReadMe

## 0) What this toolkit expects

**Files (defaults):**

* `data/processed/clean_features.csv` — main wide table (one row per student).
* `data/processed/clean_features_terms_long.csv` *or* `data/processed/clean_features_long.csv` — per-term long table with `random_id` (and/or `merge_id`), `term_gpa`, and either `term_index` or `term_slot` like “1st Fall”, “2nd Spring”, etc.

**Outcome column (default):**

* `graduated_me` (binary).
  If you don’t have this exact column, the loader tries common aliases (e.g., `"ME_BS Degree Status"`). You can also override via `--outcome YOUR_COL`.

**Python deps:** `pandas`, `numpy` (Streamlit/Altair only needed for the app).

---

## 1) Folder layout

```
src/
  analysis/
    __init__.py                # empty — no side effects
    utils.py                   # paths, loaders, typing helpers, metrics
    validate.py                # dataset health & presence report
    effect_sizes.py            # numeric effects (Cohen’s d)
    risk_ratios.py             # categorical effects (risk ratios + CIs)
    bucketing.py               # quantile buckets → grad rate by bucket
    gpa_trajectory_trends.py   # per-student GPA slope + aggregates
    slices.py                  # slice-stability stress test (Simpson’s guard)
    trend_finder.py            # auto-build Top-10 candidate bullets/table
    run_all.py                 # orchestrates a full analysis run
apps/
  streamlit_etm_viewer.py      # interactive explorer (already in your repo)
data/
  processed/
    clean_features.csv
    clean_features_terms_long.csv  (or clean_features_long.csv)
    insights/                  # all outputs land here
```

---

## 2) One-command run (recommended)

From repo root:

```bash
python -m src.analysis.run_all
```

This runs in sequence:

1. `validate` → `_data_report.md`
2. `effect_sizes` → `ins_numeric_cohens_d.csv`
3. `risk_ratios` → `ins_categorical_rr.csv`
4. `bucketing` → `ins_bucketed_grad_rates.csv`
5. `gpa_trajectory_trends` → `ins_gpa_*` CSVs
6. `trend_finder` → `top_trends_table.csv` and `top_trends.md` (first-pass 10 bullets)

**All outputs** are written to: `data/processed/insights/`.

> If your long file is named `clean_features_long.csv`, everything still works; the loader tries both names. To be explicit:
>
> ```bash
> python -m src.analysis.gpa_trajectory_trends --long data/processed/clean_features_long.csv
> python -m src.analysis.trend_finder        --long data/processed/clean_features_long.csv
> ```

---

## 3) What each script does (and how to run it)

### `validate.py`

* **Purpose:** sanity report (row/col counts, top missingness, recommended columns present/missing, long-table status).
* **Run:**

  ```bash
  python -m src.analysis.validate
  ```
* **Output:** `insights/_data_report.md`

---

### `effect_sizes.py`

* **Purpose:** for every **numeric** feature, compute standardized difference between graduates vs non-graduates (Cohen’s d).
  Score = `|d| * sqrt(n1 + n0)` to prioritize strong and well-powered signals.
* **Run:**

  ```bash
  python -m src.analysis.effect_sizes --outcome graduated_me
  ```
* **Output:** `insights/ins_numeric_cohens_d.csv`
  **Key columns:** `feature, cohens_d, abs_d, mean_grad1, mean_grad0, n1, n0, score`

---

### `risk_ratios.py`

* **Purpose:** for every **categorical** feature (skipping ID-like/high-cardinality), compute **risk ratios** per level versus the rest with 95% log-normal CIs (Haldane–Anscombe smoothing).
* **Run:**

  ```bash
  python -m src.analysis.risk_ratios --outcome graduated_me
  ```
* **Output:** `insights/ins_categorical_rr.csv`
  **Key columns:** `feature, level, n_level, grad_rate_level, risk_ratio, rr_low, rr_high, abs_log_rr, score`

---

### `bucketing.py`

* **Purpose:** for numeric features, create quantile buckets and report **graduation rate per bucket** to reveal monotone or stepwise patterns.
* **Run:**

  ```bash
  python -m src.analysis.bucketing --outcome graduated_me
  ```
* **Output:** `insights/ins_bucketed_grad_rates.csv`
  **Key columns:** `feature, bucket, n, grad_rate, edges`

---

### `gpa_trajectory_trends.py`

* **Purpose:** from the long per-term table, compute per-student **GPA slope** (simple OLS slope of `term_gpa` over `term_index`), cohort aggregates, and slope vs graduation.
* **Run:**

  ```bash
  python -m src.analysis.gpa_trajectory_trends --outcome graduated_me
  ```

  (Auto-infers `term_index` from `term_slot` names if needed.)
* **Outputs:**

  * `ins_gpa_per_student.csv` — `random_id, gpa_slope[, graduated_me]`
  * `ins_gpa_aggregates.csv` — `n_students, median_slope, pct_positive_slope`
  * `ins_gpa_slope_vs_grad.csv` — grad rate for positive vs non-positive slope

---

### `slices.py`

* **Purpose:** **stress-test** any candidate feature’s association across likely confounders (e.g., `with_math_ap`, `sat_verb_grouping`, `summer_start`, `1st_term`). Helps detect Simpson’s paradox.
* **Run (example):**

  ```bash
  python -m src.analysis.slices --feature first_two_regular_mean_gpa \
    --confounders with_math_ap sat_verb_grouping summer_start 1st_term \
    --outcome graduated_me
  ```
* **Output:** `insights/slice_stability.csv`
  Inspect whether the **direction** (mean difference or RR) is consistent across slices.

---

### `trend_finder.py`

* **Purpose:** synthesize a **Top-10 candidate list** by combining the above:

  * Numeric effects (top 6 by score)
  * Categorical RRs (top 6 by score)
  * Early GPA bucket range (if present)
  * GPA slope vs grad (if long table present)
* **Run:**

  ```bash
  python -m src.analysis.trend_finder --outcome graduated_me
  ```
* **Outputs:**

  * `top_trends_table.csv` — combined ranking table with a `kind` column
  * `top_trends.md` — ready-to-paste bullet list (≤10 items)

---

### `run_all.py`

* **Purpose:** orchestration: runs everything above in a good order.
* **Run:**

  ```bash
  python -m src.analysis.run_all
  ```
* **Output:** All artifacts in `data/processed/insights/`.

---

## 4) How to pick your 10 trends (playbook)

1. **Start broad.**
   Open:

* `ins_numeric_cohens_d.csv` → flag items with **|d| ≥ 0.40** and `n1,n0 ≥ 50`.
* `ins_categorical_rr.csv` → flag levels with **RR ≥ 1.25 or ≤ 0.80**, CI not crossing 1, `n_level ≥ 50`.
* `ins_bucketed_grad_rates.csv` → seek **monotone** or clearly separated buckets.
* `ins_gpa_slope_vs_grad.csv` → check if **positive slope** group outperforms.

2. **Stress-test.**
   For each top candidate feature `X`, run `slices.py` across 3–4 confounders. A robust trend **persists** (directionally) in most slices.

3. **Finalize 10.**
   Balance your list:

* ~4 numeric, ~4 categorical, ~2 trajectory insights.
* Prefer **actionable** and **persistent** signals.

4. **Narrative.**
   Use `top_trends.md` as the base. Add cohort qualifiers (e.g., “Entrants 2018–2022”) and any slicing context (“persists across AP-Math and Summer-Start groups”).

---

## 5) Interpreting key columns

* **Cohen’s d (numeric):**
  `0.2` small, `0.5` medium, `0.8` large (rule of thumb). Positive sign = graduates have higher mean.

* **Risk Ratio (categorical):**
  `RR = 1.40` → level has 40% higher graduation probability than the rest (approx).
  Check `rr_low/rr_high`; if they straddle `1.0`, it’s not statistically distinct.

* **Score (both):**
  Magnitude × √(sample size) → ranks well-powered, strong effects.

* **Buckets:**
  Look for clear separation between adjacent buckets or a steady trend.

* **GPA slope:**
  Proportion with positive slope and the grad-rate gap between positive vs non-positive slopes.

---

## 6) Streamlit integration (optional quick hook)

After you run the analysis once, you can surface results in the app:

* Read and display `data/processed/insights/top_trends.md`.
* Load `ins_numeric_cohens_d.csv` / `ins_categorical_rr.csv` into sortable tables.
* Add a selector to trigger `slices.py` on a chosen feature and display the slice table.

(If you want, I can drop a small `apps/streamlit_insights.py` for this panel.)

---

## 7) Customization knobs

* **Outcome column:** pass `--outcome YOUR_COL` to any script, or change `OUTCOME_COL` in `utils.py`.
* **Minimum cohort size:** change `MIN_N` in `utils.py`.
* **Categorical high-cardinality cutoff:** `MAX_LEVELS_CAT` in `utils.py`.
* **Bucket quantiles:** pass `--q` in `bucketing.py` (or edit defaults).

---

## 8) Troubleshooting

**Circular import on `gpa_trajectory_trends`:**

* Ensure `src/analysis/__init__.py` is empty (no imports).
* In `trend_finder.py`, **do not** import `compute_gpa_trends` at top-level. Import it **inside** `summarize_top10`.
* In `gpa_trajectory_trends.py`, remove any line that imports itself, like:

  ```python
  from .gpa_trajectory_trends import compute_gpa_trends  # ← delete this
  ```
* Clear caches:

  ```bash
  find src -name "__pycache__" -type d -exec rm -rf {} +
  find . -name "*.pyc" -delete
  ```

**Long file not found:**

* Place either `clean_features_terms_long.csv` or `clean_features_long.csv` in `data/processed/`, or pass `--long PATH`.

**Outcome missing / not binary:**

* Ensure your outcome column exists and is binary; text values like “Yes/No” or “Graduated/Not” are auto-coerced. Otherwise, specify `--outcome`.

**ID columns inflating categorical scan:**

* The scanner skips ID-like columns automatically; if one slips through, add it to `exclude` lists in the code or rename.

---

## 9) Example command set

```bash
# Health check
python -m src.analysis.validate

# Individual modules
python -m src.analysis.effect_sizes --outcome graduated_me
python -m src.analysis.risk_ratios  --outcome graduated_me
python -m src.analysis.bucketing    --outcome graduated_me
python -m src.analysis.gpa_trajectory_trends --outcome graduated_me

# Trend synthesis
python -m src.analysis.trend_finder --outcome graduated_me

# Full pipeline
python -m src.analysis.run_all

# Slice stress-test for a chosen feature
python -m src.analysis.slices --feature first_two_regular_mean_gpa \
  --confounders with_math_ap sat_verb_grouping summer_start 1st_term
```

---

## 10) What “100 trends” might look like (template)

When you’re ready to publish, your `top_trends.md` will read like:

* Students who graduate have **higher first_two_regular_mean_gpa** (Cohen’s d = 0.62, n1=…, n0=…).
* Being **summer_start = Yes** is associated with **lower** graduation odds (RR = 0.78, 95% CI [0.72, 0.85], n=…).
* Early GPA buckets discriminate: grad rate ranges from **32%** in `[2.00, 2.50)` to **88%** in `[3.50, 4.10)`.
* **Positive GPA slope** ⇢ higher graduation rates (…% vs …%).
* …

Edit the numbers from your outputs and add cohort qualifiers.


