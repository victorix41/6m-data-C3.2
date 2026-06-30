"""
HDB Resale Price Modeling — Results Dashboard
================================================
Streamlit app that reproduces and visualizes the modeling pipeline from
unsupervised_learning.ipynb (Part C baseline -> Option 1/2/3 -> Part D Stacking).

Run with:
    streamlit run app.py

Expects the dataset at:
    data/Resale_flat_prices_based_on_registration_date_from_Jan-2017_onwards.csv
(relative to this script's location)
"""

import re
import time
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="HDB Resale Price Modeling",
    page_icon="🏠",
    layout="wide",
)

sns.set_theme(style="whitegrid", palette="viridis")
PALETTE = "viridis"

DATA_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "data",
                                  "Resale_flat_prices_based_on_registration_date_from_Jan-2017_onwards.csv")

FEATURES_BASE = ["floor_area_sqm", "town"]
FEATURES_3 = ["floor_area_sqm", "town", "remaining_lease_years"]


# ----------------------------------------------------------------------------
# Data loading & feature engineering
# ----------------------------------------------------------------------------
def parse_lease_to_years(lease_str: str) -> float:
    """'61 years 04 months' -> 61.33 (numeric years, for use as a model feature)"""
    years = re.search(r"(\d+)\s*year", str(lease_str))
    months = re.search(r"(\d+)\s*month", str(lease_str))
    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0
    return y + m / 12


@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["remaining_lease_years"] = df["remaining_lease"].apply(parse_lease_to_years)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    return df


def find_default_csv() -> str | None:
    if os.path.exists(DATA_PATH_DEFAULT):
        return DATA_PATH_DEFAULT
    candidates = glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.csv"))
    return candidates[0] if candidates else None


# ----------------------------------------------------------------------------
# Model training (cached so the ~3 min single-core training only runs once)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training all 6 models (this runs once and is cached)...")
def train_all_models(df: pd.DataFrame):
    X = df[FEATURES_3]
    y = df["resale_price"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42
    )

    preproc_2feat = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["town"])],
        remainder="passthrough",
    )
    preproc_3feat = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["town"])],
        remainder="passthrough",
    )

    results = {}
    predictions = {}

    # --- Baseline: floor_area_sqm + town -> LinearRegression ---
    t0 = time.time()
    base_pipe = Pipeline([("prep", preproc_2feat), ("model", LinearRegression())])
    base_pipe.fit(X_train[FEATURES_BASE], y_train)
    pred_base = base_pipe.predict(X_test[FEATURES_BASE])
    results["Baseline (2 feat) LR"] = {
        "MAE": mean_absolute_error(y_test, pred_base),
        "MAPE": mean_absolute_percentage_error(y_test, pred_base),
        "R2": r2_score(y_test, pred_base),
        "fit_time": time.time() - t0,
    }
    predictions["Baseline (2 feat) LR"] = pred_base

    # --- Option 1: + remaining_lease_years -> LinearRegression ---
    t0 = time.time()
    opt1_pipe = Pipeline([("prep", preproc_3feat), ("model", LinearRegression())])
    opt1_pipe.fit(X_train, y_train)
    pred_opt1 = opt1_pipe.predict(X_test)
    results["+ remaining_lease LR"] = {
        "MAE": mean_absolute_error(y_test, pred_opt1),
        "MAPE": mean_absolute_percentage_error(y_test, pred_opt1),
        "R2": r2_score(y_test, pred_opt1),
        "fit_time": time.time() - t0,
    }
    predictions["+ remaining_lease LR"] = pred_opt1

    # --- Option 2: RandomForest 100 vs 300 trees ---
    rf_train_metrics = {}
    for n in (100, 300):
        t0 = time.time()
        rf_pipe = Pipeline(
            [("prep", preproc_3feat),
             ("model", RandomForestRegressor(
                 n_estimators=n, max_depth=15, min_samples_leaf=5,
                 random_state=42, n_jobs=-1))]
        )
        rf_pipe.fit(X_train, y_train)
        fit_time = time.time() - t0
        pred_train = rf_pipe.predict(X_train)
        pred_test = rf_pipe.predict(X_test)
        label = f"RF ({n} trees)"
        results[label] = {
            "MAE": mean_absolute_error(y_test, pred_test),
            "MAPE": mean_absolute_percentage_error(y_test, pred_test),
            "R2": r2_score(y_test, pred_test),
            "fit_time": fit_time,
        }
        rf_train_metrics[label] = {
            "train_MAE": mean_absolute_error(y_train, pred_train),
            "train_R2": r2_score(y_train, pred_train),
        }
        predictions[label] = pred_test
        if n == 100:
            rf_pipe_100 = rf_pipe
            pred_test_100 = pred_test

    # --- Option 3: worst-predicted flat (using RF n=100) ---
    errors = pd.DataFrame({
        "row_index": idx_test,
        "actual_price": y_test.values,
        "predicted_price": pred_test_100,
    })
    errors["abs_error"] = (errors["actual_price"] - errors["predicted_price"]).abs()
    errors_sorted = errors.sort_values("abs_error", ascending=False).reset_index(drop=True)
    worst = errors_sorted.iloc[0]
    worst_row = df.loc[int(worst["row_index"])]

    # --- Part D: Stacking (RF + GB -> Ridge) vs each model alone ---
    t0 = time.time()
    rf_solo = Pipeline(
        [("prep", preproc_3feat),
         ("model", RandomForestRegressor(
             n_estimators=100, max_depth=15, min_samples_leaf=5,
             random_state=42, n_jobs=-1))]
    )
    rf_solo.fit(X_train, y_train)
    pred_rf_solo = rf_solo.predict(X_test)
    results["RF alone (Part D)"] = {
        "MAE": mean_absolute_error(y_test, pred_rf_solo),
        "MAPE": mean_absolute_percentage_error(y_test, pred_rf_solo),
        "R2": r2_score(y_test, pred_rf_solo),
        "fit_time": time.time() - t0,
    }

    t0 = time.time()
    gb_solo = Pipeline(
        [("prep", preproc_3feat),
         ("model", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))]
    )
    gb_solo.fit(X_train, y_train)
    pred_gb_solo = gb_solo.predict(X_test)
    results["GradientBoosting"] = {
        "MAE": mean_absolute_error(y_test, pred_gb_solo),
        "MAPE": mean_absolute_percentage_error(y_test, pred_gb_solo),
        "R2": r2_score(y_test, pred_gb_solo),
        "fit_time": time.time() - t0,
    }
    predictions["GradientBoosting"] = pred_gb_solo

    t0 = time.time()
    base_learners = [
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
    ]
    stack_pipe = Pipeline(
        [("prep", preproc_3feat),
         ("model", StackingRegressor(estimators=base_learners, final_estimator=Ridge(), cv=3, n_jobs=-1))]
    )
    stack_pipe.fit(X_train, y_train)
    pred_stack = stack_pipe.predict(X_test)
    results["Stacked (RF+GB->Ridge)"] = {
        "MAE": mean_absolute_error(y_test, pred_stack),
        "MAPE": mean_absolute_percentage_error(y_test, pred_stack),
        "R2": r2_score(y_test, pred_stack),
        "fit_time": time.time() - t0,
    }
    predictions["Stacked (RF+GB->Ridge)"] = pred_stack

    # Feature importance from RF(100) — map one-hot town columns back to readable names
    ohe = rf_pipe_100.named_steps["prep"].named_transformers_["cat"]
    town_names = [f"town: {c}" for c in ohe.categories_[0]]
    other_cols = [c for c in FEATURES_3 if c != "town"]
    feat_names = town_names + other_cols
    importances = rf_pipe_100.named_steps["model"].feature_importances_
    feat_imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    feat_imp_df = feat_imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "results": results,
        "rf_train_metrics": rf_train_metrics,
        "predictions": predictions,
        "y_test": y_test,
        "X_test": X_test,
        "worst_row": worst_row,
        "worst": worst,
        "errors_sorted": errors_sorted,
        "feat_imp_df": feat_imp_df,
    }


# ----------------------------------------------------------------------------
# Sidebar — data source
# ----------------------------------------------------------------------------
st.sidebar.title("🏠 HDB Resale Modeling")
st.sidebar.markdown(
    "Reproduces the modeling pipeline from `unsupervised_learning.ipynb`:\n\n"
    "**Part C** baseline → **Option 1** (+lease) → **Option 2** (RF tuning) → "
    "**Option 3** (worst prediction) → **Part D** (Stacking)."
)

default_csv = find_default_csv()
uploaded = st.sidebar.file_uploader("Or upload a different CSV", type="csv")

if uploaded is not None:
    df = load_data(uploaded)
    st.sidebar.success(f"Loaded uploaded file: {uploaded.name}")
elif default_csv:
    df = load_data(default_csv)
    st.sidebar.success(f"Loaded: {os.path.basename(default_csv)}")
else:
    st.sidebar.error("No CSV found in `data/` and none uploaded.")
    st.stop()

st.sidebar.metric("Rows", f"{len(df):,}")
st.sidebar.metric("Towns", df["town"].nunique())
st.sidebar.metric("Date range", f"{df['month'].min().strftime('%Y-%m')} – {df['month'].max().strftime('%Y-%m')}")

with st.sidebar.expander("Filter exploratory charts"):
    town_filter = st.multiselect("Town(s)", sorted(df["town"].unique()), default=[])
    flat_type_filter = st.multiselect("Flat type(s)", sorted(df["flat_type"].unique()), default=[])

df_view = df.copy()
if town_filter:
    df_view = df_view[df_view["town"].isin(town_filter)]
if flat_type_filter:
    df_view = df_view[df_view["flat_type"].isin(flat_type_filter)]

# ----------------------------------------------------------------------------
# Train models (cached)
# ----------------------------------------------------------------------------
artifacts = train_all_models(df)
results = artifacts["results"]
predictions = artifacts["predictions"]
y_test = artifacts["y_test"]
X_test = artifacts["X_test"]
worst_row = artifacts["worst_row"]
worst = artifacts["worst"]
errors_sorted = artifacts["errors_sorted"]
feat_imp_df = artifacts["feat_imp_df"]
rf_train_metrics = artifacts["rf_train_metrics"]

# ----------------------------------------------------------------------------
# Header & top-level KPIs
# ----------------------------------------------------------------------------
st.title("🏠 HDB Resale Price Modeling — Results Dashboard")
st.caption(
    "Predicting `resale_price` from `floor_area_sqm`, `town`, and `remaining_lease_years` — "
    "comparing Linear Regression, Random Forest, Gradient Boosting, and a Stacked ensemble."
)

best_model_name = min(results, key=lambda k: results[k]["MAE"])
k1, k2, k3, k4 = st.columns(4)
k1.metric("Best model (by MAE)", best_model_name)
k2.metric("Best Test MAE", f"${results[best_model_name]['MAE']:,.0f}")
k3.metric("Best Test R²", f"{results[best_model_name]['R2']:.4f}")
k4.metric("Best Test MAPE", f"{results[best_model_name]['MAPE']*100:.2f}%")

st.divider()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_models, tab_diagnostics, tab_explore = st.tabs(
    ["📊 Model Comparison", "🌲 Random Forest & Overfitting", "🔍 Prediction Diagnostics", "🗺️ Data Exploration"]
)

# ============================================================================
# TAB 1 — Model comparison
# ============================================================================
with tab_overview:
    st.subheader("Summary table — all models")

    summary_df = pd.DataFrame(results).T
    summary_df.index.name = "Model"
    summary_df = summary_df.reset_index()
    display_df = summary_df.copy()
    display_df["MAE"] = display_df["MAE"].map(lambda v: f"${v:,.0f}")
    display_df["MAPE"] = display_df["MAPE"].map(lambda v: f"{v*100:.2f}%")
    display_df["R2"] = display_df["R2"].map(lambda v: f"{v:.4f}")
    display_df["fit_time"] = display_df["fit_time"].map(lambda v: f"{v:.1f}s")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Test MAE by model** (lower is better)")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_df = summary_df.sort_values("MAE", ascending=True)
        sns.barplot(data=plot_df, x="MAE", y="Model", hue="Model", palette=PALETTE, legend=False, ax=ax)
        ax.set_xlabel("Test MAE ($)")
        ax.set_ylabel("")
        for i, v in enumerate(plot_df["MAE"]):
            ax.text(v, i, f" ${v:,.0f}", va="center", fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Test R² by model** (higher is better)")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_df = summary_df.sort_values("R2", ascending=False)
        sns.barplot(data=plot_df, x="R2", y="Model", hue="Model", palette=PALETTE, legend=False, ax=ax)
        ax.set_xlabel("Test R²")
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
        for i, v in enumerate(plot_df["R2"]):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**MAE vs. fit time** — does more compute buy better accuracy?")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.scatterplot(data=summary_df, x="fit_time", y="MAE", hue="Model", s=160, palette="tab10", ax=ax)
    for _, row in summary_df.iterrows():
        ax.annotate(row["Model"], (row["fit_time"], row["MAE"]), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Fit time (seconds)")
    ax.set_ylabel("Test MAE ($)")
    ax.legend([], [], frameon=False)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"**Takeaway:** Adding `remaining_lease_years` improved the linear model substantially "
        f"(MAE ${results['Baseline (2 feat) LR']['MAE']:,.0f} → ${results['+ remaining_lease LR']['MAE']:,.0f}). "
        f"Random Forest then dominates linear models by capturing non-linear interactions between "
        f"town, area, and lease. Tripling RF's trees from 100 → 300 barely moves accuracy "
        f"(MAE changes by only ${abs(results['RF (100 trees)']['MAE'] - results['RF (300 trees)']['MAE']):,.0f}) "
        f"for ~3x the fit time — diminishing returns. Stacking RF+GB into a Ridge meta-model "
        f"does **not** beat RF alone here, since GB is the clearly weaker base learner and drags "
        f"the blend toward its mistakes."
    )

# ============================================================================
# TAB 2 — Random Forest depth/overfitting analysis
# ============================================================================
with tab_models:
    st.subheader("Random Forest: train vs. test (overfitting check)")

    rf_compare = pd.DataFrame({
        "Model": ["RF (100 trees)", "RF (300 trees)"],
        "Train MAE": [rf_train_metrics["RF (100 trees)"]["train_MAE"],
                       rf_train_metrics["RF (300 trees)"]["train_MAE"]],
        "Test MAE": [results["RF (100 trees)"]["MAE"], results["RF (300 trees)"]["MAE"]],
        "Train R2": [rf_train_metrics["RF (100 trees)"]["train_R2"],
                      rf_train_metrics["RF (300 trees)"]["train_R2"]],
        "Test R2": [results["RF (100 trees)"]["R2"], results["RF (300 trees)"]["R2"]],
    })

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**MAE: train vs. test**")
        melt_mae = rf_compare.melt(id_vars="Model", value_vars=["Train MAE", "Test MAE"],
                                    var_name="Split", value_name="MAE")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=melt_mae, x="Model", y="MAE", hue="Split", palette="mako", ax=ax)
        ax.set_ylabel("MAE ($)")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.markdown("**R²: train vs. test**")
        melt_r2 = rf_compare.melt(id_vars="Model", value_vars=["Train R2", "Test R2"],
                                   var_name="Split", value_name="R2")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=melt_r2, x="Model", y="R2", hue="Split", palette="mako", ax=ax)
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "Train and test scores stay close for both tree counts — the combination of "
        "`max_depth=15` and `min_samples_leaf=5` keeps the forest from memorizing noise, "
        "so this is **not** an overfitting problem. The gap between train/test R² "
        f"(~{abs(rf_train_metrics['RF (100 trees)']['train_R2'] - results['RF (100 trees)']['R2']):.3f}) "
        "is small and stable across 100 vs 300 trees."
    )

    st.subheader("Feature importance — Random Forest (100 trees)")
    top_n = st.slider("Show top N features", 5, min(30, len(feat_imp_df)), 15)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    plot_imp = feat_imp_df.head(top_n)
    sns.barplot(data=plot_imp, x="importance", y="feature", hue="feature", palette="rocket",
                legend=False, ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "`floor_area_sqm` and `remaining_lease_years` typically dominate individual feature "
        "importance since each one-hot `town` column only captures a single category; "
        "summed together, `town` still explains a large share of price variance, which is why "
        "removing it (the original 2-feature baseline) hurt MAE substantially."
    )

# ============================================================================
# TAB 3 — Prediction diagnostics
# ============================================================================
with tab_diagnostics:
    st.subheader("Actual vs. Predicted — by model")
    model_choice = st.selectbox("Choose a model to inspect", list(predictions.keys()),
                                 index=list(predictions.keys()).index("RF (100 trees)"))

    pred_chosen = predictions[model_choice]
    diag_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": pred_chosen,
        "town": X_test["town"].values,
    })
    diag_df["abs_error"] = (diag_df["actual"] - diag_df["predicted"]).abs()
    diag_df["residual"] = diag_df["actual"] - diag_df["predicted"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Actual vs. Predicted — {model_choice}**")
        fig, ax = plt.subplots(figsize=(6, 5.5))
        sample = diag_df.sample(min(5000, len(diag_df)), random_state=1)
        sns.scatterplot(data=sample, x="actual", y="predicted", alpha=0.25, s=15,
                         color="#2c7fb8", ax=ax)
        lims = [diag_df["actual"].min(), diag_df["actual"].max()]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual price ($)")
        ax.set_ylabel("Predicted price ($)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown(f"**Residual distribution — {model_choice}**")
        fig, ax = plt.subplots(figsize=(6, 5.5))
        sns.histplot(diag_df["residual"], bins=60, kde=True, color="#41b6c4", ax=ax)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Residual (actual − predicted, $)")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(f"**Mean absolute error by town — {model_choice}**")
    town_mae = diag_df.groupby("town")["abs_error"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=town_mae.values, y=town_mae.index, hue=town_mae.index, palette="flare",
                legend=False, ax=ax)
    ax.set_xlabel("Mean absolute error ($)")
    ax.set_ylabel("")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("🚨 Single worst-predicted flat (Random Forest, 100 trees)")

    direction = "OVER-predicted" if worst["predicted_price"] > worst["actual_price"] else "UNDER-predicted"
    c1, c2, c3 = st.columns(3)
    c1.metric("Actual price", f"${worst['actual_price']:,.0f}")
    c2.metric("Predicted price", f"${worst['predicted_price']:,.0f}", delta=direction)
    c3.metric("Absolute error", f"${worst['abs_error']:,.0f}")

    st.markdown("**Full details of this flat:**")
    detail_cols = ["town", "flat_type", "floor_area_sqm", "storey_range",
                    "flat_model", "remaining_lease", "resale_price"]
    st.dataframe(worst_row[detail_cols].to_frame().T, use_container_width=True, hide_index=True)

    st.warning(
        "**Why this happened:** the model only knows `floor_area_sqm`, `town`, and "
        "`remaining_lease_years`. It learned that this town generally commands high prices, "
        "but has no visibility into this unit's `storey_range` or `flat_model` — both of which "
        "swing price by hundreds of thousands of dollars within the same town. A low-floor, "
        "basic-model flat in an expensive town gets priced as if it were a premium high-floor "
        "unit there, because `town` alone can't distinguish them. This is the same "
        "underfitting-from-missing-features pattern as the baseline model, just relocated to a "
        "different blind spot."
    )

    st.markdown("**Top 15 worst-predicted flats**")
    worst15 = errors_sorted.head(15).copy()
    worst15_detail = df.loc[worst15["row_index"].astype(int), detail_cols].reset_index(drop=True)
    worst15_display = pd.concat([
        worst15_detail,
        worst15[["predicted_price", "abs_error"]].reset_index(drop=True)
    ], axis=1)
    worst15_display["predicted_price"] = worst15_display["predicted_price"].map(lambda v: f"${v:,.0f}")
    worst15_display["abs_error"] = worst15_display["abs_error"].map(lambda v: f"${v:,.0f}")
    worst15_display["resale_price"] = worst15_display["resale_price"].map(lambda v: f"${v:,.0f}")
    st.dataframe(worst15_display, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 4 — Data exploration (raw dataset, filterable)
# ============================================================================
with tab_explore:
    st.subheader("Exploratory charts")
    st.caption("Filtered by the sidebar town/flat-type selectors, if set.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Resale price distribution**")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.histplot(df_view["resale_price"], bins=60, kde=True, color="#225ea8", ax=ax)
        ax.set_xlabel("Resale price ($)")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Floor area vs. resale price**")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sample = df_view.sample(min(5000, len(df_view)), random_state=1)
        sns.scatterplot(data=sample, x="floor_area_sqm", y="resale_price", alpha=0.25, s=15,
                         color="#41b6c4", ax=ax)
        sns.regplot(data=sample, x="floor_area_sqm", y="resale_price", scatter=False,
                    color="red", line_kws={"linewidth": 1.5}, ax=ax)
        ax.set_xlabel("Floor area (sqm)")
        ax.set_ylabel("Resale price ($)")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Median resale price by town**")
    town_median = df_view.groupby("town")["resale_price"].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=town_median.values, y=town_median.index, hue=town_median.index, palette=PALETTE,
                legend=False, ax=ax)
    ax.set_xlabel("Median resale price ($)")
    ax.set_ylabel("")
    st.pyplot(fig)
    plt.close(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Price by flat type**")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        order = df_view.groupby("flat_type")["resale_price"].median().sort_values().index
        sns.boxplot(data=df_view, x="resale_price", y="flat_type", order=order,
                    hue="flat_type", palette="crest", legend=False, ax=ax)
        ax.set_xlabel("Resale price ($)")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Remaining lease vs. resale price**")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sample = df_view.sample(min(5000, len(df_view)), random_state=1)
        sns.scatterplot(data=sample, x="remaining_lease_years", y="resale_price", alpha=0.25,
                         s=15, color="#a1dab4", ax=ax)
        sns.regplot(data=sample, x="remaining_lease_years", y="resale_price", scatter=False,
                    color="red", line_kws={"linewidth": 1.5}, ax=ax)
        ax.set_xlabel("Remaining lease (years)")
        ax.set_ylabel("Resale price ($)")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Median price trend over time**")
    monthly = df_view.dropna(subset=["month"]).groupby(
        df_view["month"].dt.to_period("M")
    )["resale_price"].median()
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.lineplot(x=monthly.index, y=monthly.values, color="#253494", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Median resale price ($)")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Correlation heatmap — numeric features**")
    numeric_cols = ["floor_area_sqm", "remaining_lease_years", "lease_commence_date", "resale_price"]
    corr = df_view[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("View raw filtered data"):
        st.dataframe(df_view.head(500), use_container_width=True)
        st.caption(f"Showing first 500 of {len(df_view):,} filtered rows.")