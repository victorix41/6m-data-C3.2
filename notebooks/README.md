# HDB Resale Price Modeling — Streamlit Dashboard

Reproduces and visualizes the modeling pipeline from `unsupervised_learning.ipynb`:
**Part C baseline → Option 1 (+lease) → Option 2 (RF tuning) → Option 3 (worst
prediction) → Part D (Stacking)**.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Put the dataset CSV in a `data/` folder next to `app.py`, named:

```
data/Resale_flat_prices_based_on_registration_date_from_Jan-2017_onwards.csv
```

(Already included if you downloaded this from the chat.) You can also upload a
different CSV with the same columns directly in the app's sidebar.

## Run

```bash
streamlit run app.py
```

## What's inside

- **📊 Model Comparison** — summary table + bar charts of MAE/R² across all 6
  models (Baseline LR, +lease LR, RF-100, RF-300, GradientBoosting, Stacked).
- **🌲 Random Forest & Overfitting** — train vs. test MAE/R² to check for
  overfitting, plus a feature-importance chart.
- **🔍 Prediction Diagnostics** — actual-vs-predicted scatter, residual
  histogram, per-town error breakdown, and the single worst-predicted flat
  (matching the notebook's Option 3 finding) plus a top-15 worst table.
- **🗺️ Data Exploration** — price distributions, price vs. floor area / lease,
  price by town/flat type, price trend over time, and a correlation heatmap.
  Filterable by town and flat type from the sidebar.

## Notes

- Model training is wrapped in `@st.cache_resource`, so the ~2–3 minute fit
  (Random Forest x2, Gradient Boosting, Stacking) only runs once per session
  — afterward, filtering and tab-switching is instant.
- All numbers were verified against the notebook's saved output (e.g. RF-100
  Test MAE $64,904 / R² 0.7919; worst-predicted flat is the Central Area
  4-room unit, actual $483,000 vs. predicted ~$1,397,817).
