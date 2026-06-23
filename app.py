# =============================================================
# L06 — HDB Resale Price Predictor (deployable Streamlit app)
#
# The input form is built AUTOMATICALLY from features.py — add a feature
# there and a matching slider/dropdown appears here with no other changes.
#
# Run locally:
#       pip install -r requirements.txt
#       streamlit run app.py
# =============================================================

import pandas as pd
import streamlit as st

import features as ft
from model import load_or_train

st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏡", layout="centered")


@st.cache_resource(show_spinner="Training the model (first run only)…")
def get_model():
    """Load the saved model, or train one on first run. Cached across reruns."""
    return load_or_train()


bundle = get_model()
model = bundle["model"]
model_columns = bundle["columns"]

# ---- Header ----
st.title("🏡 Singapore HDB Resale Price Predictor")
st.caption(
    f"Powered by a **{bundle['model_name']}** model trained on "
    f"{bundle['n_rows']:,} real resale transactions (2017 onwards)."
)

# ---- Model report card ----
with st.expander("📊 How accurate is this model?", expanded=False):
    st.metric("Average error (MAE)", f"S${bundle['mae']:,.0f}")
    st.write(
        "On average, the prediction is off by this much. We tested it on flats "
        "the model had never seen, so this is an honest estimate."
    )
    st.write("**Models compared during training:**")
    st.table(
        pd.DataFrame(
            [{"Model": n, "Average error (MAE)": f"S${m:,.0f}"}
             for n, m in bundle["all_scores"].items()]
        )
    )

# ---- Inputs (built automatically from features.py) ----
# Categoricals are rendered FIRST (so we know the chosen flat_type), then the
# numeric sliders adapt their range to that flat_type. This stops the user
# picking impossible combos like a "1 ROOM" flat at 90 sqm — the slider simply
# won't go there, because no such flat exists in the training data.
st.sidebar.header("Flat details")
user_input = {}

# 1) Categorical dropdowns first
for f in ft.FEATURES:
    if f["type"] != "categorical":
        continue
    choices = bundle["categories"].get(f["col"], [])
    user_input[f["col"]] = st.sidebar.selectbox(f["label"], choices)

# Ranges observed in the data for the chosen flat_type (fallback: global ranges)
chosen_flat_type = user_input.get("flat_type")
ranges = bundle.get("numeric_ranges_by_flat_type", {}).get(
    chosen_flat_type, bundle.get("numeric_ranges", {})
)

# 2) Numeric sliders, clamped to what's plausible for the chosen flat_type
for f in ft.FEATURES:
    if f["type"] != "numeric":
        continue
    col, label = f["col"], f["label"]
    lo, hi = f["min"], f["max"]
    if col in ranges:
        # Stay within the feature's own bounds, but tighten to observed range
        lo = max(f["min"], int(ranges[col][0]))
        hi = min(f["max"], int(round(ranges[col][1])))
        if lo >= hi:                      # guard: a flat_type with one value
            hi = lo + f["step"]
    default = min(max(f["default"], lo), hi)   # keep default inside [lo, hi]
    user_input[col] = st.sidebar.slider(label, lo, hi, default, f["step"])

if chosen_flat_type and bundle.get("numeric_ranges_by_flat_type"):
    st.sidebar.caption(
        f"ℹ️ Slider ranges are limited to what actually exists for "
        f"**{chosen_flat_type}** flats in the data, so you can't build a "
        f"flat that was never sold."
    )

# ---- Show the chosen flat ----
st.write("### Your flat")
cols = st.columns(min(3, len(user_input)))
for i, (col, val) in enumerate(user_input.items()):
    label = next(f["label"] for f in ft.FEATURES if f["col"] == col)
    cols[i % len(cols)].write(f"**{label}:** {val}")

# ---- Prediction ----
if st.button("Predict resale price", type="primary"):
    row = pd.DataFrame([user_input])
    # Encode categories and line the columns up exactly with the trained model
    row = pd.get_dummies(row, columns=ft.categorical_cols())
    row = row.reindex(columns=model_columns, fill_value=0)

    price = model.predict(row)[0]
    mae = bundle["mae"]

    st.success(f"🇸🇬 Estimated resale price: **S${price:,.0f}**")
    st.caption(
        f"Likely range (±1 average error): "
        f"S${max(0, price - mae):,.0f} — S${price + mae:,.0f}"
    )
    st.info(
        "This is an estimate from a teaching model, not a valuation. "
        "Real prices also depend on renovation, exact location, and market timing."
    )

st.divider()
st.caption("Module 3 · Machine Learning & GenAI · L06 coaching project")
