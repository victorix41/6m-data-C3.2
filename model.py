# =============================================================
# L06 — Better, Trustworthy HDB Price Model
# Training logic (shared by the Streamlit app and command line).
#
# The feature list lives in features.py — change it there once and BOTH
# this file and app.py update automatically.
#
# Run standalone to (re)train and save the model file:
#       python model.py
# =============================================================

import os
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import features as ft

DATA_URL = (
    "https://raw.githubusercontent.com/kohjiaxuan/"
    "Predicting-HDB-Price-with-Machine-Learning/master/"
    "resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_model.pkl")


def load_data():
    """Download the live HDB resale dataset and apply the cleaning steps."""
    data = pd.read_csv(DATA_URL)
    data = ft.clean_data(data)
    return data


def build_X(data):
    """Turn the chosen features into a numeric table the model can use."""
    return pd.get_dummies(
        data[ft.all_cols()], columns=ft.categorical_cols()
    )


def train_model():
    """Train, compare two models, and return the best one as a bundle dict."""
    data = load_data()

    X = build_X(data)
    y = data[ft.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        scores[name] = (model, mae)

    best_name = min(scores, key=lambda n: scores[n][1])
    best_model, best_mae = scores[best_name]

    # Save the choices available for each dropdown (categorical) feature
    categories = {col: sorted(data[col].dropna().unique().tolist())
                  for col in ft.categorical_cols()}

    # Plausible numeric ranges CONDITIONED on flat_type, so the web form can
    # stop users picking impossible combos (e.g. a "1 ROOM" flat at 90 sqm).
    # We use the 1st–99th percentile actually seen in the data for each
    # flat_type to ignore a handful of outliers. Falls back to global ranges.
    numeric_ranges_by_flat_type = {}
    if "flat_type" in ft.categorical_cols():
        for ftype, grp in data.groupby("flat_type"):
            numeric_ranges_by_flat_type[ftype] = {
                col: [float(grp[col].quantile(0.01)),
                      float(grp[col].quantile(0.99))]
                for col in ft.numeric_cols()
            }

    numeric_ranges = {
        col: [float(data[col].quantile(0.01)), float(data[col].quantile(0.99))]
        for col in ft.numeric_cols()
    }

    return {
        "model": best_model,
        "columns": list(X.columns),
        "model_name": best_name,
        "mae": best_mae,
        "all_scores": {n: s[1] for n, s in scores.items()},
        "categories": categories,
        "numeric_ranges": numeric_ranges,
        "numeric_ranges_by_flat_type": numeric_ranges_by_flat_type,
        "n_rows": len(data),
    }


def save_model(bundle, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_or_train(path=MODEL_PATH):
    """Load a saved model, or train and save one if none exists yet.

    This is what makes the app 'deployable when trained': the first time
    the app runs on a fresh server it trains the model, then reuses it.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    bundle = train_model()
    save_model(bundle, path)
    return bundle


if __name__ == "__main__":
    print("Downloading data and training models...")
    print("Features in use:", ", ".join(ft.all_cols()))
    bundle = train_model()
    for name, mae in bundle["all_scores"].items():
        print(f"  {name:18s} -> on average off by S${mae:,.0f}")
    print(f"\nWinner: {bundle['model_name']} (lowest dollar error)")
    save_model(bundle)
    print(f"Saved model to {MODEL_PATH}")
