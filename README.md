# HDB Resale Price Predictor — C3.2 (Better & Trustworthy Models)

A teaching project that upgrades a simple 3-feature HDB price model into a more
accurate, honestly-evaluated one — and deploys it as a Streamlit web app.

Built as a 2-hour coaching add-on for **Module 3 · Machine Learning & GenAI**,
continuing from the C3.1 predictor.

## 🚀 Live demo / run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app **trains itself on first run** (downloads a live HDB resale dataset,
compares two models, keeps the better one, and caches it). No need to train
manually — but you can if you want to see the scores:

```bash
python model.py
```

## 🧠 What it does
- Predicts an HDB flat's resale price from floor area, lease year, floor level,
  **flat type**, and **town**.
- Reports its own accuracy on unseen flats using **three scorecards**: MAE
  (average dollar error), MAPE (average % error), and R² (share of price
  variation explained).
- Compares Linear Regression vs Random Forest and deploys the better one.

## ➕ Adding your own features (do it in ONE place)
The features the model uses are defined once in **`features.py`**. Both training
(`model.py`) and the web form (`app.py`) read from it, so they can never disagree.

To add a feature:
1. **If the column already exists** in the dataset (e.g. `flat_model`), add one
   entry to the `FEATURES` list in `features.py`.
2. **If it must be computed** from a raw column (e.g. remaining lease in years),
   add one line to `clean_data()` in `features.py`, then add its `FEATURES` entry.
   (A worked example is included, commented out.)
3. Retrain and redeploy: delete `house_model.pkl`, run `python model.py`, then
   `streamlit run app.py`. A matching slider/dropdown appears automatically.

That's the whole change — no edits to `model.py` or `app.py` needed.

## 📂 Repository contents
```
features.py                   ⭐ Single source of truth for the model's features
app.py                        Streamlit web app (auto-trains on first run)
model.py                      Training + load_or_train logic (run standalone too)
requirements.txt              Python dependencies
runtime.txt                   Python version pin for Streamlit Cloud
facilitator_run_sheet.md      2-hour session plan (Kolb's cycle)
explainer.html                Interactive concept aid (MAE / features / overfitting)
notebooks/L06_practice.ipynb  Self-runnable hands-on notebook (incl. model stacking)
notebooks/data/               HDB resale dataset the notebook downloads from this repo
```

## 📓 What's in the notebook (`notebooks/C32_practice.ipynb`)
A guided, self-runnable lab that takes the 3-feature baseline and makes it better
*and* more trustworthy. Recently added:
- **Plain-English metrics primer** — a no-stats-needed explanation of MAE, MAPE,
  and R² before Part A, so learners know what each score means.
- **Three scorecards everywhere** — Parts A–D now each report MAE, MAPE, and R²
  (not just MAE), including the train-vs-test gap as an overfitting signal.
- **"Reading the output" notes** — every code cell is followed by a short
  explanation interpreting what its numbers actually mean.
- **Self-hosted data** — the setup cell downloads the dataset from this repo's
  `notebooks/data/` folder rather than a third-party source.

Roughly: 3 features → ~S\$102k MAE / ~20% MAPE / R² ~0.50; add flat type + town →
~S\$83k / ~17% / ~0.70; Random Forest → ~S\$73k / ~14% / ~0.77. Stacking is shown
to barely beat a single good model — a deliberate "don't over-engineer" lesson.

## ☁️ Deploy to Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo (see steps below).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app**, pick your repo, set **Main file path** to `app.py`, deploy.
4. First load takes ~1 minute while it trains; after that it's instant.

> The trained model file (`house_model.pkl`) is intentionally **not** committed
> (see `.gitignore`) — the app regenerates it, so deployments always train fresh
> against the latest data.

## 📚 Learning concepts covered
MAE · MAPE · R² · train/test split · feature engineering · model selection · overfitting · model stacking.

## License
MIT — see `LICENSE`.
