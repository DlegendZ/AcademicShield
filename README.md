# Academic Shield — AI Burnout Meter & Student Performance Predictor

An end-to-end classical-ML web app that predicts a student's **burnout level**
and **future GPA** from daily lifestyle and mental-health inputs, with a live
**user-feedback loop**.

Built for COMP6577001 Machine Learning (final project). Two deployment fronts:

- **Streamlit app** — the original interactive app (server-side inference).
- **Static + FastAPI app** (`web/`) — a latency-optimized front end where the
  models run **100% in the browser**, so a prediction takes **< 1 ms** with no
  server round-trip (meets the < 100 ms requirement).

---

## Models

| | Model A — Burnout | Model B — GPA |
|---|---|---|
| Task | 3-class classification (Healthy / Mildly Burnout / Burnout) | Regression (GPA 0–4) |
| Algorithm | XGBClassifier | XGBRegressor |
| Dataset | `Datasets/academic_stress_level.csv` (1 M rows) | `Datasets/student_lifestyle_dataset.csv` |
| Deployed metrics | Acc 0.8465 · Macro-F1 0.5279 · CV Acc 0.8464 ± 0.0006 | R² 0.5075 · MAE 0.1662 · CV R² 0.5097 ± 0.033 |

**Research vs deployed.** The full Optuna-tuned research models live in
`notebooks/experiment/`. They are accurate but huge (Model A ≈ 210 MB / 11.4 M
nodes) — impossible to run client-side. The **deployed** models are shallow,
compact retrains (Model A 1.6 MB, Model B 0.5 MB) trained in
`notebooks/modelA.ipynb` / `notebooks/modelB.ipynb`. These compact models are the
**single source of truth** for the app.

---

## How the < 100 ms front end works

```
Browser: user input → feature engineering (JS) → XGBoost model (JS) → result
                       ↑ everything client-side, ~0.3 ms, no network
FastAPI: serves static files + /api/feedback (Supabase proxy) only
```

The trained models are transpiled from Python pickles to pure JavaScript with
[`m2cgen`](https://github.com/BayesWitnesses/m2cgen). A gradient-boosted tree is
literally nested `if/else` on feature thresholds, so the JS is the *same model*
in code form — verified identical to the Python model within 1e-6 by
`scripts/validate_parity.mjs`.

---

## Repository layout

```
Page_1.py, pages/            Streamlit app (server-side inference)
src/                         shared config, features, models, UI
models/
  modelA.pkl, modelB.pkl     deployed compact models (from notebooks/)
notebooks/
  modelA.ipynb, modelB.ipynb compact model training (+ cross-validation)
  experiment/                full research models (Optuna, EDA)
scripts/
  export_models.py           models/*.pkl  -> web/static/models/*.js  (m2cgen)
  validate_parity.mjs        asserts JS == Python within 1e-4
  requirements-build.txt     build-only deps (m2cgen, imbalanced-learn)
web/                         static + FastAPI client-side app
  api/index.py               FastAPI: serves static + /api/feedback proxy
  static/                    index.html, app.js, styles.css, models/*.js
  vercel.json                CDN-serves static, routes /api/* to FastAPI
  requirements.txt           fastapi
Datasets/  (sibling folder, not committed)   raw CSVs
```

---

## Running locally

**Streamlit app**
```bash
pip install -r requirements.txt
streamlit run Page_1.py
```

**Static + FastAPI app** (client-side inference)
```bash
pip install -r web/requirements.txt uvicorn
uvicorn api.index:app --app-dir web --port 8123
# open http://localhost:8123
# feedback POST returns 500 "env vars not configured" locally — expected
```

---

## Rebuilding the deployed models

The compact models are trained in the notebooks (single source of truth).
After (re)running `notebooks/modelA.ipynb` / `notebooks/modelB.ipynb` (which save
`models/modelA.pkl` / `models/modelB.pkl`), regenerate and verify the JS:

```bash
pip install -r scripts/requirements-build.txt   # m2cgen, imbalanced-learn
python scripts/export_models.py                  # models/*.pkl -> web/static/models/*.js
node   scripts/validate_parity.mjs               # JS == Python within 1e-4
```

> Keep Model B's `n_estimators` ≤ ~600 — m2cgen builds one deeply-nested AST per
> ensemble, and larger counts overflow Python's recursion limit during transpile.

---

## Deploying the front end (Vercel)

Set the project **root directory** to `web/`. `vercel.json` serves `static/**`
from the CDN edge (fast) and routes only `/api/*` to the FastAPI Python function.

For the feedback form, add env vars in Vercel → Settings → Environment Variables:

- `SUPABASE_URL` — your Supabase project URL
- `SUPABASE_SERVICE_KEY` — Supabase **service_role** key (server-side only; never
  shipped to the browser)

Supabase needs a `feedback` table whose columns match the payload in
`web/api/index.py`. Prediction works with or without Supabase — only the feedback
form depends on it.

> Vercel's Python runtime is serverless, so the first `/api/feedback` call after
> idle may cold-start. This never affects prediction or page load — prediction is
> client-side and static assets come from the CDN.
"# AcademicShield" 
