# Academic Shield

An end-to-end classical-ML web app that predicts a student's **burnout level** and **future GPA** from daily lifestyle and mental-health inputs, with a live **user-feedback loop**.

Prediction runs **100% client-side in the browser** (< 1 ms, no server round-trip). FastAPI only serves static assets.

---

## Table of Contents

- [Models](#models)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Datasets](#datasets)
- [EDA & Downloaded Datasets](#eda--downloaded-datasets)
- [Running Locally](#running-locally)
- [Deployed App](#deployed-app)
- [Performance Testing](#performance-testing)
- [Rebuilding Models](#rebuilding-models)
- [Deployment (Vercel)](#deployment-vercel)

---

## Models

| | Model A — Burnout Classifier | Model B — GPA Predictor |
|---|---|---|
| **Task** | 3-class classification (Healthy / Mildly Burnout / Burnout) | Regression (GPA 0–4) |
| **Algorithm** | XGBClassifier | XGBRegressor |
| **Dataset** | `datasets/academic_stress_level.csv` (1 M rows) | `datasets/student_lifestyle_dataset.csv` (2 k rows) |
| **Accuracy / R²** | Acc 0.8465 · Macro-F1 0.5279 · CV Acc 0.8464 ± 0.0006 | R² 0.5357 · MAE 0.1637 · CV R² 0.5279 ± 0.0249 |
| **Deployed size** | 1.6 MB | 0.5 MB |

**Research vs. deployed models.** Full Optuna-tuned research models live in `notebooks/experiment/` (Model A ≈ 210 MB / 11.4 M nodes — too large for client-side). Compact retrains in `notebooks/modelA.ipynb` / `notebooks/modelB.ipynb` are the **single source of truth** for the app. Performance is on par with the research versions.

---

## Architecture

```
Browser: user input → feature engineering (JS) → XGBoost model (JS) → result
                       ↑ all client-side, ~0.3 ms, zero network latency
FastAPI: serves static files only
```

Models are transpiled from Python pickles to pure JavaScript via [`m2cgen`](https://github.com/BayesWitnesses/m2cgen). The JS is verified identical to the Python model within 1e-4 by `scripts/validate_parity.mjs`.

---

## Repository Layout

```
src/                          shared config + feature engineering (config.py, features.py)
models/
  modelA.pkl, modelB.pkl      deployed compact models (from notebooks/)
notebooks/
  modelA.ipynb, modelB.ipynb  compact model training + cross-validation
  experiment/                 full research models (Optuna, EDA)
scripts/
  export_models.py            models/*.pkl → web/static/models/*.js  (m2cgen)
  validate_parity.mjs         assert JS == Python within 1e-4
  requirements-build.txt      build-only deps (m2cgen, imbalanced-learn)
web/
  api/index.py                FastAPI: serves static + /api/feedback proxy
  static/                     index.html, app.js, styles.css
  static/models/
    model_a.js                transpiled Model A (client-side inference)
    model_b.js                transpiled Model B (client-side inference)
  vercel.json                 CDN-serves static, routes /api/* to FastAPI
  requirements.txt            fastapi
datasets/                     raw CSVs (gitignored, not committed)
performance_test.py           comprehensive HTTP + inference performance test
MLflow.ipynb                  experiment tracking (MLflow runs)
artifacts.zip                 exported MLflow artifacts
```

---

## Datasets

### Dataset A — Student Mental Health & Burnout

Used to train **Model A** (burnout classifier).

- **Source:** [Kaggle — Student Mental Health and Burnout](https://www.kaggle.com/datasets/sharmajicoder/student-mental-health-and-burnout?resource=download)
- **File:** `datasets/academic_stress_level.csv`
- **Size:** ~1 million rows
- **Key features:** study hours, sleep hours, exam pressure, stress level, financial stress, social support, anxiety score, depression score, family expectation, physical activity

### Dataset B — Student Stress & Performance

Used to train **Model B** (GPA predictor).

- **Source:** [Kaggle — Student Stress Performance Insights](https://www.kaggle.com/code/sulaniishara/student-stress-performance-insights)
- **File:** `datasets/student_lifestyle_dataset.csv`
- **Size:** ~2,000 rows
- **Key features:** study hours, extracurricular hours, sleep hours, social hours, physical hours, stress level

> Datasets are gitignored. Download them from Kaggle and place in `datasets/` before running notebooks.

---

## EDA & Downloaded Datasets

Exploratory Data Analysis notebooks and pre-downloaded datasets are available on Google Drive:

**[Google Drive — EDA & Datasets](https://drive.google.com/drive/folders/1V7syPoVqk04eG6E0HG8f-sv5QUW1H1aR?usp=sharing)**

Contents include:
- Raw downloaded CSVs for both datasets
- EDA notebooks with visualizations and insights
- Exported MLflow artifacts

---

## Running Locally

### Prerequisites

- Python 3.10+
- `pip install -r web/requirements.txt uvicorn`

### Start the app

```bash
pip install -r web/requirements.txt uvicorn
uvicorn api.index:app --app-dir web --port 8123
```

Open **http://localhost:8123** in your browser.

> The feedback POST returns `500 "env vars not configured"` locally — this is expected. Predictions work fully without Supabase.

---

## Deployed App

The app is deployed on Vercel. Access it at:

**[https://academic-shield.vercel.app](https://academic-shield.vercel.app)**

> If the link above is outdated, check the latest deployment URL in the Vercel dashboard for this project.

Static assets (HTML, JS, models) are served from Vercel's CDN edge. Only `/api/feedback` hits the FastAPI serverless function (may cold-start after idle — does not affect prediction).

---

## Performance Testing

### Option 1 — `performance_test.py` (Python)

A comprehensive test covering connectivity, latency distribution, concurrent load, sustained throughput, and local model inference.

**Install dependency:**

```bash
pip install requests
```

**Test local app:**

```bash
python performance_test.py --url http://localhost:8123
```

**Test deployed app:**

```bash
python performance_test.py --url https://academic-shield.vercel.app
```

**Heavy load test:**

```bash
python performance_test.py --url http://localhost:8123 --n 200 --users 20
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--url` | `http://localhost:8123` | Target URL |
| `--n` | `50` | Sequential requests |
| `--warmup` | `3` | Warmup requests before measuring |
| `--users` | `10` | Concurrent users |
| `--duration` | `15` | Sustained load duration (seconds) |
| `--infer-n` | `100` | Local inference runs |
| `--skip-http` | — | Skip HTTP tests, run inference only |
| `--skip-infer` | — | Skip local inference test |

**Performance targets:**

| Metric | Target |
|---|---|
| TTFB mean | ≤ 500 ms |
| Response mean | ≤ 2000 ms |
| Response p95 | ≤ 3000 ms |
| Error rate | ≤ 5% |
| Inference mean (A+B) | ≤ 100 ms |
| Concurrent mean | ≤ 3000 ms |

---

### Option 2 — Postman (Manual HTTP Testing)

Use Postman to test the static model JS files served by the local app.

**Setup:**

1. Set a Postman environment variable: `BASE_URL = http://localhost:8123`
2. Make sure the local app is running (see [Running Locally](#running-locally))

**Endpoints to test:**

| Request | Method | URL | Expected |
|---|---|---|---|
| Model A JS | GET | `{{BASE_URL}}/models/model_a.js` | 200 OK, JavaScript file |
| Model B JS | GET | `{{BASE_URL}}/models/model_b.js` | 200 OK, JavaScript file |

**What to check:**

- Status code: `200 OK`
- `Content-Type`: `application/javascript`
- Response body: starts with a JS function (the transpiled XGBoost model)
- Response time: should be very fast (files are served from disk / CDN cache)
- On deployed app, response headers should include `Cache-Control: public, max-age=31536000, immutable`

---

## Rebuilding Models

After rerunning `notebooks/modelA.ipynb` / `notebooks/modelB.ipynb` (which save `models/modelA.pkl` / `models/modelB.pkl`), regenerate and verify the JS:

```bash
pip install -r scripts/requirements-build.txt   # m2cgen, imbalanced-learn
python scripts/export_models.py                  # models/*.pkl → web/static/models/*.js
node   scripts/validate_parity.mjs               # assert JS == Python within 1e-4
```

> Keep Model B's `n_estimators` ≤ ~600. m2cgen builds deeply-nested ASTs and larger counts overflow Python's recursion limit during transpile.

---

## Deployment (Vercel)

Set the project **root directory** to `web/`. `vercel.json` handles routing:

- `static/**` → served from CDN edge
- `/api/*` → routed to FastAPI Python function

**Required environment variables** (Vercel → Settings → Environment Variables):

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Supabase `service_role` key (server-side only, never sent to browser) |

Supabase needs a `feedback` table whose columns match the payload in `web/api/index.py`. Predictions work without Supabase; only the feedback form depends on it.
