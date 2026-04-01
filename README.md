# 🚦 Traffic Volume Forecasting & Analytics

A production-ready, end-to-end traffic congestion forecasting system built with
Python, scikit-learn, Plotly, and Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ Features

| Feature | Detail |
|---|---|
| **ML Models** | Gradient Boosting & Random Forest (switchable via sidebar) |
| **18 engineered features** | Lags, rolling stats, cyclic time encodings, weather, events |
| **Interactive dashboard** | 5 tabs: Trends · Actual vs Predicted · Junctions · Weather · Feature Importance |
| **Live prediction** | Adjust sliders → instant forecast |
| **Crash-resistant** | Fully self-contained; generates synthetic data on first run |
| **Dark UI** | Custom Plotly + Streamlit theme |

---

## 📂 Project Structure

```
traffic-portfolio-project/
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py        ← Main Streamlit dashboard
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── generate.py         ← Synthetic data generation & preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py         ← Lag, rolling, cyclic feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py            ← Pipeline build, train, evaluate, persist
│   └── pipeline/
│       ├── __init__.py
│       └── orchestrate.py      ← End-to-end orchestrator (cached)
│
├── outputs/
│   └── models/                 ← Saved .joblib model (auto-generated)
│
├── .streamlit/
│   └── config.toml             ← Streamlit server & theme config
│
├── app.py                      ← Root entry point (imports app/streamlit_app.py)
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Tech Stack

- **Python 3.11**
- **Streamlit ≥ 1.32** — dashboard UI
- **scikit-learn ≥ 1.4** — Gradient Boosting / Random Forest
- **Plotly ≥ 5.18** — interactive charts
- **pandas ≥ 2.0 · numpy ≥ 1.26** — data wrangling

---

## 🤖 Model Performance (on synthetic data)

| Metric | Value |
|---|---|
| MAE | **~0.98** |
| RMSE | **~1.33** |
| R² | **0.986** |
| Baseline MAE | **~5.57** |
| Improvement | **~82%** |

---

## 🚀 Deploy to Streamlit Cloud (step-by-step)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — Traffic Forecasting App"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Set:
   - **Repository**: `<YOUR_USERNAME>/<YOUR_REPO>`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
4. Click **Deploy** — done! ✅

> **Note**: No secrets or environment variables are needed. The app is fully self-contained.

---

## 💻 Run Locally

```bash
# 1. Clone
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app/streamlit_app.py
```

---

## 🔮 Future Improvements

- Real-time sensor data integration (MQTT / REST API)
- Junction-specific models for improved localization
- LSTM / Prophet for long-horizon forecasting
- Automated retraining pipeline (Airflow / GitHub Actions)
- Export predictions to SQLite / PostgreSQL

---

## 📬 Contact

For questions or collaboration, feel free to connect via GitHub.
