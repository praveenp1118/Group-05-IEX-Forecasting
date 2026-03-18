# ⚡ Group 05 — IEX RTM Electricity Price Forecasting
### ISB AMPBA | Foundation Project | CRISP-ML(Q) Framework

> **Real-time 15-minute RTM clearing price forecasting for the Indian Energy Exchange (IEX)**  
> Predicts MCP (Market Clearing Price) in Rs/MWh to support power trading decisions

---

## 🌐 Live Deployment (AWS EC2)

| | |
|---|---|
| **Public URL** | http://15.135.168.75:5000 |
| **Home Page** | http://15.135.168.75:5000/ |
| **Model Summary** | http://15.135.168.75:5000/model-summary |
| **Health Check** | http://15.135.168.75:5000/health |
| **Interactive Predictor** | http://15.135.168.75:5000/predict |
| **24h Forecast** | http://15.135.168.75:5000/forecast/24h |
| **EDA Dashboard** | http://15.135.168.75:5000/eda |
| **PESTLE Analysis** | http://15.135.168.75:5000/pestle |
| **Monitoring** | http://15.135.168.75:5000/monitoring |
| **Server** | AWS EC2 t3.small — ap-southeast-2 (Sydney) |
| **Container** | Docker (Python 3.11 + Chromium/Selenium) |

---

## 📊 Results Summary

| Metric | Value |
|---|---|
| **Best Model** | XGBoost v10 |
| **Test MAPE** | 18.31% |
| **ARIMA Baseline MAPE** | 53.46% |
| **Improvement over Baseline** | 65.7% |
| **Simulated P&L** | ₹7.7 Crore (100 MW, 3 months) |
| **Training Data** | 52,705 records (Aug 2024 – Feb 2026) |
| **Forecast Horizon** | 96 blocks × 15 min = 24 hours |
| **Distribution Shift** | 4.2% (train vs test, same regime) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                             │
│  IEX Website      OpenWeatherMap API    Yahoo Finance       │
│  (RTM 15-min)     (8 cities, live)      (Crude/Gas/FX)      │
└────────┬──────────────┬──────────────────┬──────────────────┘
         │              │                  │
         ▼              ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA PIPELINE                               │
│  scraper_iex.py          scraper_weather.py                 │
│  fetch_live_commodities.py  ← NEW (live prices daily)       │
│        │               │                   │                │
│        └───────────────┴───────────────────┘                │
│                        │                                    │
│              merge_historical.py                            │
│         (Option B: Scraped + Price.xlsx gap fill)           │
│         (Leakage-free rolling features on shifted MCP)      │
│                        │                                    │
│              sync_live_files.py                             │
│         (Copies latest rows to *_live.csv every 30 min)     │
│                        │                                    │
│              backfill_actuals.py                            │
│         (Fills actual MCP into prediction log               │
│          by matching timestamp to IEX time block)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              MASTER TRAINING DATA                           │
│  52,705 records | 18 months | 36 features                   │
│  Period: Aug 2024 – Feb 2026 (regime-aligned)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  run_pipeline.py                            │
│                                                             │
│  Step 1: Schema Validation (validator.py)                   │
│  Step 2: Feature prep — last 18 months, leakage-free        │
│  Step 3: Train ARIMA + SVM + XGBoost (3-fold CV)            │
│  Step 4: Business Evaluation (Rs P&L simulation)            │
│  Step 5: PESTLE Scenario Analysis (7 scenarios)             │
│  Step 6: Model Versioning + Rollback                        │
│  Step 7: EDA Report Generation                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 FLASK REST API  (app/app.py)                 │
│                                                             │
│  /                    → HTML home page + clickable links    │
│  /model-summary       → Model card — all models, metrics    │
│  /health              → System health + data freshness      │
│  /predict             → Interactive HTML prediction form    │
│  /predict/sample      → Live single prediction              │
│  /forecast/24h        → 96-block forecast + signals         │
│  /feature-importance  → Top 10 model drivers                │
│  /trading-simulation  → Historical P&L log                  │
│  /data/latest         → Live data freshness status          │
│  /eda                 → EDA dashboard (HTML)                │
│  /pestle              → PESTLE scenario analysis            │
│  /monitoring          → Full monitoring dashboard           │
│  /refresh             → Trigger immediate data refresh      │
│  /retrain             → Trigger model retraining            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT                                │
│   AWS EC2 t3.small          Docker Container                │
│   15.135.168.75:5000         Port 5000                       │
│   Auto-scheduler 30min      Chromium + Selenium             │
│   Auto-prediction 30min     IST timezone (UTC+5:30)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Group-05-IEX-Forecasting/
│
├── data/
│   ├── Price.xlsx                      # Mendeley dataset (2021-2023, hourly)
│   ├── iex_historical.csv              # Scraped IEX RTM data (2023-2026)
│   ├── iex_live.csv                    # Latest IEX 15-min blocks (auto-refreshed)
│   ├── weather_historical.csv          # OpenWeatherMap — 8 cities, 3 years
│   ├── weather_live.csv                # Latest weather readings (auto-refreshed)
│   ├── commodities_historical.csv      # Crude oil, gas, USD/INR
│   ├── commodities_live.csv            # Latest commodity prices (auto-refreshed daily)
│   ├── prediction_log.csv              # Every prediction logged with actual MCP backfilled
│   └── master_training_data.csv        # Final merged + engineered dataset
│
├── data_pipeline/
│   ├── scraper_iex.py                  # IEX website scraper (Selenium, headless)
│   ├── scraper_weather.py              # OpenWeatherMap API — 8 cities
│   ├── fetch_live_commodities.py       # NEW — Yahoo Finance live crude/gas/FX daily
│   ├── fetch_historical_commodities.py # Yahoo Finance + Frankfurter (historical)
│   ├── merge_historical.py             # Master merge with leakage-free features
│   ├── sync_live_files.py              # Copies historical to live CSVs
│   ├── backfill_actuals.py             # Fills actual MCP into prediction_log
│   ├── scheduler.py                    # Auto-refresh every 30 min + auto-prediction
│   ├── eda_generator.py                # Auto-generates EDA HTML report
│   ├── validator.py                    # Schema validation + dataset versioning
│   └── monitor.py                      # Drift detection + rolling metrics
│
├── models/
│   ├── best_model.pkl                  # Best model (XGBoost v10)
│   ├── scaler.pkl                      # StandardScaler
│   ├── feature_cols.pkl                # Feature column list
│   ├── model_metadata.json             # Version, MAPE, training info
│   ├── model_comparison.csv            # ARIMA vs SVM vs XGBoost results
│   ├── feature_importance.json         # Top feature importances
│   ├── business_evaluation.json        # P&L simulation results
│   ├── pestle_scenarios.json           # 7 PESTLE scenario results
│   └── archive/                        # Previous model versions (rollback)
│
├── app/
│   ├── app.py                          # Flask REST API (14 endpoints, all HTML)
│   └── static/
│       └── eda_report.html             # Auto-generated EDA dashboard
│
├── config.py
├── .env                                # OPENWEATHER_API_KEY
├── .env.example
├── run_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔬 CRISP-ML(Q) Framework Coverage

| Phase | Status | Key Deliverables |
|---|---|---|
| **1. Business Understanding** | ✅ | RTM market context, success metrics (MAPE < 25%), PESTLE analysis |
| **2. Data Understanding** | ✅ | 105K records, 3 sources, EDA report, schema validation |
| **3. Data Preparation** | ✅ | Option B merge, missing data handling, leakage-free features |
| **4. Modelling** | ✅ | ARIMA + SVM + XGBoost, TimeSeriesSplit CV, hyperparameter tuning |
| **5. Evaluation** | ✅ | MAPE/RMSE/MAE, business Rs evaluation, PESTLE scenarios |
| **6. Deployment** | ✅ | Flask API, Docker, AWS EC2, 14 endpoints, model versioning |
| **Quality** | ✅ | Schema validation, drift detection, prediction logging, rollback |

---

## 🧠 Feature Engineering (36 Features, All Leakage-Free)

All rolling statistics computed on **shifted MCP** — current price never appears in any feature.

| Category | Features |
|---|---|
| **Lag Features** | mcp_lag_1h, mcp_lag_2h, mcp_lag_24h, mcp_lag_48h, mcp_lag_1w |
| **Rolling Stats** | price_rolling_24h, price_rolling_1w, price_volatility, rolling_min, rolling_max |
| **Price Dynamics** | price_change_1h, price_change_24h, price_momentum |
| **Temporal** | hour, day_of_week, month, quarter, is_weekend, season, hour_bucket |
| **Weather (Delhi)** | temp_delhi, humidity_delhi, wind_delhi, cloud_delhi, cooling_degree, low_wind_flag |
| **Commodities** | crude_oil_usd, natural_gas_usd, usd_inr, coal_price_proxy |
| **Derived** | fuel_proxy (weighted crude + gas + coal), system_demand, load_price_ratio |

### Key Leakage Bugs Found & Fixed

| Bug | Impact | Fix |
|---|---|---|
| Price.xlsx extended to 2021 (fake ffill 15-min) | MAPE 79% | Date range locked to scraped period only |
| `seasonality = MCP - trend` used current MCP as feature | 49% importance, model saw answer | Removed entirely |
| `price_rolling_24h` included current MCP in window | Inflated lag correlation | All rolling on `MCP.shift(1)` |
| 6-month test split hit different price regime | MAPE 164% | 18-month training window |

---

## 🤖 Model Comparison

| Model | Test MAPE | RMSE (Rs/MWh) | CV MAPE | Notes |
|---|---|---|---|---|
| **XGBoost** | **18.31%** | 114 | 21% | Best — handles non-linearity |
| SVM | 69.48% | — | 21.28% | Good CV, poor extrapolation |
| ARIMA | 53.46% | — | — | Baseline, no exogenous features |

---

## 🌍 PESTLE Scenario Analysis

| Scenario | Category | Avg MCP (Rs/MWh) | Change vs Baseline |
|---|---|---|---|
| Baseline (Current) | — | 3,630 | — |
| Carbon Tax +20% | Political | 3,651 | +21 |
| Economic Recession -15% | Economic | 3,406 | -224 |
| Heatwave +8°C | Social/Environmental | 3,637 | +7 |
| Renewable Surge +30% | Technological | 3,406 | -224 |
| Price Cap Rs8,000 | Legal | 3,571 | -59 |
| Monsoon Season | Environmental | 3,548 | -82 |

---

## 📡 Monitoring Dashboard (`/monitoring`)

5-section operational health view:

**1 — Model Health Alert (Traffic Light)**
GREEN / YELLOW / RED based on model status, data age, rolling MAPE.

**2 — Data Drift (KS Test)**
KS test on MCP distribution vs 18-month training window.
Uses **KS statistic > 0.20** (practical significance) instead of p-value —
p-value is unreliable with 50,000+ samples (always significant by chance).
Only checks MCP — volume features excluded as they are not model inputs.

**3 — Last 30 Predictions (Past Blocks Only)**
Predicted vs actual MCP for past time blocks only.
Future forecast blocks are excluded — those belong in `/forecast/24h`.
Each prediction logged with its correct IST block timestamp.
Actuals auto-filled by `backfill_actuals.py` when IEX publishes data.

**4 — Data Pipeline Health**
All 6 data files — records, last modified, size, FRESH/WARNING/STALE status.

**5 — Price Distribution Shift**
Full Historical vs Last 30 Days vs Live Today — mean, median, std, percentiles, spike rate.

---

## 🔧 Production Fixes Applied (Post-Deployment)

| Issue | Root Cause | Fix |
|---|---|---|
| Commodities data stuck on Feb 23 | `fetch_historical_commodities.py` not running live | New `fetch_live_commodities.py` — Yahoo Finance BZ=F, NG=F + Frankfurter USD/INR |
| Monitoring predictions never updated | Scheduler refreshed data but never generated new predictions | Added `auto_predict()` to scheduler — hits `/forecast/24h` every 30 min |
| All 96 predictions had same timestamp | `log_pred()` used `datetime.now()` instead of forecast block time | Fixed `log_pred(log_ts=dt)` — each block gets its correct IST timestamp |
| Monitoring showed tomorrow's predictions | `tail(30)` picks newest rows = future forecast blocks | Added filter: only show predictions where timestamp ≤ now (IST) |
| Data freshness showing 330 min stale | `file_age_minutes()` used IST but CSV timestamps stored in UTC | Reverted `file_age_minutes()` to UTC `datetime.now()` |
| KS drift false positive on all features | Comparing live vs full 3-year history including 2023 high-price regime | Filter historical to 18-month training window before KS test |
| KS p-value always 0.000 | p-value meaningless with 50K+ samples | Use KS statistic > 0.20 threshold (practical significance) |
| Drift flagging non-model features | Checking purchase_bid, sell_bid, mcv which model doesn't use | Only check MCP — the target variable distribution |

---

## 🚀 Quick Start

### Local Setup
```bash
git clone https://github.com/praveenp1118/Group-05-IEX-Forecasting.git
cd Group-05-IEX-Forecasting
copy .env.example .env
pip install -r requirements.txt
python run_pipeline.py
docker-compose up -d --build
# Open http://localhost:5000
```

### Useful Docker Commands
```bash
# Logs
docker logs group05-iex-api --follow

# Test live commodities scraper
docker exec group05-iex-api python3 data_pipeline/fetch_live_commodities.py

# Backfill actual MCP into prediction log
docker exec group05-iex-api python3 data_pipeline/backfill_actuals.py

# Regenerate EDA report
docker exec group05-iex-api python3 data_pipeline/eda_generator.py

# Trigger data refresh
docker exec group05-iex-api python3 data_pipeline/scheduler.py

# Stop
docker-compose down
```

### Deploy Updated Files to EC2
```powershell
# From local PowerShell
scp -i "C:\Users\prave\Downloads\group05-key.pem" "D:\Group-05-IEX-Forecasting\app\app.py" ec2-user@15.135.168.75:~/app.py
scp -i "C:\Users\prave\Downloads\group05-key.pem" "D:\Group-05-IEX-Forecasting\data_pipeline\scheduler.py" ec2-user@15.135.168.75:~/scheduler.py
```
```bash
# From SSH
docker cp ~/app.py group05-iex-api:/app/app/app.py
docker cp ~/scheduler.py group05-iex-api:/app/data_pipeline/scheduler.py
docker restart group05-iex-api
```

---

## 🔌 API Endpoints (14 total — all HTML)

| Endpoint | Description |
|---|---|
| `/` | Home page with all endpoint links |
| `/model-summary` | Model card — models, metrics, data, bugs fixed |
| `/health` | Model version, MAPE, data freshness |
| `/predict` | Interactive form — pre-filled with live data |
| `/predict/sample` | Quick single prediction |
| `/forecast/24h` | 96-block forecast + BUY/SELL/HOLD signals |
| `/feature-importance` | Top 10 drivers with bar chart |
| `/trading-simulation` | Simulated P&L from prediction log |
| `/data/latest` | IEX + weather + commodity status |
| `/eda` | Full EDA — STL, ACF/PACF, seasonality, correlations |
| `/pestle` | 7 PESTLE scenarios + methodology |
| `/monitoring` | Health, drift, predictions, pipeline, distribution |
| `/refresh` | Trigger live data refresh |
| `/retrain` | Trigger model retraining |

---

## 📦 Dependencies

```
flask, pandas, numpy, scikit-learn, xgboost
statsmodels, matplotlib, seaborn, scipy
openpyxl, requests, gunicorn
python-dotenv, selenium, webdriver-manager
yfinance (optional — fetch_live_commodities.py falls back to requests)
```

---

## 📈 Business Value

- **65.7% improvement** over ARIMA baseline
- **Rs 7.7 Crore simulated P&L** over 3-month test period (100 MW volume)
- **24-hour forecast** with per-block BUY/SELL/HOLD signals
- **Live AWS deployment** — http://15.135.168.75:5000
- **Auto-refresh** every 30 minutes (data + predictions)
- **Full monitoring** — health, drift, rolling MAPE, pipeline, distribution shift
- **Model rollback** — previous versions in `models/archive/`

---

## 👥 Team

**Group 05 — ISB AMPBA**

---

## 📚 Data Sources

| Source | Data | Period |
|---|---|---|
| IEX Website (Selenium) | RTM 15-min MCP, volumes | Feb 2023 – present |
| Mendeley (Price.xlsx) | Hourly MCP + demand | 2021–2023 (gap fill) |
| OpenWeatherMap API | Weather — 8 Indian cities | Live + historical |
| Yahoo Finance (BZ=F, NG=F) | Crude oil, natural gas | Live + historical |
| Frankfurter API | USD/INR exchange rate | Live + historical |

---

*Built with CRISP-ML(Q) framework | ISB AMPBA Foundation Project | Deployed on AWS EC2*
