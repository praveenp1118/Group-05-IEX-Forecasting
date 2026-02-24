# ⚡ Group 05 — IEX RTM Electricity Price Forecasting
### ISB AMPBA | Foundation Project | CRISP-ML(Q) Framework

> **Real-time 15-minute RTM clearing price forecasting for the Indian Energy Exchange (IEX)**  
> Predicts MCP (Market Clearing Price) in Rs/MWh to support power trading decisions

---

## 🌐 Live Deployment (AWS EC2)

| | |
|---|---|
| **Public URL** | http://13.236.44.97:5000 |
| **Home Page** | http://13.236.44.97:5000/ |
| **Health Check** | http://13.236.44.97:5000/health |
| **Interactive Predictor** | http://13.236.44.97:5000/predict |
| **24h Forecast** | http://13.236.44.97:5000/forecast/24h |
| **EDA Dashboard** | http://13.236.44.97:5000/eda |
| **Server** | AWS EC2 t3.small — ap-southeast-2 (Sydney) |
| **Container** | Docker (Python 3.11 + Chromium/Selenium) |

---

## 📊 Results Summary

| Metric | Value |
|---|---|
| **Best Model** | XGBoost v7 |
| **Test MAPE** | 20.65% |
| **ARIMA Baseline MAPE** | 53.46% |
| **Improvement over Baseline** | 63.8% |
| **Simulated P&L** | ₹7.7 Crore (100 MW, 3 months) |
| **Training Data** | 52,705 records (Aug 2024 – Feb 2026) |
| **Forecast Horizon** | 96 blocks × 15 min = 24 hours |
| **Distribution Shift** | 4.2% (train vs test, same regime) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                             │
│  IEX Website      NASA POWER API    Yahoo Finance           │
│  (RTM 15-min)     (8 cities, 3yr)   (Crude/Gas/FX)          │
└────────┬──────────────┬──────────────────┬──────────────────┘
         │              │                  │
         ▼              ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA PIPELINE                               │
│  scraper_iex.py      scraper_weather.py                     │
│  fetch_historical_commodities.py                            │
│        │               │                   │                │
│        └───────────────┴───────────────────┘                │
│                        │                                    │
│              merge_historical.py                            │
│         (Option B: Scraped + Price.xlsx gap fill)           │
│         (Leakage-free rolling features on shifted MCP)      │
│                        │                                    │
│              sync_live_files.py                             │
│         (Copies latest rows to *_live.csv every 30 min)     │
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
│  Step 3: Train ARIMA + SVM + XGBoost (3-fold CV)           │
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
│  /health              → System health + data freshness      │
│  /predict             → Interactive HTML prediction form    │
│  /predict/sample      → Live single prediction (JSON)       │
│  /forecast/24h        → 96-block forecast + signals         │
│  /feature-importance  → Top 10 model drivers                │
│  /trading-simulation  → Historical P&L log                  │
│  /data/latest         → Live data freshness status          │
│  /eda                 → EDA dashboard (HTML)                │
│  /monitoring          → Drift detection                     │
│  /refresh             → Trigger immediate data refresh      │
│  /retrain             → Trigger model retraining            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT                                │
│   AWS EC2 t3.small          Docker Container                │
│   13.236.44.97:5000         Port 5000                       │
│   Auto-scheduler 30min      Chromium + Selenium             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Group-05-IEX-Forecasting/
│
├── data/                               # All data files
│   ├── Price.xlsx                      # Mendeley dataset (2021-2023, hourly)
│   ├── iex_historical.csv              # Scraped IEX RTM data (2023-2026)
│   ├── iex_live.csv                    # Latest IEX 15-min blocks (auto-refreshed)
│   ├── weather_historical.csv          # NASA POWER — 8 cities, 3 years
│   ├── weather_live.csv                # Latest weather readings (auto-refreshed)
│   ├── commodities_historical.csv      # Crude oil, gas, USD/INR
│   ├── commodities_live.csv            # Latest commodity prices (auto-refreshed)
│   └── master_training_data.csv        # Final merged + engineered dataset
│
├── data_pipeline/                      # All data scripts
│   ├── scraper_iex.py                  # IEX website scraper (Selenium, headless)
│   ├── scraper_weather.py              # NASA POWER API — 8 cities
│   ├── fetch_historical_commodities.py # Yahoo Finance + Frankfurter
│   ├── merge_historical.py             # Master merge with leakage-free features
│   ├── sync_live_files.py              # Copies historical to live CSVs
│   ├── scheduler.py                    # Auto-refresh every 30 min
│   ├── eda_generator.py                # Auto-generates EDA HTML report
│   ├── validator.py                    # Schema validation + dataset versioning
│   └── monitor.py                      # Drift detection + rolling metrics
│
├── models/                             # Trained model artifacts
│   ├── best_model.pkl                  # Best model (XGBoost v7)
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
│   ├── app.py                          # Flask REST API (12 endpoints)
│   └── static/
│       └── eda_report.html             # Auto-generated EDA dashboard
│
├── config.py                           # Central config — reads from .env
├── .env.example                        # API key template — safe to commit
├── .env                                # Real API keys — gitignored
├── run_pipeline.py                     # End-to-end pipeline runner
├── diagnose.py                         # Model diagnostic tool
├── Dockerfile                          # Docker + Chromium for Selenium
├── docker-compose.yml                  # Docker compose config
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
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
| **6. Deployment** | ✅ | Flask API, Docker, AWS EC2, 12 endpoints, model versioning |
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
| Price.xlsx extended data to 2021 (fake ffill 15-min) | MAPE 79% | Date range locked to scraped period only |
| `seasonality = MCP - trend` used current MCP as feature | 49% feature importance, model saw the answer | Removed entirely |
| `price_rolling_24h` included current MCP in rolling window | Inflated lag correlation | All rolling on `MCP.shift(1)` |
| 6-month test split hit different price regime (Rs4605 vs Rs3390) | MAPE 164% | 18-month training window keeps same regime |

---

## 🤖 Model Comparison

| Model | Test MAPE | RMSE (Rs/MWh) | CV MAPE | Notes |
|---|---|---|---|---|
| **XGBoost** | **20.65%** | 114 | 21% | Best — handles non-linearity |
| SVM | 69.48% | — | 21.28% | Good CV, poor extrapolation |
| ARIMA | 53.46% | — | — | Baseline, no exogenous features |

**Top 5 Features (XGBoost):**
1. `mcp_lag_1h` — 0.315 (most recent price)
2. `mcp_lag_24h` — 0.142 (same time yesterday)
3. `price_rolling_24h` — 0.098
4. `hour` — 0.071
5. `temp_delhi` — 0.052

---

## 🌍 PESTLE Scenario Analysis

| Scenario | Avg MCP (Rs/MWh) | Change vs Baseline |
|---|---|---|
| Baseline (Current) | 3,630 | — |
| Carbon Tax +20% | 3,651 | +21 |
| Economic Recession -15% | 3,406 | -224 |
| Heatwave +8C | 3,637 | +7 |
| Renewable Surge +30% | 3,406 | -224 |
| Price Cap 8000 | 3,571 | -59 |
| Monsoon Season | 3,548 | -82 |

---

## 🚀 Quick Start

### Local Setup
```bash
# 1. Clone
git clone https://github.com/praveenp1118/Group-05-IEX-Forecasting.git
cd Group-05-IEX-Forecasting

# 2. Set up environment
copy .env.example .env
# Edit .env with your API keys if needed

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run pipeline (trains model, generates EDA)
python run_pipeline.py

# 5. Start API via Docker
docker-compose up -d

# 6. Open browser
# http://localhost:5000
```

### Docker Commands
```bash
# Build and run
docker-compose up -d --build

# Check logs
docker logs group05-iex-api --follow

# Trigger data refresh
# http://localhost:5000/refresh

# Stop
docker-compose down
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | HTML home page with clickable endpoint links |
| `/health` | GET | Model version, MAPE, live data freshness |
| `/predict` | GET/POST | **Interactive HTML form** — pre-filled with live data, adjust & predict |
| `/predict/sample` | GET | Quick single prediction using live data (JSON) |
| `/forecast/24h` | GET | 96-block forecast + BUY/SELL/HOLD signals |
| `/feature-importance` | GET | Top 10 model drivers |
| `/trading-simulation` | GET | Historical P&L from prediction log |
| `/data/latest` | GET | IEX + weather + commodity freshness status |
| `/eda` | GET | EDA dashboard (HTML) |
| `/monitoring` | GET | Drift detection + rolling MAPE |
| `/refresh` | GET | **Trigger immediate live data refresh** |
| `/retrain` | GET | Trigger background model retraining |

### Sample Response — `/forecast/24h`
```json
{
  "model": "XGBoost",
  "version": "v7",
  "mape": "20.65%",
  "business_metrics": {
    "avg_mcp_24h": 3630,
    "estimated_savings": 4200000,
    "buy_windows": 18,
    "sell_windows": 12,
    "optimal_buy_time": "2026-02-24 03:15"
  },
  "forecast": [
    {"block": 1, "datetime": "2026-02-24 02:30",
     "predicted_mcp": 3245.6, "confidence": "HIGH",
     "signal": "HOLD"}
  ]
}
```

---

## 🔒 Security

API keys managed via `.env` — never hardcoded in source:

```bash
# .env (gitignored — never committed)
OPENWEATHER_API_KEY=your_real_key

# .env.example (committed — safe template for teammates)
OPENWEATHER_API_KEY=your_key_here
```

`config.py` loads `.env` at startup and provides keys to all modules.

---

## 📈 Business Value

- **63.8% improvement** over ARIMA baseline
- **Rs 7.7 Crore simulated P&L** over 3-month test period (100 MW volume)
- **24-hour forecast** with per-block BUY/SELL/HOLD signals
- **Live AWS deployment** — accessible from anywhere at `http://13.236.44.97:5000`
- **Auto-refresh** — IEX + weather + commodities updated every 30 minutes
- **Interactive predictor** — change inputs in browser, get instant price forecast
- **Model rollback** — previous versions archived in `models/archive/`

---

## 👥 Team

**Group 05 — ISB AMPBA**

---

## 📚 Data Sources

| Source | Data | Period |
|---|---|---|
| IEX Website (Selenium scraper) | RTM 15-min MCP, volumes | Feb 2023 – present |
| Mendeley (Price.xlsx) | Hourly MCP + demand | 2021–2023 (gap fill only) |
| NASA POWER API | Weather — 8 Indian cities | 3 years |
| Yahoo Finance | Crude oil, natural gas | 3 years |
| Frankfurter API | USD/INR exchange rate | 3 years |

---

*Built with CRISP-ML(Q) framework | ISB AMPBA Foundation Project | Deployed on AWS EC2*
