# CLAUDE CONTEXT SNAPSHOT
# Group 05 — IEX RTM Electricity Price Forecasting
# Upload this file to Claude to restore full project context instantly
# Last updated: 03-Mar-2026

---

## WHO I AM
- Praveen, ISB AMPBA student, Group 05
- Foundation Project: IEX RTM electricity price forecasting
- Framework: CRISP-ML(Q)
- EC2 SSH: ec2-user@15.135.168.75 (AWS ap-southeast-2 Sydney)
- PEM file: C:\Users\prave\Downloads\group05-key.pem
- Connect via: AWS EC2 Instance Connect (browser SSH) — PEM-based SSH doesn't work from inside EC2
- SCP from: local PowerShell only (not from inside SSH session)
- Local project path: D:\Group-05-IEX-Forecasting
- GitHub: https://github.com/praveenp1118/Group-05-IEX-Forecasting

---

## LIVE DEPLOYMENT
- URL: http://15.135.168.75:5000
- Container: group05-iex-api
- Docker compose file: D:\Group-05-IEX-Forecasting\docker-compose.yml
- Volumes mounted: ./data, ./models, ./app/static (persist outside container)
- Dockerfile: python:3.11-slim + Chromium + chromedriver
- ENV CHROME_BIN=/usr/bin/chromium
- ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

---

## MODEL
- Best model: XGBoost v10
- Test MAPE: 18.31%
- ARIMA baseline MAPE: 53.46%
- Training window: last 18 months only (regime-aligned — avoids 2023 high-price regime)
- Training records: 52,705
- Features: 36 (all leakage-free, all rolling on MCP.shift(1))
- Files: models/best_model.pkl, models/scaler.pkl, models/feature_cols.pkl
- Metadata: models/model_metadata.json

---

## PROJECT STRUCTURE
```
Group-05-IEX-Forecasting/
├── data/
│   ├── iex_historical.csv          # 104,960+ records, 2023-present
│   ├── iex_live.csv                # Latest 300 rows, auto-refreshed
│   ├── weather_live.csv            # 8 cities, OpenWeatherMap API
│   ├── weather_historical.csv
│   ├── commodities_live.csv        # Crude/gas/FX, refreshed daily
│   ├── commodities_historical.csv
│   ├── prediction_log.csv          # Every prediction + actual MCP backfilled
│   └── master_training_data.csv
├── data_pipeline/
│   ├── scraper_iex.py              # Selenium headless Chrome, React page
│   ├── scraper_weather.py          # OpenWeatherMap API (NOT NASA)
│   ├── fetch_live_commodities.py   # Yahoo Finance BZ=F NG=F + Frankfurter
│   ├── fetch_historical_commodities.py
│   ├── merge_historical.py
│   ├── sync_live_files.py
│   ├── backfill_actuals.py         # Fills actual_mcp into prediction_log
│   ├── scheduler.py                # 30-min refresh + auto_predict()
│   ├── eda_generator.py
│   └── validator.py
├── models/
│   └── archive/                    # Previous versions for rollback
├── app/
│   └── app.py                      # Flask API, 14 endpoints
├── run_pipeline.py                 # Full retrain pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                            # OPENWEATHER_API_KEY=your_key_here
```

---

## API ENDPOINTS (14 total)
| Endpoint | Description |
|---|---|
| / | Home |
| /model-summary | Full model card |
| /health | System health + freshness |
| /predict | Interactive form |
| /predict/sample | Quick prediction |
| /forecast/24h | 96-block forecast + signals |
| /feature-importance | Top 10 features |
| /trading-simulation | P&L log |
| /data/latest | Live data status |
| /eda | EDA dashboard |
| /pestle | 7 PESTLE scenarios |
| /monitoring | Full monitoring dashboard |
| /refresh | Trigger data refresh |
| /retrain | Trigger retraining |

---

## KEY CODE DETAILS

### app.py
- IST timezone: `IST = timezone(timedelta(hours=5, minutes=30))` + `now_ist()`
- `file_age_minutes()` uses UTC `datetime.now()` (CSV timestamps stored in UTC)
- `log_pred(features, pred, conf, version, block=1, log_ts=None)` — log_ts=dt fixes timestamp
- `build_24h_forecast()` passes `log_ts=dt` so each block gets correct IST timestamp
- Monitoring filters: `pred_df[pred_df["_ts"] <= pd.Timestamp(now_ist())]` — past only
- KS drift: compares against 18-month window, uses `stat > 0.20` not p-value
- KS drift: only checks MCP (volume features excluded — not model inputs)
- Freshness thresholds: WARN=20min, STALE=45min
- Scheduler starts in `startup()` via `from data_pipeline.scheduler import start_scheduler`

### scheduler.py
- REFRESH_INTERVAL = 30 * 60
- Calls: scraper_iex.py → scraper_weather.py → fetch_live_commodities.py
- Then: sync_live_files.py → backfill_actuals.py → auto_predict()
- auto_predict() hits http://localhost:5000/forecast/24h (3 retries, 15s apart)
- find_scraper() searches: fetch_live_commodities.py first, then fetch_historical_commodities.py

### fetch_live_commodities.py
- Fetches: BZ=F (Brent crude), NG=F (natural gas) from Yahoo Finance v8 API
- Fetches: USD/INR from Frankfurter API (free, no key)
- Falls back to yfinance library if requests fails
- Saves to commodities_live.csv (tail 30) + appends to commodities_historical.csv
- BASE_DIR = "/app" (hardcoded for Docker)

### backfill_actuals.py
- Builds lookup: 'DD-MM-YYYY|HH:MM-HH:MM' -> MCP from iex_historical + iex_live
- ts_to_key(): rounds prediction timestamp down to 15-min block
- Fills actual_mcp for past blocks, computes rolling MAPE

### scraper_iex.py
- URL: https://www.iexindia.com/market-data/real-time-market/market-snapshot
- React page — waits for table render, uses JS extraction fallback
- Saves to iex_live.csv (tail 500) + appends to iex_historical.csv
- MCP filter: 100 < MCP < 20000

---

## BUGS FIXED (03-Mar-2026 session)

| # | Issue | Fix |
|---|---|---|
| 1 | Commodities stuck on Feb 23 | New fetch_live_commodities.py |
| 2 | Predictions never auto-updated | auto_predict() added to scheduler |
| 3 | All 96 predictions same timestamp | log_pred(log_ts=dt) |
| 4 | Monitoring showed tomorrow's blocks | Filter to past timestamps only |
| 5 | Data freshness showing 330min stale | file_age_minutes uses UTC not IST |
| 6 | KS drift false positive | 18-month window + stat>0.20 threshold |
| 7 | Drift flagging non-model features | Only check MCP |

---

## DEPLOY COMMANDS

### Local to EC2 (PowerShell)
```powershell
scp -i "C:\Users\prave\Downloads\group05-key.pem" "D:\Group-05-IEX-Forecasting\app\app.py" ec2-user@15.135.168.75:~/app.py
scp -i "C:\Users\prave\Downloads\group05-key.pem" "D:\Group-05-IEX-Forecasting\data_pipeline\scheduler.py" ec2-user@15.135.168.75:~/scheduler.py
scp -i "C:\Users\prave\Downloads\group05-key.pem" "D:\Group-05-IEX-Forecasting\data_pipeline\fetch_live_commodities.py" ec2-user@15.135.168.75:~/fetch_live_commodities.py
```

### EC2 SSH (copy into container + restart)
```bash
docker cp ~/app.py group05-iex-api:/app/app/app.py
docker cp ~/scheduler.py group05-iex-api:/app/data_pipeline/scheduler.py
docker cp ~/fetch_live_commodities.py group05-iex-api:/app/data_pipeline/fetch_live_commodities.py
docker restart group05-iex-api
docker logs group05-iex-api --follow
```

### Useful one-liners
```bash
# Force fresh predictions
docker exec group05-iex-api python3 -c "import requests; print(requests.get('http://localhost:5000/forecast/24h', timeout=45).status_code)"

# Backfill actuals
docker exec group05-iex-api python3 data_pipeline/backfill_actuals.py

# Test commodities
docker exec group05-iex-api python3 data_pipeline/fetch_live_commodities.py

# Clear prediction log
docker exec group05-iex-api python3 -c "import os; os.remove('/app/data/prediction_log.csv'); print('cleared')"

# Check scheduler running
docker logs group05-iex-api --tail=50 | grep -E "Scheduler|Refresh|IEX|Weather|Commodities"
```

### Git push
```bash
cd D:\Group-05-IEX-Forecasting
git add app/app.py data_pipeline/scheduler.py data_pipeline/fetch_live_commodities.py README.md
git commit -m "your message"
git push origin main
```

---

## KNOWN ISSUES / WATCH LIST
- Rolling MAPE inflated by Feb-25 anomaly (actual=Rs10,000 regulatory cap) — will dilute over time
- EC2 server timezone is UTC; all IST conversion handled in app.py via now_ist()
- Prediction log accumulates 96 rows per /forecast/24h call — monitor file size over time
- IEX website is React-based — if scraper returns 0 records, falls back to iex_historical.csv

---

## PRESENTATION NOTES (for ISB faculty review)
- CRISP-ML(Q) all 6 phases complete
- 18-month window: justified by regime shift (2023 avg Rs5,805 vs 2025 avg Rs3,900)
- Drift methodology: KS statistic > 0.20 preferred over p-value due to large sample size (50K+)
- Feb-25 Rs10,000 actuals: genuine regulatory cap event, not model failure — demonstrates monitoring caught a real anomaly
- Test MAPE 18.31% beats target of <25%
- 63.8% improvement over ARIMA baseline
