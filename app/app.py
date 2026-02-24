"""
app.py - Group 05 IEX RTM Price Forecasting
Works both locally (python app/app.py) and in Docker
All 10 endpoints functional
"""

from flask import Flask, jsonify, request, send_file
import pickle, os, sys, json, threading, time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Path fix — works locally AND in Docker ────────────────────
_THIS_FILE = os.path.abspath(__file__)
_APP_DIR   = os.path.dirname(_THIS_FILE)
_BASE_DIR  = os.path.dirname(_APP_DIR)
for p in [_BASE_DIR, _APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Load .env keys via config.py
try:
    from config import load_env
    load_env()
except Exception as e:
    print(f"Config load: {e}")

app = Flask(__name__)

MODELS_DIR   = os.path.join(_BASE_DIR, "models")
DATA_DIR     = os.path.join(_BASE_DIR, "data")
STATIC_DIR   = os.path.join(_APP_DIR,  "static")
PRED_LOG     = os.path.join(DATA_DIR,  "prediction_log.csv")
RETRAIN_FLAG = os.path.join(MODELS_DIR,"retrain_flag.txt")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

FRESHNESS_WARN  = 20
FRESHNESS_STALE = 45

# ── Model State ───────────────────────────────────────────────
state = {
    "model": None, "scaler": None, "feature_cols": None,
    "loaded_at": None, "flag_mtime": None,
    "model_name": "unknown", "mape": None, "version": "v0",
}

def load_model():
    try:
        state["model"]        = pickle.load(open(os.path.join(MODELS_DIR,"best_model.pkl"),  "rb"))
        state["scaler"]       = pickle.load(open(os.path.join(MODELS_DIR,"scaler.pkl"),      "rb"))
        state["feature_cols"] = pickle.load(open(os.path.join(MODELS_DIR,"feature_cols.pkl"),"rb"))
        state["loaded_at"]    = datetime.now().isoformat()
        state["flag_mtime"]   = os.path.getmtime(RETRAIN_FLAG) if os.path.exists(RETRAIN_FLAG) else None
        meta = os.path.join(MODELS_DIR,"model_metadata.json")
        if os.path.exists(meta):
            with open(meta) as f: m = json.load(f)
            state["model_name"] = m.get("best_model","XGBoost")
            state["mape"]       = m.get("mape")
            state["version"]    = m.get("version","v1")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded: "
              f"{state['model_name']} {state['version']} | MAPE: {state['mape']:.2f}%")
        return True
    except Exception as e:
        print(f"Model load error: {e}"); return False

def model_watcher():
    while True:
        try:
            if os.path.exists(RETRAIN_FLAG):
                mt = os.path.getmtime(RETRAIN_FLAG)
                if state["flag_mtime"] != mt:
                    print("Retrain flag changed — reloading model...")
                    load_model()
        except: pass
        time.sleep(10)

# ── Freshness Helpers ─────────────────────────────────────────

def file_age_minutes(filepath):
    """
    Check file age — also checks fallback files if primary not found.
    e.g. commodities_live.csv → commodities_historical.csv
         iex_live.csv → iex_historical.csv
    """
    # Build list of files to check (primary + fallbacks)
    candidates = [filepath]
    fname = os.path.basename(filepath)
    fdir  = os.path.dirname(filepath)
    if "_live" in fname:
        fallback = os.path.join(fdir, fname.replace("_live", "_historical"))
        candidates.append(fallback)

    for fpath in candidates:
        if not os.path.exists(fpath): continue
        try:
            df = pd.read_csv(fpath)
            if len(df) == 0: continue
            # Try timestamp columns
            for col in ["scrape_timestamp", "timestamp", "datetime", "date"]:
                if col not in df.columns: continue
                series = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(series) == 0: continue
                latest = series.iloc[-1]
                age    = (datetime.now() - latest).total_seconds() / 60
                return max(0, int(age))
            # Fallback: file modification time
            return int((datetime.now().timestamp() - os.path.getmtime(fpath)) / 60)
        except: continue
    return 9999

def freshness_label(age):
    if age < FRESHNESS_WARN:  return "FRESH"
    if age < FRESHNESS_STALE: return "WARNING"
    return "STALE"

def ensure_fresh():
    iex = file_age_minutes(os.path.join(DATA_DIR,"iex_live.csv"))
    wx  = file_age_minutes(os.path.join(DATA_DIR,"weather_live.csv"))
    if max(iex,wx) >= FRESHNESS_STALE:
        def _scrape():
            import subprocess
            for s in ["scraper_iex.py","scraper_weather.py"]:
                sp = os.path.join(_BASE_DIR,"data_pipeline",s)
                if os.path.exists(sp):
                    try: subprocess.run(["python",sp],timeout=60,capture_output=True)
                    except: pass
        t = threading.Thread(target=_scrape, daemon=True); t.start(); t.join(timeout=90)
        iex = file_age_minutes(os.path.join(DATA_DIR,"iex_live.csv"))
    return {
        "iex_age_minutes":     iex if iex<9999 else "N/A",
        "weather_age_minutes": wx  if wx <9999 else "N/A",
        "iex_status":          freshness_label(iex),
        "weather_status":      freshness_label(wx),
        "overall":             freshness_label(max(iex,wx)),
    }

# ── Live Feature Loader ───────────────────────────────────────

def load_live_features():
    feats = {}
    iex_file = os.path.join(DATA_DIR,"iex_live.csv")
    if os.path.exists(iex_file):
        iex = pd.read_csv(iex_file).tail(200)
        if len(iex)>0 and "MCP" in iex.columns:
            mcp = iex["MCP"]
            feats["mcp_lag_1h"]          = mcp.iloc[-4]   if len(mcp)>=4   else mcp.iloc[-1]
            feats["mcp_lag_2h"]          = mcp.iloc[-8]   if len(mcp)>=8   else mcp.iloc[-1]
            feats["mcp_lag_24h"]         = mcp.iloc[-96]  if len(mcp)>=96  else mcp.iloc[-1]
            feats["mcp_lag_48h"]         = mcp.iloc[-192] if len(mcp)>=192 else mcp.iloc[-1]
            feats["mcp_lag_1w"]          = mcp.iloc[-672] if len(mcp)>=672 else mcp.iloc[-1]
            mcp_s = mcp.shift(1)
            feats["price_rolling_24h"]   = mcp_s.rolling(96,  min_periods=1).mean().iloc[-1]
            feats["price_rolling_1w"]    = mcp_s.rolling(672, min_periods=1).mean().iloc[-1]
            feats["price_volatility"]    = mcp_s.rolling(96,  min_periods=1).std().iloc[-1] or 0
            feats["price_rolling_min"]   = mcp_s.rolling(96,  min_periods=1).min().iloc[-1]
            feats["price_rolling_max"]   = mcp_s.rolling(96,  min_periods=1).max().iloc[-1]
            feats["price_change_1h"]     = feats["mcp_lag_1h"]  - feats["mcp_lag_2h"]
            feats["price_change_24h"]    = feats["mcp_lag_24h"] - feats["mcp_lag_48h"]
            feats["price_momentum"]      = feats["mcp_lag_1h"]  - feats["price_rolling_24h"]
            if "mcv_mw" in iex.columns:
                sd = iex["mcv_mw"].iloc[-1]*1.2
                feats["system_demand"]   = sd
                feats["demand_lag_24h"]  = iex["mcv_mw"].iloc[-96]*1.2 if len(iex)>=96 else sd
                feats["demand_change"]   = feats["system_demand"]-feats["demand_lag_24h"]
                feats["load_price_ratio"]= sd/(feats["mcp_lag_1h"]+1)

    wx_file = os.path.join(DATA_DIR,"weather_live.csv")
    if os.path.exists(wx_file):
        wx  = pd.read_csv(wx_file)
        row = wx[wx["city"]=="Delhi"].iloc[-1] if "city" in wx.columns and "Delhi" in wx["city"].values else wx.iloc[-1]
        feats["temp_delhi"]     = float(row.get("temperature",30))
        feats["humidity_delhi"] = float(row.get("humidity",   60))
        feats["wind_delhi"]     = float(row.get("wind_speed",  3))
        feats["cloud_delhi"]    = float(row.get("cloud_cover", 30))
        feats["pressure_delhi"] = float(row.get("pressure",1010))
        feats["cooling_degree"] = max(feats["temp_delhi"]-25, 0)
        feats["low_wind_flag"]  = 1 if feats["wind_delhi"]<2 else 0

    com_file = os.path.join(DATA_DIR,"commodities_live.csv")
    if os.path.exists(com_file):
        com = pd.read_csv(com_file).iloc[-1]
        feats["crude_oil_usd"]    = float(com.get("crude_oil_usd",   75))
        feats["natural_gas_usd"]  = float(com.get("natural_gas_usd",  3))
        feats["usd_inr"]          = float(com.get("usd_inr",         84))
        feats["coal_price_proxy"] = float(com.get("coal_price_proxy", feats.get("crude_oil_usd",75)*1.8))
        feats["coal_price"]       = feats["crude_oil_usd"]*1.8
        feats["fuel_proxy"]       = feats["crude_oil_usd"]*0.4 + feats["natural_gas_usd"]*10 + feats["coal_price"]*0.3

    now = datetime.now()
    feats["hour"]        = now.hour
    feats["day_of_week"] = now.weekday()
    feats["month"]       = now.month
    feats["quarter"]     = (now.month-1)//3+1
    feats["is_weekend"]  = 1 if now.weekday()>=5 else 0
    feats["season"]      = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[now.month]
    feats["hour_bucket"] = 0 if now.hour<=5 else 1 if now.hour<=9 else 2 if now.hour<=17 else 3 if now.hour<=21 else 4
    return feats

# ── Prediction Helpers ────────────────────────────────────────

def predict_price(features):
    cols = state["feature_cols"]
    X    = pd.DataFrame([{c: features.get(c,0) for c in cols}])[cols].fillna(0)
    try:    return float(np.clip(state["model"].predict(X)[0], 0, 20000))
    except:
        Xs = state["scaler"].transform(X)
        return float(np.clip(state["model"].predict(Xs)[0], 0, 20000))

def confidence(pred, vol, block=1):
    vp = vol/(pred+1)*100
    if block<=4  and vp<5:   return "HIGH"
    if block<=48 and vp<15:  return "MEDIUM"
    if block>48  or  vp>25:  return "LOW"
    return "MEDIUM"

def signal(pred, prev):
    if pred > prev*1.05: return "BUY",  "Price rising — buy now"
    if pred < prev*0.95: return "SELL", "Price falling — sell now"
    return "HOLD", "Price stable — hold position"

def log_pred(features, pred, conf, version, block=1):
    row = {"timestamp":datetime.now().isoformat(),"model_version":version,
           "horizon_block":block,"predicted_mcp":round(pred,2),"confidence":conf,
           "mcp_lag_1h":features.get("mcp_lag_1h"),"temp_delhi":features.get("temp_delhi"),
           "hour":features.get("hour"),"is_weekend":features.get("is_weekend"),"actual_mcp":None}
    df = pd.DataFrame([row])
    if os.path.exists(PRED_LOG): df.to_csv(PRED_LOG,mode="a",header=False,index=False)
    else: df.to_csv(PRED_LOG,index=False)

# ── 24h Forecast ──────────────────────────────────────────────

def build_24h_forecast(base):
    forecasts, cur, hist = [], dict(base), [base.get("mcp_lag_1h",3500)]*300
    now = datetime.now().replace(second=0,microsecond=0)
    for block in range(1,97):
        dt = now + timedelta(minutes=15*block)
        cur.update({"hour":dt.hour,"day_of_week":dt.weekday(),"month":dt.month,
                    "quarter":(dt.month-1)//3+1,"is_weekend":1 if dt.weekday()>=5 else 0,
                    "season":{12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[dt.month],
                    "hour_bucket":0 if dt.hour<=5 else 1 if dt.hour<=9 else 2 if dt.hour<=17 else 3 if dt.hour<=21 else 4})
        if len(hist)>=4:   cur["mcp_lag_1h"]         = hist[-4]
        if len(hist)>=8:   cur["mcp_lag_2h"]         = hist[-8]
        if len(hist)>=96:  cur["mcp_lag_24h"]        = hist[-96]
        if len(hist)>=192: cur["mcp_lag_48h"]        = hist[-192]
        cur["price_rolling_24h"] = np.mean(hist[-96:]) if len(hist)>=96 else np.mean(hist)
        cur["price_volatility"]  = np.std(hist[-96:])  if len(hist)>=96 else np.std(hist) or 0
        cur["price_rolling_min"] = np.min(hist[-96:])  if len(hist)>=96 else np.min(hist)
        cur["price_rolling_max"] = np.max(hist[-96:])  if len(hist)>=96 else np.max(hist)
        cur["price_change_1h"]   = cur["mcp_lag_1h"] - cur["mcp_lag_2h"]
        cur["price_momentum"]    = cur["mcp_lag_1h"] - cur["price_rolling_24h"]
        pred = predict_price(cur); hist.append(pred)
        conf = confidence(pred, cur.get("price_volatility",200), block)
        sig, action = signal(pred, hist[-2])
        log_pred(cur, pred, conf, state["version"], block)
        forecasts.append({"block":block,"datetime":dt.strftime("%Y-%m-%d %H:%M"),
                          "predicted_mcp":round(pred,2),"confidence":conf,
                          "signal":sig,"action":action})
    return forecasts

def business_metrics(forecasts):
    prices = [f["predicted_mcp"] for f in forecasts]
    TRADE_MW, BLK = 100, 0.25
    buys  = [f for f in forecasts if f["signal"]=="BUY"]
    sells = [f for f in forecasts if f["signal"]=="SELL"]
    savings = sum(max(max(prices)-p,0)*TRADE_MW*BLK for p in prices)
    return {
        "avg_mcp_24h":        round(np.mean(prices),2),
        "peak_mcp":           round(max(prices),2),
        "trough_mcp":         round(min(prices),2),
        "price_range":        round(max(prices)-min(prices),2),
        "estimated_savings":  round(savings,0),
        "trade_volume_mw":    TRADE_MW,
        "buy_windows":        len(buys),
        "sell_windows":       len(sells),
        "optimal_buy_time":   min(buys, key=lambda x:x["predicted_mcp"])["datetime"] if buys else "N/A",
        "optimal_sell_time":  max(sells,key=lambda x:x["predicted_mcp"])["datetime"] if sells else "N/A",
    }

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def home():
    mape_str = f"{state['mape']:.2f}%" if state["mape"] else "N/A"
    status   = "running" if state["model"] else "model_not_loaded"

    endpoints = [
        ("/health",             "System health + data freshness"),
        ("/predict/sample",     "Single prediction using live data"),
        ("/forecast/24h",       "96-block 24h forecast with confidence + trading signals"),
        ("/feature-importance", "Top 10 features driving model predictions"),
        ("/trading-simulation", "Historical P&L simulation from prediction log"),
        ("/data/latest",        "Live data status — IEX + weather + commodities"),
        ("/eda",                "EDA dashboard (HTML report)"),
        ("/monitoring",         "Drift detection + rolling MAPE"),
        ("/refresh",            "Trigger immediate live data refresh"),
        ("/retrain",            "Trigger model retraining in background"),
    ]
    rows = "".join(f"""
        <tr>
          <td><a href="{ep}" target="_blank">{ep}</a></td>
          <td>{desc}</td>
        </tr>""" for ep, desc in endpoints)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Group 05 — IEX Forecasting API</title>
  <style>
    body   {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }}
    h1     {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
    h2     {{ color: #16213e; margin-top: 30px; }}
    .badges {{ display: flex; gap: 10px; margin: 15px 0; flex-wrap: wrap; }}
    .badge  {{ background: #1a1a2e; color: white; padding: 6px 14px; border-radius: 20px; font-size: 14px; }}
    .badge.green {{ background: #27ae60; }}
    .badge.blue  {{ background: #2980b9; }}
    table  {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    th     {{ background: #1a1a2e; color: white; padding: 12px 16px; text-align: left; }}
    td     {{ padding: 12px 16px; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #f0f7ff; }}
    a      {{ color: #2980b9; font-weight: bold; text-decoration: none; font-family: monospace; font-size: 15px; }}
    a:hover {{ text-decoration: underline; }}
    .footer {{ margin-top: 30px; color: #888; font-size: 13px; text-align: center; }}
    .post  {{ background: #fff3cd; padding: 12px 16px; border-bottom: 1px solid #eee; }}
    .post td:first-child {{ color: #e67e22; font-weight: bold; font-family: monospace; font-size: 15px; }}
  </style>
</head>
<body>
  <h1>⚡ IEX RTM Electricity Price Forecasting</h1>
  <div class="badges">
    <span class="badge green">● {status.upper()}</span>
    <span class="badge blue">Model: {state['model_name']} {state['version']}</span>
    <span class="badge">MAPE: {mape_str}</span>
    <span class="badge">Group 05 — ISB AMPBA</span>
  </div>

  <h2>🔌 API Endpoints</h2>
  <table>
    <tr><th>Endpoint</th><th>Description</th></tr>
    {rows}
    <tr class="post">
      <td><a href="/predict" target="_blank" style="color:#e67e22">/predict</a></td>
      <td>Interactive prediction form — adjust inputs and get live price forecast</td>
    </tr>
  </table>

  <div class="footer">
    ISB AMPBA Foundation Project &nbsp;|&nbsp; Loaded at {state['loaded_at']}
  </div>
</body>
</html>"""
    return html

@app.route("/health")
def health():
    files = {"IEX":"iex_live.csv","Weather":"weather_live.csv","Commodities":"commodities_live.csv"}
    fresh = {n:{"age_minutes":file_age_minutes(os.path.join(DATA_DIR,f)),
                "status":freshness_label(file_age_minutes(os.path.join(DATA_DIR,f)))}
             for n,f in files.items()}
    return jsonify({
        "status":          "healthy" if state["model"] else "model_not_loaded",
        "model":           state["model_name"],
        "version":         state["version"],
        "mape":            f"{state['mape']:.2f}%" if state["mape"] else "N/A",
        "features":        len(state["feature_cols"]) if state["feature_cols"] else 0,
        "model_loaded_at": state["loaded_at"],
        "data_freshness":  fresh,
        "timestamp":       datetime.now().isoformat(),
    })

@app.route("/predict/sample")
def predict_sample():
    if state["model"] is None:
        return jsonify({"error":"Model not loaded"}), 503
    fresh    = ensure_fresh()
    features = load_live_features()
    pred     = predict_price(features)
    vol      = features.get("price_volatility", 200)
    conf     = confidence(pred, vol, 1)
    prev     = features.get("mcp_lag_1h", pred)
    sig, action = signal(pred, prev)
    log_pred(features, pred, conf, state["version"], 1)
    return jsonify({
        "predicted_mcp_rs_mwh": round(pred, 2),
        "confidence":           conf,
        "trading_signal":       sig,
        "action":               action,
        "model":                state["model_name"],
        "version":              state["version"],
        "mape":                 f"{state['mape']:.2f}%" if state["mape"] else "N/A",
        "data_freshness":       fresh,
        "timestamp":            datetime.now().isoformat(),
    })

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if state["model"] is None:
        return jsonify({"error":"Model not loaded"}), 503
    fresh    = ensure_fresh()
    features = load_live_features()
    result   = None
    if request.method == "POST":
        overrides = request.form if request.form else {}
        json_data = request.json if request.is_json else {}
        for k, v in {**overrides, **json_data}.items():
            try: features[k] = float(v)
            except: pass
        pred  = predict_price(features)
        vol   = features.get("price_volatility", 200)
        conf  = confidence(pred, vol, 1)
        prev  = features.get("mcp_lag_1h", pred)
        sig, action = signal(pred, prev)
        log_pred(features, pred, conf, state["version"], 1)
        if request.is_json:
            return jsonify({"predicted_mcp_rs_mwh": round(pred,2),
                "confidence": conf, "trading_signal": sig, "action": action})
        result = {"pred": round(pred,2), "conf": conf, "sig": sig, "action": action}
    f = features
    sig_color = {"BUY":"#27ae60","SELL":"#e74c3c","HOLD":"#f39c12"}.get(
        result["sig"] if result else "HOLD", "#888")
    result_html = ""
    if result:
        result_html = f"""
        <div class="result-box">
          <h2>Prediction Result</h2>
          <div class="result-grid">
            <div class="result-item">
              <div class="result-label">Predicted MCP</div>
              <div class="result-value">Rs {result['pred']:,.2f} /MWh</div>
            </div>
            <div class="result-item">
              <div class="result-label">Confidence</div>
              <div class="result-value">{result['conf']}</div>
            </div>
            <div class="result-item">
              <div class="result-label">Signal</div>
              <div class="result-value" style="color:{sig_color};font-size:2em">{result['sig']}</div>
            </div>
            <div class="result-item">
              <div class="result-label">Action</div>
              <div class="result-value" style="font-size:0.9em">{result['action']}</div>
            </div>
          </div>
        </div>"""
    html = f"""<!DOCTYPE html>
<html><head><title>IEX Price Predictor</title>
<style>
body{{font-family:Arial,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#f5f5f5}}
h1{{color:#1a1a2e;border-bottom:3px solid #e94560;padding-bottom:10px}}
h2{{color:#16213e}}
.form-grid{{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:20px 0}}
.form-group{{background:white;padding:15px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.08)}}
label{{display:block;font-size:13px;color:#555;margin-bottom:5px;font-weight:bold}}
input{{width:100%;padding:10px;border:1px solid #ddd;border-radius:6px;font-size:15px;box-sizing:border-box}}
.hint{{font-size:11px;color:#999;margin-top:3px}}
.btn{{background:#e94560;color:white;border:none;padding:14px 40px;font-size:16px;border-radius:8px;cursor:pointer;width:100%;margin-top:10px}}
.btn:hover{{background:#c0392b}}
.result-box{{background:#1a1a2e;color:white;border-radius:12px;padding:25px;margin:25px 0}}
.result-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:15px}}
.result-item{{background:rgba(255,255,255,0.1);padding:15px;border-radius:8px;text-align:center}}
.result-label{{font-size:12px;color:#aaa;margin-bottom:8px}}
.result-value{{font-size:1.6em;font-weight:bold}}
.back{{display:inline-block;margin-top:15px;color:#2980b9;text-decoration:none}}
.badge{{background:#27ae60;color:white;font-size:12px;padding:3px 10px;border-radius:10px;margin-left:8px}}
</style></head>
<body>
<h1>IEX RTM Price Predictor <span class="badge">LIVE</span></h1>
<p style="color:#666">Pre-filled with live market data. Adjust and click Predict.</p>
{result_html}
<form method="POST">
  <h2>Input Features</h2>
  <div class="form-grid">
    <div class="form-group">
      <label>Last Hour MCP (Rs/MWh)</label>
      <input type="number" name="mcp_lag_1h" value="{f.get('mcp_lag_1h',3500):.1f}" step="0.1">
      <div class="hint">Most recent 15-min clearing price</div>
    </div>
    <div class="form-group">
      <label>24h Ago MCP (Rs/MWh)</label>
      <input type="number" name="mcp_lag_24h" value="{f.get('mcp_lag_24h',3500):.1f}" step="0.1">
      <div class="hint">Same time slot yesterday</div>
    </div>
    <div class="form-group">
      <label>Hour of Day (0-23)</label>
      <input type="number" name="hour" value="{f.get('hour', datetime.now().hour)}" min="0" max="23">
      <div class="hint">Current hour</div>
    </div>
    <div class="form-group">
      <label>Day of Week (0=Mon, 6=Sun)</label>
      <input type="number" name="day_of_week" value="{f.get('day_of_week', datetime.now().weekday())}" min="0" max="6">
      <div class="hint">0=Monday, 6=Sunday</div>
    </div>
    <div class="form-group">
      <label>Temperature Delhi (C)</label>
      <input type="number" name="temp_delhi" value="{f.get('temp_delhi',28):.1f}" step="0.1">
      <div class="hint">Drives cooling demand</div>
    </div>
    <div class="form-group">
      <label>Crude Oil (USD/barrel)</label>
      <input type="number" name="crude_oil_usd" value="{f.get('crude_oil_usd',75):.2f}" step="0.1">
      <div class="hint">Brent crude price</div>
    </div>
    <div class="form-group">
      <label>Rolling 24h Avg MCP</label>
      <input type="number" name="price_rolling_24h" value="{f.get('price_rolling_24h',3500):.1f}" step="0.1">
      <div class="hint">24-hour rolling average</div>
    </div>
    <div class="form-group">
      <label>Price Volatility</label>
      <input type="number" name="price_volatility" value="{f.get('price_volatility',200):.1f}" step="0.1">
      <div class="hint">Std dev of recent prices</div>
    </div>
  </div>
  <button type="submit" class="btn">Predict Clearing Price</button>
</form>
<a class="back" href="/">Back to API Home</a>
</body></html>"""
    return html

@app.route("/forecast/24h")
def forecast_24h():
    if state["model"] is None:
        return jsonify({"error":"Model not loaded"}), 503
    fresh     = ensure_fresh()
    base      = load_live_features()
    forecasts = build_24h_forecast(base)
    biz       = business_metrics(forecasts)
    conf_cnt  = {"HIGH":0,"MEDIUM":0,"LOW":0}
    for f in forecasts: conf_cnt[f["confidence"]] += 1
    return jsonify({
        "generated_at":      datetime.now().isoformat(),
        "model":             state["model_name"],
        "version":           state["version"],
        "mape":              f"{state['mape']:.2f}%" if state["mape"] else "N/A",
        "data_freshness":    fresh,
        "horizon_blocks":    96,
        "horizon_hours":     24,
        "confidence_summary":conf_cnt,
        "business_metrics":  biz,
        "forecast":          forecasts,
    })

@app.route("/data/latest")
def data_latest():
    sources = {}
    for name, fname in {"IEX":"iex_live.csv","Weather":"weather_live.csv",
                        "Commodities":"commodities_live.csv"}.items():
        fpath = os.path.join(DATA_DIR,fname)
        if not os.path.exists(fpath):
            sources[name]={"status":"no_file"}; continue
        try:
            df     = pd.read_csv(fpath)
            age    = file_age_minutes(fpath)
            latest = df.iloc[-1].to_dict() if len(df)>0 else {}
            latest = {k:(None if isinstance(v,float) and np.isnan(v) else v) for k,v in latest.items()}
            sources[name]={"status":freshness_label(age),"age_minutes":age if age<9999 else "N/A",
                           "total_records":len(df),"latest":latest}
        except Exception as e:
            sources[name]={"status":"error","error":str(e)}
    return jsonify({"checked_at":datetime.now().isoformat(),"sources":sources})

@app.route("/feature-importance")
def feature_importance():
    fi_file = os.path.join(MODELS_DIR,"feature_importance.json")
    if not os.path.exists(fi_file):
        return jsonify({"error":"Not found — run run_pipeline.py first"}), 404
    with open(fi_file) as f: fi = json.load(f)
    top10 = dict(list(fi.items())[:10])
    return jsonify({
        "model":           state["model_name"],
        "version":         state["version"],
        "top_10_features": top10,
        "interpretation":  {k:("HIGH" if v>0.1 else "MEDIUM" if v>0.05 else "LOW") for k,v in top10.items()},
        "total_features":  len(fi),
    })

@app.route("/trading-simulation")
def trading_simulation():
    if not os.path.exists(PRED_LOG):
        return jsonify({"message":"No prediction history yet — call /forecast/24h first"})
    df = pd.read_csv(PRED_LOG,parse_dates=["timestamp"]).dropna(subset=["predicted_mcp"])
    if len(df)<2:
        return jsonify({"message":"Not enough predictions yet — call /forecast/24h first"})
    TRADE_MW, BLK = 100, 0.25
    trades, pnl = [], 0.0
    for i in range(1,len(df)):
        prev = df["predicted_mcp"].iloc[i-1]
        curr = df["predicted_mcp"].iloc[i]
        if   curr > prev*1.05: action = "BUY";  p = (curr-prev)*TRADE_MW*BLK
        elif curr < prev*0.95: action = "SELL"; p = (prev-curr)*TRADE_MW*BLK
        else:                  action = "HOLD"; p = 0.0
        pnl += p
        trades.append({"timestamp":str(df["timestamp"].iloc[i]),"action":action,"pnl_inr":round(p,2)})
    buy_n  = sum(1 for t in trades if t["action"]=="BUY")
    sell_n = sum(1 for t in trades if t["action"]=="SELL")
    return jsonify({
        "summary":{"total_predictions":len(df),"total_trades":len(trades),
                   "buy_signals":buy_n,"sell_signals":sell_n,"hold_signals":len(trades)-buy_n-sell_n,
                   "total_pnl_inr":round(pnl,2),"avg_pnl_per_trade":round(pnl/max(len(trades),1),2),
                   "trade_volume_mw":TRADE_MW},
        "recent_trades": trades[-20:],
    })

@app.route("/eda")
def eda():
    eda_path = os.path.join(STATIC_DIR,"eda_report.html")
    if os.path.exists(eda_path): return send_file(eda_path)
    return jsonify({"error":"EDA report not generated — run run_pipeline.py first"}), 404

@app.route("/monitoring")
def monitoring():
    try:
        from data_pipeline.monitor import run_all_checks
        return jsonify(run_all_checks())
    except Exception as e:
        files = {"IEX":"iex_live.csv","Weather":"weather_live.csv","Commodities":"commodities_live.csv"}
        freshness = {}
        for n, f in files.items():
            age = file_age_minutes(os.path.join(DATA_DIR, f))
            freshness[n] = {
                "age_minutes": age if age < 9999 else "N/A",
                "status":      freshness_label(age),
            }
        return jsonify({
            "checked_at":     datetime.now().isoformat(),
            "data_freshness": freshness,
            "note": f"Full monitoring unavailable: {e}",
        })

@app.route("/refresh")
def refresh_data():
    """Trigger immediate live data refresh"""
    def _refresh():
        try:
            from data_pipeline.scheduler import refresh_all
            refresh_all()
        except Exception as e:
            import subprocess
            for s in ["iex_scraper.py","scraper_iex.py"]:
                sp = os.path.join(_BASE_DIR,"data_pipeline",s)
                if os.path.exists(sp):
                    subprocess.run(["python",sp],timeout=120,
                                 capture_output=True,cwd=_BASE_DIR)
                    break
    threading.Thread(target=_refresh, daemon=True).start()
    return jsonify({
        "message": "Live data refresh triggered",
        "note":    "Check /health in 60s to see updated freshness"
    })

@app.route("/retrain", methods=["GET","POST"])
def retrain():
    def _run():
        import subprocess
        subprocess.run(["python", os.path.join(_BASE_DIR,"run_pipeline.py")])
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"message":"Retraining started","hint":"Monitor /health for update"})

# ── Startup ───────────────────────────────────────────────────

def startup():
    load_model()
    threading.Thread(target=model_watcher, daemon=True).start()

    # Start live data scheduler — refreshes IEX + weather + commodities every 30min
    try:
        from data_pipeline.scheduler import start_scheduler
        start_scheduler()
    except Exception as e:
        print(f"Scheduler error: {e}")
        # Fallback inline scheduler if module not found
        def _fallback_scheduler():
            import subprocess, time
            time.sleep(60)
            while True:
                try:
                    scripts = ["iex_scraper.py","scraper_iex.py"]
                    for s in scripts:
                        sp = os.path.join(_BASE_DIR,"data_pipeline",s)
                        if os.path.exists(sp):
                            subprocess.run(["python",sp],timeout=120,
                                         capture_output=True,cwd=_BASE_DIR)
                            break
                except: pass
                time.sleep(1800)
        threading.Thread(target=_fallback_scheduler, daemon=True).start()
        print("Fallback scheduler started ✅")

    # Generate EDA report
    try:
        from data_pipeline.eda_generator import generate_and_save
        generate_and_save(); print("EDA ready ✅")
    except Exception as e: print(f"EDA: {e}")

if __name__ == "__main__":
    startup()
    print(f"\nFlask running → http://localhost:5000")
    print("Endpoints: /, /health, /predict/sample, /forecast/24h,")
    print("           /data/latest, /feature-importance, /trading-simulation,")
    print("           /eda, /monitoring, POST /retrain\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
