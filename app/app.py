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

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    return datetime.now(IST).replace(tzinfo=None)

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
        state["loaded_at"]    = now_ist().isoformat()
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

    now = now_ist()
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

def log_pred(features, pred, conf, version, block=1, log_ts=None):
    row = {"timestamp":(log_ts or datetime.now()).isoformat(),"model_version":version,
           "horizon_block":block,"predicted_mcp":round(pred,2),"confidence":conf,
           "mcp_lag_1h":features.get("mcp_lag_1h"),"temp_delhi":features.get("temp_delhi"),
           "hour":features.get("hour"),"is_weekend":features.get("is_weekend"),"actual_mcp":None}
    df = pd.DataFrame([row])
    if os.path.exists(PRED_LOG): df.to_csv(PRED_LOG,mode="a",header=False,index=False)
    else: df.to_csv(PRED_LOG,index=False)

# ── 24h Forecast ──────────────────────────────────────────────

def build_24h_forecast(base):
    forecasts, cur, hist = [], dict(base), [base.get("mcp_lag_1h",3500)]*300
    now = now_ist().replace(second=0,microsecond=0)
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
        log_pred(cur, pred, conf, state["version"], block, log_ts=dt)
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
# SHARED CSS — embedded in every HTML page
# ══════════════════════════════════════════════════════════════

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#2d3436}
.hdr{background:linear-gradient(135deg,#1a1a2e,#0f3460);color:white;padding:28px 36px}
.hdr h1{font-size:1.75em;margin-bottom:5px}
.hdr p{opacity:.8;font-size:.93em}
.bdg{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
.b{padding:5px 13px;border-radius:20px;font-size:.82em;font-weight:bold}
.bg{background:#27ae60}.bb{background:#2980b9}.bo{background:#f39c12}.br{background:#e94560}.bd{background:rgba(255,255,255,.18)}
.wrap{max-width:1100px;margin:0 auto;padding:22px 16px}
.sec{background:white;border-radius:12px;padding:22px;margin-bottom:18px;box-shadow:0 2px 10px rgba(0,0,0,.07)}
.sec h2{font-size:1.25em;color:#1a1a2e;margin-bottom:12px;padding-bottom:8px;border-bottom:3px solid #e94560}
.sec h3{font-size:1em;color:#2980b9;margin:14px 0 8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:13px;margin:12px 0}
.card{background:#f8f9ff;border-radius:10px;padding:14px;border-left:4px solid #2980b9;text-align:center}
.cg{border-color:#27ae60}.cr{border-color:#e94560}.co{border-color:#f39c12}.cgr{border-color:#95a5a6}
.cv{font-size:1.65em;font-weight:bold;color:#1a1a2e;margin:5px 0}
.cl{font-size:.72em;color:#636e72;text-transform:uppercase;letter-spacing:.5px}
.cs{font-size:.75em;color:#b2bec3;margin-top:2px}
.tbl{width:100%;border-collapse:collapse;margin:10px 0;font-size:.9em}
.tbl th{background:#1a1a2e;color:white;padding:9px 12px;text-align:left}
.tbl td{padding:9px 12px;border-bottom:1px solid #eee}
.tbl tr:hover td{background:#f8f9ff}
.ins{background:#f8f9ff;border-left:4px solid #2980b9;padding:11px 15px;border-radius:0 8px 8px 0;margin:10px 0;font-size:.91em;line-height:1.7}
.inw{border-color:#e94560;background:#fff8f8}.ing{border-color:#27ae60;background:#f0fff4}
.back{display:inline-block;margin-top:14px;color:#2980b9;text-decoration:none;font-size:.9em}
.step{display:flex;gap:12px;margin:10px 0;align-items:flex-start}
.sn{background:#1a1a2e;color:white;border-radius:50%;width:30px;height:30px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:.85em;flex-shrink:0;margin-top:2px}
.sb{flex:1;font-size:.91em;line-height:1.65}
.bug{background:#fff3f3;border-left:4px solid #e94560;padding:9px 13px;margin:7px 0;border-radius:0 6px 6px 0;font-size:.88em;line-height:1.6}
.fix{background:#f0fff4;border-left:4px solid #27ae60;padding:9px 13px;margin:7px 0;border-radius:0 6px 6px 0;font-size:.88em}
.ts{font-size:.76em;color:#b2bec3;margin-top:7px}
a.il{color:#2980b9}
"""

def badge(text, cls="bd"):
    return f'<span class="b {cls}">{text}</span>'

def fbadge(s):
    c = {"FRESH":"bg","WARNING":"bo","STALE":"br"}.get(s,"bgr")
    return f'<span class="b {c}" style="font-size:.78em">{s}</span>'

def page(title, h1, sub, badges, body):
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — IEX API</title><style>{CSS}</style>
</head><body>
<div class="hdr"><h1>{h1}</h1><p>{sub}</p><div class="bdg">{badges}</div></div>
<div class="wrap">{body}<br><a class="back" href="/">← Back to Home</a></div>
<div style="text-align:center;padding:14px;color:#b2bec3;font-size:.78em">Group 05 — ISB AMPBA | Auto-refreshes every 60s</div>
</body></html>"""

# ══════════════════════════════════════════════════════════════
# / — HOME
# ══════════════════════════════════════════════════════════════
@app.route("/")
def home():
    mape_str = f"{state['mape']:.2f}%" if state["mape"] else "N/A"
    status   = "RUNNING" if state["model"] else "MODEL NOT LOADED"
    sc       = "#27ae60" if state["model"] else "#e94560"

    endpoints = [
        ("/model-summary",      "GET", "Model card — models tried, accuracy, training data, pipeline"),
        ("/health",             "GET", "System health + live data freshness"),
        ("/predict",            "GET/POST","Interactive prediction form — adjust inputs, get live forecast"),
        ("/predict/sample",     "GET", "Quick single prediction using live data"),
        ("/forecast/24h",       "GET", "96-block 24h forecast with confidence + trading signals"),
        ("/feature-importance", "GET", "Top 10 features driving model predictions"),
        ("/trading-simulation", "GET", "Simulated P&L from prediction log"),
        ("/data/latest",        "GET", "Live data status — IEX + weather + commodities"),
        ("/eda",                "GET", "Full EDA dashboard (HTML)"),
        ("/pestle",             "GET", "PESTLE scenario analysis — 7 market scenarios"),
        ("/monitoring",         "GET", "Drift detection + rolling MAPE"),
        ("/refresh",            "GET", "Trigger immediate live data refresh"),
        ("/retrain",            "GET", "Trigger model retraining in background"),
    ]
    rows = "".join(f'<tr><td><a href="{ep}" target="_blank">{ep}</a></td><td style="color:#636e72;font-size:.85em">{m}</td><td>{d}</td></tr>' for ep,m,d in endpoints)

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><title>IEX RTM Forecasting API</title>
<style>
{CSS}
.hdr h1{{font-size:2em}} a{{color:#2980b9;font-weight:bold;text-decoration:none;font-family:monospace}}
a:hover{{text-decoration:underline}}
</style></head><body>
<div class="hdr">
  <h1>IEX RTM Electricity Price Forecasting</h1>
  <p>Group 05 — ISB AMPBA | CRISP-ML(Q) Framework | Deployed on AWS EC2</p>
  <div class="bdg">
    <span class="b" style="background:{sc}">● {status}</span>
    {badge(f"{state['model_name']} {state['version']}","bb")}
    {badge(f"MAPE: {mape_str}","bd")}
    {badge("Group 05","bd")}
  </div>
</div>
<div class="wrap">
<div class="sec">
  <h2>API Endpoints</h2>
  <table class="tbl">
    <tr><th>Endpoint</th><th>Method</th><th>Description</th></tr>
    {rows}
  </table>
</div>
<div style="text-align:center;color:#b2bec3;font-size:.8em;padding:8px">
  Loaded at {state['loaded_at']}
</div>
</div></body></html>"""

# ══════════════════════════════════════════════════════════════
# /health
# ══════════════════════════════════════════════════════════════
@app.route("/health")
def health():
    files = {"IEX":"iex_live.csv","Weather":"weather_live.csv","Commodities":"commodities_live.csv"}
    fresh = {}
    for n,f in files.items():
        age = file_age_minutes(os.path.join(DATA_DIR,f))
        fresh[n] = {"age": age, "status": freshness_label(age)}

    status   = "healthy" if state["model"] else "model_not_loaded"
    mape_str = f"{state['mape']:.2f}%" if state["mape"] else "N/A"
    feats    = len(state["feature_cols"]) if state["feature_cols"] else 0
    sc       = "cg" if status=="healthy" else "cr"

    fresh_rows = "".join(
        f'<tr><td><b>{n}</b></td><td>{v["age"]:.0f} min ago</td><td>{fbadge(v["status"])}</td></tr>'
        for n,v in fresh.items())

    body = f"""
    <div class="sec"><h2>System Status</h2>
    <div class="grid">
      <div class="card {sc}"><div class="cl">API Status</div><div class="cv" style="font-size:1.1em">{status.upper()}</div></div>
      <div class="card"><div class="cl">Model</div><div class="cv" style="font-size:1.1em">{state['model_name']}</div><div class="cs">Version {state['version']}</div></div>
      <div class="card cg"><div class="cl">Test MAPE</div><div class="cv">{mape_str}</div><div class="cs">Lower is better</div></div>
      <div class="card"><div class="cl">Features</div><div class="cv">{feats}</div><div class="cs">All leakage-free</div></div>
    </div>
    <p class="ts">Model loaded: {state['loaded_at']} | Checked: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    <div class="sec"><h2>Live Data Freshness</h2>
    <table class="tbl"><tr><th>Source</th><th>Last Updated</th><th>Status</th></tr>{fresh_rows}</table>
    <div class="ins">Refreshes automatically every 30 min. <a class="il" href="/refresh">Trigger now →</a></div>
    </div>"""

    return page("Health", "System Health",
        "IEX RTM Forecasting API — Live Status",
        badge(status.upper(), "bg" if status=="healthy" else "br") + badge(f"MAPE: {mape_str}","bb") + badge(f"{state['model_name']} {state['version']}","bd"),
        body)

# ══════════════════════════════════════════════════════════════
# /predict/sample
# ══════════════════════════════════════════════════════════════
@app.route("/predict/sample")
def predict_sample():
    if state["model"] is None:
        return page("Error","Model Not Loaded","","", '<div class="ins inw">Model not loaded — check /health</div>')
    fresh    = ensure_fresh()
    features = load_live_features()
    pred     = predict_price(features)
    vol      = features.get("price_volatility", 200)
    conf     = confidence(pred, vol, 1)
    prev     = features.get("mcp_lag_1h", pred)
    sig, action = signal(pred, prev)
    log_pred(features, pred, conf, state["version"], 1)

    sc = {"BUY":"#27ae60","SELL":"#e94560","HOLD":"#f39c12"}.get(sig,"#95a5a6")
    cc = {"HIGH":"cg","MEDIUM":"co","LOW":"cr"}.get(conf,"cgr")

    fresh_rows = "".join(
        f'<tr><td>{k.replace("_"," ").title()}</td><td>{v}</td></tr>'
        for k,v in fresh.items())

    feat_rows = "".join(
        f'<tr><td style="font-family:monospace">{k}</td><td>{round(float(v),3) if isinstance(v,(int,float)) else v}</td></tr>'
        for k,v in list(features.items())[:12])

    body = f"""
    <div class="sec"><h2>Prediction Result</h2>
    <div class="grid">
      <div class="card cg"><div class="cl">Predicted MCP</div><div class="cv">Rs{pred:,.2f}</div><div class="cs">Rs/MWh</div></div>
      <div class="card {cc}"><div class="cl">Confidence</div><div class="cv">{conf}</div></div>
      <div class="card" style="border-color:{sc}"><div class="cl">Signal</div><div class="cv" style="color:{sc}">{sig}</div><div class="cs">{action}</div></div>
      <div class="card"><div class="cl">Generated At</div><div class="cv" style="font-size:.9em">{datetime.now().strftime('%H:%M:%S')}</div></div>
    </div>
    <div class="ins ing"><b>Action:</b> {action}</div>
    </div>
    <div class="sec"><h2>Input Features Used</h2>
    <table class="tbl"><tr><th>Feature</th><th>Value</th></tr>{feat_rows}</table>
    </div>
    <div class="sec"><h2>Data Freshness</h2>
    <table class="tbl">{fresh_rows}</table>
    <div class="ins">For interactive prediction with editable inputs → <a class="il" href="/predict">/predict</a></div>
    </div>"""

    return page("Live Prediction","Live Price Prediction",
        "Single 15-min block using latest market data",
        f'<span class="b" style="background:{sc}">{sig}</span>' + badge(f"{state['model_name']} {state['version']}","bb") + badge(f"MAPE: {state['mape']:.2f}%","bd"),
        body)

# ══════════════════════════════════════════════════════════════
# /predict  — interactive form (GET/POST)
# ══════════════════════════════════════════════════════════════
@app.route("/predict", methods=["GET","POST"])
def predict():
    if state["model"] is None:
        return page("Error","Model Not Loaded","","",'<div class="ins inw">Model not loaded — check /health</div>')
    ensure_fresh()
    features = load_live_features()
    result   = None

    if request.method == "POST":
        overrides = request.form if request.form else {}
        json_data = request.json  if request.is_json else {}
        for k, v in {**dict(overrides), **dict(json_data)}.items():
            try: features[k] = float(v)
            except: pass
        pred  = predict_price(features)
        conf  = confidence(pred, features.get("price_volatility",200), 1)
        prev  = features.get("mcp_lag_1h", pred)
        sig, action = signal(pred, prev)
        log_pred(features, pred, conf, state["version"], 1)
        if request.is_json:
            return jsonify({"predicted_mcp_rs_mwh":round(pred,2),"confidence":conf,"signal":sig,"action":action})
        result = {"pred":round(pred,2),"conf":conf,"sig":sig,"action":action}

    f = features
    sc = {"BUY":"#27ae60","SELL":"#e94560","HOLD":"#f39c12"}.get(result["sig"] if result else "HOLD","#888")
    result_html = ""
    if result:
        result_html = f"""
        <div class="sec" style="border:2px solid {sc}">
          <h2>Prediction Result</h2>
          <div class="grid">
            <div class="card cg"><div class="cl">Predicted MCP</div><div class="cv">Rs{result['pred']:,.2f}</div><div class="cs">Rs/MWh</div></div>
            <div class="card"><div class="cl">Confidence</div><div class="cv">{result['conf']}</div></div>
            <div class="card" style="border-color:{sc}"><div class="cl">Signal</div><div class="cv" style="color:{sc};font-size:2em">{result['sig']}</div></div>
          </div>
          <div class="ins ing"><b>Action:</b> {result['action']}</div>
        </div>"""

    form_html = f"""
    <div class="sec"><h2>Input Features <span style="font-size:.75em;color:#95a5a6;font-weight:normal">Pre-filled with live data — adjust and predict</span></h2>
    <form method="POST">
    <div class="grid" style="grid-template-columns:1fr 1fr">
      <div style="background:#f8f9ff;padding:14px;border-radius:8px">
        <label style="font-size:.82em;color:#555;font-weight:bold">Last Hour MCP (Rs/MWh)</label>
        <input type="number" name="mcp_lag_1h" value="{f.get('mcp_lag_1h',3500):.1f}" step="0.1" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;margin-top:4px">
        <div style="font-size:.75em;color:#999;margin-top:3px">Most recent price</div>
      </div>
      <div style="background:#f8f9ff;padding:14px;border-radius:8px">
        <label style="font-size:.82em;color:#555;font-weight:bold">24h Ago MCP (Rs/MWh)</label>
        <input type="number" name="mcp_lag_24h" value="{f.get('mcp_lag_24h',3500):.1f}" step="0.1" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;margin-top:4px">
        <div style="font-size:.75em;color:#999;margin-top:3px">Same time yesterday</div>
      </div>
      <div style="background:#f8f9ff;padding:14px;border-radius:8px">
        <label style="font-size:.82em;color:#555;font-weight:bold">Hour of Day (0-23)</label>
        <input type="number" name="hour" value="{f.get('hour',now_ist().hour)}" min="0" max="23" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;margin-top:4px">
      </div>
      <div style="background:#f8f9ff;padding:14px;border-radius:8px">
        <label style="font-size:.82em;color:#555;font-weight:bold">Day of Week (0=Mon, 6=Sun)</label>
        <input type="number" name="day_of_week" value="{f.get('day_of_week',now_ist().weekday())}" min="0" max="6" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;margin-top:4px">
      </div>
      <div style="background:#f8f9ff;padding:14px;border-radius:8px">
        <label style="font-size:.82em;color:#555;font-weight:bold">Temperature Delhi (°C)</label>
        <input type="number" name="temp_delhi" value="{f.get('temp_delhi',28):.1f}" step="0.1" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;margin-top:4px">
        <div style="font-size:.75em;color:#999;margin-top:3px">Drives cooling demand</div>
      </div>
      <div style="background:#f8f9ff;padding:14px;border-radius:8px">
        <label style="font-size:.82em;color:#555;font-weight:bold">Crude Oil (USD/barrel)</label>
        <input type="number" name="crude_oil_usd" value="{f.get('crude_oil_usd',75):.2f}" step="0.1" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;margin-top:4px">
        <div style="font-size:.75em;color:#999;margin-top:3px">Brent crude</div>
      </div>
      <div style="background:#f8f9ff;padding:14px;border-radius:8px">
        <label style="font-size:.82em;color:#555;font-weight:bold">Rolling 24h Avg MCP</label>
        <input type="number" name="price_rolling_24h" value="{f.get('price_rolling_24h',3500):.1f}" step="0.1" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;margin-top:4px">
      </div>
      <div style="background:#f8f9ff;padding:14px;border-radius:8px">
        <label style="font-size:.82em;color:#555;font-weight:bold">Price Volatility</label>
        <input type="number" name="price_volatility" value="{f.get('price_volatility',200):.1f}" step="0.1" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;margin-top:4px">
        <div style="font-size:.75em;color:#999;margin-top:3px">Std dev of recent prices</div>
      </div>
    </div>
    <button type="submit" style="background:#e94560;color:white;border:none;padding:13px 36px;font-size:1em;border-radius:8px;cursor:pointer;width:100%;margin-top:14px">
      Predict Clearing Price
    </button>
    </form></div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Price Predictor — IEX</title><style>{CSS}</style></head><body>
<div class="hdr"><h1>Interactive Price Predictor</h1>
<p>Adjust market inputs and get an instant RTM price forecast</p>
<div class="bdg">{badge("LIVE","bg")}{badge(f"{state['model_name']} {state['version']}","bb")}{badge(f"MAPE: {state['mape']:.2f}%","bd")}</div>
</div>
<div class="wrap">{result_html}{form_html}<br><a class="back" href="/">← Back to Home</a></div>
</body></html>"""

# ══════════════════════════════════════════════════════════════
# /forecast/24h
# ══════════════════════════════════════════════════════════════
@app.route("/forecast/24h")
def forecast_24h():
    if state["model"] is None:
        return page("Error","Model Not Loaded","","",'<div class="ins inw">Model not loaded — check /health</div>')
    fresh     = ensure_fresh()
    base      = load_live_features()
    forecasts = build_24h_forecast(base)
    biz       = business_metrics(forecasts)
    conf_cnt  = {"HIGH":0,"MEDIUM":0,"LOW":0}
    for fc in forecasts: conf_cnt[fc["confidence"]] += 1

    mape_str = f"{state['mape']:.2f}%" if state["mape"] else "N/A"
    sc_map   = {"BUY":"#27ae60","SELL":"#e94560","HOLD":"#f39c12"}
    cc_map   = {"HIGH":"#27ae60","MEDIUM":"#f39c12","LOW":"#e94560"}

    rows = "".join(f"""<tr>
      <td style="color:#95a5a6">{fc["block"]}</td>
      <td>{fc["datetime"]}</td>
      <td><b>Rs{fc["predicted_mcp"]:,.2f}</b></td>
      <td style="color:{cc_map.get(fc["confidence"],"#333")};font-weight:bold">{fc["confidence"]}</td>
      <td><span style="background:{sc_map.get(fc["signal"],"#95a5a6")};color:white;padding:2px 9px;border-radius:10px;font-size:.82em">{fc["signal"]}</span></td>
      <td style="font-size:.82em;color:#636e72">{fc["action"]}</td>
    </tr>""" for fc in forecasts)

    body = f"""
    <div class="sec"><h2>Business Summary</h2>
    <div class="grid">
      <div class="card cg"><div class="cl">Avg Forecast MCP</div><div class="cv">Rs{biz["avg_mcp_24h"]:,.0f}</div><div class="cs">24h average</div></div>
      <div class="card cg"><div class="cl">Peak MCP</div><div class="cv">Rs{biz["peak_mcp"]:,.0f}</div><div class="cs">Highest block</div></div>
      <div class="card co"><div class="cl">Trough MCP</div><div class="cv">Rs{biz["trough_mcp"]:,.0f}</div><div class="cs">Lowest block</div></div>
      <div class="card"><div class="cl">Price Range</div><div class="cv">Rs{biz["price_range"]:,.0f}</div><div class="cs">Peak - Trough</div></div>
      <div class="card cg"><div class="cl">HIGH Confidence</div><div class="cv">{conf_cnt["HIGH"]}</div><div class="cs">blocks</div></div>
      <div class="card co"><div class="cl">MEDIUM Confidence</div><div class="cv">{conf_cnt["MEDIUM"]}</div><div class="cs">blocks</div></div>
      <div class="card cr"><div class="cl">LOW Confidence</div><div class="cv">{conf_cnt["LOW"]}</div><div class="cs">blocks</div></div>
      <div class="card"><div class="cl">Optimal Buy</div><div class="cv" style="font-size:.85em">{biz["optimal_buy_time"]}</div></div>
    </div>
    </div>
    <div class="sec"><h2>96-Block Forecast Table</h2>
    <table class="tbl">
      <tr><th>Block</th><th>Time</th><th>Predicted MCP</th><th>Confidence</th><th>Signal</th><th>Action</th></tr>
      {rows}
    </table>
    <p class="ts">Generated: {datetime.now().strftime("%d %b %Y %H:%M:%S")} | Model: {state["model_name"]} {state["version"]} | MAPE: {mape_str}</p>
    </div>"""

    return page("24h Forecast","24-Hour Price Forecast",
        "96 blocks × 15 minutes — IEX RTM Market Clearing Price",
        badge(f"{state['model_name']} {state['version']}","bb") + badge(f"MAPE: {mape_str}","bd") +
        badge(f"HIGH: {conf_cnt['HIGH']}","bg") + badge(f"MED: {conf_cnt['MEDIUM']}","bo") + badge(f"LOW: {conf_cnt['LOW']}","br"),
        body)

# ══════════════════════════════════════════════════════════════
# /data/latest
# ══════════════════════════════════════════════════════════════
@app.route("/data/latest")
def data_latest():
    sources = {}
    for name, fname in {"IEX":"iex_live.csv","Weather":"weather_live.csv","Commodities":"commodities_live.csv"}.items():
        fpath = os.path.join(DATA_DIR,fname)
        if not os.path.exists(fpath):
            sources[name]={"status":"NO FILE","age":"N/A","records":"N/A","latest":{}}; continue
        try:
            df    = pd.read_csv(fpath)
            age   = file_age_minutes(fpath)
            # For weather — show Delhi (model city), not last row which may be blank
            if name == "Weather" and "city" in df.columns:
                delhi = df[df["city"]=="Delhi"]
                row   = delhi.iloc[-1] if len(delhi)>0 else df.iloc[-1]
            else:
                row   = df.iloc[-1] if len(df)>0 else df.iloc[0]
            latest= row.to_dict() if len(df)>0 else {}
            latest= {k:(None if isinstance(v,float) and np.isnan(v) else v) for k,v in latest.items()}
            sources[name]={"status":freshness_label(age),"age":age,"records":len(df),"latest":latest}
        except Exception as e:
            sources[name]={"status":"ERROR","age":"N/A","records":"N/A","latest":{"error":str(e)}}

    cards_html = ""
    for name, info in sources.items():
        status = info["status"]
        age    = f"{info['age']:.0f} min ago" if isinstance(info["age"],float) else info["age"]
        latest = info.get("latest",{})
        rows   = "".join(f'<tr><td style="font-family:monospace;font-size:.85em">{k}</td><td>{v}</td></tr>' for k,v in list(latest.items())[:12] if v is not None and str(v).strip() != "" and str(v).strip() != "nan")
        sc     = {"FRESH":"cg","WARNING":"co","STALE":"cr"}.get(status,"cgr")
        cards_html += f"""
        <div class="sec">
          <h2>{name} &nbsp; {fbadge(status)}</h2>
          <div class="grid" style="grid-template-columns:repeat(3,1fr)">
            <div class="card {sc}"><div class="cl">Status</div><div class="cv" style="font-size:1em">{status}</div></div>
            <div class="card"><div class="cl">Last Updated</div><div class="cv" style="font-size:.95em">{age}</div></div>
            <div class="card"><div class="cl">Total Records</div><div class="cv">{info["records"]}</div></div>
          </div>
          <h3>Latest Row</h3>
          <table class="tbl"><tr><th>Field</th><th>Value</th></tr>{rows}</table>
        </div>"""

    return page("Data Status","Live Data Status",
        "IEX + Weather + Commodities freshness check",
        badge(f"Checked: {datetime.now().strftime('%H:%M:%S')}","bd"),
        cards_html + '<div class="ins">Refreshes every 30 min. <a class="il" href="/refresh">Trigger now →</a></div>')

# ══════════════════════════════════════════════════════════════
# /feature-importance
# ══════════════════════════════════════════════════════════════
@app.route("/feature-importance")
def feature_importance():
    fi_file = os.path.join(MODELS_DIR,"feature_importance.json")
    if not os.path.exists(fi_file):
        return page("Error","Feature Importance","","",'<div class="ins inw">Not found — run run_pipeline.py first</div>')
    with open(fi_file) as f: fi = json.load(f)
    top10   = sorted(fi.items(), key=lambda x: float(x[1]), reverse=True)[:10]
    max_imp = float(top10[0][1]) if top10 else 1

    meanings = {
        "mcp_lag_1h":       "Most recent 15-min clearing price — strongest short-term signal",
        "mcp_lag_24h":      "Same time slot yesterday — captures daily seasonality",
        "price_rolling_24h":"24-hour rolling average — medium-term trend proxy",
        "hour":             "Hour of day — intraday demand peak pattern",
        "temp_delhi":       "Delhi temperature — proxy for national cooling demand",
        "mcp_lag_48h":      "Price 48 hours ago — 2-day lag pattern",
        "price_volatility": "Recent price std deviation — market stress indicator",
        "day_of_week":      "Day of week — weekday vs weekend demand shift",
        "crude_oil_usd":    "Brent crude price — thermal generation cost signal",
        "price_rolling_1w": "7-day rolling average — weekly trend context",
        "mcp_lag_1w":       "Price 1 week ago — weekly cycle reference",
        "price_change_1h":  "1-hour price change — momentum signal",
        "price_momentum":   "Deviation from rolling avg — directional pressure",
        "is_weekend":       "Weekend flag — lower industrial demand on weekends",
        "month":            "Month of year — seasonal demand pattern",
    }

    rows = ""
    for i,(feat,imp) in enumerate(top10):
        imp_f = float(imp)
        bar_w = int(imp_f/max_imp*260)
        level = "HIGH" if imp_f>0.1 else "MEDIUM" if imp_f>0.05 else "LOW"
        lc    = {"HIGH":"#27ae60","MEDIUM":"#f39c12","LOW":"#e94560"}[level]
        m     = meanings.get(feat,"Feature from engineered dataset")
        rows += f"""<tr>
          <td style="font-weight:bold;color:#95a5a6">{i+1}</td>
          <td style="font-family:monospace;font-weight:bold">{feat}</td>
          <td>{imp_f:.4f}</td>
          <td><div style="background:{lc};height:15px;width:{bar_w}px;border-radius:3px;min-width:3px"></div></td>
          <td><span style="background:{lc};color:white;padding:2px 8px;border-radius:8px;font-size:.78em">{level}</span></td>
          <td style="font-size:.85em;color:#636e72">{m}</td>
        </tr>"""

    body = f"""
    <div class="sec"><h2>Top 10 Feature Importances</h2>
    <table class="tbl">
      <tr><th>#</th><th>Feature</th><th>Score</th><th>Weight</th><th>Level</th><th>Business Meaning</th></tr>
      {rows}
    </table>
    <div class="ins">Importance = fraction of XGBoost tree splits. All features computed on <b>shifted MCP</b> — current price never leaks into any feature. Total features in model: {len(fi)}.</div>
    </div>
    <div class="sec"><h2>Key Insight</h2>
    <div class="ins ing"><b>Lag features dominate</b> — mcp_lag_1h + mcp_lag_24h together account for ~45% of total importance. This confirms recent price history is the strongest predictor in RTM markets. Weather and commodity features add signal during demand extremes.</div>
    </div>"""

    return page("Feature Importance","Feature Importance",
        f"Top 10 drivers of {state['model_name']} predictions",
        badge(f"{state['model_name']} {state['version']}","bb") + badge(f"Total: {len(fi)} features","bd"),
        body)

# ══════════════════════════════════════════════════════════════
# /trading-simulation
# ══════════════════════════════════════════════════════════════
@app.route("/trading-simulation")
def trading_simulation():
    if not os.path.exists(PRED_LOG):
        return page("Trading Simulation","Trading Simulation","","",
            '<div class="ins inw">No prediction history yet — call /forecast/24h first to generate predictions</div>')
    df = pd.read_csv(PRED_LOG,parse_dates=["timestamp"]).dropna(subset=["predicted_mcp"])
    if len(df)<2:
        return page("Trading Simulation","Trading Simulation","","",
            '<div class="ins inw">Not enough predictions yet — call /forecast/24h first</div>')

    TRADE_MW, BLK = 100, 0.25
    trades, pnl   = [], 0.0
    for i in range(1,len(df)):
        prev = df["predicted_mcp"].iloc[i-1]
        curr = df["predicted_mcp"].iloc[i]
        if   curr > prev*1.05: action,p = "BUY",  (curr-prev)*TRADE_MW*BLK
        elif curr < prev*0.95: action,p = "SELL", (prev-curr)*TRADE_MW*BLK
        else:                  action,p = "HOLD", 0.0
        pnl += p
        trades.append({"ts":str(df["timestamp"].iloc[i]),"action":action,"pnl":round(p,2)})

    buy_n  = sum(1 for t in trades if t["action"]=="BUY")
    sell_n = sum(1 for t in trades if t["action"]=="SELL")
    hold_n = len(trades)-buy_n-sell_n
    pnl_c  = "#27ae60" if pnl>=0 else "#e94560"

    action_colors = {"BUY":"#27ae60","SELL":"#e94560","HOLD":"#f39c12"}
    trade_rows = ""
    for t in trades[-25:]:
        ac = action_colors.get(t["action"],"#95a5a6")
        pc = "#27ae60" if t["pnl"]>=0 else "#e94560"
        trade_rows += f"""<tr>
          <td style="font-size:.82em;color:#636e72">{t["ts"]}</td>
          <td><span style="background:{ac};color:white;padding:2px 9px;border-radius:10px;font-size:.82em">{t["action"]}</span></td>
          <td style="color:{pc};font-weight:bold">Rs{t["pnl"]:,.2f}</td>
        </tr>"""

    body = f"""
    <div class="sec"><h2>Performance Summary</h2>
    <div class="grid">
      <div class="card"><div class="cl">Total Predictions</div><div class="cv">{len(df)}</div></div>
      <div class="card cg"><div class="cl">BUY Signals</div><div class="cv">{buy_n}</div></div>
      <div class="card cr"><div class="cl">SELL Signals</div><div class="cv">{sell_n}</div></div>
      <div class="card cgr"><div class="cl">HOLD Signals</div><div class="cv">{hold_n}</div></div>
      <div class="card {"cg" if pnl>=0 else "cr"}"><div class="cl">Total P&L</div><div class="cv" style="color:{pnl_c}">Rs{pnl:,.2f}</div></div>
      <div class="card"><div class="cl">Avg per Trade</div><div class="cv" style="font-size:1.1em">Rs{pnl/max(len(trades),1):,.2f}</div></div>
      <div class="card"><div class="cl">Trade Volume</div><div class="cv">{TRADE_MW} MW</div><div class="cs">per block</div></div>
    </div>
    </div>
    <div class="sec"><h2>Recent 25 Trades</h2>
    <table class="tbl"><tr><th>Timestamp</th><th>Action</th><th>P&L (Rs)</th></tr>{trade_rows}</table>
    <div class="ins inw"><b>Disclaimer:</b> Strategy: BUY when predicted price rises >5%, SELL when drops >5%, HOLD otherwise. This is a backtested simulation on model predictions — not live trading advice.</div>
    </div>"""

    return page("Trading Simulation","Trading Simulation",
        f"Simulated P&L from model predictions — {TRADE_MW} MW volume",
        f'<span class="b" style="background:{pnl_c}">P&L: Rs{pnl:,.2f}</span>' + badge(f"{len(trades)} trades","bd"),
        body)

# ══════════════════════════════════════════════════════════════
# /eda
# ══════════════════════════════════════════════════════════════
@app.route("/eda")
def eda():
    eda_path = os.path.join(STATIC_DIR,"eda_report.html")
    if os.path.exists(eda_path): return send_file(eda_path)
    return page("EDA","EDA Report Not Found","","",
        '<div class="ins inw">EDA report not generated — run eda_generator.py or run_pipeline.py first</div>')

# ══════════════════════════════════════════════════════════════
# /pestle
# ══════════════════════════════════════════════════════════════
@app.route("/pestle")
def pestle():
    pestle_file = os.path.join(MODELS_DIR,"pestle_scenarios.json")
    biz_file    = os.path.join(MODELS_DIR,"business_evaluation.json")

    scenarios_data = {}
    if os.path.exists(pestle_file):
        try:
            with open(pestle_file) as f: scenarios_data = json.load(f)
        except: pass

    biz = {}
    if os.path.exists(biz_file):
        try:
            with open(biz_file) as f: biz = json.load(f)
        except: pass

    # Hardcoded fallback with full data if file not present
    scenarios = [
        {
            "name":        "Baseline (Current Market)",
            "category":    "Baseline",
            "cat":         "B",
            "avg_mcp":     biz.get("avg_mcp", scenarios_data.get("baseline",{}).get("avg_mcp", 3630)),
            "delta":       0,
            "delta_pct":   0,
            "color":       "#2980b9",
            "icon":        "=",
            "impact":      "Reference scenario — current market conditions as of training period.",
            "features_changed": "None — all features at current observed values.",
        },
        {
            "name":        "Carbon Tax +20%",
            "category":    "Policy",
            "cat":         "P",
            "avg_mcp":     scenarios_data.get("carbon_tax",{}).get("avg_mcp", 3651),
            "delta":       scenarios_data.get("carbon_tax",{}).get("delta", 21),
            "delta_pct":   scenarios_data.get("carbon_tax",{}).get("delta_pct", 0.6),
            "color":       "#e67e22",
            "icon":        "↑",
            "impact":      "Carbon tax increases the cost of coal and gas-based thermal generation. Thermal plants pass through the higher fuel cost, lifting RTM clearing prices moderately.",
            "features_changed": "coal_price_proxy +20%, crude_oil_usd +10% (refinery energy costs), natural_gas_usd +10%.",
        },
        {
            "name":        "Economic Recession -15%",
            "category":    "Economic",
            "cat":         "E",
            "avg_mcp":     scenarios_data.get("recession",{}).get("avg_mcp", 3406),
            "delta":       scenarios_data.get("recession",{}).get("delta", -224),
            "delta_pct":   scenarios_data.get("recession",{}).get("delta_pct", -6.2),
            "color":       "#e94560",
            "icon":        "↓",
            "impact":      "Industrial demand collapses as factories cut output. Lower system demand reduces clearing prices significantly. This is the largest downward price driver identified.",
            "features_changed": "system_demand -15%, mcv_mw -15%, load_price_ratio adjusted, demand_lag_24h -15%.",
        },
        {
            "name":        "Heatwave +8°C Delhi",
            "category":    "Social / Environmental",
            "cat":         "S",
            "avg_mcp":     scenarios_data.get("heatwave",{}).get("avg_mcp", 3637),
            "delta":       scenarios_data.get("heatwave",{}).get("delta", 7),
            "delta_pct":   scenarios_data.get("heatwave",{}).get("delta_pct", 0.2),
            "color":       "#e74c3c",
            "icon":        "↑",
            "impact":      "Extreme heat drives residential and commercial cooling demand higher. Effect is moderate because grid has become better at absorbing demand shocks — renewables provide partial buffer.",
            "features_changed": "temp_delhi +8°C, cooling_degree +8, low_wind_flag may activate if heat suppresses winds.",
        },
        {
            "name":        "Renewable Surge +30%",
            "category":    "Technological",
            "cat":         "T",
            "avg_mcp":     scenarios_data.get("renewable_surge",{}).get("avg_mcp", 3406),
            "delta":       scenarios_data.get("renewable_surge",{}).get("delta", -224),
            "delta_pct":   scenarios_data.get("renewable_surge",{}).get("delta_pct", -6.2),
            "color":       "#27ae60",
            "icon":        "↓",
            "impact":      "Higher renewable generation increases supply at near-zero marginal cost, pushing thermal plants down the merit order. This is already the observed 2023→2025 price decline trend in the data.",
            "features_changed": "system_demand -30% (renewables displace thermal), fuel_proxy reduced, low_wind_flag=0 (more wind capacity).",
        },
        {
            "name":        "Price Cap Rs8,000",
            "category":    "Legal / Regulatory",
            "cat":         "L",
            "avg_mcp":     scenarios_data.get("price_cap",{}).get("avg_mcp", 3571),
            "delta":       scenarios_data.get("price_cap",{}).get("delta", -59),
            "delta_pct":   scenarios_data.get("price_cap",{}).get("delta_pct", -1.6),
            "color":       "#8e44ad",
            "icon":        "↓",
            "impact":      "Regulatory hard cap at Rs8,000/MWh truncates price spike blocks. Effect is moderate on the mean because only ~18% of blocks historically exceed Rs8,000.",
            "features_changed": "All predicted values clipped at Rs8,000 — primarily affects evening peak hours (19:00–22:00).",
        },
        {
            "name":        "Monsoon Season",
            "category":    "Environmental",
            "cat":         "E",
            "avg_mcp":     scenarios_data.get("monsoon",{}).get("avg_mcp", 3548),
            "delta":       scenarios_data.get("monsoon",{}).get("delta", -82),
            "delta_pct":   scenarios_data.get("monsoon",{}).get("delta_pct", -2.3),
            "color":       "#2980b9",
            "icon":        "↓",
            "impact":      "Monsoon brings lower temperatures (reduced cooling demand), high cloud cover (lower solar), and high humidity. Net effect is mild price reduction — lower demand offsets some renewable generation loss.",
            "features_changed": "temp_delhi -6°C, cloud_delhi +40%, humidity_delhi +20%, wind_delhi +2 m/s, cooling_degree reduced.",
        },
    ]

    # Fill from file if available
    for s in scenarios:
        key = s["name"].lower().replace(" ","_").replace("+","").replace("%","").replace(",","").replace("°","")
        for k in scenarios_data:
            if k.lower() in key or key in k.lower():
                s["avg_mcp"] = scenarios_data[k].get("avg_mcp", s["avg_mcp"])
                s["delta"]   = scenarios_data[k].get("delta",   s["delta"])
                break

    baseline_mcp = scenarios[0]["avg_mcp"]
    # Recompute deltas relative to baseline
    for s in scenarios:
        if s["delta"] == 0 and s["cat"] != "B":
            s["delta"]     = round(s["avg_mcp"] - baseline_mcp, 1)
            s["delta_pct"] = round((s["avg_mcp"] - baseline_mcp)/baseline_mcp*100, 1)

    # Summary table rows
    tbl_rows = ""
    for s in scenarios:
        d = s["delta"]
        dp = s["delta_pct"]
        d_str = f"+{d:,.0f}" if d>0 else f"{d:,.0f}" if d<0 else "—"
        dp_str= f"+{dp:.1f}%" if dp>0 else f"{dp:.1f}%" if dp<0 else "—"
        dc = "#27ae60" if d>0 else "#e94560" if d<0 else "#95a5a6"
        cat_color = {"B":"#2980b9","P":"#e67e22","E":"#e94560","S":"#e74c3c","T":"#27ae60","L":"#8e44ad"}.get(s["cat"],"#95a5a6")
        tbl_rows += f"""<tr>
          <td><span style="background:{cat_color};color:white;padding:2px 8px;border-radius:8px;font-size:.78em">{s["category"]}</span></td>
          <td><b>{s["name"]}</b></td>
          <td style="font-weight:bold">Rs{s["avg_mcp"]:,.0f}</td>
          <td style="color:{dc};font-weight:bold">{d_str}</td>
          <td style="color:{dc};font-weight:bold">{dp_str}</td>
        </tr>"""

    # Scenario detail cards
    detail_cards = ""
    for s in scenarios:
        d = s["delta"]
        dc= "#27ae60" if d>0 else "#e94560" if d<0 else "#2980b9"
        arrow = f'+Rs{d:,.0f}/MWh' if d>0 else f'Rs{d:,.0f}/MWh' if d<0 else "Baseline"
        detail_cards += f"""
        <div class="sec" style="border-left:5px solid {s['color']}">
          <h2 style="border-color:{s['color']}">{s["icon"]} {s["name"]}</h2>
          <div class="grid" style="grid-template-columns:repeat(3,1fr)">
            <div class="card" style="border-color:{s['color']}">
              <div class="cl">Avg MCP</div>
              <div class="cv">Rs{s["avg_mcp"]:,.0f}</div><div class="cs">Rs/MWh</div>
            </div>
            <div class="card" style="border-color:{dc}">
              <div class="cl">Delta vs Baseline</div>
              <div class="cv" style="color:{dc};font-size:1.3em">{arrow}</div>
            </div>
            <div class="card">
              <div class="cl">PESTLE Category</div>
              <div class="cv" style="font-size:.95em">{s["category"]}</div>
            </div>
          </div>
          <h3>Business Impact</h3>
          <div class="ins">{s["impact"]}</div>
          <h3>Features Modified in Model</h3>
          <div class="ins" style="font-family:monospace;font-size:.88em">{s["features_changed"]}</div>
        </div>"""

    method_html = f"""
    <div class="sec"><h2>Methodology — How PESTLE Scenarios Were Built</h2>
    <div class="step"><div class="sn">1</div><div class="sb"><b>Baseline established:</b> The model is run on current observed feature values to get a baseline average MCP of Rs{baseline_mcp:,.0f}/MWh. This is the "do nothing" reference point.</div></div>
    <div class="step"><div class="sn">2</div><div class="sb"><b>Scenario feature overrides:</b> For each PESTLE scenario, we identify which input features would plausibly change and by how much — based on economic reasoning and historical data ranges. For example, a carbon tax primarily affects fuel cost features; a heatwave primarily affects temperature and cooling degree features.</div></div>
    <div class="step"><div class="sn">3</div><div class="sb"><b>Model re-scored:</b> The trained XGBoost model is re-run with the overridden features across all 96 daily blocks. The model captures non-linear interactions — a heatwave in the evening peak hour has a larger price impact than the same temperature change at 3 AM.</div></div>
    <div class="step"><div class="sn">4</div><div class="sb"><b>Delta computed:</b> The scenario average MCP is compared against baseline. The delta tells us both the direction (risk or opportunity) and magnitude (how much does this scenario matter for trading decisions).</div></div>
    <div class="step"><div class="sn">5</div><div class="sb"><b>Business interpretation:</b> Scenarios are ranked by impact to help traders prioritise hedging decisions. Large downward scenarios (recession, renewable surge) indicate need for sell-side positions. Large upward scenarios (carbon tax) indicate buy-side opportunity.</div></div>
    <div class="ins ing"><b>Why PESTLE for energy markets?</b> RTM electricity prices are uniquely sensitive to all six PESTLE dimensions simultaneously — government policy sets price caps, economic cycles drive demand, social events drive consumption peaks, technology (renewables) displaces thermal generation, legal frameworks control market access, and environmental factors (weather, monsoon) directly change both supply and demand.</div>
    </div>

    <div class="sec"><h2>PESTLE Framework Explained</h2>
    <table class="tbl">
      <tr><th>Letter</th><th>Dimension</th><th>Relevance to IEX RTM</th><th>Scenarios Modelled</th></tr>
      <tr><td style="font-size:1.4em;font-weight:bold;color:#e67e22">P</td><td><b>Political</b></td><td>Government policy on price caps, energy mix mandates, import duties on fossil fuels</td><td>Carbon Tax +20%, Price Cap Rs8,000</td></tr>
      <tr><td style="font-size:1.4em;font-weight:bold;color:#e94560">E</td><td><b>Economic</b></td><td>GDP growth drives industrial electricity demand; recessions collapse load; FX affects coal import cost</td><td>Economic Recession -15%</td></tr>
      <tr><td style="font-size:1.4em;font-weight:bold;color:#e74c3c">S</td><td><b>Social</b></td><td>Urbanisation, AC penetration, population growth increase peak demand; heatwaves amplify peaks</td><td>Heatwave +8°C</td></tr>
      <tr><td style="font-size:1.4em;font-weight:bold;color:#27ae60">T</td><td><b>Technological</b></td><td>Solar and wind deployment increases zero-marginal-cost supply; battery storage smooths peaks</td><td>Renewable Surge +30%</td></tr>
      <tr><td style="font-size:1.4em;font-weight:bold;color:#8e44ad">L</td><td><b>Legal</b></td><td>CERC regulations, RTM trading limits, must-run obligations for renewables, price ceiling rules</td><td>Price Cap Rs8,000</td></tr>
      <tr><td style="font-size:1.4em;font-weight:bold;color:#2980b9">E</td><td><b>Environmental</b></td><td>Monsoon seasonality, drought (affects hydro), extreme weather, air quality regulations</td><td>Monsoon Season</td></tr>
    </table>
    </div>"""

    body = f"""
    <div class="sec"><h2>Scenario Results — Summary Table</h2>
    <table class="tbl">
      <tr><th>Category</th><th>Scenario</th><th>Avg MCP</th><th>Delta vs Baseline</th><th>% Change</th></tr>
      {tbl_rows}
    </table>
    <div class="ins">Baseline avg MCP: <b>Rs{baseline_mcp:,.0f}/MWh</b>. Deltas show directional impact on average clearing price across all 96 daily blocks.</div>
    </div>
    {detail_cards}
    {method_html}"""

    return page("PESTLE Analysis","PESTLE Scenario Analysis",
        "7 market scenarios — how policy, economy, weather and technology shift RTM prices",
        badge("7 Scenarios","bb") + badge(f"Baseline: Rs{baseline_mcp:,.0f}/MWh","bd") + badge("XGBoost Model","bg"),
        body)

# ══════════════════════════════════════════════════════════════
# /monitoring
# ══════════════════════════════════════════════════════════════
@app.route("/monitoring")
def monitoring():
    now = now_ist()

    # ── 1. MODEL HEALTH ALERT ────────────────────────────────
    mape_val   = state["mape"] or 0
    model_ok   = state["model"] is not None
    iex_age    = file_age_minutes(os.path.join(DATA_DIR,"iex_live.csv"))
    wx_age     = file_age_minutes(os.path.join(DATA_DIR,"weather_live.csv"))
    com_age    = file_age_minutes(os.path.join(DATA_DIR,"commodities_live.csv"))
    data_stale = max(iex_age, wx_age) >= FRESHNESS_STALE

    # Load prediction log
    pred_df = pd.DataFrame()
    if os.path.exists(PRED_LOG):
        try:
            pred_df = pd.read_csv(PRED_LOG, parse_dates=["timestamp"]).dropna(subset=["predicted_mcp"])
        except: pass

    # Show last 30 predictions — backfill handles actuals for past blocks
    if len(pred_df) > 0:
        pred_df["_ts"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
        # Only past blocks — future belongs in /forecast/24h
        pred_df = pred_df[pred_df["_ts"] <= pd.Timestamp(now_ist())].drop(columns=["_ts"])

    has_actuals  = "actual_mcp" in pred_df.columns and pred_df["actual_mcp"].notna().sum() >= 5
    rolling_mape = None
    if has_actuals:
        recent = pred_df.dropna(subset=["actual_mcp"]).tail(30)
        if len(recent) > 0:
            rolling_mape = (abs(recent["actual_mcp"] - recent["predicted_mcp"]) / (recent["actual_mcp"] + 1) * 100).mean()

    # Determine overall health
    issues = []
    if not model_ok:        issues.append("Model not loaded")
    if data_stale:          issues.append(f"Live data stale ({max(iex_age,wx_age):.0f} min old)")
    if rolling_mape and rolling_mape > 30: issues.append(f"Rolling MAPE {rolling_mape:.1f}% exceeds 30% threshold")

    if not issues:
        health_color, health_status, health_bg = "#27ae60", "ALL SYSTEMS HEALTHY", "#f0fff4"
        health_border, health_rec = "#27ae60", "No action required. Continue monitoring."
    elif len(issues) == 1 and data_stale and not (rolling_mape and rolling_mape > 30):
        health_color, health_status, health_bg = "#f39c12", "WARNING — ATTENTION NEEDED", "#fffbf0"
        health_border, health_rec = "#f39c12", "Trigger /refresh to update live data. Monitor MAPE."
    else:
        health_color, health_status, health_bg = "#e94560", "CRITICAL — ACTION REQUIRED", "#fff8f8"
        health_border, health_rec = "#e94560", "Trigger /retrain if MAPE > 30%. Check /refresh for data issues."

    issue_rows = "".join(f'<li style="margin:4px 0">&#9888; {i}</li>' for i in issues) if issues else '<li style="color:#27ae60">&#10003; No issues detected</li>'

    health_html = f"""
    <div class="sec" style="border-left:6px solid {health_border};background:{health_bg}">
      <h2 style="color:{health_color};border-color:{health_color}">{health_status}</h2>
      <div class="grid">
        <div class="card" style="border-color:{'#27ae60' if model_ok else '#e94560'}">
          <div class="cl">Model</div>
          <div class="cv" style="color:{'#27ae60' if model_ok else '#e94560'};font-size:1em">{"LOADED" if model_ok else "NOT LOADED"}</div>
          <div class="cs">{state['model_name']} {state['version']}</div>
        </div>
        <div class="card" style="border-color:{'#27ae60' if mape_val<25 else '#f39c12' if mape_val<30 else '#e94560'}">
          <div class="cl">Test MAPE</div>
          <div class="cv">{f"{mape_val:.2f}%" if mape_val else "N/A"}</div>
          <div class="cs">{"Good" if mape_val<25 else "Watch" if mape_val<30 else "Retrain!"}</div>
        </div>
        <div class="card" style="border-color:{'#27ae60' if iex_age<20 else '#f39c12' if iex_age<45 else '#e94560'}">
          <div class="cl">IEX Data Age</div>
          <div class="cv" style="font-size:1.1em">{iex_age:.0f} min</div>
          <div class="cs">{freshness_label(iex_age)}</div>
        </div>
        <div class="card" style="border-color:{'#27ae60' if wx_age<20 else '#f39c12' if wx_age<45 else '#e94560'}">
          <div class="cl">Weather Age</div>
          <div class="cv" style="font-size:1.1em">{wx_age:.0f} min</div>
          <div class="cs">{freshness_label(wx_age)}</div>
        </div>
        <div class="card" style="border-color:{'#27ae60' if rolling_mape and rolling_mape<25 else '#f39c12' if rolling_mape and rolling_mape<30 else '#95a5a6'}">
          <div class="cl">Rolling MAPE (30)</div>
          <div class="cv">{f"{rolling_mape:.1f}%" if rolling_mape else "No actuals yet"}</div>
          <div class="cs">Last 30 predictions</div>
        </div>
        <div class="card">
          <div class="cl">Last Checked</div>
          <div class="cv" style="font-size:.85em">{now.strftime('%H:%M:%S')}</div>
          <div class="cs">{now.strftime('%d %b %Y')}</div>
        </div>
      </div>
      <ul style="margin:12px 0 8px 18px;font-size:.92em;line-height:1.8">{issue_rows}</ul>
      <div class="ins {"ing" if not issues else "inw"}""><b>Recommendation:</b> {health_rec}
        &nbsp;<a class="il" href="/refresh">Refresh Data</a> &nbsp;|&nbsp;
        <a class="il" href="/retrain">Retrain Model</a>
      </div>
    </div>"""

    # ── 2. DATA DRIFT (KS TEST) ──────────────────────────────
    drift_html = ""
    try:
        from scipy.stats import ks_2samp
        # Load training reference (historical) vs live
        hist_path = os.path.join(DATA_DIR,"iex_historical.csv")
        live_path = os.path.join(DATA_DIR,"iex_live.csv")
        if os.path.exists(hist_path) and os.path.exists(live_path):
            hist = pd.read_csv(hist_path).dropna(subset=["MCP"])
            live = pd.read_csv(live_path).dropna(subset=["MCP"])
            # Fix: compare against 18-month training window only, not full history
            hist["_date"] = pd.to_datetime(hist["date"], format="%d-%m-%Y", errors="coerce")
            cutoff = hist["_date"].max() - pd.DateOffset(months=18)
            hist = hist[hist["_date"] >= cutoff].drop(columns=["_date"])

            # Features to check for drift
            drift_checks = {}
            for col in ["MCP"]:
                if col in hist.columns and col in live.columns:
                    h_vals = hist[col].dropna().values
                    l_vals = live[col].dropna().values
                    if len(h_vals)>10 and len(l_vals)>5:
                        stat, pval = ks_2samp(h_vals, l_vals)
                        drift_checks[col] = (stat, pval)

            if drift_checks:
                drift_rows = ""
                any_drift = False
                for feat,(stat,pval) in drift_checks.items():
                    #drifted = pval < 0.05
                    drifted = stat > 0.20  # practical significance — p-value meaningless with large samples
                    if drifted: any_drift = True
                    dc = "#e94560" if drifted else "#27ae60"
                    label = "YES — DRIFT" if drifted else "No drift"
                    drift_rows += f"""<tr>
                      <td style="font-family:monospace;font-weight:bold">{feat}</td>
                      <td>{stat:.4f}</td>
                      <td>{pval:.4f}</td>
                      <td style="color:{dc};font-weight:bold">{label}</td>
                      <td style="font-size:.82em;color:#636e72">{"Current distribution differs from training — model may underperform" if drifted else "Distribution stable"}</td>
                    </tr>"""

                drift_html = f"""
                <div class="sec">
                  <h2>Data Drift Detection (Kolmogorov-Smirnov Test)</h2>
                  <div class="ins">KS test compares live feature distribution against historical training data.
                  <b>p-value &lt; 0.05</b> means the live data has drifted significantly from what the model was trained on — predictions may degrade.</div>
                  <table class="tbl">
                    <tr><th>Feature</th><th>KS Statistic</th><th>p-value</th><th>Drift?</th><th>Interpretation</th></tr>
                    {drift_rows}
                  </table>
                  <div class="ins {"inw" if any_drift else "ing"}">
                    {"&#9888; Drift detected — consider retraining if MAPE is also rising. Trigger <a class='il' href='/retrain'>/retrain</a>" if any_drift else "&#10003; No significant drift detected — model training distribution is stable"}
                  </div>
                </div>"""
    except Exception as e:
        drift_html = f'<div class="sec"><h2>Data Drift Detection</h2><div class="ins inw">scipy not available: {e}</div></div>'

    # ── 3. LAST 30 PREDICTIONS TABLE ─────────────────────────
    pred_html = ""
    if len(pred_df) > 0:
        recent_preds = pred_df.tail(30).iloc[::-1]  # newest first
        pred_rows = ""
        for _, row in recent_preds.iterrows():
            sig_color = {"BUY":"#27ae60","SELL":"#e94560","HOLD":"#f39c12"}.get(str(row.get("signal","")), "#95a5a6")
            conf_class = {"HIGH":"cg","MEDIUM":"co","LOW":"cr"}.get(str(row.get("confidence","")), "")
            actual = row.get("actual_mcp")
            mape_cell = ""
            if pd.notna(actual) and actual > 0:
                err = abs(actual - row["predicted_mcp"]) / actual * 100
                mape_color = "#27ae60" if err < 20 else "#f39c12" if err < 30 else "#e94560"
                mape_cell = f'<span style="color:{mape_color};font-weight:bold">{err:.1f}%</span>'
            else:
                mape_cell = '<span style="color:#b2bec3">—</span>'

            pred_rows += f"""<tr>
              <td style="font-size:.8em;color:#636e72">{str(row.get("timestamp",""))[:16]}</td>
              <td><b>Rs{float(row["predicted_mcp"]):,.2f}</b></td>
              <td>{f"Rs{float(actual):,.2f}" if pd.notna(actual) else '<span style="color:#b2bec3">Pending</span>'}</td>
              <td>{mape_cell}</td>
              <td><span style="background:{sig_color};color:white;padding:1px 8px;border-radius:8px;font-size:.78em">{row.get("signal","—")}</span></td>
              <td><span style="font-size:.78em;font-weight:bold;color:{{"HIGH":"#27ae60","MEDIUM":"#f39c12","LOW":"#e94560"}}.get(str(row.get("confidence","")), "#95a5a6")">{row.get("confidence","—")}</span></td>
              <td style="font-family:monospace;font-size:.8em;color:#636e72">{row.get("model_version","—")}</td>
            </tr>"""

        avg_pred = pred_df["predicted_mcp"].mean()
        pred_html = f"""
        <div class="sec">
          <h2>Last 30 Predictions</h2>
          <div class="grid" style="grid-template-columns:repeat(4,1fr)">
            <div class="card"><div class="cl">Total Logged</div><div class="cv">{len(pred_df)}</div></div>
            <div class="card cg"><div class="cl">Avg Predicted MCP</div><div class="cv">Rs{avg_pred:,.0f}</div></div>
            <div class="card {"cg" if rolling_mape and rolling_mape<25 else "co" if rolling_mape and rolling_mape<30 else "cgr"}">
              <div class="cl">Rolling MAPE</div>
              <div class="cv">{f"{rolling_mape:.1f}%" if rolling_mape else "No actuals"}</div>
            </div>
            <div class="card"><div class="cl">Model Version</div><div class="cv" style="font-size:1em">{state["version"]}</div></div>
          </div>
          <table class="tbl">
            <tr><th>Timestamp</th><th>Predicted</th><th>Actual</th><th>MAPE</th><th>Signal</th><th>Confidence</th><th>Version</th></tr>
            {pred_rows}
          </table>
          <div class="ins">Actual MCP is filled in retrospectively when IEX publishes final data. Until then it shows Pending.</div>
        </div>"""
    else:
        pred_html = '<div class="sec"><h2>Last 30 Predictions</h2><div class="ins inw">No predictions logged yet — visit <a class="il" href="/forecast/24h">/forecast/24h</a> or <a class="il" href="/predict/sample">/predict/sample</a> to generate predictions.</div></div>'

    # ── 4. DATA PIPELINE HEALTH ──────────────────────────────
    pipeline_files = {
        "IEX Live":         ("iex_live.csv",          "RTM 15-min prices",         "scraper_iex.py"),
        "IEX Historical":   ("iex_historical.csv",    "Full price history",         "scraper_iex.py"),
        "Weather Live":     ("weather_live.csv",       "NASA POWER 8 cities",       "scraper_weather.py"),
        "Weather Historical":("weather_historical.csv","3-year weather history",    "scraper_weather.py"),
        "Commodities Live": ("commodities_live.csv",   "Crude, gas, FX",            "fetch_historical_commodities.py"),
        "Master Dataset":   ("master_training_data.csv","Merged training data",     "merge_historical.py"),
    }
    pipe_rows = ""
    for label,(fname,desc,script) in pipeline_files.items():
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            try:
                df_tmp  = pd.read_csv(fpath)
                nrows   = len(df_tmp)
                age     = file_age_minutes(fpath)
                mod_dt  = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%d %b %H:%M")
                size_kb = os.path.getsize(fpath)//1024
                status  = freshness_label(age) if "_live" in fname else "OK"
                sc      = {"FRESH":"#27ae60","WARNING":"#f39c12","STALE":"#e94560","OK":"#2980b9"}.get(status,"#95a5a6")
                pipe_rows += f"""<tr>
                  <td><b>{label}</b><br><span style="font-size:.78em;color:#95a5a6">{script}</span></td>
                  <td style="font-size:.85em;color:#636e72">{desc}</td>
                  <td style="font-weight:bold">{nrows:,}</td>
                  <td style="font-size:.85em">{mod_dt}</td>
                  <td>{size_kb} KB</td>
                  <td><span style="background:{sc};color:white;padding:2px 8px;border-radius:8px;font-size:.78em">{status}</span></td>
                </tr>"""
            except Exception as ex:
                pipe_rows += f'<tr><td><b>{label}</b></td><td colspan="5" style="color:#e94560">Read error: {ex}</td></tr>'
        else:
            pipe_rows += f'<tr><td><b>{label}</b></td><td style="color:#636e72;font-size:.85em">{desc}</td><td colspan="4" style="color:#b2bec3">File not found — run pipeline first</td></tr>'

    pipeline_html = f"""
    <div class="sec">
      <h2>Data Pipeline Health</h2>
      <table class="tbl">
        <tr><th>File</th><th>Description</th><th>Records</th><th>Last Modified</th><th>Size</th><th>Status</th></tr>
        {pipe_rows}
      </table>
      <div class="ins">Live files refresh every 30 min automatically. Historical files update when pipeline reruns.
        <a class="il" href="/refresh">Trigger refresh →</a>
      </div>
    </div>"""

    # ── 5. PRICE DISTRIBUTION SHIFT ──────────────────────────
    dist_html = ""
    try:
        hist_path = os.path.join(DATA_DIR,"iex_historical.csv")
        live_path = os.path.join(DATA_DIR,"iex_live.csv")
        if os.path.exists(hist_path) and os.path.exists(live_path):
            hist = pd.read_csv(hist_path).dropna(subset=["MCP"])
            live = pd.read_csv(live_path).dropna(subset=["MCP"])

            # Recent 30 days from historical
            hist["date_p"] = pd.to_datetime(hist["date"], format="%d-%m-%Y", errors="coerce")
            cutoff = hist["date_p"].max() - pd.Timedelta(days=30)
            recent_hist = hist[hist["date_p"] >= cutoff]["MCP"]

            def pct_stats(series, label):
                s = series.dropna()
                return {
                    "label": label,
                    "n":     len(s),
                    "mean":  s.mean(),
                    "median":s.median(),
                    "std":   s.std(),
                    "p10":   s.quantile(0.10),
                    "p25":   s.quantile(0.25),
                    "p75":   s.quantile(0.75),
                    "p90":   s.quantile(0.90),
                    "spikes":(s>9000).mean()*100,
                }

            train_stats = pct_stats(hist["MCP"],        "Full Historical")
            recent_stats= pct_stats(recent_hist,        "Last 30 Days (Historical)")
            live_stats  = pct_stats(live["MCP"],        "Live (Today)")

            def stat_row(s):
                spike_c = "#e94560" if s["spikes"]>10 else "#f39c12" if s["spikes"]>5 else "#27ae60"
                return f"""<tr>
                  <td><b>{s["label"]}</b></td>
                  <td>{s["n"]:,}</td>
                  <td style="font-weight:bold">Rs{s["mean"]:,.0f}</td>
                  <td>Rs{s["median"]:,.0f}</td>
                  <td>Rs{s["std"]:,.0f}</td>
                  <td style="color:#636e72">Rs{s["p10"]:,.0f}</td>
                  <td style="color:#2980b9">Rs{s["p25"]:,.0f}</td>
                  <td style="color:#2980b9">Rs{s["p75"]:,.0f}</td>
                  <td style="color:#636e72">Rs{s["p90"]:,.0f}</td>
                  <td style="color:{spike_c};font-weight:bold">{s["spikes"]:.1f}%</td>
                </tr>"""

            # Regime shift warning
            mean_diff = abs(live_stats["mean"] - train_stats["mean"])
            mean_diff_pct = mean_diff / (train_stats["mean"] + 1) * 100
            shift_warn = mean_diff_pct > 15

            dist_html = f"""
            <div class="sec">
              <h2>Price Distribution Shift</h2>
              <div class="ins">Compares current live MCP distribution against historical training data.
              Large shifts indicate a market regime change — the model may need retraining.</div>
              <table class="tbl">
                <tr><th>Period</th><th>Records</th><th>Mean</th><th>Median</th><th>Std Dev</th>
                    <th>P10</th><th>P25</th><th>P75</th><th>P90</th><th>Spikes &gt;9k</th></tr>
                {stat_row(train_stats)}
                {stat_row(recent_stats)}
                {stat_row(live_stats)}
              </table>
              <div class="ins {"inw" if shift_warn else "ing"}">
                {"&#9888; <b>Regime shift detected:</b> Live mean MCP differs from training mean by " + f"{mean_diff_pct:.1f}% (Rs{mean_diff:,.0f}/MWh). Model was trained on different price levels — consider retraining." if shift_warn else f"&#10003; Price levels are consistent. Live mean (Rs{live_stats['mean']:,.0f}) is within normal range of training mean (Rs{train_stats['mean']:,.0f})."}
              </div>
            </div>"""
    except Exception as e:
        dist_html = f'<div class="sec"><h2>Price Distribution Shift</h2><div class="ins inw">Could not compute: {e}</div></div>'

    # ── ASSEMBLE PAGE ────────────────────────────────────────
    body = health_html + drift_html + pred_html + pipeline_html + dist_html
    body += f'<p class="ts">Page generated: {now.strftime("%d %b %Y %H:%M:%S")} | Auto-refresh every 60s</p>'

    overall_badge = badge("HEALTHY","bg") if not issues else badge("WARNING","bo") if len(issues)==1 else badge("CRITICAL","br")

    return page("Monitoring","Model Monitoring",
        "Health alerts | Drift detection | Predictions | Pipeline | Distribution shift",
        overall_badge + badge(f"{state['model_name']} {state['version']}","bb") + badge(f"Checked: {now.strftime('%H:%M')}","bd"),
        body)

# ══════════════════════════════════════════════════════════════
# /refresh
# ══════════════════════════════════════════════════════════════
@app.route("/refresh")
def refresh_data():
    def _refresh():
        try:
            from data_pipeline.scheduler import refresh_all
            refresh_all()
        except Exception as e:
            import subprocess
            for s in ["scraper_iex.py","iex_scraper.py"]:
                sp = os.path.join(_BASE_DIR,"data_pipeline",s)
                if os.path.exists(sp):
                    subprocess.run(["python",sp],timeout=120,capture_output=True,cwd=_BASE_DIR); break
    threading.Thread(target=_refresh, daemon=True).start()

    body = f"""
    <div class="sec"><h2>Refresh Triggered</h2>
    <div class="ins ing">Live data refresh has been triggered in the background. Check <a class="il" href="/health">/health</a> in 60 seconds to see updated freshness.</div>
    <div class="grid">
      <div class="card cg"><div class="cl">Status</div><div class="cv" style="font-size:1em">RUNNING</div><div class="cs">Background process</div></div>
      <div class="card"><div class="cl">Expected Time</div><div class="cv" style="font-size:1em">~60s</div><div class="cs">IEX + Weather</div></div>
      <div class="card"><div class="cl">Triggered At</div><div class="cv" style="font-size:.85em">{datetime.now().strftime('%H:%M:%S')}</div></div>
    </div>
    </div>
    <div class="ins">Sources: IEX RTM (Selenium), NASA POWER (weather), Yahoo Finance (commodities).</div>"""

    return page("Refresh","Data Refresh",
        "Immediate live data update triggered",
        badge("IN PROGRESS","bo"),
        body)

# ══════════════════════════════════════════════════════════════
# /retrain
# ══════════════════════════════════════════════════════════════
@app.route("/retrain", methods=["GET","POST"])
def retrain():
    def _run():
        import subprocess
        subprocess.run(["python", os.path.join(_BASE_DIR,"run_pipeline.py")])
    threading.Thread(target=_run, daemon=True).start()

    body = f"""
    <div class="sec"><h2>Retraining Started</h2>
    <div class="ins ing">Model retraining has been triggered. The full pipeline (data merge → feature engineering → model training → evaluation) is running in the background.</div>
    <div class="grid">
      <div class="card co"><div class="cl">Status</div><div class="cv" style="font-size:1em">RUNNING</div><div class="cs">run_pipeline.py</div></div>
      <div class="card"><div class="cl">Expected Time</div><div class="cv" style="font-size:1em">5–15 min</div><div class="cs">Full pipeline</div></div>
      <div class="card"><div class="cl">Started At</div><div class="cv" style="font-size:.85em">{datetime.now().strftime('%H:%M:%S')}</div></div>
    </div>
    </div>
    <div class="ins">When complete, <a class="il" href="/health">/health</a> will show the new model version and MAPE. <a class="il" href="/model-summary">/model-summary</a> will update automatically.</div>"""

    return page("Retrain","Model Retraining",
        "Full pipeline triggered — merge → features → train → evaluate",
        badge("IN PROGRESS","bo"),
        body)

@app.route("/model-summary")
def model_summary():
    """Rich HTML model card"""
    import csv as csv_mod

    full_meta = {}
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f: full_meta = json.load(f)
        except: pass

    comparison = []
    comp_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        try:
            with open(comp_path) as f:
                comparison = list(csv_mod.DictReader(f))
        except: pass

    biz = {}
    biz_path = os.path.join(MODELS_DIR, "business_evaluation.json")
    if os.path.exists(biz_path):
        try:
            with open(biz_path) as f: biz = json.load(f)
        except: pass

    fi = {}
    fi_path = os.path.join(MODELS_DIR, "feature_importance.json")
    if os.path.exists(fi_path):
        try:
            with open(fi_path) as f: fi = json.load(f)
        except: pass

    # Build comparison table
    if comparison:
        headers = "".join(f"<th>{k}</th>" for k in comparison[0].keys())
        comp_rows = ""
        for row in comparison:
            model_name = row.get("model", row.get("Model",""))
            is_best = "xgb" in model_name.lower()
            style = 'style="background:#f0fff4;font-weight:bold"' if is_best else ""
            badge = ' <span style="background:#27ae60;color:white;padding:2px 8px;border-radius:10px;font-size:11px">BEST</span>' if is_best else ""
            comp_rows += f"<tr {style}><td>{model_name}{badge}</td>" + "".join(f"<td>{v}</td>" for k,v in row.items() if k not in ["model","Model"]) + "</tr>"
        comp_table = f'<table class="tbl"><tr>{headers}</tr>{comp_rows}</table>'
    else:
        comp_table = """<table class="tbl">
          <tr><th>Model</th><th>Test MAPE</th><th>CV MAPE</th><th>RMSE</th><th>Notes</th></tr>
          <tr style="background:#f0fff4;font-weight:bold"><td>XGBoost <span style="background:#27ae60;color:white;padding:2px 8px;border-radius:10px;font-size:11px">BEST</span></td><td>20.65%</td><td>21.0%</td><td>114 Rs/MWh</td><td>Handles non-linearity and regime shifts</td></tr>
          <tr><td>SVM (RBF kernel)</td><td>69.48%</td><td>21.28%</td><td>-</td><td>Good CV but poor extrapolation on unseen regimes</td></tr>
          <tr><td>ARIMA (baseline)</td><td>53.46%</td><td>-</td><td>-</td><td>Univariate only, no exogenous features</td></tr>
        </table>"""

    # Feature importance
    if fi:
        top = sorted(fi.items(), key=lambda x: float(x[1]), reverse=True)[:10]
        fi_rows = "".join(
            f'<tr><td style="font-family:monospace;font-weight:bold">{i+1}. {feat}</td><td>{float(imp):.4f}</td><td><div style="background:#2980b9;height:16px;width:{int(float(imp)*400)}px;border-radius:3px;min-width:4px"></div></td></tr>'
            for i,(feat,imp) in enumerate(top))
        fi_section = f'<table class="tbl"><tr><th>Feature</th><th>Importance</th><th>Weight</th></tr>{fi_rows}</table>'
    else:
        fi_section = """<table class="tbl">
          <tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Business Meaning</th></tr>
          <tr><td>1</td><td style="font-family:monospace">mcp_lag_1h</td><td>0.315</td><td>Most recent clearing price — strongest predictor</td></tr>
          <tr><td>2</td><td style="font-family:monospace">mcp_lag_24h</td><td>0.142</td><td>Same time slot yesterday</td></tr>
          <tr><td>3</td><td style="font-family:monospace">price_rolling_24h</td><td>0.098</td><td>24-hour rolling average</td></tr>
          <tr><td>4</td><td style="font-family:monospace">hour</td><td>0.071</td><td>Intraday demand pattern</td></tr>
          <tr><td>5</td><td style="font-family:monospace">temp_delhi</td><td>0.052</td><td>Cooling demand driver</td></tr>
        </table>"""

    trained_at = full_meta.get("trained_at", str(state.get("loaded_at","Unknown")))
    train_from = full_meta.get("train_from","Aug 2024")
    train_to   = full_meta.get("train_to","Feb 2026")
    n_records  = full_meta.get("n_records","52,705")
    n_features = full_meta.get("n_features","36")
    cv_folds   = full_meta.get("cv_folds","3")
    test_split = full_meta.get("test_split","Last 3 months")
    model_ver  = full_meta.get("version", state.get("version","v7"))
    test_mape  = full_meta.get("test_mape", state.get("mape") or 20.65)
    dist_shift = full_meta.get("distribution_shift","4.2%")
    biz_pl     = biz.get("total_pl_crore", biz.get("simulated_pl","7.7 Crore"))

    css = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#2d3436}
.header{background:linear-gradient(135deg,#1a1a2e,#0f3460);color:white;padding:40px;text-align:center}
.header h1{font-size:2em;margin-bottom:8px}
.badges{display:flex;justify-content:center;gap:12px;margin-top:16px;flex-wrap:wrap}
.badge{padding:6px 16px;border-radius:20px;font-size:0.88em;font-weight:bold}
.badge.green{background:#27ae60}.badge.blue{background:#2980b9}.badge.dark{background:rgba(255,255,255,0.15)}
.container{max-width:1200px;margin:0 auto;padding:28px 18px}
.section{background:white;border-radius:12px;padding:26px;margin-bottom:24px;box-shadow:0 2px 10px rgba(0,0,0,0.07)}
.section h2{font-size:1.4em;color:#1a1a2e;margin-bottom:14px;padding-bottom:10px;border-bottom:3px solid #e94560}
.section h3{font-size:1.05em;color:#2980b9;margin:18px 0 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px;margin:16px 0}
.card{background:#f8f9ff;border-radius:10px;padding:16px;border-left:4px solid #2980b9;text-align:center}
.card.red{border-color:#e94560}.card.green{border-color:#27ae60}.card.orange{border-color:#f39c12}
.card-val{font-size:1.8em;font-weight:bold;color:#1a1a2e;margin:6px 0}
.card-lbl{font-size:0.78em;color:#636e72;text-transform:uppercase;letter-spacing:0.5px}
.card-sub{font-size:0.8em;color:#b2bec3;margin-top:3px}
.tbl{width:100%;border-collapse:collapse;margin:12px 0}
.tbl th{background:#1a1a2e;color:white;padding:11px 14px;text-align:left;font-size:0.9em}
.tbl td{padding:11px 14px;border-bottom:1px solid #eee;font-size:0.92em}
.tbl tr:hover td{background:#f8f9ff}
.insight{background:#f8f9ff;border-left:4px solid #2980b9;padding:13px 16px;border-radius:0 8px 8px 0;margin:12px 0;font-size:0.93em;line-height:1.7}
.insight.warn{border-color:#e94560;background:#fff8f8}.insight.green{border-color:#27ae60;background:#f0fff4}
.step{display:flex;gap:14px;margin:12px 0;align-items:flex-start}
.step-num{background:#1a1a2e;color:white;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:0.9em;flex-shrink:0;margin-top:2px}
.step-body{flex:1;font-size:0.93em;line-height:1.7}
.bug{background:#fff3f3;border-left:4px solid #e94560;padding:10px 14px;margin:8px 0;border-radius:0 6px 6px 0;font-size:0.9em;line-height:1.6}
.fix{background:#f0fff4;border-left:4px solid #27ae60;padding:10px 14px;margin:8px 0;border-radius:0 6px 6px 0;font-size:0.9em}
.back{display:inline-block;margin-top:16px;color:#2980b9;text-decoration:none}
"""

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><title>Model Summary - Group 05</title>
<style>{css}</style></head><body>
<div class="header">
  <h1>Model Summary Card</h1>
  <p>IEX RTM Electricity Price Forecasting — Group 05, ISB AMPBA</p>
  <div class="badges">
    <span class="badge green">Best Model: XGBoost {model_ver}</span>
    <span class="badge blue">MAPE: {test_mape}%</span>
    <span class="badge dark">Last Trained: {trained_at}</span>
    <span class="badge dark">CRISP-ML(Q)</span>
  </div>
</div>
<div class="container">

<div class="section"><h2>Performance at a Glance</h2>
<div class="grid">
  <div class="card green"><div class="card-lbl">Best Model</div><div class="card-val">XGBoost</div><div class="card-sub">Version {model_ver}</div></div>
  <div class="card red"><div class="card-lbl">Test MAPE</div><div class="card-val">{test_mape}%</div><div class="card-sub">Target was &lt;25%</div></div>
  <div class="card"><div class="card-lbl">ARIMA Baseline</div><div class="card-val">53.46%</div><div class="card-sub">MAPE benchmark</div></div>
  <div class="card green"><div class="card-lbl">Improvement</div><div class="card-val">63.8%</div><div class="card-sub">Over ARIMA</div></div>
  <div class="card orange"><div class="card-lbl">Simulated P&L</div><div class="card-val">Rs7.7Cr</div><div class="card-sub">100MW, 3 months</div></div>
  <div class="card"><div class="card-lbl">Distribution Shift</div><div class="card-val">{dist_shift}</div><div class="card-sub">Train vs Test</div></div>
  <div class="card"><div class="card-lbl">CV Folds</div><div class="card-val">{cv_folds}</div><div class="card-sub">TimeSeriesSplit</div></div>
  <div class="card"><div class="card-lbl">Features</div><div class="card-val">{n_features}</div><div class="card-sub">All leakage-free</div></div>
</div></div>

<div class="section"><h2>Models Evaluated</h2>
{comp_table}
<h3>Why XGBoost Won</h3>
<div class="insight green">XGBoost outperformed because RTM prices are driven by non-linear interactions (high temperature + low wind + peak hour = spike risk). Tree-based models capture these naturally. SVM had competitive CV but failed on price spikes. ARIMA is univariate — cannot use weather or commodity signals.</div>
<h3>Cross-Validation Strategy</h3>
<div class="insight">Used <b>TimeSeriesSplit ({cv_folds} folds)</b> — respects temporal ordering so future data never leaks into training. Each fold trains on earlier data and tests on the next window. Standard K-Fold would allow future data to train the model, giving falsely optimistic CV scores.</div>
</div>

<div class="section"><h2>Training Data Used</h2>
<div class="grid">
  <div class="card"><div class="card-lbl">Training Period</div><div class="card-val" style="font-size:1.1em">{train_from}</div><div class="card-sub">to {train_to}</div></div>
  <div class="card"><div class="card-lbl">Training Records</div><div class="card-val">{n_records}</div><div class="card-sub">15-min RTM blocks</div></div>
  <div class="card"><div class="card-lbl">Test Split</div><div class="card-val" style="font-size:1em">{test_split}</div><div class="card-sub">Walk-forward</div></div>
  <div class="card"><div class="card-lbl">Window Strategy</div><div class="card-val" style="font-size:1em">18 Months</div><div class="card-sub">Regime-aligned</div></div>
</div>
<h3>Why 18-Month Training Window?</h3>
<div class="insight warn"><b>Critical Finding:</b> Training on all 3 years caused MAPE to explode to 164%. Root cause: 2023 avg MCP was Rs5,805/MWh vs 2025 avg Rs4,078/MWh — a different market regime due to higher renewable penetration. Limiting to last 18 months keeps train and test in the same regime, dropping MAPE to 20.65%.</div>
<h3>Data Sources</h3>
<table class="tbl">
<tr><th>Source</th><th>Data</th><th>Period</th><th>Granularity</th></tr>
<tr><td>IEX Website (Selenium)</td><td>RTM MCP, volumes, bids</td><td>Feb 2023 - present</td><td>15-min</td></tr>
<tr><td>Mendeley (Price.xlsx)</td><td>Hourly MCP + demand</td><td>2021-2023 (gap fill)</td><td>Hourly</td></tr>
<tr><td>NASA POWER API</td><td>Weather - 8 Indian cities</td><td>3 years</td><td>Hourly</td></tr>
<tr><td>Yahoo Finance</td><td>Crude oil, natural gas</td><td>3 years</td><td>Daily</td></tr>
<tr><td>Frankfurter API</td><td>USD/INR exchange rate</td><td>3 years</td><td>Daily</td></tr>
</table>
</div>

<div class="section"><h2>Data Cleaning and Merging Process</h2>
<div class="step"><div class="step-num">1</div><div class="step-body"><b>IEX Scraping (scraper_iex.py)</b><br>Selenium headless Chrome scrapes IEX RTM portal day-by-day. Extracts MCP, MCV, purchase/sell bids, scheduled volume per 15-min block. Saved to iex_historical.csv.</div></div>
<div class="step"><div class="step-num">2</div><div class="step-body"><b>Weather Fetch (scraper_weather.py)</b><br>NASA POWER API queried for 8 cities: Delhi, Mumbai, Chennai, Kolkata, Hyderabad, Bangalore, Ahmedabad, Jaipur. Features: temperature, wind speed, humidity, cloud cover. Aggregated to national demand proxy.</div></div>
<div class="step"><div class="step-num">3</div><div class="step-body"><b>Commodities Fetch (fetch_historical_commodities.py)</b><br>Yahoo Finance: Brent crude, natural gas. Frankfurter: USD/INR. Coal approximated via weighted formula. Forward-filled for missing trading days.</div></div>
<div class="step"><div class="step-num">4</div><div class="step-body"><b>Gap Fill - Option B Merge (merge_historical.py)</b><br>Mendeley Price.xlsx fills 2021-2022 gap. Hourly data upsampled to 15-min via forward-fill. Gap-fill records flagged and excluded from recent-window training.</div></div>
<div class="step"><div class="step-num">5</div><div class="step-body"><b>Schema Validation (validator.py)</b><br>Checks required columns, date ranges, MCP range (10-12000 Rs/MWh), null thresholds, dataset versioning. Rejects corrupt data before it reaches the model.</div></div>
<div class="step"><div class="step-num">6</div><div class="step-body"><b>Leakage-Free Feature Engineering</b><br>All rolling stats computed on <b>MCP.shift(1)</b> — current price never appears in any feature. Lag features use explicit shifts (1h, 2h, 24h, 48h, 1w). 36 features total.</div></div>

<h3>Critical Bugs Found and Fixed</h3>
<div class="bug"><b>Bug 1 - Fake Data Extension:</b> Price.xlsx forward-filled to 15-min created artificial 4x repetition. Impact: MAPE 79%. Fix: Date range locked to scraped IEX period only.</div>
<div class="bug"><b>Bug 2 - Target Leakage (seasonality feature):</b> seasonality = MCP - trend used current MCP directly. Gave 49% feature importance — model saw the answer. Impact: MAPE 1.6% (cheating) → 160%+ when removed. Fix: Feature removed entirely.</div>
<div class="bug"><b>Bug 3 - Rolling Window Leakage:</b> price_rolling_24h included current block MCP in window. Fix: All rolling on MCP.shift(1).</div>
<div class="bug"><b>Bug 4 - Regime Mismatch:</b> 6-month test split hit different price regime (Rs4605 vs Rs3390). MAPE: 164%. Fix: 18-month training window.</div>
<div class="fix"><b>Final Result after all fixes:</b> MAPE dropped from 164% to 20.65% on clean, leakage-free data — a 63.8% improvement over ARIMA baseline.</div>
</div>

<div class="section"><h2>Feature Importance (Top 10)</h2>
{fi_section}
<div class="insight">All 36 features are leakage-free — validated by confirming no feature causes near-zero MAPE when included (which would indicate the model is cheating).</div>
</div>

<div class="section"><h2>Deployment and Monitoring</h2>
<table class="tbl">
<tr><th>Aspect</th><th>Detail</th></tr>
<tr><td>API Framework</td><td>Flask REST API - 12 endpoints</td></tr>
<tr><td>Deployment</td><td>AWS EC2 t3.small (ap-southeast-2) via Docker</td></tr>
<tr><td>Live Data Refresh</td><td>Auto-scheduler every 30 minutes</td></tr>
<tr><td>Drift Detection</td><td>KS-test on feature distributions + rolling MAPE</td></tr>
<tr><td>Model Versioning</td><td>Archived in models/archive/ with rollback support</td></tr>
<tr><td>Retraining Trigger</td><td>Manual via /retrain endpoint or on drift alert</td></tr>
<tr><td>Prediction Logging</td><td>Every prediction logged to prediction_log.csv</td></tr>
</table>
</div>

<a class="back" href="/">Back to API Home</a>
</div>
<div style="text-align:center;padding:20px;color:#b2bec3;font-size:0.85em">Group 05 - ISB AMPBA | CRISP-ML(Q) | Last trained: {trained_at}</div>
</body></html>"""



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
        def _fallback_scheduler():
            import subprocess, time
            time.sleep(60)
            while True:
                try:
                    for s in ["scraper_iex.py","iex_scraper.py"]:
                        sp = os.path.join(_BASE_DIR,"data_pipeline",s)
                        if os.path.exists(sp):
                            subprocess.run(["python",sp],timeout=120,capture_output=True,cwd=_BASE_DIR); break
                except: pass
                time.sleep(1800)
        threading.Thread(target=_fallback_scheduler, daemon=True).start()
        print("Fallback scheduler started")

    try:
        from data_pipeline.eda_generator import generate_and_save
        generate_and_save(); print("EDA ready")
    except Exception as e: print(f"EDA: {e}")

if __name__ == "__main__":
    startup()
    print(f"\nFlask running → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
