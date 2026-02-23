"""
monitor.py
Rolling RMSE/MAPE tracking, drift detection, data quality checks
Covers: Monitoring & Retraining Loop box from slide
Group 05 - ISB AMPBA
"""

import pandas as pd
import numpy as np
import os, json
from datetime import datetime, timedelta

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
PRED_LOG      = os.path.join(DATA_DIR,  "prediction_log.csv")
MONITOR_LOG   = os.path.join(DATA_DIR,  "monitor_log.json")
ALERTS_LOG    = os.path.join(DATA_DIR,  "alerts_log.csv")

# ── Thresholds ────────────────────────────────────────────────
MAPE_ALERT_THRESHOLD  = 10.0   # alert if rolling MAPE > 10%
DRIFT_THRESHOLD       = 0.20   # alert if mean shifts by >20%
FRESHNESS_MINUTES     = 30     # data older than 30 min = stale
RETRAIN_MAPE_TRIGGER  = 15.0   # auto-trigger retrain if MAPE > 15%

def log_prediction(input_features, predicted_mcp, confidence,
                   actual_mcp=None, model_version="unknown"):
    """Log every prediction for audit trail and monitoring"""
    row = {
        "timestamp":       datetime.now().isoformat(),
        "model_version":   model_version,
        "predicted_mcp":   round(predicted_mcp, 2),
        "confidence":      confidence,
        "actual_mcp":      actual_mcp,
        "error_pct":       abs(predicted_mcp - actual_mcp) / actual_mcp * 100
                           if actual_mcp else None,
        "mcp_lag_1h":      input_features.get("mcp_lag_1h"),
        "temp_delhi":      input_features.get("temp_delhi"),
        "hour":            input_features.get("hour"),
        "is_weekend":      input_features.get("is_weekend"),
    }
    df = pd.DataFrame([row])
    if os.path.exists(PRED_LOG):
        df.to_csv(PRED_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(PRED_LOG, index=False)
    return row

def compute_rolling_metrics(window_hours=24):
    """Compute rolling MAPE/RMSE from prediction log"""
    if not os.path.exists(PRED_LOG):
        return None
    df = pd.read_csv(PRED_LOG, parse_dates=["timestamp"])
    df = df.dropna(subset=["actual_mcp","predicted_mcp"])
    if len(df) < 10:
        return None

    # Last N hours
    cutoff = datetime.now() - timedelta(hours=window_hours)
    recent = df[df["timestamp"] >= cutoff]
    if len(recent) < 5:
        recent = df.tail(50)

    y_true = recent["actual_mcp"].values
    y_pred = recent["predicted_mcp"].values

    mask = y_true != 0
    rolling_mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    rolling_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    rolling_mae  = np.mean(np.abs(y_true - y_pred))

    return {
        "window_hours":    window_hours,
        "n_predictions":   len(recent),
        "rolling_mape":    round(rolling_mape, 2),
        "rolling_rmse":    round(rolling_rmse, 2),
        "rolling_mae":     round(rolling_mae,  2),
        "computed_at":     datetime.now().isoformat(),
    }

def check_data_drift():
    """
    Detect data drift by comparing recent live data stats
    to historical training data stats
    """
    drift_results = {}

    # Check IEX price drift
    live_file = os.path.join(DATA_DIR, "iex_live.csv")
    hist_file = os.path.join(DATA_DIR, "iex_historical.csv")

    if os.path.exists(live_file) and os.path.exists(hist_file):
        live = pd.read_csv(live_file)
        hist = pd.read_csv(hist_file)

        if "MCP" in live.columns and "MCP" in hist.columns:
            live_mean = live["MCP"].mean()
            hist_mean = hist["MCP"].mean()
            drift_pct = abs(live_mean - hist_mean) / hist_mean

            drift_results["MCP_drift"] = {
                "historical_mean": round(hist_mean, 2),
                "live_mean":       round(live_mean, 2),
                "drift_pct":       round(drift_pct * 100, 2),
                "alert":           drift_pct > DRIFT_THRESHOLD,
            }

    # Check weather drift (temperature)
    wx_live = os.path.join(DATA_DIR, "weather_live.csv")
    wx_hist = os.path.join(DATA_DIR, "weather_historical.csv")

    if os.path.exists(wx_live) and os.path.exists(wx_hist):
        live_wx = pd.read_csv(wx_live)
        hist_wx = pd.read_csv(wx_hist)

        if "temperature" in live_wx.columns and "temperature" in hist_wx.columns:
            live_t = live_wx["temperature"].mean()
            hist_t = hist_wx["temperature"].mean()
            if hist_t != 0:
                drift_pct = abs(live_t - hist_t) / abs(hist_t)
                drift_results["temperature_drift"] = {
                    "historical_mean": round(hist_t, 2),
                    "live_mean":       round(live_t, 2),
                    "drift_pct":       round(drift_pct * 100, 2),
                    "alert":           drift_pct > DRIFT_THRESHOLD,
                }

    return drift_results

def check_data_freshness():
    """Check if all live data files are fresh"""
    freshness = {}
    files = {
        "IEX":         "iex_live.csv",
        "Weather":     "weather_live.csv",
        "Commodities": "commodities_live.csv",
    }
    for name, fname in files.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            freshness[name] = {"status": "MISSING", "age_minutes": None}
            continue
        df = pd.read_csv(fpath)
        if len(df) == 0:
            freshness[name] = {"status": "EMPTY", "age_minutes": None}
            continue
        # Get latest timestamp
        ts_col = next((c for c in ["scrape_timestamp","timestamp","datetime"]
                       if c in df.columns), None)
        if ts_col:
            latest = pd.to_datetime(df[ts_col].iloc[-1])
            age_min = (datetime.now() - latest).seconds // 60
            if age_min < FRESHNESS_MINUTES:
                status = "FRESH"
            elif age_min < 60:
                status = "WARNING"
            else:
                status = "STALE"
        else:
            mtime    = os.path.getmtime(fpath)
            age_min  = (datetime.now().timestamp() - mtime) // 60
            status   = "FRESH" if age_min < FRESHNESS_MINUTES else "STALE"

        freshness[name] = {
            "status":      status,
            "age_minutes": int(age_min),
        }
    return freshness

def raise_alert(alert_type, message, severity="WARNING"):
    """Log an alert"""
    row = {
        "timestamp": datetime.now().isoformat(),
        "type":      alert_type,
        "severity":  severity,
        "message":   message,
    }
    df = pd.DataFrame([row])
    if os.path.exists(ALERTS_LOG):
        df.to_csv(ALERTS_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(ALERTS_LOG, index=False)
    print(f"  🚨 ALERT [{severity}] {alert_type}: {message}")

def run_all_checks():
    """Run complete monitoring cycle"""
    print("\n" + "="*50)
    print("MONITORING CHECK")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    results = {"timestamp": datetime.now().isoformat()}

    # 1. Data freshness
    print("\n[1] Data Freshness...")
    freshness = check_data_freshness()
    for name, info in freshness.items():
        status = info["status"]
        age    = info.get("age_minutes","N/A")
        icon   = "✅" if status=="FRESH" else "⚠️" if status=="WARNING" else "❌"
        print(f"  {icon} {name}: {status} ({age} min old)")
        if status == "STALE":
            raise_alert("DATA_FRESHNESS", f"{name} data is {age} min old", "WARNING")
    results["freshness"] = freshness

    # 2. Data drift
    print("\n[2] Data Drift...")
    drift = check_data_drift()
    for metric, info in drift.items():
        icon = "❌" if info["alert"] else "✅"
        print(f"  {icon} {metric}: {info['drift_pct']:.1f}% drift "
              f"(hist: {info['historical_mean']} → live: {info['live_mean']})")
        if info["alert"]:
            raise_alert("DATA_DRIFT",
                        f"{metric} drifted {info['drift_pct']:.1f}% from historical mean",
                        "CRITICAL")
    results["drift"] = drift

    # 3. Rolling metrics
    print("\n[3] Rolling Performance Metrics...")
    metrics = compute_rolling_metrics(window_hours=24)
    if metrics:
        print(f"  Rolling MAPE (24h): {metrics['rolling_mape']:.2f}%")
        print(f"  Rolling RMSE (24h): {metrics['rolling_rmse']:.2f} Rs/MWh")
        print(f"  Based on {metrics['n_predictions']} predictions")
        if metrics["rolling_mape"] > MAPE_ALERT_THRESHOLD:
            raise_alert("PERFORMANCE_DEGRADATION",
                        f"Rolling MAPE {metrics['rolling_mape']:.2f}% exceeds {MAPE_ALERT_THRESHOLD}%",
                        "WARNING")
        if metrics["rolling_mape"] > RETRAIN_MAPE_TRIGGER:
            raise_alert("RETRAIN_REQUIRED",
                        f"MAPE {metrics['rolling_mape']:.2f}% exceeds retrain threshold {RETRAIN_MAPE_TRIGGER}%",
                        "CRITICAL")
        results["rolling_metrics"] = metrics
    else:
        print("  No prediction history yet")

    # Save monitor log
    with open(MONITOR_LOG, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nMonitoring complete ✅")
    return results

if __name__ == "__main__":
    run_all_checks()
