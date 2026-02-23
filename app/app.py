"""
app.py
Phase 5 - Deployment (CRISP-ML(Q))
Flask REST API for MCP Price Prediction
Group 05 - ISB AMPBA Final Submission
"""

from flask import Flask, request, jsonify
import pickle, numpy as np, os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_HOST, API_PORT

app = Flask(__name__)

def load_artifacts():
    with open('models/best_model.pkl','rb')    as f: model    = pickle.load(f)
    with open('models/scaler.pkl','rb')        as f: scaler   = pickle.load(f)
    with open('models/feature_cols.pkl','rb')  as f: features = pickle.load(f)
    return model, scaler, features

try:
    model, scaler, feature_cols = load_artifacts()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model load error: {e}")
    model = scaler = feature_cols = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "project": "IEX Electricity Price Forecasting",
        "group":   "Group 05 — ISB AMPBA",
        "status":  "running",
        "endpoints": {"/predict":"POST", "/predict/sample":"GET", "/health":"GET"}
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status":"healthy","model_loaded": model is not None,
                    "timestamp": datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict 15-minute Market Clearing Price (MCP)

    Input JSON:
    {
        "mcp_lag_1h": 2313.77,    "mcp_lag_2h": 2333.89,
        "mcp_lag_24h": 2328.34,   "mcp_lag_48h": 2285.62,
        "system_demand": 5257.03, "demand_lag_2h": 4851.9,
        "demand_lag_24h": 5257.03,"demand_lag_48h": 4574.5,
        "hour": 14,   "day_of_week": 0, "month": 6, "is_weekend": 0,
        "temperature": 35.0,  "humidity": 60.0,
        "wind_speed": 3.5,    "season": 2
    }
    """
    if model is None:
        return jsonify({"error":"Model not loaded"}), 500
    try:
        d = request.get_json()

        # Build all features the model expects
        mcp1  = float(d.get('mcp_lag_1h',2300))
        mcp2  = float(d.get('mcp_lag_2h',2300))
        mcp24 = float(d.get('mcp_lag_24h',2300))
        mcp48 = float(d.get('mcp_lag_48h',2300))
        dem1  = float(d.get('system_demand',5000))
        dem2  = float(d.get('demand_lag_2h',5000))
        dem24 = float(d.get('demand_lag_24h',5000))
        dem48 = float(d.get('demand_lag_48h',5000))
        hour  = float(d.get('hour',12))
        dow   = float(d.get('day_of_week',0))
        month = float(d.get('month',6))
        wknd  = float(d.get('is_weekend',0))
        temp  = float(d.get('temperature',30))
        hum   = float(d.get('humidity',60))
        wind  = float(d.get('wind_speed',3))

        # Engineered features
        feat_vals = {
            'mcp_lag_1h':mcp1,'mcp_lag_2h':mcp2,'mcp_lag_24h':mcp24,'mcp_lag_48h':mcp48,
            'system_demand':dem1,'demand_lag_2h':dem2,'demand_lag_24h':dem24,'demand_lag_48h':dem48,
            'hour':hour,'day_of_week':dow,'month':month,'is_weekend':wknd,
            'price_change_1h':mcp1-mcp2,'price_change_24h':mcp24-mcp48,
            'price_rolling_24h':(mcp1+mcp2+mcp24+mcp48)/4,
            'price_volatility': abs(mcp1-mcp24),
            'demand_change':dem1-dem24,
            'load_price_ratio':dem1/(mcp1+1),
            'coal_price':3000+(mcp1+mcp2+mcp24+mcp48)/4*0.05,
            'temperature':temp,'cooling_degree':max(temp-25,0),
            'wind_speed':wind,'low_wind_flag':int(wind<2),
            'humidity':hum,'solar_irradiance':max(0,np.sin(np.pi*(hour-6)/12))*700,
            'trend':(mcp1+mcp2+mcp24+mcp48)/4,
            'seasonality':mcp1-(mcp1+mcp2+mcp24+mcp48)/4,
        }

        row = np.array([[feat_vals.get(c, 0) for c in feature_cols]])
        row_scaled = scaler.transform(row)
        prediction = float(model.predict(row_scaled)[0])

        chg = (prediction - mcp1) / (mcp1 + 1) * 100
        signal = ("BUY — price rising" if chg > 5 else
                  "SELL — price falling" if chg < -5 else "HOLD — stable")

        return jsonify({
            "predicted_MCP_Rs_per_MWh": round(prediction, 2),
            "previous_MCP":             round(mcp1, 2),
            "expected_change_pct":      round(chg, 2),
            "trading_signal":           signal,
            "forecast_horizon":         "15 minutes",
            "timestamp":                datetime.now().isoformat(),
            "model":                    "Group 05 — XGBoost"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/sample', methods=['GET'])
def sample():
    """Demo prediction with sample IEX data"""
    with app.test_client() as c:
        resp = c.post('/predict', json={
            "mcp_lag_1h":2313.77,"mcp_lag_2h":2333.89,
            "mcp_lag_24h":2328.34,"mcp_lag_48h":2285.62,
            "system_demand":5257.03,"demand_lag_2h":4851.9,
            "demand_lag_24h":5257.03,"demand_lag_48h":4574.5,
            "hour":14,"day_of_week":1,"month":6,"is_weekend":0,
            "temperature":36.0,"humidity":55.0,"wind_speed":2.5
        })
        import json
        return app.response_class(resp.data, mimetype='application/json')

if __name__ == '__main__':
    print("Starting IEX Price Forecasting API — Group 05")
    print(f"URL: http://{API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=False)
