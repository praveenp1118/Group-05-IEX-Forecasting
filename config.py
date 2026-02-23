"""
config.py
Project Configuration - Group 05 IEX Forecasting
ISB AMPBA - Final Submission
"""

# ── OpenWeatherMap API ────────────────────────────────────────
OPENWEATHER_API_KEY = "6a4654121b86dec0ccc0f8a20961401c"

# 8 major Indian cities as committed in mid-review
CITIES = {
    'Delhi':     {'lat': 28.6139, 'lon': 77.2090},
    'Mumbai':    {'lat': 19.0760, 'lon': 72.8777},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
    'Chennai':   {'lat': 13.0827, 'lon': 80.2707},
    'Kolkata':   {'lat': 22.5726, 'lon': 88.3639},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
    'Pune':      {'lat': 18.5204, 'lon': 73.8567},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
}

# Primary city for weather features (IEX headquartered in Delhi)
PRIMARY_CITY = 'Delhi'

# ── Data Settings ─────────────────────────────────────────────
DATA_FILE        = "data/Price.xlsx"
RESAMPLE_FREQ    = "15T"   # 15-minute intervals
TEST_SIZE        = 0.2
RANDOM_STATE     = 42

# ── Model Settings ────────────────────────────────────────────
ARIMA_ORDER      = (2, 1, 2)   # (p, d, q)
XGBOOST_PARAMS   = {
    'n_estimators':  300,
    'learning_rate': 0.05,
    'max_depth':     6,
    'subsample':     0.8,
    'random_state':  42
}
SVM_PARAMS = {
    'kernel': 'rbf',
    'C':      100,
    'epsilon': 0.1
}

# ── Success Criteria (from mid-review) ───────────────────────
TARGET_MAPE             = 5.0    # ≤5% MAPE
TARGET_IMPROVEMENT      = 25.0   # ≥25% over baseline
CONFIDENCE_THRESHOLD    = 0.80   # 80% signal confidence

# ── Flask API ─────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000
