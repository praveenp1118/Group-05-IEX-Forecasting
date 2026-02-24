"""
config.py - Central config, reads from .env
Place in: D:\Group-05-IEX-Forecasting\config.py
"""
import os
from pathlib import Path

# Load .env file if it exists
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

load_env()

# API Keys — read from environment, never hardcoded
OPENWEATHER_API_KEY  = os.getenv("OPENWEATHER_API_KEY", "")
IEX_API_KEY          = os.getenv("IEX_API_KEY", "")
AWS_ACCESS_KEY_ID    = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY= os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION           = os.getenv("AWS_REGION", "ap-south-1")
FLASK_SECRET_KEY     = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-prod")

# Paths
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
