"""
scraper_posoco.py
Renewable + Commodities data collection
Group 05 - ISB AMPBA
"""

import requests
import pandas as pd
import os
from datetime import datetime
from bs4 import BeautifulSoup

DATA_FILE_RENEWABLE   = "data/renewable_live.csv"
DATA_FILE_COMMODITIES = "data/commodities_live.csv"

# ── Commodities via free APIs ─────────────────────────────────

def fetch_usd_inr():
    """Fetch USD/INR via free exchangerate API"""
    try:
        resp = requests.get(
            "https://api.exchangerate-api.com/v4/latest/USD",
            timeout=10
        )
        data = resp.json()
        return round(float(data["rates"]["INR"]), 2)
    except Exception as e:
        print(f"  USD/INR error: {e}")
        return None

def fetch_crude_oil():
    """Fetch crude oil price via commodities API"""
    try:
        # Use metals-api or open price source
        resp = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/CL=F",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        data = resp.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return round(float(price), 2)
    except:
        try:
            # Fallback: use open commodity price source
            resp = requests.get(
                "https://api.allorigins.win/get?url=" +
                "https://query1.finance.yahoo.com/v8/finance/chart/CL%3DF",
                timeout=10
            )
            import json
            content = json.loads(resp.json()["contents"])
            price = content["chart"]["result"][0]["meta"]["regularMarketPrice"]
            return round(float(price), 2)
        except Exception as e:
            print(f"  Crude oil error: {e}")
            return None

def fetch_natural_gas():
    """Fetch natural gas price"""
    try:
        resp = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/NG=F",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        data = resp.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return round(float(price), 2)
    except:
        try:
            resp = requests.get(
                "https://api.allorigins.win/get?url=" +
                "https://query1.finance.yahoo.com/v8/finance/chart/NG%3DF",
                timeout=10
            )
            import json
            content = json.loads(resp.json()["contents"])
            price = content["chart"]["result"][0]["meta"]["regularMarketPrice"]
            return round(float(price), 2)
        except Exception as e:
            print(f"  Natural gas error: {e}")
            return None

def scrape_commodities():
    """Fetch all commodity prices"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching commodities...")
    record = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    record["usd_inr"]          = fetch_usd_inr()
    record["crude_oil_usd"]    = fetch_crude_oil()
    record["natural_gas_usd"]  = fetch_natural_gas()

    # Coal price proxy — derive from crude oil (correlated)
    if record["crude_oil_usd"]:
        record["coal_price_proxy"] = round(record["crude_oil_usd"] * 1.8, 2)
    else:
        record["coal_price_proxy"] = None

    for k, v in record.items():
        if k != "timestamp":
            print(f"  {k}: {v}")

    return record

def save_commodities(record):
    if not record:
        return
    os.makedirs("data", exist_ok=True)
    df_new = pd.DataFrame([record])
    if os.path.exists(DATA_FILE_COMMODITIES):
        df = pd.concat([pd.read_csv(DATA_FILE_COMMODITIES), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(DATA_FILE_COMMODITIES, index=False)
    print(f"  Commodities saved → {DATA_FILE_COMMODITIES}")

# ── Renewable — derive from weather as proxy ──────────────────

def scrape_renewable():
    """
    Renewable generation proxy derived from weather data.
    POSOCO site uses JavaScript charts — not directly scrapable.
    We use Delhi weather to estimate solar/wind generation.
    This is the production architecture design.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Estimating renewable generation...")

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "solar_mw":   None,
        "wind_mw":    None,
        "hydro_mw":   None,
        "total_renewable_mw": None,
        "source": "weather_proxy"
    }

    try:
        # Use live weather to estimate renewable
        weather_file = "data/weather_live.csv"
        if os.path.exists(weather_file):
            df = pd.read_csv(weather_file)
            if len(df) > 0:
                # Average across all cities
                avg_cloud  = df["cloud_cover"].mean()  if "cloud_cover"  in df.columns else 50
                avg_wind   = df["wind_speed"].mean()   if "wind_speed"   in df.columns else 3
                avg_temp   = df["temperature"].mean()  if "temperature"  in df.columns else 30

                hour = datetime.now().hour
                # Solar estimate (MW) based on time of day and cloud cover
                solar_factor = max(0, __import__('math').sin(
                    __import__('math').pi * (hour - 6) / 12
                )) if 6 <= hour <= 18 else 0
                record["solar_mw"]  = round(solar_factor * (1 - avg_cloud/100) * 60000, 0)
                record["wind_mw"]   = round(avg_wind * 8000, 0)
                record["hydro_mw"]  = round(45000 + avg_temp * 100, 0)
                record["total_renewable_mw"] = round(
                    (record["solar_mw"] or 0) +
                    (record["wind_mw"]  or 0) +
                    (record["hydro_mw"] or 0), 0
                )
                print(f"  Solar: {record['solar_mw']} MW | "
                      f"Wind: {record['wind_mw']} MW | "
                      f"Hydro: {record['hydro_mw']} MW (weather proxy)")
    except Exception as e:
        print(f"  Renewable proxy error: {e}")

    return record

def save_renewable(record):
    if not record:
        return
    os.makedirs("data", exist_ok=True)
    df_new = pd.DataFrame([record])
    if os.path.exists(DATA_FILE_RENEWABLE):
        df = pd.concat([pd.read_csv(DATA_FILE_RENEWABLE), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(DATA_FILE_RENEWABLE, index=False)
    print(f"  Renewable saved → {DATA_FILE_RENEWABLE}")

if __name__ == "__main__":
    r = scrape_renewable();  save_renewable(r)
    c = scrape_commodities(); save_commodities(c)
    print("Done!")
