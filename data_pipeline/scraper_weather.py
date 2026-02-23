"""
scraper_weather.py
Fetches live weather data from OpenWeatherMap API every 15 minutes
8 major Indian cities as committed in mid-review
Group 05 - ISB AMPBA
"""

import requests
import pandas as pd
import os
from datetime import datetime

API_KEY = "6a4654121b86dec0ccc0f8a20961401c"
DATA_FILE = "data/weather_live.csv"

CITIES = {
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Pune":      {"lat": 18.5204, "lon": 73.8567},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
}

def fetch_weather(city_name):
    """Fetch current weather for one city"""
    city = CITIES[city_name]
    url  = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat":   city["lat"],
        "lon":   city["lon"],
        "appid": API_KEY,
        "units": "metric"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            d = resp.json()
            return {
                "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "city":         city_name,
                "temperature":  d["main"]["temp"],
                "feels_like":   d["main"]["feels_like"],
                "humidity":     d["main"]["humidity"],
                "pressure":     d["main"]["pressure"],
                "wind_speed":   d["wind"]["speed"],
                "wind_deg":     d["wind"].get("deg", 0),
                "cloud_cover":  d["clouds"]["all"],
                "weather_main": d["weather"][0]["main"],
                "visibility":   d.get("visibility", 0),
                # Derived features (committed in mid-review)
                "cooling_degree":   max(d["main"]["temp"] - 25, 0),
                "low_wind_flag":    1 if d["wind"]["speed"] < 2.0 else 0,
            }
        else:
            print(f"  Weather API error {city_name}: {resp.status_code}")
            return None
    except Exception as e:
        print(f"  Weather fetch error {city_name}: {e}")
        return None

def scrape_weather_all_cities():
    """Fetch weather for all 8 cities"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching weather...")
    records = []
    for city in CITIES:
        record = fetch_weather(city)
        if record:
            records.append(record)
            print(f"  {city}: {record['temperature']}°C | "
                  f"Wind: {record['wind_speed']}m/s | "
                  f"Humidity: {record['humidity']}%")
    return records

def save_weather_data(records):
    """Append new records to CSV"""
    if not records:
        return False
    os.makedirs("data", exist_ok=True)
    df_new = pd.DataFrame(records)
    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(DATA_FILE, index=False)
    print(f"  Saved {len(records)} city records → {DATA_FILE}")
    return True

def get_latest_weather(city="Delhi"):
    """Get most recent weather for a city"""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        city_df = df[df["city"] == city]
        if len(city_df) > 0:
            return city_df.iloc[-1].to_dict()
    return None

if __name__ == "__main__":
    records = scrape_weather_all_cities()
    if records:
        save_weather_data(records)
        print(f"Weather scrape successful! {len(records)} cities")
    else:
        print("Weather scrape failed")
