"""
fetch_historical_weather.py
Fetches historical weather for 8 Indian cities via NASA POWER API
- If no data exists: fetches last 3 years up to yesterday
- If data exists: fetches only missing cities (gap fill)
- Used by retrain pipeline automatically
NASA POWER: FREE, no API key, government data
Group 05 - ISB AMPBA
"""

import requests
import pandas as pd
import os, time
from datetime import datetime, timedelta

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE      = os.path.join(BASE_DIR, "data", "weather_historical.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, "data", "weather_fetch_progress.csv")
NASA_URL      = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_PARAMS   = "T2M,WS10M,RH2M,CLOUD_AMT,PS"

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

def get_date_range():
    end   = datetime.now().date() - timedelta(days=1)
    start = end - timedelta(days=3*365)
    return start, end

def load_progress():
    """Load set of already-fetched cities"""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    try:
        df = pd.read_csv(PROGRESS_FILE)
        # Handle both old and new column formats
        col = df.columns[0]
        return set(df[col].tolist())
    except:
        return set()

def save_progress(city):
    done = load_progress()
    done.add(city)
    pd.DataFrame({"city": sorted(done)}).to_csv(PROGRESS_FILE, index=False)

def get_missing_cities():
    done = load_progress()
    # If existing weather data covers different date range, refetch all
    start, end = get_date_range()
    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE, parse_dates=["datetime"])
        if len(df) > 0:
            existing_start = df["datetime"].min().date()
            existing_end   = df["datetime"].max().date()
            # If date range shifted by more than 30 days, refetch
            if abs((existing_start - start).days) > 30:
                print(f"  Date range shifted — refetching all cities")
                return list(CITIES.keys())
    return [c for c in CITIES if c not in done]

def fetch_city(city_name, lat, lon, start, end):
    """Fetch full date range for one city from NASA POWER"""
    print(f"  Fetching {city_name} ({start} to {end})...")
    params = {
        "parameters": NASA_PARAMS,
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      start.strftime("%Y%m%d"),
        "end":        end.strftime("%Y%m%d"),
        "format":     "JSON",
    }
    try:
        resp = requests.get(NASA_URL, params=params, timeout=120)
        if resp.status_code != 200:
            print(f"    Error {resp.status_code}: {resp.text[:100]}")
            return None

        props      = resp.json()["properties"]["parameter"]
        timestamps = list(props["T2M"].keys())
        records    = []

        for ts in timestamps:
            try:
                dt = pd.to_datetime(ts, format="%Y%m%d%H")
                def clean(v): return None if (v is None or v == -999) else v
                records.append({
                    "datetime":    dt,
                    "city":        city_name,
                    "temperature": clean(props["T2M"].get(ts)),
                    "wind_speed":  clean(props["WS10M"].get(ts)),
                    "humidity":    clean(props["RH2M"].get(ts)),
                    "cloud_cover": clean(props["CLOUD_AMT"].get(ts)),
                    "pressure":    clean(props["PS"].get(ts)),
                })
            except: pass

        df = pd.DataFrame(records)
        df["cooling_degree"] = df["temperature"].apply(
            lambda t: max(t-25,0) if pd.notna(t) else 0)
        df["low_wind_flag"]  = (df["wind_speed"] < 2.0).astype(int)
        print(f"    {city_name}: {len(df):,} records ✅")
        return df

    except Exception as e:
        print(f"    {city_name} error: {e}")
        return None

def fetch_all(force_cities=None):
    start, end = get_date_range()
    print("="*55)
    print("HISTORICAL WEATHER — NASA POWER API (FREE)")
    print(f"Range: {start} to {end}")
    print(f"Cities: {list(CITIES.keys())}")
    print("="*55)

    cities_todo = force_cities if force_cities else get_missing_cities()
    all_records = []

    # Load existing data
    if os.path.exists(OUT_FILE):
        existing = pd.read_csv(OUT_FILE, parse_dates=["datetime"])
        # Remove cities being refetched
        if cities_todo:
            existing = existing[~existing["city"].isin(cities_todo)]
        if len(existing) > 0:
            all_records.append(existing)
            print(f"Loaded {len(existing):,} existing records")

    if not cities_todo:
        print("All cities already fetched! ✅")
        return

    print(f"Cities to fetch: {cities_todo}\n")

    for city_name in cities_todo:
        coords = CITIES[city_name]
        df = fetch_city(city_name, coords["lat"], coords["lon"], start, end)

        if df is not None and len(df) > 0:
            all_records.append(df)
            save_progress(city_name)

            # Save checkpoint after each city
            combined = pd.concat(all_records, ignore_index=True)
            combined = combined.drop_duplicates(subset=["datetime","city"])
            combined.to_csv(OUT_FILE, index=False)
            print(f"    Checkpoint: {len(combined):,} total records saved\n")

        time.sleep(2)

    # Final save
    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        combined = combined.drop_duplicates(subset=["datetime","city"])
        combined = combined.sort_values(["city","datetime"]).reset_index(drop=True)
        combined.to_csv(OUT_FILE, index=False)
        print("\n" + "="*55)
        print(f"COMPLETE!")
        print(f"Total records : {len(combined):,}")
        print(f"Cities        : {combined['city'].nunique()}/8")
        print(f"Period        : {combined['datetime'].min()} to {combined['datetime'].max()}")

if __name__ == "__main__":
    fetch_all()
