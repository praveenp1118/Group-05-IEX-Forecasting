"""
preprocessor.py
Cleans and preprocesses incoming live data
Runs automatically after each scrape cycle
Group 05 - ISB AMPBA
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# File paths
RAW_IEX         = "data/iex_live.csv"
RAW_WEATHER     = "data/weather_live.csv"
RAW_RENEWABLE   = "data/renewable_live.csv"
RAW_COMMODITIES = "data/commodities_live.csv"
CLEAN_MERGED    = "data/merged_clean.csv"

def clean_iex(df):
    """Clean IEX price data"""
    if df is None or len(df) == 0:
        return df
    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"])
    # Remove invalid prices
    if "MCP" in df.columns:
        df = df[df["MCP"].notna()]
        df = df[(df["MCP"] > 0) & (df["MCP"] <= 10000)]
    # Sort by time
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def clean_weather(df):
    """Clean weather data"""
    if df is None or len(df) == 0:
        return df
    df = df.drop_duplicates(subset=["timestamp", "city"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Validate temperature range for India
    if "temperature" in df.columns:
        df = df[(df["temperature"] >= 0) & (df["temperature"] <= 55)]
    df["cooling_degree"] = df["temperature"].apply(lambda t: max(t - 25, 0))
    df["low_wind_flag"]  = (df["wind_speed"] < 2.0).astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def merge_latest(iex_df, weather_df, renewable_df, commodities_df):
    """Merge all sources on timestamp for the latest records"""
    try:
        # Get Delhi weather as primary weather source
        if weather_df is not None and len(weather_df) > 0:
            delhi_weather = weather_df[weather_df["city"] == "Delhi"].copy()
            delhi_weather = delhi_weather.rename(columns={
                "temperature": "temp_delhi",
                "humidity":    "humidity_delhi",
                "wind_speed":  "wind_delhi",
            })
        else:
            delhi_weather = pd.DataFrame()

        if iex_df is None or len(iex_df) == 0:
            print("  No IEX data available yet")
            return None

        merged = iex_df.copy()
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])

        # Merge weather
        if len(delhi_weather) > 0:
            delhi_weather["timestamp"] = pd.to_datetime(delhi_weather["timestamp"])
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                delhi_weather[["timestamp","temp_delhi","humidity_delhi",
                               "wind_delhi","cooling_degree","low_wind_flag"]].sort_values("timestamp"),
                on="timestamp", direction="nearest", tolerance=pd.Timedelta("1h")
            )

        # Merge commodities
        if commodities_df is not None and len(commodities_df) > 0:
            commodities_df["timestamp"] = pd.to_datetime(commodities_df["timestamp"])
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                commodities_df[["timestamp","crude_oil_usd","natural_gas_usd","usd_inr"]].sort_values("timestamp"),
                on="timestamp", direction="nearest", tolerance=pd.Timedelta("24h")
            )

        return merged

    except Exception as e:
        print(f"  Merge error: {e}")
        return iex_df  # Return just IEX if merge fails

def run_preprocessing():
    """Main preprocessing pipeline"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running preprocessor...")

    # Load all sources
    iex_df         = pd.read_csv(RAW_IEX)         if os.path.exists(RAW_IEX)         else None
    weather_df     = pd.read_csv(RAW_WEATHER)     if os.path.exists(RAW_WEATHER)     else None
    renewable_df   = pd.read_csv(RAW_RENEWABLE)   if os.path.exists(RAW_RENEWABLE)   else None
    commodities_df = pd.read_csv(RAW_COMMODITIES) if os.path.exists(RAW_COMMODITIES) else None

    # Clean each source
    iex_df     = clean_iex(iex_df)
    weather_df = clean_weather(weather_df)

    # Merge
    merged = merge_latest(iex_df, weather_df, renewable_df, commodities_df)

    if merged is not None and len(merged) > 0:
        merged.to_csv(CLEAN_MERGED, index=False)
        print(f"  Clean merged data saved → {CLEAN_MERGED} ({len(merged)} records)")
        return merged
    else:
        print("  No data to merge yet")
        return None

def get_data_status():
    """Return status of all data sources for /data/latest endpoint"""
    status = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {}
    }

    for name, filepath in [
        ("IEX",         RAW_IEX),
        ("Weather",     RAW_WEATHER),
        ("Renewable",   RAW_RENEWABLE),
        ("Commodities", RAW_COMMODITIES),
    ]:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if len(df) > 0:
                last_row = df.iloc[-1].to_dict()
                status["sources"][name] = {
                    "status":        "live",
                    "records":       len(df),
                    "last_timestamp": str(last_row.get("timestamp","N/A")),
                    "latest_data":   {k: v for k, v in last_row.items()
                                      if k != "timestamp" and str(v) != "nan"}
                }
            else:
                status["sources"][name] = {"status": "empty", "records": 0}
        else:
            status["sources"][name] = {"status": "not started", "records": 0}

    return status

if __name__ == "__main__":
    run_preprocessing()
    print(get_data_status())
