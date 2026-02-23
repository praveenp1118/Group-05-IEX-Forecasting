"""
merge_historical.py - FINAL FIX
All rolling features computed on shifted MCP (no leakage)
Group 05 - ISB AMPBA
"""

import pandas as pd
import numpy as np
import os, sys
from datetime import datetime, timedelta

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICE_XLSX     = os.path.join(BASE_DIR, "data", "Price.xlsx")
IEX_HIST       = os.path.join(BASE_DIR, "data", "iex_historical.csv")
WEATHER_HIST   = os.path.join(BASE_DIR, "data", "weather_historical.csv")
COMMODITY_HIST = os.path.join(BASE_DIR, "data", "commodities_historical.csv")
IEX_LIVE       = os.path.join(BASE_DIR, "data", "iex_live.csv")
OUTPUT_FILE    = os.path.join(BASE_DIR, "data", "master_training_data.csv")

sys.path.insert(0, BASE_DIR)

# ══════════════════════════════════════════════════════════════
# MISSING DATA FILLER
# ══════════════════════════════════════════════════════════════

def fill_missing(df, cols, group_keys, label=""):
    for col in cols:
        if col not in df.columns: continue
        n = df[col].isna().sum()
        if n == 0: continue
        try:
            df[col] = df[col].fillna(df.groupby(group_keys)[col].transform("mean"))
        except: pass
        if len(group_keys) > 1:
            try:
                df[col] = df[col].fillna(df.groupby(group_keys[1:])[col].transform("mean"))
            except: pass
        df[col] = df[col].interpolate(limit=24).ffill().bfill().fillna(df[col].mean())
        filled = n - df[col].isna().sum()
        if filled > 0:
            print(f"    {label}{col}: {n} missing → filled {filled}")
    return df

# ══════════════════════════════════════════════════════════════
# AUTO FETCH
# ══════════════════════════════════════════════════════════════

def ensure_fresh_data():
    print("Checking data sources...")
    try:
        from data_pipeline.fetch_historical_weather import fetch_all as fw, get_missing_cities
        missing = get_missing_cities()
        if missing: fw(force_cities=missing)
        else: print("  Weather: up to date ✅")
    except Exception as e: print(f"  Weather: {e}")
    try:
        from data_pipeline.fetch_historical_commodities import fetch_all as fc, get_missing_dates
        missing, _, _ = get_missing_dates()
        if missing: fc()
        else: print("  Commodities: up to date ✅")
    except Exception as e: print(f"  Commodities: {e}")

# ══════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════

def load_iex_scraped():
    dfs = []
    for filepath, label in [(IEX_HIST,"historical"),(IEX_LIVE,"live")]:
        if not os.path.exists(filepath): continue
        try:
            df = pd.read_csv(filepath)
            try:
                df["datetime"] = pd.to_datetime(
                    df["date"] + " " + df["time_block"].str[:5],
                    format="%d-%m-%Y %H:%M", errors="coerce")
            except:
                df["datetime"] = pd.to_datetime(
                    df.get("scrape_timestamp", df.get("timestamp","")), errors="coerce")
            df = df.dropna(subset=["datetime","MCP"])
            df = df[(df["MCP"] > 0) & (df["MCP"] <= 20000)]
            keep = ["datetime","MCP","purchase_bid_mw","sell_bid_mw","mcv_mw","scheduled_vol_mw"]
            df   = df[[c for c in keep if c in df.columns]]
            dfs.append(df)
            print(f"  IEX {label}: {len(df):,} records")
        except Exception as e:
            print(f"  IEX {label} error: {e}")
    if not dfs: return None
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"])
    combined = combined.sort_values("datetime").set_index("datetime")
    print(f"  IEX combined: {len(combined):,} | {combined.index[0].date()} → {combined.index[-1].date()}")
    return combined

def load_price_xlsx_for_gaps(start, end):
    if not os.path.exists(PRICE_XLSX): return None
    try:
        df = pd.read_excel(PRICE_XLSX)
        df.index = pd.date_range(start="2021-01-01", periods=len(df), freq="h")
        df.index.name = "datetime"
        df = df.rename(columns={"P(T)":"MCP","L(T-1)":"system_demand"})
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        if len(df) == 0: return None
        df_15 = df[["MCP","system_demand"]].resample("15min").ffill()
        print(f"  Price.xlsx gap fill: {len(df_15):,} records")
        return df_15
    except Exception as e:
        print(f"  Price.xlsx error: {e}"); return None

def load_weather(start, end):
    if not os.path.exists(WEATHER_HIST): return None
    df    = pd.read_csv(WEATHER_HIST, parse_dates=["datetime"])
    delhi = df[df["city"]=="Delhi"].copy().set_index("datetime").drop(columns=["city"])
    delhi = delhi[(delhi.index >= pd.Timestamp(start)) & (delhi.index <= pd.Timestamp(end))]
    delhi = delhi.rename(columns={
        "temperature":"temp_delhi","humidity":"humidity_delhi",
        "wind_speed":"wind_delhi","cloud_cover":"cloud_delhi","pressure":"pressure_delhi"})
    delhi["_h"] = delhi.index.hour
    delhi["_m"] = delhi.index.month
    wx = ["temp_delhi","humidity_delhi","wind_delhi","cloud_delhi","pressure_delhi"]
    delhi = fill_missing(delhi, wx, ["_h","_m"], "wx.")
    delhi.drop(columns=["_h","_m"], inplace=True, errors="ignore")
    delhi["cooling_degree"] = delhi["temp_delhi"].apply(lambda t: max(t-25,0) if pd.notna(t) else 0)
    delhi["low_wind_flag"]  = (delhi["wind_delhi"] < 2.0).astype(int)
    delhi = delhi.resample("15min").ffill()
    print(f"  Weather: {len(delhi):,} records ✅")
    return delhi

def load_commodities(start, end):
    if not os.path.exists(COMMODITY_HIST): return None
    df = pd.read_csv(COMMODITY_HIST, parse_dates=["date"], index_col="date")
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    df["_m"] = df.index.month; df["_y"] = df.index.year
    cols = [c for c in df.columns if c not in ["_m","_y"]]
    df = fill_missing(df, cols, ["_y","_m"], "com.")
    df.drop(columns=["_m","_y"], inplace=True, errors="ignore")
    df = df.resample("15min").ffill()
    print(f"  Commodities: {len(df):,} records ✅")
    return df

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — ALL LEAKAGE FREE
# ══════════════════════════════════════════════════════════════

def add_features(df):
    """
    CRITICAL RULE: Every feature must use ONLY past MCP values
    - All rolling stats computed on df["MCP"].shift(1) — excludes current value
    - Lag features use shift(4), shift(96) etc — all past only
    - NO seasonality = MCP - trend (was leaking current MCP)
    - NO rolling mean/std on current MCP (was leaking)
    """
    df = df.sort_index()

    # Shifted MCP series — use this for ALL rolling calculations
    mcp_shifted = df["MCP"].shift(1)   # shift by 1 block = exclude current value

    # Lag features (pure past values)
    df["mcp_lag_1h"]   = df["MCP"].shift(4)    # 1 hour ago
    df["mcp_lag_2h"]   = df["MCP"].shift(8)    # 2 hours ago
    df["mcp_lag_24h"]  = df["MCP"].shift(96)   # 24 hours ago
    df["mcp_lag_48h"]  = df["MCP"].shift(192)  # 48 hours ago
    df["mcp_lag_1w"]   = df["MCP"].shift(96*7) # 1 week ago

    # Rolling stats on SHIFTED series — no leakage
    df["price_rolling_24h"] = mcp_shifted.rolling(96,   min_periods=1).mean()
    df["price_rolling_1w"]  = mcp_shifted.rolling(96*7, min_periods=1).mean()
    df["price_volatility"]  = mcp_shifted.rolling(96,   min_periods=1).std().fillna(0)
    df["price_rolling_min"] = mcp_shifted.rolling(96,   min_periods=1).min()
    df["price_rolling_max"] = mcp_shifted.rolling(96,   min_periods=1).max()

    # Price dynamics (lag vs lag — no leakage)
    df["price_change_1h"]   = df["mcp_lag_1h"]  - df["mcp_lag_2h"]
    df["price_change_24h"]  = df["mcp_lag_24h"] - df["mcp_lag_48h"]
    df["price_momentum"]    = df["mcp_lag_1h"]  - df["price_rolling_24h"]

    # Temporal features
    df["hour"]         = df.index.hour
    df["day_of_week"]  = df.index.dayofweek
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["is_weekend"]   = (df.index.dayofweek >= 5).astype(int)
    df["season"]       = df["month"].map({
        12:1,1:1,2:1, 3:2,4:2,5:2, 6:3,7:3,8:3, 9:4,10:4,11:4})
    # Hour buckets (peak/off-peak)
    df["hour_bucket"]  = pd.cut(df.index.hour,
                                bins=[-1,5,9,17,21,23],
                                labels=[0,1,2,3,4]).astype(int)

    # Fuel proxy (no MCP involved — safe)
    if "crude_oil_usd" in df.columns and "natural_gas_usd" in df.columns:
        coal = df.get("coal_price_proxy", df["crude_oil_usd"]*1.8)
        df["coal_price"] = (df["crude_oil_usd"]*1.8).round(2)
        df["fuel_proxy"] = (df["crude_oil_usd"]*0.4 +
                            df["natural_gas_usd"]*10 +
                            coal*0.3).round(2)
    else:
        df["coal_price"] = 3000
        df["fuel_proxy"] = 3000

    # Demand features (no current MCP)
    if "system_demand" in df.columns:
        df["demand_lag_24h"]   = df["system_demand"].shift(96)
        df["demand_change"]    = df["system_demand"] - df["demand_lag_24h"]
        df["load_price_ratio"] = df["system_demand"] / (df["mcp_lag_1h"] + 1)
    elif "mcv_mw" in df.columns:
        df["system_demand"]    = df["mcv_mw"] * 1.2
        df["demand_lag_24h"]   = df["system_demand"].shift(96)
        df["demand_change"]    = df["system_demand"] - df["demand_lag_24h"]
        df["load_price_ratio"] = df["system_demand"] / (df["mcp_lag_1h"] + 1)

    df = df.rename(columns={"MCP": "target_mcp"})
    return df

# ══════════════════════════════════════════════════════════════
# MAIN MERGE
# ══════════════════════════════════════════════════════════════

def merge_all(auto_fetch=True):
    print("="*55)
    print("MERGING ALL HISTORICAL DATA")
    print("Leakage fix: ALL rolling on shifted MCP")
    print("="*55 + "\n")

    try:
        from data_pipeline.validator import validate_all_sources
        validate_all_sources()
    except Exception as e:
        print(f"Validator skipped: {e}")

    if auto_fetch:
        ensure_fresh_data()

    print("\nLoading IEX scraped data...")
    iex = load_iex_scraped()
    if iex is None:
        print("ERROR: No scraped IEX data"); return None

    data_start = iex.index.min()
    data_end   = iex.index.max()
    print(f"Date range locked: {data_start.date()} → {data_end.date()}")

    # Gap fill with xlsx (same range only)
    print("\nLoading Price.xlsx for gap fill...")
    xlsx = load_price_xlsx_for_gaps(data_start, data_end)

    # Build full 15-min index from scraped range
    print("\nBuilding full 15-min index...")
    full_idx   = pd.date_range(data_start, data_end, freq="15min")
    merged     = iex.reindex(full_idx)
    missing_n  = merged["MCP"].isna().sum()
    print(f"  Missing slots: {missing_n:,} ({missing_n/len(full_idx)*100:.1f}%)")

    # Fill gaps with xlsx first
    if xlsx is not None and missing_n > 0:
        xlsx_r = xlsx.reindex(full_idx)
        merged["MCP"] = merged["MCP"].combine_first(xlsx_r["MCP"])
        if "system_demand" in xlsx_r.columns:
            merged["system_demand"] = xlsx_r["system_demand"]
        filled = missing_n - merged["MCP"].isna().sum()
        print(f"  Gaps filled by xlsx: {filled:,}")

    # Fill remaining with historical average
    merged["_h"] = merged.index.hour
    merged["_d"] = merged.index.dayofweek
    merged["_m"] = merged.index.month
    merged = fill_missing(merged, ["MCP"], ["_h","_d","_m"], "iex.")
    merged.drop(columns=["_h","_d","_m"], inplace=True, errors="ignore")

    # Weather
    wx = load_weather(data_start, data_end)
    if wx is not None:
        merged = merged.join(wx, how="left")
        merged["_h"] = merged.index.hour
        merged["_m"] = merged.index.month
        merged = fill_missing(merged,
            ["temp_delhi","humidity_delhi","wind_delhi","cloud_delhi","cooling_degree"],
            ["_h","_m"], "wx.")
        merged.drop(columns=["_h","_m"], inplace=True, errors="ignore")
    else:
        doy = merged.index.dayofyear
        merged["temp_delhi"]     = 25 + 10*np.sin(2*np.pi*doy/365)
        merged["humidity_delhi"] = 60
        merged["wind_delhi"]     = 3.0
        merged["cooling_degree"] = merged["temp_delhi"].apply(lambda t: max(t-25,0))
        merged["low_wind_flag"]  = 0

    # Commodities
    com = load_commodities(data_start, data_end)
    if com is not None:
        merged = merged.join(com, how="left")
        merged["_m"] = merged.index.month
        merged["_y"] = merged.index.year
        merged = fill_missing(merged,
            ["crude_oil_usd","natural_gas_usd","usd_inr","coal_price_proxy"],
            ["_y","_m"], "com.")
        merged.drop(columns=["_m","_y"], inplace=True, errors="ignore")

    # Feature engineering (leakage-free)
    print("\nAdding leakage-free features...")
    merged = add_features(merged)

    # Fill lag NaN at start of series
    lag_cols = ["mcp_lag_1h","mcp_lag_2h","mcp_lag_24h","mcp_lag_48h","mcp_lag_1w",
                "price_rolling_24h","price_rolling_1w","price_volatility",
                "price_rolling_min","price_rolling_max"]
    for col in lag_cols:
        if col in merged.columns:
            merged[col] = merged[col].bfill().ffill().fillna(merged["target_mcp"].mean())

    merged = merged.ffill().bfill()

    # Remove invalid
    before = len(merged)
    merged = merged.dropna(subset=["target_mcp"])
    merged = merged[(merged["target_mcp"] > 0) & (merged["target_mcp"] <= 20000)]
    print(f"Removed {before-len(merged):,} invalid rows")
    print(f"Remaining NaN: {merged.isna().sum().sum()}")

    # Quick leakage check
    corr_lag1 = merged["target_mcp"].corr(merged["mcp_lag_1h"])
    corr_roll = merged["target_mcp"].corr(merged["price_rolling_24h"])
    print(f"\nLeakage check (should both be <0.95):")
    print(f"  mcp_lag_1h corr       : {corr_lag1:.4f}")
    print(f"  price_rolling_24h corr: {corr_roll:.4f}")
    if corr_lag1 > 0.95 or corr_roll > 0.95:
        print("  ⚠️  WARNING: correlation suspiciously high — check for leakage")
    else:
        print("  ✅ No obvious leakage detected")

    merged.to_csv(OUTPUT_FILE)

    try:
        from data_pipeline.validator import save_version
        save_version("master_training_data", OUTPUT_FILE, len(merged),
                     notes=f"Leakage-fixed {datetime.now().strftime('%Y-%m-%d')}")
    except: pass

    print(f"\n{'='*55}")
    print(f"MASTER DATA READY")
    print(f"Records  : {len(merged):,}")
    print(f"Features : {len(merged.columns)}")
    print(f"Period   : {merged.index[0].date()} → {merged.index[-1].date()}")
    return merged

if __name__ == "__main__":
    merge_all(auto_fetch=True)
