"""
fetch_historical_commodities.py
Fetches historical commodity prices
- If no data exists: fetches last 3 years up to yesterday
- If data exists: fetches only missing dates (gap fill)
- Used by retrain pipeline automatically
Group 05 - ISB AMPBA
"""

import pandas as pd
import requests
import os, time
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE = os.path.join(BASE_DIR, "data", "commodities_historical.csv")

def get_date_range():
    """Dynamic: yesterday back 3 years"""
    end   = datetime.now().date() - timedelta(days=1)
    start = end - timedelta(days=3*365)
    return start, end

def get_missing_dates():
    """Return only dates not already in CSV"""
    start, end = get_date_range()
    all_dates  = pd.date_range(start=start, end=end, freq="D")
    if os.path.exists(OUT_FILE):
        existing = pd.read_csv(OUT_FILE, parse_dates=["date"])
        existing_dates = set(existing["date"].dt.date.tolist())
        missing = [d.date() for d in all_dates if d.date() not in existing_dates]
    else:
        missing = [d.date() for d in all_dates]
    return missing, start, end

def fetch_yahoo_history(ticker, name, start, end):
    """Fetch via yfinance"""
    print(f"  Fetching {name} ({ticker})...")
    try:
        import yfinance as yf
        df = yf.download(ticker,
                         start=str(start), end=str(end),
                         progress=False, auto_adjust=True)
        if len(df) > 0:
            result = df[["Close"]].copy()
            result.columns = [name]
            result.index.name = "date"
            print(f"    {name}: {len(result):,} records ✅")
            return result
        return None
    except Exception as e:
        print(f"    {name} error: {e}")
        return None

def fetch_usd_inr(start, end):
    """Fetch USD/INR via frankfurter.app"""
    print("  Fetching USD/INR...")
    try:
        url  = f"https://api.frankfurter.app/{start}..{end}?from=USD&to=INR"
        resp = requests.get(url, timeout=15)
        data = resp.json()
        records = [{"date": d, "usd_inr": r["INR"]}
                   for d, r in data["rates"].items()]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        full_idx = pd.date_range(str(start), str(end), freq="D")
        df = df.reindex(full_idx).ffill().bfill()
        df.index.name = "date"
        print(f"    USD/INR: {len(df):,} records ✅")
        return df
    except Exception as e:
        print(f"    USD/INR error: {e} — using proxy")
        dates = pd.date_range(str(start), str(end), freq="D")
        df = pd.DataFrame({"usd_inr": [83.5]*len(dates)}, index=dates)
        df.index.name = "date"
        return df

def fetch_all():
    missing, start, end = get_missing_dates()
    print("="*55)
    print("HISTORICAL COMMODITIES — Yahoo Finance + Frankfurter")
    print(f"Range: Last 3 years → {start} to {end}")
    print(f"Missing dates: {len(missing)}")
    print("="*55)

    if not missing:
        print("All dates already fetched!")
        return

    all_dfs = []
    for name, ticker in [("crude_oil_usd","CL=F"), ("natural_gas_usd","NG=F")]:
        df = fetch_yahoo_history(ticker, name, start, end)
        if df is not None:
            all_dfs.append(df)
        time.sleep(1)

    usd_df = fetch_usd_inr(start, end)
    if usd_df is not None:
        all_dfs.append(usd_df)

    if not all_dfs:
        print("No data fetched — using proxy")
        dates  = pd.date_range(str(start), str(end), freq="D")
        merged = pd.DataFrame({
            "crude_oil_usd":   [70.0]*len(dates),
            "natural_gas_usd": [3.5] *len(dates),
            "usd_inr":         [83.5]*len(dates),
        }, index=dates)
    else:
        merged = all_dfs[0]
        for df in all_dfs[1:]:
            merged = merged.join(df, how="outer")
        merged = merged.sort_index().ffill().bfill()

    merged["coal_price_proxy"] = (merged["crude_oil_usd"] * 1.8).round(2)
    merged.index = pd.to_datetime(merged.index)
    merged.index.name = "date"

    # Merge with existing data
    if os.path.exists(OUT_FILE):
        existing = pd.read_csv(OUT_FILE, parse_dates=["date"], index_col="date")
        merged   = pd.concat([existing, merged])
        merged   = merged[~merged.index.duplicated(keep="last")]
        merged   = merged.sort_index()

    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    merged.to_csv(OUT_FILE)
    print(f"\nSaved {len(merged):,} records → {OUT_FILE}")
    return merged

if __name__ == "__main__":
    fetch_all()
