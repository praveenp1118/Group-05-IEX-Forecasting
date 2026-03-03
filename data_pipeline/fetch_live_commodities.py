"""
fetch_live_commodities.py - Live Commodities Scraper
Fetches latest crude oil, natural gas, USD/INR using yfinance + frankfurter
Replaces stale commodities_live.csv with today's data
Group 05 - ISB AMPBA
"""
import os, sys, requests
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_FILE = os.path.join(BASE_DIR, "data", "commodities_live.csv")
HIST_FILE = os.path.join(BASE_DIR, "data", "commodities_historical.csv")

def fetch_yfinance(ticker, label):
    """Fetch latest price from Yahoo Finance via yfinance library"""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if len(hist) > 0:
            price = float(hist["Close"].iloc[-1])
            date  = hist.index[-1].strftime("%Y-%m-%d")
            print(f"  {label}: {price:.4f} (yfinance, date={date})")
            return price, date
    except ImportError:
        pass  # yfinance not installed — fall back to requests
    except Exception as e:
        print(f"  {label} yfinance error: {e}")
    return None, None

def fetch_via_requests(ticker, label):
    """Fallback: fetch via Yahoo Finance v8 API (no library needed)"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1d", "range": "5d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data   = resp.json()
            closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            ts     = data["chart"]["result"][0]["timestamp"]
            # Get last non-None value
            for c, t in zip(reversed(closes), reversed(ts)):
                if c is not None:
                    date = datetime.fromtimestamp(t).strftime("%Y-%m-%d")
                    print(f"  {label}: {c:.4f} (requests, date={date})")
                    return float(c), date
    except Exception as e:
        print(f"  {label} requests error: {e}")
    return None, None

def get_price(ticker, label):
    """Try yfinance first, then requests fallback"""
    price, date = fetch_yfinance(ticker, label)
    if price is None:
        price, date = fetch_via_requests(ticker, label)
    return price, date

def fetch_usd_inr():
    """Fetch USD/INR from Frankfurter API (free, no key needed)"""
    try:
        resp = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": "INR"},
            timeout=10
        )
        if resp.status_code == 200:
            rate = resp.json()["rates"]["INR"]
            date = resp.json()["date"]
            print(f"  USD/INR: {rate:.2f} (frankfurter, date={date})")
            return float(rate), date
    except Exception as e:
        print(f"  USD/INR frankfurter error: {e}")

    # Fallback: Yahoo Finance
    return get_price("INR=X", "USD/INR (yahoo)")

def fetch_commodities():
    """Fetch all commodity prices"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching live commodities...")

    crude_price, crude_date   = get_price("BZ=F", "Crude Oil Brent (USD)")
    gas_price, gas_date       = get_price("NG=F", "Natural Gas (USD)")
    usd_inr, fx_date          = fetch_usd_inr()

    # Use most recent available date across sources
    today = datetime.now().strftime("%Y-%m-%d")
    dates = [d for d in [crude_date, gas_date, fx_date] if d]
    record_date = max(dates) if dates else today

    # Fallbacks if any fetch failed
    if crude_price is None:
        print("  Crude: using fallback Rs 70")
        crude_price = 70.0
    if gas_price is None:
        print("  Gas: using fallback 3.0")
        gas_price = 3.0
    if usd_inr is None:
        print("  USD/INR: using fallback 84")
        usd_inr = 84.0

    coal_price_proxy = crude_price * 1.8

    record = {
        "date":             record_date,
        "crude_oil_usd":    round(crude_price, 4),
        "natural_gas_usd":  round(gas_price,   4),
        "usd_inr":          round(usd_inr,      2),
        "coal_price_proxy": round(coal_price_proxy, 2),
        "scrape_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"  Record date: {record_date}")
    return record

def save_commodities(record):
    """Update live CSV with new record; append to historical"""
    os.makedirs(os.path.dirname(LIVE_FILE), exist_ok=True)
    df_new = pd.DataFrame([record])

    # ── Update live file ──────────────────────────────────────
    if os.path.exists(LIVE_FILE):
        try:
            existing = pd.read_csv(LIVE_FILE)
            # Remove any row with same date to avoid duplicates
            existing = existing[existing["date"] != record["date"]]
            combined = pd.concat([existing, df_new], ignore_index=True)
        except:
            combined = df_new
    else:
        combined = df_new

    # Sort by date, keep latest 30
    combined["_d"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values("_d").drop(columns=["_d"]).tail(30)
    combined.to_csv(LIVE_FILE, index=False)
    print(f"  Saved -> commodities_live.csv (date={record['date']}) ✅")

    # ── Append to historical ──────────────────────────────────
    if os.path.exists(HIST_FILE):
        try:
            hist = pd.read_csv(HIST_FILE)
            hist = hist[hist["date"] != record["date"]]  # dedup
            hist = pd.concat([hist, df_new], ignore_index=True)
            hist["_d"] = pd.to_datetime(hist["date"], errors="coerce")
            hist = hist.sort_values("_d").drop(columns=["_d"])
            hist.to_csv(HIST_FILE, index=False)
            print(f"  Appended -> commodities_historical.csv ✅")
        except Exception as e:
            print(f"  Historical append failed: {e}")

if __name__ == "__main__":
    record = fetch_commodities()
    save_commodities(record)
    print("Done ✅")
