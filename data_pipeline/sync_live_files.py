"""
sync_live_files.py
Copies MOST RECENT records from historical CSVs -> live CSVs
Sorts by date before taking tail — so live file always has newest data
Place in: D:\\Group-05-IEX-Forecasting\\data_pipeline\\sync_live_files.py
"""
import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def sync_iex(n_rows=300):
    """Sync IEX — sort by date+time_block before taking tail"""
    hist_path = os.path.join(DATA_DIR, "iex_historical.csv")
    live_path = os.path.join(DATA_DIR, "iex_live.csv")

    if not os.path.exists(hist_path):
        print("  IEX: iex_historical.csv not found"); return

    df = pd.read_csv(hist_path)
    if len(df) == 0:
        print("  IEX: empty"); return

    # Parse date properly and sort — CRITICAL FIX
    df["date_p"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df = df.sort_values(["date_p", "time_block"]).drop(columns=["date_p"])

    # Add scrape timestamp
    df["scrape_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # If live file already exists and has newer data (from scraper), merge it
    if os.path.exists(live_path):
        try:
            live_df = pd.read_csv(live_path)
            # Check if live has newer dates than historical
            live_df["date_p"] = pd.to_datetime(live_df["date"], format="%d-%m-%Y", errors="coerce")
            hist_max = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce").max()
            live_max = live_df["date_p"].max()
            if live_max > hist_max:
                # Live has newer data — merge and deduplicate
                live_df = live_df.drop(columns=["date_p"])
                combined = pd.concat([df.tail(n_rows), live_df], ignore_index=True)
                combined["date_p"] = pd.to_datetime(combined["date"], format="%d-%m-%Y", errors="coerce")
                combined = combined.sort_values(["date_p","time_block"]).drop(columns=["date_p"])
                combined = combined.drop_duplicates(subset=["date","time_block"], keep="last")
                combined = combined.tail(n_rows)
                combined.to_csv(live_path, index=False)
                print(f"  IEX: merged {len(combined)} rows (live had newer data up to {live_max.strftime('%d-%m-%Y')}) -> iex_live.csv ✅")
                return
        except: pass

    # Default — save most recent n_rows from historical
    recent = df.tail(n_rows)
    recent.to_csv(live_path, index=False)
    max_date = pd.to_datetime(recent["date"], format="%d-%m-%Y", errors="coerce").max().strftime("%d-%m-%Y")
    print(f"  IEX: synced {len(recent)} rows (most recent: {max_date}) -> iex_live.csv ✅")

def sync_sorted(historical_file, live_file, date_col, n_rows=200, label=""):
    """Generic sync with date sorting"""
    hist_path = os.path.join(DATA_DIR, historical_file)
    live_path = os.path.join(DATA_DIR, live_file)

    if not os.path.exists(hist_path):
        print(f"  {label}: {historical_file} not found"); return

    df = pd.read_csv(hist_path)
    if len(df) == 0:
        print(f"  {label}: empty"); return

    # Sort by date
    if date_col in df.columns:
        df["_date_p"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values("_date_p").drop(columns=["_date_p"])

    df["scrape_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.tail(n_rows).to_csv(live_path, index=False)
    print(f"  {label}: synced {min(n_rows, len(df))} rows -> {live_file} ✅")

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Syncing live files...")
    sync_iex(300)
    sync_sorted("commodities_historical.csv", "commodities_live.csv", "Date", 30,  "Commodities")
    sync_sorted("weather_historical.csv",     "weather_live.csv",     "datetime", 100, "Weather")
    print("Done ✅")
