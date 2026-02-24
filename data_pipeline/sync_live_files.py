"""
sync_live_files.py
Copies latest records from historical CSVs → live CSVs
Run after each scrape to keep live files fresh
Place in: D:\Group-05-IEX-Forecasting\data_pipeline\sync_live_files.py
"""
import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def sync(historical_file, live_file, n_rows=200, label=""):
    hist_path = os.path.join(DATA_DIR, historical_file)
    live_path = os.path.join(DATA_DIR, live_file)

    if not os.path.exists(hist_path):
        print(f"  {label}: {historical_file} not found"); return

    df = pd.read_csv(hist_path)
    if len(df) == 0:
        print(f"  {label}: empty"); return

    # Add/update scrape_timestamp
    df["scrape_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save latest n rows to live file
    df.tail(n_rows).to_csv(live_path, index=False)
    print(f"  {label}: synced {min(n_rows, len(df))} rows → {live_file} ✅")

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Syncing live files...")
    sync("iex_historical.csv",         "iex_live.csv",         200, "IEX")
    sync("commodities_historical.csv", "commodities_live.csv", 30,  "Commodities")
    sync("weather_historical.csv",     "weather_live.csv",     100, "Weather")
    print("Done ✅")
