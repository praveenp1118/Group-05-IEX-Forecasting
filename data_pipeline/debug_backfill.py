"""
debug_backfill.py — Check why backfill matched 0 rows
docker exec group05-iex-api python3 data_pipeline/debug_backfill.py
"""
import os, sys, pandas as pd
from datetime import datetime

_BASE    = "/app"
DATA_DIR = "/app/data"
PRED_LOG = "/app/data/prediction_log.csv"

# Step 1 — show sample prediction timestamps
print("="*60)
print("STEP 1: Sample prediction timestamps")
df = pd.read_csv(PRED_LOG)
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample timestamps:\n{df['timestamp'].head(5).tolist()}")

# Step 2 — show what keys we generate from those timestamps
print("\nSTEP 2: Keys generated from timestamps")
def ts_to_key(ts):
    try:
        t = pd.Timestamp(ts)
        block_start = t.floor("15min")
        block_end   = block_start + pd.Timedelta(minutes=15)
        date_str    = block_start.strftime("%d-%m-%Y")
        time_str    = f"{block_start.strftime('%H:%M')}-{block_end.strftime('%H:%M')}"
        return f"{date_str}|{time_str}"
    except:
        return None

sample_keys = [ts_to_key(ts) for ts in df['timestamp'].head(5)]
print(f"Sample generated keys: {sample_keys}")

# Step 3 — show what keys exist in IEX data
print("\nSTEP 3: Sample keys in IEX data")
for fname in ["iex_historical.csv", "iex_live.csv"]:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        iex = pd.read_csv(fpath)
        iex['key'] = iex['date'] + '|' + iex['time_block']
        print(f"\n{fname}:")
        print(f"  Rows: {len(iex)}")
        print(f"  Date range: {iex['date'].min()} to {iex['date'].max()}")
        print(f"  Sample keys: {iex['key'].head(3).tolist()}")
        print(f"  Last 3 keys: {iex['key'].tail(3).tolist()}")

# Step 4 — try to manually match first prediction
print("\nSTEP 4: Manual match attempt")
first_ts  = df['timestamp'].iloc[0]
first_key = ts_to_key(first_ts)
print(f"First prediction ts: {first_ts}")
print(f"Generated key:       {first_key}")

iex_hist = pd.read_csv(os.path.join(DATA_DIR, "iex_historical.csv"))
iex_hist['key'] = iex_hist['date'] + '|' + iex_hist['time_block']
match = iex_hist[iex_hist['key'] == first_key]
print(f"Match in historical: {len(match)} rows")

iex_live = pd.read_csv(os.path.join(DATA_DIR, "iex_live.csv"))
iex_live['key'] = iex_live['date'] + '|' + iex_live['time_block']
match2 = iex_live[iex_live['key'] == first_key]
print(f"Match in live:       {len(match2)} rows")

# Step 5 — check date overlap
print("\nSTEP 5: Date overlap check")
pred_dates = pd.to_datetime(df['timestamp']).dt.strftime('%d-%m-%Y').unique()
print(f"Prediction dates (unique): {sorted(pred_dates)}")
hist_dates = iex_hist['date'].unique()
print(f"Historical dates available: {sorted(hist_dates)[-5:]} (last 5)")
live_dates = iex_live['date'].unique()
print(f"Live dates available: {sorted(live_dates)}")

overlap = set(pred_dates) & set(hist_dates) | set(pred_dates) & set(live_dates)
print(f"Overlapping dates: {overlap}")
