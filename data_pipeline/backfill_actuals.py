"""
backfill_actuals.py — Fills actual_mcp into prediction_log.csv
by matching each prediction timestamp to the IEX historical/live data.

How it works:
  1. Reads prediction_log.csv
  2. For each row where actual_mcp is null:
       - Rounds the prediction timestamp down to the nearest 15-min block
       - Looks up that date + time_block in iex_historical.csv and iex_live.csv
       - Fills in the actual MCP if found
  3. Saves updated prediction_log.csv

Run:  python data_pipeline/backfill_actuals.py
Also called automatically from scheduler.py after every refresh.

Place at: D:\\Group-05-IEX-Forecasting\\data_pipeline\\backfill_actuals.py
"""

import os, sys
import pandas as pd
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────
_THIS  = os.path.abspath(__file__)
_BASE  = os.path.dirname(os.path.dirname(_THIS))
DATA_DIR = os.path.join(_BASE, "data")
PRED_LOG = os.path.join(DATA_DIR, "prediction_log.csv")

def build_iex_lookup():
    """
    Build a dict: 'DD-MM-YYYY|HH:MM-HH:MM' -> MCP float
    from both iex_historical.csv and iex_live.csv
    """
    lookup = {}
    for fname in ["iex_historical.csv", "iex_live.csv"]:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath).dropna(subset=["MCP"])
            for _, row in df.iterrows():
                key = f"{row['date']}|{row['time_block']}"
                # Live file wins over historical for same key
                lookup[key] = float(row["MCP"])
        except Exception as e:
            print(f"  Warning: could not read {fname}: {e}")
    return lookup

def ts_to_key(ts):
    """
    Convert a prediction timestamp to IEX lookup key.
    e.g. '2026-02-24 12:01:00' -> '24-02-2026|12:00-12:15'
    """
    try:
        t = pd.Timestamp(ts)
        block_start = t.floor("15min")
        block_end   = block_start + pd.Timedelta(minutes=15)
        date_str    = block_start.strftime("%d-%m-%Y")
        time_str    = f"{block_start.strftime('%H:%M')}-{block_end.strftime('%H:%M')}"
        return f"{date_str}|{time_str}"
    except:
        return None

def backfill():
    if not os.path.exists(PRED_LOG):
        print("prediction_log.csv not found — nothing to backfill")
        return 0

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Backfilling actuals...")

    # Load prediction log
    df = pd.read_csv(PRED_LOG)
    if "actual_mcp" not in df.columns:
        df["actual_mcp"] = None

    # Find rows needing backfill
    needs_fill = df["actual_mcp"].isna()
    n_needed   = needs_fill.sum()
    if n_needed == 0:
        print("  All rows already have actuals — nothing to do")
        return 0

    print(f"  {n_needed} rows need actuals | building IEX lookup...")

    # Build lookup from IEX data
    lookup = build_iex_lookup()
    print(f"  IEX lookup built: {len(lookup):,} time blocks available")

    # Fill actuals
    filled = 0
    for idx in df[needs_fill].index:
        ts  = df.at[idx, "timestamp"]
        key = ts_to_key(ts)
        if key and key in lookup:
            df.at[idx, "actual_mcp"] = round(lookup[key], 2)
            filled += 1

    # Save back
    df.to_csv(PRED_LOG, index=False)

    # Report
    total     = len(df)
    now_null  = df["actual_mcp"].isna().sum()
    now_filled= total - now_null

    # Compute rolling MAPE on filled rows
    has_both = df.dropna(subset=["actual_mcp", "predicted_mcp"])
    if len(has_both) > 0:
        mape = (abs(has_both["actual_mcp"] - has_both["predicted_mcp"]) /
                (has_both["actual_mcp"] + 1) * 100).mean()
        print(f"  Filled {filled} new rows | Total with actuals: {now_filled}/{total}")
        print(f"  Rolling MAPE (all filled): {mape:.2f}%")
    else:
        print(f"  Filled {filled} new rows | Still no actuals available in IEX data")

    return filled

if __name__ == "__main__":
    n = backfill()
    print(f"  Done — {n} rows backfilled")
