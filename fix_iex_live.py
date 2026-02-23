"""
fix_iex_live.py
Run once to clean iex_live.csv, then this logic is built into the scraper
Place in: D:\Group-05-IEX-Forecasting\
Run: python fix_iex_live.py
"""
import pandas as pd, os

BASE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE, "data", "iex_live.csv")

if not os.path.exists(path):
    print("iex_live.csv not found"); exit()

df = pd.read_csv(path)
before = len(df)

# Remove rows with no MCP (junk rows from old scraper format)
df = df.dropna(subset=["MCP"])

# Remove rows with no scrape_timestamp
if "scrape_timestamp" in df.columns:
    df = df.dropna(subset=["scrape_timestamp"])

# Remove the old-format timestamp column if it's mostly empty
if "timestamp" in df.columns and df["timestamp"].notna().mean() < 0.05:
    df = df.drop(columns=["timestamp"])
    print("  Dropped mostly-empty 'timestamp' column")

df.to_csv(path, index=False)
print(f"Cleaned iex_live.csv: {before} → {len(df)} rows")
print(f"Last scrape: {df['scrape_timestamp'].iloc[-1]}")
print("Done ✅")
