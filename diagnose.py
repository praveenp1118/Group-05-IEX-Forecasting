"""
diagnose.py - Run from D:\Group-05-IEX-Forecasting\
python diagnose.py
"""
import pandas as pd
import numpy as np
import os, json, pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER   = os.path.join(BASE_DIR, "data", "master_training_data.csv")
MODELS   = os.path.join(BASE_DIR, "models")

print("="*55)
print("FULL DIAGNOSIS")
print("="*55)

df = pd.read_csv(MASTER, index_col=0, parse_dates=True)
mcp = df["target_mcp"]

print(f"\n--- BASIC ---")
print(f"Records   : {len(df):,}")
print(f"Period    : {df.index[0]} → {df.index[-1]}")
print(f"Features  : {len(df.columns)}")

print(f"\n--- TARGET MCP DISTRIBUTION ---")
print(f"Mean      : {mcp.mean():.2f}")
print(f"Median    : {mcp.median():.2f}")
print(f"Std       : {mcp.std():.2f}")
print(f"Min       : {mcp.min():.2f}")
print(f"Max       : {mcp.max():.2f}")
print(f"= 10000   : {(mcp==10000).sum():,} ({(mcp==10000).mean()*100:.1f}%)")
print(f"> 9000    : {(mcp>9000).sum():,}  ({(mcp>9000).mean()*100:.1f}%)")
print(f"< 1000    : {(mcp<1000).sum():,}  ({(mcp<1000).mean()*100:.1f}%)")

print(f"\n--- LAG FEATURE QUALITY ---")
for col in ["mcp_lag_1h","mcp_lag_24h","mcp_lag_48h"]:
    if col in df.columns:
        corr   = df["target_mcp"].corr(df[col])
        n_zero = (df[col] == 0).sum()
        n_same = (df[col] == df["target_mcp"]).sum()
        print(f"  {col}:")
        print(f"    corr with target : {corr:.4f}  ← should be >0.85")
        print(f"    zero values      : {n_zero}")
        print(f"    identical to MCP : {n_same} ← bad if high")

print(f"\n--- YEAR BREAKDOWN ---")
df["year"] = df.index.year
yr = df.groupby("year")["target_mcp"].agg(["mean","count","std"])
print(yr.to_string())

print(f"\n--- MCP = 10000 BY MONTH ---")
df["month"] = df.index.month
cap_by_month = df.groupby("month").apply(lambda x: (x["target_mcp"]==10000).mean()*100).round(1)
print(cap_by_month.to_string())

print(f"\n--- TRAIN/TEST SPLIT PREVIEW ---")
split = int(len(df)*0.8)
train_mcp = df["target_mcp"].iloc[:split]
test_mcp  = df["target_mcp"].iloc[split:]
print(f"Train: {len(train_mcp):,} records | mean={train_mcp.mean():.0f} | "
      f"10k%={(train_mcp==10000).mean()*100:.1f}%")
print(f"Test : {len(test_mcp):,} records  | mean={test_mcp.mean():.0f}  | "
      f"10k%={(test_mcp==10000).mean()*100:.1f}%")

print(f"\n--- FEATURE IMPORTANCE ---")
fi_file = os.path.join(MODELS, "feature_importance.json")
if os.path.exists(fi_file):
    with open(fi_file) as f:
        fi = json.load(f)
    for k,v in list(fi.items())[:8]:
        print(f"  {k}: {v:.4f}")

print(f"\n--- MODEL RESULTS ---")
comp = os.path.join(MODELS,"model_comparison.csv")
if os.path.exists(comp):
    print(pd.read_csv(comp)[["model","mape","rmse","cv_mape_mean"]].to_string())

print(f"\n--- SCALER CHECK ---")
sc_file = os.path.join(MODELS,"scaler.pkl")
if os.path.exists(sc_file):
    sc = pickle.load(open(sc_file,"rb"))
    print(f"  Scaler type     : {type(sc).__name__}")
    print(f"  n_features_in   : {sc.n_features_in_}")
    print(f"  mean range      : {sc.mean_.min():.2f} – {sc.mean_.max():.2f}")

print("\nDone.")
