"""
validator.py
Schema validation, cap checks, dataset versioning
Covers: Data Validation & Versioning box from slide
Group 05 - ISB AMPBA
"""

import pandas as pd
import numpy as np
import os, json, hashlib
from datetime import datetime

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
VERSIONS_FILE = os.path.join(DATA_DIR,  "dataset_versions.json")
SCHEMA_LOG    = os.path.join(DATA_DIR,  "schema_validation_log.csv")

# ── Expected Schema ──────────────────────────────────────────
IEX_SCHEMA = {
    "MCP":              {"type": float, "min": 0,    "max": 20000, "required": True},
    "purchase_bid_mw":  {"type": float, "min": 0,    "max": 200000,"required": False},
    "sell_bid_mw":      {"type": float, "min": 0,    "max": 200000,"required": False},
    "mcv_mw":           {"type": float, "min": 0,    "max": 200000,"required": False},
}
WEATHER_SCHEMA = {
    "temperature":  {"type": float, "min": -10, "max": 55,   "required": True},
    "wind_speed":   {"type": float, "min": 0,   "max": 50,   "required": True},
    "humidity":     {"type": float, "min": 0,   "max": 100,  "required": True},
    "cloud_cover":  {"type": float, "min": 0,   "max": 100,  "required": False},
    "pressure":     {"type": float, "min": 800, "max": 1100, "required": False},
}
COMMODITY_SCHEMA = {
    "crude_oil_usd":    {"type": float, "min": 0,  "max": 300,  "required": True},
    "natural_gas_usd":  {"type": float, "min": 0,  "max": 50,   "required": True},
    "usd_inr":          {"type": float, "min": 50, "max": 150,  "required": True},
}

def validate_dataframe(df, schema, source_name):
    """Validate a dataframe against schema rules"""
    issues   = []
    warnings = []

    for col, rules in schema.items():
        # Check required columns
        if rules["required"] and col not in df.columns:
            issues.append(f"MISSING required column: {col}")
            continue
        if col not in df.columns:
            continue

        series = df[col]

        # Check missing values
        n_missing = series.isna().sum()
        if n_missing > 0:
            pct = n_missing / len(df) * 100
            if pct > 20:
                issues.append(f"{col}: {pct:.1f}% missing (>{20}% threshold)")
            else:
                warnings.append(f"{col}: {n_missing} missing values ({pct:.1f}%)")

        # Check value ranges (cap checks)
        valid = series.dropna()
        n_below = (valid < rules["min"]).sum()
        n_above = (valid > rules["max"]).sum()
        if n_below > 0:
            issues.append(f"{col}: {n_below} values below min ({rules['min']})")
        if n_above > 0:
            issues.append(f"{col}: {n_above} values above max ({rules['max']})")

    # Check timestamp alignment
    if "datetime" in df.columns or df.index.name == "datetime":
        idx = pd.to_datetime(df.index if df.index.name == "datetime" else df["datetime"])
        n_missing_ts = idx.isna().sum()
        if n_missing_ts > 0:
            issues.append(f"Missing timestamps: {n_missing_ts}")
        # Check for duplicates
        n_dupes = idx.duplicated().sum()
        if n_dupes > 0:
            warnings.append(f"Duplicate timestamps: {n_dupes}")

    result = {
        "source":    source_name,
        "records":   len(df),
        "issues":    issues,
        "warnings":  warnings,
        "passed":    len(issues) == 0,
        "timestamp": datetime.now().isoformat(),
    }

    # Log result
    log_validation(result)

    status = "✅ PASSED" if result["passed"] else "❌ FAILED"
    print(f"  Schema validation {source_name}: {status}")
    if issues:
        for i in issues:   print(f"    ❌ {i}")
    if warnings:
        for w in warnings: print(f"    ⚠️  {w}")

    return result

def log_validation(result):
    """Append validation result to log CSV"""
    row = {
        "timestamp":  result["timestamp"],
        "source":     result["source"],
        "records":    result["records"],
        "passed":     result["passed"],
        "n_issues":   len(result["issues"]),
        "n_warnings": len(result["warnings"]),
        "issues":     " | ".join(result["issues"]),
    }
    df = pd.DataFrame([row])
    if os.path.exists(SCHEMA_LOG):
        df.to_csv(SCHEMA_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(SCHEMA_LOG, index=False)

def validate_all_sources():
    """Run validation on all data sources"""
    print("\nRunning schema validation on all sources...")
    results = []
    checks = [
        ("iex_historical.csv",    IEX_SCHEMA,       "IEX"),
        ("weather_historical.csv",WEATHER_SCHEMA,    "Weather"),
        ("commodities_historical.csv", COMMODITY_SCHEMA, "Commodities"),
    ]
    for fname, schema, name in checks:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  {name}: file not found, skipping")
            continue
        df = pd.read_csv(fpath)
        results.append(validate_dataframe(df, schema, name))
    return results

# ── Dataset Versioning ────────────────────────────────────────

def get_file_hash(filepath):
    """MD5 hash of file for change detection"""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_versions():
    if os.path.exists(VERSIONS_FILE):
        with open(VERSIONS_FILE) as f:
            return json.load(f)
    return {}

def save_version(dataset_name, filepath, n_records, notes=""):
    """Record a new dataset version"""
    versions = load_versions()
    if dataset_name not in versions:
        versions[dataset_name] = []

    file_hash = get_file_hash(filepath) if os.path.exists(filepath) else "N/A"
    version_num = len(versions[dataset_name]) + 1

    entry = {
        "version":    version_num,
        "timestamp":  datetime.now().isoformat(),
        "records":    n_records,
        "hash":       file_hash,
        "notes":      notes,
    }
    versions[dataset_name].append(entry)
    with open(VERSIONS_FILE, "w") as f:
        json.dump(versions, f, indent=2)
    print(f"  Dataset version saved: {dataset_name} v{version_num} ({n_records:,} records)")
    return version_num

def get_latest_version(dataset_name):
    versions = load_versions()
    if dataset_name in versions and versions[dataset_name]:
        return versions[dataset_name][-1]
    return None

if __name__ == "__main__":
    validate_all_sources()
