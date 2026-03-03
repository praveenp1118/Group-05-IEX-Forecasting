"""
scheduler.py - Auto-refresh live data every 30 minutes
+ Auto-generate predictions so monitoring stays current
Place in: D:\Group-05-IEX-Forecasting\data_pipeline\scheduler.py
"""
import threading, time, os, sys, subprocess, requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

REFRESH_INTERVAL = 30 * 60   # 30 minutes
FLASK_URL        = "http://localhost:5000"

# ── Find actual scraper file ──────────────────────────────────
def find_scraper(candidates):
    """Return first existing script from candidate list"""
    for name in candidates:
        path = os.path.join(BASE_DIR, "data_pipeline", name)
        if os.path.exists(path):
            return path
    return None

def run_script(path, label):
    if not path:
        print(f"  ⚠️  {label}: scraper not found")
        return False
    try:
        result = subprocess.run(
            ["python", path],
            timeout=180, capture_output=True,
            text=True, cwd=BASE_DIR
        )
        if result.returncode == 0:
            print(f"  ✅ {label} done")
            return True
        else:
            print(f"  ❌ {label}: {result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {label} timed out"); return False
    except Exception as e:
        print(f"  ❌ {label}: {e}"); return False

def refresh_commodities():
    """
    Run live commodities scraper (fetch_live_commodities.py).
    Falls back to fetch_historical_commodities.py if not found.
    """
    com = find_scraper([
        "fetch_live_commodities.py",        # NEW — always try this first
        "fetch_historical_commodities.py",  # legacy fallback
        "scraper_commodities.py",
    ])
    return run_script(com, "Commodities")

def auto_predict():
    """
    Hit /forecast/24h so prediction_log.csv gets fresh entries.
    This keeps the monitoring endpoint's 'Last 30 Predictions' current.
    Retries up to 3 times if Flask is temporarily busy.
    """
    for attempt in range(1, 4):
        try:
            resp = requests.get(f"{FLASK_URL}/forecast/24h", timeout=45)
            if resp.status_code == 200:
                print(f"  ✅ Auto-prediction logged (96 blocks)")
                return True
            else:
                print(f"  ⚠️  Auto-predict HTTP {resp.status_code} (attempt {attempt})")
        except requests.exceptions.ConnectionError:
            print(f"  ⚠️  Flask not reachable yet (attempt {attempt}) — retrying in 15s")
            time.sleep(15)
        except Exception as e:
            print(f"  ❌ Auto-predict error: {e}")
            break
    print("  ❌ Auto-prediction failed after 3 attempts")
    return False

def refresh_all():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Refreshing live data...")

    # ── 1. IEX scraper ────────────────────────────────────────
    iex = find_scraper(["scraper_iex.py", "iex_scraper.py", "fetch_historical_iex.py"])
    run_script(iex, "IEX")

    # ── 2. Weather scraper ────────────────────────────────────
    wx = find_scraper(["scraper_weather.py", "fetch_historical_weather.py"])
    run_script(wx, "Weather")

    # ── 3. Commodities (live prices, not historical) ──────────
    refresh_commodities()

    # ── 4. Sync historical → live files ──────────────────────
    try:
        sync_script = os.path.join(BASE_DIR, "data_pipeline", "sync_live_files.py")
        if os.path.exists(sync_script):
            subprocess.run(["python", sync_script], timeout=30,
                           capture_output=True, cwd=BASE_DIR)
            print("  ✅ Live files synced")
    except Exception as e:
        print(f"  Sync error: {e}")

    # ── 5. Backfill actual MCP into prediction log ────────────
    try:
        backfill_script = os.path.join(BASE_DIR, "data_pipeline", "backfill_actuals.py")
        if os.path.exists(backfill_script):
            subprocess.run(["python", backfill_script], timeout=30,
                           capture_output=True, cwd=BASE_DIR)
            print("  ✅ Actuals backfilled")
    except Exception as e:
        print(f"  Backfill error: {e}")

    # ── 6. Auto-generate predictions (keeps monitoring fresh) ─
    auto_predict()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Refresh complete\n")

def scheduler_loop():
    print(f"[Scheduler] First refresh in 60s, then every 30min")

    # Show what scrapers were found on startup
    for label, candidates in [
        ("IEX",         ["scraper_iex.py", "iex_scraper.py", "iex_rtm_scraper.py"]),
        ("Weather",     ["scraper_weather.py", "fetch_live_weather.py"]),
        ("Commodities", ["fetch_live_commodities.py", "fetch_historical_commodities.py",
                         "scraper_commodities.py"]),
    ]:
        found = find_scraper(candidates)
        print(f"  {label}: {os.path.basename(found) if found else 'NOT FOUND'}")

    time.sleep(60)
    while True:
        try:
            refresh_all()
        except Exception as e:
            print(f"[Scheduler] Error: {e}")
        time.sleep(REFRESH_INTERVAL)

def start_scheduler():
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    print("✅ Scheduler started — live data refreshes every 30min + auto-prediction")
    return t

if __name__ == "__main__":
    refresh_all()
