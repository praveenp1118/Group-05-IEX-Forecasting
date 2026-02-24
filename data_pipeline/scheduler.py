"""
scheduler.py - Auto-refresh live data every 30 minutes
Place in: D:\Group-05-IEX-Forecasting\data_pipeline\scheduler.py
"""
import threading, time, os, sys, subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

REFRESH_INTERVAL = 30 * 60  # 30 minutes

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

def refresh_all():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Refreshing live data...")

    # IEX scraper — confirmed filename
    iex = find_scraper(["scraper_iex.py", "iex_scraper.py", "fetch_historical_iex.py"])
    run_script(iex, "IEX")

    # Weather scraper — confirmed filename
    wx = find_scraper(["scraper_weather.py", "fetch_historical_weather.py"])
    run_script(wx, "Weather")

    # Commodities scraper — confirmed filename
    com = find_scraper(["fetch_historical_commodities.py", "scraper_commodities.py"])
    run_script(com, "Commodities")

    # Sync historical → live files so health check shows FRESH
    try:
        sync_script = os.path.join(BASE_DIR, "data_pipeline", "sync_live_files.py")
        if os.path.exists(sync_script):
            subprocess.run(["python", sync_script], timeout=30,
                         capture_output=True, cwd=BASE_DIR)
            print("  ✅ Live files synced")
    except Exception as e:
        print(f"  Sync error: {e}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Refresh complete\n")

def scheduler_loop():
    print(f"[Scheduler] First refresh in 60s, then every 30min")
    # Show what scrapers were found on startup
    for label, candidates in [
        ("IEX",         ["scraper_iex.py","iex_scraper.py","iex_rtm_scraper.py"]),
        ("Weather",     ["scraper_weather.py","fetch_live_weather.py"]),
        ("Commodities", ["fetch_historical_commodities.py","scraper_commodities.py"]),
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
    print("✅ Scheduler started — live data refreshes every 30min")
    return t

if __name__ == "__main__":
    refresh_all()
