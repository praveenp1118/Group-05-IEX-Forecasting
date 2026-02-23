"""
scheduler.py
Runs all data scrapers on schedule:
- IEX + Weather: every 15 minutes
- Renewable + Commodities: every 24 hours
- Preprocessor: after every scrape cycle
Group 05 - ISB AMPBA
"""

import schedule
import time
import threading
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.scraper_iex import scrape_iex, save_iex_data
from data_pipeline.scraper_weather import scrape_weather_all_cities, save_weather_data
from data_pipeline.scraper_posoco import scrape_renewable, save_renewable, scrape_commodities, save_commodities
from data_pipeline.preprocessor import run_preprocessing

def run_15min_pipeline():
    """Runs every 15 minutes — IEX + Weather"""
    print(f"\n{'='*50}")
    print(f"15-MIN PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # IEX scrape
    try:
        iex_record = scrape_iex()
        save_iex_data(iex_record)
    except Exception as e:
        print(f"IEX scraper error: {e}")

    # Weather scrape
    try:
        weather_records = scrape_weather_all_cities()
        save_weather_data(weather_records)
    except Exception as e:
        print(f"Weather scraper error: {e}")

    # Preprocess after every scrape
    try:
        run_preprocessing()
    except Exception as e:
        print(f"Preprocessor error: {e}")

    print(f"15-min cycle complete — next run in 15 mins")

def run_daily_pipeline():
    """Runs once daily — Renewable + Commodities"""
    print(f"\n{'='*50}")
    print(f"DAILY PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    try:
        r = scrape_renewable(); save_renewable(r)
    except Exception as e:
        print(f"Renewable scraper error: {e}")

    try:
        c = scrape_commodities(); save_commodities(c)
    except Exception as e:
        print(f"Commodities scraper error: {e}")

def start_scheduler():
    """Start the scheduler in background thread"""
    print("Starting data pipeline scheduler...")
    print("  15-min jobs: IEX + Weather + Preprocessor")
    print("  Daily jobs : Renewable + Commodities")

    # Run immediately on start
    threading.Thread(target=run_15min_pipeline, daemon=True).start()
    threading.Thread(target=run_daily_pipeline, daemon=True).start()

    # Schedule recurring jobs
    schedule.every(15).minutes.do(
        lambda: threading.Thread(target=run_15min_pipeline, daemon=True).start()
    )
    schedule.every(24).hours.do(
        lambda: threading.Thread(target=run_daily_pipeline, daemon=True).start()
    )

    # Run scheduler in background thread
    def run_schedule():
        while True:
            schedule.run_pending()
            time.sleep(30)

    scheduler_thread = threading.Thread(target=run_schedule, daemon=True)
    scheduler_thread.start()
    print("Scheduler started — running in background")
    return scheduler_thread

if __name__ == "__main__":
    # Standalone mode — keeps running
    start_scheduler()
    print("Scheduler running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Scheduler stopped.")
