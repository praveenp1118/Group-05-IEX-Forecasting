"""
fetch_historical_iex.py
Fetches IEX historical 15-min RTM data — 1 day at a time
FROM = TO = same date (bulletproof, no pagination issues)
- If no data: fetches last 3 years up to yesterday
- If data exists: fetches only missing dates (gap fill)
Group 05 - ISB AMPBA
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import os, time
from datetime import datetime, timedelta, date

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE      = os.path.join(BASE_DIR, "data", "iex_historical.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, "data", "iex_fetch_progress.csv")
IEX_URL       = "https://www.iexindia.com/market-data/real-time-market/market-snapshot"

def get_date_range():
    end   = datetime.now().date() - timedelta(days=1)
    start = max(end - timedelta(days=3*365), date(2022, 4, 1))
    return start, end

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            return set(pd.read_csv(PROGRESS_FILE)["date"].tolist())
        except: return set()
    return set()

def save_progress(date_str):
    done = load_progress()
    done.add(date_str)
    pd.DataFrame({"date": sorted(done)}).to_csv(PROGRESS_FILE, index=False)

def get_missing_dates(start, end):
    done      = load_progress()
    all_dates = pd.date_range(start=start, end=end, freq="D")
    return [d.date() for d in all_dates if str(d.date()) not in done]

def get_chrome_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

def fill_mui_date(driver, element, date_str):
    """Fill MUI React date input properly using JS events"""
    try:
        driver.execute_script("arguments[0].click();", element)
        time.sleep(0.2)

        # Clear field
        element.send_keys(Keys.CONTROL + "a")
        element.send_keys(Keys.DELETE)
        time.sleep(0.2)

        # Type character by character
        for char in date_str:
            element.send_keys(char)
            time.sleep(0.04)

        # Trigger React change events via JS
        driver.execute_script("""
            var nativeSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value').set;
            nativeSetter.call(arguments[0], arguments[1]);
            arguments[0].dispatchEvent(new Event('input',  {bubbles:true}));
            arguments[0].dispatchEvent(new Event('change', {bubbles:true}));
        """, element, date_str)
        time.sleep(0.2)
        element.send_keys(Keys.TAB)
        time.sleep(0.3)
    except Exception as e:
        print(f"fill error: {e}")

def scrape_day(driver, target_date):
    """Scrape a single day — FROM = TO = target_date"""
    date_str = target_date.strftime("%d-%m-%Y")
    try:
        driver.get(IEX_URL)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "MuiSelect-select"))
        )
        time.sleep(3)

        # Click Delivery Period dropdown
        selects = driver.find_elements(By.CLASS_NAME, "MuiSelect-select")
        driver.execute_script("arguments[0].click();", selects[1])
        time.sleep(1.5)

        # Click SELECT_RANGE
        items = driver.find_elements(By.CSS_SELECTOR, "li[role='option'], li.MuiMenuItem-root")
        for item in items:
            if item.get_attribute("data-value") == "SELECT_RANGE":
                driver.execute_script("arguments[0].click();", item)
                break
        time.sleep(1.5)

        # Fill both FROM and TO with same date
        date_inputs = driver.find_elements(By.CSS_SELECTOR, "input[placeholder='DD-MM-YYYY']")
        if len(date_inputs) < 2:
            return []

        fill_mui_date(driver, date_inputs[0], date_str)
        time.sleep(0.5)

        # Re-fetch after first fill (DOM may update)
        date_inputs = driver.find_elements(By.CSS_SELECTOR, "input[placeholder='DD-MM-YYYY']")
        fill_mui_date(driver, date_inputs[1], date_str)
        time.sleep(1)

        # Wait for Update Report to become enabled then click
        for _ in range(10):
            btns = driver.find_elements(By.TAG_NAME, "button")
            for btn in btns:
                if "update" in btn.text.lower() and "report" in btn.text.lower():
                    cls = btn.get_attribute("class") or ""
                    if "disabled" not in cls and btn.is_enabled():
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(4)
                        return parse_table(driver, date_str)
            time.sleep(1)

        # Try clicking anyway
        for btn in driver.find_elements(By.TAG_NAME, "button"):
            if "update" in btn.text.lower():
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(4)
                break

        return parse_table(driver, date_str)

    except Exception as e:
        print(f"error: {e}")
        return []

def parse_table(driver, expected_date):
    """Parse 15-min blocks from table, verify date matches"""
    try:
        tables = driver.find_elements(By.TAG_NAME, "table")
        if not tables:
            return []
        rows    = tables[0].find_elements(By.TAG_NAME, "tr")
        records = []
        current_date = None

        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) < 5:
                continue
            texts = [c.text.strip() for c in cells]

            for i, t in enumerate(texts):
                if len(t) == 10 and t.count("-") == 2:
                    current_date = t
                if len(t) == 11 and t.count(":") == 2 and t.count("-") == 1:
                    try:
                        mcp = float(texts[i+5].replace(",","")) if i+5 < len(texts) else None
                        if mcp:
                            records.append({
                                "date":             current_date or expected_date,
                                "time_block":       t,
                                "purchase_bid_mw":  float(texts[i+1].replace(",","")) if i+1 < len(texts) else None,
                                "sell_bid_mw":      float(texts[i+2].replace(",","")) if i+2 < len(texts) else None,
                                "mcv_mw":           float(texts[i+3].replace(",","")) if i+3 < len(texts) else None,
                                "scheduled_vol_mw": float(texts[i+4].replace(",","")) if i+4 < len(texts) else None,
                                "MCP":              mcp,
                            })
                    except: pass
                    break

        # Verify we got the right date (not just today's data)
        if records:
            dates_in_records = set(r["date"] for r in records if r["date"])
            if expected_date not in dates_in_records and dates_in_records:
                # Wrong date returned — mark as today's data bleed
                return []
        return records

    except Exception as e:
        print(f"parse error: {e}")
        return []

def fetch_all():
    start, end   = get_date_range()
    missing      = get_missing_dates(start, end)
    total        = len(missing)
    total_days   = (end - start).days + 1
    done_count   = total_days - total

    print("="*55)
    print("IEX HISTORICAL — 1 Day Per Request (Bulletproof)")
    print(f"Range  : {start} to {end} ({total_days} days)")
    print(f"Done   : {done_count} | Remaining: {total}")
    print(f"Approx : ~{total * 8 // 60} mins remaining")
    print("Auto-resumes if interrupted — just run again!")
    print("="*55 + "\n")

    if not missing:
        print("All dates fetched! ✅")
        return

    all_records = []
    if os.path.exists(OUT_FILE):
        existing = pd.read_csv(OUT_FILE)
        all_records.append(existing)
        print(f"Loaded {len(existing):,} existing records\n")

    driver = get_chrome_driver()
    fail_streak = 0

    try:
        for i, target_date in enumerate(missing):
            date_str = str(target_date)
            print(f"[{done_count+i+1}/{total_days}] {target_date.strftime('%d-%m-%Y')}...",
                  end=" ", flush=True)

            records = scrape_day(driver, target_date)

            if records:
                all_records.append(pd.DataFrame(records))
                save_progress(date_str)
                fail_streak = 0
                print(f"{len(records)} blocks | MCP: {records[0]['MCP']:.2f}–{records[-1]['MCP']:.2f}")
            else:
                fail_streak += 1
                print("no data")
                # Restart driver if 3 consecutive failures
                if fail_streak >= 3:
                    print("  Restarting Chrome driver...")
                    try: driver.quit()
                    except: pass
                    driver = get_chrome_driver()
                    fail_streak = 0

            # Checkpoint every 20 days
            if (i+1) % 20 == 0 and all_records:
                combined = pd.concat(all_records, ignore_index=True)
                combined = combined.drop_duplicates(subset=["date","time_block"])
                combined.to_csv(OUT_FILE, index=False)
                unique_dates = combined["date"].nunique()
                print(f"  → Checkpoint: {len(combined):,} records | {unique_dates} dates")

            time.sleep(4)

    except KeyboardInterrupt:
        print("\nInterrupted — progress saved!")
    finally:
        try: driver.quit()
        except: pass
        if all_records:
            combined = pd.concat(all_records, ignore_index=True)
            combined = combined.drop_duplicates(subset=["date","time_block"])
            combined = combined.sort_values(["date","time_block"]).reset_index(drop=True)
            combined.to_csv(OUT_FILE, index=False)
            print(f"\nSaved {len(combined):,} records → {OUT_FILE}")
            print(f"Unique dates: {combined['date'].nunique()}")
            print("Run again to continue!")

if __name__ == "__main__":
    fetch_all()
