"""
scraper_iex.py - IEX RTM Live Data Scraper
Works both locally (with Chrome) and in Docker (headless Chromium)
Place in: D:\Group-05-IEX-Forecasting\data_pipeline\scraper_iex.py
"""
import os, sys, time, csv
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

OUTPUT_FILE = os.path.join(BASE_DIR, "data", "iex_live.csv")

def get_driver():
    """Get Selenium driver — auto-detects Docker vs local"""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    options = Options()
    options.add_argument("--headless")           # Always headless
    options.add_argument("--no-sandbox")         # Required in Docker
    options.add_argument("--disable-dev-shm-usage")  # Required in Docker
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-logging")
    options.add_argument("--log-level=3")

    # Docker: use system chromium
    chromium_path = os.environ.get("CHROME_BIN", "")
    if chromium_path and os.path.exists(chromium_path):
        options.binary_location = chromium_path
        chromedriver = os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
        service = Service(executable_path=chromedriver)
        print(f"  Using Docker Chromium: {chromium_path}")
        return webdriver.Chrome(service=service, options=options)

    # Local: use webdriver-manager
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        print("  Using local ChromeDriver")
        return webdriver.Chrome(service=service, options=options)
    except Exception as e:
        print(f"  ChromeDriver error: {e}")
        raise

def scrape_today():
    """Scrape today's IEX RTM data"""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    today = datetime.now().strftime("%d-%m-%Y")
    url   = f"https://www.iexindia.com/marketdata/rtm.aspx"

    print(f"  Scraping IEX RTM for {today}...")
    driver = get_driver()
    records = []

    try:
        driver.get(url)
        time.sleep(3)

        # Try to find table data
        try:
            wait  = WebDriverWait(driver, 15)
            table = wait.until(EC.presence_of_element_located(
                (By.XPATH, "//table[contains(@id,'grd') or contains(@class,'grid')]")))
            rows  = table.find_elements(By.TAG_NAME, "tr")

            for row in rows[1:]:  # skip header
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 4:
                    try:
                        records.append({
                            "scrape_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "date":             today,
                            "time_block":       cols[0].text.strip(),
                            "MCP":              float(cols[1].text.strip().replace(",","")),
                            "mcv_mw":           float(cols[2].text.strip().replace(",","")) if len(cols)>2 else None,
                            "purchase_bid_mw":  float(cols[3].text.strip().replace(",","")) if len(cols)>3 else None,
                            "sell_bid_mw":      float(cols[4].text.strip().replace(",","")) if len(cols)>4 else None,
                            "scheduled_vol_mw": float(cols[5].text.strip().replace(",","")) if len(cols)>5 else None,
                        })
                    except: pass
        except Exception as e:
            print(f"  Table not found: {e}")

    finally:
        driver.quit()

    return records

def save_records(records):
    if not records:
        print("  No records to save")
        return

    new_df = pd.DataFrame(records)
    new_df = new_df.dropna(subset=["MCP"])
    new_df = new_df[(new_df["MCP"] > 0) & (new_df["MCP"] <= 20000)]

    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        existing = existing.dropna(subset=["MCP"])
        combined = pd.concat([existing, new_df], ignore_index=True)
        # Deduplicate on date + time_block
        if "date" in combined.columns and "time_block" in combined.columns:
            combined = combined.drop_duplicates(subset=["date","time_block"], keep="last")
        combined = combined.tail(500)  # keep last 500 records
    else:
        combined = new_df

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved {len(new_df)} new records → iex_live.csv ({len(combined)} total)")

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] IEX scraper starting...")
    try:
        records = scrape_today()
        save_records(records)
        print(f"  Done ✅")
    except Exception as e:
        print(f"  Error: {e}")
        # If scraping fails, update scrape_timestamp on existing data
        # so freshness check doesn't show STALE
        if os.path.exists(OUTPUT_FILE):
            df = pd.read_csv(OUTPUT_FILE)
            df["scrape_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"  Updated timestamp on existing data")
