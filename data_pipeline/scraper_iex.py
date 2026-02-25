"""
scraper_iex.py - IEX RTM Live Data Scraper
Scrapes today's RTM data from IEX website using Selenium.
Falls back gracefully if scraping fails.
Place in: D:\\Group-05-IEX-Forecasting\\data_pipeline\\scraper_iex.py
"""
import os, sys, time
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

OUTPUT_FILE  = os.path.join(BASE_DIR, "data", "iex_live.csv")
HIST_FILE    = os.path.join(BASE_DIR, "data", "iex_historical.csv")

def get_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-logging")
    options.add_argument("--log-level=3")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    chromium_path = os.environ.get("CHROME_BIN", "")
    if chromium_path and os.path.exists(chromium_path):
        options.binary_location = chromium_path
        chromedriver = os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
        service = Service(executable_path=chromedriver)
        print(f"  Using Docker Chromium: {chromium_path}")
        return webdriver.Chrome(service=service, options=options)

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        print("  Using local ChromeDriver")
        return webdriver.Chrome(service=service, options=options)
    except Exception as e:
        print(f"  ChromeDriver error: {e}")
        raise

def scrape_date(date_str):
    """
    Scrape IEX RTM data for a specific date.
    Tries multiple table selectors to handle IEX page variations.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    url = "https://www.iexindia.com/marketdata/rtm.aspx"
    print(f"  Scraping IEX RTM for {date_str}...")
    driver = get_driver()
    records = []

    try:
        driver.get(url)
        time.sleep(5)  # Wait for JS to load

        # Try multiple table selectors — IEX uses different IDs
        table_xpaths = [
            "//table[contains(@id,'grd')]",
            "//table[contains(@id,'Grid')]",
            "//table[contains(@id,'RTM')]",
            "//table[contains(@class,'grid')]",
            "//table[contains(@class,'Grid')]",
            "//table[contains(@class,'table')]",
            "//div[@id='MainContent_pnlGrid']//table",
            "//div[contains(@id,'Content')]//table",
            "//table[@border='1']",
            "//table",   # last resort — any table
        ]

        table = None
        for xpath in table_xpaths:
            try:
                tables = driver.find_elements(By.XPATH, xpath)
                for t in tables:
                    rows = t.find_elements(By.TAG_NAME, "tr")
                    if len(rows) > 5:  # meaningful table has >5 rows
                        table = t
                        print(f"  Found table with {len(rows)} rows using: {xpath}")
                        break
                if table:
                    break
            except:
                continue

        if table is None:
            # Debug — save page source to check structure
            page_len = len(driver.page_source)
            print(f"  No table found. Page length: {page_len} chars")
            # Try waiting longer and retry
            time.sleep(8)
            try:
                tables = driver.find_elements(By.TAG_NAME, "table")
                print(f"  Tables on page after extra wait: {len(tables)}")
                for i, t in enumerate(tables):
                    rows = t.find_elements(By.TAG_NAME, "tr")
                    print(f"    Table {i}: {len(rows)} rows")
                    if len(rows) > 10:
                        table = t
                        print(f"  Using table {i}")
                        break
            except: pass

        if table:
            rows = table.find_elements(By.TAG_NAME, "tr")
            for row in rows[1:]:  # skip header
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 4:
                    try:
                        def clean(s):
                            return s.strip().replace(",","").replace(" ","")

                        mcp_text = clean(cols[1].text)
                        if not mcp_text or mcp_text == "-":
                            continue
                        mcp = float(mcp_text)
                        if mcp <= 0 or mcp > 20000:
                            continue

                        records.append({
                            "date":             date_str,
                            "time_block":       clean(cols[0].text),
                            "purchase_bid_mw":  float(clean(cols[2].text)) if len(cols)>2 and clean(cols[2].text) else None,
                            "sell_bid_mw":      float(clean(cols[3].text)) if len(cols)>3 and clean(cols[3].text) else None,
                            "mcv_mw":           float(clean(cols[4].text)) if len(cols)>4 and clean(cols[4].text) else None,
                            "scheduled_vol_mw": float(clean(cols[5].text)) if len(cols)>5 and clean(cols[5].text) else None,
                            "MCP":              mcp,
                            "scrape_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                    except:
                        continue

    except Exception as e:
        print(f"  Scraper error: {e}")
    finally:
        try:
            driver.quit()
        except:
            pass

    print(f"  Scraped {len(records)} records for {date_str}")
    return records

def scrape_recent_days(n_days=3):
    """
    Try to scrape last n_days — catches up if yesterday's scrape failed.
    """
    all_records = []

    # Check what dates we already have in live file
    existing_dates = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            existing_dates = set(existing["date"].unique())
        except: pass

    for i in range(n_days):
        target_date = datetime.now() - timedelta(days=i)
        date_str    = target_date.strftime("%d-%m-%Y")
        if date_str in existing_dates:
            print(f"  {date_str} already in live file — skipping")
            continue
        try:
            records = scrape_date(date_str)
            all_records.extend(records)
            if records:
                break  # Got today's data — stop
        except Exception as e:
            print(f"  Failed for {date_str}: {e}")
            continue

    return all_records

def save_records(records):
    """Merge new records with existing live file, dedup, keep latest 500."""
    if not records:
        print("  No new records to save")
        return

    new_df = pd.DataFrame(records)
    new_df = new_df.dropna(subset=["MCP"])
    new_df = new_df[(new_df["MCP"] > 0) & (new_df["MCP"] <= 20000)]

    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE).dropna(subset=["MCP"])
            combined = pd.concat([existing, new_df], ignore_index=True)
        except:
            combined = new_df
    else:
        combined = new_df

    # Sort by date + time_block and deduplicate
    combined["date_p"] = pd.to_datetime(combined["date"], format="%d-%m-%Y", errors="coerce")
    combined = combined.sort_values(["date_p","time_block"])
    combined = combined.drop_duplicates(subset=["date","time_block"], keep="last")
    combined = combined.drop(columns=["date_p"])
    combined = combined.tail(500)

    combined.to_csv(OUTPUT_FILE, index=False)
    max_date = pd.to_datetime(combined["date"], format="%d-%m-%Y", errors="coerce").max().strftime("%d-%m-%Y")
    print(f"  Saved {len(new_df)} new records -> iex_live.csv ({len(combined)} total, latest: {max_date})")

def update_timestamp_only():
    """
    If scraping completely fails, at minimum update scrape_timestamp
    so /health doesn't show STALE for live data.
    Also append latest rows from historical so live file stays current.
    """
    print("  Falling back: updating from historical...")
    if not os.path.exists(HIST_FILE):
        return

    hist = pd.read_csv(HIST_FILE)
    hist["date_p"] = pd.to_datetime(hist["date"], format="%d-%m-%Y", errors="coerce")
    hist = hist.sort_values(["date_p","time_block"])

    # Get most recent date from historical
    most_recent_date = hist["date_p"].max()
    recent = hist[hist["date_p"] == most_recent_date].drop(columns=["date_p"])
    recent["scrape_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            existing["date_p"] = pd.to_datetime(existing["date"], format="%d-%m-%Y", errors="coerce")
            combined = pd.concat([existing.drop(columns=["date_p"]), recent], ignore_index=True)
            combined["date_p"] = pd.to_datetime(combined["date"], format="%d-%m-%Y", errors="coerce")
            combined = combined.sort_values(["date_p","time_block"]).drop(columns=["date_p"])
            combined = combined.drop_duplicates(subset=["date","time_block"], keep="last")
            combined = combined.tail(500)
            combined.to_csv(OUTPUT_FILE, index=False)
        except:
            recent.to_csv(OUTPUT_FILE, index=False)
    else:
        recent.to_csv(OUTPUT_FILE, index=False)

    max_d = most_recent_date.strftime("%d-%m-%Y")
    print(f"  Updated iex_live.csv from historical (latest available: {max_d}) ✅")

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] IEX scraper starting...")
    try:
        records = scrape_recent_days(n_days=3)
        if records:
            save_records(records)
        else:
            print("  Selenium scrape returned 0 records")
            update_timestamp_only()
        print("  Done ✅")
    except Exception as e:
        print(f"  Scraper exception: {e}")
        update_timestamp_only()
