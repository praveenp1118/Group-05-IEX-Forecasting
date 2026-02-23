"""
scraper_iex.py
Scrapes IEX Real-Time Market data every 15 minutes
Source: https://www.iexindia.com/market-data/real-time-market/market-snapshot
Group 05 - ISB AMPBA
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import os, time
from datetime import datetime

IEX_URL   = "https://www.iexindia.com/market-data/real-time-market/market-snapshot"
DATA_FILE = "data/iex_live.csv"

def get_chrome_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def scrape_iex():
    """
    Scrapes all 15-min time blocks from IEX RTM page.
    Confirmed columns: Date | Hour | Session ID | Time Block |
    Purchase Bid | Sell Bid | MCV | Scheduled Volume | MCP
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scraping IEX...")
    driver = None
    try:
        driver = get_chrome_driver()
        driver.get(IEX_URL)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        time.sleep(3)

        tables = driver.find_elements(By.TAG_NAME, "table")
        if not tables:
            print("  No tables found")
            return []

        main_table = tables[0]
        rows = main_table.find_elements(By.TAG_NAME, "tr")

        records = []
        scrape_time  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_date = None

        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) < 5:
                continue
            texts = [c.text.strip() for c in cells]

            time_block = None
            mcp = purchase = sell = mcv = scheduled = None

            for i, t in enumerate(texts):
                # Date format dd-mm-yyyy
                if len(t) == 10 and t.count("-") == 2:
                    current_date = t
                # Time block format HH:MM-HH:MM
                if len(t) == 11 and t.count(":") == 2 and t.count("-") == 1:
                    time_block = t
                    try:
                        purchase  = float(texts[i+1].replace(",","")) if i+1 < len(texts) else None
                        sell      = float(texts[i+2].replace(",","")) if i+2 < len(texts) else None
                        mcv       = float(texts[i+3].replace(",","")) if i+3 < len(texts) else None
                        scheduled = float(texts[i+4].replace(",","")) if i+4 < len(texts) else None
                        mcp_str   = texts[i+5].replace(",","")        if i+5 < len(texts) else ""
                        mcp       = float(mcp_str) if mcp_str else None
                    except:
                        pass
                    break

            if time_block and mcp:
                records.append({
                    "scrape_timestamp": scrape_time,
                    "date":             current_date,
                    "time_block":       time_block,
                    "purchase_bid_mw":  purchase,
                    "sell_bid_mw":      sell,
                    "mcv_mw":           mcv,
                    "scheduled_vol_mw": scheduled,
                    "MCP":              mcp,
                })

        print(f"  Scraped {len(records)} time blocks")
        if records:
            latest = records[-1]
            print(f"  Latest: {latest['time_block']} | MCP: {latest['MCP']} Rs/MWh")
        return records

    except Exception as e:
        print(f"  IEX scrape error: {e}")
        return []
    finally:
        if driver:
            driver.quit()

def save_iex_data(records):
    if not records:
        return False
    os.makedirs("data", exist_ok=True)
    df_new = pd.DataFrame(records)
    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["date","time_block"], keep="last")
    else:
        df_combined = df_new
    df_combined = df_combined.sort_values(["date","time_block"]).reset_index(drop=True)
    df_combined.to_csv(DATA_FILE, index=False)
    print(f"  Saved → {DATA_FILE} | Total records: {len(df_combined)}")
    return True

def get_latest_iex():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if len(df) > 0:
            return df.iloc[-1].to_dict()
    return None

if __name__ == "__main__":
    records = scrape_iex()
    if records:
        save_iex_data(records)
        print(f"\nSuccess! {len(records)} blocks scraped")
        print(f"Latest MCP: {records[-1]['MCP']} Rs/MWh at {records[-1]['time_block']}")
    else:
        print("Scrape failed")
