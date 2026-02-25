"""
scraper_iex.py - IEX RTM Live Data Scraper
Loads the IEX market-snapshot page, waits for React to render,
then reads the table directly from DOM.
Place in: D:\\Group-05-IEX-Forecasting\\data_pipeline\\scraper_iex.py
"""
import os, sys, time
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "iex_live.csv")
HIST_FILE   = os.path.join(BASE_DIR, "data", "iex_historical.csv")

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
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36")

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

def parse_float(s):
    try:
        return float(str(s).replace(",","").strip())
    except:
        return None

def scrape_date(date_str, driver=None):
    """
    Scrape IEX RTM data for a specific date.
    Waits for React to render the table, then reads DOM.
    date_str: DD-MM-YYYY format
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    import json

    url = "https://www.iexindia.com/market-data/real-time-market/market-snapshot"
    print(f"  Loading {url} for {date_str}...")

    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    records = []

    try:
        driver.get(url)

        # Wait for React to hydrate — look for the table
        print("  Waiting for React table to render...")
        wait = WebDriverWait(driver, 30)

        # Wait for a cell with time-block format (00:00-00:15)
        try:
            wait.until(EC.presence_of_element_located(
                (By.XPATH, "//*[contains(text(),'00:00-00:15') or contains(text(),'00:00 - 00:15')]")
            ))
            print("  Table rendered!")
        except:
            print("  Table wait timed out — trying anyway...")

        time.sleep(3)  # Extra buffer for all rows to load

        # If requesting a non-today date, change the delivery period dropdown
        today_str = datetime.now().strftime("%d-%m-%Y")
        if date_str != today_str:
            print(f"  Changing date to {date_str}...")
            try:
                # Find the delivery period dropdown
                dropdowns = driver.find_elements(By.XPATH,
                    "//button[contains(@class,'dropdown') or contains(text(),'Today') or contains(text(),'Yesterday')]"
                    " | //div[contains(@class,'select') and contains(text(),'Today')]"
                    " | //*[contains(text(),'Delivery Period')]/following-sibling::*"
                )
                print(f"  Found {len(dropdowns)} dropdown candidates")
                for d in dropdowns[:3]:
                    print(f"    dropdown: tag={d.tag_name} text={d.text[:50]}")

                # Try clicking any element that says Today
                today_el = driver.find_elements(By.XPATH, "//*[text()='Today']")
                if today_el:
                    today_el[0].click()
                    time.sleep(1)
                    # Now look for specific date option
                    date_options = driver.find_elements(By.XPATH, f"//*[contains(text(),'{date_str}')]")
                    for opt in date_options[:3]:
                        print(f"    date option: {opt.text}")
                        if date_str in opt.text:
                            opt.click()
                            time.sleep(3)
                            break
            except Exception as e:
                print(f"  Date change failed: {e} — reading today's data instead")

        # Read the table from DOM
        # Look for all table rows with time block pattern
        rows = driver.find_elements(By.XPATH,
            "//tr[td[contains(text(),':') and contains(text(),'-') and string-length(text())=11]]"
        )
        print(f"  Found {len(rows)} data rows")

        # If no rows found try alternate approach — read all td elements
        if not rows:
            # Try JavaScript to extract table data
            print("  Trying JavaScript extraction...")
            try:
                js_data = driver.execute_script("""
                    var rows = document.querySelectorAll('table tr');
                    var result = [];
                    for(var i=0; i<rows.length; i++){
                        var cells = rows[i].querySelectorAll('td');
                        if(cells.length >= 5){
                            var row = [];
                            for(var j=0; j<cells.length; j++){
                                row.push(cells[j].innerText.trim());
                            }
                            result.push(row);
                        }
                    }
                    return result;
                """)
                print(f"  JS extracted {len(js_data)} rows")
                if js_data:
                    print(f"  Sample: {js_data[:2]}")
                    for row in js_data:
                        if len(row) >= 5 and ':' in str(row[0]) and '-' in str(row[0]) and len(str(row[0])) == 11:
                            # Row format: [time_block, purchase_bid, sell_bid, mcv, scheduled_vol, MCP]
                            mcp = parse_float(row[-1])
                            if mcp and 100 < mcp < 20000:
                                records.append({
                                    "date":             date_str,
                                    "time_block":       str(row[0]).strip(),
                                    "purchase_bid_mw":  parse_float(row[1]) if len(row) > 1 else None,
                                    "sell_bid_mw":      parse_float(row[2]) if len(row) > 2 else None,
                                    "mcv_mw":           parse_float(row[3]) if len(row) > 3 else None,
                                    "scheduled_vol_mw": parse_float(row[4]) if len(row) > 4 else None,
                                    "MCP":              mcp,
                                    "scrape_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                })
            except Exception as e:
                print(f"  JS extraction failed: {e}")

        else:
            # Read from found rows
            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 5:
                        continue
                    # Find which cell is the time block
                    time_block = None
                    for cell in cells:
                        txt = cell.text.strip()
                        if len(txt) == 11 and ':' in txt and txt.count('-') == 1:
                            time_block = txt
                            break
                    if not time_block:
                        continue

                    texts = [c.text.strip().replace(",","") for c in cells]
                    mcp = parse_float(texts[-1])
                    if not mcp or mcp < 100 or mcp > 20000:
                        continue

                    records.append({
                        "date":             date_str,
                        "time_block":       time_block,
                        "purchase_bid_mw":  parse_float(texts[1]) if len(texts) > 1 else None,
                        "sell_bid_mw":      parse_float(texts[2]) if len(texts) > 2 else None,
                        "mcv_mw":           parse_float(texts[3]) if len(texts) > 3 else None,
                        "scheduled_vol_mw": parse_float(texts[4]) if len(texts) > 4 else None,
                        "MCP":              mcp,
                        "scrape_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                except:
                    continue

    except Exception as e:
        print(f"  Error: {e}")
    finally:
        if own_driver:
            try: driver.quit()
            except: pass

    print(f"  Scraped {len(records)} records for {date_str}")
    return records

def append_to_historical(new_df):
    """Append new records to iex_historical.csv so data persists across container restarts."""
    if not os.path.exists(HIST_FILE):
        print("  Historical file not found — skipping historical append")
        return
    try:
        hist = pd.read_csv(HIST_FILE)
        combined = pd.concat([hist, new_df], ignore_index=True)
        combined["date_p"] = pd.to_datetime(combined["date"], format="%d-%m-%Y", errors="coerce")
        combined = combined.sort_values(["date_p","time_block"])
        combined = combined.drop_duplicates(subset=["date","time_block"], keep="last")
        combined = combined.drop(columns=["date_p"])
        combined.to_csv(HIST_FILE, index=False)
        print(f"  Appended new rows -> iex_historical.csv ({len(combined)} total) ✅")
    except Exception as e:
        print(f"  Historical append failed: {e}")

def save_records(records):
    """Merge with existing live file, sort by date, dedup, keep latest 500.
    Also appends to iex_historical.csv for persistence across restarts."""
    if not records:
        print("  No new records to save")
        return

    new_df = pd.DataFrame(records)
    new_df = new_df.dropna(subset=["MCP"])
    new_df = new_df[(new_df["MCP"] > 100) & (new_df["MCP"] <= 20000)]

    if len(new_df) == 0:
        print("  No valid records after filtering")
        return

    # Append to historical first (persists across container restarts)
    append_to_historical(new_df)

    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE).dropna(subset=["MCP"])
            combined = pd.concat([existing, new_df], ignore_index=True)
        except:
            combined = new_df
    else:
        combined = new_df

    combined["date_p"] = pd.to_datetime(combined["date"], format="%d-%m-%Y", errors="coerce")
    combined = combined.sort_values(["date_p","time_block"])
    combined = combined.drop_duplicates(subset=["date","time_block"], keep="last")
    combined = combined.drop(columns=["date_p"])
    combined = combined.tail(500)

    combined.to_csv(OUTPUT_FILE, index=False)
    max_date = pd.to_datetime(combined["date"], format="%d-%m-%Y", errors="coerce").max().strftime("%d-%m-%Y")
    print(f"  Saved {len(new_df)} new records -> iex_live.csv ({len(combined)} total, latest: {max_date}) ✅")

def update_from_historical():
    """Fallback: pull most recent date from historical into live file."""
    print("  Fallback: updating from historical...")
    if not os.path.exists(HIST_FILE):
        return

    hist = pd.read_csv(HIST_FILE)
    hist["date_p"] = pd.to_datetime(hist["date"], format="%d-%m-%Y", errors="coerce")
    hist = hist.sort_values(["date_p","time_block"])
    most_recent = hist["date_p"].max()
    recent = hist[hist["date_p"] == most_recent].drop(columns=["date_p"])
    recent["scrape_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            existing["date_p"] = pd.to_datetime(existing["date"], format="%d-%m-%Y", errors="coerce")
            combined = pd.concat([existing.drop(columns=["date_p"]), recent], ignore_index=True)
            combined["date_p"] = pd.to_datetime(combined["date"], format="%d-%m-%Y", errors="coerce")
            combined = combined.sort_values(["date_p","time_block"]).drop(columns=["date_p"])
            combined = combined.drop_duplicates(subset=["date","time_block"], keep="last")
            combined.tail(500).to_csv(OUTPUT_FILE, index=False)
        except:
            recent.to_csv(OUTPUT_FILE, index=False)
    else:
        recent.to_csv(OUTPUT_FILE, index=False)

    print(f"  Updated from historical (latest: {most_recent.strftime('%d-%m-%Y')}) ✅")

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] IEX scraper starting...")
    today = datetime.now().strftime("%d-%m-%Y")
    try:
        records = scrape_date(today)
        if records:
            save_records(records)
        else:
            print("  Selenium returned 0 records — falling back to historical")
            update_from_historical()
        print("  Done ✅")
    except Exception as e:
        print(f"  Scraper exception: {e}")
        update_from_historical()
