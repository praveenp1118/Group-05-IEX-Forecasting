"""
find_iex_api4.py — Explore iexrtmprice.com + RSC endpoint
docker cp find_iex_api4.py group05-iex-api:/app/find_iex_api4.py
docker exec group05-iex-api python3 /app/find_iex_api4.py 2>&1
"""
import requests, re, json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

feb22 = "22-02-2026"
feb22_slash = "22/02/2026"

# ── 1. Explore iexrtmprice.com homepage ─────────────────────
print("="*60)
print("1. iexrtmprice.com homepage")
print("="*60)
try:
    r = requests.get("https://iexrtmprice.com/", headers=HEADERS, timeout=15)
    print(f"Status: {r.status_code} | Size: {len(r.content)}b")
    soup = BeautifulSoup(r.text, "html.parser")

    # Find all links
    links = [a.get("href","") for a in soup.find_all("a")]
    print(f"Links: {links[:20]}")

    # Find all script src
    scripts = [s.get("src","") for s in soup.find_all("script") if s.get("src")]
    print(f"Scripts: {scripts[:10]}")

    # Find all tables
    tables = soup.find_all("table")
    print(f"Tables: {len(tables)}")
    for i, t in enumerate(tables[:3]):
        rows = t.find_all("tr")
        print(f"  Table {i}: {len(rows)} rows")
        if rows:
            print(f"  First row: {rows[0].get_text()[:100]}")
            if len(rows) > 1:
                print(f"  Second row: {rows[1].get_text()[:100]}")

    # Find any JS variables with data
    inline_scripts = [s.string for s in soup.find_all("script") if s.string]
    for sc in inline_scripts[:5]:
        if sc and len(sc) > 50:
            print(f"\nInline script preview: {sc[:300]}")
except Exception as e:
    print(f"Error: {e}")

# ── 2. Explore iexrtmprice.com/DSM_Data/ ─────────────────────
print("\n" + "="*60)
print("2. iexrtmprice.com/DSM_Data/")
print("="*60)
try:
    r = requests.get("https://iexrtmprice.com/DSM_Data/", headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    links = [a.get("href","") for a in soup.find_all("a")]
    print(f"Links: {links[:20]}")
    tables = soup.find_all("table")
    print(f"Tables: {len(tables)}")
    for i, t in enumerate(tables[:2]):
        rows = t.find_all("tr")
        print(f"  Table {i}: {len(rows)} rows")
        for row in rows[:3]:
            print(f"    {row.get_text()[:120]}")
except Exception as e:
    print(f"Error: {e}")

# ── 3. Try iexrtmprice.com with date params ───────────────────
print("\n" + "="*60)
print("3. iexrtmprice.com date variants")
print("="*60)
date_urls = [
    f"https://iexrtmprice.com/?date={feb22}",
    f"https://iexrtmprice.com/index.php?date={feb22}",
    f"https://iexrtmprice.com/data?date={feb22}",
    f"https://iexrtmprice.com/api/data?date={feb22}",
    f"https://iexrtmprice.com/price?date={feb22}",
    f"https://iexrtmprice.com/rtm?date={feb22}",
    f"https://iexrtmprice.com/getdata?date={feb22}",
    f"https://iexrtmprice.com/fetch?date={feb22}",
    f"https://iexrtmprice.com/getData.php?date={feb22}",
    f"https://iexrtmprice.com/data.php?date={feb22}",
]
for url in date_urls:
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        ct = r.headers.get("Content-Type","")
        print(f"{r.status_code} | {len(r.content)}b | {url}")
        if r.status_code == 200 and "html" not in ct.lower() and len(r.content) > 100:
            print(f"  DATA FOUND: {r.text[:300]}")
    except Exception as e:
        print(f"  Error: {e}")

# ── 4. Try RSC endpoint ───────────────────────────────────────
print("\n" + "="*60)
print("4. Try Next.js RSC endpoint")
print("="*60)
rsc_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/x-component",
    "Next-Router-State-Tree": "%5B%22%22%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%5D%7D%2Cnull%2Cnull%2Ctrue%5D",
    "RSC": "1",
    "Referer": "https://www.iexindia.com/market-data/real-time-market/market-snapshot",
}
rsc_urls = [
    f"https://www.iexindia.com/market-data/real-time-market/market-snapshot?_rsc=1wo8m",
    f"https://www.iexindia.com/market-data/real-time-market/market-snapshot?date={feb22}&_rsc=1wo8m",
]
for url in rsc_urls:
    try:
        r = requests.get(url, headers=rsc_headers, timeout=10)
        print(f"\n{r.status_code} | {len(r.content)}b | {r.headers.get('Content-Type','')}")
        print(f"URL: {url}")
        print(f"Preview: {r.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")

print("\nDone.")
