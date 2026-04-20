import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

TICKERS = {
    "FROTO": "FROTO.IS", "DOAS": "DOAS.IS", "TOASO": "TOASO.IS",
    "SOKM": "SOKM.IS", "BIMAS": "BIMAS.IS", "MGROS": "MGROS.IS",
    "THYAO": "THYAO.IS", "PGSUS": "PGSUS.IS", "TAVHL": "TAVHL.IS",
    "EREGL": "EREGL.IS", "KRDMD": "KRDMD.IS", "ISDMR": "ISDMR.IS",
    "USDTRY": "USDTRY=X", "XU100": "^XU100", "GOLD": "GC=F"
}

INV_MAP = {v: k for k, v in TICKERS.items()}

import requests
from bs4 import BeautifulSoup

def get_google_finance_price(ticker, exchange="IST"):
    """
    Scrapes the last price from Google Finance.
    Used as a fallback for yfinance during market hours.
    """
    # Mapping for special tickers
    if ticker == "USDTRY": url = "https://www.google.com/finance/quote/USD-TRY"
    elif ticker == "XU100": url = "https://www.google.com/finance/quote/XU100:INDEXIST"
    elif ticker == "GOLD": url = "https://www.google.com/finance/quote/GCW00:COMEX"
    else: url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for price in specific div or by data-last-price attribute
        price_div = soup.find("div", {"class": "YMlS7e"})
        if not price_div:
            # Try finding by looking for any div with data-last-price
            for div in soup.find_all("div"):
                if div.has_attr("data-last-price"):
                    return float(div["data-last-price"])
            return None
            
        # Clean price text: handles "1.234,50 TL" or "14,426.98"
        txt = price_div.text.replace("₺", "").replace("TL", "").replace("$", "").strip()
        
        # Robust number cleaning
        if "," in txt and "." in txt:
            # Standard international: 1,234.56 -> 1234.56
            if txt.find(",") < txt.find("."):
                txt = txt.replace(",", "")
            # Turkish/European: 1.234,56 -> 1234.56
            else:
                txt = txt.replace(".", "").replace(",", ".")
        elif "," in txt:
            # Could be 1,234 (international) or 1234,56 (TR)
            # If it's BIST, it's likely TR format for decimals
            # But let's check position of comma
            if len(txt.split(",")[-1]) == 2: # 1234,56
                txt = txt.replace(",", ".")
            else: # 1,234
                txt = txt.replace(",", "")
                
        return float(txt)
    except Exception as e:
        print(f"⚠️ Google Finance error for {ticker}: {e}")
        return None

def get_live_prices(ticker_map):
    """
    Fetches the most recent price for all tickers.
    Prioritizes Google Finance for speed and accuracy during market hours.
    """
    live_data = {}
    
    # We fetch Google Finance prices for EVERYTHING first because it's more reliable for intraday
    for short_name in ticker_map.keys():
        # print(f"🚀 Fetching {short_name} from Google Finance...")
        price = get_google_finance_price(short_name)
        
        # If Google fails, only then try yfinance as a secondary backup
        if price is None:
            try:
                full_symbol = ticker_map[short_name]
                ticker_obj = yf.Ticker(full_symbol)
                price = ticker_obj.fast_info['lastPrice']
                if price == 0: price = None
            except:
                price = None
                
        live_data[short_name] = price
        
    return pd.Series(live_data)

def update_database(force_rebuild=False):
    """
    Updates the local CSV database with the latest data.
    """
    nom_file = "db_nominal.csv"
    adj_file = "db_adjusted.csv"
    
    if force_rebuild or not os.path.exists(nom_file) or not os.path.exists(adj_file):
        # Rebuild logic remains the same as it's for historical data
        print("🚀 Rebuilding entire database from 2016...")
        raw = yf.download(list(TICKERS.values()), start="2016-01-01", progress=True, auto_adjust=False)
        if raw.empty:
            return False, None, None
        
        df_nom = raw['Close']; df_adj = raw['Adj Close']
        df_nom.columns = [INV_MAP.get(c, c) for c in df_nom.columns]
        df_adj.columns = [INV_MAP.get(c, c) for c in df_adj.columns]
        
        try:
            df_nom.to_csv(nom_file); df_adj.to_csv(adj_file)
        except: pass
        return True, df_nom, df_adj

    # Incremental update
    try:
        df_nom = pd.read_csv(nom_file, index_col='Date', parse_dates=True)
        df_adj = pd.read_csv(adj_file, index_col='Date', parse_dates=True)
    except:
        return False, None, None
    
    last_date = df_nom.index.max()
    today = datetime.now()
    start_sync = last_date - timedelta(days=3)
    
    new_nom = pd.DataFrame(); new_adj = pd.DataFrame()

    # 1. Attempt yfinance history sync (wrapped in try-except to not block live data)
    try:
        print(f"📡 Checking history from {start_sync.date()}...")
        raw = yf.download(list(TICKERS.values()), start=start_sync, progress=False, auto_adjust=False)
        if not raw.empty:
            new_nom = raw['Close']; new_adj = raw['Adj Close']
            new_nom.columns = [INV_MAP.get(c, c) for c in new_nom.columns]
            new_adj.columns = [INV_MAP.get(c, c) for c in new_adj.columns]
    except Exception as e:
        print(f"⚠️ History sync failed (non-critical): {e}")

    # 2. LIVE PRICE INJECTION (Google Finance Primary)
    is_trading_day = (today.weekday() < 5)
    today_date = today.date()
    
    if is_trading_day:
        print("⚡ Fetching primary live prices from Google Finance...")
        live_prices = get_live_prices(TICKERS)
        if not live_prices.isna().all():
            # Round to nearest 15 minutes to capture intraday movements without bloating DB
            now = datetime.now()
            minute = (now.minute // 15) * 15
            today_ts = pd.Timestamp(now.date()).replace(hour=now.hour, minute=minute, second=0, microsecond=0)

            # Create or update today's specific 15-min row
            live_row = pd.DataFrame([live_prices], index=[today_ts])

            # Merge with new_nom/new_adj
            if not new_nom.empty:
                # If yf.download also returned today (usually as 00:00), 
                # we prefer our timestamped Google data for intraday
                new_nom = pd.concat([new_nom[new_nom.index.date != today_date], live_row]).sort_index()
                new_adj = pd.concat([new_adj[new_adj.index.date != today_date], live_row]).sort_index()
            else:
                new_nom = live_row
                new_adj = live_row

            print(f"✅ Intraday data injected for {today_ts}")
    if new_nom.empty:
        return True, df_nom, df_adj

    # 3. Merge and Deduplicate
    df_nom = pd.concat([df_nom[df_nom.index < new_nom.index[0]], new_nom]).sort_index()
    df_adj = pd.concat([df_adj[df_adj.index < new_adj.index[0]], new_adj]).sort_index()
    df_nom = df_nom[~df_nom.index.duplicated(keep='last')]
    df_adj = df_adj[~df_adj.index.duplicated(keep='last')]

    # Save to disk
    try:
        df_nom.to_csv(nom_file); df_adj.to_csv(adj_file)
    except: pass
    
    print(f"✅ DB Updated. Latest: {df_nom.index.max()}")
    return True, df_nom, df_adj

if __name__ == "__main__":
    update_database()
