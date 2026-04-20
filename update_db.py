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
            
        # Clean price text: handles "1.234,50 TL" -> "1234.50"
        txt = price_div.text.replace("₺", "").replace("TL", "").replace("$", "").replace(",", "").strip()
        # Note: BIST uses comma for decimal in Turkish locale, but Google Finance UI 
        # often adapts to headers. Let's be robust.
        if "." in txt and "," in txt: # "1,234.50"
             txt = txt.replace(",", "")
        elif "," in txt: # "1234,50"
             txt = txt.replace(",", ".")
             
        return float(txt)
    except Exception as e:
        print(f"⚠️ Google Finance error for {ticker}: {e}")
        return None

def get_live_prices(ticker_map):
    """
    Fetches the most recent price for all tickers.
    Tries yfinance first, but prefers Google Finance for BIST tickers during market hours.
    """
    now = datetime.now()
    is_bist_hours = (now.weekday() < 5) and (10 <= now.hour < 18)
    
    tickers_obj = yf.Tickers(list(ticker_map.values()))
    live_data = {}
    
    for short_name, full_symbol in ticker_map.items():
        price = None
        is_bist = full_symbol.endswith(".IS") or full_symbol == "^XU100"
        
        # 1. During BIST hours, try Google Finance FIRST for BIST stocks
        if is_bist and is_bist_hours:
            # print(f"🚀 Preferring Google Finance for {short_name} during market hours...")
            price = get_google_finance_price(short_name)
        
        # 2. If not BIST, or Google failed, or outside hours, try yfinance
        if price is None:
            try:
                fast_info = tickers_obj.tickers[full_symbol].fast_info
                price = fast_info['lastPrice']
                # Basic check for 0 or None
                if price == 0: price = None
            except: pass
            
        # 3. Final Fallback if yfinance also failed
        if price is None:
            # print(f"🔍 Final fallback to Google Finance for {short_name}...")
            price = get_google_finance_price(short_name)
            
        live_data[short_name] = price
        
    return pd.Series(live_data)

def update_database(force_rebuild=False):
    """
    Updates the local CSV database with the latest data from Yahoo Finance.
    Saves results back to disk to ensure subsequent loads are fast.
    """
    nom_file = "db_nominal.csv"
    adj_file = "db_adjusted.csv"
    
    if force_rebuild or not os.path.exists(nom_file) or not os.path.exists(adj_file):
        print("🚀 Rebuilding entire database from 2016...")
        raw = yf.download(list(TICKERS.values()), start="2016-01-01", progress=True, auto_adjust=False)
        if raw.empty:
            print("❌ Error: No data received.")
            return False
        
        df_nom = raw['Close']
        df_adj = raw['Adj Close']
        df_nom.columns = [INV_MAP.get(c, c) for c in df_nom.columns]
        df_adj.columns = [INV_MAP.get(c, c) for c in df_adj.columns]
        
        df_nom.to_csv(nom_file)
        df_adj.to_csv(adj_file)
        print(f"✅ Rebuild complete. {len(df_nom)} rows saved.")
        return True

    # Incremental update
    print("🔄 Checking for updates...")
    df_nom = pd.read_csv(nom_file, index_col='Date', parse_dates=True)
    df_adj = pd.read_csv(adj_file, index_col='Date', parse_dates=True)
    
    last_date = df_nom.index.max()
    today = datetime.now()
    
    # Use a small buffer to handle potential adjustments/corrections in recent days
    start_sync = last_date - timedelta(days=3)
    
    print(f"📡 Syncing from {start_sync.date()}...")
    try:
        # Batch download for history
        raw = yf.download(list(TICKERS.values()), start=start_sync, progress=False, auto_adjust=False)
        
        if not raw.empty:
            new_nom = raw['Close']
            new_adj = raw['Adj Close']
            new_nom.columns = [INV_MAP.get(c, c) for c in new_nom.columns]
            new_adj.columns = [INV_MAP.get(c, c) for c in new_adj.columns]
        else:
            new_nom = pd.DataFrame()
            new_adj = pd.DataFrame()

        # LIVE PRICE INJECTION
        is_trading_day = (today.weekday() < 5)
        
        # Check if we need to supplement with live data
        today_date = today.date()
        
        # We want to fetch live prices if:
        # 1. Today is a trading day AND
        # 2. Today's date is missing from our new data OR it has NaNs in stock columns
        needs_live = False
        if is_trading_day:
            stock_cols = [k for k in TICKERS.keys() if k not in ["USDTRY", "XU100", "GOLD"]]
            if not new_nom.empty and today_date in new_nom.index.date:
                # Check if today's row has NaNs for stocks
                today_row = new_nom[new_nom.index.date == today_date]
                if today_row[stock_cols].isna().any().any():
                    needs_live = True
            elif not df_nom.empty and today_date in df_nom.index.date:
                today_row = df_nom[df_nom.index.date == today_date]
                if today_row[stock_cols].isna().any().any():
                    needs_live = True
            else:
                needs_live = True

        if needs_live:
            print("⚡ Fetching live intraday prices to fill gaps...")
            live_prices = get_live_prices(TICKERS)
            if not live_prices.isna().all():
                today_ts = pd.Timestamp(today_date).replace(hour=today.hour, minute=today.minute)
                
                # If today already exists in new_nom, update it
                # Otherwise, create it
                if not new_nom.empty and today_date in new_nom.index.date:
                    idx = new_nom.index[new_nom.index.date == today_date][0]
                    for col, val in live_prices.items():
                        if pd.isna(new_nom.at[idx, col]) or new_nom.at[idx, col] == 0:
                            new_nom.at[idx, col] = val
                            if col in new_adj.columns:
                                new_adj.at[idx, col] = val
                else:
                    live_row = pd.DataFrame([live_prices], index=[today_ts])
                    new_nom = pd.concat([new_nom, live_row])
                    new_adj = pd.concat([new_adj, live_row])
                print(f"✅ Supplemental live data injected for {today_date}")

        if new_nom.empty:
            print("ℹ️ No new data found.")
            return True

        # Combine: Keep old data, but update with new data for overlapping dates
        df_nom = pd.concat([df_nom[df_nom.index < new_nom.index[0]], new_nom]).sort_index()
        df_adj = pd.concat([df_adj[df_adj.index < new_adj.index[0]], new_adj]).sort_index()
        
        # Deduplicate indices just in case
        df_nom = df_nom[~df_nom.index.duplicated(keep='last')]
        df_adj = df_adj[~df_adj.index.duplicated(keep='last')]

        # Save to disk
        df_nom.to_csv(nom_file)
        df_adj.to_csv(adj_file)
        
        print(f"✅ Database updated. Last date: {df_nom.index.max()}")
        return True
    except Exception as e:
        print(f"❌ Update failed: {e}")
        return False

if __name__ == "__main__":
    update_database()
