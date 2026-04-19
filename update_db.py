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
    # We want to check from the last date onwards. 
    # yfinance 'start' is inclusive, so we'll get last_date again and anything newer.
    
    # Use a small buffer to handle potential adjustments/corrections in recent days
    start_sync = last_date - timedelta(days=3)
    
    print(f"📡 Syncing from {start_sync.date()}...")
    try:
        # Batch download is MUCH faster than individual Ticker calls
        raw = yf.download(list(TICKERS.values()), start=start_sync, progress=False, auto_adjust=False)
        
        if raw.empty:
            print("ℹ️ No new data found.")
            return True

        new_nom = raw['Close']
        new_adj = raw['Adj Close']
        new_nom.columns = [INV_MAP.get(c, c) for c in new_nom.columns]
        new_adj.columns = [INV_MAP.get(c, c) for c in new_adj.columns]

        # Combine: Keep old data, but update with new data for overlapping dates
        # This handles cases where recent dates were NaNs or had old prices
        df_nom = pd.concat([df_nom[df_nom.index < new_nom.index[0]], new_nom]).sort_index()
        df_adj = pd.concat([df_adj[df_adj.index < new_adj.index[0]], new_adj]).sort_index()
        
        # Deduplicate indices just in case
        df_nom = df_nom[~df_nom.index.duplicated(keep='last')]
        df_adj = df_adj[~df_adj.index.duplicated(keep='last')]

        # Save to disk
        df_nom.to_csv(nom_file)
        df_adj.to_csv(adj_file)
        
        print(f"✅ Database updated. Last date: {df_nom.index.max().date()}")
        return True
    except Exception as e:
        print(f"❌ Update failed: {e}")
        return False

if __name__ == "__main__":
    update_database()
