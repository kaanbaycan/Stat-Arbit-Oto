import yfinance as yf
import pandas as pd
from datetime import datetime
import os

TICKERS = {
    "FROTO": "FROTO.IS", "DOAS": "DOAS.IS", "TOASO": "TOASO.IS",
    "SOKM": "SOKM.IS", "BIMAS": "BIMAS.IS", "MGROS": "MGROS.IS",
    "THYAO": "THYAO.IS", "PGSUS": "PGSUS.IS", "TAVHL": "TAVHL.IS",
    "EREGL": "EREGL.IS", "KRDMD": "KRDMD.IS", "ISDMR": "ISDMR.IS",
    "USDTRY": "USDTRY=X", "XU100": "^XU100", "GOLD": "GC=F"
}

def build_database():
    print("🚀 Initializing Master Database Build (2016 - Present)...")
    
    # 1. Download ALL data in one go
    # auto_adjust=False to get both Close and Adj Close
    raw = yf.download(list(TICKERS.values()), start="2016-01-01", progress=True, auto_adjust=False)
    
    if raw.empty:
        print("❌ Error: No data received from yfinance.")
        return

    # 2. Extract Nominal (Close) and Adjusted (Adj Close)
    df_nom = raw['Close']
    df_adj = raw['Adj Close']
    
    # Clean Column Names (e.g., 'FROTO.IS' -> 'FROTO')
    inv_map = {v: k for k, v in TICKERS.items()}
    df_nom.columns = [inv_map.get(c, c) for c in df_nom.columns]
    df_adj.columns = [inv_map.get(c, c) for c in df_adj.columns]
    
    # 3. Save as CSVs (Faster than Excel)
    df_nom.to_csv("db_nominal.csv")
    df_adj.to_csv("db_adjusted.csv")
    
    print("-" * 40)
    print(f"✅ SUCCESS: Database built.")
    print(f"📁 db_nominal.csv: {len(df_nom)} rows")
    print(f"📁 db_adjusted.csv: {len(df_adj)} rows")
    print(f"📅 Last Date: {df_nom.index.max().date()}")
    print("-" * 40)

if __name__ == "__main__":
    build_database()
