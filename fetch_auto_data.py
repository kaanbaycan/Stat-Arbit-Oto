import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_and_export_auto():
    # 1. Define Tickers
    tickers = {
        "FROTO": "FROTO.IS", 
        "DOAS": "DOAS.IS", 
        "TOASO": "TOASO.IS"
    }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching data from yfinance...")
    
    try:
        # 2. Download Close Prices
        data = yf.download(list(tickers.values()), start="2016-01-01", progress=True)['Close']
        
        if data.empty:
            print("Error: No data found.")
            return

        # 3. Clean Column Names
        data.columns = [c.replace('.IS', '') for c in data.columns]
        
        # 4. Final Processing
        # Forward fill to handle non-trading day gaps, then drop remaining NaNs
        combined = data.ffill().dropna()
        
        # 5. Export to Excel
        output_file = "automotive_data_check.xlsx"
        combined.to_excel(output_file)
        
        print("-" * 30)
        print(f"SUCCESS: Data saved to {output_file}")
        print(f"Total trading days: {len(combined)}")
        print(f"Date range: {combined.index.min().date()} to {combined.index.max().date()}")
        print("-" * 30)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_and_export_auto()
