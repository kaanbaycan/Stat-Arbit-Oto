import yfinance as yf
import pandas as pd

def check():
    ticker = "FROTO.IS"
    # Fetching a small window to ensure 17th is included
    data = yf.download(ticker, start="2026-04-16", end="2026-04-18", auto_adjust=False)
    
    print(f"--- ALL COLUMNS FOR {ticker} ON 2026-04-17 ---")
    if not data.empty:
        # Filter for just the 17th
        row_17 = data[data.index == "2026-04-17"]
        print(row_17)
    else:
        print("No data found for this range.")

if __name__ == "__main__":
    check()
