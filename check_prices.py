import yfinance as yf
import pandas as pd

def check():
    ticker = "FROTO.IS"
    data = yf.download(ticker, start="2024-01-01", progress=False)
    
    print("Columns available:", data.columns.tolist())
    
    # yfinance sometimes uses 'Adj Close' or just 'Close' depending on version/settings
    # Usually it's 'Close' and 'Adj Close'
    cols = [c for c in data.columns if 'Close' in c]
    print("\nLATEST DATA:")
    print(data[cols].tail(5))

if __name__ == "__main__":
    check()
