import yfinance as yf
import pandas as pd

def debug():
    print("Checking FROTO.IS from yfinance...")
    raw = yf.download("FROTO.IS", start="2026-04-10", auto_adjust=False)
    print("\nRAW DATA:")
    print(raw[['Close', 'Adj Close']].tail(5))
    
    print("\nCOLUMNS FOUND:", raw.columns.tolist())

if __name__ == "__main__":
    debug()
