import yfinance as yf
import pandas as pd

def check():
    ticker = "FROTO.IS"
    data = yf.download(ticker, start="2024-01-01", progress=False)
    
    print(f"YFINANCE DATA FOR {ticker}:")
    print(data[['Close', 'Adj Close']].tail(5))
    
    print("\nLOCAL EXCEL DATA (Last 5 days):")
    file_path = 'veri_havuzu-2.xlsx'
    xl = pd.ExcelFile(file_path)
    sheet = xl.sheet_names[0] # Assuming FROTO is first or similar
    df = pd.read_excel(file_path, sheet_name=sheet)
    print(df.tail(5))

if __name__ == "__main__":
    check()
