import yfinance as yf

def test():
    t = yf.Ticker("FROTO.IS")
    try:
        # fast_info is the modern attribute for quick stats
        last_price = t.fast_info['lastPrice']
        print(f"FROTO Last Price (Fast Info): {last_price}")
    except Exception as e:
        print(f"Fast info failed: {e}")

if __name__ == "__main__":
    test()
