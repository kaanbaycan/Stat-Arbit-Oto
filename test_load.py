from app import load_all_data
import pandas as pd
from datetime import datetime

data, msg = load_all_data()
print(f"Message: {msg}")
if data:
    df = data['nom']
    print(f"Last date in df: {df.index.max()}")
    print("Last row values:")
    print(df.iloc[-1])
else:
    print("Data loading failed")
