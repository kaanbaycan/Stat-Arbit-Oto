import pandas as pd
import os
from datetime import datetime, timedelta
import yfinance as yf
from update_db import update_database, TICKERS, INV_MAP

SECTORS = {
    "Automotive": {"FROTO": "FROTO.IS", "DOAS": "DOAS.IS", "TOASO": "TOASO.IS"},
    "Food Retail": {"SOKM": "SOKM.IS", "BIMAS": "BIMAS.IS", "MGROS": "MGROS.IS"},
    "Aviation": {"THYAO": "THYAO.IS", "PGSUS": "PGSUS.IS", "TAVHL": "TAVHL.IS"},
    "Steel & Iron": {"EREGL": "EREGL.IS", "KRDMD": "KRDMD.IS", "ISDMR": "ISDMR.IS"}
}
ALL_STOCKS_MAP = {}
for s in SECTORS.values(): ALL_STOCKS_MAP.update(s)

def load_all_data_test():
    nom_file = "db_nominal.csv"
    adj_file = "db_adjusted.csv"
    
    # 1. Initial Check & Update
    # In test, we just call the function
    success = update_database()
    
    # 2. Read existing data
    if not os.path.exists(nom_file) or not os.path.exists(adj_file):
        return None, "❌ Database Error"

    df_nom = pd.read_csv(nom_file, index_col='Date', parse_dates=True)
    df_adj = pd.read_csv(adj_file, index_col='Date', parse_dates=True)
    
    stock_cols = list(ALL_STOCKS_MAP.keys())
    last_dt = df_nom.index.max()
    
    # Format the sync message
    current_time = datetime.now().strftime("%H:%M:%S")
    if hasattr(last_dt, 'hour') and last_dt.hour != 0:
        sync_msg = f"Last Data: {last_dt.strftime('%Y-%m-%d %H:%M')} | Refreshed: {current_time}"
    else:
        sync_msg = f"Last Data: {last_dt.date()} | Refreshed: {current_time}"
        
    if not success:
        sync_msg += " | ⚠️ Sync Issue"
    else:
        sync_msg += " | ⚡ Live (G-Finance)"

    # Final Cleanup
    # Ensure they are aligned and have the same index
    df_nom_orig = df_nom.copy()
    df_nom = df_nom.ffill().dropna(subset=stock_cols, how='all')
    df_adj = df_adj.ffill().dropna(subset=stock_cols, how='all')
    
    # Intersect indices to ensure perfect alignment
    common_index = df_nom.index.intersection(df_adj.index)
    df_nom = df_nom.loc[common_index]
    df_adj = df_adj.loc[common_index]
    
    return {"nom": df_nom, "adj": df_adj, "orig": df_nom_orig}, sync_msg

res, msg = load_all_data_test()
print(f"Msg: {msg}")
print(f"DF Index Max: {res['nom'].index.max()}")
print(f"Orig DF Index Max: {res['orig'].index.max()}")
print(f"Stock columns used for dropna: {list(ALL_STOCKS_MAP.keys())}")
print(f"Row for 2026-04-20 in orig:\n{res['orig'].tail(1)}")
