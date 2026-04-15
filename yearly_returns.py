import pandas as pd
import numpy as np
import os

def load_data(file_path):
    xl = pd.ExcelFile(file_path)
    dfs = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        name = df.columns[1].split(' - ')[1] if ' - ' in df.columns[1] else df.columns[1]
        df.columns = ['Date', name]
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df.set_index('Date', inplace=True)
        dfs.append(df)
    return pd.concat(dfs, axis=1).ffill().dropna()

def run_yearly_backtest(df, year, window=30, entry_z=-2.0, exit_z=0.5):
    # Calculate global stats for accurate signals
    mean_price = df.mean(axis=1)
    ratios = df.divide(mean_price, axis=0)
    rolling_mean = ratios.rolling(window=window).mean()
    rolling_std = ratios.rolling(window=window).std()
    z_scores = (ratios - rolling_mean) / rolling_std
    
    # Filter for the specific year
    df_year = df[df.index.year == year]
    if df_year.empty: return None
    
    start_date = df_year.index[0]
    cash = 100000
    positions = {col: 0 for col in df.columns}
    current_stock = None
    
    for date, row in df.iterrows():
        if date < start_date: continue
        if date.year > year: break
        
        daily_z = z_scores.loc[date]
        prices = row
        
        if current_stock is not None:
            if daily_z[current_stock] >= exit_z:
                cash = positions[current_stock] * prices[current_stock]
                positions[current_stock] = 0
                current_stock = None
        
        if current_stock is None:
            min_z = daily_z.min()
            if min_z <= entry_z:
                best_stock = daily_z.idxmin()
                positions[best_stock] = cash / prices[best_stock]
                cash = 0
                current_stock = best_stock
                
    final_val = cash + sum(positions[s] * df_year.iloc[-1][s] for s in df.columns)
    return ((final_val - 100000) / 100000) * 100

if __name__ == "__main__":
    df = load_data('veri_havuzu-2.xlsx')
    years = sorted(df.index.year.unique())
    
    print(f"{'Year':<10} | {'Return (%)':<12}")
    print("-" * 25)
    
    total_compounded = 1.0
    for yr in years:
        ret = run_yearly_backtest(df, yr)
        if ret is not None:
            print(f"{yr:<10} | {ret:>10.2f}%")
            total_compounded *= (1 + ret/100)
            
    print("-" * 25)
    print(f"Total Multiplier since {years[0]}: {total_compounded:.2f}x")
