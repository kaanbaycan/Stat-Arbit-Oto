import pandas as pd
import numpy as np

def cross_check():
    # 1. Load yfinance data
    yf_df = pd.read_excel('automotive_data_check.xlsx', index_col='Date', parse_dates=True)
    
    # 2. Load Local Excel data
    file_path = 'veri_havuzu-2.xlsx'
    xl = pd.ExcelFile(file_path)
    local_dfs = []
    for sheet in xl.sheet_names:
        temp_df = pd.read_excel(file_path, sheet_name=sheet)
        if len(temp_df.columns) < 2: continue
        col_name = temp_df.columns[1]
        name = col_name.split(' - ')[1] if ' - ' in col_name else col_name
        # Match names to yf columns
        if name not in yf_df.columns: continue
        temp_df.columns = ['Date', name]
        temp_df['Date'] = pd.to_datetime(temp_df['Date'], format='%d-%m-%Y')
        temp_df.set_index('Date', inplace=True)
        local_dfs.append(temp_df)
    
    local_df = pd.concat(local_dfs, axis=1).sort_index()
    
    # 3. Find Overlapping Dates
    common_dates = yf_df.index.intersection(local_df.index)
    
    if len(common_dates) == 0:
        print("No overlapping dates found between the two files.")
        return

    print(f"Comparing data for {len(common_dates)} overlapping trading days...")
    print("-" * 50)
    
    for stock in yf_df.columns:
        if stock in local_df.columns:
            diff = (yf_df.loc[common_dates, stock] - local_df.loc[common_dates, stock]).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            print(f"STOCK: {stock}")
            print(f"  Max Price Difference: {max_diff:.4f}")
            print(f"  Avg Price Difference: {mean_diff:.4f}")
            
            # Check for significant discrepancies (> 0.1 TL)
            sig_diffs = diff[diff > 0.1]
            if not sig_diffs.empty:
                print(f"  Significant Differences Found: {len(sig_diffs)} days")
            else:
                print("  No significant price differences found.")
            print("-" * 50)

if __name__ == "__main__":
    cross_check()
