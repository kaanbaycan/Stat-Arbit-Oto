import pandas as pd
import numpy as np
import scipy.stats as stats

def run_backtest(df, initial_capital=100000, window=30, entry_z=-2.0, exit_z=0.5):
    # Calculate relative performance
    # For each stock, calculate Ratio = Stock_Price / Index (Mean of all 3)
    mean_price = df.mean(axis=1)
    ratios = df.divide(mean_price, axis=0)
    
    # Calculate Rolling Mean and Std for Ratios
    rolling_mean = ratios.rolling(window=window).mean()
    rolling_std = ratios.rolling(window=window).std()
    
    # Calculate Z-Scores
    z_scores = (ratios - rolling_mean) / rolling_std
    
    # Backtest logic
    cash = initial_capital
    positions = {col: 0 for col in df.columns} # Shares held
    current_stock = None
    
    history = []
    
    for date, row in df.iterrows():
        daily_z = z_scores.loc[date]
        prices = row
        
        # Calculate other stocks sum for Buy Price calculation
        others_sum = {}
        for col in df.columns:
            others_sum[col] = prices.sum() - prices[col]
            
        # Confidence Level
        # Map Z <= entry_z to 50-100%, Z=0 to 0%
        # Let's say Confidence = min(100, max(0, abs(daily_z) * 25)) if daily_z < 0 else 0
        confidences = daily_z.apply(lambda z: min(100, max(0, abs(z) * 25)) if z < 0 else 0)
        
        # Buy Price for next day (approximate, using current others_sum)
        buy_prices = {}
        for col in df.columns:
            target_ratio = entry_z * rolling_std.loc[date, col] + rolling_mean.loc[date, col]
            # P_new = target_ratio * (P_new + Others) / 3
            # 3*P_new = target_ratio * P_new + target_ratio * Others
            # P_new * (3 - target_ratio) = target_ratio * Others
            # P_new = (target_ratio * Others) / (3 - target_ratio)
            if 3 - target_ratio > 0:
                buy_prices[col] = (target_ratio * others_sum[col]) / (3 - target_ratio)
            else:
                buy_prices[col] = np.nan
        
        # Check for exit
        if current_stock is not None:
            if daily_z[current_stock] >= exit_z:
                cash = positions[current_stock] * prices[current_stock]
                positions[current_stock] = 0
                current_stock = None
        
        # Check for entry if in cash
        if current_stock is None:
            min_z = daily_z.min()
            if min_z <= entry_z:
                best_stock = daily_z.idxmin()
                positions[best_stock] = cash / prices[best_stock]
                cash = 0
                current_stock = best_stock
        
        # Calculate total value
        total_value = cash
        for stock, shares in positions.items():
            total_value += shares * prices[stock]
            
        # Record current state
        state = {
            'Date': date, 
            'TotalValue': total_value, 
            'Cash': cash, 
            'InPosition': current_stock
        }
        for stock in df.columns:
            state[f'{stock}_Price'] = prices[stock]
            state[f'{stock}_Z'] = daily_z[stock]
            state[f'{stock}_Conf'] = confidences[stock]
            state[f'{stock}_BuyPrice'] = buy_prices[stock]
            state[f'{stock}_Shares'] = positions[stock]
            
        history.append(state)
        
    result_df = pd.DataFrame(history).set_index('Date')
    return result_df

if __name__ == "__main__":
    df = pd.read_csv('cleaned_data.csv', index_col='Date', parse_dates=True)
    results = run_backtest(df)
    print(results[['TotalValue', 'InPosition']].tail())
