import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_backtest_2025(df, initial_capital=100000, window=30, entry_z=-2.0, exit_z=0.5):
    # Calculate relative performance (Full history for warm-up)
    mean_price = df.mean(axis=1)
    ratios = df.divide(mean_price, axis=0)
    rolling_mean = ratios.rolling(window=window).mean()
    rolling_std = ratios.rolling(window=window).std()
    z_scores = (ratios - rolling_mean) / rolling_std
    
    # Filter for 2025 (and a bit of 2024 for the start price)
    df_2025 = df[df.index.year == 2025]
    if df_2025.empty:
        return None, "No data for 2025"
    
    start_date = df_2025.index[0]
    
    # Reset simulation to start in 2025 with 100k
    cash = initial_capital
    positions = {col: 0 for col in df.columns} 
    current_stock = None
    
    history = []
    
    # Loop through the full range but only act/record in 2025
    for date, row in df.iterrows():
        if date < start_date:
            continue
        if date.year > 2025:
            break
            
        daily_z = z_scores.loc[date]
        prices = row
        
        # Trade Logic
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
        
        total_value = cash + sum(positions[s] * prices[s] for s in df.columns)
            
        state = {'Date': date, 'TotalValue': total_value, 'Cash': cash, 'InPosition': current_stock}
        for stock in df.columns:
            state[f'{stock}_Price'] = prices[stock]
        history.append(state)
        
    return pd.DataFrame(history).set_index('Date'), None

def generate_report():
    df = pd.read_csv('cleaned_data.csv', index_col='Date', parse_dates=True)
    results, err = run_backtest_2025(df)
    
    if err:
        print(err)
        return

    # Calculations
    start_val = 100000
    end_val = results['TotalValue'].iloc[-1]
    roic = (end_val - start_val) / start_val * 100
    
    # Create Figure
    fig = plt.figure(figsize=(12, 10), facecolor='#f0f0f0')
    
    # 1. Equity Curve
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(results.index, results['TotalValue'], color='#2ca02c', linewidth=2.5, label='Portfolio Value')
    ax1.fill_between(results.index, 100000, results['TotalValue'], color='#2ca02c', alpha=0.1)
    ax1.axhline(100000, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('2025 Portfolio Performance (Starting 100k TL)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value (TL)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Stats Summary Text
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax2.axis('off')
    stats_text = (
        f"2025 ROIC REPORT\n"
        f"-----------------\n"
        f"Initial Capital: 100,000.00 TL\n"
        f"Final Value: {end_val:,.2f} TL\n"
        f"Total Profit: {end_val - start_val:,.2f} TL\n\n"
        f"ROIC: {roic:.2f}%\n"
        f"Year: 2025\n"
        f"Strategy: Long-Only Stat-Arb"
    )
    ax2.text(0.1, 0.5, stats_text, fontsize=12, family='monospace', verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    # 3. Position Allocation (Simplified)
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    pos_counts = results['InPosition'].fillna('Cash').value_counts()
    ax3.pie(pos_counts, labels=pos_counts.index, autopct='%1.1f%%', colors=['#ff7f0e', '#1f77b4', '#d62728', '#9467bd'])
    ax3.set_title('2025 Time Allocation')

    # 4. Stock Prices with Signals
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    # Plot normalized prices for comparison
    for stock in df.columns:
        norm_price = (results[f'{stock}_Price'] / results[f'{stock}_Price'].iloc[0]) * 100
        ax4.plot(results.index, norm_price, label=stock, alpha=0.7)
    
    ax4.set_title('Relative Stock Performance (Base 100)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('roic_2025_report.png', dpi=300)
    print(f"Report saved as roic_2025_report.png. ROIC: {roic:.2f}%")

if __name__ == "__main__":
    generate_report()
