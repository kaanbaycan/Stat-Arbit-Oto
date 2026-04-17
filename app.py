import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yfinance as yf
from datetime import datetime, timedelta
import scipy.stats as stats
import time

# Set page config
st.set_page_config(page_title="Ultra-Fast Stat-Arb Dashboard", layout="wide")

# Custom CSS for Terminal look
st.markdown("""
<style>
    .terminal {
        background-color: #0d0d0d;
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .terminal-header {
        color: #ff9900;
        border-bottom: 1px solid #333;
        margin-bottom: 10px;
        padding-bottom: 5px;
        font-weight: bold;
    }
    .sync-info { color: #00e5ff; font-size: 0.8em; margin-bottom: 10px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

SECTORS = {
    "Automotive": {"FROTO": "FROTO.IS", "DOAS": "DOAS.IS", "TOASO": "TOASO.IS"},
    "Food Retail": {"SOKM": "SOKM.IS", "BIMAS": "BIMAS.IS", "MGROS": "MGROS.IS"},
    "Aviation": {"THYAO": "THYAO.IS", "PGSUS": "PGSUS.IS", "TAVHL": "TAVHL.IS"},
    "Steel & Iron": {"EREGL": "EREGL.IS", "KRDMD": "KRDMD.IS", "ISDMR": "ISDMR.IS"}
}

ALL_STOCKS = []
for s in SECTORS.values(): 
    ALL_STOCKS.extend(list(s.keys()))

TICKER_MAP = {}
for s in SECTORS.values(): 
    TICKER_MAP.update(s)
TICKER_MAP.update({"USDTRY": "USDTRY=X", "XU100": "^XU100", "GOLD": "GC=F"})

# --- OPTIMIZED DATA LOADING ---
@st.cache_data(ttl=1800)
def load_all_data():
    if not os.path.exists("db_nominal.csv") or not os.path.exists("db_adjusted.csv"):
        return None, "Database missing"

    df_nom = pd.read_csv("db_nominal.csv", index_col='Date', parse_dates=True)
    df_adj = pd.read_csv("db_adjusted.csv", index_col='Date', parse_dates=True)
    
    last_date = df_nom.index.max()
    today = datetime.now()
    sync_msg = f"Database: {len(df_nom)} days (Last: {last_date.date()})"
    
    if (today - last_date).days >= 1:
        try:
            raw = yf.download(list(TICKER_MAP.values()), start=last_date, progress=False, auto_adjust=False)
            if not raw.empty:
                new_nom = raw['Close']
                new_adj = raw['Adj Close']
                inv_map = {v: k for k, v in TICKER_MAP.items()}
                new_nom.columns = [inv_map.get(c, c) for c in new_nom.columns]
                new_adj.columns = [inv_map.get(c, c) for c in new_adj.columns]
                
                added = new_nom[new_nom.index > last_date]
                if not added.empty:
                    df_nom = pd.concat([df_nom, added]).sort_index()
                    df_adj = pd.concat([df_adj, new_adj[new_adj.index > last_date]]).sort_index()
                    df_nom.to_csv("db_nominal.csv")
                    df_adj.to_csv("db_adjusted.csv")
                    sync_msg += f" | ⚡ Synced +{len(added)} days live"
        except:
            sync_msg += " | ⚠️ Live sync failed"
    
    # --- CRITICAL DATA CLEANING ---
    # Forward fill to handle small gaps
    df_nom = df_nom.ffill()
    df_adj = df_adj.ffill()
    
    # Remove trailing rows that are missing STOCK prices (yfinance often returns NaN for today)
    # We check only columns that are part of our SECTORS
    stock_cols = [c for c in ALL_STOCKS if c in df_nom.columns]
    valid_mask = df_nom[stock_cols].notna().any(axis=1)
    df_nom = df_nom[valid_mask]
    df_adj = df_adj[valid_mask]
    
    return {"nom": df_nom, "adj": df_adj}, sync_msg

# 3. Model Logic
def run_model(df_nom, df_adj, initial_capital=100000, window=30, entry_z=-2.0, exit_z=0.5, stop_z=-4.0, abs_stop=0.10, interest_rate=0.35):
    if df_nom.empty or len(df_nom) < window: 
        return None
    
    mean_adj = df_adj.mean(axis=1)
    ratios_adj = df_adj.divide(mean_adj, axis=0)
    rolling_mean = ratios_adj.rolling(window=window).mean()
    rolling_std = ratios_adj.rolling(window=window).std()
    z_scores = (ratios_adj - rolling_mean) / rolling_std
    
    half_lives = {}
    for col in df_adj.columns:
        try:
            y = ratios_adj[col].dropna()
            y_lag = y.shift(1).dropna()
            y_curr = y.iloc[1:]
            y_diff = y_curr.values - y_lag.values
            res = stats.linregress(y_lag.values, y_diff)
            beta = res.slope
            half_lives[col] = -np.log(2)/beta if beta < 0 else 99.9
        except:
            half_lives[col] = 99.9

    cash = initial_capital
    positions = {col: 0 for col in df_nom.columns}
    current_stock = None
    entry_p_nom = 0
    prev_date = None
    history = []
    outcomes = {col: [] for col in df_nom.columns}
    
    for date in df_nom.index:
        row_nom = df_nom.loc[date]
        row_adj = df_adj.loc[date]
        daily_z = z_scores.loc[date]
        
        if prev_date is not None and cash > 0:
            days = (date - prev_date).days
            if days > 0: 
                cash *= (1 + interest_rate / 365) ** days
        
        prev_date = date
        reversion_probs = daily_z.apply(lambda z: (1 - stats.norm.cdf(z)) * 100 if pd.notnull(z) else 0)
        
        buy_nom = {}
        sell_nom = {}
        for col in df_nom.columns:
            n = len(df_nom.columns)
            conv = row_nom[col] / row_adj[col] if row_adj[col] != 0 else 1
            tr_buy = entry_z * rolling_std.loc[date, col] + rolling_mean.loc[date, col]
            buy_p_adj = (tr_buy * (row_adj.sum() - row_adj[col])) / (n - tr_buy) if n - tr_buy > 0 else np.nan
            buy_nom[col] = buy_p_adj * conv
            tr_sell = exit_z * rolling_std.loc[date, col] + rolling_mean.loc[date, col]
            sell_p_adj = (tr_sell * (row_adj.sum() - row_adj[col])) / (n - tr_sell) if n - tr_sell > 0 else np.nan
            sell_nom[col] = sell_p_adj * conv

        if current_stock is not None:
            p_loss = (row_nom[current_stock] - entry_p_nom) / entry_p_nom
            if pd.notnull(daily_z[current_stock]):
                if daily_z[current_stock] >= exit_z or daily_z[current_stock] <= stop_z or p_loss <= -abs_stop:
                    outcomes[current_stock].append(1 if daily_z[current_stock] >= exit_z else 0)
                    cash = positions[current_stock] * row_nom[current_stock]
                    positions[current_stock] = 0
                    current_stock = None
                    entry_p_nom = 0
        
        if current_stock is None:
            if not daily_z.dropna().empty:
                min_z = daily_z.min()
                if min_z <= entry_z:
                    best_stock = daily_z.idxmin()
                    positions[best_stock] = cash / row_nom[best_stock]
                    cash = 0
                    current_stock = best_stock
                    entry_p_nom = row_nom[best_stock]
        
        total_val = cash + sum(positions[s] * row_nom[s] for s in df_nom.columns)
        state = {'Date': date, 'TotalValue': total_val, 'Cash': cash, 'InPosition': current_stock, 'EntryPrice': entry_p_nom}
        for s in df_nom.columns:
            state[f'{s}_Price'] = row_nom[s]
            state[f'{s}_Z'] = daily_z[s]
            state[f'{s}_RevProb'] = reversion_probs[s]
            state[f'{s}_BuyPrice'] = buy_nom.get(s, np.nan)
            state[f'{s}_SellPrice'] = sell_nom.get(s, np.nan)
        history.append(state)
        
    result_df = pd.DataFrame(history).set_index('Date')
    win_rates = {col: (sum(outcomes[col])/len(outcomes[col])*100 if outcomes[col] else 0) for col in df_nom.columns}
    return result_df, half_lives, win_rates

# --- App UI ---
master_data, sync_msg = load_all_data()

if master_data:
    st.sidebar.markdown(f"<div class='sync-info'>{sync_msg}</div>", unsafe_allow_html=True)
    st.sidebar.header("Global Settings")
    window = st.sidebar.slider("Rolling Window", 10, 100, 30)
    entry_z = st.sidebar.slider("Entry Z (Buy)", -5.0, -1.0, -2.0)
    exit_z = st.sidebar.slider("Exit Z (Sell)", -1.0, 2.0, 0.5)
    stop_z = st.sidebar.slider("Stop Z (Relative)", -10.0, -3.0, -4.0)
    abs_stop = st.sidebar.slider("Absolute Stop Loss %", 0.01, 0.30, 0.10)
    interest_rate = st.sidebar.slider("Idle Cash Yearly Interest", 0.0, 1.0, 0.35)

    # 0. EXECUTIVE SUMMARY RADAR
    st.subheader("🎯 Market Opportunity Radar (Nominal Prices)")
    radar_data = []
    for s_name, s_tickers in SECTORS.items():
        stocks = [s for s in s_tickers.keys() if s in master_data['nom'].columns]
        if not stocks: 
            continue
        s_nom = master_data['nom'][stocks]
        s_adj = master_data['adj'][stocks]
        m_res = run_model(s_nom, s_adj, 100000, window, entry_z, exit_z, stop_z, abs_stop, interest_rate)
        if m_res:
            s_res, _, _ = m_res
            latest_r = s_res.iloc[-1]
            active_r = latest_r['InPosition']
            if pd.notnull(active_r):
                radar_data.append({
                    'Sector': s_name, 
                    'Status': '🔴 HOLDING', 
                    'Stock': active_r, 
                    'Price': f"{latest_r[f'{active_r}_Price']:,.2f}", 
                    'Rev Prob': f"{latest_r[f'{active_r}_RevProb']:.1f}%", 
                    'Target': f"{latest_r[f'{active_r}_SellPrice']:,.2f}"
                })
            else:
                z_cols = [c for c in latest_r.index if c.endswith('_Z')]
                z_series = pd.to_numeric(latest_r[z_cols], errors='coerce').dropna()
                if not z_series.empty:
                    try:
                        best_ticker_z = z_series.idxmin()
                        min_z_s = best_ticker_z.replace('_Z', '')
                        radar_data.append({
                            'Sector': s_name, 
                            'Status': '🟢 MONITORING', 
                            'Stock': min_z_s, 
                            'Price': f"{latest_r[f'{min_z_s}_Price']:,.2f}", 
                            'Rev Prob': f"{latest_r[f'{min_z_s}_RevProb']:.1f}%", 
                            'Target': f"{latest_r[f'{min_z_s}_BuyPrice']:,.2f} (BUY)"
                        })
                    except: 
                        pass
    
    if radar_data: 
        st.table(pd.DataFrame(radar_data).set_index('Sector'))

    st.markdown("---")
    selected_sector = st.sidebar.selectbox("Select Detail Sector", list(SECTORS.keys()))
    if st.sidebar.button("🔄 Force Fresh Refresh"): 
        st.cache_data.clear()
        st.rerun()

    years = sorted(master_data['nom'].index.year.unique().tolist())
    year_range = st.sidebar.slider("Analysis Year Range", min_value=years[0], max_value=years[-1], value=(years[0], years[-1]))
    stocks_detail = list(SECTORS[selected_sector].keys())
    d_nom = master_data['nom'][stocks_detail][(master_data['nom'].index.year >= year_range[0]) & (master_data['nom'].index.year <= year_range[1])]
    d_adj = master_data['adj'][stocks_detail][(master_data['adj'].index.year >= year_range[0]) & (master_data['adj'].index.year <= year_range[1])]
    
    res_out = run_model(d_nom, d_adj, 100000, window, entry_z, exit_z, stop_z, abs_stop, interest_rate)
    if res_out:
        results, half_lives, win_rates = res_out
        latest = results.iloc[-1]
        active_pos = latest['InPosition']
        
        st.subheader(f"📡 {selected_sector} Detail Terminal")
        terminal_html = f"<div class='terminal'><div class='terminal-header'>DETAIL TERMINAL | {results.index[-1].strftime('%Y-%m-%d')}</div>"
        if pd.notnull(active_pos):
            p = (latest[f'{active_pos}_Price'] - latest['EntryPrice']) / latest['EntryPrice'] * 100
            terminal_html += f"<div>[ACTIVE]: <span style='color:#ffcc00'>{active_pos}</span> | PROFIT: <span style='color:{'#00ff00' if p >= 0 else '#ff5555'}'>{p:+.2f}%</span></div>"
        else: 
            terminal_html += "<div>[ACTIVE]: <span style='color:#888'>CASH_LIQUID</span></div>"
        
        terminal_html += "<br><div style='display: grid; grid-template-columns: 1fr 1fr 1.2fr 1.2fr 1fr 1fr; border-bottom: 1px solid #333;'>"
        terminal_html += "<span>TICKER</span><span>PRICE</span><span>T_BUY</span><span>T_SELL</span><span>WIN%</span><span>REV%</span></div>"
        for s in stocks_detail:
            terminal_html += f"<div style='display: grid; grid-template-columns: 1fr 1fr 1.2fr 1.2fr 1fr 1fr; padding-top:5px;'>"
            terminal_html += f"<span>{s}</span><span>{latest[f'{s}_Price']:,.2f}</span><span style='color:#00e5ff'>{latest[f'{s}_BuyPrice']:,.2f}</span><span style='color:#ff5555'>{latest[f'{s}_SellPrice']:,.2f}</span><span>{win_rates[s]:.0f}%</span><span>{latest[f'{s}_RevProb']:.1f}%</span></div>"
        st.html(terminal_html + "</div>")
        
        m1, m2 = st.columns(2)
        m1.metric("Portfolio Value", f"{latest['TotalValue']:,.0f} TL")
        m2.metric("Nominal Profit", f"{(latest['TotalValue']-100000)/1000:.1f}%")

        tabs = st.tabs(["Sector Comparison", "History", "Stats"] + stocks_detail)
        with tabs[0]:
            comp_results = []
            for s_n, s_t in SECTORS.items():
                stocks_c = [s for s in s_t.keys() if s in master_data['nom'].columns]
                if not stocks_c: 
                    continue
                sc_nom = master_data['nom'][stocks_c][(master_data['nom'].index.year >= year_range[0]) & (master_data['nom'].index.year <= year_range[1])]
                sc_adj = master_data['adj'][stocks_c][(master_data['adj'].index.year >= year_range[0]) & (master_data['adj'].index.year <= year_range[1])]
                model_res = run_model(sc_nom, sc_adj, 100000, window, entry_z, exit_z, stop_z, abs_stop, interest_rate)
                if model_res: 
                    s_r, _, _ = model_res
                    comp_results.append({'Sector': s_n, 'Series': s_r['TotalValue'], 'Profit %': (s_r['TotalValue'].iloc[-1]-100000)/1000})
            if comp_results:
                fig_c, ax_c = plt.subplots(figsize=(10, 4))
                for r in comp_results: 
                    ax_c.plot(r['Series'].index, r['Series'], label=f"{r['Sector']} ({r['Profit %']:.1f}%)")
                ax_c.legend()
                st.pyplot(fig_c)
        with tabs[1]:
            moves = []
            prev = None
            for d, r in results.iterrows():
                curr_p = r['InPosition']
                if curr_p != prev:
                    if pd.notnull(prev): 
                        moves.append({'Date': d.date(), 'Ticker': prev, 'Action': 'SELL', 'Price': r[f'{prev}_Price']})
                    if pd.notnull(curr_p): 
                        moves.append({'Date': d.date(), 'Ticker': curr_p, 'Action': 'BUY', 'Price': r[f"{curr_p}_Price"]})
                prev = curr_p
            if moves:
                m_df = pd.DataFrame(moves)
                m_df['Profit %'] = ""
                m_df['Hold Duration'] = ""
                for i in range(len(m_df)):
                    if m_df.iloc[i]['Action'] == 'SELL':
                        for j in range(i-1, -1, -1):
                            if m_df.iloc[j]['Action'] == 'BUY' and m_df.iloc[j]['Ticker'] == m_df.iloc[i]['Ticker']:
                                bp, sp = m_df.iloc[j]['Price'], m_df.iloc[i]['Price']
                                m_df.at[i, 'Profit %'] = f"{(sp-bp)/bp*100:+.2f}%"
                                m_df.at[i, 'Hold Duration'] = f"{(m_df.iloc[i]['Date']-m_df.iloc[j]['Date']).days} days"
                                break
                st.dataframe(m_df.sort_values('Date', ascending=False), use_container_width=True)
        with tabs[2]: 
            st.table(pd.DataFrame({'Win Rate (%)': win_rates, 'Half-Life (Days)': half_lives}))
        for i, name in enumerate(stocks_detail):
            with tabs[i+3]:
                c1, c2 = st.columns(2)
                with c1:
                    fig_p, ax_p = plt.subplots()
                    ax_p.plot(results.index, results[f'{name}_Price'], color='gray', alpha=0.5)
                    pos = results['InPosition']
                    b = results.index[(pos == name) & (pos.shift(1) != name)]
                    s = results.index[(pos != name) & (pos.shift(1) == name)]
                    ax_p.scatter(b, results.loc[b, f'{name}_Price'], color='green', marker='^')
                    ax_p.scatter(s, results.loc[s, f'{name}_Price'], color='red', marker='v')
                    st.pyplot(fig_p)
                with c2:
                    fig_z, ax_z = plt.subplots()
                    ax_z.plot(results.index, results[f'{name}_Z'], color='purple')
                    ax_z.axhline(entry_z, color='green', linestyle='--')
                    ax_z.axhline(exit_z, color='red', linestyle='--')
                    st.pyplot(fig_z)
else: 
    st.error("Data error.")
