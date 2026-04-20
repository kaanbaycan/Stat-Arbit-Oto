import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yfinance as yf
from datetime import datetime, timedelta
import scipy.stats as stats
import time

import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Real-Time Stat-Arb Dashboard", layout="wide")

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

ALL_STOCKS_MAP = {}
for s in SECTORS.values(): ALL_STOCKS_MAP.update(s)

TICKER_MAP = ALL_STOCKS_MAP.copy()
TICKER_MAP.update({"USDTRY": "USDTRY=X", "XU100": "^XU100", "GOLD": "GC=F"})

# --- OPTIMIZED DATA LOADING ---
from update_db import update_database, TICKERS, INV_MAP

@st.cache_data(ttl=300) # Reduced to 5 minutes for better intraday flow
def load_all_data():
    nom_file = "db_nominal.csv"
    adj_file = "db_adjusted.csv"
    
    # 1. Initial Check & Update
    # Always attempt an update, update_database handles the logic of whether it's needed
    with st.spinner("📡 Syncing latest market data..."):
        success = update_database()
    
    # 2. Read existing data
    if not os.path.exists(nom_file) or not os.path.exists(adj_file):
        return None, "❌ Database Error"

    df_nom = pd.read_csv(nom_file, index_col='Date', parse_dates=True)
    df_adj = pd.read_csv(adj_file, index_col='Date', parse_dates=True)
    
    stock_cols = list(ALL_STOCKS_MAP.keys())
    last_date = df_nom.index.max()
    
    sync_msg = f"DB: {last_date.strftime('%Y-%m-%d %H:%M') if hasattr(last_date, 'hour') else last_date.date()}"
    if not success:
        sync_msg += " | ⚠️ Sync Issue"
    else:
        sync_msg += " | ⚡ Live"

    # Final Cleanup
    df_nom = df_nom.ffill().dropna(subset=stock_cols, how='all')
    df_adj = df_adj.ffill().dropna(subset=stock_cols, how='all')
    return {"nom": df_nom, "adj": df_adj}, sync_msg

# 3. Model Logic
@st.cache_data(show_spinner=False)
def run_model(df_nom, df_adj, initial_capital=100000, window=30, entry_z=-2.0, exit_z=0.5, stop_z=-4.0, abs_stop=0.10, interest_rate=0.35):
    if df_nom.empty or len(df_nom) < window: return None
    
    # Vectorized Pre-calculations
    mean_adj = df_adj.mean(axis=1); ratios_adj = df_adj.divide(mean_adj, axis=0)
    rolling_mean = ratios_adj.rolling(window=window).mean(); rolling_std = ratios_adj.rolling(window=window).std()
    z_scores = (ratios_adj - rolling_mean) / rolling_std
    
    # Optimization: Calculate half-lives only once
    half_lives = {}
    for col in df_adj.columns:
        try:
            y = ratios_adj[col].dropna(); y_lag = y.shift(1).dropna(); y_curr = y.iloc[1:]; y_diff = y_curr.values - y_lag.values
            res = stats.linregress(y_lag.values, y_diff); beta = res.slope
            half_lives[col] = -np.log(2)/beta if beta < 0 else 99.9
        except: half_lives[col] = 99.9

    # Vectorized Reversion Probabilities (pre-calculate for all dates)
    # Using scipy.stats.norm.cdf is fast but vectorizing over the whole dataframe is faster
    rev_probs = (1 - stats.norm.cdf(z_scores.fillna(0))) * 100
    
    # Pre-calculate Buy/Sell prices (Targets) for all dates to avoid the inner loop
    n_stocks = len(df_nom.columns)
    sum_adj = df_adj.sum(axis=1)
    conv_ratios = df_nom / df_adj
    
    buy_targets = (entry_z * rolling_std + rolling_mean)
    sell_targets = (exit_z * rolling_std + rolling_mean)
    
    # Simplified buy/sell price calculation
    # P_adj = (Target * (Sum - P_adj)) / (n - Target) => P_adj = (Target * Sum) / n
    # (Approximation for performance, original was more precise but this is very close)
    buy_nom_all = (buy_targets.mul(sum_adj, axis=0) / n_stocks) * conv_ratios
    sell_nom_all = (sell_targets.mul(sum_adj, axis=0) / n_stocks) * conv_ratios

    # --- INITIALIZATION ---
    cash = initial_capital; positions = {col: 0 for col in df_nom.columns}; 
    current_stock = None; entry_p_nom = 0; prev_date = None; history = []; outcomes = {col: [] for col in df_nom.columns}
    
    # Date loop (Hard to vectorize fully due to path dependency)
    # Convert to values for faster access
    dates = df_nom.index
    nom_values = df_nom.values; z_values = z_scores.values; col_names = list(df_nom.columns)
    buy_nom_values = buy_nom_all.values; sell_nom_values = sell_nom_all.values
    rev_prob_values = rev_probs # This is already a numpy array from stats.norm.cdf

    for i in range(len(dates)):
        date = dates[i]; row_nom = nom_values[i]; daily_z = z_values[i]
        
        if prev_date is not None and cash > 0:
            days = (date - prev_date).days
            if days > 0: cash *= (1 + interest_rate / 365) ** days
        prev_date = date
        
        if current_stock is not None:
            idx = col_names.index(current_stock)
            p_loss = (row_nom[idx] - entry_p_nom) / entry_p_nom
            z_val = daily_z[idx]
            if not np.isnan(z_val):
                if z_val >= exit_z or z_val <= stop_z or p_loss <= -abs_stop:
                    outcomes[current_stock].append(1 if z_val >= exit_z else 0)
                    cash = positions[current_stock] * row_nom[idx]; positions[current_stock] = 0; current_stock = None; entry_p_nom = 0
        
        if current_stock is None:
            if not np.isnan(daily_z).all():
                min_idx = np.nanargmin(daily_z)
                min_z = daily_z[min_idx]
                if min_z <= entry_z:
                    best_stock = col_names[min_idx]; positions[best_stock] = cash / row_nom[min_idx]; cash = 0; current_stock = best_stock; entry_p_nom = row_nom[min_idx]
        
        total_val = cash + sum(positions[s] * row_nom[col_names.index(s)] for s in col_names)
        state = {'Date': date, 'TotalValue': total_val, 'Cash': cash, 'InPosition': current_stock, 'EntryPrice': entry_p_nom}
        for j, s in enumerate(col_names):
            state[f'{s}_Price'] = row_nom[j]; state[f'{s}_Z'] = daily_z[j]; state[f'{s}_RevProb'] = rev_prob_values[i, j]
            state[f'{s}_BuyPrice'] = buy_nom_values[i, j]; state[f'{s}_SellPrice'] = sell_nom_values[i, j]
        history.append(state)
        
    result_df = pd.DataFrame(history).set_index('Date')
    win_rates = {col: (sum(outcomes[col])/len(outcomes[col])*100 if outcomes[col] else 0) for col in df_nom.columns}
    return result_df, half_lives, win_rates

# --- App UI ---
master_data, sync_msg = load_all_data()
if master_data:
    st.sidebar.markdown(f"<div class='sync-info'>{sync_msg}</div>", unsafe_allow_html=True)
    st.sidebar.header("Global Settings")
    window = st.sidebar.slider("Rolling Window", 10, 100, 30); entry_z = st.sidebar.slider("Entry Z (Buy)", -5.0, -1.0, -2.0)
    exit_z = st.sidebar.slider("Exit Z (Sell)", -1.0, 2.0, 0.5); stop_z = st.sidebar.slider("Stop Z (Relative)", -10.0, -3.0, -4.0)
    abs_stop = st.sidebar.slider("Absolute Stop Loss %", 0.01, 0.30, 0.10); interest_rate = st.sidebar.slider("Idle Cash Yearly Interest", 0.0, 1.0, 0.35)

    st.subheader("🎯 Market Opportunity Radar")
    radar_data = []
    for s_name, s_tickers in SECTORS.items():
        stocks = [s for s in s_tickers.keys() if s in master_data['nom'].columns]
        if not stocks: continue
        s_nom, s_adj = master_data['nom'][stocks], master_data['adj'][stocks]
        m_res = run_model(s_nom, s_adj, 100000, window, entry_z, exit_z, stop_z, abs_stop, interest_rate)
        if m_res:
            s_res, _, _ = m_res; latest_r = s_res.iloc[-1]; active_r = latest_r['InPosition']
            if pd.notnull(active_r):
                radar_data.append({'Sector': s_name, 'Status': '🔴 HOLDING', 'Stock': active_r, 'Price': f"{latest_r[f'{active_r}_Price']:,.2f}", 'Rev Prob': f"{latest_r[f'{active_r}_RevProb']:.1f}%", 'Target': f"{latest_r[f'{active_r}_SellPrice']:,.2f}"})
            else:
                z_cols = [c for c in latest_r.index if c.endswith('_Z')]
                z_series = pd.to_numeric(latest_r[z_cols], errors='coerce').dropna()
                if not z_series.empty:
                    min_z_s = z_series.idxmin().replace('_Z', '')
                    radar_data.append({'Sector': s_name, 'Status': '🟢 MONITORING', 'Stock': min_z_s, 'Price': f"{latest_r[f'{min_z_s}_Price']:,.2f}", 'Rev Prob': f"{latest_r[f'{min_z_s}_RevProb']:.1f}%", 'Target': f"{latest_r[f'{min_z_s}_BuyPrice']:,.2f} (BUY)"})
    if radar_data: st.table(pd.DataFrame(radar_data).set_index('Sector'))

    st.markdown("---")
    selected_sector = st.sidebar.selectbox("Select Detail Sector", list(SECTORS.keys()))
    if st.sidebar.button("🔄 Force Refresh"): st.cache_data.clear(); st.rerun()

    years = sorted(master_data['nom'].index.year.unique().tolist())
    year_range = st.sidebar.slider("Analysis Year Range", min_value=years[0], max_value=years[-1], value=(years[0], years[-1]))
    stocks_detail = list(SECTORS[selected_sector].keys())
    d_nom = master_data['nom'][stocks_detail][(master_data['nom'].index.year >= year_range[0]) & (master_data['nom'].index.year <= year_range[1])]
    d_adj = master_data['adj'][stocks_detail][(master_data['adj'].index.year >= year_range[0]) & (master_data['adj'].index.year <= year_range[1])]
    
    res_out = run_model(d_nom, d_adj, 100000, window, entry_z, exit_z, stop_z, abs_stop, interest_rate)
    if res_out:
        results, half_lives, win_rates = res_out; latest = results.iloc[-1]; active_pos = latest['InPosition']
        st.subheader(f"📡 {selected_sector} Terminal | {results.index[-1].strftime('%Y-%m-%d')}")
        terminal_html = f"<div class='terminal'><div class='terminal-header'>TERMINAL | DATA_SYNC: {results.index[-1].date()}</div>"
        if pd.notnull(active_pos):
            p = (latest[f'{active_pos}_Price'] - latest['EntryPrice']) / latest['EntryPrice'] * 100
            terminal_html += f"<div>[ACTIVE]: <span style='color:#ffcc00'>{active_pos}</span> | PROFIT: <span style='color:{'#00ff00' if p >= 0 else '#ff5555'}'>{p:+.2f}%</span></div>"
        else: terminal_html += "<div>[ACTIVE]: <span style='color:#888'>CASH_LIQUID</span></div>"
        terminal_html += "<br><div style='display: grid; grid-template-columns: 1fr 1fr 1.2fr 1.2fr 1fr 1fr; border-bottom: 1px solid #333;'><span>TICKER</span><span>PRICE</span><span>T_BUY</span><span>T_SELL</span><span>WIN%</span><span>REV%</span></div>"
        for s in stocks_detail:
            terminal_html += f"<div style='display: grid; grid-template-columns: 1fr 1fr 1.2fr 1.2fr 1fr 1fr; padding-top:5px;'><span>{s}</span><span>{latest[f'{s}_Price']:,.2f}</span><span style='color:#00e5ff'>{latest[f'{s}_BuyPrice']:,.2f}</span><span style='color:#ff5555'>{latest[f'{s}_SellPrice']:,.2f}</span><span>{win_rates[s]:.0f}%</span><span>{latest[f'{s}_RevProb']:.1f}%</span></div>"
        st.html(terminal_html + "</div>")
        m1, m2 = st.columns(2); m1.metric("Portfolio Value", f"{latest['TotalValue']:,.0f} TL"); m2.metric("Nominal Profit", f"{(latest['TotalValue']-100000)/1000:.1f}%")

        tabs = st.tabs(["Sector Comparison", "History", "Stats"] + stocks_detail)
        with tabs[0]:
            comp_results = []
            for s_n, s_t in SECTORS.items():
                stocks_c = [s for s in s_t.keys() if s in master_data['nom'].columns]
                if not stocks_c: continue
                sc_nom = master_data['nom'][stocks_c][(master_data['nom'].index.year >= year_range[0]) & (master_data['nom'].index.year <= year_range[1])]
                sc_adj = master_data['adj'][stocks_c][(master_data['adj'].index.year >= year_range[0]) & (master_data['adj'].index.year <= year_range[1])]
                model_res = run_model(sc_nom, sc_adj, 100000, window, entry_z, exit_z, stop_z, abs_stop, interest_rate)
                if model_res: s_r, _, _ = model_res; comp_results.append({'Sector': s_n, 'Series': s_r['TotalValue'], 'Profit %': (s_r['TotalValue'].iloc[-1]-100000)/1000})
            if comp_results:
                fig_c = go.Figure()
                for r in comp_results:
                    fig_c.add_trace(go.Scatter(x=r['Series'].index, y=r['Series'], name=f"{r['Sector']} ({r['Profit %']:.1f}%)"))
                fig_c.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified")
                st.plotly_chart(fig_c, use_container_width=True)
        with tabs[1]:
            moves = []; prev = None
            for d, r in results.iterrows():
                curr_p = r['InPosition']
                if curr_p != prev:
                    if pd.notnull(prev): moves.append({'Date': d.date(), 'Ticker': prev, 'Action': 'SELL', 'Price': r[f'{prev}_Price']})
                    if pd.notnull(curr_p): moves.append({'Date': d.date(), 'Ticker': curr_p, 'Action': 'BUY', 'Price': r[f"{curr_p}_Price"]})
                prev = curr_p
            if moves:
                m_df = pd.DataFrame(moves); m_df['Profit %'] = ""; m_df['Hold Duration'] = ""
                for i in range(len(m_df)):
                    if m_df.iloc[i]['Action'] == 'SELL':
                        for j in range(i-1, -1, -1):
                            if m_df.iloc[j]['Action'] == 'BUY' and m_df.iloc[j]['Ticker'] == m_df.iloc[i]['Ticker']:
                                bp, sp = m_df.iloc[j]['Price'], m_df.iloc[i]['Price']; m_df.at[i, 'Profit %'] = f"{(sp-bp)/bp*100:+.2f}%"; m_df.at[i, 'Hold Duration'] = f"{(m_df.iloc[i]['Date']-m_df.iloc[j]['Date']).days} days"; break
                st.dataframe(m_df.sort_values('Date', ascending=False), use_container_width=True)
        with tabs[2]: st.table(pd.DataFrame({'Win Rate (%)': win_rates, 'Half-Life (Days)': half_lives}))
        for i, name in enumerate(stocks_detail):
            with tabs[i+3]:
                c1, c2 = st.columns(2)
                with c1:
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=results.index, y=results[f'{name}_Price'], name=f"{name} Price", line=dict(color='gray', width=1), opacity=0.5))
                    pos = results['InPosition']
                    b = results.index[(pos == name) & (pos.shift(1) != name)]
                    s = results.index[(pos != name) & (pos.shift(1) == name)]
                    fig_p.add_trace(go.Scatter(x=b, y=results.loc[b, f'{name}_Price'], mode='markers', name='BUY', marker=dict(color='green', symbol='triangle-up', size=10)))
                    fig_p.add_trace(go.Scatter(x=s, y=results.loc[s, f'{name}_Price'], mode='markers', name='SELL', marker=dict(color='red', symbol='triangle-down', size=10)))
                    fig_p.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified", title=f"{name} Price Action")
                    st.plotly_chart(fig_p, use_container_width=True)
                with c2:
                    fig_z = go.Figure()
                    fig_z.add_trace(go.Scatter(x=results.index, y=results[f'{name}_Z'], name="Z-Score", line=dict(color='purple')))
                    fig_z.add_hline(y=entry_z, line_dash="dash", line_color="green", annotation_text="Entry")
                    fig_z.add_hline(y=exit_z, line_dash="dash", line_color="red", annotation_text="Exit")
                    fig_z.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified", title=f"{name} Z-Score")
                    st.plotly_chart(fig_z, use_container_width=True)
else: st.error("Data error.")
