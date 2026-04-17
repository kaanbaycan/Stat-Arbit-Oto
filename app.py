import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yfinance as yf
from datetime import datetime
import scipy.stats as stats
import time

# Set page config
st.set_page_config(page_title="Multi-Sector Stat-Arb Dashboard", layout="wide")

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
    .sync-info { color: #00e5ff; font-size: 0.9em; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

SECTORS = {
    "Automotive": {"FROTO": "FROTO.IS", "DOAS": "DOAS.IS", "TOASO": "TOASO.IS"},
    "Food Retail": {"SOKM": "SOKM.IS", "BIMAS": "BIMAS.IS", "MGROS": "MGROS.IS"},
    "Aviation": {"THYAO": "THYAO.IS", "PGSUS": "PGSUS.IS", "TAVHL": "TAVHL.IS"},
    "Steel & Iron": {"EREGL": "EREGL.IS", "KRDMD": "KRDMD.IS", "ISDMR": "ISDMR.IS"}
}

# 1. Data Loading
@st.cache_data(ttl=1800)
def load_data_source(tickers_map):
    try:
        raw_data = yf.download(list(tickers_map.values()), start="2016-01-01", progress=False, auto_adjust=False)
        if raw_data.empty: return None
        df_nom = raw_data['Close'].ffill().dropna()
        df_adj = raw_data['Adj Close'].ffill().dropna()
        if df_nom.empty or df_adj.empty: return None
        df_nom.columns = [c.replace('.IS', '') for c in df_nom.columns]
        df_adj.columns = [c.replace('.IS', '') for c in df_adj.columns]
        return {"nom": df_nom, "adj": df_adj}
    except: return None

@st.cache_data(ttl=3600)
def load_benchmarks(start_date):
    try:
        bench_tickers = ["USDTRY=X", "^XU100", "GC=F"]
        data = yf.download(bench_tickers, start=start_date, progress=False, auto_adjust=False)['Close']
        data.columns = ["Gold_ONS", "USDTRY", "XU100"]
        return data.ffill()
    except: return None

# 3. Model Logic
def run_model(data_dict, initial_capital=100000, window=30, entry_z=-2.0, exit_z=0.5, stop_z=-4.0, abs_stop=0.10, interest_rate=0.35, year_range=None):
    if not data_dict or data_dict['nom'].empty: return None
    df_nom, df_adj = data_dict['nom'], data_dict['adj']
    if len(df_nom) < window: return None

    mean_adj = df_adj.mean(axis=1); ratios_adj = df_adj.divide(mean_adj, axis=0)
    rolling_mean = ratios_adj.rolling(window=window).mean(); rolling_std = ratios_adj.rolling(window=window).std()
    z_scores = (ratios_adj - rolling_mean) / rolling_std
    
    half_lives = {}
    for col in df_adj.columns:
        y = ratios_adj[col].dropna(); y_lag = y.shift(1).dropna(); y_curr = y.iloc[1:]; y_diff = y_curr.values - y_lag.values
        res = stats.linregress(y_lag.values, y_diff); beta = res.slope
        half_lives[col] = -np.log(2)/beta if beta < 0 else 99.9

    if year_range:
        s_y, e_y = year_range; sim_df = df_nom[(df_nom.index.year >= s_y) & (df_nom.index.year <= e_y)]
        if sim_df.empty: return None
        start_date, end_date = sim_df.index[0], sim_df.index[-1]
    else: start_date, end_date = df_nom.index[0], df_nom.index[-1]

    cash = initial_capital; positions = {col: 0 for col in df_nom.columns}; current_stock = None; entry_p_nom = 0; prev_date = None; history = []; outcomes = {col: [] for col in df_nom.columns}
    
    for date in df_nom.index:
        if date < start_date: continue
        if date > end_date: break
        row_nom, row_adj = df_nom.loc[date], df_adj.loc[date]
        if prev_date is not None and cash > 0:
            days = (date - prev_date).days
            if days > 0: cash *= (1 + interest_rate / 365) ** days
        prev_date = date; daily_z = z_scores.loc[date]; reversion_probs = daily_z.apply(lambda z: (1 - stats.norm.cdf(z)) * 100)
        
        buy_nom, sell_nom = {}, {}
        for col in df_nom.columns:
            n = len(df_nom.columns); conv = row_nom[col] / row_adj[col] if row_adj[col] != 0 else 1
            tr_buy = entry_z * rolling_std.loc[date, col] + rolling_mean.loc[date, col]
            buy_p_adj = (tr_buy * (row_adj.sum() - row_adj[col])) / (n - tr_buy) if n - tr_buy > 0 else np.nan
            buy_nom[col] = buy_p_adj * conv
            tr_sell = exit_z * rolling_std.loc[date, col] + rolling_mean.loc[date, col]
            sell_p_adj = (tr_sell * (row_adj.sum() - row_adj[col])) / (n - tr_sell) if n - tr_sell > 0 else np.nan
            sell_nom[col] = sell_p_adj * conv

        if current_stock is not None:
            p_loss = (row_nom[current_stock] - entry_p_nom) / entry_p_nom
            if daily_z[current_stock] >= exit_z or daily_z[current_stock] <= stop_z or p_loss <= -abs_stop:
                outcomes[current_stock].append(1 if daily_z[current_stock] >= exit_z else 0)
                cash = positions[current_stock] * row_nom[current_stock]; positions[current_stock] = 0; current_stock = None; entry_p_nom = 0
        if current_stock is None:
            min_z = daily_z.min()
            if min_z <= entry_z:
                best_stock = daily_z.idxmin(); positions[best_stock] = cash / row_nom[best_stock]; cash = 0; current_stock = best_stock; entry_p_nom = row_nom[best_stock]
        
        total_val = cash + sum(positions[s] * row_nom[s] for s in df_nom.columns)
        state = {'Date': date, 'TotalValue': total_val, 'Cash': cash, 'InPosition': current_stock, 'EntryPrice': entry_p_nom}
        for s in df_nom.columns:
            state[f'{s}_Price'] = row_nom[s]; state[f'{s}_Z'] = daily_z[s]; state[f'{s}_RevProb'] = reversion_probs[s]
            state[f'{s}_BuyPrice'] = buy_nom[s]; state[f'{s}_SellPrice'] = sell_nom[s]
        history.append(state)
        
    result_df = pd.DataFrame(history).set_index('Date')
    win_rates = {col: (sum(outcomes[col])/len(outcomes[col])*100 if outcomes[col] else 0) for col in df_nom.columns}
    return result_df, half_lives, win_rates

# --- App UI ---
st.title("Sector Stat-Arb Dashboard")

st.sidebar.header("Global Settings")
window = st.sidebar.slider("Rolling Window", 10, 100, 30); entry_z = st.sidebar.slider("Entry Z (Buy)", -5.0, -1.0, -2.0)
exit_z = st.sidebar.slider("Exit Z (Sell)", -1.0, 2.0, 0.5); stop_z = st.sidebar.slider("Stop Z (Relative)", -10.0, -3.0, -4.0)
abs_stop = st.sidebar.slider("Absolute Stop Loss %", 0.01, 0.30, 0.10); interest_rate = st.sidebar.slider("Idle Cash Yearly Interest", 0.0, 1.0, 0.35)

st.subheader("🎯 Market Opportunity Radar (Nominal Prices)")
radar_data = []
for s_name, s_tickers in SECTORS.items():
    s_data = load_data_source(s_tickers)
    if s_data:
        m_res = run_model(s_data, 100000, window, entry_z, exit_z, stop_z, abs_stop, interest_rate)
        if m_res:
            s_res, _, _ = m_res; latest_r = s_res.iloc[-1]; active_r = latest_r['InPosition']
            if active_r and not pd.isna(active_r):
                radar_data.append({'Sector': s_name, 'Status': '🔴 HOLDING', 'Stock': active_r, 'Price': f"{latest_r[f'{active_r}_Price']:,.2f}", 'Rev Prob': f"{latest_r[f'{active_r}_RevProb']:.1f}%", 'Target': f"{latest_r[f'{active_r}_SellPrice']:,.2f}"})
            else:
                z_cols = [c for c in latest_r.index if c.endswith('_Z')]; min_z_s = latest_r[z_cols].idxmin().replace('_Z', '')
                radar_data.append({'Sector': s_name, 'Status': '🟢 MONITORING', 'Stock': min_z_s, 'Price': f"{latest_r[f'{min_z_s}_Price']:,.2f}", 'Rev Prob': f"{latest_r[f'{min_z_s}_RevProb']:.1f}%", 'Target': f"{latest_r[f'{min_z_s}_BuyPrice']:,.2f} (BUY)"})
    time.sleep(0.3)
if radar_data: st.table(pd.DataFrame(radar_data).set_index('Sector'))

st.markdown("---")
selected_sector = st.sidebar.selectbox("Select Detail Sector", list(SECTORS.keys()))
if st.sidebar.button("🔄 Refresh All Data"): st.cache_data.clear()

detail_data = load_data_source(SECTORS[selected_sector])
if detail_data:
    years = sorted(detail_data['nom'].index.year.unique().tolist())
    year_range = st.sidebar.slider("Analysis Year Range", min_value=years[0], max_value=years[-1], value=(years[0], years[-1]))
    res_out = run_model(detail_data, 100000, window, entry_z, exit_z, stop_z, abs_stop, interest_rate, year_range=year_range)
    if res_out:
        results, half_lives, win_rates = res_out; bench_data = load_benchmarks(results.index[0].strftime('%Y-%m-%d')); latest = results.iloc[-1]; active_pos = latest['InPosition']
        st.subheader(f"📡 {selected_sector} Detail Terminal")
        terminal_html = f"<div class='terminal'><div class='terminal-header'>DETAIL TERMINAL | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>"
        if active_pos and not pd.isna(active_pos):
            p = (latest[f'{active_pos}_Price'] - latest['EntryPrice']) / latest['EntryPrice'] * 100
            terminal_html += f"<div>[ACTIVE]: <span style='color:#ffcc00'>{active_pos}</span> | PROFIT: <span style='color:{'#00ff00' if p >= 0 else '#ff5555'}'>{p:+.2f}%</span></div>"
        else: terminal_html += "<div>[ACTIVE]: <span style='color:#888'>CASH_LIQUID</span></div>"
        terminal_html += "<br><div style='display: grid; grid-template-columns: 1fr 1fr 1.2fr 1.2fr 1fr 1fr; border-bottom: 1px solid #333;'><span>TICKER</span><span>PRICE</span><span>T_BUY</span><span>T_SELL</span><span>WIN%</span><span>REV%</span></div>"
        for s in detail_data['nom'].columns:
            terminal_html += f"<div style='display: grid; grid-template-columns: 1fr 1fr 1.2fr 1.2fr 1fr 1fr; padding-top:5px;'><span>{s}</span><span>{latest[f'{s}_Price']:,.2f}</span><span style='color:#00e5ff'>{latest[f'{s}_BuyPrice']:,.2f}</span><span style='color:#ff5555'>{latest[f'{s}_SellPrice']:,.2f}</span><span>{win_rates[s]:.0f}%</span><span>{latest[f'{s}_RevProb']:.1f}%</span></div>"
        st.html(terminal_html + "</div>")
        m1, m2, m3, m4 = st.columns(4); m1.metric("Portfolio Value", f"{latest['TotalValue']:,.0f} TL"); m2.metric("Nominal Profit", f"{(latest['TotalValue']-100000)/1000:.1f}%")
        if bench_data is not None:
            bench_aligned = bench_data.reindex(results.index).ffill(); s_u = float(bench_aligned['USDTRY'].iloc[0]); c_u = float(bench_aligned['USDTRY'].iloc[-1])
            real_p = ((latest['TotalValue'] / c_u) / (100000 / s_u) - 1) * 100
            m3.metric("vs USDTRY", f"{real_p:,.1f}%"); m4.metric("USDTRY", f"{c_u:,.2f} ₺")

        tabs = st.tabs(["Sector Comparison", "Benchmark Growth", "History", "Stats"] + list(detail_data['nom'].columns))
        with tabs[0]:
            comp_results = []
            for s_n, s_t in SECTORS.items():
                s_d = load_data_source(s_t)
                model_res = run_model(s_d, 100000, window, entry_z, exit_z, stop_z, abs_stop, interest_rate, year_range=year_range)
                if model_res: s_r, _, _ = model_res; comp_results.append({'Sector': s_n, 'Series': s_r['TotalValue'], 'Profit %': (s_r['TotalValue'].iloc[-1]-100000)/1000})
            if comp_results:
                fig_c, ax_c = plt.subplots(figsize=(10, 4))
                for r in comp_results: ax_c.plot(r['Series'].index, r['Series'], label=f"{r['Sector']} ({r['Profit %']:.1f}%)")
                ax_c.legend(); st.pyplot(fig_c)
        with tabs[1]:
            if bench_data is not None:
                gold_p = bench_aligned['Gold_ONS'] * bench_aligned['USDTRY']; usd_g = (results['TotalValue'] / bench_aligned['USDTRY']) / (100000 / s_u) * 100; gold_g = (results['TotalValue'] / gold_p) / (100000 / gold_p.iloc[0]) * 100
                fig_r, ax_r = plt.subplots(figsize=(10, 4)); ax_r.plot(results.index, usd_g, label="vs USD"); ax_r.plot(results.index, gold_g, label="vs Gold", color='gold'); ax_r.axhline(100, color='black'); ax_r.legend(); st.pyplot(fig_r)
        with tabs[2]:
            moves = []; prev = None
            for d, r in results.iterrows():
                curr_p = r['InPosition']
                if curr_p != prev:
                    if prev and not pd.isna(prev): moves.append({'Date': d.date(), 'Ticker': prev, 'Action': 'SELL', 'Price': r[f'{prev}_Price']})
                    if curr_p and not pd.isna(curr_p): moves.append({'Date': d.date(), 'Ticker': curr_p, 'Action': 'BUY', 'Price': r[f"{curr_p}_Price"]})
                prev = curr_p
            if moves:
                m_df = pd.DataFrame(moves); m_df['Profit %'] = ""; m_df['Hold Duration'] = ""
                for i in range(len(m_df)):
                    if m_df.iloc[i]['Action'] == 'SELL':
                        for j in range(i-1, -1, -1):
                            if m_df.iloc[j]['Action'] == 'BUY' and m_df.iloc[j]['Ticker'] == m_df.iloc[i]['Ticker']:
                                bp, sp = m_df.iloc[j]['Price'], m_df.iloc[i]['Price']; m_df.at[i, 'Profit %'] = f"{(sp-bp)/bp*100:+.2f}%"
                                m_df.at[i, 'Hold Duration'] = f"{(m_df.iloc[i]['Date']-m_df.iloc[j]['Date']).days} days"; break
                st.dataframe(m_df.sort_values('Date', ascending=False), use_container_width=True)
        with tabs[3]: st.table(pd.DataFrame({'Win Rate (%)': win_rates, 'Half-Life (Days)': half_lives}))
        for i, name in enumerate(detail_data['nom'].columns):
            with tabs[i+4]:
                c1, c2 = st.columns(2)
                with c1:
                    fig_p, ax_p = plt.subplots(); ax_p.plot(results.index, results[f'{name}_Price'], color='gray', alpha=0.5)
                    pos = results['InPosition']; b = results.index[(pos == name) & (pos.shift(1) != name)]; s = results.index[(pos != name) & (pos.shift(1) == name)]
                    ax_p.scatter(b, results.loc[b, f'{name}_Price'], color='green', marker='^'); ax_p.scatter(s, results.loc[s, f'{name}_Price'], color='red', marker='v'); st.pyplot(fig_p)
                with c2:
                    fig_z, ax_z = plt.subplots(); ax_z.plot(results.index, results[f'{name}_Z'], color='purple'); ax_z.axhline(entry_z, color='green', linestyle='--'); ax_z.axhline(exit_z, color='red', linestyle='--'); st.pyplot(fig_z)
else: st.error("Data error.")
