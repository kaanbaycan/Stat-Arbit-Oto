# 🚀 Automotive & Multi-Sector Stat-Arb Dashboard

A high-performance, real-time Statistical Arbitrage (Stat-Arb) dashboard built with **Streamlit** and **yfinance**. This tool monitors price deviations within industry sectors (Automotive, Food Retail, Aviation, Steel) and identifies mean-reversion opportunities based on mathematical Z-scores.

## 🌟 Key Features

*   **Market Opportunity Radar:** Real-time summary of all sectors. Automatically identifies the stock closest to a "Buy" signal.
*   **Multi-Sector Support:** Switch between Automotive, Food Retail, Aviation, and Steel & Iron sectors with one click.
*   **Ultra-Fast Performance:** Uses a hybrid local CSV database (`db_nominal.csv`) with incremental live syncing from yfinance to ensure near-instant loading.
*   **Advanced Risk Management:**
    *   **Relative Stop Loss:** Based on sector underperformance (Z-score).
    *   **Absolute Stop Loss:** Protective exit to cash if a stock drops by a fixed percentage (e.g., -10%).
*   **Professional Statistics:**
    *   **REV% (Reversion Probability):** Theoretical probability of profit based on Normal Distribution.
    *   **WIN% (Historical Win Rate):** Success track record of the strategy for each specific ticker.
    *   **Half-Life:** Estimated days to reach the profit target using an Ornstein-Uhlenbeck process.
*   **Nominal Pricing:** All terminal displays and trade history match the **Raw Market Price** seen on your broker screen.

## 🛠️ Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Stat-Arbit-Oto.git
    cd Stat-Arbit-Oto
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Initialize the Database:**
    (Run this once to download historical data from 2016 to today)
    ```bash
    python database_builder.py
    ```

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## 📈 How the Strategy Works

This dashboard uses a **Price Ratio** strategy:
1.  **Ratio Calculation:** The price of each stock is divided by the average price of the sector.
2.  **Z-Score:** We calculate how many standard deviations the current ratio is from its rolling mean.
3.  **The Signal:**
    *   **Entry (Z <= -2.0):** The stock is significantly "cheaper" than its peers. Probability of recovery is high.
    *   **Exit (Z >= 0.5):** The stock has returned to its mean relative value.
    *   **Relative Stop (Z <= -4.0):** The stock has decoupled and is no longer following the sector pattern.

## ☁️ Deployment

This project is ready for one-click deployment on **Streamlit Cloud**:
1.  Push the code (including the `.csv` database files) to GitHub.
2.  Connect your GitHub repo to [share.streamlit.io](https://share.streamlit.io).
3.  The app will automatically sync new daily closes from yfinance every time it is opened.

## ⚠️ Disclaimer
*This tool is for educational and research purposes only. Trading involves risk, and past performance is not indicative of future results.*
