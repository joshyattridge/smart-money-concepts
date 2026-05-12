import streamlit as st
import pandas as pd
import os
import json

st.set_page_config(page_title="ICT-AI Trading Dashboard", layout="wide", page_icon="📈")

def load_config():
    with open("config.json", 'r') as f:
        return json.load(f)

import time

def main():
    st.title("📈 ICT-AI Trading Bot Dashboard")
    st.markdown("Monitor real-time paper and live trading state.")

    # Auto-refresh setup
    st.sidebar.header("Dashboard Controls")
    auto_refresh = st.sidebar.checkbox("Auto-Refresh (10s)", value=True)

    config = load_config()
    st.sidebar.header("Bot Configuration")
    st.sidebar.write(f"**Mode:** {config['trading']['mode'].upper()}")
    st.sidebar.write(f"**Algorithm:** {config['trading']['algorithm_version'].upper()}")
    st.sidebar.write(f"**Risk:** {config['trading']['risk_per_trade_pct']}% per trade")
    st.sidebar.write(f"**Symbols:** {', '.join(config['trading']['symbols'])}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚨 Latest Signals Generated")
        if os.path.exists("data/signals.csv") and os.path.getsize("data/signals.csv") > 0:
            try:
                signals_df = pd.read_csv("data/signals.csv")
                # Show latest 10 signals
                st.dataframe(signals_df.tail(10).iloc[::-1], use_container_width=True)
            except pd.errors.EmptyDataError:
                st.info("No signals generated yet. Ensure main.py is running.")
        else:
            st.info("No signals generated yet. Ensure main.py is running.")

    with col2:
        st.subheader("✅ Executed Trades")
        if os.path.exists("data/trades.csv") and os.path.getsize("data/trades.csv") > 0:
            try:
                trades_df = pd.read_csv("data/trades.csv")
                # Calculate simple stats
                total_trades = len(trades_df)
                buy_trades = len(trades_df[trades_df['action'] == 'BUY'])
                sell_trades = len(trades_df[trades_df['action'] == 'SELL'])

                st.write(f"**Total Executions:** {total_trades} (Buys: {buy_trades} | Sells: {sell_trades})")

                # Show latest 10 trades
                display_cols = ['timestamp', 'symbol', 'action', 'quantity', 'entry_price', 'mode']
                st.dataframe(trades_df[display_cols].tail(10).iloc[::-1], use_container_width=True)

                st.subheader("Trade Context / Reasons")
                for index, row in trades_df.tail(3).iloc[::-1].iterrows():
                    with st.expander(f"{row['timestamp']} - {row['action']} {row['symbol']}"):
                        st.write(row['reasons'].replace("; ", "\n- "))
            except pd.errors.EmptyDataError:
                st.info("No trades executed yet.")
        else:
            st.info("No trades executed yet.")

    st.divider()
    st.subheader("🧠 Bot 'Live Brain' Output Log")
    if os.path.exists("data/bot_logs.csv") and os.path.getsize("data/bot_logs.csv") > 0:
        try:
            logs_df = pd.read_csv("data/bot_logs.csv")
            # Show the last 25 thought processes in a scrolling text area
            recent_logs = logs_df.tail(25).iloc[::-1]
            log_text = ""
            for idx, row in recent_logs.iterrows():
                log_text += f"[{row['timestamp']}] {row['symbol']}: {row['message']}\n"
            st.text_area("Latest Operations", log_text, height=300)
        except pd.errors.EmptyDataError:
            st.info("No bot logs generated yet. Ensure main.py is running.")
    else:
        st.info("No bot logs generated yet. Ensure main.py is running.")

    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
