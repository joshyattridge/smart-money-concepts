import os
import json
import logging
import pandas as pd
import numpy as np

from src.features.ict_features import ICTFeatures
from src.strategy.generator import ICTStrategy
from src.models.lstm_model import MultiTaskICTLSTM, ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest(data_path="data/training_data.csv", config_path="config.json"):
    """
    Simulates trading over the historical dataset using the trained AI model.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}. Please run train.py first.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    df.sort_index(inplace=True)

    symbols = df['symbol'].unique() if 'symbol' in df.columns else ["BTC/USDT"]

    # Init Strategy
    strategy = ICTStrategy(algorithm_version=config["trading"]["algorithm_version"])

    # Load AI Model
    seq_length = 60
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'momentum', 'avg_volume', 'premium_discount']
    target_cols = ['mss', 'fvg', 'ob', 'liquidity_grab', 'breaker_block', 'ote']

    input_size = len(feature_cols)
    hidden_size = config["model"]["lstm_hidden_size"]
    num_layers = config["model"]["lstm_num_layers"]

    model = MultiTaskICTLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_tasks=len(target_cols)
    )
    trainer = ModelTrainer(model, learning_rate=config["model"]["learning_rate"])

    model_path = config["model"]["model_path"]
    scaler_path = os.path.join(os.path.dirname(model_path), "scaler.json")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        trainer.load_model(model_path)
        with open(scaler_path, "r") as f:
            scaler_data = json.load(f)
            scaler_mean = pd.Series(scaler_data["mean"])
            scaler_std = pd.Series(scaler_data["std"])
        logger.info(f"Loaded trained AI model and global scaler from {os.path.dirname(model_path)} for backtesting")
    else:
        logger.error(f"AI Model or scaler not found at {model_path}. Please run train.py first.")
        return

    # Backtest Metrics
    initial_balance = 10000.0
    balance = initial_balance
    risk_pct = 1.0

    wins = 0
    losses = 0
    total_trades = 0

    trade_log = []

    logger.info("Starting backtest simulation over historical data...")

    # We will step through the dataframe row by row, starting from seq_length
    for i in range(seq_length, len(df)):
        # Normalize the past 60 candles using the global training scaler
        window_df = df.iloc[i-seq_length:i]

        normalized_features = (window_df[feature_cols] - scaler_mean) / (scaler_std + 1e-8)

        x_seq = normalized_features.values

        # Predict ICT Concepts
        predictions = trainer.predict(x_seq)

        # Current data snapshot
        current_data = window_df.iloc[-1].to_dict()
        for j, col in enumerate(target_cols):
            current_data[col] = predictions[j]

        # Simplified HTF mock for backtest
        htf_data = {'liquidity_grab': predictions[3], 'ob': predictions[2]}

        # Generate Signal
        signal = strategy.generate_signal(current_data, htf_data)

        if signal and signal.get('action') is not None:
            total_trades += 1
            action = signal['action']
            entry_price = current_data['close']

            # Track HTF context
            direction = 1 if action == "BUY" else -1
            htf_aligned = (htf_data['liquidity_grab'] == direction) or (htf_data['ob'] == direction)

            # Simple TP/SL calculation for backtest
            min_rr = 3.0 if config["trading"]["algorithm_version"] == "v4" else 1.2
            if action == "BUY":
                sl = current_data.get('low', entry_price * 0.99) * 0.999
                tp = entry_price + (entry_price - sl) * min_rr
            else:
                sl = current_data.get('high', entry_price * 1.01) * 1.001
                tp = entry_price - (sl - entry_price) * min_rr

            risk_amount = balance * (risk_pct / 100)

            # Simulate Outcome: Look ahead 50 candles to see what hit first
            future_df = df.iloc[i:min(i+50, len(df))]

            outcome = "TIMED_OUT"
            pnl_val = 0.0

            for _, row in future_df.iterrows():
                if action == "BUY":
                    if row['low'] <= sl:
                        outcome = "LOSS"
                        pnl_val = -risk_amount
                        balance -= risk_amount
                        losses += 1
                        break
                    elif row['high'] >= tp:
                        outcome = "WIN"
                        pnl_val = (risk_amount * min_rr)
                        balance += pnl_val
                        wins += 1
                        break
                elif action == "SELL":
                    if row['high'] >= sl:
                        outcome = "LOSS"
                        pnl_val = -risk_amount
                        balance -= risk_amount
                        losses += 1
                        break
                    elif row['low'] <= tp:
                        outcome = "WIN"
                        pnl_val = (risk_amount * min_rr)
                        balance += pnl_val
                        wins += 1
                        break

            trade_log.append({
                "timestamp": window_df.index[-1],
                "action": action,
                "entry_price": entry_price,
                "outcome": outcome,
                "pnl": pnl_val,
                "htf_aligned": htf_aligned
            })

    # Final Report
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    pnl = balance - initial_balance
    pnl_pct = (pnl / initial_balance) * 100

    print("\n" + "="*60)
    print("🎯 ICT-AI BOT BACKTEST REPORT")
    print("="*60)
    print(f"Total Trades Taken: {total_trades}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Ending Balance:   ${balance:,.2f}")
    print(f"Total P/L:        ${pnl:,.2f} ({pnl_pct:.2f}%)")
    print("="*60)

    if total_trades > 0:
        trades_df = pd.DataFrame(trade_log)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # 1. Performance by Signal Type (Action)
        print("\n📊 PERFORMANCE BY SIGNAL TYPE:")
        action_group = trades_df.groupby('action').agg(
            Trades=('action', 'count'),
            Wins=('outcome', lambda x: (x == 'WIN').sum()),
            PnL=('pnl', 'sum')
        )
        action_group['Win Rate (%)'] = (action_group['Wins'] / action_group['Trades'] * 100).round(2)
        print(action_group.to_string())

        # 2. Performance by HTF Alignment
        print("\n🔍 PERFORMANCE BY HTF ALIGNMENT:")
        htf_group = trades_df.groupby('htf_aligned').agg(
            Trades=('htf_aligned', 'count'),
            Wins=('outcome', lambda x: (x == 'WIN').sum()),
            PnL=('pnl', 'sum')
        )
        htf_group.index = htf_group.index.map({True: 'Aligned', False: 'Not Aligned'})
        htf_group['Win Rate (%)'] = (htf_group['Wins'] / htf_group['Trades'] * 100).round(2)
        print(htf_group.to_string())

        # 3. Performance by Time of Year (Month)
        print("\n📅 PERFORMANCE BY MONTH:")
        trades_df['Month'] = trades_df['timestamp'].dt.strftime('%b')
        month_group = trades_df.groupby('Month').agg(
            Trades=('Month', 'count'),
            PnL=('pnl', 'sum')
        )
        print(month_group.to_string())

        # 4. Performance by Price Range
        print("\n📈 PERFORMANCE BY PRICE RANGE:")
        try:
            trades_df['Price Tier'] = pd.qcut(trades_df['entry_price'], q=3, labels=['Low', 'Medium', 'High'])
            price_group = trades_df.groupby('Price Tier').agg(
                Trades=('entry_price', 'count'),
                Wins=('outcome', lambda x: (x == 'WIN').sum()),
                PnL=('pnl', 'sum')
            )
            price_group['Win Rate (%)'] = (price_group['Wins'] / price_group['Trades'] * 100).round(2)
            print(price_group.to_string())
        except Exception:
            print("Not enough price variance to calculate price tiers.")

    print("="*60 + "\n")

if __name__ == "__main__":
    run_backtest()
