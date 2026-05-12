import json
import time
import logging
from typing import Dict, Any

from src.data.fetcher import DataFetcher
from src.features.ict_features import ICTFeatures
import os
import torch
import pandas as pd
import numpy as np
import requests
from src.strategy.generator import ICTStrategy
from src.execution.engine import ExecutionEngine
from src.models.lstm_model import MultiTaskICTLSTM, ModelTrainer, ICTDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingBotDaemon:
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.fetcher = DataFetcher(
            exchange_id=self.config["exchange"]["name"],
            testnet=False,
            api_key=self.config["exchange"].get("api_key", ""),
            api_secret=self.config["exchange"].get("api_secret", "")
        )

        # In live mode, pass the initialized ccxt exchange object to the Execution Engine
        self.strategy = ICTStrategy(algorithm_version=self.config["trading"]["algorithm_version"])
        self.execution = ExecutionEngine(
            mode=self.config["trading"]["mode"],
            risk_pct=self.config["trading"]["risk_per_trade_pct"],
            exchange=self.fetcher.exchange
        )

        # Initialize Telegram Notifier (Mocked for now, integrate python-telegram-bot later)
        self.telegram_enabled = self.config["telegram"]["enabled"]

        # Load AI Model
        self._init_ai_model()

    def _init_ai_model(self):
        """Initializes and loads the trained PyTorch LSTM model."""
        self.seq_length = 60
        # Dummy dataset to get feature cols and target cols info
        self.feature_cols = ['open', 'high', 'low', 'close', 'volume', 'momentum', 'avg_volume', 'premium_discount']
        self.target_cols = ['mss', 'fvg', 'ob', 'liquidity_grab', 'breaker_block', 'ote']

        input_size = len(self.feature_cols)
        hidden_size = self.config["model"]["lstm_hidden_size"]
        num_layers = self.config["model"]["lstm_num_layers"]

        self.ai_model = MultiTaskICTLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_tasks=len(self.target_cols)
        )
        self.trainer = ModelTrainer(self.ai_model, learning_rate=self.config["model"]["learning_rate"])

        import json
        model_path = self.config["model"]["model_path"]
        scaler_path = os.path.join(os.path.dirname(model_path), "scaler.json")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.trainer.load_model(model_path)

            with open(scaler_path, "r") as f:
                scaler_data = json.load(f)
                self.scaler_mean = pd.Series(scaler_data["mean"])
                self.scaler_std = pd.Series(scaler_data["std"])

            logger.info(f"Loaded trained AI model and global scaler from {os.path.dirname(model_path)}")
        else:
            logger.warning(f"AI Model or Scaler not found at {model_path}. Please run train.py first. Proceeding with deterministic logic fallback.")
            self.ai_model = None

    def _send_telegram_alert(self, message: str):
        if self.telegram_enabled:
            bot_token = self.config["telegram"].get("bot_token")
            chat_id = self.config["telegram"].get("chat_id")
            if bot_token and chat_id and bot_token != "YOUR_BOT_TOKEN":
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }
                try:
                    requests.post(url, json=payload, timeout=5)
                    logger.info(f"[TELEGRAM ALERT] Successfully sent message.")
                except Exception as e:
                    logger.error(f"[TELEGRAM ALERT FAILED] {e}")
            else:
                logger.info(f"[TELEGRAM ALERT MOCKED] {message}")

    def run_cycle(self):
        """
        Executes a single monitoring cycle.
        """
        symbols = self.config["trading"]["symbols"]
        base_tf = self.config["trading"]["base_timeframe"]
        htf_tfs = self.config["trading"]["htf_timeframes"]

        for symbol in symbols:
            self._log_thought_process(symbol, f"Starting analysis cycle for {symbol} on {base_tf} timeframe.")

            # 1. Fetch Multi-Timeframe Data
            all_timeframes = [base_tf] + htf_tfs
            try:
                data_dict = self.fetcher.fetch_multi_timeframe_data(symbol, all_timeframes, limit=100)
                self._log_thought_process(symbol, f"Successfully fetched 100 recent candles for {all_timeframes}.")
            except Exception as e:
                self._log_thought_process(symbol, f"Failed to fetch data: {e}")
                continue

            if base_tf not in data_dict:
                continue

            # 2. Feature Engineering (ICT Concepts)
            base_df = data_dict[base_tf]
            features_engine = ICTFeatures(base_df)
            base_features_df = features_engine.generate_all_features()

            # Use AI Model for inference if available, otherwise use deterministic features
            # Use iloc[-2] to evaluate the most recently CLOSED candle. iloc[-1] is the current open/forming candle.
            current_data = base_features_df.iloc[-2].to_dict()

            use_ai = self.config["model"].get("use_ai_inference", True)

            o, h, l, c = current_data.get('open', 0), current_data.get('high', 0), current_data.get('low', 0), current_data.get('close', 0)
            volume = current_data.get('volume', 0)
            self._log_thought_process(symbol, f"Candle closed at OHLC: [{o:.4f}, {h:.4f}, {l:.4f}, {c:.4f}] | Vol: {volume:.4f}")

            # Skip processing if candle has absolutely zero volume (e.g. frozen/delisted market)
            if volume == 0.0 or (o == h and h == l and l == c):
                self._log_thought_process(symbol, "Candle has zero volume or flat OHLC. Market frozen or low liquidity. Skipping cycle.")
                continue

            # To capture raw math before AI overrides it
            raw_math_data = current_data.copy()

            if self.ai_model is not None and len(base_features_df) >= self.seq_length and use_ai:
                # Normalize inputs using the global scaler saved during training
                normalized_features = (base_features_df[self.feature_cols] - self.scaler_mean) / (self.scaler_std + 1e-8)

                # Get the last sequence (ending at the closed candle)
                x_seq = normalized_features.values[-self.seq_length - 1:-1]

                # Predict
                predictions = self.trainer.predict(x_seq)

                # Diagnostic Comparison
                math_mss = current_data.get('mss', 0)
                math_fvg = current_data.get('fvg', 0)
                math_ob = current_data.get('ob', 0)

                self._log_thought_process(symbol, f"[DIAGNOSTIC] Math logic found: MSS={math_mss}, FVG={math_fvg}, OB={math_ob}")
                self._log_thought_process(symbol, f"[DIAGNOSTIC] AI Model predicted: MSS={predictions[0]}, FVG={predictions[1]}, OB={predictions[2]}")

                # Override deterministic labels with AI predictions
                for i, col in enumerate(self.target_cols):
                    current_data[col] = predictions[i]
            else:
                predictions = []
                self._log_thought_process(symbol, f"Using standard deterministic math for ICT detection. (AI Mode: {'ON' if use_ai else 'OFF'})")

            # Log continuous market data ledger (using raw math data to preserve what math saw before AI override)
            self._save_candle_data(symbol, base_features_df.index[-2].isoformat(), raw_math_data, predictions)

            # Calculate HTF Features
            # We aggregate HTF concepts to pass to the strategy generator
            htf_data = {'liquidity_grab': 0, 'ob': 0}
            for htf in htf_tfs:
                if htf in data_dict:
                    htf_engine = ICTFeatures(data_dict[htf])
                    htf_df = htf_engine.generate_all_features()

                    # Evaluate the closed candle for HTF as well
                    latest_htf = htf_df.iloc[-2]

                    # If we find a confirmation on ANY higher timeframe, save it. Don't overwrite it with a missing confirmation on a higher-higher timeframe.
                    if latest_htf.get('liquidity_grab', 0) != 0:
                        htf_data['liquidity_grab'] = latest_htf['liquidity_grab']
                    if latest_htf.get('ob', 0) != 0:
                        htf_data['ob'] = latest_htf['ob']
                        htf_data['ob_upper'] = latest_htf.get('ob_upper', 0)
                        htf_data['ob_lower'] = latest_htf.get('ob_lower', 0)

            # 3. Strategy & Signal Generation
            signal = self.strategy.generate_signal(current_data, htf_data)

            # Log the step-by-step thinking process regardless of whether a signal was generated
            for thought in signal.get("thought_process", []):
                self._log_thought_process(symbol, thought)

            if signal.get("action") is None:
                continue

            if signal.get("action"):
                # Stale Signal Checking
                candle_timestamp_utc = pd.to_datetime(base_features_df.index[-2], utc=True)
                current_time_utc = pd.Timestamp.utcnow()

                # Parse timeframe safely into minutes
                tf_minutes = 15 # default
                if base_tf.endswith('m'): tf_minutes = int(base_tf[:-1])
                elif base_tf.endswith('h'): tf_minutes = int(base_tf[:-1]) * 60
                elif base_tf.endswith('d'): tf_minutes = int(base_tf[:-1]) * 1440
                elif base_tf.endswith('w'): tf_minutes = int(base_tf[:-1]) * 10080

                # CCXT timestamp is the OPEN time. So the candle closes at timestamp + duration.
                # The time since it closed is the total age minus its duration.
                age_since_open_minutes = (current_time_utc - candle_timestamp_utc).total_seconds() / 60.0
                age_since_close_minutes = age_since_open_minutes - tf_minutes

                # If the candle closed more than a few minutes ago (e.g., 5 min grace period), skip it
                if age_since_close_minutes > 5.0:
                    self._log_thought_process(symbol, f"⚠️ Valid Signal Generated, BUT candle is STALE! Closed {age_since_close_minutes:.1f} mins ago. Discarding trade.")
                    continue

                reasons_str = "\n".join([f"- {r}" for r in signal.get("reasons", [])])
                self._log_thought_process(symbol, f"🔥 Valid Signal Generated! Action: {signal['action']} | Confidence: {signal['confidence']:.2f}\nConfirmations:\n{reasons_str}")

                self._send_telegram_alert(f"🚨 Signal Alert: {signal['action']} {symbol} (Conf: {signal['confidence']:.2f})\n<b>Confirmations:</b>\n{reasons_str}")

                # Save signal to CSV for dashboard
                self._save_signal(symbol, signal)

                # 4. Execution & Risk Management
                # Pass recent swing levels for SL calculation
                market_data = {
                    "close": current_data["close"],
                    "recent_swing_high": current_data.get("recent_swing_high", current_data["high"] * 1.01),
                    "recent_swing_low": current_data.get("recent_swing_low", current_data["low"] * 0.99)
                }

                trade = self.execution.execute_trade(symbol, signal, market_data)
                if trade:
                    trade_msg = (
                        f"✅ Trade Executed: {trade['action']} {trade['quantity']} {symbol} @ {trade['entry_price']:.2f}\n"
                        f"SL: {trade['stop_loss']:.2f}\n"
                        f"TP: {trade['take_profit']:.2f}\n\n"
                        f"<b>Trade Setup Context:</b>\n{reasons_str}"
                    )
                    self._send_telegram_alert(trade_msg)

    def _log_thought_process(self, symbol: str, message: str):
        """Saves detailed bot logic to a log file for the Streamlit dashboard."""
        import os
        import pandas as pd
        import datetime
        file_path = "data/bot_logs.csv"

        log_data = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "symbol": symbol,
            "message": message
        }
        df_new = pd.DataFrame([log_data])
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df_new.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(file_path, mode='w', header=True, index=False)

        logger.info(message)

    def _save_candle_data(self, symbol: str, timestamp: str, current_data: dict, predictions: list):
        """Saves a continuous ledger of every single closed candle, its calculated math features, and its AI predictions."""
        import os
        import pandas as pd
        file_path = "data/market_history.csv"

        # Prepare the row data
        row_data = {
            "timestamp": timestamp,
            "symbol": symbol,
            "open": current_data.get('open', 0),
            "high": current_data.get('high', 0),
            "low": current_data.get('low', 0),
            "close": current_data.get('close', 0),
            "volume": current_data.get('volume', 0),
            "math_mss": current_data.get('mss', 0),
            "math_fvg": current_data.get('fvg', 0),
            "math_ob": current_data.get('ob', 0),
            "math_liquidity_grab": current_data.get('liquidity_grab', 0),
            "ai_mss": predictions[0] if len(predictions) > 0 else 0,
            "ai_fvg": predictions[1] if len(predictions) > 0 else 0,
            "ai_ob": predictions[2] if len(predictions) > 0 else 0,
            "ai_liquidity_grab": predictions[3] if len(predictions) > 0 else 0
        }

        df_new = pd.DataFrame([row_data])
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df_new.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(file_path, mode='w', header=True, index=False)

    def _save_signal(self, symbol: str, signal: dict):
        import datetime
        import pandas as pd
        import os

        file_path = "data/signals.csv"
        sig_data = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "symbol": symbol,
            "action": signal["action"],
            "confidence": signal["confidence"],
            "version": signal["version"],
            "reasons": "; ".join(signal.get("reasons", []))
        }
        df_new = pd.DataFrame([sig_data])
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df_new.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(file_path, mode='w', header=True, index=False)

    def start(self):
        """
        Starts the continuous daemon loop, calculating sleep interval dynamically.
        """
        logger.info("Starting ICT Trading Bot Daemon...")

        # Calculate interval dynamically based on base_timeframe config
        base_tf = self.config["trading"]["base_timeframe"]
        tf_minutes = 15
        if base_tf.endswith('m'): tf_minutes = int(base_tf[:-1])
        elif base_tf.endswith('h'): tf_minutes = int(base_tf[:-1]) * 60
        elif base_tf.endswith('d'): tf_minutes = int(base_tf[:-1]) * 1440

        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")

            # Synchronize with the next candle close
            import datetime
            import math
            now = datetime.datetime.now(datetime.timezone.utc)
            minutes_passed = now.minute + (now.hour * 60)

            # Find the next interval boundary
            next_interval_minutes = math.ceil((minutes_passed + 1e-9) / tf_minutes) * tf_minutes

            # Create a datetime object for the exact next boundary
            next_run_time = now.replace(hour=(next_interval_minutes // 60) % 24,
                                      minute=next_interval_minutes % 60,
                                      second=0, microsecond=0)

            # If the calculation rolled over to the next day
            if next_interval_minutes >= 1440 and now.hour > 0:
                next_run_time += datetime.timedelta(days=1)

            # Add a 2-second buffer to ensure the exchange API has finalised the closed candle
            next_run_time += datetime.timedelta(seconds=2)

            sleep_seconds = (next_run_time - now).total_seconds()

            # Fallback failsafe
            if sleep_seconds <= 0:
                sleep_seconds = tf_minutes * 60

            logger.info(f"Syncing to clock: Sleeping for {sleep_seconds:.1f} seconds until next {base_tf} candle closes...")
            time.sleep(sleep_seconds)

if __name__ == "__main__":
    bot = TradingBotDaemon()
    bot.start()
