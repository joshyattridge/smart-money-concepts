import logging
from typing import Dict, Any, Optional

class ExecutionEngine:
    def __init__(self, mode: str = "paper", risk_pct: float = 1.0, account_balance: float = 10000.0, exchange=None):
        """
        mode: "paper" or "live"
        risk_pct: Percentage of account balance to risk per trade (e.g., 1.0 for 1%)
        account_balance: Starting balance for paper trading
        exchange: The initialized ccxt exchange object for live trading
        """
        self.mode = mode
        self.risk_pct = risk_pct
        self.account_balance = account_balance
        self.exchange = exchange
        self.positions = []

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """
        Calculates the position size based on account balance and risk percentage.
        """
        balance = self.account_balance
        if self.mode == "live" and self.exchange:
            try:
                # Fetch actual live balance for the quote currency (e.g. USDT in BTC/USDT)
                quote_currency = symbol.split('/')[1]
                free_balance = self.exchange.fetch_free_balance()
                balance = float(free_balance.get(quote_currency, 0.0))
            except Exception as e:
                self.logger.warning(f"Failed to fetch live balance: {e}. Falling back to default balance.")

        risk_amount = balance * (self.risk_pct / 100)
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return 0.0

        return risk_amount / risk_per_unit

    def validate_risk_reward(self, entry: float, sl: float, tp: float, min_rr: float = 3.0) -> bool:
        """
        Validates if the trade meets the minimum Risk:Reward ratio.
        """
        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return False

        rr_ratio = reward / risk
        return rr_ratio >= min_rr

    def calculate_levels(self, action: str, current_price: float, recent_swing_high: float, recent_swing_low: float, min_rr: float) -> Dict[str, float]:
        """
        Calculates Stop Loss and Take Profit levels.
        SL is placed just beyond recent swing structure.
        TP is calculated mechanically to meet minimum RR.
        """
        if action == "BUY":
            sl = recent_swing_low * 0.999 # Slightly below the swing low
            risk = current_price - sl
            tp = current_price + (risk * min_rr)
        else: # SELL
            sl = recent_swing_high * 1.001 # Slightly above the swing high
            risk = sl - current_price
            tp = current_price - (risk * min_rr)

        return {"entry": current_price, "sl": sl, "tp": tp}

    def execute_trade(self, symbol: str, signal: Dict[str, Any], market_data: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Processes a generated signal, validates risk, calculates position size, and executes.
        """
        action = signal["action"]
        current_price = market_data["close"]

        # Determine minimum RR based on algorithm version
        min_rr = 3.0 if signal["version"] == "v4" else 1.2

        levels = self.calculate_levels(
            action,
            current_price,
            market_data["recent_swing_high"],
            market_data["recent_swing_low"],
            min_rr
        )

        # Validate Tier 5: Risk-Reward
        if not self.validate_risk_reward(levels["entry"], levels["sl"], levels["tp"], min_rr):
            self.logger.warning(f"Signal rejected: Did not meet minimum R:R of 1:{min_rr}")
            return None

        qty = self.calculate_position_size(symbol, levels["entry"], levels["sl"])

        import datetime
        trade_details = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "symbol": symbol,
            "action": action,
            "quantity": round(qty, 4),
            "entry_price": levels["entry"],
            "stop_loss": levels["sl"],
            "take_profit": levels["tp"],
            "confidence": signal["confidence"],
            "version": signal["version"],
            "reasons": "; ".join(signal.get("reasons", [])),
            "mode": self.mode
        }

        if self.mode == "paper":
            self.logger.info(f"[PAPER TRADE] Executing {action} on {symbol}: Qty {qty:.4f} @ {current_price:.2f} | SL: {levels['sl']:.2f} | TP: {levels['tp']:.2f}")
            self.positions.append(trade_details)
            self._save_trade(trade_details)
            return trade_details

        elif self.mode == "live":
            self.logger.info(f"[LIVE TRADE] Sending order to exchange for {symbol}...")
            if not self.exchange:
                self.logger.error("Exchange instance not provided for live trading. Cannot execute.")
                return None

            try:
                # Basic market order implementation
                if action == "BUY":
                    order = self.exchange.create_market_buy_order(symbol, qty)
                else:
                    order = self.exchange.create_market_sell_order(symbol, qty)

                self.logger.info(f"[LIVE TRADE SUCCESS] Order id: {order.get('id')} placed successfully.")

                # Note: In a full production environment you would also create stop loss and take profit orders here.
                # E.g. self.exchange.create_order(symbol, 'stop_loss', 'sell' if action=='BUY' else 'buy', qty, None, {'stopPrice': levels['sl']})

                trade_details['exchange_response_id'] = order.get('id')
                self.positions.append(trade_details)
                self._save_trade(trade_details)
                return trade_details
            except Exception as e:
                self.logger.error(f"[LIVE TRADE FAILED] Error executing {action} on {symbol}: {e}")
                return None

        return None

    def _save_trade(self, trade_details: Dict[str, Any]):
        """Saves executed trade to local CSV for dashboard tracking."""
        import pandas as pd
        import os

        file_path = "data/trades.csv"
        df_new = pd.DataFrame([trade_details])

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df_new.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(file_path, mode='w', header=True, index=False)
