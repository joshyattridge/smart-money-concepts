import ccxt
import pandas as pd
import time
from typing import List, Dict, Optional

class DataFetcher:
    def __init__(self, exchange_id: str = 'binance', testnet: bool = False, api_key: str = "", api_secret: str = ""):
        """
        Initializes the data fetcher using CCXT.
        """
        exchange_class = getattr(ccxt, exchange_id)
        exchange_config = {
            'enableRateLimit': True,
        }
        if api_key and api_secret and api_key != "YOUR_API_KEY":
            exchange_config['apiKey'] = api_key
            exchange_config['secret'] = api_secret

        self.exchange = exchange_class(exchange_config)

        if testnet and 'testnet' in self.exchange.urls:
            self.exchange.set_sandbox_mode(True)

    def fetch_historical_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetches historical OHLCV data for a given symbol and timeframe.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol} on {timeframe}: {e}")
            return pd.DataFrame()

    def fetch_multi_timeframe_data(self, symbol: str, timeframes: List[str], limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Fetches data for multiple timeframes.
        """
        data = {}
        for tf in timeframes:
            df = self.fetch_historical_data(symbol, tf, limit=limit)
            if not df.empty:
                data[tf] = df
            time.sleep(self.exchange.rateLimit / 1000)  # Respect rate limits
        return data

    def fetch_recent_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetches the most recent data (for live monitoring).
        """
        return self.fetch_historical_data(symbol, timeframe, limit=limit)
