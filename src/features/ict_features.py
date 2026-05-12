import pandas as pd
import numpy as np

class ICTFeatures:
    """
    Computes the 12 ICT concepts from OHLCV data.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _swing_highs_lows(self, window: int = 20):
        """Helper to find local swing highs and lows without future leak."""
        # Using a backward looking window instead of center=True to prevent future leak in live trading
        self.df['swing_high'] = self.df['high'] == self.df['high'].rolling(window=window).max()
        self.df['swing_low'] = self.df['low'] == self.df['low'].rolling(window=window).min()

        # Forward fill the recent swing prices for reference
        # We shift by 1 before ffill so the current candle compares itself to the PREVIOUS swing high,
        # not a swing high it might be forming right now.
        self.df['recent_swing_high'] = self.df['high'].where(self.df['swing_high']).shift(1).ffill()
        self.df['recent_swing_low'] = self.df['low'].where(self.df['swing_low']).shift(1).ffill()

    def detect_mss(self, window: int = 20, momentum_threshold: float = 0.003, volume_multiplier: float = 1.5):
        """
        1. Market Structure Shift (MSS)
        Bullish: Closes ABOVE recent 20-period high, >0.3% upward momentum, volume 50% above average
        Bearish: Closes BELOW recent 20-period low, >0.3% downward momentum, volume 50% above average
        """
        self._swing_highs_lows(window)

        self.df['avg_volume'] = self.df['volume'].rolling(window=window).mean()
        self.df['momentum'] = (self.df['close'] - self.df['open']) / self.df['open']

        # We strictly check that the current close is > the established recent swing high.
        # recent_swing_high is already shifted backwards by 1 in the _swing_highs_lows function.
        bullish_mss_cond = (
            (self.df['close'] > self.df['recent_swing_high']) &
            (self.df['momentum'] > momentum_threshold) &
            (self.df['volume'] > self.df['avg_volume'] * volume_multiplier)
        )

        bearish_mss_cond = (
            (self.df['close'] < self.df['recent_swing_low']) &
            (self.df['momentum'] < -momentum_threshold) &
            (self.df['volume'] > self.df['avg_volume'] * volume_multiplier)
        )

        self.df['bullish_mss'] = bullish_mss_cond.astype(int)
        self.df['bearish_mss'] = bearish_mss_cond.astype(int)
        self.df['mss'] = np.where(bullish_mss_cond, 1, np.where(bearish_mss_cond, -1, 0))

    def detect_fvg(self):
        """
        2. Fair Value Gaps (FVG)
        Bullish: Candle[i-1].high < Candle[i+1].low
        Bearish: Candle[i-1].low > Candle[i+1].high
        """
        # Bullish FVG: Current candle's low is higher than the high of 2 candles ago
        bullish_fvg = self.df['low'] > self.df['high'].shift(2)
        # Bearish FVG: Current candle's high is lower than the low of 2 candles ago
        bearish_fvg = self.df['high'] < self.df['low'].shift(2)

        self.df['bullish_fvg'] = bullish_fvg.astype(int)
        self.df['bearish_fvg'] = bearish_fvg.astype(int)
        self.df['fvg'] = np.where(bullish_fvg, 1, np.where(bearish_fvg, -1, 0))

        # Calculate FVG Exact Boundaries
        # For Bullish FVG, the gap is between candle i-1 High and candle i+1 Low
        self.df['fvg_upper'] = np.where(bullish_fvg, self.df['low'], np.where(bearish_fvg, self.df['low'].shift(2), np.nan))
        self.df['fvg_lower'] = np.where(bullish_fvg, self.df['high'].shift(2), np.where(bearish_fvg, self.df['high'], np.nan))

    def detect_order_blocks(self, move_threshold: float = 0.004):
        """
        3. Order Blocks (OB)
        Bullish OB: Strong bullish candle (>0.4% move) preceded by consolidation (down candle)
        Bearish OB: Strong bearish candle (>-0.4% move) preceded by consolidation (up candle)
        """
        # Simplified OB detection:
        # A strong move that breaks structure, the opposite colored candle before the move is the OB.
        self.df['candle_pct_change'] = (self.df['close'] - self.df['open']) / self.df['open']

        strong_bullish = self.df['candle_pct_change'] > move_threshold
        strong_bearish = self.df['candle_pct_change'] < -move_threshold

        # Bullish OB: strong bullish candle preceded by a bearish candle
        bullish_ob = strong_bullish & (self.df['candle_pct_change'].shift(1) < 0)
        # Bearish OB: strong bearish candle preceded by a bullish candle
        bearish_ob = strong_bearish & (self.df['candle_pct_change'].shift(1) > 0)

        self.df['bullish_ob'] = bullish_ob.astype(int)
        self.df['bearish_ob'] = bearish_ob.astype(int)
        self.df['ob'] = np.where(bullish_ob, 1, np.where(bearish_ob, -1, 0))

        # Calculate OB Exact Boundaries (The consolidation candle before the break)
        self.df['ob_upper'] = np.where(bullish_ob | bearish_ob, self.df['high'].shift(1), np.nan)
        self.df['ob_lower'] = np.where(bullish_ob | bearish_ob, self.df['low'].shift(1), np.nan)

    def detect_liquidity_grabs(self, window: int = 10, spike_threshold: float = 0.001):
        """
        4. Liquidity Grab
        Above High: Price spikes ABOVE 10-period high by >0.1%, then quick reversal with high volume
        Below Low: Price spikes BELOW 10-period low by >0.1%, then quick reversal with high volume
        """
        recent_high_10 = self.df['high'].rolling(window=window).max().shift(1)
        recent_low_10 = self.df['low'].rolling(window=window).min().shift(1)

        # Spike above high but close below it (reversal)
        grab_above = (self.df['high'] > recent_high_10 * (1 + spike_threshold)) & (self.df['close'] < recent_high_10)

        # Spike below low but close above it (reversal)
        grab_below = (self.df['low'] < recent_low_10 * (1 - spike_threshold)) & (self.df['close'] > recent_low_10)

        self.df['liquidity_grab_up'] = grab_above.astype(int)
        self.df['liquidity_grab_down'] = grab_below.astype(int)
        self.df['liquidity_grab'] = np.where(grab_above, -1, np.where(grab_below, 1, 0))

    def calculate_premium_discount(self, window: int = 20):
        """
        5. Premium/Discount Zones
        Premium Zone (-1): Top 33%
        Equilibrium (0): Middle 33%
        Discount Zone (1): Bottom 33%
        """
        rolling_high = self.df['high'].rolling(window=window).max()
        rolling_low = self.df['low'].rolling(window=window).min()
        range_size = rolling_high - rolling_low

        # Normalize current close within the range
        normalized_pos = (self.df['close'] - rolling_low) / range_size

        # 1 = Discount (buy territory), 0 = Eq, -1 = Premium (sell territory)
        conditions = [
            normalized_pos <= 0.33,
            (normalized_pos > 0.33) & (normalized_pos < 0.67),
            normalized_pos >= 0.67
        ]
        choices = [1, 0, -1]
        self.df['premium_discount'] = np.select(conditions, choices, default=np.nan)

    def detect_breaker_blocks(self):
        """
        6. Breaker Blocks
        Failed Order Blocks that act as support/resistance.
        """
        # A simple approximation: An old OB that is broken through strongly in the opposite direction
        # We will use a proxy: if a strong move happens against a recent swing, mark it.
        # This requires historical tracking, so we approximate with MSS logic combined with recent sweeps.

        # If we had a liquidity grab (swept low) followed by a bullish MSS, the old lower high is a bullish breaker.
        # We'll create a simplified placeholder for the feature matrix.
        self.df['bullish_breaker'] = ((self.df['liquidity_grab'] == 1).shift(1) & (self.df['mss'] == 1)).astype(int)
        self.df['bearish_breaker'] = ((self.df['liquidity_grab'] == -1).shift(1) & (self.df['mss'] == -1)).astype(int)
        self.df['breaker_block'] = np.where(self.df['bullish_breaker'], 1, np.where(self.df['bearish_breaker'], -1, 0))

    def detect_ote(self):
        """
        7. OTE Zones (Optimal Trade Entry)
        Typically 62% to 79% retracement of a strong impulse leg.
        """
        # We look for a recent swing low to swing high impulse, and if current price is within 62-79% retracement
        # This is complex to do purely vectorised without a complex zigzag indicator, but we can approximate:

        # Need swing high/lows first (already calculated)
        swing_range = self.df['recent_swing_high'] - self.df['recent_swing_low']

        # Retracement level from the high (for a long setup)
        retracement = (self.df['recent_swing_high'] - self.df['close']) / swing_range

        # Bullish OTE: Trend is up (recent low < previous low, recent high > previous high), and price retraces 0.62-0.79
        bullish_ote = (retracement >= 0.62) & (retracement <= 0.79) & (self.df['close'] > self.df['recent_swing_low'])

        # Bearish OTE
        retracement_down = (self.df['close'] - self.df['recent_swing_low']) / swing_range
        bearish_ote = (retracement_down >= 0.62) & (retracement_down <= 0.79) & (self.df['close'] < self.df['recent_swing_high'])

        self.df['ote_bullish'] = bullish_ote.astype(int)
        self.df['ote_bearish'] = bearish_ote.astype(int)
        self.df['ote'] = np.where(bullish_ote, 1, np.where(bearish_ote, -1, 0))

    def generate_all_features(self):
        """Calculates all features and returns the DataFrame."""
        self.detect_mss()
        self.detect_fvg()
        self.detect_order_blocks()
        self.detect_liquidity_grabs()
        self.calculate_premium_discount()
        self.detect_breaker_blocks()
        self.detect_ote()

        # Clean up NaNs
        self.df.fillna(0, inplace=True)
        return self.df
