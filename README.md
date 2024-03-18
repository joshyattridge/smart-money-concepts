# Smart Money Concepts (smc) BETA

The Smart Money Concepts Python Indicator is a sophisticated financial tool developed for traders and investors to gain insights into market sentiment, trends, and potential reversals. This indicator is inspired by Inner Circle Trader (ICT) concepts like Order blocks, Liquidity, Fair Value Gap, Swing Highs and Lows, Break of Structure, Change of Character, and more. Please Take a look and contribute to the project.

![alt text](https://github.com/joshyattridge/smart-money-concepts/blob/53dd0b0a5e598d04b0cd3d10af71a0c252ad850d/tests/test_binance.png)

## Installation

```bash
pip install smartmoneyconcepts
```

## Usage

```python
from smartmoneyconcepts import smc
```

Prepare data to use with smc:

smc expects properly formated ohlc DataFrame, with column names in lowercase: ["open", "high", "low", "close"] and ["volume"] for indicators that expect ohlcv input.

## Indicators

### Fair Value Gap (FVG)

```python
smc.fvg(ohlc)
```

A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
Or when the previous low is higher than the next high if the current candle is bearish.

returns:<br>
FVG = 1 if bullish fair value gap, -1 if bearish fair value gap<br>
Top = the top of the fair value gap<br>
Bottom = the bottom of the fair value gap<br>
MitigatedIndex = the index of the candle that mitigated the fair value gap<br>

### Swing Highs and Lows

```python
smc.swing_highs_lows(ohlc, swing_length = 50)
```

A swing high is when the current high is the highest high out of the swing_length amount of candles before and after.
A swing low is when the current low is the lowest low out of the swing_length amount of candles before and after.

parameters:<br>
swing_length: int - the amount of candles to look back and forward to determine the swing high or low<br>

returns:<br>
HighLow = 1 if swing high, -1 if swing low<br>
Level = the level of the swing high or low<br>

### Break of Structure (BOS) & Change of Character (CHoCH)

```python
smc.bos_choch(ohlc, swing_highs_lows, close_break = True)
```

These are both indications of market structure changing

parameters:<br>
swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function<br>
close_break: bool - if True then the break of structure will be mitigated based on the close of the candle otherwise it will be the high/low.<br>

returns:<br>
BOS = 1 if bullish break of structure, -1 if bearish break of structure<br>
CHOCH = 1 if bullish change of character, -1 if bearish change of character<br>
Level = the level of the break of structure or change of character<br>
BrokenIndex = the index of the candle that broke the level<br>

### Order Blocks (OB)

```python
smc.ob(ohlc, swing_highs_lows, close_mitigation = False)
```

This method detects order blocks when there is a high amount of market orders exist on a price range.

parameters:<br>
swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function<br>
close_mitigation: bool - if True then the order block will be mitigated based on the close of the candle otherwise it will be the high/low.

returns:<br>
OB = 1 if bullish order block, -1 if bearish order block<br>
Top = top of the order block<br>
Bottom = bottom of the order block<br>
OBVolume = volume + 2 last volumes amounts<br>
Percentage = strength of order block (min(highVolume, lowVolume)/max(highVolume,lowVolume))<br>

### Liquidity

```python
smc.liquidity(ohlc, swing_highs_lows, range_percent = 0.01)
```

Liquidity is when there are multiply highs within a small range of each other.
or multiply lows within a small range of each other.

parameters:<br>
swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function<br>
range_percent: float - the percentage of the range to determine liquidity<br>

returns:<br>
Liquidity = 1 if bullish liquidity, -1 if bearish liquidity<br>
Level = the level of the liquidity<br>
End = the index of the last liquidity level<br>
Swept = the index of the candle that swept the liquidity<br>

### Previous High And Low

```python
smc.previous_high_low(ohlc, time_frame = "1D")
```

This method returns the previous high and low of the given time frame.

parameters:<br>
time_frame: str - the time frame to get the previous high and low 15m, 1H, 4H, 1D, 1W, 1M<br>

returns:<br>
PreviousHigh = the previous high<br>
PreviousLow = the previous low<br>

## Contributing

This project is still in BETA so please feel free to contribute to the project. By creating your own indicators or improving the existing ones. If you are stuggling to find something to do then please check out the issues tab for requested changes.

1. Fork it (https://github.com/joshyattridge/smartmoneyconcepts/fork).
2. Study how it's implemented.
3. Create your feature branch (git checkout -b my-new-feature).
4. Commit your changes (git commit -am 'Add some feature').
5. Push to the branch (git push origin my-new-feature).
6. Create a new Pull Request.
