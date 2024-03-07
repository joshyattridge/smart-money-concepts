# Smart Money Concepts (smc) BETA

The Smart Money Concepts Python Indicator is a sophisticated financial tool developed for traders and investors to gain insights into market sentiment, trends, and potential reversals. This indicator is built using Python, a versatile programming language known for its data analysis and visualization capabilities.

![alt text](https://github.com/joshyattridge/smart-money-concepts/blob/21656dd807c4077f345b6cbf29b1bc37672628e9/tests/test_binance.png)

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

returns:
FVG = 1 if bullish fair value gap, -1 if bearish fair value gap
Top = the top of the fair value gap
Bottom = the bottom of the fair value gap
MitigatedIndex = the index of the candle that mitigated the fair value gap

### Swing Highs and Lows

```python
smc.swing_highs_lows(ohlc, swing_length = 50)
```

A swing high is when the current high is the highest high out of the swing_length amount of candles before and after.
A swing low is when the current low is the lowest low out of the swing_length amount of candles before and after.

parameters:
swing_length: int - the amount of candles to look back and forward to determine the swing high or low

returns:
HighLow = 1 if swing high, -1 if swing low
Level = the level of the swing high or low

### Break of Structure (BOS) & Change of Character (CHoCH)

```python
smc.bos_choch(ohlc, swing_highs_lows, close_break = True)
```

These are both indications of market structure changing

parameters:
swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
close_break: bool - if True then the break of structure will be mitigated based on the close of the candle otherwise it will be the high/low.

returns:
BOS = 1 if bullish break of structure, -1 if bearish break of structure
CHOCH = 1 if bullish change of character, -1 if bearish change of character
Level = the level of the break of structure or change of character
BrokenIndex = the index of the candle that broke the level

### Order Blocks (OB)

```python
smc.ob(ohlc, swing_highs_lows, close_mitigation = False)
```

This method detects order blocks when there is a high amount of market orders exist on a price range.

parameters:
swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
close_mitigation: bool - if True then the order block will be mitigated based on the close of the candle otherwise it will be the high/low.

returns:
OB = 1 if bullish order block, -1 if bearish order block
Top = top of the order block
Bottom = bottom of the order block
OBVolume = volume + 2 last volumes amounts
Percentage = strength of order block (min(highVolume, lowVolume)/max(highVolume,lowVolume))

### Liquidity

```python
smc.liquidity(ohlc, swing_highs_lows, range_percent = 0.01)
```

Liquidity is when there are multiply highs within a small range of each other.
or multiply lows within a small range of each other.

parameters:
swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
range_percent: float - the percentage of the range to determine liquidity

returns:
Liquidity = 1 if bullish liquidity, -1 if bearish liquidity
Level = the level of the liquidity
End = the index of the last liquidity level
Swept = the index of the candle that swept the liquidity

## Contributing

This project is still in BETA so please feel free to contribute to the project. By creating your own indicators or improving the existing ones.

1. Fork it (https://github.com/joshyattridge/smartmoneyconcepts/fork).
2. Study how it's implemented.
3. Create your feature branch (git checkout -b my-new-feature).
4. Commit your changes (git commit -am 'Add some feature').
5. Push to the branch (git push origin my-new-feature).
6. Create a new Pull Request.
