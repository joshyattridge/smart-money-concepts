# Smart Money Concepts (smc)

The Smart Money Concepts Python Indicator is a sophisticated financial tool developed for traders and investors to gain insights into market sentiment, trends, and potential reversals. This indicator is built using Python, a versatile programming language known for its data analysis and visualization capabilities.

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

FVG - Fair Value Gap
Highs and Lows
BOS/CHoCH - Break of Structure / Change of Character
OB - Order Block

## Examples

```python
    smc.fvg(ohlc)
    smc.highs_lows(ohlc)
    smc.bos_choch(ohlc)
    smc.ob(ohlc)
```
