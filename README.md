# Smart Money Concepts (smc) BETA

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

- FVG - Fair Value Gap
- Highs and Lows
- BOS and CHoCH
- OB - Order Block
- Liquidity

## Examples

Please take a look at smc.test.py for more detailed examples on how each indicator works.

```python
    smc.fvg(ohlc) # Fair Value Gap
    smc.highs_lows(ohlc) # Highs and Lows
    smc.bos_choch(ohlc, close_break=True, filter_liquidity=False) # Detect BOS and CHoCH
    smc.ob(ohlc) # Order Block
    smc.liquidity(ohlc) # Liquidity
```

## Contributing

This project is still in BETA so please feel free to contribute to the project. By creating your own indicators or improving the existing ones.

1. Fork it (https://github.com/joshyattridge/smartmoneyconcepts/fork).
2. Study how it's implemented.
3. Create your feature branch (git checkout -b my-new-feature).
4. Run black code formatter on the finta.py to ensure uniform code style.
5. Commit your changes (git commit -am 'Add some feature').
6. Push to the branch (git push origin my-new-feature).
7. Create a new Pull Request.
