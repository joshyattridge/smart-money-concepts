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

## Examples

Test out the fair value gap indicator:

```python
smc.fvg(ohlc)
```

## Contributing

1. Fork it (https://github.com/peerchemist/finta/fork)
2. Study how it's implemented.
3. Create your feature branch (git checkout -b my-new-feature).
4. Run black code formatter on the finta.py to ensure uniform code style.
5. Commit your changes (git commit -am 'Add some feature').
6. Push to the branch (git push origin my-new-feature).
7. Create a new Pull Request.
