# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`smartmoneyconcepts` (PyPI package name, imported as `from smartmoneyconcepts import smc`) is a Python technical-analysis library implementing ICT (Inner Circle Trader) style "smart money concepts" indicators: Fair Value Gap, Swing Highs/Lows, Break of Structure/Change of Character, Order Blocks, Liquidity, Previous High/Low, Sessions, and Retracements.

The entire indicator implementation lives in one file: `smartmoneyconcepts/smc.py`, as a single `smc` class with one `@classmethod` per indicator. There is no other application code — no CLI, server, or UI in this package.

## Commands

Install in editable mode from repo root:
```bash
pip install -e .
```

Run the test suite (uses stdlib `unittest`, not pytest, despite the `.pytest_cache` directory):
```bash
cd tests
python unit_tests.py
```

Run a single test:
```bash
cd tests
python -m unittest unit_tests.TestSmartMoneyConcepts.test_fvg
```

CI (`.github/workflows/unittest.yaml`) runs on every PR: installs the package via `pip install .` on Python 3.8, then runs `python unit_tests.py` from `./tests`.

## Architecture

### The `inputvalidator` / `apply` decorator pattern

Every method on `smc` is wrapped by `@apply(inputvalidator(input_="ohlc"))` at the class level (`smartmoneyconcepts/smc.py:51`). This automatically:
- Lowercases all DataFrame column names on the first positional arg (or second, if the first is `self`/`cls`-like and not a DataFrame).
- Validates that required OHLC(V) columns exist, raising `LookupError` if not.
- Remaps a custom `column` kwarg (e.g., for using something other than "close").

Because of this, every `smc.*` method transparently accepts DataFrames with mixed-case column names — don't add manual lowercasing/validation in new methods; it's handled centrally.

### Data flow between indicators

Several indicators are NOT independent — they take the output of `smc.swing_highs_lows()` as an input DataFrame:
- `smc.bos_choch(ohlc, swing_highs_lows)`
- `smc.ob(ohlc, swing_highs_lows)`
- `smc.liquidity(ohlc, swing_highs_lows)`
- `smc.retracements(ohlc, swing_highs_lows)`

So `swing_highs_lows` must be computed first and threaded through. This is also how the test suite is structured — see `tests/unit_tests.py`.

### Output convention

Every method returns a `pd.concat([...], axis=1)` of named `pd.Series`, aligned to the input OHLC index (same row count as input, using `NaN`/`0` for non-signal rows). Preserve this shape and NaN-padding convention in any new/modified indicator — the test suite does exact frame comparisons (`pd.testing.assert_frame_equal(..., check_dtype=False)`) against fixed CSV fixtures in `tests/test_data/EURUSD/`.

### Performance-sensitive code

Recent history on this branch (`55d9czt4sg-ui-performance-improvements`) has focused on vectorizing/optimizing hot loops in `smc.py` (order blocks, swing highs/lows) — e.g. replacing per-row `.iloc` access with precomputed numpy arrays, binary search (`np.searchsorted`) instead of linear scans, and O(1) set-based removal instead of O(n) list scans. When touching these functions, preserve output parity with the CSV fixtures in `tests/test_data/` (run the full test suite) while optimizing — this codebase treats "faster but changes output" as a regression, not an improvement.

### Test fixtures

`tests/test_data/EURUSD/EURUSD_15M.csv` is the single shared input dataset. Each indicator's expected output is a corresponding `*_result_data.csv` (e.g. `fvg_result_data.csv`, `ob_result_data.csv`). If you intentionally change an indicator's output, the fixture CSVs need to be regenerated/updated alongside — there's no fixture-generation script beyond what's in `tests/unit_tests.py` and `tests/generate_gif.py`.

## Contribution norms (from CONTRIBUTING.md)

Keep PRs minimal and focused on a single function or small feature — sweeping changes across multiple indicators in one PR are discouraged by project convention.
