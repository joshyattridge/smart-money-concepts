from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

def inputvalidator(input_="ohlc"):
    def dfcheck(func):
        @wraps(func)
        def wrap(*args, **kwargs):

            args = list(args)
            i = 0 if isinstance(args[0], pd.DataFrame) else 1

            args[i] = args[i].rename(columns={c: c.lower() for c in args[i].columns})

            inputs = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": kwargs.get("column", "close").lower(),
                "v": "volume",
            }

            if inputs["c"] != "close":
                kwargs["column"] = inputs["c"]

            for l in input_:
                if inputs[l] not in args[i].columns:
                    raise LookupError(
                        'Must have a dataframe column named "{0}"'.format(inputs[l])
                    )

            return func(*args, **kwargs)

        return wrap

    return dfcheck


def apply(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))

        return cls

    return decorate


@apply(inputvalidator(input_="ohlc"))
class SMC:

    __version__ = "0.01"

    @classmethod
    def FVG(cls, ohlc: DataFrame) -> Series:
        """
        FVG - Fair Value Gap
        A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
        Or when the previous low is higher than the next high if the current candle is bearish.
        """
        
        fvg = np.where(((ohlc["high"].shift(1) < ohlc["low"].shift(-1)) & (ohlc["close"] > ohlc["open"])) | ((ohlc["low"].shift(1) > ohlc["high"].shift(-1)) & (ohlc["close"] < ohlc["open"])),1,0)
        direction = np.where(ohlc["close"] > ohlc["open"], 1, 0)
        start = np.where(ohlc["close"] > ohlc["open"], ohlc["high"].shift(1), ohlc["low"].shift(1))
        end = np.where(ohlc["close"] > ohlc["open"], ohlc["low"].shift(-1), ohlc["high"].shift(-1))
        size = abs(ohlc["high"].shift(1) - ohlc["low"].shift(-1))

        # create a series for each of the keys in the dictionary
        fvg = pd.Series(fvg, name="FVG")
        direction = pd.Series(direction, name="Direction")
        start = pd.Series(start, name="Start")
        end = pd.Series(end, name="End")
        size = pd.Series(size, name="Size")

        print(pd.concat([fvg, direction, start, end, size], axis=1))

        return pd.concat([fvg, direction, start, end, size], axis=1)


