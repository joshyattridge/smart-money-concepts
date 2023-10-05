from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from zigzag import *


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
class smc:

    __version__ = "0.01"

    @classmethod
    def fvg(cls, ohlc: DataFrame) -> Series:
        """
        FVG - Fair Value Gap
        A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
        Or when the previous low is higher than the next high if the current candle is bearish.
        """

        fvg = np.where(
            (
                (ohlc["high"].shift(1) < ohlc["low"].shift(-1))
                & (ohlc["close"] > ohlc["open"])
            )
            | (
                (ohlc["low"].shift(1) > ohlc["high"].shift(-1))
                & (ohlc["close"] < ohlc["open"])
            ),
            np.where(ohlc["close"] > ohlc["open"], 1, -1),
            0,
        )
        top = np.where(
            ohlc["close"] > ohlc["open"], ohlc["low"].shift(-1), ohlc["low"].shift(1)
        )
        bottom = np.where(
            ohlc["close"] > ohlc["open"], ohlc["high"].shift(1), ohlc["high"].shift(-1)
        )

        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(fvg != 0)[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool)
            if fvg[i] == 1:
                mask = ohlc["low"][i + 2 :] <= top[i]
            elif fvg[i] == -1:
                mask = ohlc["high"][i + 2 :] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        # create a series for each of the keys in the dictionary
        fvg = pd.Series(fvg, name="FVG")
        top = pd.Series(top, name="Top")
        bottom = pd.Series(bottom, name="Bottom")
        mitigated_index = pd.Series(mitigated_index, name="MitigatedIndex")

        return pd.concat(
            [fvg, top, bottom, mitigated_index], axis=1
        )

    @classmethod
    def highs_lows(cls, ohlc: DataFrame, up_thresh=0.05, down_thresh=-0.05) -> Series:
        highs_lows = peak_valley_pivots(ohlc["close"], up_thresh, down_thresh)
        still_adjusting = True
        while still_adjusting:
            still_adjusting = False
            for i in range(1, len(highs_lows) - 1):
                if highs_lows[i] == 1:
                    previous_high = ohlc["high"][i - 1]
                    current_high = ohlc["high"][i]
                    next_high = ohlc["high"][i + 1]
                    if (previous_high > current_high and highs_lows[i-1] == 0) or (next_high > current_high and highs_lows[i+1] == 0):
                        highs_lows[i] = 0
                        still_adjusting = True
                        if (previous_high > next_high and highs_lows[i-1] == 0):
                            highs_lows[i - 1] = 1
                        else:
                            highs_lows[i + 1] = 1
                if highs_lows[i] == -1:
                    previous_low = ohlc["low"][i - 1]
                    current_low = ohlc["low"][i]
                    next_low = ohlc["low"][i + 1]
                    if (previous_low < current_low and highs_lows[i-1] == 0) or (next_low < current_low and highs_lows[i+1] == 0):
                        highs_lows[i] = 0
                        still_adjusting = True
                        if (previous_low < next_low and highs_lows[i-1] == 0):
                            highs_lows[i - 1] = -1
                        else:
                            highs_lows[i + 1] = -1

        levels = np.where(highs_lows != 0, np.where(highs_lows == 1, ohlc["high"], ohlc["low"]), np.nan)

        highs_lows = pd.Series(highs_lows, name="HighsLows")
        levels = pd.Series(levels, name="Levels")

        return pd.concat([highs_lows, levels], axis=1)

    @classmethod
    def ob(cls, ohlc: DataFrame) -> Series:
        """
        OB - Order Block
        This is the last candle before a FVG
        """

        # get the FVG
        fvg = cls.fvg(ohlc)

        ob = np.where((fvg["FVG"].shift(-1) != 0) & (fvg["FVG"] == 0), fvg["FVG"].shift(-1), 0)
        # top is equal to the current candles high unless the ob is -1 and the next candles high is higher than the current candles high then top is equal to the next candles high
        top = np.where(
            (ob == -1) & (ohlc["high"].shift(-1) > ohlc["high"]), ohlc["high"].shift(-1), ohlc["high"]
        )
        # bottom is equal to the current candles low unless the ob is 1 and the next candles low is lower than the current candles low then bottom is equal to the next candles low
        bottom = np.where(
            (ob == 1) & (ohlc["low"].shift(-1) < ohlc["low"]), ohlc["low"].shift(-1), ohlc["low"]
        )

        # set mitigated to np.nan
        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(ob != 0)[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool)
            if ob[i] == 1:
                mask = ohlc["low"][i + 2 :] <= top[i]
            elif ob[i] == -1:
                mask = ohlc["high"][i + 2 :] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        # create a series for each of the keys in the dictionary
        ob = pd.Series(ob, name="OB")
        top = pd.Series(top, name="Top")
        bottom = pd.Series(bottom, name="Bottom")
        mitigated_index = pd.Series(mitigated_index, name="MitigatedIndex")

        return pd.concat(
            [ob, top, bottom, mitigated_index], axis=1
        )

    @classmethod
    def liquidity(cls, ohlc: DataFrame, range_percent=0.01, up_thresh=0.05, down_thresh=-0.05) -> Series:
        """
        Liquidity
        Liquidity is when there are multiply highs within a small range of each other.
        or multiply lows within a small range of each other.
        """

        # subtract the highest high from the lowest low
        pip_range = (max(ohlc["high"]) - min(ohlc["low"])) * range_percent

        # get the highs and lows
        highs_lows = cls.highs_lows(ohlc, up_thresh, down_thresh)
        levels = highs_lows["Levels"]
        highs_lows = highs_lows["HighsLows"]

        # go through all of the high levels and if there are more than 1 within the pip range, then it is liquidity
        liquidity = np.zeros(len(ohlc), dtype=np.int32)
        buy_sell_side = np.zeros(len(ohlc), dtype=np.int32)
        liquidity_level = np.zeros(len(ohlc), dtype=np.int32)
        liquidity_end = np.zeros(len(ohlc), dtype=np.int32)
        liquidity_swept = np.zeros(len(ohlc), dtype=np.int32)

        for i in range(len(ohlc)):
            if highs_lows[i] == 1:
                high_level = levels[i]
                range_low = high_level - pip_range
                range_high = high_level + pip_range
                temp_liquidity_levels = [high_level]
                start = i
                end = i
                swept = 0
                for c in range(i + 1, len(ohlc)):
                    if highs_lows[c] == 1 and range_low <= levels[c] <= range_high:
                        end = c
                        temp_liquidity_levels.append(levels[c])
                        highs_lows.loc[c] = 0
                    if ohlc["high"][c] >= range_high:
                        swept = c
                        break
                if len(temp_liquidity_levels) > 1:
                    average_high = sum(temp_liquidity_levels) / len(
                        temp_liquidity_levels
                    )
                    liquidity[i] = 1
                    buy_sell_side[i] = 2  # 2 is buy
                    liquidity_level[i] = average_high
                    liquidity_end[i] = end
                    liquidity_swept[i] = swept

        # now do the same for the lows
        for i in range(len(ohlc)):
            if highs_lows[i] == -1:
                low_level = levels[i]
                range_low = low_level - pip_range
                range_high = low_level + pip_range
                temp_liquidity_levels = [low_level]
                start = i
                end = i
                swept = 0
                for c in range(i + 1, len(ohlc)):
                    if highs_lows[c] == -1 and range_low <= levels[c] <= range_high:
                        end = c
                        temp_liquidity_levels.append(levels[c])
                        highs_lows.loc[c] = 0
                    if ohlc["low"][c] <= range_low:
                        swept = c
                        break
                if len(temp_liquidity_levels) > 1:
                    average_low = sum(temp_liquidity_levels) / len(
                        temp_liquidity_levels
                    )
                    liquidity[i] = 1
                    buy_sell_side[i] = 1
                    liquidity_level[i] = average_low
                    liquidity_end[i] = end
                    liquidity_swept[i] = swept

        liquidity = pd.Series(liquidity, name="Liquidity")
        buy_sell_side = pd.Series(buy_sell_side, name="BuySellSide")
        level = pd.Series(liquidity_level, name="Level")
        liquidity_end = pd.Series(liquidity_end, name="End")
        liquidity_swept = pd.Series(liquidity_swept, name="Swept")

        return pd.concat(
            [liquidity, buy_sell_side, level, liquidity_end, liquidity_swept], axis=1
        )


# TODO: correct mask error for ob and fvg