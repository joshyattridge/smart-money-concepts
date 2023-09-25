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
            1,
            0,
        )
        direction = np.where(ohlc["close"] > ohlc["open"], 1, 0)
        top = np.where(
            ohlc["close"] > ohlc["open"], ohlc["low"].shift(-1), ohlc["low"].shift(1)
        )
        bottom = np.where(
            ohlc["close"] > ohlc["open"], ohlc["high"].shift(1), ohlc["high"].shift(-1)
        )
        size = abs(ohlc["high"].shift(1) - ohlc["low"].shift(-1))

        mitigated = np.zeros(len(ohlc), dtype=np.int32)
        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(fvg == 1)[0]:
            if direction[i] == 1:
                mask = ohlc["low"][i + 2 :] <= top[i]
            elif direction[i] == 0:
                mask = ohlc["high"][i + 2 :] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated[i] = 1
                mitigated_index[i] = j

        # create a series for each of the keys in the dictionary
        fvg = pd.Series(fvg, name="FVG")
        direction = pd.Series(direction, name="Direction")
        top = pd.Series(top, name="Top")
        bottom = pd.Series(bottom, name="Bottom")
        size = pd.Series(size, name="Size")
        mitigated = pd.Series(mitigated, name="Mitigated")
        mitigated_index = pd.Series(mitigated_index, name="MitigatedIndex")

        return pd.concat(
            [fvg, direction, top, bottom, size, mitigated, mitigated_index], axis=1
        )

    @classmethod
    def highs_lows(cls, ohlc: DataFrame, window=5) -> Series:
        """
        Highs and Lows
        if the current candles high is higher than the previous 5 and next 5 candles high, then it is a high.
        if the current candles low is lower than the previous 5 and next 5 candles low, then it is a low.
        """
        # create a series of highs and lows
        highs = np.where(
            (ohlc["high"] == ohlc["high"].rolling(window=window, center=True).max()),
            1,
            0,
        )
        lows = np.where(
            (ohlc["low"] == ohlc["low"].rolling(window=window, center=True).min()), 1, 0
        )

        # merge these lists together
        highs_and_lows = np.where(
            (highs == 1) | (lows == 1), np.where(highs == 1, 2, 1), 0
        )
        highs_and_lows_levels = np.where(
            (highs == 1) | (lows == 1),
            np.where(highs == 1, ohlc["high"], ohlc["low"]),
            0,
        )

        highs_and_lows_indexes = []
        for i in range(len(highs_and_lows)):
            if highs_and_lows[i] != 0:
                highs_and_lows_indexes.append(i)

        # remove the 0s from the highs and lows
        highs_and_lows = highs_and_lows[highs_and_lows != 0]
        highs_and_lows_levels = highs_and_lows_levels[highs_and_lows_levels != 0]

        for i in range(1, len(highs_and_lows)):
            if highs_and_lows[i] == 1:
                if highs_and_lows[i - 1] == 1:
                    # remove the highest one
                    if highs_and_lows_levels[i] > highs_and_lows_levels[i - 1]:
                        lows[highs_and_lows_indexes[i]] = 0
                    else:
                        lows[highs_and_lows_indexes[i - 1]] = 0
            if highs_and_lows[i] == 2:
                if highs_and_lows[i - 1] == 2:
                    # remove the lowest one
                    if highs_and_lows_levels[i] < highs_and_lows_levels[i - 1]:
                        highs[highs_and_lows_indexes[i]] = 0
                    else:
                        highs[highs_and_lows_indexes[i - 1]] = 0

        # get the value of the highs and lows
        highs_level = np.where(highs == 1, ohlc["high"], np.nan)
        lows_level = np.where(lows == 1, ohlc["low"], np.nan)

        highs = pd.Series(highs, name="Highs")
        lows = pd.Series(lows, name="Lows")
        highs_level = pd.Series(highs_level, name="HighsLevel")
        lows_level = pd.Series(lows_level, name="LowsLevel")

        return pd.concat([highs, lows, highs_level, lows_level], axis=1)

    @classmethod
    def bos_choch(cls, ohlc: DataFrame, window=5, range_percent=0.01) -> Series:
        """
        BOS - Break Of Structure
        CHOCH - Change Of Character
        A break of structure is when price breaks a previous high or low.
        A change of character is when the direction of the BOS changes
        """

        # subtract the highest high from the lowest low
        pip_range = (max(ohlc["high"]) - min(ohlc["low"])) * range_percent

        # get the highs and lows
        highs_lows = cls.highs_lows(ohlc, window=window)
        highs = highs_lows["Highs"]
        lows = highs_lows["Lows"]
        highs_level = highs_lows["HighsLevel"]
        lows_level = highs_lows["LowsLevel"]

        bos = np.zeros(len(ohlc), dtype=np.int32)
        direction = np.zeros(len(ohlc), dtype=np.int32)
        level = np.zeros(len(ohlc), dtype=np.int32)
        broken_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in range(len(ohlc)):
            if highs[i] == 1:
                high_level = highs_level[i]
                for c in range(i, len(ohlc)):
                    # if the candles high is higher than the high level, then it is a break of structure
                    if ohlc["high"][c] > (high_level + pip_range):
                        bos[i] = 1
                        direction[i] = 1
                        level[i] = high_level
                        broken_index[i] = c
                        break
            if lows[i] == 1:
                low_level = lows_level[i]
                for c in range(i, len(ohlc)):
                    # if the candles low is lower than the low level, then it is a break of structure
                    if ohlc["low"][c] < (low_level - pip_range):
                        bos[i] = 1
                        direction[i] = 0
                        level[i] = low_level
                        broken_index[i] = c
                        break

        choch = np.zeros(len(ohlc), dtype=np.int32)
        # if the previosu bos direction is different to the last bos direction, then it is a change of character
        for i in range(1, len(ohlc)):
            if bos[i] == 1 and direction[i] != direction[i - 1]:
                choch[i] = 1
                bos[i] = 0

        # create a series for each of the keys in the dictionary
        bos = pd.Series(bos, name="BOS")
        choch = pd.Series(choch, name="CHOCH")
        direction = pd.Series(direction, name="Direction")
        level = pd.Series(level, name="Level")
        broken_index = pd.Series(broken_index, name="BrokenIndex")

        return pd.concat([bos, choch, direction, level, broken_index], axis=1)

    @classmethod
    def ob(cls, ohlc: DataFrame) -> Series:
        """
        OB - Order Block
        This is the last candle before a FVG
        """

        # get the FVG
        fvg = cls.fvg(ohlc)

        ob = np.where((fvg["FVG"].shift(-1) == 1) & (fvg["FVG"] == 0), 1, 0)
        direction = fvg["Direction"].shift(-1)
        top = ohlc["high"]
        bottom = ohlc["low"]
        size = abs(ohlc["high"] - ohlc["low"])

        mitigated = np.zeros(len(ohlc), dtype=np.int32)
        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(ob == 1)[0]:
            if direction[i] == 1:
                mask = ohlc["low"][i + 2 :] <= top[i]
            elif direction[i] == 0:
                mask = ohlc["high"][i + 2 :] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated[i] = 1
                mitigated_index[i] = j

        # create a series for each of the keys in the dictionary
        ob = pd.Series(ob, name="OB")
        direction = pd.Series(direction, name="Direction")
        top = pd.Series(top, name="Top")
        bottom = pd.Series(bottom, name="Bottom")
        size = pd.Series(size, name="Size")
        mitigated = pd.Series(mitigated, name="Mitigated")
        mitigated_index = pd.Series(mitigated_index, name="MitigatedIndex")

        return pd.concat(
            [ob, direction, top, bottom, size, mitigated, mitigated_index], axis=1
        )

    @classmethod
    def liquidity(cls, ohlc: DataFrame, range_percent=0.01) -> Series:
        """
        Liquidity
        Liquidity is when there are multiply highs within a small range of each other.
        or multiply lows within a small range of each other.
        """

        # subtract the highest high from the lowest low
        pip_range = (max(ohlc["high"]) - min(ohlc["low"])) * range_percent

        # get the highs and lows
        highs_lows = cls.highs_lows(ohlc)
        highs = highs_lows["Highs"]
        high_levels = highs_lows["HighsLevel"]
        lows = highs_lows["Lows"]
        low_levels = highs_lows["LowsLevel"]

        # go through all of the high levels and if there are more than 1 within the pip range, then it is liquidity
        liquidity = np.zeros(len(ohlc), dtype=np.int32)
        buy_sell_side = np.zeros(len(ohlc), dtype=np.int32)
        liquidity_level = np.zeros(len(ohlc), dtype=np.int32)
        liquidity_end = np.zeros(len(ohlc), dtype=np.int32)
        liquidity_swept = np.zeros(len(ohlc), dtype=np.int32)

        for i in range(len(ohlc)):
            if highs[i] == 1:
                high_level = high_levels[i]
                range_low = high_level - pip_range
                range_high = high_level + pip_range
                temp_liquidity_levels = [high_level]
                start = i
                end = i
                swept = 0
                for c in range(i + 1, len(ohlc)):
                    if highs[c] == 1 and range_low <= high_levels[c] <= range_high:
                        end = c
                        temp_liquidity_levels.append(high_levels[c])
                        highs.loc[c] = 0
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
            if lows[i] == 1:
                low_level = low_levels[i]
                range_low = low_level - pip_range
                range_high = low_level + pip_range
                temp_liquidity_levels = [low_level]
                start = i
                end = i
                swept = 0
                for c in range(i + 1, len(ohlc)):
                    if lows[c] == 1 and range_low <= low_levels[c] <= range_high:
                        end = c
                        temp_liquidity_levels.append(low_levels[c])
                        lows.loc[c] = 0
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
