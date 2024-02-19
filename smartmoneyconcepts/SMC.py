from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from zigzag import *
from finta import TA


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

    __version__ = "0.0.13"

    atr_multiplier = 1.5
    range_percent = 0.01

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
            mask = np.zeros(len(ohlc), dtype=np.bool_)
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

        return pd.concat([fvg, top, bottom, mitigated_index], axis=1)

    @classmethod
    def highs_lows(cls, ohlc: DataFrame) -> Series:
        pip_range = TA.ATR(ohlc, 14) / ohlc["close"].iloc[-1] * cls.atr_multiplier
        pip_range = pip_range.iloc[-1]

        highs_lows = peak_valley_pivots(ohlc["close"], abs(pip_range), -abs(pip_range))

        still_adjusting = True
        while still_adjusting:
            still_adjusting = False
            for i in range(1, len(highs_lows) - 1):
                if highs_lows[i] == 1:
                    previous_high = ohlc["high"][i - 1]
                    current_high = ohlc["high"][i]
                    next_high = ohlc["high"][i + 1]
                    if (previous_high > current_high and highs_lows[i - 1] == 0) or (
                        next_high > current_high and highs_lows[i + 1] == 0
                    ):
                        highs_lows[i] = 0
                        still_adjusting = True
                        if previous_high > next_high and highs_lows[i - 1] == 0:
                            highs_lows[i - 1] = 1
                        else:
                            highs_lows[i + 1] = 1
                if highs_lows[i] == -1:
                    previous_low = ohlc["low"][i - 1]
                    current_low = ohlc["low"][i]
                    next_low = ohlc["low"][i + 1]
                    if (previous_low < current_low and highs_lows[i - 1] == 0) or (
                        next_low < current_low and highs_lows[i + 1] == 0
                    ):
                        highs_lows[i] = 0
                        still_adjusting = True
                        if previous_low < next_low and highs_lows[i - 1] == 0:
                            highs_lows[i - 1] = -1
                        else:
                            highs_lows[i + 1] = -1

        levels = np.where(
            highs_lows != 0,
            np.where(highs_lows == 1, ohlc["high"], ohlc["low"]),
            np.nan,
        )

        highs_lows = pd.Series(highs_lows, name="HighsLows")
        levels = pd.Series(levels, name="Levels")

        return pd.concat([highs_lows, levels], axis=1)

    @classmethod
    def bos_choch(
        cls, ohlc: DataFrame, close_break=True, filter_liquidity=False
    ) -> Series:
        """
        BOS - Breakout Signal
        CHoCH - Change of Character signal
        This is when the current candle is the first candle to break out of a range.
        """

        # get the highs and lows
        highs_lows = cls.highs_lows(ohlc)
        levels = highs_lows["Levels"]
        highs_lows = highs_lows["HighsLows"]

        # filter out the highs and lows used if it is aligned with liquidity
        if filter_liquidity:
            liquidity = cls.liquidity(ohlc)
            liquidity = liquidity["Liquidity"]
            for i in range(len(highs_lows)):
                if liquidity[i] != 0 and highs_lows[i] != 0:
                    highs_lows[i] = 0

        levels_order = []
        highs_lows_order = []

        bos = np.zeros(len(ohlc), dtype=np.int32)
        choch = np.zeros(len(ohlc), dtype=np.int32)
        level = np.zeros(len(ohlc), dtype=np.float32)

        last_positions = []

        for i in range(len(highs_lows)):
            if highs_lows[i] != 0:
                levels_order.append(levels[i])
                highs_lows_order.append(highs_lows[i])
                if len(levels_order) >= 4:
                    # bullish bos
                    bos[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                levels_order[-4]
                                < levels_order[-2]
                                < levels_order[-3]
                                < levels_order[-1]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        levels_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # bearish bos
                    bos[last_positions[-2]] = (
                        -1
                        if (
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                levels_order[-4]
                                > levels_order[-2]
                                > levels_order[-3]
                                > levels_order[-1]
                            )
                        )
                        else bos[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        levels_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # bullish choch
                    choch[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                levels_order[-1]
                                > levels_order[-3]
                                > levels_order[-4]
                                > levels_order[-2]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        levels_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                    # bearish choch
                    choch[last_positions[-2]] = (
                        -1
                        if (
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                levels_order[-1]
                                < levels_order[-3]
                                < levels_order[-4]
                                < levels_order[-2]
                            )
                        )
                        else choch[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        levels_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                last_positions.append(i)

        broken = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            # if the bos is 1 then check if the candles high has gone above the level
            if bos[i] == 1 or choch[i] == 1:
                mask = ohlc["close" if close_break else "high"][i + 2 :] > level[i]
            # if the bos is -1 then check if the candles low has gone below the level
            elif bos[i] == -1 or choch[i] == -1:
                mask = ohlc["close" if close_break else "low"][i + 2 :] < level[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                broken[i] = j

        # remove the ones that aren't broken
        for i in np.where(
            np.logical_and(np.logical_or(bos != 0, choch != 0), broken == 0)
        )[0]:
            bos[i] = 0
            choch[i] = 0
            level[i] = 0

        # there can only be one high or low between the bos/choch and the broken index
        for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
            # count the number of highs or lows between the bos/choch and the broken index
            count = 0
            for j in range(i, broken[i]):
                if highs_lows[j] != 0:
                    count += 1
            # if there is more than 1 high or low then remove the bos/choch
            if count > 2:
                bos[i] = 0
                choch[i] = 0
                level[i] = 0

        bos = pd.Series(bos, name="BOS")
        choch = pd.Series(choch, name="CHOCH")
        level = pd.Series(level, name="Level")
        broken = pd.Series(broken, name="BrokenIndex")

        return pd.concat([bos, choch, level, broken], axis=1)

    @classmethod
    def ob(cls, ohlc: DataFrame) -> Series:
        """
        OB - Order Block
        This is the last candle before a FVG
        """

        # get the FVG
        fvg = cls.fvg(ohlc)

        ob = np.where(
            (fvg["FVG"].shift(-1) != 0) & (fvg["FVG"] == 0), fvg["FVG"].shift(-1), 0
        )
        # top is equal to the current candles high unless the ob is -1 and the next candles high is higher than the current candles high then top is equal to the next candles high
        top = np.where(
            (ob == -1) & (ohlc["high"].shift(-1) > ohlc["high"]),
            ohlc["high"].shift(-1),
            ohlc["high"],
        )
        # bottom is equal to the current candles low unless the ob is 1 and the next candles low is lower than the current candles low then bottom is equal to the next candles low
        bottom = np.where(
            (ob == 1) & (ohlc["low"].shift(-1) < ohlc["low"]),
            ohlc["low"].shift(-1),
            ohlc["low"],
        )

        # set mitigated to np.nan
        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(ob != 0)[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
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

        return pd.concat([ob, top, bottom, mitigated_index], axis=1)

    @classmethod
    def liquidity(cls, ohlc: DataFrame) -> Series:
        """
        Liquidity
        Liquidity is when there are multiply highs within a small range of each other.
        or multiply lows within a small range of each other.
        """

        # subtract the highest high from the lowest low
        pip_range = (max(ohlc["high"]) - min(ohlc["low"])) * cls.range_percent

        # get the highs and lows
        highs_lows = cls.highs_lows(ohlc)
        levels = highs_lows["Levels"]
        highs_lows = highs_lows["HighsLows"]

        # go through all of the high levels and if there are more than 1 within the pip range, then it is liquidity
        liquidity = np.zeros(len(ohlc), dtype=np.int32)
        liquidity_level = np.zeros(len(ohlc), dtype=np.float32)
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
                    liquidity[i] = -1
                    liquidity_level[i] = average_low
                    liquidity_end[i] = end
                    liquidity_swept[i] = swept

        liquidity = pd.Series(liquidity, name="Liquidity")
        level = pd.Series(liquidity_level, name="Level")
        liquidity_end = pd.Series(liquidity_end, name="End")
        liquidity_swept = pd.Series(liquidity_swept, name="Swept")

        return pd.concat([liquidity, level, liquidity_end, liquidity_swept], axis=1)

    @classmethod
    def volumized_ob(
        cls,
        ohlc: pd.DataFrame,
        maxDistanceToLastBar: int,
        swingLength: int,
        obEndMethod: str,
        maxOrderBlocks: int,
    ):
        """
        Volumized Order Blocks
        This method calculates volumized order blocks based on the provided OHLC data and parameters.
        """
        ob_swings = {"top": [], "bottom": []}
        swing_type = 0

        # Calculate the highest high and lowest low for the specified length
        upper = ohlc["high"].rolling(window=swingLength).max()
        lower = ohlc["low"].rolling(window=swingLength).min()

        # Concatenate upper and lower to df
        ohlc["upper"] = upper
        ohlc["lower"] = lower

        # Initialize the previous swing type
        prev_swing_type = 0

        # Iterate over each index in the dataframe
        for i in range(len(ohlc)):
            try:
                # Determine the swing type
                if ohlc["high"].iloc[i] > upper.iloc[i + swingLength]:
                    swing_type = 0
                elif ohlc["low"].iloc[i] < lower.iloc[i + swingLength]:
                    swing_type = 1

                # Check if it's a new top or bottom
                if swing_type == 0 and prev_swing_type != 0:
                    ob_swings["top"].append(
                        {
                            "index": i,
                            "loc": ohlc.index[i],
                            "price": ohlc["high"].iloc[i],
                            "volume": ohlc["volume"].iloc[i],
                            "crossed": False,
                        }
                    )
                elif swing_type == 1 and prev_swing_type != 1:
                    ob_swings["bottom"].append(
                        {
                            "index": i,
                            "loc": ohlc.index[i],
                            "price": ohlc["low"].iloc[i],
                            "volume": ohlc["volume"].iloc[i],
                            "crossed": False,
                        }
                    )

                # Update the previous swing type
                prev_swing_type = swing_type
            except IndexError:
                pass

        bullish_order_blocks = []
        bearish_order_blocks = []

        last_bar_index = len(ohlc)
        bar_index = max(0, last_bar_index - maxDistanceToLastBar)

        if bar_index >= 0:

            # Bullish Order Block
            for close_index in range(bar_index, last_bar_index):
                close_price = ohlc["close"].iloc[close_index]

                if len(bullish_order_blocks) > 0:
                    for i in range(len(bullish_order_blocks) - 1, -1, -1):
                        currentOB = bullish_order_blocks[i]
                        if not currentOB["breaker"]:
                            if (
                                obEndMethod == "Wick"
                                and ohlc.low.iloc[close_index] < currentOB["bottom"]
                            ) or (
                                obEndMethod != "Wick"
                                and min(
                                    ohlc.open.iloc[close_index],
                                    ohlc.close.iloc[close_index],
                                )
                                < currentOB["bottom"]
                            ):
                                currentOB["breaker"] = True
                                currentOB["breakTime"] = ohlc.index[close_index - 1]
                                currentOB["bbvolume"] = ohlc["volume"].iloc[
                                    close_index - 1
                                ]
                        else:
                            if ohlc.high.iloc[close_index] > currentOB["top"]:
                                bullish_order_blocks.pop(i)

                last_top_index = None
                for i in range(len(ob_swings["top"])):
                    if ob_swings["top"][i]["index"] < close_index:
                        last_top_index = i
                if last_top_index is not None:
                    swing_top_price = ob_swings["top"][last_top_index]["price"]
                    if (
                        close_price > swing_top_price
                        and not ob_swings["top"][last_top_index]["crossed"]
                    ):
                        ob_swings["top"][last_top_index]["crossed"] = True
                        boxBtm = ohlc.high.iloc[close_index - 1]
                        boxTop = ohlc.low.iloc[close_index - 1]
                        boxLoc = ohlc.index[close_index - 1]
                        for j in range(
                            1, close_index - ob_swings["top"][last_top_index]["index"]
                        ):
                            boxBtm = min(
                                ohlc.low.iloc[
                                    ob_swings["top"][last_top_index]["index"] + j
                                ],
                                boxBtm,
                            )
                            if (
                                boxBtm
                                == ohlc.low.iloc[
                                    ob_swings["top"][last_top_index]["index"] + j
                                ]
                            ):
                                boxTop = ohlc.high.iloc[
                                    ob_swings["top"][last_top_index]["index"] + j
                                ]
                            boxLoc = (
                                ohlc.index[
                                    ob_swings["top"][last_top_index]["index"] + j
                                ]
                                if boxBtm
                                == ohlc.low.iloc[
                                    ob_swings["top"][last_top_index]["index"] + j
                                ]
                                else boxLoc
                            )

                        newOrderBlockInfo = {
                            "top": boxTop,
                            "bottom": boxBtm,
                            "volume": ohlc["volume"].iloc[close_index]
                            + ohlc["volume"].iloc[close_index - 1]
                            + ohlc["volume"].iloc[close_index - 2],
                            "type": "Bull",
                            "loc": boxLoc,
                            "loc_number": close_index,
                            "index": len(bullish_order_blocks),
                            "oblowvolume": ohlc["volume"].iloc[close_index - 2],
                            "obhighvolume": (
                                ohlc["volume"].iloc[close_index]
                                + ohlc["volume"].iloc[close_index - 1]
                            ),
                            "breaker": False,
                        }
                        bullish_order_blocks.insert(0, newOrderBlockInfo)
                        if len(bullish_order_blocks) > maxOrderBlocks:
                            bullish_order_blocks.pop()

            for close_index in range(bar_index, last_bar_index):
                close_price = ohlc["close"].iloc[close_index]

                # Bearish Order Block
                if len(bearish_order_blocks) > 0:
                    for i in range(len(bearish_order_blocks) - 1, -1, -1):
                        currentOB = bearish_order_blocks[i]
                        if not currentOB["breaker"]:
                            if (
                                obEndMethod == "Wick"
                                and ohlc.high.iloc[close_index] > currentOB["top"]
                            ) or (
                                obEndMethod != "Wick"
                                and max(
                                    ohlc.open.iloc[close_index],
                                    ohlc.close.iloc[close_index],
                                )
                                > currentOB["top"]
                            ):
                                currentOB["breaker"] = True
                                currentOB["breakTime"] = ohlc.index[close_index]
                                currentOB["bbvolume"] = ohlc["volume"].iloc[close_index]
                        else:
                            if ohlc.low.iloc[close_index] < currentOB["bottom"]:
                                bearish_order_blocks.pop(i)

                last_btm_index = None
                for i in range(len(ob_swings["bottom"])):
                    if ob_swings["bottom"][i]["index"] < close_index:
                        last_btm_index = i
                if last_btm_index is not None:
                    swing_btm_price = ob_swings["bottom"][last_btm_index]["price"]
                    if (
                        close_price < swing_btm_price
                        and not ob_swings["bottom"][last_btm_index]["crossed"]
                    ):
                        ob_swings["bottom"][last_btm_index]["crossed"] = True
                        boxBtm = ohlc.low.iloc[close_index - 1]
                        boxTop = ohlc.high.iloc[close_index - 1]
                        boxLoc = ohlc.index[close_index - 1]
                        for j in range(
                            1,
                            close_index - ob_swings["bottom"][last_btm_index]["index"],
                        ):
                            boxTop = max(
                                ohlc.high.iloc[
                                    ob_swings["bottom"][last_btm_index]["index"] + j
                                ],
                                boxTop,
                            )
                            boxBtm = (
                                ohlc.low.iloc[
                                    ob_swings["bottom"][last_btm_index]["index"] + j
                                ]
                                if boxTop
                                == ohlc.high.iloc[
                                    ob_swings["bottom"][last_btm_index]["index"] + j
                                ]
                                else boxBtm
                            )
                            boxLoc = (
                                ohlc.index[
                                    ob_swings["bottom"][last_btm_index]["index"] + j
                                ]
                                if boxTop
                                == ohlc.high.iloc[
                                    ob_swings["bottom"][last_btm_index]["index"] + j
                                ]
                                else boxLoc
                            )

                        newOrderBlockInfo = {
                            "top": boxTop,
                            "bottom": boxBtm,
                            "volume": ohlc["volume"].iloc[close_index]
                            + ohlc["volume"].iloc[close_index - 1]
                            + ohlc["volume"].iloc[close_index - 2],
                            "type": "Bear",
                            "loc": boxLoc,
                            "loc_number": close_index,
                            "index": len(bearish_order_blocks),
                            "oblowvolume": (
                                ohlc["volume"].iloc[close_index]
                                + ohlc["volume"].iloc[close_index - 1]
                            ),
                            "obhighvolume": ohlc["volume"].iloc[close_index - 2],
                            "breaker": False,
                        }
                        bearish_order_blocks.insert(0, newOrderBlockInfo)
                        if len(bearish_order_blocks) > maxOrderBlocks:
                            bearish_order_blocks.pop()

        return bullish_order_blocks, bearish_order_blocks
