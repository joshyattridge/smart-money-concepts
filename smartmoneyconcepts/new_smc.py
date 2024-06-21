from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from datetime import datetime


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
    __version__ = "0.0.21"

    @classmethod
    def fvg(cls, ohlc: DataFrame, join_consecutive=False) -> Series:
        """
        FVG - Fair Value Gap
        A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
        Or when the previous low is higher than the next high if the current candle is bearish.

        parameters:
        join_consecutive: bool - if there are multiple FVG in a row then they will be merged into one using the highest top and the lowest bottom

        returns:
        FVG = 1 if bullish fair value gap, -1 if bearish fair value gap
        Top = the top of the fair value gap
        Bottom = the bottom of the fair value gap
        MitigatedIndex = the index of the candle that mitigated the fair value gap
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
            np.nan,
        )

        top = np.where(
            ~np.isnan(fvg),
            np.where(
                ohlc["close"] > ohlc["open"],
                ohlc["low"].shift(-1),
                ohlc["low"].shift(1),
            ),
            np.nan,
        )

        bottom = np.where(
            ~np.isnan(fvg),
            np.where(
                ohlc["close"] > ohlc["open"],
                ohlc["high"].shift(1),
                ohlc["high"].shift(-1),
            ),
            np.nan,
        )

        # if there are multiple consecutive fvg then join them together using the highest top and lowest bottom and the last index
        if join_consecutive:
            for i in range(len(fvg) - 1):
                if fvg[i] == fvg[i + 1]:
                    top[i + 1] = max(top[i], top[i + 1])
                    bottom[i + 1] = min(bottom[i], bottom[i + 1])
                    fvg[i] = top[i] = bottom[i] = np.nan

        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(~np.isnan(fvg))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            if fvg[i] == 1:
                mask = ohlc["low"][i + 2 :] <= top[i]
            elif fvg[i] == -1:
                mask = ohlc["high"][i + 2 :] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        mitigated_index = np.where(np.isnan(fvg), np.nan, mitigated_index)

        return pd.concat(
            [
                pd.Series(fvg, name="FVG"),
                pd.Series(top, name="Top"),
                pd.Series(bottom, name="Bottom"),
                pd.Series(mitigated_index, name="MitigatedIndex"),
            ],
            axis=1,
        )

    @classmethod
    def swing_highs_lows(cls, ohlc: DataFrame, swing_length: int = 50) -> Series:
        """
        Swing Highs and Lows
        A swing high is when the current high is the highest high out of the swing_length amount of candles before and after.
        A swing low is when the current low is the lowest low out of the swing_length amount of candles before and after.

        parameters:
        swing_length: int - the amount of candles to look back and forward to determine the swing high or low

        returns:
        HighLow = 1 if swing high, -1 if swing low
        Level = the level of the swing high or low
        """

        swing_length *= 2
        # set the highs to 1 if the current high is the highest high in the last 5 candles and next 5 candles
        swing_highs_lows = np.where(
            ohlc["high"]
            == ohlc["high"].shift(-(swing_length // 2)).rolling(swing_length).max(),
            1,
            np.where(
                ohlc["low"]
                == ohlc["low"].shift(-(swing_length // 2)).rolling(swing_length).min(),
                -1,
                np.nan,
            ),
        )

        while True:
            positions = np.where(~np.isnan(swing_highs_lows))[0]

            if len(positions) < 2:
                break

            current = swing_highs_lows[positions[:-1]]
            next = swing_highs_lows[positions[1:]]

            highs = ohlc["high"].iloc[positions[:-1]].values
            lows = ohlc["low"].iloc[positions[:-1]].values

            next_highs = ohlc["high"].iloc[positions[1:]].values
            next_lows = ohlc["low"].iloc[positions[1:]].values

            index_to_remove = np.zeros(len(positions), dtype=bool)

            consecutive_highs = (current == 1) & (next == 1)
            index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
            index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)

            consecutive_lows = (current == -1) & (next == -1)
            index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
            index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)

            if not index_to_remove.any():
                break

            swing_highs_lows[positions[index_to_remove]] = np.nan

        positions = np.where(~np.isnan(swing_highs_lows))[0]

        if len(positions) > 0:
            if swing_highs_lows[positions[0]] == 1:
                swing_highs_lows[0] = -1
            if swing_highs_lows[positions[0]] == -1:
                swing_highs_lows[0] = 1
            if swing_highs_lows[positions[-1]] == -1:
                swing_highs_lows[-1] = 1
            if swing_highs_lows[positions[-1]] == 1:
                swing_highs_lows[-1] = -1

        level = np.where(
            ~np.isnan(swing_highs_lows),
            np.where(swing_highs_lows == 1, ohlc["high"], ohlc["low"]),
            np.nan,
        )

        return pd.concat(
            [
                pd.Series(swing_highs_lows, name="HighLow"),
                pd.Series(level, name="Level"),
            ],
            axis=1,
        )

    @classmethod
    def swing_highs_lows_forward(
        cls,
        ohlc: pd.DataFrame,
        swing_length: int,
        remove_consecutive_points: bool = False,
    ) -> pd.DataFrame:
        """
        Swing Highs and Lows
        A swing high is when the current high is the highest high out of the swing_length amount of candles after current candle.
        A swing low is when the current low is the lowest low out of the swing_length amount of candles after current candle.

        parameters:
        swing_length: int - the amount of candles to look forward to determine the swing high or low
        remove_consecutive_points: bool - remove consecutive highs and lows points.

        returns:
        HighLow = 1 if swing high, -1 if swing low
        Level = the level of the swing high or low
        """

        if swing_length <= 0:
            raise ValueError("swing_length should be positive integer")

        swing_highs_lows = np.full(len(ohlc), np.nan)

        future_highs = ohlc["high"].shift(1)
        future_lows = ohlc["low"].shift(1)

        swing_highs = future_highs == future_highs.rolling(
            window=swing_length + 1, min_periods=swing_length
        ).max().shift(-swing_length)

        swing_lows = future_lows == future_lows.rolling(
            window=swing_length + 1, min_periods=swing_length
        ).min().shift(-swing_length)

        # set swing high lows
        swing_highs_lows[swing_highs] = 1
        swing_highs_lows[swing_lows] = -1
        swing_highs_lows = np.roll(swing_highs_lows, -1)

        while True:
            positions = np.where(~np.isnan(swing_highs_lows))[0]

            if len(positions) < 2:
                break

            current = swing_highs_lows[positions[:-1]]
            next = swing_highs_lows[positions[1:]]

            highs = ohlc["high"].iloc[positions[:-1]].values
            lows = ohlc["low"].iloc[positions[:-1]].values

            next_highs = ohlc["high"].iloc[positions[1:]].values
            next_lows = ohlc["low"].iloc[positions[1:]].values

            index_to_remove = np.zeros(len(positions), dtype=bool)

            consecutive_highs = (current == 1) & (next == 1)

            if remove_consecutive_points:
                index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
                index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)
            else:

                index_to_remove[:-1] |= (
                    consecutive_highs
                    & (highs < next_highs)
                    & (positions[1:] - positions[:-1] < swing_length)
                )
                index_to_remove[1:] |= (
                    consecutive_highs
                    & (highs >= next_highs)
                    & (positions[1:] - positions[:-1] < swing_length)
                )

            consecutive_lows = (current == -1) & (next == -1)
            if remove_consecutive_points:
                index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
                index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)
            else:
                index_to_remove[:-1] |= (
                    consecutive_lows
                    & (lows > next_lows)
                    & (positions[1:] - positions[:-1] < swing_length)
                )
                index_to_remove[1:] |= (
                    consecutive_lows
                    & (lows <= next_lows)
                    & (positions[1:] - positions[:-1] < swing_length)
                )

            if not index_to_remove.any():
                break

            swing_highs_lows[positions[index_to_remove]] = np.nan

        positions = np.where(~np.isnan(swing_highs_lows))[0]

        if len(positions) > 0:
            before_first = ohlc.iloc[: positions[0]]
            if len(before_first) > 0:
                if swing_highs_lows[positions[0]] == 1:
                    swing_highs_lows[before_first["low"].idxmin()] = -1
                else:
                    swing_highs_lows[before_first["high"].idxmax()] = 1

            after_last = ohlc.iloc[positions[-1] :]
            if len(after_last) > 0:
                if swing_highs_lows[positions[-1]] == 1:
                    swing_highs_lows[after_last["low"].idxmin()] = -1
                else:
                    swing_highs_lows[after_last["high"].idxmax()] = 1

        level = np.where(
            ~np.isnan(swing_highs_lows),
            np.where(swing_highs_lows == 1, ohlc["high"], ohlc["low"]),
            np.nan,
        )

        return pd.concat(
            [
                pd.Series(swing_highs_lows, name="HighLow"),
                pd.Series(level, name="Level"),
            ],
            axis=1,
        )

    @classmethod
    def bos_choch(
        cls, ohlc: DataFrame, swing_highs_lows: pd.DataFrame, close_break: bool = True
    ):
        """
        BOS - Break of Structure
        CHoCH - Change of Character
        these are both indications of market structure changing

        parameters:
        swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
        close_break: bool - if True then the break of structure will be mitigated based on the close of the candle otherwise it will be the high/low.

        returns:
        BOS = 1 if bullish break of structure, -1 if bearish break of structure
        CHOCH = 1 if bullish change of character, -1 if bearish change of character
        Level = the level of the break of structure or change of character
        BrokenIndex = the index of the candle that broke the level
        BrokenDate = the datetime of the candle that broke the level
        """
        shl = swing_highs_lows.copy()

        _high_low = shl["HighLow"].values
        _dates = ohlc["date"].values.astype("datetime64[ns]")
        _high = ohlc["high"].values
        _low = ohlc["low"].values
        _close = ohlc["close"].values

        shl_len = len(_dates)
        bos = np.zeros(shl_len, dtype=np.int16)
        choch = np.zeros(shl_len, dtype=np.int16)
        bos_level = np.zeros(shl_len, dtype=np.float64)
        broken_index = np.zeros(shl_len, dtype=np.int64)
        broken_date = np.full(shl_len, np.datetime64("NaT"), dtype="datetime64[ns]")

        very_high_no = 999999
        very_low_no = -1
        last_swing_high = very_high_no
        last_swing_low = very_low_no
        last_swing_high_index = 0
        last_swing_low_index = 0

        trend = _high_low[np.flatnonzero(~np.isnan(_high_low))[0]]

        if close_break:
            for i in range(shl_len):

                # new structure completed on up side
                if _close[i] >= last_swing_high:
                    if trend == 1:
                        bos[last_swing_high_index] = 1
                    else:
                        choch[last_swing_high_index] = 1
                        trend = 1

                    bos_level[last_swing_high_index] = last_swing_high
                    broken_date[last_swing_high_index] = _dates[i]
                    broken_index[last_swing_high_index] = i

                    last_swing_high = very_high_no
                    last_swing_high_index = very_low_no

                # new structure completed on down side
                if _close[i] <= last_swing_low:
                    if trend == -1:
                        bos[last_swing_low_index] = -1
                    else:
                        choch[last_swing_low_index] = -1
                        trend = -1

                    bos_level[last_swing_low_index] = last_swing_low
                    broken_date[last_swing_low_index] = _dates[i]
                    broken_index[last_swing_low_index] = i

                    last_swing_low = 0
                    last_swing_low_index = 0

                if _high_low[i] == 1:
                    last_swing_high = _high[i]
                    last_swing_high_index = i

                if _high_low[i] == -1:
                    last_swing_low = _low[i]
                    last_swing_low_index = i

        else:
            for i in range(shl_len):

                # new structure completed on up side
                if _high[i] >= last_swing_high:
                    if trend == 1:
                        bos[last_swing_high_index] = 1
                    else:
                        choch[last_swing_high_index] = 1
                        trend = 1

                    bos_level[last_swing_high_index] = last_swing_high
                    broken_date[last_swing_high_index] = _dates[i]
                    broken_index[i] = i

                    last_swing_high = very_high_no
                    last_swing_high_index = very_low_no

                # new structure completed on down side
                if _low[i] <= last_swing_low:

                    if trend == -1:
                        bos[last_swing_low_index] = -1
                    else:
                        choch[last_swing_low_index] = -1
                        trend = -1

                    bos_level[last_swing_low_index] = last_swing_low
                    broken_date[last_swing_low_index] = _dates[i]
                    broken_index[i] = i

                    last_swing_low = 0
                    last_swing_low_index = 0

                if _high_low[i] == 1:
                    last_swing_high = _high[i]
                    last_swing_high_index = i

                if _high_low[i] == -1:
                    last_swing_low = _low[i]
                    last_swing_low_index = i

        # replace all the 0s with np.nan
        bos = np.where(bos != 0, bos, np.nan)
        choch = np.where(choch != 0, choch, np.nan)
        bos_level = np.where(bos_level != 0, bos_level, np.nan)
        broken_index = np.where(broken_index != 0, bos_level, np.nan)

        bos = pd.Series(bos, name="BOS")
        choch = pd.Series(choch, name="CHOCH")
        bos_level = pd.Series(bos_level, name="Level")
        broken_index = pd.Series(broken_index, name="BrokenIndex")
        broken_date = pd.Series(broken_date, name="BrokenDate")

        return pd.concat(
            [
                bos,
                choch,
                bos_level,
                broken_index,
                broken_date,
            ],
            axis=1,
        )

    @classmethod
    def ob(
        cls,
        ohlc: DataFrame,
        shl: pd.DataFrame,
        close_mitigation: bool = True,
        use_bos: bool = True,
        use_choch: bool = True,
    ):
        """
        OB - Order Blocks
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
        """

        _high_low = shl["HighLow"].values
        _dates = ohlc["date"].values.astype("datetime64[ns]")
        _high = ohlc["high"].values
        _low = ohlc["low"].values
        _close = ohlc["close"].values

        shl_len = len(_dates)
        bos = np.zeros(shl_len, dtype=np.int16)
        choch = np.zeros(shl_len, dtype=np.int16)
        bos_level = np.zeros(shl_len, dtype=np.float64)
        structure_complete_date = np.full(
            shl_len, np.datetime64("NaT"), dtype="datetime64[ns]"
        )

        ob = np.zeros(shl_len, dtype=np.int16)
        ob_confirm_date = np.full(shl_len, np.datetime64("NaT"), dtype="datetime64[ns]")
        ob_mitigation_date = np.full(
            shl_len, np.datetime64("NaT"), dtype="datetime64[ns]"
        )
        ob_mitigated_index = np.zeros(shl_len, dtype=np.int64)
        obVolume = np.zeros(len(ohlc), dtype=np.float32)
        lowVolume = np.zeros(len(ohlc), dtype=np.float32)
        highVolume = np.zeros(len(ohlc), dtype=np.float32)
        percentage = np.zeros(len(ohlc), dtype=np.int32)

        very_high_no = 9999999
        very_low_no = -1
        last_swing_high_index = 0
        last_swing_low_index = 0
        last_swing_high = very_high_no
        last_swing_low = very_low_no
        last_ob_high = very_high_no
        last_ob_low = very_low_no
        ob_high_indices = np.zeros((0, 2), dtype=np.int64)
        ob_low_indices = np.zeros((0, 2), dtype=np.int64)

        trend = _high_low[np.flatnonzero(~np.isnan(_high_low))[0]]
        if close_mitigation:
            for i in range(shl_len):
                use_structure = False

                # new structure completed on up side
                if _close[i] >= last_swing_high:
                    if trend == 1:
                        bos[last_swing_high_index] = 1
                        if use_bos:
                            use_structure = True
                    else:
                        choch[last_swing_high_index] = 1
                        trend = 1
                        if use_choch:
                            use_structure = True

                    bos_level[last_swing_high_index] = last_swing_high
                    structure_complete_date[last_swing_high_index] = _dates[i]

                    if use_structure:
                        ob[last_swing_low_index] = -1
                        ob_confirm_date[last_swing_low_index] = _dates[i]
                        ob_low_indices = np.concatenate(
                            (
                                ob_low_indices,
                                np.array([[last_swing_low_index, last_swing_low]]),
                            ),
                            axis=0,
                        )

                    if last_swing_low > last_ob_low:
                        last_ob_low = last_swing_low

                    last_swing_high = very_high_no
                    last_swing_high_index = very_low_no

                # upper order block mitigated
                if _close[i] >= last_ob_high:
                    idx_to_delete = None
                    for row_idx, row in enumerate(ob_high_indices):
                        ob_high_index, swing_high = row
                        if last_ob_high == swing_high:
                            idx_to_delete = row_idx
                            ob_mitigation_date[int(ob_high_index)] = _dates[i]
                            ob_mitigated_index[int(ob_high_index)] = i

                    mask = np.ones(ob_high_indices.shape[0], dtype=bool)
                    if idx_to_delete is not None:
                        mask[idx_to_delete] = False
                        ob_high_indices = ob_high_indices[mask]
                    if np.any(ob_high_indices):
                        last_ob_high = ob_high_indices[:, 1].min()
                    else:
                        last_ob_high = very_high_no

                # new structure completed on down side
                if _close[i] <= last_swing_low:
                    if trend == -1:
                        bos[last_swing_low_index] = -1
                        if use_bos:
                            use_structure = True
                    else:
                        choch[last_swing_low_index] = -1
                        trend = -1
                        if use_choch:
                            use_structure = True

                    bos_level[last_swing_low_index] = last_swing_low
                    structure_complete_date[last_swing_low_index] = _dates[i]

                    if use_structure:
                        ob[last_swing_high_index] = 1
                        ob_confirm_date[last_swing_high_index] = _dates[i]

                        ob_high_indices = np.concatenate(
                            (
                                ob_high_indices,
                                np.array([[last_swing_high_index, last_swing_high]]),
                            ),
                            axis=0,
                        )

                    if last_swing_high < last_ob_high:
                        last_ob_high = last_swing_high

                    last_swing_low = 0
                    last_swing_low_index = 0

                # lower order block mitigated
                if _close[i] <= last_ob_low:

                    idx_to_delete = None
                    for row_idx, row in enumerate(ob_low_indices):
                        ob_low_index, swing_low = row
                        if last_ob_low == swing_low:
                            idx_to_delete = row_idx
                            ob_mitigation_date[int(ob_low_index)] = _dates[i]
                            ob_mitigated_index[int(ob_low_index)] = i

                    mask = np.ones(ob_low_indices.shape[0], dtype=bool)
                    if idx_to_delete is not None:
                        mask[idx_to_delete] = False
                        ob_low_indices = ob_low_indices[mask]

                    if np.any(ob_low_indices):
                        last_ob_low = ob_low_indices[:, 1].max()
                    else:
                        last_ob_low = very_low_no

                if _high_low[i] == 1:
                    last_swing_high = _high[i]
                    last_swing_high_index = i

                if _high_low[i] == -1:
                    last_swing_low = _low[i]
                    last_swing_low_index = i

        else:
            for i in range(shl_len):

                # new structure completed on up side
                if _high[i] >= last_swing_high:
                    if trend == 1:
                        bos[last_swing_high_index] = 1
                        if use_bos:
                            use_structure = True
                    else:
                        choch[last_swing_high_index] = 1
                        trend = 1
                        if use_choch:
                            use_structure = True

                    bos_level[last_swing_high_index] = last_swing_high
                    structure_complete_date[last_swing_high_index] = _dates[i]

                    if use_structure:
                        ob[last_swing_low_index] = -1
                        ob_confirm_date[last_swing_low_index] = _dates[i]

                        ob_low_indices = np.concatenate(
                            (
                                ob_low_indices,
                                np.array([[last_swing_low_index, last_swing_low]]),
                            ),
                            axis=0,
                        )

                    if last_swing_low > last_ob_low:
                        last_ob_low = last_swing_low

                    last_swing_high = very_high_no
                    last_swing_high_index = very_low_no

                # upper order block mitigated
                if _high[i] >= last_ob_high:
                    idx_to_delete = 0
                    for row_idx, row in enumerate(ob_high_indices):
                        ob_high_index, swing_high = row
                        if last_ob_high == swing_high:
                            idx_to_delete = row_idx
                            ob_mitigation_date[int(ob_high_index)] = _dates[i]

                    mask = np.ones(ob_high_indices.shape[0], dtype=bool)
                    mask[idx_to_delete] = False
                    ob_high_indices = ob_high_indices[mask]
                    if np.any(ob_high_indices):
                        last_ob_high = ob_high_indices[:, 1].min()
                    else:
                        last_ob_high = very_high_no

                # new structure completed on down side
                if _low[i] <= last_swing_low:

                    if trend == -1:
                        bos[last_swing_low_index] = -1
                        if use_bos:
                            use_structure = True
                    else:
                        choch[last_swing_low_index] = -1
                        trend = -1
                        if use_choch:
                            use_structure = True

                    bos_level[last_swing_low_index] = last_swing_low
                    structure_complete_date[last_swing_low_index] = _dates[i]

                    if use_structure:
                        ob[last_swing_high_index] = 1
                        ob_confirm_date[last_swing_high_index] = _dates[i]

                        ob_high_indices = np.concatenate(
                            (
                                ob_high_indices,
                                np.array([[last_swing_high_index, last_swing_high]]),
                            ),
                            axis=0,
                        )

                    if last_swing_high < last_ob_high:
                        last_ob_high = last_swing_high

                    last_swing_low = 0
                    last_swing_low_index = 0

                # lower order block mitigated
                if _low[i] <= last_ob_low:

                    idx_to_delete = 0
                    for row_idx, row in enumerate(ob_low_indices):
                        ob_low_index, swing_low = row
                        if last_ob_low == swing_low:
                            idx_to_delete = row_idx
                            ob_mitigation_date[int(ob_low_index)] = _dates[i]

                    mask = np.ones(ob_low_indices.shape[0], dtype=bool)
                    mask[idx_to_delete] = False
                    ob_low_indices = ob_low_indices[mask]

                    if np.any(ob_low_indices):
                        last_ob_low = ob_low_indices[:, 1].max()
                    else:
                        last_ob_low = very_low_no

                if _high_low[i] == 1:
                    last_swing_high = _high[i]
                    last_swing_high_index = i

                if _high_low[i] == -1:
                    last_swing_low = _low[i]
                    last_swing_low_index = i

        # replace all the 0s with np.nan
        bos = np.where(bos != 0, bos, np.nan)
        choch = np.where(choch != 0, choch, np.nan)
        bos_level = np.where(bos_level != 0, bos_level, np.nan)
        ob = np.where(ob != 0, ob, np.nan)

        top = np.where(np.isnan(ob), np.nan, _high)
        bottom = np.where(np.isnan(ob), np.nan, _low)

        ob = pd.Series(ob, name="OB")
        top_series = pd.Series(top, name="Top")
        bottom_series = pd.Series(bottom, name="Bottom")
        ob_confirm_date = pd.Series(ob_confirm_date, name="ConfirmDate")
        ob_mitigation_date = pd.Series(ob_mitigation_date, name="MitigationDate")

        return pd.concat(
            [
                ob,
                top_series,
                bottom_series,
                ob_confirm_date,
                ob_mitigation_date,
            ],
            axis=1,
        )

    @classmethod
    def liquidity(
        cls, ohlc: DataFrame, swing_highs_lows: DataFrame, range_percent: float = 0.01
    ) -> Series:
        """
        Liquidity
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
        """

        swing_highs_lows = swing_highs_lows.copy()

        # subtract the highest high from the lowest low
        pip_range = (max(ohlc["high"]) - min(ohlc["low"])) * range_percent

        # go through all of the high level and if there are more than 1 within the pip range, then it is liquidity
        liquidity = np.zeros(len(ohlc), dtype=np.int32)
        liquidity_level = np.zeros(len(ohlc), dtype=np.float32)
        liquidity_end = np.zeros(len(ohlc), dtype=np.int32)
        liquidity_swept = np.zeros(len(ohlc), dtype=np.int32)

        for i in range(len(ohlc)):
            if swing_highs_lows["HighLow"][i] == 1:
                high_level = swing_highs_lows["Level"][i]
                range_low = high_level - pip_range
                range_high = high_level + pip_range
                temp_liquidity_level = [high_level]
                start = i
                end = i
                swept = 0
                for c in range(i + 1, len(ohlc)):
                    if (
                        swing_highs_lows["HighLow"][c] == 1
                        and range_low <= swing_highs_lows["Level"][c] <= range_high
                    ):
                        end = c
                        temp_liquidity_level.append(swing_highs_lows["Level"][c])
                        swing_highs_lows.loc[c, "HighLow"] = 0
                    if ohlc["high"].iloc[c] >= range_high:
                        swept = c
                        break
                if len(temp_liquidity_level) > 1:
                    average_high = sum(temp_liquidity_level) / len(temp_liquidity_level)
                    liquidity[i] = 1
                    liquidity_level[i] = average_high
                    liquidity_end[i] = end
                    liquidity_swept[i] = swept

        # now do the same for the lows
        for i in range(len(ohlc)):
            if swing_highs_lows["HighLow"][i] == -1:
                low_level = swing_highs_lows["Level"][i]
                range_low = low_level - pip_range
                range_high = low_level + pip_range
                temp_liquidity_level = [low_level]
                start = i
                end = i
                swept = 0
                for c in range(i + 1, len(ohlc)):
                    if (
                        swing_highs_lows["HighLow"][c] == -1
                        and range_low <= swing_highs_lows["Level"][c] <= range_high
                    ):
                        end = c
                        temp_liquidity_level.append(swing_highs_lows["Level"][c])
                        swing_highs_lows.loc[c, "HighLow"] = 0
                    if ohlc["low"].iloc[c] <= range_low:
                        swept = c
                        break
                if len(temp_liquidity_level) > 1:
                    average_low = sum(temp_liquidity_level) / len(temp_liquidity_level)
                    liquidity[i] = -1
                    liquidity_level[i] = average_low
                    liquidity_end[i] = end
                    liquidity_swept[i] = swept

        liquidity = np.where(liquidity != 0, liquidity, np.nan)
        liquidity_level = np.where(~np.isnan(liquidity), liquidity_level, np.nan)
        liquidity_end = np.where(~np.isnan(liquidity), liquidity_end, np.nan)
        liquidity_swept = np.where(~np.isnan(liquidity), liquidity_swept, np.nan)

        liquidity = pd.Series(liquidity, name="Liquidity")
        level = pd.Series(liquidity_level, name="Level")
        liquidity_end = pd.Series(liquidity_end, name="End")
        liquidity_swept = pd.Series(liquidity_swept, name="Swept")

        return pd.concat([liquidity, level, liquidity_end, liquidity_swept], axis=1)

    @classmethod
    def previous_high_low(cls, ohlc: DataFrame, time_frame: str = "1D") -> Series:
        """
        Previous High Low
        This method returns the previous high and low of the given time frame.

        parameters:
        time_frame: str - the time frame to get the previous high and low 15m, 1H, 4H, 1D, 1W, 1M

        returns:
        PreviousHigh = the previous high
        PreviousLow = the previous low
        """

        ohlc.index = pd.to_datetime(ohlc.index)

        previous_high = np.zeros(len(ohlc), dtype=np.float32)
        previous_low = np.zeros(len(ohlc), dtype=np.float32)
        broken_high = np.zeros(len(ohlc), dtype=np.int32)
        broken_low = np.zeros(len(ohlc), dtype=np.int32)

        resampled_ohlc = (
            ohlc.resample(time_frame)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        currently_broken_high = False
        currently_broken_low = False
        last_broken_time = None
        for i in range(len(ohlc)):
            resampled_previous_index = np.where(resampled_ohlc.index < ohlc.index[i])[0]
            if len(resampled_previous_index) <= 1:
                previous_high[i] = np.nan
                previous_low[i] = np.nan
                continue
            resampled_previous_index = resampled_previous_index[-2]

            if last_broken_time != resampled_previous_index:
                currently_broken_high = False
                currently_broken_low = False
                last_broken_time = resampled_previous_index

            previous_high[i] = resampled_ohlc["high"].iloc[resampled_previous_index]
            previous_low[i] = resampled_ohlc["low"].iloc[resampled_previous_index]
            currently_broken_high = (
                ohlc["high"].iloc[i] > previous_high[i] or currently_broken_high
            )
            currently_broken_low = (
                ohlc["low"].iloc[i] < previous_low[i] or currently_broken_low
            )
            broken_high[i] = 1 if currently_broken_high else 0
            broken_low[i] = 1 if currently_broken_low else 0

        previous_high = pd.Series(previous_high, name="PreviousHigh")
        previous_low = pd.Series(previous_low, name="PreviousLow")
        broken_high = pd.Series(broken_high, name="BrokenHigh")
        broken_low = pd.Series(broken_low, name="BrokenLow")

        return pd.concat([previous_high, previous_low, broken_high, broken_low], axis=1)

    @classmethod
    def sessions(
        cls,
        ohlc: DataFrame,
        session: str,
        start_time: str = "",
        end_time: str = "",
        time_zone: str = "UTC",
    ) -> Series:
        """
        Sessions
        This method returns wwhich candles are within the session specified

        parameters:
        session: str - the session you want to check (Sydney, Tokyo, London, New York, Asian kill zone, London open kill zone, New York kill zone, london close kill zone, Custom)
        start_time: str - the start time of the session in the format "HH:MM" only required for custom session.
        end_time: str - the end time of the session in the format "HH:MM" only required for custom session.
        time_zone: str - the time zone of the candles can be in the format "UTC+0" or "GMT+0"

        returns:
        Active = 1 if the candle is within the session, 0 if not
        High = the highest point of the session
        Low = the lowest point of the session
        """

        if session == "Custom" and (start_time == "" or end_time == ""):
            raise ValueError("Custom session requires a start and end time")

        default_sessions = {
            "Sydney": {
                "start": "21:00",
                "end": "06:00",
            },
            "Tokyo": {
                "start": "00:00",
                "end": "09:00",
            },
            "London": {
                "start": "07:00",
                "end": "16:00",
            },
            "New York": {
                "start": "13:00",
                "end": "22:00",
            },
            "Asian kill zone": {
                "start": "00:00",
                "end": "04:00",
            },
            "London open kill zone": {
                "start": "6:00",
                "end": "9:00",
            },
            "New York kill zone": {
                "start": "11:00",
                "end": "14:00",
            },
            "london close kill zone": {
                "start": "14:00",
                "end": "16:00",
            },
            "Custom": {
                "start": start_time,
                "end": end_time,
            },
        }

        ohlc.index = pd.to_datetime(ohlc.index)
        if time_zone != "UTC":
            time_zone = time_zone.replace("GMT", "Etc/GMT")
            time_zone = time_zone.replace("UTC", "Etc/GMT")
            ohlc.index = ohlc.index.tz_localize(time_zone).tz_convert("UTC")

        start_time = datetime.strptime(
            default_sessions[session]["start"], "%H:%M"
        ).strftime("%H:%M")
        start_time = datetime.strptime(start_time, "%H:%M")
        end_time = datetime.strptime(
            default_sessions[session]["end"], "%H:%M"
        ).strftime("%H:%M")
        end_time = datetime.strptime(end_time, "%H:%M")

        # if the candles are between the start and end time then it is an active session
        active = np.zeros(len(ohlc), dtype=np.int32)
        high = np.zeros(len(ohlc), dtype=np.float32)
        low = np.zeros(len(ohlc), dtype=np.float32)

        for i in range(len(ohlc)):
            current_time = ohlc.index[i].strftime("%H:%M")
            # convert current time to the second of the day
            current_time = datetime.strptime(current_time, "%H:%M")
            if (start_time < end_time and start_time <= current_time <= end_time) or (
                start_time >= end_time
                and (start_time <= current_time or current_time <= end_time)
            ):
                active[i] = 1
                high[i] = max(ohlc["high"].iloc[i], high[i - 1] if i > 0 else 0)
                low[i] = min(
                    ohlc["low"].iloc[i],
                    low[i - 1] if i > 0 and low[i - 1] != 0 else float("inf"),
                )

        active = pd.Series(active, name="Active")
        high = pd.Series(high, name="High")
        low = pd.Series(low, name="Low")

        return pd.concat([active, high, low], axis=1)

    @classmethod
    def retracements(cls, ohlc: DataFrame, swing_highs_lows: DataFrame) -> Series:
        """
        Retracement
        This method returns the percentage of a retracement from the swing high or low

        parameters:
        swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function

        returns:
        Direction = 1 if bullish retracement, -1 if bearish retracement
        CurrentRetracement% = the current retracement percentage from the swing high or low
        DeepestRetracement% = the deepest retracement percentage from the swing high or low
        """

        swing_highs_lows = swing_highs_lows.copy()

        direction = np.zeros(len(ohlc), dtype=np.int32)
        current_retracement = np.zeros(len(ohlc), dtype=np.float64)
        deepest_retracement = np.zeros(len(ohlc), dtype=np.float64)

        top = 0
        bottom = 0
        for i in range(len(ohlc)):
            if swing_highs_lows["HighLow"][i] == 1:
                direction[i] = 1
                top = swing_highs_lows["Level"][i]
                # deepest_retracement[i] = 0
            elif swing_highs_lows["HighLow"][i] == -1:
                direction[i] = -1
                bottom = swing_highs_lows["Level"][i]
                # deepest_retracement[i] = 0
            else:
                direction[i] = direction[i - 1] if i > 0 else 0

            if direction[i - 1] == 1:
                current_retracement[i] = round(
                    100 - (((ohlc["low"].iloc[i] - bottom) / (top - bottom)) * 100), 1
                )
                deepest_retracement[i] = max(
                    (
                        deepest_retracement[i - 1]
                        if i > 0 and direction[i - 1] == 1
                        else 0
                    ),
                    current_retracement[i],
                )
            if direction[i] == -1:
                current_retracement[i] = round(
                    100 - ((ohlc["high"].iloc[i] - top) / (bottom - top)) * 100, 1
                )
                deepest_retracement[i] = max(
                    (
                        deepest_retracement[i - 1]
                        if i > 0 and direction[i - 1] == -1
                        else 0
                    ),
                    current_retracement[i],
                )

        # shift the arrays by 1
        current_retracement = np.roll(current_retracement, 1)
        deepest_retracement = np.roll(deepest_retracement, 1)
        direction = np.roll(direction, 1)

        # remove the first 3 retracements as they get calculated incorrectly due to not enough data
        remove_first_count = 0
        for i in range(len(direction)):
            if i + 1 == len(direction):
                break
            if direction[i] != direction[i + 1]:
                remove_first_count += 1
            direction[i] = 0
            current_retracement[i] = 0
            deepest_retracement[i] = 0
            if remove_first_count == 3:
                direction[i + 1] = 0
                current_retracement[i + 1] = 0
                deepest_retracement[i + 1] = 0
                break

        direction = pd.Series(direction, name="Direction")
        current_retracement = pd.Series(current_retracement, name="CurrentRetracement%")
        deepest_retracement = pd.Series(deepest_retracement, name="DeepestRetracement%")

        return pd.concat([direction, current_retracement, deepest_retracement], axis=1)
