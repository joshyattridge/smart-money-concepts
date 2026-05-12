from smartmoneyconcepts.smc import smc
import time
import pandas as pd
import pytest
import os


def test_fvg(df, fvg_result_data):
    start_time = time.time()
    fvg_data = smc.fvg(df, join_consecutive=True)
    print("fvg test time: ", time.time() - start_time)
    pd.testing.assert_frame_equal(fvg_data, fvg_result_data, check_dtype=False)


def test_swing_highs_lows(df, swing_highs_lows_result_data):
    start_time = time.time()
    swing_highs_lows_data = smc.swing_highs_lows(df, swing_length=5)
    print("swing_highs_lows test time: ", time.time() - start_time)
    pd.testing.assert_frame_equal(
        swing_highs_lows_data, swing_highs_lows_result_data, check_dtype=False
    )


def test_bos_choch(df, bos_choch_result_data):
    start_time = time.time()
    swing_highs_lows_data = smc.swing_highs_lows(df, swing_length=5)
    bos_choch_data = smc.bos_choch(df, swing_highs_lows_data)
    print("bos_choch test time: ", time.time() - start_time)
    pd.testing.assert_frame_equal(
        bos_choch_data, bos_choch_result_data, check_dtype=False
    )


@pytest.mark.skipif(
    os.getenv("HEAD", None) != "refs/heads/master", reason="Takes too long"
)
def test_ob(df, ob_result_data):
    start_time = time.time()
    swing_highs_lows_data = smc.swing_highs_lows(df, swing_length=5)
    ob_data = smc.ob(df, swing_highs_lows_data)
    print("ob test time: ", time.time() - start_time)
    pd.testing.assert_frame_equal(ob_data, ob_result_data, check_dtype=False)


def test_liquidity(df, liquidity_result_data):
    start_time = time.time()
    swing_highs_lows_data = smc.swing_highs_lows(df, swing_length=5)
    liquidity_data = smc.liquidity(df, swing_highs_lows_data)
    print("liquidity test time: ", time.time() - start_time)
    pd.testing.assert_frame_equal(
        liquidity_data, liquidity_result_data, check_dtype=False
    )


def test_previous_high_low(df, previous_high_low_result_data):
    start_time = time.time()
    previous_high_low_data = smc.previous_high_low(df, time_frame="4h")
    print("previous_high_low test time: ", time.time() - start_time)
    pd.testing.assert_frame_equal(
        previous_high_low_data, previous_high_low_result_data, check_dtype=False
    )


def test_sessions(df, sessions_result_data):
    start_time = time.time()
    sessions = smc.sessions(df, session="London")
    print("sessions test time: ", time.time() - start_time)
    pd.testing.assert_frame_equal(sessions, sessions_result_data, check_dtype=False)


def test_retracements(df, retracements_result_data):
    start_time = time.time()
    swing_highs_lows_data = smc.swing_highs_lows(df, swing_length=5)
    retracements = smc.retracements(df, swing_highs_lows_data)
    print("retracements test time: ", time.time() - start_time)
    pd.testing.assert_frame_equal(
        retracements, retracements_result_data, check_dtype=False
    )
