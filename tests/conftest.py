import pytest
import pandas as pd
import os


@pytest.fixture
def df():
    test_instrument = "EURUSD"
    instrument_data = f"{test_instrument}_15M.csv"
    df = pd.read_csv(
        os.path.join("tests/test_data", test_instrument, instrument_data)
    )
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df

@pytest.fixture
def fvg_result_data():
    test_instrument = "EURUSD"
    return pd.read_csv(
        os.path.join("tests/test_data", test_instrument, "fvg_result_data.csv")
    )


@pytest.fixture
def swing_highs_lows_result_data():
    test_instrument = "EURUSD"
    return pd.read_csv(
        os.path.join(
            "tests/test_data", test_instrument, "swing_highs_lows_result_data.csv"
        )
    )


@pytest.fixture
def bos_choch_result_data():
    test_instrument = "EURUSD"
    return pd.read_csv(
        os.path.join("tests/test_data", test_instrument, "bos_choch_result_data.csv")
    )


@pytest.fixture
def liquidity_result_data():
    test_instrument = "EURUSD"
    return pd.read_csv(
        os.path.join("tests/test_data", test_instrument, "liquidity_result_data.csv")
    )


@pytest.fixture
def ob_result_data():
    test_instrument = "EURUSD"
    return pd.read_csv(
        os.path.join("tests/test_data", test_instrument, "ob_result_data.csv")
    )


@pytest.fixture
def previous_high_low_result_data():
    test_instrument = "EURUSD"
    return pd.read_csv(
        os.path.join(
            "tests/test_data", test_instrument, "previous_high_low_result_data_4h.csv"
        )
    )


@pytest.fixture
def sessions_result_data():
    test_instrument = "EURUSD"
    return pd.read_csv(
        os.path.join("tests/test_data", test_instrument, "sessions_result_data.csv")
    )


@pytest.fixture
def retracements_result_data():
    test_instrument = "EURUSD"
    return pd.read_csv(
        os.path.join("tests/test_data", test_instrument, "retracements_result_data.csv")
    )
