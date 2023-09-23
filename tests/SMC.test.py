import pandas as pd
import numpy as np
from smartmoneyconcepts.SMC import SMC

def test_FVG():
    df = pd.read_csv('bittrex_btc-usdt.csv')
    SMC.FVG(df)

test_FVG()