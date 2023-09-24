import pandas as pd
import numpy as np
from smartmoneyconcepts.smc import smc
import plotly.graph_objects as go

def test_FVG():
    df = pd.read_csv('bittrex_btc-usdt.csv')
    # use the last 200 candles
    df = df.iloc[-200:]
    # reset the index
    df = df.reset_index(drop=True)

    fvg_data = smc.fvg(df)

    fvg = fvg_data["FVG"]
    direction = fvg_data["Direction"]
    top = fvg_data["Top"]
    bottom = fvg_data["Bottom"]
    size = fvg_data["Size"]
    mitigated = fvg_data["Mitigated"]
    mitigated_index = fvg_data["MitigatedIndex"]

    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                    open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'])
                        ])
    
    # plot a rectangle for each fvg
    for i in range(len(fvg)):
        if fvg[i] == 1:
            if mitigated[i] == 1:
                fig.add_shape(
                    # unfilled Rectangle
                    type="rect",
                    x0=df['date'][i],
                    y0=top[i],
                    x1=df['date'][mitigated_index[i]],
                    y1=bottom[i],
                    line=dict(
                        color=("rgba(255,0,0,0.5)" if direction[i] == 1 else "rgba(0,255,0,0.5)"),
                        width=1,
                    ),
                )
            else:
                fig.add_shape(
                    # unfilled Rectangle
                    type="rect",
                    x0=df['date'][i],
                    y0=top[i],
                    x1=df['date'][len(df)-1],
                    y1=bottom[i],
                    line=dict(
                        color=("rgba(255,0,0,0.5)" if direction[i] == 1 else "rgba(0,255,0,0.5)"),
                        width=1,
                    ),
                )
    fig.show()

test_FVG()

def test_HighsLows():
    df = pd.read_csv('bittrex_btc-usdt.csv')
    # use the last 200 candles
    df = df.iloc[-200:]
    # reset the index
    df = df.reset_index(drop=True)

    highs_lows = smc.highs_lows(df)
    
    highs = highs_lows["Highs"]
    lows = highs_lows["Lows"]

    fig = go.Figure(data=[go.Candlestick(x=df['date'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'])
    ])

    # plot a dot on each high and low
    for i in range(len(highs)):
        if highs[i] == 1:
            fig.add_trace(go.Scatter(
                x=[df['date'][i]],
                y=[df['high'][i]],
                mode="markers",
                marker=dict(
                    size=10,
                    color="rgba(255,0,0,1)"
                )
            ))
        if lows[i] == 1:
            fig.add_trace(go.Scatter(
                x=[df['date'][i]],
                y=[df['low'][i]],
                mode="markers",
                marker=dict(
                    size=10,
                    color="rgba(0,255,0,1)"
                )
            ))
    fig.show()

test_HighsLows()

def test_BOS():
    df = pd.read_csv('bittrex_btc-usdt.csv')
    # use the last 200 candles
    df = df.iloc[-200:]
    # reset the index
    df = df.reset_index(drop=True)

    bos_data = smc.bos_choch(df)

    bos = bos_data["BOS"]
    choch = bos_data["CHOCH"]
    direction = bos_data["Direction"]
    level = bos_data["Level"]
    broken_index = bos_data["BrokenIndex"]

    # plot these lines on a graph
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'])
    ])

    # plot a line for each bos

    for i in range(len(bos)):
        if bos[i] == 1:
            fig.add_trace(go.Scatter(
                x=[df['date'][i], df['date'][broken_index[i]]],
                y=[level[i], level[i]],
                mode="lines",
                line=dict(
                    color="rgba(255,0,0,1)"
                )
            ))
        if choch[i] == 1:
            # do the same with a different color
            fig.add_trace(go.Scatter(
                x=[df['date'][i], df['date'][broken_index[i]]],
                y=[level[i], level[i]],
                mode="lines",
                line=dict(
                    color="rgba(0,255,0,1)"
                )
            ))

    fig.show()

test_BOS()

def test_OB():
    df = pd.read_csv('bittrex_btc-usdt.csv')
    # use the last 200 candles
    df = df.iloc[-200:]
    # reset the index
    df = df.reset_index(drop=True)

    ob_data = smc.ob(df)

    print(ob_data)

    ob = ob_data["OB"]
    direction = ob_data["Direction"]
    top = ob_data["Top"]
    bottom = ob_data["Bottom"]
    size = ob_data["Size"]
    mitigated = ob_data["Mitigated"]
    mitigated_index = ob_data["MitigatedIndex"]

    fig = go.Figure(data=[go.Candlestick(x=df['date'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'])
    ])

    # plot the same way as FVG
    for i in range(len(ob)):
        if ob[i] == 1:
            if mitigated[i] == 1:
                fig.add_shape(
                    # unfilled Rectangle
                    type="rect",
                    x0=df['date'][i],
                    y0=top[i],
                    x1=df['date'][mitigated_index[i]],
                    y1=bottom[i],
                    line=dict(
                        color=("rgba(255,0,0,0.5)" if direction[i] == 1 else "rgba(0,255,0,0.5)"),
                        width=1,
                    ),
                )
            else:
                fig.add_shape(
                    # unfilled Rectangle
                    type="rect",
                    x0=df['date'][i],
                    y0=top[i],
                    x1=df['date'][len(df)-1],
                    y1=bottom[i],
                    line=dict(
                        color=("rgba(255,0,0,0.5)" if direction[i] == 1 else "rgba(0,255,0,0.5)"),
                        width=1,
                    ),
                )

    fig.show()

test_OB()

        
