import pandas as pd
import plotly.graph_objects as go
import sys
import os
from binance.client import Client
from datetime import datetime
from smartmoneyconcepts.smc import smc

sys.path.append(os.path.abspath("../"))


def import_data(symbol, start_str, timeframe):
    client = Client()
    start_str = str(start_str)
    end_str = f"{datetime.now()}"
    df = pd.DataFrame(
        client.get_historical_klines(
            symbol=symbol, interval=timeframe, start_str=start_str, end_str=end_str
        )
    ).astype(float)
    df = df.iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms")
    return df


df = import_data("BTCUSDT", "2021-01-01", "1d")

fig = go.Figure(
    data=[
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        )
    ]
)


def add_FVG(fig):
    fvg_data = smc.fvg(df)
    # plot a rectangle for each fvg
    for i in range(len(fvg_data["FVG"])):
        if fvg_data["FVG"][i] != 0:
            x1 = (
                fvg_data["MitigatedIndex"][i]
                if fvg_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                # filled Rectangle
                type="rect",
                x0=df.index[i],
                y0=fvg_data["Top"][i],
                x1=df.index[x1],
                y1=fvg_data["Bottom"][i],
                line=dict(
                    width=0,
                ),
                fillcolor="yellow",
                opacity=0.5,
            )
    return fig


def add_highs_lows(fig):
    highs_lows_data = smc.highs_lows(df)

    # remove from highs_lows_data
    indexs = []
    levels = []
    for i in range(len(highs_lows_data)):
        if highs_lows_data["HighsLows"][i] != 0:
            indexs.append(i)
            levels.append(highs_lows_data["Levels"][i])

    # plot these lines on a graph
    for i in range(len(indexs) - 1):
        fig.add_trace(
            go.Scatter(
                x=[df.index[indexs[i]], df.index[indexs[i + 1]]],
                y=[levels[i], levels[i + 1]],
                mode="lines",
                line=dict(
                    color=(
                        "green"
                        if highs_lows_data["HighsLows"][indexs[i]] == -1
                        else "red"
                    ),
                ),
            )
        )

    return fig


def add_OB(fig):
    ob_data = smc.ob(df)

    # plot the same way as FVG
    for i in range(len(ob_data["OB"])):
        if ob_data["OB"][i] == 1:
            x1 = (
                ob_data["MitigatedIndex"][i]
                if ob_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=ob_data["Top"][i],
                x1=df.index[x1],
                y1=ob_data["Bottom"][i],
                line=dict(
                    width=0,
                ),
                fillcolor="purple",
                opacity=0.5,
            )
    return fig


def add_liquidity(fig):
    liquidity_data = smc.liquidity(df)

    # draw a line horizontally for each liquidity level
    for i in range(len(liquidity_data["Liquidity"])):
        if liquidity_data["Liquidity"][i] != 0:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[liquidity_data["End"][i]]],
                    y=[liquidity_data["Level"][i], liquidity_data["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="orange",
                    ),
                )
            )
        if liquidity_data["Swept"][i] != 0:
            # draw a red line between the end and the swept point
            fig.add_trace(
                go.Scatter(
                    x=[
                        df.index[liquidity_data["End"][i]],
                        df.index[liquidity_data["Swept"][i]],
                    ],
                    y=[
                        liquidity_data["Level"][i],
                        (
                            df["high"][liquidity_data["Swept"][i]]
                            if liquidity_data["Liquidity"][i] == 1
                            else df["low"][liquidity_data["Swept"][i]]
                        ),
                    ],
                    mode="lines",
                    line=dict(
                        color="red",
                    ),
                )
            )
    return fig


# fig = add_FVG(fig)
# fig = add_highs_lows(fig)
fig = add_OB(fig)
# fig = add_liquidity(fig)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()
