import pandas as pd
import plotly.graph_objects as go
import sys
import os
from binance.client import Client
from datetime import datetime
import numpy as np
import time

sys.path.append(os.path.abspath("../"))
from smartmoneyconcepts.smc import smc


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
    df.index = pd.to_datetime(df.index, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
    return df


df = import_data("BTCUSDT", "2021-01-01", "1d")
df = df.iloc[-1000:]

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


def add_FVG(fig, fvg_data):
    for i in range(len(fvg_data["FVG"])):
        if not np.isnan(fvg_data["FVG"][i]):
            x1 = int(
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


def add_swing_highs_lows(fig, swing_highs_lows_data):
    indexs = []
    level = []
    for i in range(len(swing_highs_lows_data)):
        if not np.isnan(swing_highs_lows_data["HighLow"][i]):
            indexs.append(i)
            level.append(swing_highs_lows_data["Level"][i])

    # plot these lines on a graph
    for i in range(len(indexs) - 1):
        fig.add_trace(
            go.Scatter(
                x=[df.index[indexs[i]], df.index[indexs[i + 1]]],
                y=[level[i], level[i + 1]],
                mode="lines",
                line=dict(
                    color=(
                        "green"
                        if swing_highs_lows_data["HighLow"][indexs[i]] == -1
                        else "red"
                    ),
                ),
            )
        )

    return fig


def add_bos_choch(fig, bos_choch_data):
    for i in range(len(bos_choch_data["BOS"])):
        if not np.isnan(bos_choch_data["BOS"][i]):
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[int(bos_choch_data["BrokenIndex"][i])]],
                    y=[bos_choch_data["Level"][i], bos_choch_data["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="orange",
                    ),
                )
            )
        if not np.isnan(bos_choch_data["CHOCH"][i]):
            fig.add_trace(
                go.Scatter(
                    x=[df.index[int(bos_choch_data["BrokenIndex"][i])], df.index[i]],
                    y=[bos_choch_data["Level"][i], bos_choch_data["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="blue",
                    ),
                )
            )

    return fig


def add_OB(fig, ob_data):
    def format_volume(volume):
        if volume >= 1e12:
            return f"{volume / 1e12:.3f}T"
        elif volume >= 1e9:
            return f"{volume / 1e9:.3f}B"
        elif volume >= 1e6:
            return f"{volume / 1e6:.3f}M"
        elif volume >= 1e3:
            return f"{volume / 1e3:.3f}k"
        else:
            return f"{volume:.2f}"

    for i in range(len(ob_data["OB"])):
        if ob_data["OB"][i] == 1:
            x1 = int(
                ob_data["MitigatedIndex"][i]
                if ob_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=ob_data["Bottom"][i],
                x1=df.index[x1],
                y1=ob_data["Top"][i],
                line=dict(color="Green"),
                fillcolor="Green",
                opacity=0.3,
                name="Bullish OB",
                legendgroup="bullish ob",
                showlegend=True,
            )

            if ob_data["MitigatedIndex"][i] > 0:
                x_center = df.index[int(i + (ob_data["MitigatedIndex"][i] - i) / 2)]
            else:
                x_center = df.index[int(i + (len(df) - i) / 2)]

            y_center = (ob_data["Bottom"][i] + ob_data["Top"][i]) / 2
            volume_text = format_volume(ob_data["OBVolume"][i])
            # Add annotation text
            annotation_text = f'{volume_text} ({ob_data["Percentage"][i]}%)'

            fig.add_annotation(
                x=x_center,
                y=y_center,
                xref="x",
                yref="y",
                align="center",
                text=annotation_text,
                font=dict(color="white", size=8),
                showarrow=False,
            )

    for i in range(len(ob_data["OB"])):
        if ob_data["OB"][i] == -1:
            x1 = int(
                ob_data["MitigatedIndex"][i]
                if ob_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=ob_data["Bottom"][i],
                x1=df.index[x1],
                y1=ob_data["Top"][i],
                line=dict(color="Red"),
                fillcolor="Red",
                opacity=0.3,
                name="Bearish OB",
                legendgroup="bearish ob",
                showlegend=True,
            )

            if ob_data["MitigatedIndex"][i] > 0:
                x_center = df.index[int(i + (ob_data["MitigatedIndex"][i] - i) / 2)]
            else:
                x_center = df.index[int(i + (len(df) - i) / 2)]

            y_center = (ob_data["Bottom"][i] + ob_data["Top"][i]) / 2
            volume_text = format_volume(ob_data["OBVolume"][i])
            # Add annotation text
            annotation_text = f'{volume_text} ({ob_data["Percentage"][i]}%)'

            fig.add_annotation(
                x=x_center,
                y=y_center,
                xref="x",
                yref="y",
                align="center",
                text=annotation_text,
                font=dict(color="white", size=8),
                showarrow=False,
            )
    return fig


def add_liquidity(fig, liquidity_data):
    # draw a line horizontally for each liquidity level
    for i in range(len(liquidity_data["Liquidity"])):
        if not np.isnan(liquidity_data["Liquidity"][i]):
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[int(liquidity_data["End"][i])]],
                    y=[liquidity_data["Level"][i], liquidity_data["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="orange",
                    ),
                )
            )
        if liquidity_data["Swept"][i] != 0 and not np.isnan(liquidity_data["Swept"][i]):
            # draw a red line between the end and the swept point
            fig.add_trace(
                go.Scatter(
                    x=[
                        df.index[int(liquidity_data["End"][i])],
                        df.index[int(liquidity_data["Swept"][i])],
                    ],
                    y=[
                        liquidity_data["Level"][i],
                        (
                            df["high"].iloc[int(liquidity_data["Swept"][i])]
                            if liquidity_data["Liquidity"][i] == 1
                            else df["low"].iloc[int(liquidity_data["Swept"][i])]
                        ),
                    ],
                    mode="lines",
                    line=dict(
                        color="red",
                    ),
                )
            )
    return fig

def add_previous_high_low(fig, previous_high_low_data):
    high = previous_high_low_data["PreviousHigh"]
    low = previous_high_low_data["PreviousLow"]
    # draw a line horizontally for each high where the highs are the same consecutively
    for i in range(len(high)-1):
        if high.iloc[i] == high.iloc[i+1]:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[i+1]],
                    y=[high.iloc[i], high.iloc[i+1]],
                    mode="lines",
                    line=dict(
                        color="lightblue",
                    ),
                )
            )
    # draw a line horizontally for each low where the lows are the same consecutively
    for i in range(len(low)-1):
        if low.iloc[i] == low.iloc[i+1]:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[i+1]],
                    y=[low.iloc[i], low.iloc[i+1]],
                    mode="lines",
                    line=dict(
                        color="lightblue",
                    ),
                )
            )

    return fig

def add_sessions(fig, sessions):
    for i in range(len(sessions["Active"])):
        if sessions["Active"][i] == 1:
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=sessions["Low"][i],
                x1=df.index[i+1],
                y1=sessions["High"][i],
                line=dict(
                    width=0,
                ),
                fillcolor="blue",
                opacity=0.5,
            )
    return fig

fvg_data = smc.fvg(df)
swing_highs_lows_data = smc.swing_highs_lows(df, swing_length=50)
bos_choch_data = smc.bos_choch(df, swing_highs_lows_data)
ob_data = smc.ob(df, swing_highs_lows_data)
liquidity_data = smc.liquidity(df, swing_highs_lows_data)
previous_high_low_data = smc.previous_high_low(df, time_frame="1W")
sessions = smc.sessions(df, session="London open kill zone")
fig = add_FVG(fig, fvg_data)
fig = add_swing_highs_lows(fig, swing_highs_lows_data)
fig = add_bos_choch(fig, bos_choch_data)
fig = add_OB(fig, ob_data)
fig = add_liquidity(fig, liquidity_data)
fig = add_previous_high_low(fig, previous_high_low_data)
fig = add_sessions(fig, sessions)

fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(showlegend=False)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.update_xaxes(visible=False, showticklabels=False)
fig.update_yaxes(visible=False, showticklabels=False)
fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
fig.update_layout(paper_bgcolor="rgba(94, 94, 134, 1)")
fig.update_layout(font=dict(color="white"))
fig.write_image("test_binance.png")
