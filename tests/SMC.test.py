import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath("../"))
from smartmoneyconcepts.smc import smc

df = pd.read_csv("bittrex_btc-usdt.csv")
df = df.iloc[-200:]
df = df.reset_index(drop=True)
fig = go.Figure(
    data=[
        go.Candlestick(
            x=df["date"],
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
        if fvg_data["FVG"][i] == 1:
            x1 = (
                fvg_data["MitigatedIndex"][i]
                if fvg_data["Mitigated"][i] == 1
                else len(df) - 1
            )
            fig.add_shape(
                # filled Rectangle
                type="rect",
                x0=df["date"][i],
                y0=fvg_data["Top"][i],
                x1=df["date"][x1],
                y1=fvg_data["Bottom"][i],
                line=dict(
                    width=0,
                ),
                fillcolor="yellow",
                opacity=0.5,
            )
    return fig


def add_highs_lows(fig):
    highs_lows = smc.highs_lows(df)

    # plot a dot on each high and low
    for i in range(len(highs_lows["Highs"])):
        if highs_lows["Highs"][i] == 1 or highs_lows["Lows"][i] == 1:
            high_or_low = "high" if highs_lows["Highs"][i] == 1 else "low"
            fig.add_trace(
                go.Scatter(
                    x=[df["date"][i]],
                    y=[df[high_or_low][i]],
                    mode="markers",
                    marker=dict(size=10, color="blue", opacity=0.5),
                )
            )
    return fig


def add_BOS_CHoCH(fig):
    bos_data = smc.bos_choch(df)

    # plot these lines on a graph
    for i in range(len(bos_data["BOS"])):
        if bos_data["BOS"][i] == 1 or bos_data["CHOCH"][i] == 1:
            fig.add_trace(
                go.Scatter(
                    x=[df["date"][i], df["date"][bos_data["BrokenIndex"][i]]],
                    y=[bos_data["Level"][i], bos_data["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="red" if bos_data["BOS"][i] == 1 else "green",
                    ),
                    opacity=0.5,
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
                if ob_data["Mitigated"][i] == 1
                else len(df) - 1
            )
            fig.add_shape(
                type="rect",
                x0=df["date"][i],
                y0=ob_data["Top"][i],
                x1=df["date"][x1],
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
        if liquidity_data["Liquidity"][i] == 1:
            fig.add_trace(
                go.Scatter(
                    x=[df["date"][i], df["date"][liquidity_data["End"][i]]],
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
                        df["date"][liquidity_data["End"][i]],
                        df["date"][liquidity_data["Swept"][i]],
                    ],
                    y=[
                        liquidity_data["Level"][i],
                        (
                            df["high"][liquidity_data["Swept"][i]]
                            if liquidity_data["BuySellSide"][i] == 2
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


fig = add_FVG(fig)
fig = add_highs_lows(fig)
fig = add_BOS_CHoCH(fig)
fig = add_OB(fig)
fig = add_liquidity(fig)
fig.show()
