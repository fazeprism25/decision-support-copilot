"""Plotly rendering shared by KPI and Ask tabs. One series = one hue; no dual axes."""
import pandas as pd
import plotly.express as px

SERIES_1 = "#2a78d6"


def figure(chart: dict, table: pd.DataFrame, title: str = ""):
    t, x, y = chart["type"], chart.get("x"), chart.get("y")
    if t == "hist":
        fig = px.histogram(table, x=x, nbins=40, color_discrete_sequence=[SERIES_1])
        fig.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.6)")
    elif t == "line":
        fig = px.line(table, x=x, y=y, markers=True, color_discrete_sequence=[SERIES_1])
        fig.update_traces(line_width=2, marker_size=8)
    else:
        data = table.head(20)
        data = data.assign(**{x: data[x].astype(str)})
        fig = px.bar(data, x=x, y=y, color_discrete_sequence=[SERIES_1])
        fig.update_layout(bargap=0.25)
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10), height=340,
                      hovermode="x unified" if t == "line" else "closest")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig
