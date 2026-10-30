"""Plotly rendering shared by KPI and Ask tabs. One series = one hue; no dual axes. Dark, transparent background."""
import pandas as pd
import plotly.express as px

from ui.style import GREEN, INK_MUTED, LINE

SERIES_1 = GREEN


def dark(fig, height: int = 340):
    """Shared dark layout: transparent canvas, muted ticks, faint horizontal grid only."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                      font=dict(color=INK_MUTED, size=12), margin=dict(l=10, r=10, t=40, b=10),
                      title_font=dict(size=13, color="#d6d3d1"),
                      hoverlabel=dict(bgcolor="#24221f", bordercolor=LINE, font_color="#e7e5e4"))
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    return fig


def figure(chart: dict, table: pd.DataFrame, title: str = ""):
    t, x, y = chart["type"], chart.get("x"), chart.get("y")
    if t == "hist":
        fig = px.histogram(table, x=x, nbins=40, color_discrete_sequence=[SERIES_1])
        fig.update_traces(marker_line_width=1, marker_line_color="#141413")
    elif t == "line":
        fig = px.line(table, x=x, y=y, markers=True, color_discrete_sequence=[SERIES_1])
        fig.update_traces(line_width=2, marker_size=7)
    else:
        data = table.head(20)
        data = data.assign(**{x: data[x].astype(str)})
        fig = px.bar(data, x=x, y=y, color_discrete_sequence=[SERIES_1])
        fig.update_layout(bargap=0.3)
    fig.update_layout(title=title, hovermode="x unified" if t == "line" else "closest")
    return dark(fig)
