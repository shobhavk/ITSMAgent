"""Plotly figures: category/priority donuts, heatmap, trend charts."""
import pandas as pd
import plotly.graph_objects as go

from app.services import overview_metrics, trend_metrics

from ui.constants import _CATEGORY_PALETTE, _TREND_BAR_PALETTE


def _category_color(index: int) -> str:
    return _CATEGORY_PALETTE[index % len(_CATEGORY_PALETTE)]


def _category_chart_df(category_counts: dict) -> pd.DataFrame:
    if not category_counts:
        return pd.DataFrame({"Category": [], "Count": []})
    # Descending here since a donut chart reads clockwise from the top,
    # largest slice first.
    items = sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame(items, columns=["Category", "Count"])


def _donut_figure(
    counts: dict, title: str, color_fn=None, show_count_in_legend: bool = False,
    legend_style: str = "paren",
) -> go.Figure:
    """Generic donut chart from a {label: count} dict. color_fn, if given,
    maps a label to a hex color so semantically meaningful groups (e.g.
    priority tiers) get consistent colors instead of Plotly's defaults.

    show_count_in_legend appends the count to each legend entry, right
    next to that entry's color swatch - Plotly ties legend text to the
    slice `labels`, so this is the only hook available for that without
    also changing the on-slice text. Off by default so the only current
    caller that doesn't want it (_category_chart_figure) is unaffected;
    the live priority-chart call site below passes True explicitly.

    legend_style controls the count's format when show_count_in_legend is
    True: "paren" -> "Label (N)" (the original format), "dash_incidents"
    -> "Label - N incidents" (used for the priority donut).
    """
    chart_df = _category_chart_df(counts)
    if chart_df.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No data yet", showarrow=False, font=dict(size=14))],
            height=230,
            margin=dict(t=20, b=5, l=10, r=10),
        )
        return fig

    raw_labels = list(chart_df["Category"])
    if show_count_in_legend:
        if legend_style == "dash_incidents":
            legend_labels = [f"{label} - {count} incidents" for label, count in zip(raw_labels, chart_df["Count"])]
        else:
            legend_labels = [f"{label} ({count})" for label, count in zip(raw_labels, chart_df["Count"])]
    else:
        legend_labels = raw_labels
    colors = [color_fn(c) for c in raw_labels] if color_fn else [_category_color(i) for i in range(len(raw_labels))]
    marker = dict(colors=colors, line=dict(color="#ffffff", width=3))
    total = int(chart_df["Count"].sum())
    fig = go.Figure(
        data=[
            go.Pie(
                labels=legend_labels,
                values=chart_df["Count"],
                customdata=raw_labels,
                hole=0.66,
                sort=False,
                textinfo="percent",
                textposition="inside",
                textfont=dict(color="#ffffff", size=12, family="Inter, Segoe UI, sans-serif"),
                insidetextorientation="horizontal",
                marker=marker,
                hovertemplate="%{customdata}: %{value} tickets (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=230,
        margin=dict(t=8, b=8, l=8, r=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, Inter, Segoe UI, sans-serif", size=12, color="#14202b"),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0, font=dict(size=12)),
        annotations=[
            dict(text=f"<b style='font-size:22px'>{total}</b><br><span style='color:#64748b'>incidents</span>",
                 x=0.19, y=0.5, xref="paper", yref="paper", showarrow=False, align="center"),
        ],
    )
    fig.update_traces(domain=dict(x=[0.0, 0.38], y=[0.0, 1.0]))
    return fig


def _priority_sort_key(label: str) -> tuple:
    """Orders priority columns most-severe first, using the same wording rules
    as _priority_color; unknown labels go last, alphabetically."""
    l = (label or "").lower()
    if "critical" in l or "p1" in l:
        rank = 0
    elif "very high" in l:
        rank = 1
    elif "high" in l or "p2" in l:
        rank = 2
    elif "medium" in l or "p3" in l:
        rank = 3
    elif "low" in l or "p4" in l:
        rank = 4
    else:
        rank = 5
    return (rank, l)


def _category_priority_heatmap(full_df: pd.DataFrame) -> go.Figure:
    """Category x Priority count matrix for the Categorization tab: rows are
    categories (busiest first), columns are priorities (most severe first)."""
    if (
        full_df is None or len(full_df) == 0
        or "Category" not in full_df.columns or "Priority" not in full_df.columns
    ):
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No data yet", showarrow=False, font=dict(size=14))],
            height=260, margin=dict(t=30, b=10, l=10, r=10),
        )
        return fig

    category = full_df["Category"].fillna("").astype(str).str.strip().replace("", "Uncategorized")
    priority = full_df["Priority"].fillna("").astype(str).str.strip().replace("", "Unspecified")
    table = pd.crosstab(category, priority)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    table = table[sorted(table.columns, key=_priority_sort_key)]

    z = table.values
    text = [["" if v == 0 else str(int(v)) for v in row] for row in z]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z, x=list(table.columns), y=list(table.index),
                text=text, texttemplate="%{text}",
                colorscale=[[0.0, "#eef4f6"], [0.35, "#a3cbd1"], [0.7, "#0f6e7a"], [1.0, "#08434b"]],
                xgap=3, ygap=3, showscale=False,
                hovertemplate="%{y} · %{x}: %{z} tickets<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=max(260, 44 * len(table.index) + 90),
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(side="top", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    return fig


def _category_priority_stacked_bar(full_df: pd.DataFrame) -> go.Figure:
    """Category x Priority as horizontal stacked bars: one bar per category
    (busiest first), split by priority. Reads better than the heatmap when
    counts are small, and the priority colours match the rest of the UI."""
    if (
        full_df is None or len(full_df) == 0
        or "Category" not in full_df.columns or "Priority" not in full_df.columns
    ):
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No data yet", showarrow=False, font=dict(size=14))],
            height=260, margin=dict(t=30, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    category = full_df["Category"].fillna("").astype(str).str.strip().replace("", "Uncategorized")
    priority = full_df["Priority"].fillna("").astype(str).str.strip().replace("", "Unspecified")
    table = pd.crosstab(category, priority)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    table = table[sorted(table.columns, key=_priority_sort_key)]

    fig = go.Figure()
    for pri in table.columns:
        values = [int(v) for v in table[pri].tolist()]
        fig.add_bar(
            y=list(table.index), x=values, name=str(pri), orientation="h",
            marker=dict(color=_priority_color(str(pri))),
            text=[str(v) if v else "" for v in values], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#ffffff"),
            hovertemplate="%{y} · " + str(pri) + ": %{x} tickets<extra></extra>",
        )
    longest = max(len(str(c)) for c in table.index)
    fig.update_layout(
        barmode="stack", bargap=0.35,
        height=max(260, 46 * len(table.index) + 110),
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal"),
        yaxis=dict(autorange="reversed", showgrid=False, automargin=True),
        xaxis=dict(showgrid=True, zeroline=False, dtick=1 if int(table.sum(axis=1).max()) <= 10 else None),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _refresh_category_heatmap(full_df: pd.DataFrame):
    # Name kept so the existing .then() wiring is unchanged; the Categorization
    # tab now shows stacked bars instead of the old heatmap.
    return _category_priority_stacked_bar(full_df)


def _category_chart_figure(category_counts: dict) -> go.Figure:
    """Donut chart of ticket volume by category. Kept as a thin wrapper
    around _donut_figure so any existing caller keeps working unchanged."""
    return _donut_figure(category_counts, "Tickets by Category")


def _priority_color(label: str) -> str:
    """Maps a priority label to a fixed color regardless of exact wording
    ("P1 - Critical", "Very High", etc.) so the priority donut reads
    consistently: red = critical, orange = very high, amber = high,
    yellow = medium, green = low. "very high" is checked before the plain
    "high" substring match, since "high" also occurs inside it."""
    l = (label or "").lower()
    if "critical" in l or "p1" in l:
        return "#b3261e"
    if "very high" in l:
        return "#c05621"
    if "high" in l or "p2" in l:
        return "#b7791f"
    if "medium" in l or "p3" in l:
        return "#2b7a9b"
    if "low" in l or "p4" in l:
        return "#2e7d5b"
    return "#94a3b8"


def _priority_counts(full_df: pd.DataFrame) -> dict:
    if full_df is None or len(full_df) == 0 or "Priority" not in full_df.columns:
        return {}
    series = full_df["Priority"].fillna("").astype(str).str.strip()
    series = series.replace("", "Unspecified")
    return series.value_counts().to_dict()


def _high_priority_count(full_df: pd.DataFrame) -> int:
    if full_df is None or len(full_df) == 0 or "Priority" not in full_df.columns:
        return 0
    mask = full_df["Priority"].fillna("").astype(str).str.contains(
        "critical|high|p1|p2", case=False, regex=True
    )
    return int(mask.sum())


def _overview_trend_figure(full_df: pd.DataFrame) -> go.Figure:
    """Simple single-line chart of daily incident volume for the Overview
    page - deliberately simpler than the dual-axis volume/quality chart on
    the Trends & Insights tab, and built from the same underlying series
    (overview_metrics.compute_volume_trend -> trend_metrics.compute_time_series)
    so the two tabs never disagree on ticket counts."""
    try:
        series = overview_metrics.compute_volume_trend(full_df)
        fig = go.Figure()
        if not series:
            fig.update_layout(
                annotations=[dict(
                    text="No Opened-date data yet - run an analysis first.",
                    showarrow=False, font=dict(size=13),
                )],
                height=260,
                margin=dict(t=20, b=10, l=10, r=10),
            )
            return fig
        periods = [p["period"] for p in series]
        counts = [p["count"] for p in series]
        fig.add_trace(go.Scatter(
            x=periods, y=counts, mode="lines+markers", name="Incidents",
            line=dict(color="#0f6e7a", width=2), marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.08)",
        ))
        fig.update_layout(
            height=260,
            margin=dict(t=20, b=10, l=10, r=10),
            xaxis=dict(title=""),
            yaxis=dict(title="Incidents", rangemode="tozero"),
            showlegend=False,
        )
        return fig
    except Exception:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="Could not render the volume trend.", showarrow=False, font=dict(size=13))],
            height=260,
            margin=dict(t=20, b=10, l=10, r=10),
        )
        return fig


def _trend_figure(full_df: pd.DataFrame, granularity: str) -> go.Figure:
    """Ticket volume as a colorful bar chart, bucketed by day/week/month
    on Opened At, with the count labeled on top of each bar and the
    overall total shown in the title. (Previously a dual-axis chart that
    also plotted average worklog score as a line - removed per request;
    that number is still available via the KPI cards elsewhere.)"""
    series = trend_metrics.compute_time_series(full_df, granularity)
    fig = go.Figure()
    if not series:
        fig.update_layout(
            annotations=[dict(
                text="No Opened-date data yet - run an analysis first (dates come from the "
                     "Opened/Created column in your upload).",
                showarrow=False, font=dict(size=13),
            )],
            height=340,
            margin=dict(t=30, b=10, l=10, r=10),
        )
        return fig

    periods = [p["period"] for p in series]
    counts = [p["ticket_count"] for p in series]
    total = sum(counts)
    colors = [_TREND_BAR_PALETTE[i % len(_TREND_BAR_PALETTE)] for i in range(len(periods))]

    fig.add_trace(go.Bar(
        x=periods, y=counts, name="Ticket volume",
        marker_color=colors,
        text=counts, textposition="outside",
    ))
    fig.update_layout(
        title=f"Ticket volume - {granularity}  ·  Total incidents: {total}",
        height=340,
        margin=dict(t=50, b=10, l=10, r=10),
        xaxis=dict(title=""),
        yaxis=dict(title="Tickets", rangemode="tozero"),
        showlegend=False,
        bargap=0.25,
    )
    return fig
