"""Overview tab: KPI/health refresh and the Executive Summary handler."""
import gradio as gr
import pandas as pd

from app.services import overview_metrics

from ui.constants import _ATTENTION_PLACEHOLDER, _HEALTH_PLACEHOLDER
from ui.charts import _overview_trend_figure
from ui.components import (
    _OVERVIEW_KPI_PLACEHOLDER,
    _attention_html,
    _health_indicators_html,
    _kpi_deltas,
    _overview_kpi_html,
    _volume_sparkline_html,
)
from ui.llm import _llm_executive_summary


def _refresh_overview(full_df: pd.DataFrame, summary_stats: dict):
    """Recomputes every Overview section (KPI cards, health indicators,
    attention list, volume trend) from the latest analysis."""
    try:
        kpis = overview_metrics.compute_overview_kpis(full_df, summary_stats or {})
        health = overview_metrics.compute_health_indicators(full_df, kpis)
        attention = overview_metrics.compute_attention_items(health)
        return (
            _overview_kpi_html(kpis, _kpi_deltas(full_df), _volume_sparkline_html(full_df)),
            _health_indicators_html(health),
            _attention_html(attention),
            _overview_trend_figure(full_df),
        )
    except Exception:
        return _OVERVIEW_KPI_PLACEHOLDER, _HEALTH_PLACEHOLDER, _ATTENTION_PLACEHOLDER, _overview_trend_figure(None)


async def _generate_executive_summary(full_df: pd.DataFrame, summary_stats: dict):
    """Click handler for "Generate Executive Summary": computes all KPIs/
    health/attention data via pandas (overview_metrics), sends only that
    small aggregated dict to the LLM, and renders the resulting text.
    Never sends per-ticket rows/descriptions/worklog text to the LLM."""
    try:
        if full_df is None or len(full_df) == 0:
            return gr.update(value="Run an analysis on the Overview tab first, then generate the summary.")

        kpis = overview_metrics.compute_overview_kpis(full_df, summary_stats or {})
        health = overview_metrics.compute_health_indicators(full_df, kpis)
        attention = overview_metrics.compute_attention_items(health)
        volume_trend = overview_metrics.compute_volume_trend(full_df)
        payload = overview_metrics.build_aggregated_payload(kpis, health, attention, volume_trend)

        summary_text, used_llm = await _llm_executive_summary(payload)
        note = "" if used_llm else "\n\n_(LLM unavailable right now - showing the rule-based summary of the same metrics.)_"
        return gr.update(value=f"{summary_text}{note}")
    except Exception as exc:
        return gr.update(value=f"Could not generate the executive summary right now ({exc}).")
