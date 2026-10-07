"""Trends & Insights tab: trend refresh, incident timeline, semantic recurrence."""
import gradio as gr
import pandas as pd

from app.services import incident_timeline, recurring_issues

from ui.charts import _trend_figure
from ui.components import _resolution_metrics_html, _semantic_clusters_html, _timeline_kpi_html
from ui.llm import _llm_timeline_summary


def _refresh_trend(full_df: pd.DataFrame, granularity: str):
    return _trend_figure(full_df, granularity), _resolution_metrics_html(full_df)


def _refresh_incident_timeline(full_df: pd.DataFrame):
    """Recomputes the Incident Timeline table + KPI cards from the latest
    analysis. Deterministic and LLM-free, so it runs automatically at the
    end of every analysis, same as Recommendations. Returns the timeline
    dataframe too, cached as state, so the Generate Summary button reuses
    exactly these numbers instead of recomputing them."""
    try:
        timeline_df = incident_timeline.compute_incident_timeline(full_df)
        aggregate = incident_timeline.compute_timeline_aggregate(timeline_df)
        return timeline_df, _timeline_kpi_html(aggregate), aggregate
    except Exception:
        return pd.DataFrame(), _timeline_kpi_html({}), {}


async def _generate_timeline_summary(full_df: pd.DataFrame, cached_aggregate: dict):
    """Click handler for the Incident Timeline's "Generate summary"
    button - mirrors _generate_recommendations_writeup exactly: reuse the
    aggregate already computed when the analysis finished, send only that
    to the LLM, and fall back to a deterministic rendering of the same
    numbers if no LLM is available."""
    try:
        if full_df is None or len(full_df) == 0:
            return gr.update(value="Run an analysis on the Overview tab first, then generate the summary.")

        aggregate = cached_aggregate if (cached_aggregate or {}).get("available") else \
            incident_timeline.compute_timeline_aggregate(incident_timeline.compute_incident_timeline(full_df))

        if not aggregate.get("available"):
            return gr.update(value=incident_timeline.fallback_summary_markdown(aggregate))

        payload = incident_timeline.build_llm_payload(aggregate)
        text, used_llm = await _llm_timeline_summary(payload)
        if not used_llm:
            return gr.update(
                value=incident_timeline.fallback_summary_markdown(aggregate)
                + "\n\n_(LLM unavailable right now - showing the rule-based summary of the same metrics.)_"
            )
        return gr.update(value=text)
    except Exception as exc:
        return gr.update(value=f"Could not generate the incident timeline summary right now ({exc}).")


async def _detect_semantic_recurrence(index) -> str:
    result = await recurring_issues.detect_semantic_recurrence(index)
    return _semantic_clusters_html(result)
