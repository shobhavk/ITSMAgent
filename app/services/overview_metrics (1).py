"""
Aggregation layer for the Overview tab's executive KPIs, health
indicators, attention list, volume trend, and executive-summary payload.

This module deliberately does NOT recompute anything trend_metrics.py or
chat_tools.py already compute correctly - it reshapes their output into
the specific dict/list shapes ui/gradio_app.py's Overview section
renders, and adds the traffic-light thresholding (good/warning/critical)
on top. Two consequences of that:
  - chat_tools.get_incident_summary(full_df) is the single source of
    truth for total/P1+P2/avg-resolution/avg-worklog/poor-worklog%, so
    the Overview tab and the chat agent can never disagree on those
    numbers.
  - trend_metrics.compute_time_series(full_df, "Daily") is the single
    source of truth for the volume trend, so the Overview tab's simple
    line chart and the Trends & Insights tab's dual-axis chart always
    show the same underlying counts.

Every function is wrapped in try/except and degrades to an "unknown"/
empty result rather than raising, consistent with the rest of this
codebase's "never crash the dashboard, report what's missing" approach.

Thresholds (HEALTH_THRESHOLDS below) are reasonable defaults, not a
standard - tune them to your organization's actual expectations the same
way trend_metrics.SLA_TARGET_HOURS is meant to be tuned.
"""
import logging

import pandas as pd

from app.services import chat_tools, trend_metrics

logger = logging.getLogger(__name__)

# Tune these to your organization's actual expectations - they're
# reasonable general-purpose defaults, not a standard.
HEALTH_THRESHOLDS = {
    "high_priority_pct": {"warning": 20.0, "critical": 40.0},
    "poor_worklog_pct": {"warning": 20.0, "critical": 40.0},
    "avg_resolution_hours": {"warning": 24.0, "critical": 48.0},
    "rejected_pct": {"warning": 5.0, "critical": 15.0},
}


def _status_from_threshold(value, thresholds: dict, higher_is_worse: bool = True) -> str:
    if value is None:
        return "unknown"
    if higher_is_worse:
        if value >= thresholds["critical"]:
            return "critical"
        if value >= thresholds["warning"]:
            return "warning"
        return "good"
    if value <= thresholds["critical"]:
        return "critical"
    if value <= thresholds["warning"]:
        return "warning"
    return "good"


def compute_overview_kpis(full_df: pd.DataFrame, summary_stats: dict | None = None) -> dict:
    """The five Overview KPI cards. Thin reshape of
    chat_tools.get_incident_summary(full_df) - see that function for the
    actual computation (it's the same number the chat agent uses)."""
    try:
        summary_stats = summary_stats or {}
        if full_df is None or len(full_df) == 0:
            total = int(summary_stats.get("valid_records", 0) or 0)
            return {
                "total_incidents": total,
                "high_priority_count": 0,
                "high_priority_pct": 0.0,
                "avg_resolution_hours": None,
                "avg_worklog_score": summary_stats.get("average_worklog_score"),
                "poor_worklog_pct": None,
                "poor_worklog_count": 0,
            }

        summary = chat_tools.get_incident_summary(full_df)
        if "error" in summary:
            logger.info("compute_overview_kpis: get_incident_summary reported %s", summary["error"])
            return {
                "total_incidents": int(len(full_df)),
                "high_priority_count": 0,
                "high_priority_pct": 0.0,
                "avg_resolution_hours": None,
                "avg_worklog_score": None,
                "poor_worklog_pct": None,
                "poor_worklog_count": 0,
            }

        total = summary["total_incidents"]
        high_priority_count = summary["p1_incidents"] + summary["p2_incidents"]
        high_priority_pct = round(100.0 * high_priority_count / total, 1) if total else 0.0
        poor_pct = summary.get("poor_worklog_percentage")
        poor_count = round(poor_pct / 100.0 * total) if poor_pct is not None and total else 0

        return {
            "total_incidents": total,
            "high_priority_count": high_priority_count,
            "high_priority_pct": high_priority_pct,
            "avg_resolution_hours": summary.get("average_resolution_time_hours"),
            "avg_worklog_score": summary.get("average_worklog_score"),
            "poor_worklog_pct": poor_pct,
            "poor_worklog_count": poor_count,
        }
    except Exception as exc:
        logger.warning("compute_overview_kpis failed: %s", exc)
        return {
            "total_incidents": 0, "high_priority_count": 0, "high_priority_pct": 0.0,
            "avg_resolution_hours": None, "avg_worklog_score": None,
            "poor_worklog_pct": None, "poor_worklog_count": 0,
        }


def compute_health_indicators(full_df: pd.DataFrame, kpis: dict) -> dict:
    """Four traffic-light indicators (good/warning/critical/unknown):
    Priority Health, Resolution Performance, Worklog Quality, Data
    Quality. Thresholds in HEALTH_THRESHOLDS above."""
    try:
        kpis = kpis or {}

        priority_status = _status_from_threshold(kpis.get("high_priority_pct"), HEALTH_THRESHOLDS["high_priority_pct"])
        priority_detail = (
            f'{kpis.get("high_priority_count", 0)} of {kpis.get("total_incidents", 0)} incidents are P1/P2 '
            f'({kpis.get("high_priority_pct", 0)}%).'
        )

        resolution_hours = kpis.get("avg_resolution_hours")
        resolution_status = _status_from_threshold(resolution_hours, HEALTH_THRESHOLDS["avg_resolution_hours"])
        resolution_detail = (
            f"Average resolution time is {resolution_hours}h." if resolution_hours is not None
            else "No resolution-time data in this batch (needs Opened/Created and Resolved/Closed dates)."
        )

        worklog_status = _status_from_threshold(kpis.get("poor_worklog_pct"), HEALTH_THRESHOLDS["poor_worklog_pct"])
        worklog_detail = (
            f'{kpis.get("poor_worklog_count", 0)} incident(s) ({kpis.get("poor_worklog_pct", 0)}%) have a poor worklog.'
            if kpis.get("poor_worklog_pct") is not None
            else "No worklog score data in this batch."
        )

        if full_df is None or len(full_df) == 0:
            data_status, data_detail = "unknown", "No analyzed tickets yet."
        else:
            missing_category = int((full_df.get("Category", pd.Series(dtype=str)).fillna("") == "").sum())
            missing_pct = round(100.0 * missing_category / len(full_df), 1) if len(full_df) else 0.0
            data_status = _status_from_threshold(missing_pct, HEALTH_THRESHOLDS["rejected_pct"])
            data_detail = f"{missing_category} incident(s) ({missing_pct}%) have no category assigned."

        return {
            "priority_health": {"status": priority_status, "detail": priority_detail},
            "resolution_performance": {"status": resolution_status, "detail": resolution_detail},
            "worklog_quality": {"status": worklog_status, "detail": worklog_detail},
            "data_quality": {"status": data_status, "detail": data_detail},
        }
    except Exception as exc:
        logger.warning("compute_health_indicators failed: %s", exc)
        unknown = {"status": "unknown", "detail": "Could not compute this indicator."}
        return {"priority_health": unknown, "resolution_performance": unknown, "worklog_quality": unknown, "data_quality": unknown}


_HEALTH_LABELS = {
    "priority_health": "Priority mix",
    "resolution_performance": "Resolution time",
    "worklog_quality": "Worklog quality",
    "data_quality": "Data quality",
}


def compute_attention_items(health: dict) -> list:
    """One item per health indicator currently at "warning" or "critical" -
    empty list means everything is healthy (the "all clear" banner)."""
    try:
        items = []
        for key, entry in (health or {}).items():
            status = entry.get("status")
            if status in ("warning", "critical"):
                items.append({
                    "severity": status,
                    "title": _HEALTH_LABELS.get(key, key),
                    "detail": entry.get("detail", ""),
                })
        # Critical first, then warning.
        items.sort(key=lambda it: 0 if it["severity"] == "critical" else 1)
        return items
    except Exception as exc:
        logger.warning("compute_attention_items failed: %s", exc)
        return []


def compute_volume_trend(full_df: pd.DataFrame) -> list:
    """Daily incident volume as [{"period": ..., "count": ...}, ...] -
    thin reshape of trend_metrics.compute_time_series so the Overview
    tab's simple line chart and the Trends & Insights tab's dual-axis
    chart always agree on the underlying counts."""
    try:
        series = trend_metrics.compute_time_series(full_df, "Daily")
        return [{"period": p["period"], "count": p["ticket_count"]} for p in series]
    except Exception as exc:
        logger.warning("compute_volume_trend failed: %s", exc)
        return []


def build_aggregated_payload(kpis: dict, health: dict, attention: list, volume_trend: list) -> dict:
    """Bundles the already-computed, already-small aggregates for the
    executive-summary LLM call - never includes ticket rows/descriptions/
    worklog text, only these pre-aggregated figures."""
    try:
        trend_direction = "flat"
        if volume_trend and len(volume_trend) >= 2:
            first_half = volume_trend[: len(volume_trend) // 2]
            second_half = volume_trend[len(volume_trend) // 2:]
            first_avg = sum(p["count"] for p in first_half) / len(first_half)
            second_avg = sum(p["count"] for p in second_half) / len(second_half)
            if second_avg > first_avg * 1.1:
                trend_direction = "increasing"
            elif second_avg < first_avg * 0.9:
                trend_direction = "decreasing"

        return {
            "kpis": kpis or {},
            "health_indicators": health or {},
            "attention_items": attention or [],
            "volume_trend_direction": trend_direction,
            "volume_trend_periods": len(volume_trend or []),
        }
    except Exception as exc:
        logger.warning("build_aggregated_payload failed: %s", exc)
        return {"kpis": kpis or {}, "health_indicators": health or {}, "attention_items": attention or []}


def fallback_summary_text(payload: dict) -> str:
    """Deterministic, rule-based executive summary from the same
    aggregated payload - no LLM. Used when no API key is configured or
    the LLM call fails, so the "Generate summary" button always returns
    something useful."""
    try:
        kpis = payload.get("kpis", {})
        attention = payload.get("attention_items", [])
        trend = payload.get("volume_trend_direction", "flat")

        total = kpis.get("total_incidents", 0)
        sentences = [f"This batch has {total} analyzed incident(s)."]

        if kpis.get("high_priority_count") is not None:
            sentences.append(
                f'{kpis["high_priority_count"]} ({kpis.get("high_priority_pct", 0)}%) are P1/P2 priority.'
            )
        if kpis.get("avg_resolution_hours") is not None:
            sentences.append(f'Average resolution time is {kpis["avg_resolution_hours"]} hours.')
        if kpis.get("poor_worklog_pct") is not None:
            sentences.append(f'{kpis.get("poor_worklog_count", 0)} incident(s) ({kpis["poor_worklog_pct"]}%) have a poor worklog.')

        trend_phrase = {
            "increasing": "Incident volume has been trending upward over this period.",
            "decreasing": "Incident volume has been trending downward over this period.",
            "flat": "Incident volume has stayed roughly steady over this period.",
        }.get(trend)
        if trend_phrase:
            sentences.append(trend_phrase)

        if attention:
            titles = ", ".join(item["title"] for item in attention)
            sentences.append(f"Areas needing attention: {titles}.")
        else:
            sentences.append("No metrics are currently outside a healthy range.")

        return " ".join(sentences)
    except Exception as exc:
        logger.warning("fallback_summary_text failed: %s", exc)
        return "Could not generate a summary from the current data."
