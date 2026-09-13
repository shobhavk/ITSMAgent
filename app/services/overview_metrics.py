"""
Calculation/analytics helpers for the Overview (executive-summary) page.

Kept separate from gradio_app.py (the UI layer) on purpose - same pattern
already used by trend_metrics.py. Every function here is a pure function
over a pandas DataFrame (or the small dicts these functions return) and has
no knowledge of Gradio, HTML, or any other UI concern. This also means the
Overview page's numbers can be unit-tested independently of the dashboard.

Nothing in here re-derives a metric that already exists elsewhere in the
app: resolution time / SLA compliance are pulled from
trend_metrics.compute_resolution_metrics, and the volume trend re-uses
trend_metrics.compute_time_series, so there is a single source of truth for
each number no matter which tab shows it.
"""
from __future__ import annotations

import pandas as pd

from app.services import trend_metrics

# --- Thresholds -------------------------------------------------------
# Simple, named constants (not magic numbers scattered through the body)
# so they're easy to retune later without touching any UI code.
HIGH_PRIORITY_WARN_PCT = 15.0
HIGH_PRIORITY_CRITICAL_PCT = 30.0

POOR_WORKLOG_WARN_PCT = 15.0
POOR_WORKLOG_CRITICAL_PCT = 30.0

DATA_QUALITY_WARN_PCT = 3.0
DATA_QUALITY_CRITICAL_PCT = 10.0

SLA_WARN_PCT = 90.0
SLA_CRITICAL_PCT = 75.0

MTTR_WARN_HOURS = 24.0
MTTR_CRITICAL_HOURS = 48.0

# Matches the "Poor" cutoff already used by gradio_app._score_badge, so
# "poor worklog %" here means exactly what the 🔴 badge means elsewhere.
POOR_WORKLOG_SCORE_THRESHOLD = 50

_EMPTY_KPIS = {
    "total_incidents": 0,
    "high_priority_count": 0,
    "high_priority_pct": 0.0,
    "avg_resolution_hours": None,
    "avg_worklog_score": 0,
    "poor_worklog_count": 0,
    "poor_worklog_pct": 0.0,
    "rejected_records": 0,
    "total_records_seen": 0,
}


def _pct(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return round(numerator / denominator * 100, 1)


def compute_overview_kpis(full_df: pd.DataFrame, summary_stats: dict) -> dict:
    """The five headline KPIs for the executive Overview page: Total
    Incidents, P1/P2 Incidents, Average Resolution Time, Average Worklog
    Score, and Poor Worklog %."""
    try:
        summary_stats = summary_stats or {}
        total_incidents = int(summary_stats.get("valid_records", 0) or 0)

        if full_df is None or len(full_df) == 0:
            high_priority_count = 0
            poor_worklog_count = 0
        else:
            high_priority_count = (
                int(
                    full_df["Priority"].fillna("").astype(str).str.contains(
                        "critical|high|p1|p2", case=False, regex=True
                    ).sum()
                )
                if "Priority" in full_df.columns
                else 0
            )
            poor_worklog_count = (
                int((full_df["Worklog Score"] < POOR_WORKLOG_SCORE_THRESHOLD).sum())
                if "Worklog Score" in full_df.columns
                else 0
            )

        resolution = trend_metrics.compute_resolution_metrics(full_df)
        mttr = resolution.get("mttr", {"available": False})
        avg_resolution_hours = (
            round(float(mttr["value_hours"]), 1) if mttr.get("available") else None
        )

        return {
            "total_incidents": total_incidents,
            "high_priority_count": high_priority_count,
            "high_priority_pct": _pct(high_priority_count, total_incidents),
            "avg_resolution_hours": avg_resolution_hours,
            "avg_worklog_score": summary_stats.get("average_worklog_score", 0),
            "poor_worklog_count": poor_worklog_count,
            "poor_worklog_pct": _pct(poor_worklog_count, total_incidents),
            "rejected_records": int(summary_stats.get("rejected_records", 0) or 0),
            "total_records_seen": int(summary_stats.get("total_records", 0) or 0),
        }
    except Exception:
        # A missing/unexpected column should degrade to a clearly-empty
        # KPI set rather than break the Overview page.
        return dict(_EMPTY_KPIS)


def _status(value, warn: float, critical: float, higher_is_worse: bool = True) -> str:
    if value is None:
        return "unknown"
    if higher_is_worse:
        if value >= critical:
            return "critical"
        if value >= warn:
            return "warning"
        return "good"
    if value <= critical:
        return "critical"
    if value <= warn:
        return "warning"
    return "good"


def compute_health_indicators(full_df: pd.DataFrame, kpis: dict) -> dict:
    """Four simple traffic-light indicators management can scan in a
    second: Priority Health, Resolution Performance, Worklog Quality, and
    Data Quality. Each is just a threshold applied to a KPI already
    computed above (or, for Resolution Performance, to SLA compliance from
    trend_metrics when that's available) - no new heavy computation here."""
    try:
        priority_status = _status(
            kpis.get("high_priority_pct"), HIGH_PRIORITY_WARN_PCT, HIGH_PRIORITY_CRITICAL_PCT
        )

        resolution_metrics = trend_metrics.compute_resolution_metrics(full_df)
        sla = resolution_metrics.get("sla", {"available": False})
        if sla.get("available"):
            resolution_status = _status(
                sla["compliance_pct"], SLA_WARN_PCT, SLA_CRITICAL_PCT, higher_is_worse=False
            )
            resolution_detail = f'{sla["compliance_pct"]}% of resolved incidents met their SLA target.'
        elif kpis.get("avg_resolution_hours") is not None:
            resolution_status = _status(kpis["avg_resolution_hours"], MTTR_WARN_HOURS, MTTR_CRITICAL_HOURS)
            resolution_detail = f'Average resolution time is {kpis["avg_resolution_hours"]}h.'
        else:
            resolution_status = "unknown"
            resolution_detail = "Not enough Opened/Closed timestamp data to assess yet."

        worklog_status = _status(kpis.get("poor_worklog_pct"), POOR_WORKLOG_WARN_PCT, POOR_WORKLOG_CRITICAL_PCT)

        total_seen = kpis.get("total_records_seen", 0)
        rejected_pct = _pct(kpis.get("rejected_records", 0), total_seen)
        data_quality_status = _status(rejected_pct, DATA_QUALITY_WARN_PCT, DATA_QUALITY_CRITICAL_PCT)

        return {
            "priority_health": {
                "status": priority_status,
                "detail": f'{kpis.get("high_priority_pct", 0)}% of incidents are P1/P2.',
            },
            "resolution_performance": {
                "status": resolution_status,
                "detail": resolution_detail,
            },
            "worklog_quality": {
                "status": worklog_status,
                "detail": f'{kpis.get("poor_worklog_pct", 0)}% of worklogs are rated poor.',
            },
            "data_quality": {
                "status": data_quality_status,
                "detail": f'{rejected_pct}% of uploaded records were rejected during validation.',
            },
        }
    except Exception:
        unknown = {"status": "unknown", "detail": "Not enough data yet."}
        return {
            "priority_health": dict(unknown),
            "resolution_performance": dict(unknown),
            "worklog_quality": dict(unknown),
            "data_quality": dict(unknown),
        }


_ATTENTION_TITLES = [
    ("priority_health", "High number of P1/P2 incidents"),
    ("worklog_quality", "Poor worklog quality"),
    ("resolution_performance", "High average resolution time / low SLA compliance"),
    ("data_quality", "Data quality issues in uploaded records"),
]


def compute_attention_items(health: dict) -> list:
    """Turns any non-"good" health indicator into a management-facing
    "needs attention" line item. This is deliberately just a roll-up of
    the four health checks above - it does NOT duplicate the
    Recurring/Repetitive Issues panel (Trends & Insights) or the
    per-category breakdown (Categorization tab)."""
    try:
        items = []
        for key, title in _ATTENTION_TITLES:
            entry = health.get(key) or {}
            if entry.get("status") in ("warning", "critical"):
                items.append(
                    {"title": title, "severity": entry["status"], "detail": entry.get("detail", "")}
                )
        return items
    except Exception:
        return []


def compute_volume_trend(full_df: pd.DataFrame) -> list:
    """Daily incident-volume series for the Overview's simple trend line.
    Re-uses trend_metrics.compute_time_series (the same source the Trends
    & Insights tab uses) instead of re-deriving bucket counts here."""
    try:
        series = trend_metrics.compute_time_series(full_df, "Daily")
        return [{"period": p["period"], "count": p["ticket_count"]} for p in series]
    except Exception:
        return []


def build_aggregated_payload(kpis: dict, health: dict, attention: list, volume_trend: list) -> dict:
    """Bundles everything the executive summary needs into one small,
    JSON-serializable dict of AGGREGATED numbers only - no ticket rows,
    descriptions, or worklog text. This dict (and only this dict) is what
    gets sent to the LLM for the "Generate Executive Summary" button."""
    try:
        trend_direction = "flat"
        if len(volume_trend) >= 2:
            midpoint = len(volume_trend) // 2
            first_half = volume_trend[:midpoint]
            second_half = volume_trend[midpoint:]
            avg_first = sum(p["count"] for p in first_half) / max(1, len(first_half))
            avg_second = sum(p["count"] for p in second_half) / max(1, len(second_half))
            if avg_second > avg_first * 1.1:
                trend_direction = "rising"
            elif avg_second < avg_first * 0.9:
                trend_direction = "falling"

        return {
            "kpis": kpis,
            "health": {k: v.get("status") for k, v in (health or {}).items()},
            "attention_items": [{"title": a["title"], "severity": a["severity"]} for a in (attention or [])],
            "volume_trend_direction": trend_direction,
            "volume_days_covered": len(volume_trend),
        }
    except Exception:
        return {"kpis": kpis or {}, "health": {}, "attention_items": [], "volume_trend_direction": "flat", "volume_days_covered": 0}


def fallback_summary_text(payload: dict) -> str:
    """Deterministic, rule-based executive summary built purely from the
    aggregated payload above. Used when the LLM call is unavailable or
    fails, so the "Generate Executive Summary" button always produces a
    useful result instead of an error."""
    try:
        k = payload.get("kpis", {})
        lines = [
            f"{k.get('total_incidents', 0)} incidents were analyzed, "
            f"{k.get('high_priority_count', 0)} of which ({k.get('high_priority_pct', 0)}%) are P1/P2."
        ]
        if k.get("avg_resolution_hours") is not None:
            lines.append(f"Average resolution time is {k['avg_resolution_hours']} hours.")
        else:
            lines.append("Average resolution time could not be calculated from the available timestamps.")
        lines.append(
            f"Average worklog quality score is {k.get('avg_worklog_score', 0)}/100, with "
            f"{k.get('poor_worklog_pct', 0)}% of worklogs rated poor."
        )
        lines.append(f"Incident volume is currently {payload.get('volume_trend_direction', 'flat')}.")
        attention_items = payload.get("attention_items", [])
        if attention_items:
            titles = "; ".join(a["title"] for a in attention_items)
            lines.append(f"Attention needed: {titles}.")
        else:
            lines.append("No metrics are currently outside healthy thresholds.")
        return " ".join(lines)
    except Exception:
        return "Executive summary is not available - run an analysis first."
