"""HTML renderers: progress bar, KPI cards, bar lists, tables, health, attention, recommendations."""
from datetime import datetime
from html import escape as _html_escape

import pandas as pd

from app.services import chat_tools, trend_metrics

from ui.constants import (
    _ATTENTION_ICON,
    _ATTENTION_PLACEHOLDER,
    _DELTA_MIN_DATED,
    _DELTA_MIN_PER_HALF,
    _HEALTH_LABELS,
    _HEALTH_PLACEHOLDER,
    _HEALTH_STATUS_STYLE,
    _RECOMMENDATIONS_PLACEHOLDER,
    _REC_ATTENTION_STYLE,
)
from ui.helpers import _strip_html
from ui.charts import _category_color, _high_priority_count


def _agent_progress_html(stage: str = "Agent analyzing tickets", step: int | None = None, total: int | None = None) -> str:
    """Animated progress card. `stage` is the current stage label; `step`/`total`
    (optional) add a "Step N of M" tag so the bar says what is actually happening."""
    step_html = f'<span class="agent-progress-step">Step {step} of {total}</span>' if step and total else ""
    return f"""
<div class="agent-progress">
  <span class="agent-progress-icon">🤖</span>
  <div class="agent-progress-track"><div class="agent-progress-fill"></div></div>
  <span class="agent-progress-text">{stage}<span class="dots"><span>.</span><span>.</span><span>.</span></span></span>
  {step_html}
</div>
"""


AGENT_PROGRESS_HTML = _agent_progress_html()


def _bar_list_html(items: list, max_items: int = 6, color: str = "#3b82f6", multicolor: bool = False) -> str:
    """Renders a sorted list of (label, count) tuples as a horizontal
    bar-list panel, matching the "Incidents by Category" / "Top Affected
    Servers" style in the reference dashboard."""
    if not items:
        return '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">No data to show yet - run an analysis first.</p>'
    total = sum(c for _, c in items) or 1
    items = items[:max_items]
    max_count = max(c for _, c in items) or 1
    rows = []
    for i, (label, count) in enumerate(items):
        pct = max(4, round(count / max_count * 100))
        bar_color = _category_color(i) if multicolor else color
        share = round(count / total * 100)
        rows.append(
            '<div class="bar-row">'
            f'<span class="bar-dot" style="background:{bar_color};"></span>'
            f'<span class="bar-label" title="{label}">{label}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%; background:{bar_color};"></div></div>'
            f'<span class="bar-count">{count}</span>'
            f'<span class="bar-share">{share}%</span>'
            "</div>"
        )
    return '<div class="bar-list">' + "".join(rows) + "</div>"


def _recurring_issues_table_html(rows: list) -> str:
    """Renders detect_exact_recurrence()'s output - each row is a (host,
    category) combo meeting the recurrence threshold, with a real time
    dimension instead of just a count."""
    if not rows:
        return (
            '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">'
            "No recurring issues at the current threshold - run an analysis first, "
            "or this batch simply doesn't have repeat offenders yet.</p>"
        )
    body = "".join(
        "<tr>"
        f'<td>{r["host"] or "(no host)"}</td><td>{r["category"]}</td><td>{r["count"]}</td>'
        f'<td>{r["avg_interval_days"] if r["avg_interval_days"] is not None else "—"}</td>'
        f'<td>{r["first_seen"] or "—"} → {r["last_seen"] or "—"}</td>'
        "</tr>"
        for r in rows
    )
    return (
        '<table class="mini-table"><thead><tr>'
        "<th>Host / Server</th><th>Issue Type</th><th>Count</th>"
        "<th>Avg. days between</th><th>First → last seen</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _semantic_clusters_html(result: dict) -> str:
    """Renders detect_semantic_recurrence()'s output."""
    if not result.get("available"):
        note = result.get("note") or "Click \"Detect Similar Recurring Issues\" to run this."
        return f'<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">{note}</p>'
    clusters = result.get("clusters", [])
    if not clusters:
        return (
            '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">'
            "No semantically similar recurring clusters found at this threshold.</p>"
        )
    body = "".join(
        "<tr>"
        f'<td>{c["count"]}</td>'
        f'<td>{", ".join(c["categories"]) or "—"}</td>'
        f'<td>{", ".join(c["sample_ticket_ids"])}</td>'
        f'<td style="max-width:320px; white-space:normal;">{c["excerpt"]}</td>'
        "</tr>"
        for c in clusters
    )
    return (
        '<table class="mini-table"><thead><tr>'
        "<th>Count</th><th>Categories involved</th><th>Sample ticket IDs</th><th>Representative text</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _assignment_group_performance(full_df: pd.DataFrame, top_n: int = 6) -> list:
    """Per assignment group: ticket count, average worklog score, and the
    share of "well-documented" tickets (score >= 75). Stands in for the
    reference's Avg Resolution Time / SLA % columns, which need
    timestamp/SLA data the current pipeline doesn't produce."""
    if full_df is None or len(full_df) == 0 or "Assignment Group" not in full_df.columns:
        return []
    d = full_df[full_df["Assignment Group"].fillna("").astype(str).str.strip() != ""]
    if d.empty:
        return []
    rows = []
    for group, sub in d.groupby("Assignment Group"):
        count = len(sub)
        avg_score = round(sub["Worklog Score"].mean(), 1) if "Worklog Score" in sub.columns else 0
        pct_good = round((sub["Worklog Score"] >= 75).mean() * 100) if "Worklog Score" in sub.columns else 0
        rows.append({"group": group, "count": count, "avg_score": avg_score, "pct_good": pct_good})
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows[:top_n]


def _assignment_group_table_html(rows: list) -> str:
    if not rows:
        return '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">No assignment group data yet - run an analysis first.</p>'
    body = "".join(
        f'<tr><td>{r["group"]}</td><td>{r["count"]}</td><td>{r["avg_score"]}</td><td>{r["pct_good"]}%</td></tr>'
        for r in rows
    )
    return (
        '<table class="mini-table"><thead><tr><th>Assignment Group</th><th>Incidents</th>'
        f'<th>Avg Worklog Score</th><th>Well-Documented</th></tr></thead><tbody>{body}</tbody></table>'
    )


def _kpi_card(label: str, value, accent: str) -> str:
    return (
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"</div>"
    )


_KPI_PLACEHOLDER = (
    '<div class="kpi-grid">'
    + _kpi_card("Total Records Seen", "—", "#3b82f6")
    + _kpi_card("Valid Records Analyzed", "—", "#10b981")
    + _kpi_card("Rejected Records", "—", "#ef4444")
    + _kpi_card("Average Worklog Score", "—", "#f59e0b")
    + "</div>"
)


def _summary_kpi_html(stats: dict) -> str:
    """Renders the same summary_stats dict used by _summary_markdown as a
    row of KPI cards instead of a markdown table - same underlying data,
    a management-dashboard-style presentation."""
    return (
        '<div class="kpi-grid">'
        + _kpi_card("Total Records Seen", stats["total_records"], "#3b82f6")
        + _kpi_card("Valid Records Analyzed", stats["valid_records"], "#10b981")
        + _kpi_card("Rejected Records", stats["rejected_records"], "#ef4444")
        + _kpi_card("Average Worklog Score", f'{stats["average_worklog_score"]} / 100', "#f59e0b")
        + "</div>"
    )


def _kpi_card_v2(
    icon: str, label: str, value, accent: str, note: str = "",
    delta_html: str = "", spark_html: str = "", link_tab: str = "",
) -> str:
    """KPI card. The last three arguments are optional: a trend delta line, a
    small sparkline, and `link_tab` (a tab-label fragment such as
    "Categorization") that makes the whole card click through to that tab."""
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    link_cls, link_attrs = "", ""
    if link_tab:
        js = (
            "var t=[...document.querySelectorAll('#main-tabs button')]"
            f".find(function(b){{return b.textContent.indexOf('{link_tab}')!==-1}});"
            "if(t){t.click();}"
        )
        link_cls = " kpi-link"
        link_attrs = f' role="button" tabindex="0" title="Open {link_tab}" onclick="{js}"'
    return (
        f'<div class="kpi-card kpi-card-v2{link_cls}" style="--accent:{accent}"{link_attrs}>'
        f'<div class="kpi-icon" style="background:{accent}1a; color:{accent};">{icon}</div>'
        f'<div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{note_html}{delta_html}</div>'
        f"{spark_html}"
        "</div>"
    )


def _kpi_deltas(full_df: pd.DataFrame) -> dict:
    """{kpi_key: (change, unit, higher_is_worse)} for the later vs earlier half
    of the batch's date range, or {} when it can't be computed reliably."""
    try:
        if full_df is None or len(full_df) == 0:
            return {}
        start = trend_metrics.effective_open_resolve_times(full_df)["start"]
        dated = start.notna()
        if int(dated.sum()) < _DELTA_MIN_DATED:
            return {}
        lo, hi = start[dated].min(), start[dated].max()
        if (hi - lo) < pd.Timedelta(days=1):
            return {}
        mid = lo + (hi - lo) / 2
        earlier = full_df[dated & (start < mid)]
        later = full_df[dated & (start >= mid)]
        if len(earlier) < _DELTA_MIN_PER_HALF or len(later) < _DELTA_MIN_PER_HALF:
            return {}
        e = chat_tools.get_incident_summary(earlier)
        l = chat_tools.get_incident_summary(later)
        if "error" in e or "error" in l:
            return {}

        def pct(old, new):
            if old in (None, 0) or new is None:
                return None
            return round(100.0 * (new - old) / old, 1)

        def diff(old, new):
            if old is None or new is None:
                return None
            return round(new - old, 1)

        e_hp = e["p1_incidents"] + e["p2_incidents"]
        l_hp = l["p1_incidents"] + l["p2_incidents"]
        return {
            "total_incidents": (pct(e["total_incidents"], l["total_incidents"]), "%", True),
            "high_priority_count": (pct(e_hp, l_hp), "%", True),
            "avg_resolution_hours": (
                pct(e.get("average_resolution_time_hours"), l.get("average_resolution_time_hours")), "%", True,
            ),
            "avg_worklog_score": (
                diff(e.get("average_worklog_score"), l.get("average_worklog_score")), " pts", False,
            ),
            "poor_worklog_pct": (
                diff(e.get("poor_worklog_percentage"), l.get("poor_worklog_percentage")), " pts", True,
            ),
        }
    except Exception:
        return {}


def _delta_html(deltas: dict, key: str) -> str:
    entry = (deltas or {}).get(key)
    if not entry or entry[0] is None:
        return ""
    change, unit, higher_is_worse = entry
    sub = '<span class="kpi-delta-sub">vs earlier half</span>'
    if abs(change) < 0.5:
        return f'<div class="kpi-delta"><span class="kpi-delta-flat">▬ no change</span>{sub}</div>'
    going_up = change > 0
    cls = "kpi-delta-bad" if going_up == higher_is_worse else "kpi-delta-good"
    arrow = "▲" if going_up else "▼"
    return f'<div class="kpi-delta"><span class="{cls}">{arrow} {abs(change):g}{unit}</span>{sub}</div>'


def _volume_sparkline_html(full_df: pd.DataFrame, color: str = "#3b82f6") -> str:
    """Tiny inline-SVG daily-volume sparkline for the Total Incidents card.
    Empty string when there aren't enough dated days to draw a line."""
    try:
        series = trend_metrics.compute_time_series(full_df, "Daily")
        counts = [pt["ticket_count"] for pt in series]
        if len(counts) < 3:
            return ""
        w, h, pad = 84, 30, 3
        top = max(counts) or 1
        step = (w - 2 * pad) / (len(counts) - 1)
        points = " ".join(
            f"{pad + i * step:.1f},{h - pad - (c / top) * (h - 2 * pad):.1f}" for i, c in enumerate(counts)
        )
        return (
            f'<svg class="kpi-spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-hidden="true">'
            f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round" points="{points}"/></svg>'
        )
    except Exception:
        return ""


_KPI_PLACEHOLDER_V2 = (
    '<div class="kpi-grid kpi-grid-5">'
    + _kpi_card_v2("🎫", "Total Records Seen", "—", "#3b82f6")
    + _kpi_card_v2("✅", "Valid Records Analyzed", "—", "#10b981")
    + _kpi_card_v2("⚠️", "Rejected Records", "—", "#ef4444")
    + _kpi_card_v2("📝", "Average Worklog Score", "—", "#f59e0b")
    + _kpi_card_v2("🔴", "High Priority Tickets", "—", "#f97316")
    + "</div>"
)


def _summary_kpi_html_v2(stats: dict, full_df: pd.DataFrame) -> str:
    """Five-card KPI strip matching the reference dashboard: the four
    existing summary_stats values plus a High Priority Tickets count
    derived from the Priority column already present in full_df."""
    high_priority = _high_priority_count(full_df)
    return (
        '<div class="kpi-grid kpi-grid-5">'
        + _kpi_card_v2("🎫", "Total Records Seen", stats["total_records"], "#3b82f6")
        + _kpi_card_v2("✅", "Valid Records Analyzed", stats["valid_records"], "#10b981")
        + _kpi_card_v2("⚠️", "Rejected Records", stats["rejected_records"], "#ef4444")
        + _kpi_card_v2("📝", "Average Worklog Score", f'{stats["average_worklog_score"]} / 100', "#f59e0b")
        + _kpi_card_v2("🔴", "High Priority Tickets", high_priority, "#f97316")
        + "</div>"
    )


_OVERVIEW_KPI_PLACEHOLDER = (
    '<div class="kpi-grid kpi-grid-5">'
    + _kpi_card_v2("🎫", "Total Incidents", "—", "#3b82f6")
    + _kpi_card_v2("🔴", "P1/P2 Incidents", "—", "#ef4444")
    + _kpi_card_v2("⏱️", "Avg Resolution Time", "—", "#0ea5e9")
    + _kpi_card_v2("📝", "Avg Worklog Score", "—", "#f59e0b")
    + _kpi_card_v2("⚠️", "Poor Worklog %", "—", "#f97316")
    + "</div>"
)


def _overview_kpi_html(kpis: dict, deltas: dict | None = None, spark_html: str = "") -> str:
    """Renders the five executive KPI cards from overview_metrics'
    aggregated kpis dict. Pure presentation - all the numbers are already
    computed by overview_metrics.compute_overview_kpis. `deltas` / `spark_html`
    (optional) add the trend context from _kpi_deltas / _volume_sparkline_html;
    each card links to the tab where that number is explored further."""
    deltas = deltas or {}
    try:
        resolution_value = (
            f'{kpis["avg_resolution_hours"]}h' if kpis.get("avg_resolution_hours") is not None else "N/A"
        )
        return (
            '<div class="kpi-grid kpi-grid-5">'
            + _kpi_card_v2(
                "🎫", "Total Incidents", kpis.get("total_incidents", 0), "#3b82f6",
                delta_html=_delta_html(deltas, "total_incidents"), spark_html=spark_html,
                link_tab="Categorization",
            )
            + _kpi_card_v2(
                "🔴", "P1/P2 Incidents", kpis.get("high_priority_count", 0), "#ef4444",
                note=f'{kpis.get("high_priority_pct", 0)}% of total',
                delta_html=_delta_html(deltas, "high_priority_count"), link_tab="Categorization",
            )
            + _kpi_card_v2(
                "⏱️", "Avg Resolution Time", resolution_value, "#0ea5e9",
                delta_html=_delta_html(deltas, "avg_resolution_hours"), link_tab="Trends",
            )
            + _kpi_card_v2(
                "📝", "Avg Worklog Score", f'{kpis.get("avg_worklog_score", 0)} / 100', "#f59e0b",
                delta_html=_delta_html(deltas, "avg_worklog_score"), link_tab="Trends",
            )
            + _kpi_card_v2(
                "⚠️", "Poor Worklog %", f'{kpis.get("poor_worklog_pct", 0)}%', "#f97316",
                note=f'{kpis.get("poor_worklog_count", 0)} incident(s)',
                delta_html=_delta_html(deltas, "poor_worklog_pct"), link_tab="Recommendations",
            )
            + "</div>"
        )
    except Exception:
        return _OVERVIEW_KPI_PLACEHOLDER


def _health_indicators_html(health: dict) -> str:
    """Renders the four Incident Health traffic-light cards (Priority
    Health, Resolution Performance, Worklog Quality, Data Quality) from
    overview_metrics.compute_health_indicators's output."""
    if not health:
        return _HEALTH_PLACEHOLDER
    try:
        cards = []
        for key, label in _HEALTH_LABELS:
            entry = health.get(key, {"status": "unknown", "detail": ""})
            accent, status_label = _HEALTH_STATUS_STYLE.get(entry.get("status"), _HEALTH_STATUS_STYLE["unknown"])
            cards.append(
                f'<div class="health-card" style="--accent:{accent}">'
                f'<div class="health-card-top"><span class="health-dot"></span><span class="health-label">{label}</span></div>'
                f'<div class="health-status">{status_label}</div>'
                f'<div class="health-detail">{entry.get("detail", "")}</div>'
                "</div>"
            )
        return '<div class="health-grid">' + "".join(cards) + "</div>"
    except Exception:
        return _HEALTH_PLACEHOLDER


def _attention_html(items: list) -> str:
    """Renders the Attention Required panel from
    overview_metrics.compute_attention_items's output - one card per
    metric currently outside a healthy threshold, or a single "all clear"
    banner when nothing needs attention."""
    if items is None:
        return _ATTENTION_PLACEHOLDER
    try:
        if not items:
            return '<div class="attention-ok">✅ No urgent issues - all monitored metrics are within healthy ranges.</div>'
        accents = {"critical": "#ef4444", "warning": "#f59e0b"}
        rows = []
        for item in items:
            accent = accents.get(item.get("severity"), "#94a3b8")
            icon = _ATTENTION_ICON.get(item.get("severity"), "⚪")
            rows.append(
                f'<div class="attention-item" style="--accent:{accent}">'
                f'<span class="attention-icon">{icon}</span>'
                f'<div><div class="attention-title">{item.get("title", "")}</div>'
                f'<div class="attention-detail">{item.get("detail", "")}</div></div>'
                "</div>"
            )
        return '<div class="attention-list">' + "".join(rows) + "</div>"
    except Exception:
        return _ATTENTION_PLACEHOLDER


def _recommendations_html(result: dict) -> str:
    """Renders recommendations.generate_recommendations()'s output as one
    card per recommendation, each showing the fixed five-part structure:
    Observation / Evidence / Recommendation / Attention / Expected
    benefit. A batch that crosses no thresholds renders an explicit
    "nothing to flag" banner rather than padding the panel with generic
    advice."""
    try:
        if not result or not result.get("available"):
            note = (result or {}).get("note") or ""
            return (
                f'<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">{_strip_html(note)}</p>'
                if note else _RECOMMENDATIONS_PLACEHOLDER
            )

        recommendations_list = result.get("recommendations") or []
        if not recommendations_list:
            return f'<div class="rec-empty">✅ {_strip_html(result.get("note", ""))}</div>'

        chips = "".join(
            f'<span class="rec-chip">{_strip_html(area)}'
            f'<span class="rec-chip-count">{count}</span></span>'
            for area, count in (result.get("counts_by_area") or {}).items()
        )
        header = f'<div class="rec-summary-bar">{chips}</div>' if chips else ""

        cards = []
        for rec in recommendations_list:
            accent, tint, label = _REC_ATTENTION_STYLE.get(
                rec.get("attention"), _REC_ATTENTION_STYLE["Low"]
            )
            cards.append(
                f'<div class="rec-card" style="--accent:{accent}">'
                '<div class="rec-card-top">'
                f'<div class="rec-observation">{_strip_html(rec.get("observation", ""))}</div>'
                '<div class="rec-badges">'
                f'<span class="rec-badge" style="background:{tint}; color:{accent}">{label}</span>'
                f'<span class="rec-badge rec-badge-area">{_strip_html(rec.get("area", ""))}</span>'
                "</div></div>"
                f'<div class="rec-row"><div class="rec-row-label">Evidence</div>'
                f'<div class="rec-row-value">{_strip_html(rec.get("evidence", ""))}</div></div>'
                f'<div class="rec-row"><div class="rec-row-label">Recommendation</div>'
                f'<div class="rec-row-value">{_strip_html(rec.get("recommendation", ""))}</div></div>'
                f'<div class="rec-row"><div class="rec-row-label">Expected benefit</div>'
                f'<div class="rec-row-value">{_strip_html(rec.get("expected_benefit", ""))}</div></div>'
                "</div>"
            )

        note = result.get("note") or ""
        footer = (
            f'<p style="color:var(--dash-text-muted); font-size:0.78rem; margin:12px 0 0;">{_strip_html(note)}</p>'
            if note else ""
        )
        return f'{header}<div class="rec-list">{"".join(cards)}</div>{footer}'
    except Exception:
        return _RECOMMENDATIONS_PLACEHOLDER


_RESOLUTION_METRICS_PLACEHOLDER = (
    '<div class="kpi-grid">'
    + _kpi_card_v2("🔍", "MTTD (Detect)", "—", "#6366f1")
    + _kpi_card_v2("📨", "MTTA (Acknowledge)", "—", "#0ea5e9")
    + _kpi_card_v2("🛠️", "MTTR (Resolve)", "—", "#10b981")
    + _kpi_card_v2("🎯", "SLA Compliance", "—", "#f59e0b")
    + "</div>"
)


def _resolution_metrics_html(full_df: pd.DataFrame) -> str:
    """Four KPI cards: MTTD, MTTA, MTTR, SLA compliance. Any metric whose
    required timestamp column wasn't in the uploaded data shows "N/A" with
    an explanatory note instead of a fabricated number - see
    trend_metrics.compute_resolution_metrics for why MTTD/MTTA are often
    unavailable while MTTR/SLA (which only need Opened/Closed) are not."""
    metrics = trend_metrics.compute_resolution_metrics(full_df)

    def _duration_card(icon, label, accent, m):
        if not m["available"]:
            return _kpi_card_v2(icon, label, "N/A", "#94a3b8", note=m["note"])
        return _kpi_card_v2(icon, label, f'{m["value_hours"]}h', accent, note=f'Based on {m["sample_size"]} ticket(s)')

    sla = metrics["sla"]
    if not sla["available"]:
        sla_card = _kpi_card_v2("🎯", "SLA Compliance", "N/A", "#94a3b8", note=sla["note"])
    else:
        sla_card = _kpi_card_v2(
            "🎯", "SLA Compliance", f'{sla["compliance_pct"]}%', "#f59e0b",
            note=f'{sla["sample_size"]} resolved ticket(s), target by priority',
        )

    return (
        '<div class="kpi-grid">'
        + _duration_card("🔍", "MTTD (Detect)", "#6366f1", metrics["mttd"])
        + _duration_card("📨", "MTTA (Acknowledge)", "#0ea5e9", metrics["mtta"])
        + _duration_card("🛠️", "MTTR (Resolve)", "#10b981", metrics["mttr"])
        + sla_card
        + "</div>"
    )


def _timeline_kpi_html(aggregate: dict) -> str:
    if not aggregate or not aggregate.get("available"):
        return (
            '<div class="kpi-grid">'
            + _kpi_card_v2("📨", "Avg time to acknowledge", "—", "#0ea5e9")
            + _kpi_card_v2("🔁", "Reassignments detected", "—", "#8b5cf6")
            + _kpi_card_v2("📝", "External Info missing", "—", "#94a3b8")
            + "</div>"
        )
    ack = aggregate.get("avg_time_to_acknowledge_hours")
    quality = aggregate.get("external_info_quality_breakdown") or {}
    missing = quality.get("Missing", 0)
    return (
        '<div class="kpi-grid">'
        + _kpi_card_v2(
            "📨", "Avg time to acknowledge", f"{ack}h" if ack is not None else "N/A", "#0ea5e9",
            note=f"{aggregate.get('acknowledgment_data_available_for', 0)} ticket(s) with data" if ack is not None
            else "No acknowledgment timestamp in this data",
        )
        + _kpi_card_v2(
            "🔁", "Reassignments detected", aggregate.get("incidents_with_reassignment_detected", 0), "#8b5cf6",
            note="Detected in External Info text",
        )
        + _kpi_card_v2(
            "📝", "External Info missing", missing, "#94a3b8",
            note=f"of {aggregate.get('total_incidents', 0)} incident(s)",
        )
        + "</div>"
    )


def _loaded_strip_html(label: str, n_tickets: int) -> str:
    return (
        '<div class="loaded-strip-text">✅ Loaded <b>' + _html_escape(label or "input") + "</b>"
        f' <span class="loaded-meta">· {n_tickets:,} ticket{"" if n_tickets == 1 else "s"}'
        f' · {datetime.now().strftime("%H:%M")}</span></div>'
    )
