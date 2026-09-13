"""
Gradio dashboard for the ITSM Quality Analysis Agent.

Runs in-process (mounted into the FastAPI app in main.py) so it calls the
pipeline directly rather than round-tripping through HTTP - simpler, faster,
and avoids needing an API key inside the browser session. The REST API
(/api/v1/...) remains available separately for machine-to-machine/automation
use cases, secured with its own API key as usual.
"""
import io
import json
import os
import re
from datetime import datetime

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from app.services import overview_metrics, rag, trend_metrics
from app.services.pipeline import run_pipeline_from_bytes, run_pipeline_from_text
from app.services.persistence import compute_file_hash, get_cached_result, save_result

CUSTOM_CSS = """
:root {
    --dash-bg: #f5f6fa;
    --dash-border: #e5e9f0;
    --dash-text-muted: #64748b;
    --dash-text: #0f172a;
    --dash-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}

.gradio-container {
    max-width: 1680px !important; margin: auto; padding: 0 !important;
    background: var(--dash-bg) !important; font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}
footer {display: none !important;}

/* App shell: dark nav rail on the left, everything else scrolls in the
   main column on the right - matches the reference management dashboard. */
#app-shell {gap: 0 !important; align-items: stretch !important;}
#sidebar-col {
    background: #0f1f33 !important; padding: 4px 16px !important; min-height: 100vh;
    border-radius: 0 !important;
}
#main-col {padding: 20px 28px 32px !important;}

.side-brand {display: flex; align-items: center; gap: 10px; padding: 0 6px 8px; margin-bottom: 4px; border-bottom: 1px solid rgba(255,255,255,0.08);}
.side-brand-icon {
    width: 34px; height: 34px; border-radius: 9px; background: #2563eb;
    display: flex; align-items: center; justify-content: center; font-size: 1.05rem; flex-shrink: 0;
}
.side-brand-title {color: #fff; font-weight: 700; font-size: 0.92rem; line-height: 1.2;}
.side-brand-sub {color: #8291a8; font-size: 0.72rem; line-height: 1.2;}

.nav-item {
    display: flex; align-items: center; gap: 10px; padding: 9px 12px; border-radius: 9px;
    color: #aab6c7; font-size: 0.85rem; font-weight: 500; margin-bottom: 2px;
}
.nav-item .nav-icon {font-size: 0.95rem; width: 18px; text-align: center;}
.nav-item.active {background: #1d5fe0; color: #fff; font-weight: 600;}

/* Nav items are now real Gradio buttons (so they can switch tabs), styled
   to look like the plain divs they replaced instead of default buttons. */
#sidebar-col {gap: 2px !important;}
#nav-buttons {gap: 2px !important;}
button.nav-item, button.nav-item:active, button.nav-item:focus {
    all: unset; box-sizing: border-box; cursor: pointer;
    display: flex; align-items: center; gap: 10px; width: 100%;
    padding: 9px 12px; border-radius: 9px;
    color: #aab6c7; font-size: 0.85rem; font-weight: 500; margin-bottom: 2px;
}
button.nav-item:hover {background: rgba(255, 255, 255, 0.08); color: #fff;}
button.nav-item.active {background: #1d5fe0 !important; color: #fff !important; font-weight: 600 !important;}

/* The sidebar now drives navigation, so hide Gradio's own tab strip -
   otherwise there would be two competing sets of tab controls. */
#main-tabs > .tab-nav {display: none !important;}
/* Extra safety net: if this Gradio version nests the tab strip
   differently than the selector above expects (so it stays visible),
   this trims the default top margin/padding it ships with so the strip
   sits closer to the title above it instead of floating lower down. */
#main-tabs .tab-nav {margin-top: 0 !important; padding-top: 0 !important;}
#main-tabs .tabitem, #main-tabs {border: none !important; padding: 0 !important; background: transparent !important;}

/* Top bar - plain title/subtitle on the left, a status badge on the
   right, replacing the old solid-color banner to match the reference. */
#topbar {align-items: center !important; margin-bottom: 6px; gap: 14px !important;}
#topbar h1 {margin: 0; font-size: 1.3rem; font-weight: 700; color: var(--dash-text); letter-spacing: -0.01em;}
#topbar p {margin: 3px 0 0; font-size: 0.85rem; color: var(--dash-text-muted);}
.severity-note {font-size: 0.78rem; color: var(--dash-text-muted); margin: 0 0 8px 2px; font-style: italic;}
.severity-note p {margin: 0;}

/* Section headings used above each dash-card block on the dashboard. */
.section-heading {
    font-size: 0.95rem !important; font-weight: 700 !important; color: var(--dash-text) !important;
    margin: 0 0 14px 0 !important; padding-bottom: 10px !important; border-bottom: 1px solid var(--dash-border) !important;
    letter-spacing: -0.01em;
}

#cache-notice {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
    padding: 8px 14px; font-size: 0.85rem; color: #166534; margin-bottom: 14px;
}

/* Card wrapper used around every major section - gives the dribbble-style
   raised-panel look instead of controls floating on the bare page. */
.dash-card {
    background: #ffffff !important; border: 1px solid var(--dash-border) !important;
    border-radius: 16px !important; padding: 18px 20px !important; box-shadow: var(--dash-shadow);
}

/* KPI strip */
#metrics-row {margin-bottom: 18px; gap: 14px !important;}
.kpi-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;}
.kpi-grid-5 {grid-template-columns: repeat(5, 1fr);}
.kpi-card {
    background: #ffffff; border: 1px solid var(--dash-border); border-radius: 14px;
    padding: 16px 20px; box-shadow: var(--dash-shadow); border-left: 4px solid var(--accent, #3b82f6);
    transition: box-shadow 0.15s ease;
}
.kpi-card:hover {box-shadow: 0 4px 10px rgba(15, 23, 42, 0.09);}
.kpi-label {font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--dash-text-muted); margin-bottom: 8px;}
.kpi-value {font-size: 1.65rem; font-weight: 700; color: var(--dash-text); line-height: 1; font-variant-numeric: tabular-nums;}

/* v2 KPI cards - icon chip + label/value, border-left removed in favor
   of a plain card since the icon already carries the accent color. */
.kpi-card-v2 {display: flex; align-items: center; gap: 12px; border-left: 1px solid var(--dash-border);}
.kpi-icon {width: 40px; height: 40px; border-radius: 11px; display: flex; align-items: center; justify-content: center; font-size: 1.15rem; flex-shrink: 0;}
.kpi-note {font-size: 0.68rem; color: var(--dash-text-muted); margin-top: 4px; line-height: 1.3;}

/* Horizontal bar-list panels (category / host breakdowns). */
.bar-list {display: flex; flex-direction: column; gap: 12px;}
.bar-row {display: flex; align-items: center; gap: 10px;}
.bar-label {
    flex: 0 0 120px; font-size: 0.82rem; color: var(--dash-text); font-weight: 500;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.bar-track {flex: 1 1 auto; height: 10px; border-radius: 999px; background: #eef1f5; overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.bar-count {flex: 0 0 40px; font-size: 0.82rem; color: var(--dash-text-muted); text-align: right; font-variant-numeric: tabular-nums;}

/* Mini data tables (repeated issues / assignment group performance). */
.mini-table {width: 100%; border-collapse: collapse; font-size: 0.82rem;}
.mini-table th {
    text-align: left; padding: 6px 8px; font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.03em; color: var(--dash-text-muted); border-bottom: 1px solid var(--dash-border);
}
.mini-table td {padding: 7px 8px; border-bottom: 1px solid #f1f4f8; color: var(--dash-text);}
.mini-table tbody tr:last-child td {border-bottom: none;}

#panel-row-1 {gap: 14px !important; margin-bottom: 14px;}
#panel-row-2 {gap: 14px !important; margin-bottom: 20px;}

/* Full-width input bar - a single horizontal card housing the upload,
   paste, analyze, and download controls with consistent alignment. */
#input-row {gap: 20px !important; margin-bottom: 18px; align-items: end !important; padding: 12px 20px !important;}
#input-row label {font-weight: 600; font-size: 0.82rem; color: var(--dash-text);}
#input-row .gr-button.primary, #input-row button.primary {
    border-radius: 10px !important; font-weight: 600 !important; box-shadow: 0 1px 2px rgba(15,23,42,.18);
    height: 36px !important;
}
#action-col {display: flex; flex-direction: column; gap: 6px; justify-content: flex-end;}
#action-col .gr-file, #action-col [data-testid="file"] {min-height: 0 !important;}

/* Compact upload dropzone - the default Gradio File drop area is tall
   and mostly empty space; shrink it down to a slim strip. */
#file-upload {min-height: 0 !important;}
#file-upload .wrap {
    min-height: 38px !important; padding: 6px 10px !important;
}
#file-upload .wrap svg {width: 15px !important; height: 15px !important; margin-bottom: 1px !important;}
#file-upload .wrap > * {font-size: 0.72rem !important;}

/* Paste-text box in the same row - trim its line height/padding so it
   matches the shrunk upload dropzone instead of towering over it. */
#input-row textarea {min-height: 0 !important; padding: 6px 10px !important; font-size: 0.8rem !important;}

#results-section {margin-bottom: 20px;}

/* Results table - fixed-height, single-line rows instead of letting long
   Description/Worklog text blow rows out. The Python side already
   truncates + strips HTML tags (see _preview_html); this just makes sure
   the CSS doesn't fight that by re-wrapping or auto-growing rows. Full
   text is available on hover via the native title tooltip. */
#results-table table th {
    background: #f8fafc !important; font-weight: 600 !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.03em; color: var(--dash-text-muted) !important;
    padding: 6px 10px !important;
}
#results-table table td {
    padding: 4px 10px !important; font-size: 0.78rem !important;
    height: 26px !important; max-height: 26px !important; line-height: 1.1 !important;
    vertical-align: middle !important;
    white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important;
    border-bottom: 1px solid #eef1f5 !important;
}
#results-table table td span[title] {cursor: help;}
#results-table table tbody tr:nth-child(even) td {background: #fbfcfe !important;}
#results-table table tbody tr:hover td {background: #f8fafc !important;}

#pagination-row {
    margin-top: 14px; display: flex; align-items: center; justify-content: center; gap: 16px;
}
#pagination-row button {border-radius: 8px !important; font-weight: 600 !important;}
#page-indicator {text-align: center; font-size: 0.85rem; color: var(--dash-text-muted); padding-top: 8px; font-weight: 500;}

.plotly {border-radius: 10px;}

/* Animated "agent working" progress bar - shown in one fixed, centered
   spot (a compact card, not a full-width strip) while an analysis is
   running, instead of Gradio's default per-component loading overlays
   scattered across the summary/donut chart section. */
#agent-progress {margin: 0 auto 16px; max-width: 480px;}
.agent-progress {
    display: flex; align-items: center; gap: 14px;
    background: #ffffff; border: 1px solid var(--dash-border); border-radius: 14px;
    padding: 12px 20px; box-shadow: var(--dash-shadow);
}
.agent-progress-icon {
    font-size: 1.3rem; flex-shrink: 0;
    animation: agent-bounce 1s ease-in-out infinite;
}
@keyframes agent-bounce {
    0%, 100% {transform: translateY(0) rotate(0deg);}
    50% {transform: translateY(-4px) rotate(-6deg);}
}
.agent-progress-track {
    position: relative; flex: 1; height: 8px; border-radius: 999px;
    background: #eef1f5; overflow: hidden;
}
.agent-progress-fill {
    position: absolute; top: 0; left: -40%; width: 40%; height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #16345c, #3b82f6, #16345c);
    animation: agent-slide 1.15s ease-in-out infinite;
}
@keyframes agent-slide {
    0% {left: -40%;}
    100% {left: 100%;}
}
.agent-progress-text {
    font-size: 0.85rem; font-weight: 600; color: var(--dash-text);
    white-space: nowrap; flex-shrink: 0;
}
.agent-progress-text .dots span {
    animation: agent-dot 1.4s infinite; opacity: 0;
}
.agent-progress-text .dots span:nth-child(2) {animation-delay: 0.2s;}
.agent-progress-text .dots span:nth-child(3) {animation-delay: 0.4s;}
@keyframes agent-dot {
    0% {opacity: 0;}
    20% {opacity: 1;}
    100% {opacity: 0;}
}

/* Q&A (Agent) - bounded, dribbble-style chat panel. Intro, transcript,
   and composer all live inside one fixed-height card instead of loosely
   stacked components, so the tab stays a fixed height and only the
   transcript scrolls internally, like a real chat product. */
#chat-panel {
    display: flex !important; flex-direction: column !important;
    height: 640px !important; max-height: 78vh !important;
    padding: 0 !important; overflow: hidden !important;
}
.chat-intro {
    margin: 0 !important; padding: 14px 20px 12px !important;
    border-bottom: 1px solid var(--dash-border); flex-shrink: 0;
}
#chatbot {flex: 1 1 auto !important; min-height: 0 !important; border: none !important;}
#chatbot .wrap {background: #f8fafc !important;}
#chatbot .bubble-wrap, #chatbot .message-wrap {padding: 16px 20px !important; gap: 12px !important;}

/* Message bubbles - best-effort selectors covering the class names used
   across recent Gradio chatbot versions; harmless no-ops if a selector
   doesn't match the installed version's DOM. */
#chatbot .message, #chatbot [class*="user-row"] .bubble, #chatbot [class*="bot-row"] .bubble {
    border-radius: 16px !important; font-size: 0.88rem !important; line-height: 1.45 !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important; border: none !important;
}
#chatbot .message.user, #chatbot [class*="user-row"] .bubble, #chatbot .role-user {
    background: #2563eb !important; color: #ffffff !important;
}
#chatbot .message.bot, #chatbot [class*="bot-row"] .bubble, #chatbot .role-assistant {
    background: #ffffff !important; color: var(--dash-text) !important;
    border: 1px solid var(--dash-border) !important;
}
#chatbot .avatar-container {border-radius: 10px !important; box-shadow: var(--dash-shadow);}

/* Composer - pinned to the bottom of the panel, pill-shaped input/button
   so it reads as a real chat composer rather than a generic form row. */
#chat-input-row {
    flex-shrink: 0 !important; margin: 0 !important; gap: 10px !important;
    align-items: center !important; padding: 14px 20px !important;
    border-top: 1px solid var(--dash-border) !important; background: #ffffff !important;
}
#chat-input-row textarea {
    border-radius: 22px !important; padding: 10px 18px !important; font-size: 0.88rem !important;
    border: 1px solid var(--dash-border) !important; background: #f8fafc !important;
}
#chat-input-row textarea:focus {border-color: #2563eb !important; background: #ffffff !important;}
#chat-input-row button.primary {
    border-radius: 22px !important; font-weight: 600 !important; height: 42px !important; padding: 0 22px !important;
}
#chat-clear-btn {
    flex-shrink: 0 !important; margin: 10px 20px 14px !important; align-self: flex-start !important;
    border-radius: 8px !important; font-size: 0.78rem !important; font-weight: 500 !important;
    color: var(--dash-text-muted) !important; background: transparent !important;
    border: 1px solid var(--dash-border) !important; box-shadow: none !important;
}

/* Overview - Incident Health traffic-light grid. Reuses the same card
   look as .kpi-card (white panel, left accent bar) so it reads as part
   of the same dashboard rather than a new visual language. */
.health-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;}
.health-card {
    background: #ffffff; border: 1px solid var(--dash-border); border-radius: 14px;
    padding: 14px 16px; box-shadow: var(--dash-shadow); border-left: 4px solid var(--accent, #94a3b8);
}
.health-card-top {display: flex; align-items: center; gap: 8px; margin-bottom: 6px;}
.health-dot {width: 9px; height: 9px; border-radius: 999px; background: var(--accent, #94a3b8); flex-shrink: 0;}
.health-label {font-size: 0.78rem; font-weight: 600; color: var(--dash-text);}
.health-status {font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; color: var(--accent, #94a3b8);}
.health-detail {font-size: 0.76rem; color: var(--dash-text-muted); line-height: 1.35;}

/* Overview - Attention Required list. Same card language as the mini
   bar-list rows elsewhere, colored by severity via a left accent bar. */
.attention-list {display: flex; flex-direction: column; gap: 10px;}
.attention-item {
    display: flex; align-items: flex-start; gap: 10px; padding: 10px 14px;
    border-radius: 10px; border-left: 4px solid var(--accent, #94a3b8);
    background: #f8fafc; font-size: 0.82rem;
}
.attention-icon {flex-shrink: 0; font-size: 0.95rem; line-height: 1.4;}
.attention-title {font-weight: 600; color: var(--dash-text);}
.attention-detail {color: var(--dash-text-muted); font-size: 0.78rem; margin-top: 2px;}
.attention-ok {
    padding: 10px 14px; border-radius: 10px; background: #f0fdf4; border: 1px solid #bbf7d0;
    color: #166534; font-size: 0.85rem;
}

/* Overview - Executive Summary card/button. Heading and button share a
   row so the button sits inline, top-right, instead of a full-width
   block button below the heading. */
#exec-summary-card {margin-top: 14px;}
#exec-summary-header {
    align-items: center !important; justify-content: space-between !important;
    gap: 12px !important; margin-bottom: 4px !important; flex-wrap: nowrap !important;
}
#exec-summary-title {margin: 0 !important; padding-bottom: 0 !important; border-bottom: none !important; flex: 1 1 auto;}
#exec-summary-title h3 {margin: 0 !important;}
#exec-summary-btn {
    flex-shrink: 0 !important; width: auto !important; min-width: 0 !important;
    background: transparent !important; color: #2563eb !important;
    border: 1px solid #bfdbfe !important; border-radius: 999px !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    padding: 6px 14px !important; height: auto !important; box-shadow: none !important;
}
#exec-summary-btn:hover {background: #eff6ff !important; border-color: #93c5fd !important;}
#exec-summary-output {
    margin-top: 12px; font-size: 0.88rem; color: var(--dash-text); line-height: 1.55;
}
#exec-summary-output p {margin: 0 0 8px 0;}
"""

AGENT_PROGRESS_HTML = """
<div class="agent-progress">
  <span class="agent-progress-icon">🤖</span>
  <div class="agent-progress-track"><div class="agent-progress-fill"></div></div>
  <span class="agent-progress-text">Agent analyzing tickets<span class="dots"><span>.</span><span>.</span><span>.</span></span></span>
</div>
"""

SEVERITY_NOTE = (
    "Free-text fields (description/worklog) are treated as untrusted data end-to-end - "
    "they are never executed as instructions by the underlying models."
)


def _score_badge(score: int) -> str:
    if score >= 75:
        return "🟢 Good"
    if score >= 50:
        return "🟡 Needs improvement"
    return "🔴 Poor"


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Strips HTML tags and collapses whitespace/newlines so table cells
    render as a single clean line instead of wrapping across many lines."""
    text = text or ""
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _truncate(text: str, limit: int) -> str:
    text = _strip_html(text)
    return text[:limit] + "…" if len(text) > limit else text


ALL_COLUMNS = [
    "Ticket ID", "Category", "Category Confidence", "Category Method",
    "Short Description", "Description", "Worklog Notes", "Worklog Score",
    "Worklog Rating", "Worklog Flags", "Priority", "Status",
    "Assignment Group", "Host / CI", "Validation Notes",
    "Opened At", "Closed At", "Responded At", "Detected At",
]
DEFAULT_VISIBLE_COLUMNS = [
    "Ticket ID", "Category", "Priority", "Host / CI", "Assignment Group",
    "Worklog Score", "Status", "Category Confidence",
]


def _results_to_full_dataframe(analysis) -> pd.DataFrame:
    """Untruncated version for CSV export - the on-screen table truncates
    long text for readability, but "download full results" should contain
    the actual full text, not the display-truncated version."""
    rows = []
    for r in analysis.results:
        rows.append(
            {
                "Ticket ID": r.ticket_id,
                "Category": r.category,
                "Category Confidence": r.category_confidence,
                "Category Method": r.category_method,
                "Short Description": r.short_description,
                "Description": r.description,
                "Worklog Notes": r.worklog,
                "Worklog Score": r.worklog_score,
                "Worklog Rating": _score_badge(r.worklog_score),
                "Worklog Flags": "; ".join(r.worklog_flags) if r.worklog_flags else "",
                "Priority": r.priority or "",
                "Status": r.status or "",
                "Assignment Group": r.assignment_group or "",
                "Host / CI": r.host or "",
                "Validation Notes": "; ".join(r.validation_flags) if r.validation_flags else "",
                # Trend/resolution-metric inputs (Trends & Insights tab).
                # Kept as real datetimes (not strings) so trend_metrics.py
                # can parse them without a round-trip through text.
                "Opened At": r.opened_at,
                "Closed At": r.closed_at,
                "Responded At": r.responded_at,
                "Detected At": r.detected_at,
            }
        )
    return pd.DataFrame(rows)


def _ranked_markdown(title: str, counts: dict, column_label: str, top_n: int = 5) -> str:
    """Small ranked table for a dashboard insight panel, e.g. top
    categories or top hosts by ticket count."""
    if not counts:
        return f"### {title}\nNo data to rank yet - run an analysis first."
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    lines = [f"### {title}", f"| Rank | {column_label} | Tickets |", "|---|---|---|"]
    for i, (name, count) in enumerate(items, start=1):
        lines.append(f"| {i} | {name} | {count} |")
    return "\n".join(lines)


def _category_chart_df(category_counts: dict) -> pd.DataFrame:
    if not category_counts:
        return pd.DataFrame({"Category": [], "Count": []})
    # Descending here since a donut chart reads clockwise from the top,
    # largest slice first.
    items = sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame(items, columns=["Category", "Count"])


def _donut_figure(counts: dict, title: str, color_fn=None) -> go.Figure:
    """Generic donut chart from a {label: count} dict. color_fn, if given,
    maps a label to a hex color so semantically meaningful groups (e.g.
    priority tiers) get consistent colors instead of Plotly's defaults."""
    chart_df = _category_chart_df(counts)
    if chart_df.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No data yet", showarrow=False, font=dict(size=14))],
            height=300,
            margin=dict(t=30, b=10, l=10, r=10),
        )
        return fig

    marker = dict(colors=[color_fn(c) for c in chart_df["Category"]]) if color_fn else {}
    fig = go.Figure(
        data=[
            go.Pie(
                labels=chart_df["Category"],
                values=chart_df["Count"],
                hole=0.55,
                sort=False,
                textinfo="percent",
                marker=marker,
                hovertemplate="%{label}: %{value} tickets (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
    )
    return fig


def _category_chart_figure(category_counts: dict) -> go.Figure:
    """Donut chart of ticket volume by category. Kept as a thin wrapper
    around _donut_figure so any existing caller keeps working unchanged."""
    return _donut_figure(category_counts, "Tickets by Category")


def _priority_color(label: str) -> str:
    """Maps a priority label to a fixed color regardless of exact wording
    ("P1 - Critical", "Critical", etc.) so the priority donut reads
    consistently: red = critical, amber/orange = high/medium, green = low."""
    l = (label or "").lower()
    if "critical" in l or "p1" in l:
        return "#ef4444"
    if "high" in l or "p2" in l:
        return "#f97316"
    if "medium" in l or "p3" in l:
        return "#f59e0b"
    if "low" in l or "p4" in l:
        return "#10b981"
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


def _bar_list_html(items: list, max_items: int = 6, color: str = "#3b82f6") -> str:
    """Renders a sorted list of (label, count) tuples as a horizontal
    bar-list panel, matching the "Incidents by Category" / "Top Affected
    Servers" style in the reference dashboard."""
    if not items:
        return '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">No data to show yet - run an analysis first.</p>'
    items = items[:max_items]
    max_count = max(c for _, c in items) or 1
    rows = []
    for label, count in items:
        pct = max(4, round(count / max_count * 100))
        rows.append(
            '<div class="bar-row">'
            f'<span class="bar-label" title="{label}">{label}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%; background:{color};"></div></div>'
            f'<span class="bar-count">{count}</span>'
            "</div>"
        )
    return '<div class="bar-list">' + "".join(rows) + "</div>"


def _top_host_issues(full_df: pd.DataFrame, top_n: int = 6) -> list:
    """Cross-tabs Host / CI x Category to find each host's most frequent
    issue type - the "Repeated Issues (Top Hosts)" panel."""
    if full_df is None or len(full_df) == 0 or "Host / CI" not in full_df.columns:
        return []
    d = full_df[full_df["Host / CI"].fillna("").astype(str).str.strip() != ""]
    if d.empty:
        return []
    grp = d.groupby(["Host / CI", "Category"]).size().reset_index(name="Count")
    grp = grp.sort_values("Count", ascending=False).head(top_n)
    return grp.to_dict("records")


def _repeated_issues_table_html(rows: list) -> str:
    if not rows:
        return '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">No repeated issues yet - run an analysis first.</p>'
    body = "".join(
        f'<tr><td>{r["Host / CI"]}</td><td>{r["Category"]}</td><td>{r["Count"]}</td></tr>'
        for r in rows
    )
    return (
        '<table class="mini-table"><thead><tr><th>Host / Server</th><th>Issue Type</th><th>Count</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
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


DEFAULT_PAGE_SIZE = 10


def _summary_markdown(stats: dict) -> str:
    return (
        f"### Summary\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total records seen | {stats['total_records']} |\n"
        f"| Valid records analyzed | {stats['valid_records']} |\n"
        f"| Rejected records | {stats['rejected_records']} |\n"
        f"| Average worklog score | {stats['average_worklog_score']} / 100 |\n"
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


def _kpi_card_v2(icon: str, label: str, value, accent: str, note: str = "") -> str:
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    return (
        f'<div class="kpi-card kpi-card-v2" style="--accent:{accent}">'
        f'<div class="kpi-icon" style="background:{accent}1a; color:{accent};">{icon}</div>'
        f'<div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{note_html}</div>'
        "</div>"
    )


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


def _overview_kpi_html(kpis: dict) -> str:
    """Renders the five executive KPI cards from overview_metrics'
    aggregated kpis dict. Pure presentation - all the numbers are already
    computed by overview_metrics.compute_overview_kpis."""
    try:
        resolution_value = (
            f'{kpis["avg_resolution_hours"]}h' if kpis.get("avg_resolution_hours") is not None else "N/A"
        )
        return (
            '<div class="kpi-grid kpi-grid-5">'
            + _kpi_card_v2("🎫", "Total Incidents", kpis.get("total_incidents", 0), "#3b82f6")
            + _kpi_card_v2(
                "🔴", "P1/P2 Incidents", kpis.get("high_priority_count", 0), "#ef4444",
                note=f'{kpis.get("high_priority_pct", 0)}% of total',
            )
            + _kpi_card_v2("⏱️", "Avg Resolution Time", resolution_value, "#0ea5e9")
            + _kpi_card_v2("📝", "Avg Worklog Score", f'{kpis.get("avg_worklog_score", 0)} / 100', "#f59e0b")
            + _kpi_card_v2(
                "⚠️", "Poor Worklog %", f'{kpis.get("poor_worklog_pct", 0)}%', "#f97316",
                note=f'{kpis.get("poor_worklog_count", 0)} incident(s)',
            )
            + "</div>"
        )
    except Exception:
        return _OVERVIEW_KPI_PLACEHOLDER


_HEALTH_STATUS_STYLE = {
    "good": ("#10b981", "Good"),
    "warning": ("#f59e0b", "Needs Attention"),
    "critical": ("#ef4444", "Critical"),
    "unknown": ("#94a3b8", "No Data"),
}

_HEALTH_LABELS = [
    ("priority_health", "Priority Health"),
    ("resolution_performance", "Resolution Performance"),
    ("worklog_quality", "Worklog Quality"),
    ("data_quality", "Data Quality"),
]

_HEALTH_PLACEHOLDER = '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'


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


_ATTENTION_ICON = {"critical": "🔴", "warning": "🟡"}
_ATTENTION_PLACEHOLDER = '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'


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
            line=dict(color="#2563eb", width=2), marker=dict(size=5),
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


def _refresh_overview(full_df: pd.DataFrame, summary_stats: dict):
    """Recomputes every Overview section (KPI cards, health indicators,
    attention list, volume trend) from the latest analysis. Wrapped in
    try/except per-section isn't needed here since each underlying
    overview_metrics function already degrades gracefully on its own."""
    try:
        kpis = overview_metrics.compute_overview_kpis(full_df, summary_stats or {})
        health = overview_metrics.compute_health_indicators(full_df, kpis)
        attention = overview_metrics.compute_attention_items(health)
        return (
            _overview_kpi_html(kpis),
            _health_indicators_html(health),
            _attention_html(attention),
            _overview_trend_figure(full_df),
        )
    except Exception:
        return _OVERVIEW_KPI_PLACEHOLDER, _HEALTH_PLACEHOLDER, _ATTENTION_PLACEHOLDER, _overview_trend_figure(None)


async def _llm_executive_summary(payload: dict) -> "tuple[str, bool]":
    """Sends ONLY the small aggregated `payload` dict (KPIs/health/
    attention/trend direction - no ticket rows, descriptions, or worklog
    text) to the LLM and asks for a short management-friendly summary.
    Falls back to a deterministic, rule-based summary (same underlying
    numbers, no LLM) if the API key/package isn't available or the call
    fails for any reason, so the button never errors out for the user.

    Returns (summary_text, used_llm).
    """
    try:
        import anthropic  # local import - optional dependency for this feature only

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        prompt = (
            "You are an ITSM reporting assistant writing for a management audience. "
            "Using ONLY the aggregated incident metrics in the JSON below, write a short "
            "(4-6 sentence) executive summary in plain English. Do not invent any numbers "
            "that aren't present in the data, and do not mention individual tickets - these "
            "are already-aggregated figures.\n\n"
            f"Aggregated metrics (JSON):\n{json.dumps(payload, default=str)}"
        )
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
        if not summary:
            raise RuntimeError("empty LLM response")
        return summary, True
    except Exception:
        return overview_metrics.fallback_summary_text(payload), False


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
        note = "" if used_llm else "\n\n_(LLM unavailable right now - showing a rule-based summary of the same metrics.)_"
        return gr.update(value=f"{summary_text}{note}")
    except Exception as exc:
        return gr.update(value=f"Could not generate the executive summary right now ({exc}).")


def _analysis_to_summary_stats(analysis) -> dict:
    return {
        "total_records": analysis.total_records,
        "valid_records": analysis.valid_records,
        "rejected_records": analysis.rejected_records,
        "average_worklog_score": analysis.average_worklog_score,
    }


_RESOLUTION_METRICS_PLACEHOLDER = (
    '<div class="kpi-grid">'
    + _kpi_card_v2("🔍", "MTTD (Detect)", "—", "#6366f1")
    + _kpi_card_v2("📨", "MTTA (Acknowledge)", "—", "#0ea5e9")
    + _kpi_card_v2("🛠️", "MTTR (Resolve)", "—", "#10b981")
    + _kpi_card_v2("🎯", "SLA Compliance", "—", "#f59e0b")
    + "</div>"
)


def _trend_figure(full_df: pd.DataFrame, granularity: str) -> go.Figure:
    """Dual-axis chart: ticket volume as bars (left axis), average worklog
    score as a line (right axis), bucketed by day/week/month on Opened At."""
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
    scores = [p["avg_worklog_score"] for p in series]

    fig.add_trace(go.Bar(x=periods, y=counts, name="Ticket volume", marker_color="#3b82f6", yaxis="y1"))
    fig.add_trace(go.Scatter(
        x=periods, y=scores, name="Avg worklog score", mode="lines+markers",
        line=dict(color="#f59e0b", width=2), marker=dict(size=6), yaxis="y2",
    ))
    fig.update_layout(
        title=f"Ticket volume & worklog quality - {granularity}",
        height=340,
        margin=dict(t=40, b=10, l=10, r=10),
        xaxis=dict(title=""),
        yaxis=dict(title="Tickets", side="left", rangemode="tozero"),
        yaxis2=dict(title="Avg score", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.25,
    )
    return fig


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


def _refresh_trend(full_df: pd.DataFrame, granularity: str):
    return _trend_figure(full_df, granularity), _resolution_metrics_html(full_df)


def _truncate_full_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the on-screen truncated view from a cached full (untruncated)
    dataframe - same truncation rules applied to a fresh analysis in
    _results_to_full_dataframe, just starting from a dataframe already on
    disk instead of a freshly-computed AnalysisResponse."""
    if full_df is None or len(full_df) == 0:
        return pd.DataFrame(columns=ALL_COLUMNS)
    df = full_df.copy()
    if "Short Description" in df.columns:
        df["Short Description"] = df["Short Description"].apply(lambda t: _truncate(t, 100))
    if "Description" in df.columns:
        df["Description"] = df["Description"].apply(lambda t: _truncate(t, 140))
    if "Worklog Notes" in df.columns:
        df["Worklog Notes"] = df["Worklog Notes"].apply(lambda t: _truncate(t, 140))
    return df


async def _analyze(file_obj, pasted_text):
    has_file = file_obj is not None
    has_text = bool(pasted_text and pasted_text.strip())

    if not has_file and not has_text:
        raise gr.Error("Upload a file (CSV/XLSX/TXT) or paste incident text first.")
    if has_file and has_text:
        # Both fields are populated - almost always a leftover value from a
        # previous analysis still sitting in the file uploader, not two
        # inputs the user actually intends together. Rather than silently
        # picking the file (which is what used to happen, and produced a
        # confusing "already categorized" result while the newly-pasted
        # text was quietly ignored), make the user disambiguate.
        raise gr.Error(
            "Both a file and pasted text are present - please clear one of them. "
            "(If you analyzed a file earlier, use the ✕ on the file box to remove it "
            "before pasting text, or vice versa.)"
        )

    # Show the animated agent progress bar in its single fixed spot before
    # doing any work; other outputs are left untouched (gr.update()) so
    # nothing under them flickers or shows its own loading state.
    yield (gr.update(visible=True),) + (gr.update(),) * 13

    # Hash the raw input (file bytes, or the pasted text) so an identical
    # upload/paste can be served straight from the DB instead of hitting
    # the LLM pipeline again.
    if has_file:
        with open(file_obj.name, "rb") as f:
            content = f.read()
        input_label = file_obj.name
    else:
        content = pasted_text.strip().encode("utf-8")
        input_label = "pasted_text"

    file_hash = compute_file_hash(content)
    llm_worklog_scoring_enabled = os.environ.get("ENABLE_LLM_WORKLOG_SCORING", "false").lower() == "true"
    cached = get_cached_result(file_hash, current_llm_worklog_scoring_enabled=llm_worklog_scoring_enabled)

    if cached is not None:
        full_df = cached["full_df"]
        summary_stats = cached["summary_stats"]
        category_counts = cached["category_counts"]
        host_counts = cached["host_counts"]
        cached_on = cached["uploaded_at"][:19].replace("T", " ")
        cache_notice = gr.update(
            value=f"✅ Identical input already analyzed on {cached_on} — showing cached results, no LLM calls made.",
            visible=True,
        )
    else:
        if has_file:
            analysis = await run_pipeline_from_bytes(file_obj.name, content)
        else:
            analysis = await run_pipeline_from_text(pasted_text)

        full_df = _results_to_full_dataframe(analysis)
        summary_stats = _analysis_to_summary_stats(analysis)
        category_counts = analysis.category_counts
        host_counts = analysis.host_counts

        save_result(
            file_hash=file_hash,
            filename=input_label,
            full_df=full_df,
            summary_stats=summary_stats,
            category_counts=category_counts,
            host_counts=host_counts,
            llm_worklog_scoring_enabled=llm_worklog_scoring_enabled,
        )
        cache_notice = gr.update(value="", visible=False)

    df = _truncate_full_df(full_df)
    overview_kpis = overview_metrics.compute_overview_kpis(full_df, summary_stats)
    summary = _overview_kpi_html(overview_kpis)

    # Prepare CSV for download - always the FULL untruncated result set,
    # independent of whatever filter/page/truncation the on-screen table
    # is showing.
    csv_buf = io.StringIO()
    full_df.to_csv(csv_buf, index=False)
    csv_path = f"/tmp/itsm_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w") as f:
        f.write(csv_buf.getvalue())

    # Priority isn't pre-aggregated like category/host counts are, so it's
    # derived here from full_df - works the same for a fresh analysis and
    # a cached result, since both always carry a full_df.
    priority_counts = _priority_counts(full_df)
    chart = _donut_figure(priority_counts, "Incidents by Priority", color_fn=_priority_color)

    category_bar_html = _bar_list_html(
        sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True), color="#3b82f6"
    )
    host_bar_html = _bar_list_html(
        sorted(host_counts.items(), key=lambda kv: kv[1], reverse=True), color="#3b82f6"
    )
    repeated_issues_html = _repeated_issues_table_html(_top_host_issues(full_df))
    assignment_group_html = _assignment_group_table_html(_assignment_group_performance(full_df))

    # Aggregate stats handed to the chat/RAG tab alongside the retrieved
    # ticket excerpts - same numbers already shown on the dashboard, just
    # collected into one dict for the chat prompt.
    chat_stats = {
        **summary_stats,
        "category_counts": category_counts,
        "host_counts": host_counts,
    }

    yield (
        gr.update(visible=False), summary, chart, csv_path, df, cache_notice,
        category_bar_html, host_bar_html, repeated_issues_html, assignment_group_html,
        chat_stats,
        # Clear both inputs now that this run is done, so a stale file or
        # stale pasted text can't get silently reused (and mistaken for
        # "both provided") on the next click.
        gr.update(value=None), gr.update(value=""),
        # summary_stats_state - feeds the Overview health/attention/exec
        # summary calculations without needing to re-run the pipeline.
        summary_stats,
    )


def _build_chat_index(full_df: pd.DataFrame) -> "rag.TicketIndex":
    """Rebuilds the chat/RAG index from the latest analysis dataframe.
    Cheap and synchronous - embeddings are computed lazily on first
    question (see rag.TicketIndex._ensure_vectors), not here."""
    return rag.build_index_from_dataframe(full_df)


def _history_to_messages(history: list[tuple[str, str]]) -> list[dict]:
    """Converts the internal (question, answer) tuple history - the shape
    rag.answer_question expects for prompt context - into the role/content
    message dicts gr.Chatbot renders in this Gradio version."""
    messages = []
    for question, answer in history:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return messages


async def _chat_respond(message: str, history: list, index, stats: dict):
    message = (message or "").strip()
    if not message:
        return _history_to_messages(history), history, ""

    if index is None or not getattr(index, "rows", None):
        answer = "Run an analysis on the Dashboard tab first - then come back and ask away."
    else:
        answer = await rag.answer_question(message, index, stats or {}, history)

    history = history + [(message, answer)]
    return _history_to_messages(history), history, ""


def _chat_clear():
    return [], []


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Restricts a dataframe to the fixed default column set for on-screen
    display, in ALL_COLUMNS order. Filtering/pagination/CSV export always
    operate on the full, un-reduced dataframe - only this final display
    step drops columns."""
    cols = [c for c in ALL_COLUMNS if c in DEFAULT_VISIBLE_COLUMNS]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols or ["Ticket ID"])
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = ["Ticket ID"]  # never render a fully empty table
    return df[cols]


def _apply_filters(full_df: pd.DataFrame, category: str, min_score: int) -> pd.DataFrame:
    if full_df is None or len(full_df) == 0:
        return pd.DataFrame() if full_df is None else full_df
    filtered = full_df.copy()
    if category and category != "All":
        filtered = filtered[filtered["Category"] == category]
    filtered = filtered[filtered["Worklog Score"] >= min_score]
    return filtered.reset_index(drop=True)


def _paginate(filtered_df: pd.DataFrame, page: int, page_size: int):
    total = len(filtered_df) if filtered_df is not None else 0
    page_size = page_size or DEFAULT_PAGE_SIZE
    total_pages = max(1, -(-total // page_size))  # ceil division
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = filtered_df.iloc[start:end] if total else filtered_df
    indicator = f"Page {page} of {total_pages}  ·  {total} ticket{'s' if total != 1 else ''}"
    return page_df, indicator, page


def _refresh_view(full_df, category, min_score, page_size):
    """Re-applies filters, resets to page 1, and returns everything the
    table/pagination controls need. Used after a new analysis runs or
    whenever a filter/page-size control changes."""
    filtered = _apply_filters(full_df, category, min_score)
    page_df, indicator, page = _paginate(filtered, 1, page_size)
    return _select_columns(page_df), indicator, filtered, page


def _go_to_page(filtered_df, page, page_size, delta):
    page_df, indicator, new_page = _paginate(filtered_df, (page or 1) + delta, page_size)
    return _select_columns(page_df), indicator, new_page


SIDEBAR_BRAND_HTML = """
<div class="side-brand">
  <div class="side-brand-icon">🤖</div>
  <div>
    <div class="side-brand-title">ITSM Agent</div>
    <div class="side-brand-sub">Incident Analytics</div>
  </div>
</div>
"""

TOPBAR_HTML = """
<h1>ITSM Incident Analytics</h1>
<p>AI-powered insights for better service and faster resolution</p>
"""

# Sidebar nav items -> the id of the gr.Tab each one opens. Order here
# drives both the buttons drawn in the sidebar and the tabs built below.
NAV_ITEMS = [
    ("overview", "🏠", "Overview"),
    ("analysis", "📈", "Incident Analysis"),
    ("categorization", "🗂️", "Categorization"),
    ("trends", "📊", "Trends & Insights"),
    ("qa", "💬", "Q&A (Agent)"),
    ("export", "⬇️", "Export"),
    ("settings", "⚙️", "Settings"),
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ITSM Quality Analysis Agent", css=CUSTOM_CSS) as demo:
        # App shell: dark nav rail on the left, everything else scrolls in
        # the main column on the right. The nav items below are real
        # buttons that switch between the gr.Tab sections built further
        # down, so the sidebar is an actual working nav, not a static mock.
        with gr.Row(elem_id="app-shell", equal_height=False):
            with gr.Column(scale=0, min_width=210, elem_id="sidebar-col"):
                gr.HTML(SIDEBAR_BRAND_HTML)
                nav_buttons = []
                with gr.Column(elem_id="nav-buttons"):
                    for i, (key, icon, label) in enumerate(NAV_ITEMS):
                        btn = gr.Button(
                            f"{icon}  {label}",
                            elem_classes=["nav-item", "active"] if i == 0 else ["nav-item"],
                        )
                        nav_buttons.append(btn)

            with gr.Column(scale=1, elem_id="main-col"):
                gr.HTML(TOPBAR_HTML, elem_id="topbar")
                gr.Markdown(f"_{SEVERITY_NOTE}_", elem_classes=["severity-note"])

                with gr.Tabs(elem_id="main-tabs") as main_tabs:
                    with gr.Tab("Overview", id=0):
                        agent_progress = gr.HTML(AGENT_PROGRESS_HTML, elem_id="agent-progress", visible=False)
                        cache_notice = gr.Markdown(visible=False, elem_id="cache-notice")

                        # KPI strip - headline numbers for management at a glance,
                        # shown above the upload bar so totals are the first thing seen.
                        with gr.Row(elem_id="metrics-row"):
                            summary_md = gr.HTML(_OVERVIEW_KPI_PLACEHOLDER)

                        # Single full-width input bar - upload, paste, and the
                        # analyze action sit on one row (download moved to Export).
                        with gr.Row(elem_id="input-row", elem_classes=["dash-card"], equal_height=False):
                            with gr.Column(scale=3, min_width=260):
                                file_input = gr.File(
                                    label="Upload incident file (.xlsx, .csv, .txt)",
                                    file_types=[".xlsx", ".xls", ".csv", ".txt"],
                                    elem_id="file-upload",
                                )
                            with gr.Column(scale=4, min_width=320):
                                text_input = gr.Textbox(label="...or paste unstructured incident text", lines=2,
                                                          placeholder="INC0012345\nShort description: ...\nWorklog: ...")
                            with gr.Column(scale=2, min_width=180, elem_id="action-col"):
                                analyze_btn = gr.Button("Analyze", variant="primary")

                        # Incident Health - four traffic-light indicators so
                        # management can scan overall status in a second.
                        with gr.Column(elem_classes=["dash-card"]):
                            gr.Markdown("### 🩺 Incident Health", elem_classes=["section-heading"])
                            health_html = gr.HTML(_HEALTH_PLACEHOLDER)

                        # Incident Volume Trend - a simple line chart, distinct
                        # from the dual-axis chart on Trends & Insights.
                        with gr.Column(elem_classes=["dash-card"]):
                            gr.Markdown("### 📈 Incident Volume Trend", elem_classes=["section-heading"])
                            overview_trend_chart = gr.Plot(show_label=False)

                        # Attention Required - a short, management-facing
                        # roll-up of anything currently outside a healthy range.
                        with gr.Column(elem_classes=["dash-card"]):
                            gr.Markdown("### 🚩 Attention Required", elem_classes=["section-heading"])
                            attention_html = gr.HTML(_ATTENTION_PLACEHOLDER)

                        # Executive Summary - computes KPIs/trends with
                        # pandas first, then sends only that small
                        # aggregated dict to the LLM for a short write-up.
                        # Heading + button share a row (button top-right,
                        # small outlined pill) instead of a full-width
                        # button sitting below the heading.
                        with gr.Column(elem_classes=["dash-card"], elem_id="exec-summary-card"):
                            with gr.Row(elem_id="exec-summary-header"):
                                gr.Markdown(
                                    "### 🧾 Executive Summary", elem_classes=["section-heading"],
                                    elem_id="exec-summary-title",
                                )
                                exec_summary_btn = gr.Button(
                                    "✨ Generate summary", size="sm", elem_id="exec-summary-btn",
                                )
                            exec_summary_output = gr.Markdown(
                                "Run an analysis, then click **Generate summary** for a "
                                "management-friendly write-up of the KPIs above.",
                                elem_id="exec-summary-output",
                            )

                    with gr.Tab("Incident Analysis", id=1):
                        # Recent Incidents: the primary, full-width table.
                        with gr.Column(elem_id="results-section", elem_classes=["dash-card"]):
                            gr.Markdown("### 📋 Recent Incidents (Analyzed &amp; Categorized)", elem_classes=["section-heading"])
                            results_table = gr.Dataframe(
                                label=None,
                                show_label=False,
                                interactive=False,
                                wrap=False,
                                max_height=460,
                                elem_id="results-table",
                            )
                            with gr.Row(elem_id="pagination-row"):
                                prev_btn = gr.Button("← Previous", size="sm")
                                page_indicator = gr.Markdown("Page 1 of 1  ·  0 tickets", elem_id="page-indicator")
                                next_btn = gr.Button("Next →", size="sm")

                    with gr.Tab("Categorization", id=2):
                        # Category breakdown + priority donut.
                        with gr.Row(elem_id="panel-row-1"):
                            with gr.Column(scale=1, elem_classes=["dash-card"]):
                                gr.Markdown("### 🗂️ Incidents by Category", elem_classes=["section-heading"])
                                category_bar_html = gr.HTML(
                                    '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'
                                )
                            with gr.Column(scale=1, elem_classes=["dash-card"]):
                                gr.Markdown("### 🎯 Incidents by Priority", elem_classes=["section-heading"])
                                category_chart = gr.Plot(show_label=False)

                    with gr.Tab("Trends & Insights", id=3):
                        # KPI trend - ticket volume + worklog quality over
                        # time, toggle between daily/weekly/monthly views.
                        with gr.Column(elem_classes=["dash-card"]):
                            gr.Markdown("### 📈 Ticket Trend", elem_classes=["section-heading"])
                            trend_granularity = gr.Radio(
                                ["Daily", "Weekly", "Monthly"], value="Daily",
                                show_label=False, elem_id="trend-granularity",
                            )
                            trend_chart = gr.Plot(show_label=False)

                        # Resolution metrics - MTTD/MTTA/MTTR/SLA. Any
                        # metric whose timestamp column isn't in the
                        # uploaded data shows "N/A" with an explanation
                        # rather than a fabricated number.
                        with gr.Column(elem_classes=["dash-card"]):
                            gr.Markdown("### ⏱️ Resolution Metrics", elem_classes=["section-heading"])
                            resolution_metrics_html = gr.HTML(_RESOLUTION_METRICS_PLACEHOLDER)

                        # Top hosts, repeated issues, assignment group performance.
                        with gr.Row(elem_id="panel-row-2"):
                            with gr.Column(scale=1, elem_classes=["dash-card"]):
                                gr.Markdown("### 🖥️ Top Affected Servers / Hosts", elem_classes=["section-heading"])
                                host_bar_html = gr.HTML(
                                    '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'
                                )
                            with gr.Column(scale=1, elem_classes=["dash-card"]):
                                gr.Markdown("### 🔁 Repeated Issues (Top Hosts)", elem_classes=["section-heading"])
                                repeated_issues_html = gr.HTML(
                                    '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'
                                )
                            with gr.Column(scale=1, elem_classes=["dash-card"]):
                                gr.Markdown("### 👥 Assignment Group Performance", elem_classes=["section-heading"])
                                assignment_group_html = gr.HTML(
                                    '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'
                                )

                    with gr.Tab("Q&A (Agent)", id=4):
                        # Single bounded chat panel (intro + transcript +
                        # composer) instead of loosely stacked components -
                        # keeps the tab a fixed height with the transcript
                        # scrolling internally like a normal chat app.
                        with gr.Column(elem_id="chat-panel", elem_classes=["dash-card"]):
                            gr.Markdown(
                                "Ask a question about the tickets you just analyzed - e.g. "
                                "*\"what's driving high-priority incidents?\"*, "
                                "*\"which assignment group has the worst worklog quality?\"*, or "
                                "*\"summarize the recurring issues on our database servers.\"* "
                                "Answers are grounded only in the analyzed batch (Overview tab) - "
                                "run an analysis first if you haven't yet.",
                                elem_classes=["severity-note", "chat-intro"],
                            )
                            chatbot = gr.Chatbot(height=440, show_label=False, elem_id="chatbot")
                            with gr.Row(elem_id="chat-input-row"):
                                chat_input = gr.Textbox(
                                    placeholder="Ask a question about the analyzed tickets...",
                                    show_label=False, scale=5, container=False,
                                )
                                chat_send = gr.Button("Send", variant="primary", scale=1)
                            chat_clear_btn = gr.Button("Clear conversation", size="sm", elem_id="chat-clear-btn")

                    with gr.Tab("Export", id=5):
                        gr.Markdown(
                            "Run an analysis on the Overview tab, then download the full, "
                            "untruncated results as a CSV here.",
                            elem_classes=["severity-note"],
                        )
                        with gr.Column(elem_classes=["dash-card"], elem_id="export-card"):
                            gr.Markdown("### ⬇️ Download Results", elem_classes=["section-heading"])
                            download_file = gr.File(label="Full results (CSV)", interactive=False)

                    with gr.Tab("Settings", id=6):
                        with gr.Column(elem_classes=["dash-card"]):
                            gr.Markdown("### ⚙️ Settings", elem_classes=["section-heading"])
                            gr.Markdown(
                                "Nothing configurable here yet - this tab is a placeholder for "
                                "future options (e.g. default page size, scoring thresholds).",
                                elem_classes=["severity-note"],
                            )

        full_results_state = gr.State(pd.DataFrame())
        filtered_results_state = gr.State(pd.DataFrame())
        page_state = gr.State(1)
        # Category/score filters and the rows-per-page control have been
        # removed from the UI; these fixed states keep _apply_filters /
        # _paginate (and their existing behavior) unchanged underneath.
        category_state = gr.State("All")
        min_score_state = gr.State(0)
        page_size_state = gr.State(DEFAULT_PAGE_SIZE)

        # Chat/RAG state: the semantic index + aggregate stats are rebuilt
        # from the latest analysis; chat_history_state is the running
        # (question, answer) transcript fed back into the prompt for
        # multi-turn context.
        chat_index_state = gr.State(None)
        chat_stats_state = gr.State({})
        chat_history_state = gr.State([])

        # Raw summary_stats dict (total/valid/rejected records, avg
        # worklog score) from the latest analysis - feeds the Overview
        # KPI/health/attention calculations and the executive summary
        # without needing to re-run the pipeline.
        summary_stats_state = gr.State({})

        analyze_btn.click(
            fn=_analyze,
            inputs=[file_input, text_input],
            outputs=[
                agent_progress, summary_md, category_chart, download_file,
                full_results_state, cache_notice,
                category_bar_html, host_bar_html, repeated_issues_html, assignment_group_html,
                chat_stats_state,
                file_input, text_input,
                summary_stats_state,
            ],
        ).then(
            fn=_refresh_view,
            inputs=[full_results_state, category_state, min_score_state, page_size_state],
            outputs=[results_table, page_indicator, filtered_results_state, page_state],
        ).then(
            fn=_build_chat_index,
            inputs=[full_results_state],
            outputs=[chat_index_state],
        ).then(
            fn=_refresh_trend,
            inputs=[full_results_state, trend_granularity],
            outputs=[trend_chart, resolution_metrics_html],
        ).then(
            fn=_refresh_overview,
            inputs=[full_results_state, summary_stats_state],
            outputs=[summary_md, health_html, attention_html, overview_trend_chart],
        )

        exec_summary_btn.click(
            fn=_generate_executive_summary,
            inputs=[full_results_state, summary_stats_state],
            outputs=[exec_summary_output],
        )

        trend_granularity.change(
            fn=_refresh_trend,
            inputs=[full_results_state, trend_granularity],
            outputs=[trend_chart, resolution_metrics_html],
        )

        prev_btn.click(
            fn=lambda filtered_df, page, page_size: _go_to_page(filtered_df, page, page_size, -1),
            inputs=[filtered_results_state, page_state, page_size_state],
            outputs=[results_table, page_indicator, page_state],
        )
        next_btn.click(
            fn=lambda filtered_df, page, page_size: _go_to_page(filtered_df, page, page_size, 1),
            inputs=[filtered_results_state, page_state, page_size_state],
            outputs=[results_table, page_indicator, page_state],
        )

        chat_send.click(
            fn=_chat_respond,
            inputs=[chat_input, chat_history_state, chat_index_state, chat_stats_state],
            outputs=[chatbot, chat_history_state, chat_input],
        )
        chat_input.submit(
            fn=_chat_respond,
            inputs=[chat_input, chat_history_state, chat_index_state, chat_stats_state],
            outputs=[chatbot, chat_history_state, chat_input],
        )
        chat_clear_btn.click(fn=_chat_clear, outputs=[chatbot, chat_history_state])

        # Wire each sidebar nav button to (a) switch the visible tab and
        # (b) move the "active" highlight to itself and off the rest.
        def _make_nav_handler(selected_idx: int):
            def _handler():
                updates = [gr.Tabs(selected=selected_idx)]
                for i in range(len(NAV_ITEMS)):
                    classes = ["nav-item", "active"] if i == selected_idx else ["nav-item"]
                    updates.append(gr.update(elem_classes=classes))
                return updates
            return _handler

        for i, btn in enumerate(nav_buttons):
            btn.click(fn=_make_nav_handler(i), inputs=[], outputs=[main_tabs, *nav_buttons])

    return demo
