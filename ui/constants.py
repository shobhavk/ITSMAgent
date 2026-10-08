"""Static constants: placeholder text, static HTML snippets, column lists and palettes."""
import os
import re


HERO_HTML = """
<div class="hero-card">
  <div class="hero-main">
    <div>
      <div class="hero-title">Analyze your incident data</div>
      <div class="hero-sub">Upload an export or paste ticket text - the agent validates, categorizes and scores every ticket.</div>
    </div>
    <div class="hero-steps">
      <span class="hero-step"><span class="hero-step-num">1</span>Upload or paste</span>
      <span class="hero-step"><span class="hero-step-num">2</span>Analyze</span>
      <span class="hero-step"><span class="hero-step-num">3</span>Explore</span>
    </div>
  </div>
</div>
"""


_SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "sample_incidents.csv"
)


_EXEC_SUMMARY_PLACEHOLDER = (
    "Click **Generate summary** for a management-friendly write-up of the KPIs above."
)


SEVERITY_NOTE = (
    "Free-text fields (description/worklog) are treated as untrusted data end-to-end - "
    "they are never executed as instructions by the underlying models."
)


_TAG_RE = re.compile(r"<[^>]+>")


_WS_RE = re.compile(r"\s+")


ALL_COLUMNS = [
    "Ticket ID", "Category", "Category Confidence", "Review", "Category Method",
    "Short Description", "Description", "Worklog Notes", "Worklog Score",
    "Worklog Rating", "Worklog Flags", "Priority", "Status",
    "Assignment Group", "Host / CI", "Validation Notes",
    "Opened At", "Closed At", "Created At", "Resolved At", "Responded At", "Detected At",
]


DEFAULT_VISIBLE_COLUMNS = [
    "Ticket ID", "Category", "Priority", "Host / CI", "Assignment Group",
    "Worklog Score", "Status", "Category Confidence", "Review",
]


# Categorization tab: a ticket is flagged "Needs review" when the category
# confidence is below this value, or when it could not be categorised at all.
LOW_CONFIDENCE_THRESHOLD = 0.8
UNCATEGORISED_LABELS = {"", "other", "uncategorized", "uncategorised", "unknown", "unclassified"}


TIMELINE_COLUMNS = [
    "Opened/Created At", "Acknowledged At", "Time to Acknowledge (h)",
    "Reassignment Detected", "Reassigned To", "Reassigned At",
    "External Info Quality", "Resolved At",
]


_CATEGORY_PALETTE = [
    "#0f6e7a", "#3b6ea5", "#b7791f", "#b3261e",
    "#2e7d5b", "#7a5c99", "#c05621", "#64748b",
]


DEFAULT_PAGE_SIZE = 10


_DELTA_MIN_DATED = 8


_DELTA_MIN_PER_HALF = 3


_HEALTH_STATUS_STYLE = {
    "good": ("#2e7d5b", "Good"),
    "warning": ("#b7791f", "Needs Attention"),
    "critical": ("#b3261e", "Critical"),
    "unknown": ("#94a3b8", "No Data"),
}


_HEALTH_LABELS = [
    ("priority_health", "Priority Health"),
    ("resolution_performance", "Resolution Performance"),
    ("worklog_quality", "Worklog Quality"),
    ("data_quality", "Data Quality"),
]


_HEALTH_PLACEHOLDER = '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'


_ATTENTION_ICON = {"critical": "🔴", "warning": "🟡"}


_ATTENTION_PLACEHOLDER = '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'


# --- Recommendations -------------------------------------------------
# Rendering + handlers for the Recommendations tab. All numbers come from
# app/services/recommendations.py (pure pandas over the existing
# analytics); nothing is computed here. The deterministic cards below are
# always what's shown - the LLM write-up is an optional, opt-in extra
# layer on the same already-computed content, exactly like the Overview
# tab's Executive Summary.

_RECOMMENDATIONS_PLACEHOLDER = (
    '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">'
    "Run an analysis on the Overview tab - recommendations are generated automatically "
    "from that batch.</p>"
)


_REC_ATTENTION_STYLE = {
    "Critical": ("#b3261e", "#b3261e1f", "🔴 Critical"),
    "High": ("#b7791f", "#b7791f1f", "🟠 High"),
    "Medium": ("#2b7a9b", "#2b7a9b1f", "🔵 Medium"),
    "Low": ("#94a3b8", "#94a3b81f", "⚪ Low"),
}


_TREND_BAR_PALETTE = ["#0f6e7a"]  # single series colour; indexed with % len() by the trend chart


# --- Incident Timeline -------------------------------------------------
# Per-incident audit table (Trends & Insights tab): when each incident
# came in, how long it took to acknowledge, any reassignment detected in
# External Info, and whether External Info carries a proper timestamp
# trail. All numbers come from app/services/incident_timeline.py (pure
# pandas + regex over the existing analytics) - nothing computed here.
# The LLM write-up is the same optional, opt-in extra layer used by the
# Overview tab's Executive Summary and the Recommendations write-up.

_TIMELINE_PLACEHOLDER_MD = (
    "Run an analysis on the Overview tab, then click **Generate summary** to turn the "
    "table below into a management-ready briefing on acknowledgment times and External "
    "Info audit-trail quality."
)


_ANALYZE_N_OUTPUTS = 17


_ANALYZE_STEPS = 3


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
<button type="button" class="theme-toggle" title="Switch between light and dark theme" aria-label="Toggle dark mode"
  onclick="var d=document.body.classList.toggle('dark');try{localStorage.setItem('itsm-theme',d?'dark':'light')}catch(e){}">
  <span class="to-dark">🌙 Dark</span><span class="to-light">☀️ Light</span>
</button>
"""
