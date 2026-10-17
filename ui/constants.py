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
    "Opened At", "Closed At", "Created At", "Resolved At", "Responded At", "Detected At", "Ack Start At",
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
    "#2dd4bf", "#60a5fa", "#f2b01e", "#ef6b62",
    "#34d399", "#a78bfa", "#fb923c", "#64748b",
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
  <div class="side-brand-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 17l5-5 4 4 7-8"/></svg></div>
  <div class="side-brand-title">ITSM Quality Agent</div>
</div>
<label class="nav-search">
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg>
  <input id="nav-search" type="text" placeholder="Search or jump to..." autocomplete="off" spellcheck="false" aria-label="Search or jump to a page">
  <kbd>⌘K</kbd>
</label>
"""


TOPBAR_HTML = """
<div class="pg-text">
  <div class="pg-crumb" id="pg-crumb">Analyze / Overview</div>
  <h1 id="pg-title">Overview</h1>
  <p id="pg-sub">Upload data, run the analysis, then explore the results.</p>
</div>
<button type="button" class="theme-toggle" title="Switch between light and dark theme" aria-label="Toggle dark mode"
  onclick="var d=document.body.classList.toggle('dark');try{localStorage.setItem('itsm-theme',d?'dark':'light')}catch(e){}">
  <svg class="to-dark" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/></svg>
  <svg class="to-light" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4l1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
</button>
"""


# Presentation-only shell behaviour (runs once on page load): the page heading follows the open tab,
# the sidebar search jumps to a page, Ctrl/Cmd+K focuses it. It only reads the tab bar and clicks tab buttons.
SHELL_JS = """() => {
  if (window.__itsmShell) return; window.__itsmShell = true;
  var PAGES = {
    '0': ['Analyze / Overview', 'Overview', 'Upload data, run the analysis, then explore the results.'],
    '1': ['Analyze / Categorization', 'Categorization', 'Every ticket validated, categorized and scored.'],
    '2': ['Analyze / Trends & Insights', 'Trends & Insights', 'Patterns over time, recurring issues and the servers behind them.'],
    '3': ['Analyze / Recommendations', 'Recommendations', 'Evidence-based next actions for this batch.'],
    '4': ['Knowledge / Knowledge Base', 'Knowledge Base', 'Runbooks, SOPs and guides the agent can cite.'],
    '5': ['Knowledge / Q&A', 'Q&A (Agent)', 'Ask about your incidents and your documented procedures.'],
    '6': ['System / Settings', 'Settings', 'Analysis thresholds and defaults.'],
    '7': ['System / Observability', 'Observability', 'Usage, latency and quality of the agent.'],
    '8': ['System / Admin', 'Access control', '']
  };
  function setText(id, t) { var e = document.getElementById(id); if (e && e.textContent !== t) e.textContent = t; }
  var queued = false;
  function sync() {
    queued = false;
    var sel = document.querySelector('#main-tabs button[role="tab"].selected'); if (!sel) return;
    var id = sel.getAttribute('data-tab-id') || '0';
    var p = (PAGES[id] || ['', sel.textContent.trim(), '']).slice();
    var viewer = !!document.querySelector('#user-chip .uc-role-viewer');
    if (id === '0' && viewer) p[0] = 'Workspace / Overview';
    if (id === '5') { var h = document.querySelector('.qa-head');
      if (h) { var c = h.querySelector('.qa-crumb'), t = h.querySelector('h2'), s = h.querySelector('p');
        p = [c ? c.textContent : p[0], t ? t.textContent : p[1], s ? s.textContent : '']; } }
    setText('pg-crumb', p[0]); setText('pg-title', p[1]); setText('pg-sub', p[2]);
    document.body.setAttribute('data-pg', id);
  }
  function queue() { if (!queued) { queued = true; requestAnimationFrame(sync); } }
  new MutationObserver(queue).observe(document.body, {subtree: true, childList: true, attributes: true, attributeFilter: ['class']});
  sync();
  var inp = document.getElementById('nav-search');
  function tabs() { return Array.prototype.slice.call(document.querySelectorAll('#main-tabs .tab-container button[role="tab"]')); }
  function filter() {
    var q = (inp.value || '').trim().toLowerCase();
    tabs().forEach(function (b) {
      if (!q || b.textContent.toLowerCase().indexOf(q) > -1) b.style.removeProperty('display');
      else b.style.setProperty('display', 'none', 'important');
    });
  }
  function reset() { if (inp) { inp.value = ''; filter(); } }
  if (inp) {
    inp.addEventListener('input', filter);
    inp.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { var m = tabs().filter(function (b) { return b.style.display !== 'none' && !b.disabled; })[0];
        if (m) { m.click(); reset(); inp.blur(); } }
      else if (e.key === 'Escape') { reset(); inp.blur(); }
    });
    inp.addEventListener('blur', function () { setTimeout(reset, 150); });
  }
  document.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && String(e.key).toLowerCase() === 'k' && inp) { e.preventDefault(); inp.focus(); inp.select(); }
  });
}"""
