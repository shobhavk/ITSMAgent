"""
Pure, deterministic tool-backing functions for the Q&A agent.

See rag.py for how these get wrapped as LangChain tools and bound to the
chat model - this module has NO LangChain/agent-framework dependency on
purpose, so every function here is independently callable/testable, and
reusable from both the Gradio chat tab and the REST /api/v1/chat endpoint
(see tickets_to_dataframe() below for how the API's AnalyzedTicket list
gets normalized into the same column shape the Gradio dataframe already
uses, so every function below works identically for both callers).

Design principles this module exists to enforce:
- Every function takes the existing analysis dataframe (same column
  shape used by the dashboard/CSV export) and returns a SMALL,
  JSON-serializable dict - never the dataframe itself, never more than a
  capped sample of rows (see search_incidents' `limit`). The agent's
  system prompt (rag.py) is told never to state a number that didn't
  come from one of these; this module is how that's enforced on the data
  side - there is no path from here back to the raw dataframe reaching
  the LLM.
- Every function is wrapped in try/except and returns {"error": "..."}
  rather than raising, so a malformed/partial dataframe degrades to an
  honest "couldn't compute this" instead of crashing the agent turn -
  consistent with trend_metrics.py/recurring_issues.py's existing
  "report unavailable, never fabricate" philosophy, which get_incident_summary
  and get_recurring_issues below wrap rather than duplicate.
- Missing/unavailable data (e.g. no resolution timestamps) is reported
  as such, not silently omitted or guessed at - same rule.

Note on _priority_counts-style logic: ui/gradio_app.py has its own tiny
`_priority_counts()`/`_score_badge()` helpers for the dashboard. This
module intentionally does NOT import them - app/services/ must not depend
on ui/ (the REST API path has no UI layer at all). The one-line pandas
idioms below are kept in sync with those by convention, not by import;
POOR_WORKLOG_THRESHOLD documents exactly which gradio_app.py constant it
mirrors.
"""
import logging

import pandas as pd

from app.services import recurring_issues, trend_metrics

logger = logging.getLogger(__name__)

# Mirrors the "Poor" band in ui/gradio_app.py's _score_badge() (score < 50
# -> "🔴 Poor"). Kept here as a separate constant rather than an import
# because app/services/ must not depend on ui/ - see module docstring.
POOR_WORKLOG_THRESHOLD = 50

DEFAULT_TOP_N = 10
DEFAULT_SEARCH_LIMIT = 20


def tickets_to_dataframe(tickets: list) -> pd.DataFrame:
    """Converts a list of AnalyzedTicket (pydantic - the REST API's
    result shape) into the same column names ui/gradio_app.py's full_df
    already uses, so every function below works identically regardless
    of which surface (Gradio or the API) is calling it."""
    rows = [
        {
            "Ticket ID": t.ticket_id,
            "Category": t.category,
            "Priority": t.priority or "",
            "Status": t.status or "",
            "Assignment Group": t.assignment_group or "",
            "Host / CI": t.host or "",
            "Worklog Score": t.worklog_score,
            "Short Description": t.short_description,
            "Description": t.description,
            "Worklog Notes": t.worklog,
            "Opened At": t.opened_at,
            "Closed At": t.closed_at,
            "Created At": t.created_at,
            "Resolved At": t.resolved_at,
        }
        for t in tickets
    ]
    return pd.DataFrame(rows)


def _no_data_error() -> dict:
    return {"error": "No analyzed tickets are available yet - run an analysis first."}


def _to_native(value):
    """Recursively converts numpy/pandas scalar types (int64, float64, ...)
    to plain Python types. Every tool result below passes through this
    before returning, so json.dumps() in rag.py's tool wrapper never
    chokes on a numpy type leaking out of a pandas aggregation."""
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def get_incident_summary(full_df: pd.DataFrame) -> dict:
    """Total incidents, P1/P2 counts, average resolution time (MTTR),
    average worklog score, and the percentage of tickets with a "poor"
    worklog score. Reuses trend_metrics.compute_resolution_metrics for
    the resolution-time figure rather than recomputing it."""
    try:
        if full_df is None or len(full_df) == 0:
            return _no_data_error()

        total = int(len(full_df))

        priority = full_df.get("Priority", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
        p1_count = int(priority.str.contains("P1|CRITICAL", regex=True).sum())
        p2_count = int(priority.str.contains("P2|HIGH", regex=True).sum())

        scores = pd.to_numeric(full_df.get("Worklog Score"), errors="coerce")
        has_scores = scores.notna().any()
        avg_score = round(float(scores.mean()), 1) if has_scores else None
        poor_pct = (
            round(100.0 * (scores < POOR_WORKLOG_THRESHOLD).sum() / scores.notna().sum(), 1)
            if has_scores
            else None
        )

        mttr = trend_metrics.compute_resolution_metrics(full_df)["mttr"]

        return _to_native({
            "total_incidents": total,
            "p1_incidents": p1_count,
            "p2_incidents": p2_count,
            "average_resolution_time_hours": mttr["value_hours"] if mttr["available"] else None,
            "average_resolution_time_note": "" if mttr["available"] else mttr["note"],
            "average_worklog_score": avg_score,
            "poor_worklog_percentage": poor_pct,
        })
    except Exception as exc:
        logger.warning("get_incident_summary failed: %s", exc)
        return {"error": f"Could not compute the incident summary: {exc}"}


def get_category_analysis(full_df: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> dict:
    """Incident count by category, plus the top N categories."""
    try:
        if full_df is None or len(full_df) == 0 or "Category" not in full_df.columns:
            return _no_data_error()
        counts = full_df["Category"].fillna("Unknown").replace("", "Unknown").value_counts()
        return _to_native({
            "total_categories": int(counts.shape[0]),
            "counts_by_category": {str(k): int(v) for k, v in counts.items()},
            "top_categories": [{"category": str(c), "count": int(n)} for c, n in counts.head(top_n).items()],
        })
    except Exception as exc:
        logger.warning("get_category_analysis failed: %s", exc)
        return {"error": f"Could not compute category analysis: {exc}"}


def get_priority_analysis(full_df: pd.DataFrame) -> dict:
    """Incident count by priority, plus the percentage distribution."""
    try:
        if full_df is None or len(full_df) == 0 or "Priority" not in full_df.columns:
            return _no_data_error()
        series = full_df["Priority"].fillna("").astype(str).str.strip().replace("", "Unspecified")
        counts = series.value_counts()
        total = int(counts.sum())
        distribution_pct = {str(k): round(100.0 * v / total, 1) for k, v in counts.items()} if total else {}
        return _to_native({
            "counts_by_priority": {str(k): int(v) for k, v in counts.items()},
            "distribution_pct": distribution_pct,
        })
    except Exception as exc:
        logger.warning("get_priority_analysis failed: %s", exc)
        return {"error": f"Could not compute priority analysis: {exc}"}


def get_server_analysis(full_df: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> dict:
    """Incident count by server/host, plus the top N most-affected servers."""
    try:
        if full_df is None or len(full_df) == 0 or "Host / CI" not in full_df.columns:
            return _no_data_error()
        hosts = full_df["Host / CI"].fillna("").astype(str).str.strip()
        hosts = hosts[hosts != ""]
        if hosts.empty:
            return {"error": "No tickets in this batch have a detected server/host."}
        counts = hosts.value_counts()
        return _to_native({
            "counts_by_server": {str(k): int(v) for k, v in counts.items()},
            "top_servers": [{"server": str(s), "count": int(n)} for s, n in counts.head(top_n).items()],
        })
    except Exception as exc:
        logger.warning("get_server_analysis failed: %s", exc)
        return {"error": f"Could not compute server analysis: {exc}"}


def get_recurring_issues(full_df: pd.DataFrame, threshold: int = recurring_issues.DEFAULT_RECURRENCE_THRESHOLD) -> dict:
    """Recurring issues (exact-match: same host+category recurring at
    least `threshold` times), their frequency, and affected servers.
    Thin wrapper over recurring_issues.detect_exact_recurrence - the
    same function the dashboard's "Recurring Issues" panel already
    calls, so the agent's answer and the dashboard table can never
    disagree."""
    try:
        rows = recurring_issues.detect_exact_recurrence(full_df, threshold=threshold)
        if not rows:
            return {"recurring_issues": [], "note": f"No issue recurred {threshold}+ times in this batch."}
        return _to_native({
            "recurring_issues": [
                {
                    "server": r["host"] or "(no host detected)",
                    "category": r["category"],
                    "frequency": r["count"],
                    "first_seen": r["first_seen"],
                    "last_seen": r["last_seen"],
                    "avg_days_between_occurrences": r["avg_interval_days"],
                }
                for r in rows
            ]
        })
    except Exception as exc:
        logger.warning("get_recurring_issues failed: %s", exc)
        return {"error": f"Could not compute recurring issues: {exc}"}


def search_incidents(
    full_df: pd.DataFrame,
    incident_id: str = "",
    server: str = "",
    category: str = "",
    priority: str = "",
    keyword: str = "",
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict:
    """Filters the analyzed tickets by any combination of incident ID,
    server, category, priority, and/or a free-text keyword (matched
    against short description, description, and worklog notes). Returns
    the true total match count plus a capped sample - never the full
    matching set - so a broad search still can't leak the whole
    dataframe to the LLM."""
    try:
        if full_df is None or len(full_df) == 0:
            return _no_data_error()

        d = full_df
        if incident_id:
            d = d[d["Ticket ID"].astype(str).str.contains(incident_id, case=False, na=False, regex=False)]
        if server and "Host / CI" in d.columns:
            d = d[d["Host / CI"].astype(str).str.contains(server, case=False, na=False, regex=False)]
        if category and "Category" in d.columns:
            d = d[d["Category"].astype(str).str.contains(category, case=False, na=False, regex=False)]
        if priority and "Priority" in d.columns:
            d = d[d["Priority"].astype(str).str.contains(priority, case=False, na=False, regex=False)]
        if keyword:
            text_cols = [c for c in ["Short Description", "Description", "Worklog Notes"] if c in d.columns]
            if text_cols:
                mask = pd.Series(False, index=d.index)
                for col in text_cols:
                    mask = mask | d[col].astype(str).str.contains(keyword, case=False, na=False, regex=False)
                d = d[mask]

        total_matching = int(len(d))
        sample_cols = [c for c in ["Ticket ID", "Category", "Priority", "Status", "Host / CI", "Worklog Score"] if c in d.columns]
        results = d[sample_cols].head(limit).to_dict("records") if sample_cols else []
        return _to_native({"total_matching": total_matching, "results": results})
    except Exception as exc:
        logger.warning("search_incidents failed: %s", exc)
        return {"error": f"Could not search incidents: {exc}"}
