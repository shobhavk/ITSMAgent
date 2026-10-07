"""Small text/data helpers shared across the UI (formatting, DataFrame preparation)."""
import pandas as pd

from ui.constants import ALL_COLUMNS, _TAG_RE, _WS_RE


def _score_badge(score: int) -> str:
    if score >= 75:
        return "🟢 Good"
    if score >= 50:
        return "🟡 Needs improvement"
    return "🔴 Poor"


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
                "External Info": r.external_info,
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
                "Created At": r.created_at,
                "Resolved At": r.resolved_at,
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


def _summary_markdown(stats: dict) -> str:
    return (
        f"### Summary\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total records seen | {stats['total_records']} |\n"
        f"| Valid records analyzed | {stats['valid_records']} |\n"
        f"| Rejected records | {stats['rejected_records']} |\n"
        f"| Average worklog score | {stats['average_worklog_score']} / 100 |\n"
    )


def _analysis_to_summary_stats(analysis) -> dict:
    return {
        "total_records": analysis.total_records,
        "valid_records": analysis.valid_records,
        "rejected_records": analysis.rejected_records,
        "average_worklog_score": analysis.average_worklog_score,
    }


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


def _fmt_ts(value) -> str:
    """Timestamp -> short, table-friendly string ('' when missing)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value)
