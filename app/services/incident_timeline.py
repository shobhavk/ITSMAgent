"""
Per-incident timeline audit for the "Trends & Insights" tab.

For every ticket this reconstructs a simple lifecycle view:
  - when it came in (effective open time - see trend_metrics.py)
  - how long it took to acknowledge (MTTA for that one ticket)
  - whether the raw "External Info" text shows the ticket being
    assigned/reassigned to someone else, and when
  - whether "External Info" was updated with a proper timestamp trail at
    all, or is missing/untimed

Nothing here invents a number: every field is either a real timestamp
from the upload or an explicit "not found in this data" flag, matching
the MTTA/MTTD "report unavailable, never fabricate" approach already
used in trend_metrics.py. The reassignment/timestamp detection is a
best-effort regex pass over free text - it is a *data quality signal*
("does this incident have a legible audit trail"), not a guaranteed
extraction, and is presented that way.
"""
import re

import pandas as pd

from app.services.trend_metrics import _opened_closed_series

# Broad enough to catch the common ITSM export formats (ISO, US slash
# dates, "01-Aug-2026") without trying to be a general date parser -
# pd.to_datetime(..., errors="coerce") does the actual parsing below, this
# regex just finds the substrings worth handing to it.
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}(?::\d{2})?"   # 2026-08-01 09:15
    r"|\d{1,2}/\d{1,2}/\d{2,4}[ ,]+\d{1,2}:\d{2}"       # 08/01/2026 09:15
    r"|\d{1,2}-[A-Za-z]{3,9}-\d{2,4}[ ,]*\d{1,2}:\d{2}" # 01-Aug-2026 09:15
    r"|\d{4}-\d{2}-\d{2}"                                # 2026-08-01
)

_ASSIGNMENT_RE = re.compile(
    # Scoped (?i:...) so only the "assigned/reassigned to" keyword is
    # case-insensitive - the name group must still start with a capital
    # letter, or a lower-case word like "at"/"the" following the keyword
    # gets swept in as a fake "assignee".
    r"(?i:re[- ]?assign(?:ed|ment)?|assign(?:ed)?)\s*(?:(?i:to)|:)?\s*"
    r"([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,2})"
)


def _extract_assignment_events(text: str) -> list[dict]:
    """Best-effort scan of External Info for "assigned/reassigned to
    <name>" mentions, paired with the nearest timestamp in the same
    sentence/line if one is present. Returns [] for blank text or no
    match - callers treat that as "no reassignment visible in the data",
    not "no reassignment happened"."""
    if not text:
        return []
    events = []
    for segment in re.split(r"(?<=[.;\n])\s+", text):
        match = _ASSIGNMENT_RE.search(segment)
        if not match:
            continue
        ts_match = _TIMESTAMP_RE.search(segment)
        events.append(
            {
                "assignee": match.group(1).strip(),
                "timestamp": ts_match.group(0) if ts_match else None,
                "note": segment.strip()[:160],
            }
        )
    return events


def _external_info_quality(text: str, events: list[dict]) -> str:
    if not text or not text.strip():
        return "Missing"
    if any(e["timestamp"] for e in events):
        return "Good - timestamped"
    if _TIMESTAMP_RE.search(text):
        return "Present - has timestamp(s)"
    return "Present - no timestamp"


def compute_incident_timeline(full_df: pd.DataFrame) -> pd.DataFrame:
    """One row per ticket: effective opened time, time-to-acknowledge
    (hours), assignment group, any reassignment events detected in
    External Info, and the External Info quality flag above. Returns an
    empty dataframe (not None) if full_df has no usable rows, so callers
    can treat "no data yet" and "nothing found" the same way."""
    columns = [
        "Ticket ID", "Opened/Created At", "Acknowledged At", "Time to Acknowledge (h)",
        "Assignment Group", "Reassignment Detected", "Reassigned To", "Reassigned At",
        "External Info Quality", "Resolved At",
    ]
    if full_df is None or len(full_df) == 0:
        return pd.DataFrame(columns=columns)

    ts = _opened_closed_series(full_df)
    external_info = full_df.get("External Info", pd.Series([""] * len(full_df), index=full_df.index))
    external_info = external_info.fillna("").astype(str)

    ack_hours = (ts["responded_at"] - ts["opened_at"]).dt.total_seconds() / 3600.0

    rows = []
    for idx in full_df.index:
        events = _extract_assignment_events(external_info.loc[idx])
        first_event = events[0] if events else None
        rows.append(
            {
                "Ticket ID": full_df.at[idx, "Ticket ID"] if "Ticket ID" in full_df.columns else str(idx),
                "Opened/Created At": ts["opened_at"].loc[idx],
                "Acknowledged At": ts["responded_at"].loc[idx],
                "Time to Acknowledge (h)": (
                    round(ack_hours.loc[idx], 1)
                    if pd.notna(ts["opened_at"].loc[idx]) and pd.notna(ts["responded_at"].loc[idx])
                    else None
                ),
                "Assignment Group": full_df.at[idx, "Assignment Group"] if "Assignment Group" in full_df.columns else "",
                "Reassignment Detected": bool(events),
                "Reassigned To": first_event["assignee"] if first_event else "",
                "Reassigned At": first_event["timestamp"] if first_event else "",
                "External Info Quality": _external_info_quality(external_info.loc[idx], events),
                "Resolved At": ts["closed_at"].loc[idx],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_timeline_aggregate(timeline_df: pd.DataFrame) -> dict:
    """Small aggregated dict for the KPI cards and the LLM summary payload
    - counts and percentages only, never per-ticket text, matching how
    overview_metrics/recommendations already keep LLM payloads small."""
    if timeline_df is None or len(timeline_df) == 0:
        return {"available": False, "note": "Run an analysis first."}

    total = len(timeline_df)
    ack_known = timeline_df["Time to Acknowledge (h)"].dropna()
    reassigned = timeline_df[timeline_df["Reassignment Detected"]]
    quality_counts = timeline_df["External Info Quality"].value_counts().to_dict()

    slowest = (
        timeline_df.dropna(subset=["Time to Acknowledge (h)"])
        .sort_values("Time to Acknowledge (h)", ascending=False)
        .head(5)[["Ticket ID", "Time to Acknowledge (h)"]]
        .to_dict("records")
    )

    return {
        "available": True,
        "total_incidents": total,
        "acknowledgment_data_available_for": int(len(ack_known)),
        "avg_time_to_acknowledge_hours": round(float(ack_known.mean()), 1) if len(ack_known) else None,
        "max_time_to_acknowledge_hours": round(float(ack_known.max()), 1) if len(ack_known) else None,
        "incidents_with_reassignment_detected": int(len(reassigned)),
        "external_info_quality_breakdown": quality_counts,
        "slowest_to_acknowledge": slowest,
    }


def build_llm_payload(aggregate: dict) -> dict:
    """Strips the "available" flag before handing the aggregate to the
    LLM - it's a routing flag for our own code, not something the model
    should comment on."""
    return {k: v for k, v in aggregate.items() if k != "available"}


def fallback_summary_markdown(aggregate: dict) -> str:
    """Deterministic, rule-based rendering of the same aggregate used for
    the LLM prompt - shown when no LLM is available, and identical in
    substance to what the LLM would be asked to re-voice."""
    if not aggregate or not aggregate.get("available"):
        return aggregate.get("note", "Run an analysis first, then generate the summary.") if aggregate else \
            "Run an analysis first, then generate the summary."

    lines = [f"### Incident Timeline Summary\n", f"- **{aggregate['total_incidents']}** incidents in this batch."]

    if aggregate["avg_time_to_acknowledge_hours"] is not None:
        lines.append(
            f"- Average time to acknowledge: **{aggregate['avg_time_to_acknowledge_hours']}h** "
            f"(based on {aggregate['acknowledgment_data_available_for']} ticket(s) with an "
            f"acknowledgment timestamp); slowest was **{aggregate['max_time_to_acknowledge_hours']}h**."
        )
    else:
        lines.append("- No acknowledgment/first-response timestamp found in this data - MTTA per ticket is unavailable.")

    lines.append(
        f"- **{aggregate['incidents_with_reassignment_detected']}** incident(s) show a "
        "reassignment to another person in their External Info text."
    )

    quality = aggregate.get("external_info_quality_breakdown") or {}
    if quality:
        quality_parts = ", ".join(f"{k}: {v}" for k, v in quality.items())
        lines.append(f"- External Info audit-trail quality across the batch - {quality_parts}.")

    slowest = aggregate.get("slowest_to_acknowledge") or []
    if slowest:
        lines.append("\n**Slowest to acknowledge:**")
        for item in slowest:
            lines.append(f"- {item['Ticket ID']}: {item['Time to Acknowledge (h)']}h")

    return "\n".join(lines)
