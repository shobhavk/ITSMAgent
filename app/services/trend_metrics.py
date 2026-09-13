"""
Trend and resolution-metric calculations for the "Trend Analysis" tab.

Two families of numbers here, and they have very different reliability:

1. Time-bucketed KPIs (daily/weekly/monthly ticket volume + average worklog
   score) - always computable from "Opened At", which every ticket has a
   good chance of carrying (it's just the creation date). Straightforward
   pandas groupby, no assumptions.

2. Resolution metrics - MTTD, MTTA, MTTR, SLA compliance - are NOT equally
   available:
     - MTTR (mean time to resolve) = closed_at - opened_at. Computable for
       any ticket that has both, which is most ITSM exports.
     - MTTA (mean time to acknowledge) = responded_at - opened_at. Needs a
       "first response"/"acknowledged at" column, which many exports don't
       have.
     - MTTD (mean time to detect) = opened_at - detected_at. Needs a true
       detection/alert timestamp separate from ticket creation - this is
       usually a monitoring-system field, not a ticket field, so most
       ITSM exports will NOT have it.
   Rather than substituting a proxy and presenting it as the real thing,
   each metric below reports `available: False` with an explanatory note
   when its required column wasn't present in the uploaded data, so the
   UI can show "N/A - no detection timestamp in this data" instead of a
   fabricated number. See TicketRecord/AnalyzedTicket in schemas.py for
   which source columns map to which of these.

SLA compliance is the one exception that doesn't need a dedicated
timestamp column: it's evaluated against a per-priority target resolution
time (SLA_TARGET_HOURS below), using the same opened_at/closed_at pair
MTTR already uses. Adjust SLA_TARGET_HOURS to match your organization's
actual SLA policy - the defaults here are common ITSM conventions, not
universal.
"""
import pandas as pd

# Target resolution time per priority, in hours. Tune to your org's real
# SLA policy - these are reasonable ITSM defaults, not a standard.
SLA_TARGET_HOURS: dict[str, float] = {
    "P1": 4, "CRITICAL": 4,
    "P2": 8, "HIGH": 8,
    "P3": 24, "MEDIUM": 24,
    "P4": 72, "LOW": 72,
}
DEFAULT_SLA_TARGET_HOURS = 48  # fallback for unrecognized/missing priority

_GRANULARITY_FREQ = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
_GRANULARITY_LABEL_FMT = {"Daily": "%Y-%m-%d", "Weekly": "%Y-%m-%d", "Monthly": "%Y-%m"}


def _priority_bucket(priority: str) -> str:
    p = (priority or "").strip().upper()
    for key in SLA_TARGET_HOURS:
        if key in p:
            return key
    return ""


def _opened_closed_series(full_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with parsed opened_at/closed_at datetime columns
    (NaT where missing/unparsable), from whatever the display column names
    are ("Opened At" / "Closed At")."""
    if full_df is None or len(full_df) == 0:
        return pd.DataFrame(columns=["opened_at", "closed_at", "responded_at", "detected_at", "priority"])
    out = pd.DataFrame(
        {
            "opened_at": pd.to_datetime(full_df.get("Opened At"), errors="coerce"),
            "closed_at": pd.to_datetime(full_df.get("Closed At"), errors="coerce"),
            "responded_at": pd.to_datetime(full_df.get("Responded At"), errors="coerce"),
            "detected_at": pd.to_datetime(full_df.get("Detected At"), errors="coerce"),
            "priority": full_df.get("Priority", ""),
        }
    )
    return out


def compute_time_series(full_df: pd.DataFrame, granularity: str = "Daily") -> list[dict]:
    """Ticket volume + average worklog score, bucketed by day/week/month
    on Opened At. Returns [] if there's no usable Opened At data at all -
    the UI should treat that as "insufficient data", not an empty period."""
    if full_df is None or len(full_df) == 0 or "Opened At" not in full_df.columns:
        return []

    opened = pd.to_datetime(full_df["Opened At"], errors="coerce")
    scores = pd.to_numeric(full_df.get("Worklog Score"), errors="coerce")
    valid = opened.notna()
    if not valid.any():
        return []

    freq = _GRANULARITY_FREQ.get(granularity, "D")
    label_fmt = _GRANULARITY_LABEL_FMT.get(granularity, "%Y-%m-%d")

    frame = pd.DataFrame({"opened_at": opened[valid], "worklog_score": scores[valid]})
    period = frame["opened_at"].dt.to_period(freq)
    grouped = frame.groupby(period).agg(ticket_count=("opened_at", "count"), avg_worklog_score=("worklog_score", "mean"))
    grouped = grouped.sort_index()

    return [
        {
            "period": idx.start_time.strftime(label_fmt),
            "ticket_count": int(row["ticket_count"]),
            "avg_worklog_score": round(row["avg_worklog_score"], 1) if pd.notna(row["avg_worklog_score"]) else None,
        }
        for idx, row in grouped.iterrows()
    ]


def _duration_hours(start: pd.Series, end: pd.Series) -> pd.Series:
    both_present = start.notna() & end.notna()
    delta = (end - start).dt.total_seconds() / 3600.0
    return delta[both_present]


def _metric_from_durations(durations: pd.Series, unavailable_note: str) -> dict:
    if durations is None or len(durations) == 0:
        return {"available": False, "value_hours": None, "sample_size": 0, "note": unavailable_note}
    return {
        "available": True,
        "value_hours": round(float(durations.mean()), 1),
        "sample_size": int(len(durations)),
        "note": "",
    }


def compute_resolution_metrics(full_df: pd.DataFrame) -> dict:
    """Returns {"mttd": {...}, "mtta": {...}, "mttr": {...}, "sla": {...}}.
    Each of mttd/mtta/mttr has: available, value_hours, sample_size, note.
    sla has: available, compliance_pct, sample_size, by_priority, note."""
    ts = _opened_closed_series(full_df)

    mttd_durations = _duration_hours(ts["detected_at"], ts["opened_at"])
    mtta_durations = _duration_hours(ts["opened_at"], ts["responded_at"])
    mttr_durations = _duration_hours(ts["opened_at"], ts["closed_at"])

    mttd = _metric_from_durations(
        mttd_durations, "No detection timestamp found in this data - MTTD needs a monitoring/alert time distinct from ticket creation."
    )
    mtta = _metric_from_durations(
        mtta_durations, "No acknowledgment/first-response timestamp found in this data."
    )
    mttr = _metric_from_durations(
        mttr_durations, "No resolved/closed timestamp found in this data."
    )

    # SLA compliance reuses the same opened/closed pair as MTTR, evaluated
    # per-ticket against SLA_TARGET_HOURS for that ticket's priority.
    resolved_mask = ts["opened_at"].notna() & ts["closed_at"].notna()
    if not resolved_mask.any():
        sla = {"available": False, "compliance_pct": None, "sample_size": 0, "by_priority": {}, "note": mttr["note"]}
    else:
        resolved = ts[resolved_mask].copy()
        resolved["duration_hours"] = (resolved["closed_at"] - resolved["opened_at"]).dt.total_seconds() / 3600.0
        resolved["priority_bucket"] = resolved["priority"].apply(_priority_bucket)
        resolved["target_hours"] = resolved["priority_bucket"].map(SLA_TARGET_HOURS).fillna(DEFAULT_SLA_TARGET_HOURS)
        resolved["met_sla"] = resolved["duration_hours"] <= resolved["target_hours"]

        by_priority = {}
        for bucket, group in resolved.groupby(resolved["priority"].fillna("Unspecified").replace("", "Unspecified")):
            by_priority[bucket] = {
                "compliance_pct": round(100.0 * group["met_sla"].mean(), 1),
                "sample_size": int(len(group)),
                "target_hours": float(group["target_hours"].iloc[0]),
            }

        sla = {
            "available": True,
            "compliance_pct": round(100.0 * resolved["met_sla"].mean(), 1),
            "sample_size": int(len(resolved)),
            "by_priority": by_priority,
            "note": "",
        }

    return {"mttd": mttd, "mtta": mtta, "mttr": mttr, "sla": sla}
