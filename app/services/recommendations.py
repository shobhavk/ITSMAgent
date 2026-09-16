"""
Data-backed ITSM recommendations.

This module answers "so what should we actually do about it?" on top of
the analytics that already exist - it deliberately does NOT recompute
anything chat_tools.py, recurring_issues.py, or trend_metrics.py already
compute correctly. Every number that ends up in a recommendation comes
from one of those existing functions (or from a small pandas aggregation
here for the per-group cuts none of them expose yet), so the
Recommendations tab, the Overview KPIs, the Trends tab, and the chat
agent can never disagree on a figure.

Two strictly separated layers, which is the whole point of the design:

  1. FACTS (compute_recommendation_facts) - pure pandas/Python. Produces
     a small, JSON-serializable dict of aggregates: recurrence groups,
     priority mix, SLA compliance, per-category/server/assignment-group
     resolution times, worklog-quality cuts, host and category volumes.
     No LLM involved, so these numbers are reproducible and auditable.

  2. RULES (generate_recommendations) - deterministic thresholding over
     those facts, producing fully-formed recommendations with the
     evidence sentence already written from the computed numbers. This
     layer ALWAYS runs and is always what the dashboard renders, so the
     feature works with no LLM configured at all.

An optional third step lives in ui/gradio_app.py: the LLM is handed
build_llm_payload(...) - the facts and the already-written
recommendations, never the incident dataframe, never ticket
descriptions or worklog text - and asked only to re-voice them for a
management audience. It cannot introduce a number that isn't in the
payload because it never sees the rows those numbers came from, and the
deterministic text below is what's shown if that call fails.

Thresholds (RECOMMENDATION_THRESHOLDS) are reasonable defaults, not a
standard - tune them the same way overview_metrics.HEALTH_THRESHOLDS and
trend_metrics.SLA_TARGET_HOURS are meant to be tuned. Where a threshold
mirrors one that already exists elsewhere, that's noted inline.

Every public function is wrapped in try/except and degrades to an empty/
"unavailable" result rather than raising, consistent with the rest of
app/services/.
"""
import logging

import pandas as pd

from app.services import chat_tools, recurring_issues, trend_metrics

logger = logging.getLogger(__name__)

RECOMMENDATION_THRESHOLDS = {
    # --- Recurrence ---
    # Mirrors recurring_issues.DEFAULT_RECURRENCE_THRESHOLD as the floor;
    # anything at/above "high" is escalated to a High-attention item.
    "recurrence_min_count": recurring_issues.DEFAULT_RECURRENCE_THRESHOLD,
    "recurrence_high_count": 5,
    "recurrence_top_n": 5,
    # --- Priority mix --- (mirrors overview_metrics.HEALTH_THRESHOLDS)
    "high_priority_pct_warning": 20.0,
    "high_priority_pct_critical": 40.0,
    # --- SLA ---
    "sla_warning_pct": 90.0,
    "sla_critical_pct": 70.0,
    # --- Resolution performance ---
    # A group is "slow" only if it has enough resolved tickets to mean
    # anything AND is materially worse than this batch's own average -
    # a relative comparison, so it adapts to the dataset instead of
    # asserting an absolute target the organization may not share.
    "slow_group_min_sample": 3,
    "slow_group_multiplier": 1.5,
    "slow_group_min_hours": 8.0,
    "slow_group_top_n": 3,
    # --- Worklog quality --- (mirrors overview_metrics.HEALTH_THRESHOLDS)
    "poor_worklog_pct_warning": 20.0,
    "poor_worklog_pct_critical": 40.0,
    "worklog_group_min_sample": 3,
    "worklog_group_poor_pct": 40.0,
    "worklog_group_top_n": 3,
    # --- Volume concentration ---
    "server_min_incidents": 5,
    "server_share_pct_warning": 15.0,
    "server_top_n": 3,
    "category_share_pct_warning": 30.0,
    "category_top_n": 3,
    # Hard ceiling on the whole panel. A pathological batch (one host
    # causing everything) can legitimately trip every rule in every area
    # at once; the list is sorted most-urgent-first before this cap
    # applies, so what gets dropped is always the least urgent.
    "max_recommendations": 15,
}

# Attention levels, most urgent first. Used for sorting and for the
# badge colour in the UI.
ATTENTION_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

# The six analysis areas, in the order they're presented.
AREA_RECURRING = "Recurring Issues"
AREA_PRIORITY = "High-Priority Incidents"
AREA_RESOLUTION = "Resolution Performance"
AREA_WORKLOG = "Worklog Quality"
AREA_SERVERS = "Server / Host Patterns"
AREA_CATEGORIES = "Category Patterns"


def _recommendation(area, observation, evidence, recommendation, attention, benefit, metrics=None) -> dict:
    """One recommendation in the fixed five-part shape the UI renders:
    Observation / Evidence / Recommendation / Attention level / Expected
    benefit. `metrics` carries the raw numbers behind the evidence
    sentence so the LLM re-voicing step (and any future export) has them
    as data, not just prose."""
    return {
        "area": area,
        "observation": observation,
        "evidence": evidence,
        "recommendation": recommendation,
        "attention": attention,
        "expected_benefit": benefit,
        "metrics": metrics or {},
    }


# --------------------------------------------------------------------------
# Layer 1: facts (pure pandas - no LLM, no prose)
# --------------------------------------------------------------------------

def _group_resolution_hours(full_df: pd.DataFrame, column: str, min_sample: int) -> dict:
    """Average resolution time per value of `column` (Category, Host / CI,
    Assignment Group), using the SAME effective open/resolve pair
    trend_metrics.compute_resolution_metrics uses for MTTR - so a group
    average here is directly comparable to the MTTR shown on the Trends
    tab rather than being a second, subtly different calculation.

    Returns {"overall_avg_hours": float|None, "groups": [...]} where
    overall_avg_hours is computed over exactly the rows that survived
    filtering, so the "1.8x the batch average" comparisons below are
    like-for-like. Groups with fewer than `min_sample` resolved tickets
    are dropped - a single slow ticket is not a performance signal."""
    try:
        if full_df is None or len(full_df) == 0 or column not in full_df.columns:
            return {"overall_avg_hours": None, "groups": []}

        effective = trend_metrics.effective_open_resolve_times(full_df)
        hours = (effective["end"] - effective["start"]).dt.total_seconds() / 3600.0

        frame = pd.DataFrame({
            "group": full_df[column].fillna("").astype(str).str.strip(),
            "hours": hours,
        })
        frame = frame[(frame["group"] != "") & frame["hours"].notna() & (frame["hours"] >= 0)]
        if frame.empty:
            return {"overall_avg_hours": None, "groups": []}

        overall = round(float(frame["hours"].mean()), 1)
        agg = frame.groupby("group")["hours"].agg(["mean", "count"])
        agg = agg[agg["count"] >= min_sample]
        groups = [
            {"name": str(name), "avg_hours": round(float(row["mean"]), 1), "count": int(row["count"])}
            for name, row in agg.iterrows()
        ]
        groups.sort(key=lambda g: g["avg_hours"], reverse=True)
        return {"overall_avg_hours": overall, "groups": groups}
    except Exception as exc:
        logger.warning("_group_resolution_hours(%s) failed: %s", column, exc)
        return {"overall_avg_hours": None, "groups": []}


def _group_worklog_quality(full_df: pd.DataFrame, column: str, min_sample: int) -> list:
    """Share of "poor" worklogs per value of `column`. Uses
    chat_tools.POOR_WORKLOG_THRESHOLD so the definition of "poor" here is
    the same one the Overview KPI, the chat agent, and the dashboard's
    score badge all use."""
    try:
        if full_df is None or len(full_df) == 0 or column not in full_df.columns:
            return []

        scores = pd.to_numeric(full_df.get("Worklog Score"), errors="coerce")
        frame = pd.DataFrame({
            "group": full_df[column].fillna("").astype(str).str.strip(),
            "score": scores,
        })
        frame = frame[(frame["group"] != "") & frame["score"].notna()]
        if frame.empty:
            return []

        frame["is_poor"] = frame["score"] < chat_tools.POOR_WORKLOG_THRESHOLD
        agg = frame.groupby("group").agg(
            count=("score", "count"), avg_score=("score", "mean"), poor=("is_poor", "sum")
        )
        agg = agg[agg["count"] >= min_sample]
        rows = [
            {
                "name": str(name),
                "count": int(row["count"]),
                "avg_score": round(float(row["avg_score"]), 1),
                "poor_count": int(row["poor"]),
                "poor_pct": round(100.0 * float(row["poor"]) / float(row["count"]), 1),
            }
            for name, row in agg.iterrows()
        ]
        rows.sort(key=lambda r: (r["poor_pct"], r["count"]), reverse=True)
        return rows
    except Exception as exc:
        logger.warning("_group_worklog_quality(%s) failed: %s", column, exc)
        return []


def compute_recommendation_facts(full_df: pd.DataFrame, thresholds: dict | None = None) -> dict:
    """Every aggregate the rule layer below needs, and nothing else - no
    ticket rows, no descriptions, no worklog text. Reuses
    chat_tools.get_incident_summary / get_priority_analysis /
    get_server_analysis / get_category_analysis / get_recurring_issues and
    trend_metrics.compute_resolution_metrics rather than recomputing any
    of them; only the per-group resolution/worklog cuts (which no
    existing function exposes) are computed here.

    Returns a dict that is safe to hand straight to an LLM."""
    t = {**RECOMMENDATION_THRESHOLDS, **(thresholds or {})}
    facts: dict = {"available": False, "total_incidents": 0}

    try:
        if full_df is None or len(full_df) == 0:
            facts["note"] = "No analyzed tickets are available yet - run an analysis first."
            return facts

        facts["available"] = True
        facts["total_incidents"] = int(len(full_df))

        summary = chat_tools.get_incident_summary(full_df)
        facts["summary"] = {} if "error" in summary else summary

        priority = chat_tools.get_priority_analysis(full_df)
        facts["priority"] = {} if "error" in priority else priority

        servers = chat_tools.get_server_analysis(full_df, top_n=t["server_top_n"])
        facts["servers"] = {} if "error" in servers else servers

        categories = chat_tools.get_category_analysis(full_df, top_n=t["category_top_n"])
        facts["categories"] = {} if "error" in categories else categories

        recurring = chat_tools.get_recurring_issues(full_df, threshold=t["recurrence_min_count"])
        facts["recurring"] = [] if "error" in recurring else recurring.get("recurring_issues", [])

        resolution = trend_metrics.compute_resolution_metrics(full_df)
        facts["mttr"] = resolution.get("mttr", {})
        facts["sla"] = resolution.get("sla", {})

        facts["resolution_by_category"] = _group_resolution_hours(full_df, "Category", t["slow_group_min_sample"])
        facts["resolution_by_server"] = _group_resolution_hours(full_df, "Host / CI", t["slow_group_min_sample"])
        facts["resolution_by_assignment_group"] = _group_resolution_hours(
            full_df, "Assignment Group", t["slow_group_min_sample"]
        )

        facts["worklog_by_assignment_group"] = _group_worklog_quality(
            full_df, "Assignment Group", t["worklog_group_min_sample"]
        )
        facts["worklog_by_category"] = _group_worklog_quality(full_df, "Category", t["worklog_group_min_sample"])

        return chat_tools._to_native(facts)
    except Exception as exc:
        logger.warning("compute_recommendation_facts failed: %s", exc)
        facts["note"] = f"Could not compute the recommendation facts: {exc}"
        return facts


# --------------------------------------------------------------------------
# Layer 2: rules (deterministic - every sentence built from the facts above)
# --------------------------------------------------------------------------

def _recurring_rules(facts: dict, t: dict) -> list:
    """High recurring issues: root-cause investigation, named hosts."""
    out = []
    rows = facts.get("recurring") or []
    if not rows:
        return out

    for row in rows[: t["recurrence_top_n"]]:
        count = row.get("frequency", 0)
        server = row.get("server") or "(no host detected)"
        category = row.get("category", "Unknown")
        has_host = bool(row.get("server")) and row["server"] != "(no host detected)"
        attention = "High" if count >= t["recurrence_high_count"] else "Medium"

        interval = row.get("avg_days_between_occurrences")
        window = ""
        if row.get("first_seen") and row.get("last_seen"):
            window = f" between {row['first_seen']} and {row['last_seen']}"
        cadence = f", recurring on average every {interval} day(s)" if interval is not None else ""

        subject = f"{server} " if has_host else ""
        out.append(_recommendation(
            area=AREA_RECURRING,
            observation=(
                f"{server} shows a repeating '{category}' issue."
                if has_host
                else f"'{category}' incidents keep recurring, but no host could be attributed to them."
            ),
            evidence=(
                f"{subject}accounts for {count} '{category}' incidents{window}{cadence}."
                if has_host
                else f"'{category}' recurred {count} time(s){window}{cadence}, with no host/CI captured on the tickets."
            ),
            recommendation=(
                f"Investigate the recurring failure pattern on {server} under '{category}' and review "
                f"whether a permanent corrective action (problem record, configuration fix, or capacity "
                f"change) can remove the cause rather than the symptom."
                if has_host
                else f"Review the '{category}' tickets together to identify a common root cause, and make "
                     f"host/CI capture mandatory for this category so the pattern can be attributed next time."
            ),
            attention=attention,
            benefit=(
                f"Eliminating the cause would remove roughly {count} repeat incident(s) per comparable "
                f"period and free the effort currently spent re-fixing the same problem."
            ),
            metrics={
                "server": server, "category": category, "count": count,
                "first_seen": row.get("first_seen"), "last_seen": row.get("last_seen"),
                "avg_days_between_occurrences": interval,
            },
        ))
    return out


def _priority_rules(facts: dict, t: dict) -> list:
    """P1/P2 mix and SLA compliance on high-priority work."""
    out = []
    summary = facts.get("summary") or {}
    total = facts.get("total_incidents", 0)

    p1 = summary.get("p1_incidents")
    p2 = summary.get("p2_incidents")
    if p1 is not None and p2 is not None and total:
        high = p1 + p2
        pct = round(100.0 * high / total, 1)
        if pct >= t["high_priority_pct_warning"]:
            attention = "Critical" if pct >= t["high_priority_pct_critical"] else "High"
            out.append(_recommendation(
                area=AREA_PRIORITY,
                observation="A large share of this batch is raised at P1/P2 severity.",
                evidence=f"{high} of {total} incidents ({pct}%) are P1 or P2 - {p1} P1 and {p2} P2.",
                recommendation=(
                    "Review how severity is assigned at intake. Either the environment genuinely warrants "
                    "this much critical attention - in which case major-incident capacity and on-call "
                    "coverage should be sized for it - or tickets are being over-classified, which dilutes "
                    "the meaning of P1 and pulls responders away from real emergencies."
                ),
                attention=attention,
                benefit=(
                    "Correct severity assignment keeps escalation paths credible and lets genuinely "
                    "critical incidents get responder attention first."
                ),
                metrics={"p1": p1, "p2": p2, "high_priority_pct": pct, "total": total},
            ))

    sla = facts.get("sla") or {}
    if sla.get("available") and sla.get("compliance_pct") is not None:
        by_priority = sla.get("by_priority") or {}
        breaching = [
            (label, data) for label, data in by_priority.items()
            if data.get("compliance_pct") is not None and data["compliance_pct"] < t["sla_warning_pct"]
        ]
        breaching.sort(key=lambda item: item[1]["compliance_pct"])
        for label, data in breaching[:3]:
            compliance = data["compliance_pct"]
            # A P3/P4 SLA miss is a throughput problem, not a high-priority
            # one - file it under Resolution Performance so the
            # High-Priority section stays about genuinely severe work.
            is_high_severity = any(
                token in str(label).upper() for token in ("P1", "P2", "CRITICAL", "HIGH")
            )
            attention = (
                ("Critical" if compliance < t["sla_critical_pct"] else "High")
                if is_high_severity
                else ("High" if compliance < t["sla_critical_pct"] else "Medium")
            )
            out.append(_recommendation(
                area=AREA_PRIORITY if is_high_severity else AREA_RESOLUTION,
                observation=f"{label} incidents are frequently resolved outside their target time.",
                evidence=(
                    f"{compliance}% of {data.get('sample_size', 0)} resolved {label} incident(s) met the "
                    f"{data.get('target_hours')}h target."
                ),
                recommendation=(
                    f"Walk through the {label} incidents that missed the target and identify where the time "
                    f"goes - detection, assignment, escalation, or the fix itself. Address the dominant "
                    f"delay before adjusting the target itself."
                ),
                attention=attention,
                benefit="Higher SLA attainment on the severities where breaches are most visible to the business.",
                metrics={
                    "priority": label, "compliance_pct": compliance,
                    "sample_size": data.get("sample_size"), "target_hours": data.get("target_hours"),
                },
            ))
    return out


def _resolution_rules(facts: dict, t: dict) -> list:
    """Categories / servers / assignment groups resolving unusually slowly
    relative to this batch's own average."""
    out = []
    label_for = {
        "resolution_by_category": ("category", "Category"),
        "resolution_by_server": ("server", "Server / host"),
        "resolution_by_assignment_group": ("assignment group", "Assignment group"),
    }

    for key, (noun, title_noun) in label_for.items():
        block = facts.get(key) or {}
        overall = block.get("overall_avg_hours")
        groups = block.get("groups") or []
        if overall is None or not groups:
            continue

        floor = max(overall * t["slow_group_multiplier"], t["slow_group_min_hours"])
        slow = [g for g in groups if g["avg_hours"] >= floor]
        for group in slow[: t["slow_group_top_n"]]:
            ratio = round(group["avg_hours"] / overall, 1) if overall else None
            attention = "High" if ratio is not None and ratio >= 2 else "Medium"
            out.append(_recommendation(
                area=AREA_RESOLUTION,
                observation=f"{title_noun} '{group['name']}' resolves incidents far more slowly than the rest of the batch.",
                evidence=(
                    f"{group['count']} resolved incident(s) in '{group['name']}' took {group['avg_hours']}h on "
                    f"average, against a batch average of {overall}h"
                    + (f" ({ratio}x)." if ratio is not None else ".")
                ),
                recommendation=(
                    f"Examine the resolution path for this {noun}: check whether the delay is queue/wait time, "
                    f"a hand-off between teams, a dependency on a third party, or genuine technical complexity. "
                    f"Target the largest contributor rather than pushing for faster closure generally."
                ),
                attention=attention,
                benefit=(
                    f"Bringing '{group['name']}' toward the {overall}h batch average would measurably reduce "
                    f"overall MTTR and the customer-visible wait on these incidents."
                ),
                metrics={
                    "dimension": noun, "name": group["name"], "avg_hours": group["avg_hours"],
                    "batch_avg_hours": overall, "sample_size": group["count"], "ratio": ratio,
                },
            ))
    return out


def _worklog_rules(facts: dict, t: dict) -> list:
    """Overall and per-group documentation quality."""
    out = []
    summary = facts.get("summary") or {}
    poor_pct = summary.get("poor_worklog_percentage")
    total = facts.get("total_incidents", 0)

    if poor_pct is not None and poor_pct >= t["poor_worklog_pct_warning"]:
        attention = "High" if poor_pct >= t["poor_worklog_pct_critical"] else "Medium"
        poor_count = round(poor_pct / 100.0 * total) if total else 0
        avg_score = summary.get("average_worklog_score")
        avg_clause = f" The average worklog score is {avg_score}." if avg_score is not None else ""
        out.append(_recommendation(
            area=AREA_WORKLOG,
            observation="A significant share of incidents is closed with poor-quality worklog notes.",
            evidence=(
                f"{poor_count} of {total} incidents ({poor_pct}%) score below "
                f"{chat_tools.POOR_WORKLOG_THRESHOLD} on worklog quality.{avg_clause}"
            ),
            recommendation=(
                "Set a minimum closure standard - symptom, diagnosis, action taken, and verification - and "
                "make it part of the closure check rather than a post-hoc audit. Publish a short worked "
                "example so the standard is concrete."
            ),
            attention=attention,
            benefit=(
                "Better notes make recurring-issue detection and root-cause analysis reliable, and cut the "
                "rediscovery effort when the same problem reappears."
            ),
            metrics={"poor_worklog_pct": poor_pct, "poor_count": poor_count, "total": total,
                     "average_worklog_score": avg_score},
        ))

    for key, noun in (("worklog_by_assignment_group", "Assignment group"), ("worklog_by_category", "Category")):
        rows = [r for r in (facts.get(key) or []) if r["poor_pct"] >= t["worklog_group_poor_pct"]]
        for row in rows[: t["worklog_group_top_n"]]:
            out.append(_recommendation(
                area=AREA_WORKLOG,
                observation=f"{noun} '{row['name']}' documents incidents noticeably worse than the rest.",
                evidence=(
                    f"{row['poor_count']} of {row['count']} incident(s) ({row['poor_pct']}%) in '{row['name']}' "
                    f"have a poor worklog; its average score is {row['avg_score']}."
                ),
                recommendation=(
                    f"Take this up directly with '{row['name']}' rather than issuing a general reminder - "
                    f"review a handful of their recent closures together and agree what a complete note "
                    f"looks like for the work they actually do."
                ),
                attention="Medium",
                benefit=f"Raising '{row['name']}' to the batch standard removes the largest single gap in documentation coverage.",
                metrics={"dimension": noun, **row},
            ))
    return out


def _server_rules(facts: dict, t: dict) -> list:
    """Hosts carrying a disproportionate share of the incident volume."""
    out = []
    servers = (facts.get("servers") or {}).get("top_servers") or []
    total = facts.get("total_incidents", 0)
    if not servers or not total:
        return out

    for row in servers[: t["server_top_n"]]:
        count = row.get("count", 0)
        share = round(100.0 * count / total, 1)
        if count < t["server_min_incidents"] or share < t["server_share_pct_warning"]:
            continue
        out.append(_recommendation(
            area=AREA_SERVERS,
            observation=f"Server {row['server']} carries a disproportionate share of the incident volume.",
            evidence=f"{row['server']} accounts for {count} of {total} incidents ({share}% of the batch).",
            recommendation=(
                f"Review {row['server']} as a unit rather than ticket by ticket: check capacity/utilisation "
                f"headroom, patch and firmware currency, recent change history, and monitoring thresholds. "
                f"If the volume is driven by a single failing component, plan preventive maintenance or "
                f"replacement instead of continuing to absorb the tickets."
            ),
            attention="High" if share >= t["server_share_pct_warning"] * 2 else "Medium",
            benefit=(
                f"A single fix on {row['server']} addresses {share}% of this batch's incident load - the "
                f"highest-leverage host-level action available from this data."
            ),
            metrics={"server": row["server"], "count": count, "share_pct": share, "total": total},
        ))
    return out


def _category_rules(facts: dict, t: dict) -> list:
    """Categories dominating the volume, and categories that both
    dominate and recur (the strongest problem-management signal)."""
    out = []
    top = (facts.get("categories") or {}).get("top_categories") or []
    total = facts.get("total_incidents", 0)
    if not top or not total:
        return out

    recurring_by_category: dict = {}
    for row in facts.get("recurring") or []:
        recurring_by_category[row.get("category", "")] = (
            recurring_by_category.get(row.get("category", ""), 0) + row.get("frequency", 0)
        )

    for row in top[: t["category_top_n"]]:
        count = row.get("count", 0)
        share = round(100.0 * count / total, 1)
        if share < t["category_share_pct_warning"]:
            continue
        recurring_count = recurring_by_category.get(row["category"], 0)
        recurrence_clause = (
            f" {recurring_count} of them fall into groups already flagged as recurring."
            if recurring_count else ""
        )
        out.append(_recommendation(
            area=AREA_CATEGORIES,
            observation=f"'{row['category']}' dominates the incident mix for this period.",
            evidence=f"'{row['category']}' accounts for {count} of {total} incidents ({share}%).{recurrence_clause}",
            recommendation=(
                f"Open a problem record for '{row['category']}' and break the volume down by host and by "
                f"failure mode to see whether it is one systemic cause or many unrelated issues sharing a "
                f"label. Where the pattern is repeatable, consider automation or a self-service path "
                f"instead of handling each occurrence manually."
                if recurring_count else
                f"Break '{row['category']}' down by host and failure mode to confirm whether the volume "
                f"reflects one systemic cause or simply a broad category. Where demand is predictable, "
                f"consider a knowledge article or self-service path to divert it from the queue."
            ),
            attention="High" if recurring_count else "Medium",
            benefit=(
                f"'{row['category']}' is the largest single lever on total volume - reducing it by even a "
                f"third would remove roughly {round(count / 3)} incident(s) per comparable period."
            ),
            metrics={"category": row["category"], "count": count, "share_pct": share,
                     "recurring_count": recurring_count, "total": total},
        ))
    return out


_RULE_FUNCTIONS = (
    _recurring_rules, _priority_rules, _resolution_rules,
    _worklog_rules, _server_rules, _category_rules,
)


def generate_recommendations(full_df: pd.DataFrame, facts: dict | None = None, thresholds: dict | None = None) -> dict:
    """Main entry point. Computes the facts (unless already supplied by
    the caller, so a caller that needs both doesn't pay for them twice)
    and runs every rule group over them.

    Returns {"available": bool, "recommendations": [...], "facts": {...},
    "note": str, "counts_by_area": {...}}. Never raises - a rule group
    that fails is logged and skipped, so one bad aggregate can't take the
    whole panel down."""
    try:
        t = {**RECOMMENDATION_THRESHOLDS, **(thresholds or {})}
        facts = facts if facts is not None else compute_recommendation_facts(full_df, thresholds=t)

        if not facts.get("available"):
            return {
                "available": False, "recommendations": [], "facts": facts,
                "note": facts.get("note", "No analyzed tickets are available yet - run an analysis first."),
                "counts_by_area": {},
            }

        recommendations: list = []
        for rule_fn in _RULE_FUNCTIONS:
            try:
                recommendations.extend(rule_fn(facts, t) or [])
            except Exception as exc:
                logger.warning("Recommendation rule %s failed: %s", rule_fn.__name__, exc)

        recommendations.sort(key=lambda r: ATTENTION_ORDER.get(r.get("attention"), 99))
        trimmed = max(0, len(recommendations) - t["max_recommendations"])
        recommendations = recommendations[: t["max_recommendations"]]

        counts_by_area: dict = {}
        for rec in recommendations:
            counts_by_area[rec["area"]] = counts_by_area.get(rec["area"], 0) + 1

        if not recommendations:
            note = (
                "No recommendation thresholds were crossed in this batch - recurrence, priority mix, "
                "resolution times, worklog quality, and volume concentration are all within the configured "
                "ranges. Nothing here is fabricated to fill the panel."
            )
        elif trimmed:
            note = f"Showing the {len(recommendations)} most urgent of {len(recommendations) + trimmed} findings."
        else:
            note = ""
        return chat_tools._to_native({
            "available": True,
            "recommendations": recommendations,
            "facts": facts,
            "note": note,
            "counts_by_area": counts_by_area,
        })
    except Exception as exc:
        logger.warning("generate_recommendations failed: %s", exc)
        return {
            "available": False, "recommendations": [], "facts": facts or {},
            "note": f"Could not generate recommendations: {exc}", "counts_by_area": {},
        }


# --------------------------------------------------------------------------
# Layer 3 support: the small payload the LLM is allowed to see
# --------------------------------------------------------------------------

def build_llm_payload(result: dict, max_recommendations: int = 12) -> dict:
    """The ONLY thing sent to the LLM for the management write-up: the
    already-computed recommendations plus a handful of headline
    aggregates. Deliberately excludes the full facts blob (which carries
    every per-group cut) and, absolutely, any ticket row, description, or
    worklog text - there is no path from here back to the dataframe.

    Because each recommendation already contains its own evidence
    sentence AND the numbers behind it in `metrics`, the model has
    everything it needs to re-voice them and no source for a number that
    wasn't calculated in Python."""
    try:
        facts = result.get("facts") or {}
        summary = facts.get("summary") or {}
        return {
            "total_incidents": facts.get("total_incidents", 0),
            "headline_metrics": {
                "p1_incidents": summary.get("p1_incidents"),
                "p2_incidents": summary.get("p2_incidents"),
                "average_resolution_time_hours": summary.get("average_resolution_time_hours"),
                "average_worklog_score": summary.get("average_worklog_score"),
                "poor_worklog_percentage": summary.get("poor_worklog_percentage"),
                "sla_compliance_pct": (facts.get("sla") or {}).get("compliance_pct"),
            },
            "counts_by_area": result.get("counts_by_area", {}),
            "recommendations": [
                {
                    "area": r["area"], "observation": r["observation"], "evidence": r["evidence"],
                    "recommendation": r["recommendation"], "attention": r["attention"],
                    "expected_benefit": r["expected_benefit"], "metrics": r.get("metrics", {}),
                }
                for r in (result.get("recommendations") or [])[:max_recommendations]
            ],
        }
    except Exception as exc:
        logger.warning("build_llm_payload failed: %s", exc)
        return {"total_incidents": 0, "headline_metrics": {}, "counts_by_area": {}, "recommendations": []}


def fallback_recommendations_markdown(result: dict) -> str:
    """Deterministic markdown write-up built from the same
    recommendations - no LLM. This is what the "management write-up"
    button shows when no model is configured or the call fails, so the
    button is never a dead end."""
    try:
        if not result.get("available"):
            return result.get("note", "Run an analysis first, then generate recommendations.")

        recommendations = result.get("recommendations") or []
        if not recommendations:
            return result.get("note", "No recommendation thresholds were crossed in this batch.")

        facts = result.get("facts") or {}
        lines = [
            f"**{len(recommendations)} recommendation(s)** from "
            f"{facts.get('total_incidents', 0)} analyzed incident(s), most urgent first.",
            "",
        ]
        for i, rec in enumerate(recommendations, start=1):
            lines.extend([
                f"**{i}. {rec['observation']}**  _({rec['attention']} attention · {rec['area']})_",
                f"- **Evidence:** {rec['evidence']}",
                f"- **Recommendation:** {rec['recommendation']}",
                f"- **Expected benefit:** {rec['expected_benefit']}",
                "",
            ])
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("fallback_recommendations_markdown failed: %s", exc)
        return "Could not render the recommendations write-up from the current data."


def get_recommendations(full_df: pd.DataFrame, thresholds: dict | None = None) -> dict:
    """Agent-tool-shaped wrapper (see rag.py): same small, JSON-safe
    contract as the chat_tools functions - the recommendations without
    the full facts blob, so a tool result stays compact."""
    try:
        result = generate_recommendations(full_df, thresholds=thresholds)
        if not result.get("available"):
            return {"recommendations": [], "note": result.get("note", "")}
        return {
            "recommendations": [
                {k: v for k, v in rec.items() if k != "metrics"}
                for rec in result.get("recommendations", [])
            ],
            "counts_by_area": result.get("counts_by_area", {}),
            "note": result.get("note", ""),
        }
    except Exception as exc:
        logger.warning("get_recommendations failed: %s", exc)
        return {"error": f"Could not compute recommendations: {exc}"}
