"""
Recurring issue detection - two complementary, independently-useful layers.

1. Exact-match recurrence (detect_exact_recurrence): groups by (host,
   category) - or category alone when host is blank - and flags any
   group whose ticket count meets/exceeds a threshold, with a real time
   dimension (first/last seen, span, average interval between
   occurrences). Fast, free (no LLM/embeddings), deterministic, and not
   affected by wording variation - this is the more reliable of the two.
   It replaces the old "Repeated Issues (Top Hosts)" panel, which showed
   the top 6 host+category combos unconditionally (even a single
   occurrence would show up) with no time information at all.

2. Semantic recurrence (detect_semantic_recurrence): clusters tickets by
   embedding similarity, so a recurring issue that keeps getting logged
   under different categories, or worded differently each time, still
   gets caught even though it wouldn't share an exact (host, category)
   key. This is opt-in, not automatic on every analysis - clustering
   needs the whole batch embedded first (one embedding call), which is a
   cost/latency tradeoff the exact-match layer doesn't have. See the
   "Detect Similar Recurring Issues" button wiring in gradio_app.py,
   which reuses the same TicketIndex the chat tab builds (so if chat
   already ran, this is free; if this runs first, chat becomes free).
"""
import pandas as pd

from app.services import trend_metrics

DEFAULT_RECURRENCE_THRESHOLD = 3
DEFAULT_SIMILARITY_CUTOFF = 0.85
# Semantic clustering is O(n^2) in memory (a full similarity matrix) - cap
# it so a very large batch degrades to "use the exact-match table instead"
# rather than a multi-GB allocation.
MAX_TICKETS_FOR_SEMANTIC_CLUSTERING = 3000


def detect_exact_recurrence(full_df: pd.DataFrame, threshold: int = DEFAULT_RECURRENCE_THRESHOLD) -> list[dict]:
    """Groups tickets by (Host / CI, Category) - falling back to Category
    alone when a ticket has no host - and returns every group whose count
    meets `threshold`, richest-first (highest count first). Each result:
    host, category, count, first_seen, last_seen, span_days,
    avg_interval_days (None when fewer than 2 dated occurrences)."""
    if full_df is None or len(full_df) == 0 or "Category" not in full_df.columns:
        return []

    d = full_df.copy()
    host_col = (
        d["Host / CI"].fillna("").astype(str).str.strip()
        if "Host / CI" in d.columns
        else pd.Series([""] * len(d), index=d.index)
    )
    category_col = d["Category"].fillna("Unknown").astype(str)
    group_key = host_col.where(host_col != "", "(no host)") + "\u241f" + category_col

    opened = trend_metrics.effective_open_resolve_times(d)["start"]

    results = []
    for key, idx in group_key.groupby(group_key).groups.items():
        count = len(idx)
        if count < threshold:
            continue
        host, _, category = key.partition("\u241f")

        dates = opened.loc[idx].dropna()
        if len(dates) >= 2:
            first_seen, last_seen = dates.min(), dates.max()
            span_days = (last_seen - first_seen).total_seconds() / 86400.0
            avg_interval_days = round(span_days / (count - 1), 1) if count > 1 else None
            first_seen_s, last_seen_s = first_seen.strftime("%Y-%m-%d"), last_seen.strftime("%Y-%m-%d")
            span_days = round(span_days, 1)
        else:
            span_days = avg_interval_days = first_seen_s = last_seen_s = None

        results.append(
            {
                "host": "" if host == "(no host)" else host,
                "category": category,
                "count": count,
                "first_seen": first_seen_s,
                "last_seen": last_seen_s,
                "span_days": span_days,
                "avg_interval_days": avg_interval_days,
            }
        )

    results.sort(key=lambda r: r["count"], reverse=True)
    return results


async def detect_semantic_recurrence(
    index,
    threshold: int = DEFAULT_RECURRENCE_THRESHOLD,
    similarity_cutoff: float = DEFAULT_SIMILARITY_CUTOFF,
) -> dict:
    """Clusters tickets in `index` (a rag.TicketIndex) by text similarity
    and returns clusters with >= threshold members. Returns
    {"available": bool, "clusters": [...], "note": str} - available is
    False (with an explanatory note) if there's no data yet or the batch
    is too large for the O(n^2) similarity matrix; never raises."""
    if index is None or not index.rows:
        return {"available": False, "clusters": [], "note": "Run an analysis first."}

    if len(index.rows) > MAX_TICKETS_FOR_SEMANTIC_CLUSTERING:
        return {
            "available": False,
            "clusters": [],
            "note": (
                f"This batch has {len(index.rows)} tickets - semantic clustering is capped at "
                f"{MAX_TICKETS_FOR_SEMANTIC_CLUSTERING} to keep memory bounded. Use the exact-match "
                f"recurrence table above instead."
            ),
        }

    member_groups = await index.cluster_similar(similarity_cutoff=similarity_cutoff, min_cluster_size=threshold)

    clusters = []
    for members in member_groups:
        rows = [index.rows[i] for i in members]
        categories = sorted({r["category"] for r in rows if r["category"]})
        clusters.append(
            {
                "count": len(rows),
                "categories": categories,
                "sample_ticket_ids": [r["ticket_id"] for r in rows[:5]],
                "excerpt": rows[0]["text"][:220],
            }
        )
    clusters.sort(key=lambda c: c["count"], reverse=True)
    return {"available": True, "clusters": clusters, "note": ""}
