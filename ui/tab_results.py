"""Recent Incidents table: timeline merge, filtering, pagination, column selection."""
import pandas as pd

from app.services import incident_timeline

from ui.constants import ALL_COLUMNS, DEFAULT_PAGE_SIZE, DEFAULT_VISIBLE_COLUMNS, TIMELINE_COLUMNS
from ui.helpers import _fmt_ts


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Restricts a dataframe to the fixed default column set for on-screen
    display, in ALL_COLUMNS order. Filtering/pagination/CSV export always
    operate on the full, un-reduced dataframe - only this final display
    step drops columns."""
    cols = [c for c in ALL_COLUMNS if c in DEFAULT_VISIBLE_COLUMNS] + TIMELINE_COLUMNS
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols or ["Ticket ID"])
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = ["Ticket ID"]  # never render a fully empty table
    return df[cols]


def _attach_timeline(full_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of full_df with the incident-timeline columns
    (acknowledgment time, reassignment, External Info quality, ...) added,
    so they show up in the Recent Incidents table. compute_incident_timeline
    emits exactly one row per ticket in full_df's own order, so the join is
    positional. Any failure just returns full_df unchanged - the table
    then simply shows its normal columns."""
    if full_df is None or len(full_df) == 0:
        return full_df
    try:
        tl = incident_timeline.compute_incident_timeline(full_df)
        if len(tl) != len(full_df):
            return full_df
        out = full_df.copy()
        for col in TIMELINE_COLUMNS:
            if col not in tl.columns:
                continue
            values = tl[col].tolist()
            if col in ("Opened/Created At", "Acknowledged At", "Resolved At"):
                values = [_fmt_ts(v) for v in values]
            elif col == "Reassignment Detected":
                values = ["Yes" if v else "No" for v in values]
            elif col == "Time to Acknowledge (h)":
                values = ["" if (v is None or pd.isna(v)) else v for v in values]
            else:
                values = ["" if v is None else v for v in values]
            out[col] = values
        return out
    except Exception:
        return full_df


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
    filtered = _apply_filters(_with_review_flag(_attach_timeline(full_df)), category, min_score)
    page_df, indicator, page = _paginate(filtered, 1, page_size)
    return _select_columns(page_df), indicator, filtered, page


def _go_to_page(filtered_df, page, page_size, delta):
    page_df, indicator, new_page = _paginate(filtered_df, (page or 1) + delta, page_size)
    return _select_columns(page_df), indicator, new_page


def _change_page_size(filtered_df, new_page_size):
    """Rows-per-page dropdown handler: re-paginates the already-filtered
    table from page 1 at the newly chosen page size, and updates
    page_size_state so prev_btn/next_btn keep using it afterward."""
    page_df, indicator, page = _paginate(filtered_df, 1, new_page_size)
    return _select_columns(page_df), indicator, page, new_page_size


# ---------------------------------------------------------------------------
# Categorization tab: review flag, filter bar, summary strip, row detail drawer
# ---------------------------------------------------------------------------
import html as _html

import gradio as gr

from ui.constants import LOW_CONFIDENCE_THRESHOLD, UNCATEGORISED_LABELS
from ui.helpers import _strip_html, _truncate

_CAT_SUMMARY_PLACEHOLDER = (
    '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">'
    "Run an analysis to see the categorization summary.</p>"
)


def _review_mask(df: pd.DataFrame) -> pd.Series:
    """True for tickets that should be reviewed by a person: category
    confidence below LOW_CONFIDENCE_THRESHOLD, or no real category."""
    conf = pd.to_numeric(df.get("Category Confidence"), errors="coerce")
    low = conf.fillna(0) < LOW_CONFIDENCE_THRESHOLD
    cat = df["Category"].fillna("").astype(str).str.strip().str.lower()
    return low | cat.isin(UNCATEGORISED_LABELS)


def _with_review_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a display column 'Review' ('⚠ Needs review' / '✓ OK'). Only the
    table view carries it; the full dataframe used for export and charts is
    left untouched."""
    if df is None or len(df) == 0 or "Category" not in df.columns:
        return df
    out = df.copy()
    out["Review"] = _review_mask(out).map({True: "⚠ Needs review", False: "✓ OK"})
    return out


def _filter_view(full_df, search, category, priority, review_only, min_score, page_size):
    """Filter-bar handler: applies search / category / priority / needs-review
    on top of the existing category + min-score filter, resets to page 1."""
    base = _with_review_flag(_attach_timeline(full_df))
    filtered = _apply_filters(base, category, min_score)
    if filtered is not None and len(filtered):
        if priority and priority != "All":
            filtered = filtered[filtered["Priority"].fillna("").astype(str) == priority]
        if review_only and "Review" in filtered.columns:
            filtered = filtered[filtered["Review"] == "⚠ Needs review"]
        q = (search or "").strip().lower()
        if q:
            hay_cols = [c for c in ("Ticket ID", "Short Description", "Description", "Worklog Notes", "Host / CI") if c in filtered.columns]
            mask = pd.Series(False, index=filtered.index)
            for c in hay_cols:
                mask |= filtered[c].fillna("").astype(str).str.lower().str.contains(q, regex=False)
            filtered = filtered[mask]
        filtered = filtered.reset_index(drop=True)
    page_df, indicator, page = _paginate(filtered, 1, page_size)
    return _select_columns(page_df), indicator, filtered, page


def _categorization_summary_html(full_df: pd.DataFrame) -> str:
    from ui.components import _kpi_card_v2  # local import: components imports charts

    if full_df is None or len(full_df) == 0 or "Category" not in full_df.columns:
        return _CAT_SUMMARY_PLACEHOLDER
    total = len(full_df)
    cats = full_df["Category"].fillna("").astype(str).str.strip()
    n_categories = cats[~cats.str.lower().isin(UNCATEGORISED_LABELS)].nunique()
    conf = pd.to_numeric(full_df.get("Category Confidence"), errors="coerce").dropna()
    avg_conf = f"{conf.mean() * 100:.0f}%" if len(conf) else "n/a"
    n_review = int(_review_mask(full_df).sum())
    review_accent = "#b3261e" if n_review else "#2e7d5b"
    cards = [
        _kpi_card_v2("🗂️", "Tickets categorised", total, "#0f6e7a"),
        _kpi_card_v2("🏷️", "Distinct categories", n_categories, "#2b7a9b"),
        _kpi_card_v2("🎯", "Avg category confidence", avg_conf, "#2e7d5b"),
        _kpi_card_v2(
            "⚠️", "Needs review", n_review, review_accent,
            note=f"below {LOW_CONFIDENCE_THRESHOLD * 100:.0f}% confidence or uncategorised",
        ),
    ]
    return '<div class="cat-summary-grid">' + "".join(cards) + "</div>"


def _refresh_categorization_controls(full_df):
    """After each analysis: reset the filter bar, refill its choices from the
    new batch, refresh the summary strip and close any open detail drawer."""
    if full_df is None or len(full_df) == 0:
        cats, pris = ["All"], ["All"]
    else:
        cats = ["All"] + sorted({str(c) for c in full_df["Category"].fillna("").astype(str) if str(c).strip()})
        pris = ["All"] + sorted({str(p) for p in full_df["Priority"].fillna("").astype(str) if str(p).strip()})
    return (
        gr.update(choices=cats, value="All"),
        gr.update(choices=pris, value="All"),
        gr.update(value=""),
        gr.update(value=False),
        _categorization_summary_html(full_df),
        gr.update(visible=False),
    )


def _row_detail_html(row: pd.Series) -> str:
    esc = lambda v: _html.escape(str(v if v is not None and not (isinstance(v, float) and pd.isna(v)) else ""))

    def block(title, text, limit=1500):
        text = _truncate(_strip_html(str(text or "")), limit)
        body = esc(text) if text.strip() else '<span class="drw-empty">Not provided</span>'
        return f'<div class="drw-sec"><div class="drw-sec-title">{title}</div><div class="drw-text">{body}</div></div>'

    conf = pd.to_numeric(row.get("Category Confidence"), errors="coerce")
    conf_pct = 0 if pd.isna(conf) else max(0, min(100, int(round(float(conf) * 100))))
    conf_cls = "good" if conf_pct >= LOW_CONFIDENCE_THRESHOLD * 100 else "poor"
    score = pd.to_numeric(row.get("Worklog Score"), errors="coerce")
    score_pct = 0 if pd.isna(score) else max(0, min(100, int(score)))
    score_cls = "good" if score_pct >= 75 else "fair" if score_pct >= 50 else "poor"
    needs_review = str(row.get("Review", "")).startswith("⚠")
    review_chip = (
        '<span class="drw-chip drw-chip-warn">⚠ Needs review</span>' if needs_review
        else '<span class="drw-chip drw-chip-ok">✓ Category OK</span>'
    )
    meta = [
        ("Assignment group", row.get("Assignment Group")), ("Host / CI", row.get("Host / CI")),
        ("Status", row.get("Status")), ("Category method", row.get("Category Method")),
    ]
    meta_html = "".join(
        f'<div><div class="drw-meta-k">{k}</div><div class="drw-meta-v">{esc(v) or "-"}</div></div>' for k, v in meta
    )
    flags = str(row.get("Worklog Flags") or "").strip()
    flags_html = "".join(f'<span class="drw-chip">{esc(f.strip())}</span>' for f in flags.split(";") if f.strip())
    return (
        '<div class="drw">'
        f'<div class="drw-id">{esc(row.get("Ticket ID"))}</div>'
        f'<div class="drw-title">{esc(_truncate(_strip_html(str(row.get("Short Description") or "")), 160)) or "(no short description)"}</div>'
        f'<div class="drw-chips"><span class="drw-chip drw-chip-cat">{esc(row.get("Category"))}</span>'
        f'<span class="drw-chip">{esc(row.get("Priority")) or "No priority"}</span>{review_chip}</div>'
        '<div class="drw-meters">'
        f'<div><div class="drw-meter-label"><span>Category confidence</span><b>{conf_pct}%</b></div>'
        f'<div class="drw-track"><i class="{conf_cls}" style="width:{conf_pct}%"></i></div></div>'
        f'<div><div class="drw-meter-label"><span>Worklog score</span><b>{score_pct} / 100</b></div>'
        f'<div class="drw-track"><i class="{score_cls}" style="width:{score_pct}%"></i></div></div>'
        '</div>'
        + (f'<div class="drw-chips">{flags_html}</div>' if flags_html else "")
        + f'<div class="drw-meta">{meta_html}</div>'
        + block("Description", row.get("Description"))
        + block("Worklog notes", row.get("Worklog Notes"))
        + block("Validation notes", row.get("Validation Notes"), 600)
        + "</div>"
    )


def _show_row_detail(filtered_df, page, page_size, evt: gr.SelectData):
    """Row click in the results table -> open the detail drawer for that ticket."""
    try:
        idx = evt.index
        row_pos = int(idx[0] if isinstance(idx, (list, tuple)) else idx)
        size = page_size or DEFAULT_PAGE_SIZE
        pos = ((page or 1) - 1) * size + row_pos
        if filtered_df is None or pos < 0 or pos >= len(filtered_df):
            return gr.update(), gr.update()
        return gr.update(visible=True), _row_detail_html(filtered_df.iloc[pos])
    except Exception:
        return gr.update(), gr.update()


def _close_row_detail():
    return gr.update(visible=False)
