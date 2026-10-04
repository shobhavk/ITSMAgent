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
    filtered = _apply_filters(_attach_timeline(full_df), category, min_score)
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
