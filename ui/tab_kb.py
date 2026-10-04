"""Knowledge Base tab handlers."""
import os

import gradio as gr
import pandas as pd

from app.services import knowledge_base


# Tab order/icons (Overview, Categorization, Trends & Insights,
# Recommendations, Knowledge Base, Q&A (Agent), Settings) now live
# directly on each gr.Tab(...) label below - there's no sidebar anymore
# to keep a separate icon/label list in sync with.


# --- Knowledge Base (Step 13) ------------------------------------------
# Thin UI-layer wrappers around knowledge_base.py - no ingestion/retrieval
# logic lives here, only building the table/dropdown Gradio renders and
# turning a dropdown selection back into a document_id.

def _kb_documents_dataframe() -> pd.DataFrame:
    """Backs the Indexed Documents table. Uploaded At is reformatted to
    match the rest of the dashboard's date display; Error is blank unless
    that document's status is Failed."""
    try:
        docs = knowledge_base.list_documents()
        if not docs:
            return pd.DataFrame(columns=["Document Name", "Type", "Status", "Chunks", "Uploaded At", "Error"])
        rows = []
        for d in docs:
            uploaded = d.get("uploaded_at") or ""
            try:
                uploaded = pd.to_datetime(uploaded).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
            rows.append({
                "Document Name": d["document_name"],
                "Type": (d["document_type"] or "").upper(),
                "Status": d["status"],
                "Chunks": d.get("chunk_count", 0),
                "Uploaded At": uploaded,
                "Error": d.get("error_message") or "",
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["Document Name", "Type", "Status", "Chunks", "Uploaded At", "Error"])


def _kb_document_choices() -> list[tuple[str, int]]:
    """(display_label, document_id) pairs for the manage dropdown - the id
    is the actual dropdown value so two documents with the same filename
    (re-uploaded, or coincidentally identical names) never get confused
    for each other, even though their labels look the same."""
    try:
        return [(f"{d['document_name']} ({d['status']})", d["id"]) for d in knowledge_base.list_documents()]
    except Exception:
        return []


def _kb_refresh():
    """Recomputes both the table and the manage-dropdown choices - called
    after every upload/delete/reindex so they can never show stale data
    relative to each other."""
    return _kb_documents_dataframe(), gr.update(choices=_kb_document_choices(), value=None)


async def _kb_upload(file_obj):
    """Upload & Index button handler. Reads the uploaded file's bytes and
    runs the full ingestion pipeline (knowledge_base.ingest_document) -
    extraction, chunking, embedding, persistence - then refreshes the
    table/dropdown and reports success/failure in plain language."""
    if file_obj is None:
        table, dropdown = _kb_refresh()
        return table, dropdown, "Choose a file first."
    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        filename = os.path.basename(path)
        with open(path, "rb") as f:
            file_bytes = f.read()
        result = await knowledge_base.ingest_document(filename, file_bytes)
        table, dropdown = _kb_refresh()
        if result["success"]:
            status_msg = f"✅ **{filename}** indexed successfully ({result['chunk_count']} chunks)."
        else:
            status_msg = f"❌ **{filename}** failed to index: {result.get('error', 'unknown error')}"
        return table, dropdown, status_msg
    except Exception as exc:
        table, dropdown = _kb_refresh()
        return table, dropdown, f"❌ Upload failed: {exc}"


def _kb_delete(document_id):
    if document_id is None:
        table, dropdown = _kb_refresh()
        return table, dropdown, "Select a document first."
    try:
        deleted = knowledge_base.delete_document(int(document_id))
        table, dropdown = _kb_refresh()
        msg = "🗑️ Document deleted." if deleted else "Document was already gone."
        return table, dropdown, msg
    except Exception as exc:
        table, dropdown = _kb_refresh()
        return table, dropdown, f"❌ Delete failed: {exc}"


async def _kb_reindex(document_id):
    if document_id is None:
        table, dropdown = _kb_refresh()
        return table, dropdown, "Select a document first."
    try:
        result = await knowledge_base.reindex_document(int(document_id))
        table, dropdown = _kb_refresh()
        if result["success"]:
            return table, dropdown, f"🔁 Reindexed successfully ({result['chunk_count']} chunks)."
        return table, dropdown, f"❌ Reindex failed: {result.get('error', 'unknown error')}"
    except Exception as exc:
        table, dropdown = _kb_refresh()
        return table, dropdown, f"❌ Reindex failed: {exc}"
