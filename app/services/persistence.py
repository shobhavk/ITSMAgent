"""
persistence.py
---------------
Persistent storage for the ITSM Quality Analysis app.

Purpose:
  When a user uploads a file, we hash its raw bytes. If we've already
  processed a file with that exact hash, we return the stored results
  directly from the DB instead of re-running classification/scoring
  through the LLM pipeline (LangGraph classify + score nodes).

Storage: SQLite via SQLAlchemy (file-based, no extra infra).
  - Mount the DB file on a Docker volume so it survives container
    restarts, e.g.:
      volumes:
        - itsm_db_data:/app/data
    and point DB_PATH at /app/data/itsm_analysis.db in that environment.

Design notes:
  - Dedup key is SHA-256 of the raw uploaded file bytes -> exact-file
    duplicate detection (not near-duplicate / fuzzy matching).
  - Full processed DataFrame is stored as JSON (records orient) so it
    round-trips exactly, including any truncated/full text columns.
  - We also store which config flags were active (e.g.
    ENABLE_LLM_WORKLOG_SCORING) when the file was processed. If the
    current config no longer matches, the cache is treated as stale
    so results aren't silently wrong after a config change.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text,
    Float,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.environ.get("ITSM_DB_PATH", "./data/itsm_analysis.db")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ticket_count = Column(Integer, default=0)

    # Config snapshot at time of processing — used to invalidate cache
    # if the pipeline behavior has since changed.
    llm_worklog_scoring_enabled = Column(Boolean, default=False)
    pipeline_version = Column(String(64), default="v1")

    # Full result set, exact round-trip (pandas records-orient JSON).
    results_json = Column(Text, nullable=False)

    __table_args__ = (UniqueConstraint("file_hash", name="uq_file_hash"),)


class TicketCache(Base):
    """
    Real duplicate detection: keyed on Incident/Ticket ID, not file hash.
    Whenever a ticket with this exact ID and unchanged content has been
    analyzed before - in ANY upload, not just an identical file - the
    pipeline reuses this row instead of calling the LLM again for that
    ticket. content_hash guards against silently serving stale results
    if the same Incident ID was re-exported with edited text.
    """
    __tablename__ = "ticket_cache"

    ticket_id = Column(String(128), primary_key=True)
    content_hash = Column(String(64), nullable=False)
    category = Column(String(256))
    category_confidence = Column(Float)
    category_method = Column(String(64))
    worklog_score = Column(Integer)
    worklog_flags_json = Column(Text, default="[]")
    host = Column(String(256), nullable=True)
    configuration_item = Column(String(256), nullable=True)
    llm_worklog_scoring_enabled = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    # Carried through so a cache-hit ticket still has its timestamps for
    # trend/resolution-metric analysis - these describe the ticket itself,
    # not the LLM processing, so they're safe to reuse across re-uploads
    # even though category/score are keyed on unchanged content.
    opened_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    responded_at = Column(DateTime, nullable=True)
    detected_at = Column(DateTime, nullable=True)


class KnowledgeDocument(Base):
    """
    Step 13 - Knowledge Base / RAG. One row per uploaded document (PDF,
    DOCX, TXT, or MD) - metadata + ingestion status only; the actual
    chunk text and embeddings live in KnowledgeChunk below, keyed by
    document_id. Global to the app instance (like TicketCache above),
    not scoped per API key - a runbook or SOP is organizational
    knowledge, not something tied to one analysis session.
    """
    __tablename__ = "kb_documents"

    id = Column(Integer, primary_key=True)
    document_name = Column(String(512), nullable=False)
    document_type = Column(String(16), nullable=False)  # pdf / docx / txt / md
    # Processing -> Indexed, or Processing -> Failed (see error_message).
    status = Column(String(32), nullable=False, default="Processing")
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    # Extracted-and-cleaned text units (JSON list of {text, page_number,
    # section_title}) BEFORE chunking - kept so "reindex" (re-chunk +
    # re-embed, e.g. after a KB_CHUNK_SIZE change) doesn't require the
    # user to re-upload the original file. The original binary is not
    # kept - only its already-extracted text, which is all re-chunking
    # needs and is far smaller than a PDF/DOCX.
    source_units_json = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class KnowledgeChunk(Base):
    """One retrievable chunk of a KnowledgeDocument, with its embedding
    vector stored as a JSON list of floats (None if no embeddings model
    was configured at ingestion time - see knowledge_base.py's keyword
    fallback for that case, mirroring rag.py's TicketIndex pattern)."""
    __tablename__ = "kb_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("kb_documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=True)
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(512), nullable=True)


Base.metadata.create_all(engine)


def compute_file_hash(file_bytes: bytes) -> str:
    """SHA-256 hash of raw file bytes. Same file contents => same hash,
    regardless of filename or upload time."""
    return hashlib.sha256(file_bytes).hexdigest()


# Same six columns _results_to_full_dataframe (ui/gradio_app.py) puts real
# datetimes into for a fresh analysis - trend_metrics.py reads these by
# name for MTTR/SLA/volume-trend, so a cache-hit load has to restore the
# exact same dtype or those calculations silently go wrong.
_CACHED_DATETIME_COLUMNS = [
    "Opened At", "Closed At", "Created At", "Resolved At", "Responded At", "Detected At",
]


def _restore_cached_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Un-does what full_df.to_json()/pd.DataFrame(records) does to
    datetime columns on the DB cache round-trip.

    pandas' to_json() serializes a datetime column as epoch-milliseconds
    (or, once we started passing date_format="iso" below, as an ISO 8601
    string); pd.DataFrame(json.loads(...)) then reconstructs that column
    as a plain int64 or string - never back as datetime64. If that's fed
    straight into pd.to_datetime() with no unit given, an epoch-ms int is
    silently misread as epoch-*nanoseconds*, landing every date around
    1970-01-01 instead of its real value - resolution hours, SLA
    compliance, and the volume-trend chart all go quietly wrong, with no
    error raised anywhere. Detecting numeric vs. string per column here
    handles both old rows already sitting in the DB from before this fix
    (numeric epoch-ms) and new ones (ISO strings) without needing a
    migration."""
    for col in _CACHED_DATETIME_COLUMNS:
        if col not in df.columns:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            df[col] = pd.NaT
    return df


def compute_ticket_content_hash(
    short_description: str, description: str, worklog: str, subject: str = "", external_info: str = ""
) -> str:
    """Hash of a ticket's actual content (not its ID) - used to detect
    when an Incident ID has been re-exported with edited text, so the
    cache isn't served stale in that case. subject/external_info default
    to "" so this stays backward-compatible with any other caller."""
    raw = (
        f"{short_description or ''}|{subject or ''}|{description or ''}|"
        f"{worklog or ''}|{external_info or ''}"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_ticket_cache_bulk(ticket_ids: list[str]) -> dict[str, dict]:
    """One query for a whole batch of ticket IDs, since the pipeline needs
    to check every ticket in an upload, not just one at a time."""
    if not ticket_ids:
        return {}
    session = SessionLocal()
    try:
        rows = session.query(TicketCache).filter(TicketCache.ticket_id.in_(ticket_ids)).all()
        return {
            r.ticket_id: {
                "content_hash": r.content_hash,
                "category": r.category,
                "category_confidence": r.category_confidence,
                "category_method": r.category_method,
                "worklog_score": r.worklog_score,
                "worklog_flags": json.loads(r.worklog_flags_json or "[]"),
                "host": r.host,
                "configuration_item": r.configuration_item,
                "llm_worklog_scoring_enabled": r.llm_worklog_scoring_enabled,
                "opened_at": r.opened_at,
                "closed_at": r.closed_at,
                "created_at": r.created_at,
                "resolved_at": r.resolved_at,
                "responded_at": r.responded_at,
                "detected_at": r.detected_at,
            }
            for r in rows
        }
    finally:
        session.close()


def save_ticket_cache_bulk(entries: list[dict]) -> None:
    """Upserts a batch of freshly-processed tickets into the cache.
    Each entry needs: ticket_id, content_hash, category, category_confidence,
    category_method, worklog_score, worklog_flags, host, configuration_item,
    llm_worklog_scoring_enabled, opened_at, closed_at, created_at,
    resolved_at, responded_at, detected_at (the last six may be None)."""
    if not entries:
        return
    session = SessionLocal()
    try:
        existing_by_id = {
            r.ticket_id: r
            for r in session.query(TicketCache)
            .filter(TicketCache.ticket_id.in_([e["ticket_id"] for e in entries]))
            .all()
        }
        for e in entries:
            row = existing_by_id.get(e["ticket_id"])
            if row is None:
                row = TicketCache(ticket_id=e["ticket_id"])
                session.add(row)
            row.content_hash = e["content_hash"]
            row.category = e["category"]
            row.category_confidence = e["category_confidence"]
            row.category_method = e["category_method"]
            row.worklog_score = e["worklog_score"]
            row.worklog_flags_json = json.dumps(e["worklog_flags"])
            row.host = e["host"]
            row.configuration_item = e["configuration_item"]
            row.llm_worklog_scoring_enabled = e["llm_worklog_scoring_enabled"]
            row.opened_at = e.get("opened_at")
            row.closed_at = e.get("closed_at")
            row.created_at = e.get("created_at")
            row.resolved_at = e.get("resolved_at")
            row.responded_at = e.get("responded_at")
            row.detected_at = e.get("detected_at")
            row.updated_at = datetime.now(timezone.utc)
        session.commit()
    finally:
        session.close()


def get_cached_result(
    file_hash: str,
    current_llm_worklog_scoring_enabled: Optional[bool] = None,
) -> Optional[dict]:
    """
    Returns a dict {"full_df": DataFrame, "summary_stats": dict,
    "category_counts": dict, "uploaded_at": str} if this exact file has
    been processed before AND the relevant config hasn't changed since.
    Returns None on cache miss or config mismatch (caller should then
    run the normal LLM pipeline).
    """
    session = SessionLocal()
    try:
        record = session.query(UploadedFile).filter_by(file_hash=file_hash).first()
        if record is None:
            return None

        if (
            current_llm_worklog_scoring_enabled is not None
            and record.llm_worklog_scoring_enabled != current_llm_worklog_scoring_enabled
        ):
            # Config changed since this file was last processed —
            # treat as a miss so results reflect current settings.
            return None

        payload = json.loads(record.results_json)
        return {
            "full_df": _restore_cached_datetime_columns(pd.DataFrame(payload["full_df"])),
            "summary_stats": payload["summary_stats"],
            "category_counts": payload["category_counts"],
            "host_counts": payload.get("host_counts", {}),
            "uploaded_at": record.uploaded_at.isoformat(),
        }
    finally:
        session.close()


def save_result(
    file_hash: str,
    filename: str,
    full_df: pd.DataFrame,
    summary_stats: dict,
    category_counts: dict,
    host_counts: dict,
    llm_worklog_scoring_enabled: bool = False,
    pipeline_version: str = "v1",
) -> None:
    """Persist processed results for a newly-analyzed file. full_df should
    be the untruncated result set (same shape used for CSV export) so the
    cached load can rebuild the on-screen truncated view from it."""
    payload = json.dumps(
        {
            # date_format="iso" avoids the epoch-ms/nanosecond ambiguity
            # _restore_cached_datetime_columns above has to work around for
            # rows saved before this fix - new rows serialize their six
            # date columns as unambiguous ISO 8601 strings instead.
            "full_df": json.loads(full_df.to_json(orient="records", date_format="iso")),
            "summary_stats": summary_stats,
            "category_counts": category_counts,
            "host_counts": host_counts,
        }
    )

    session = SessionLocal()
    try:
        existing = session.query(UploadedFile).filter_by(file_hash=file_hash).first()
        if existing:
            # Same hash re-saved (e.g. forced re-run) — overwrite in place.
            existing.results_json = payload
            existing.ticket_count = len(full_df)
            existing.llm_worklog_scoring_enabled = llm_worklog_scoring_enabled
            existing.pipeline_version = pipeline_version
            existing.uploaded_at = datetime.now(timezone.utc)
        else:
            record = UploadedFile(
                file_hash=file_hash,
                filename=filename,
                ticket_count=len(full_df),
                llm_worklog_scoring_enabled=llm_worklog_scoring_enabled,
                pipeline_version=pipeline_version,
                results_json=payload,
            )
            session.add(record)

        session.commit()
    finally:
        session.close()


def file_metadata(file_hash: str) -> Optional[dict]:
    """Small helper for surfacing 'this file was already processed on X' in the UI."""
    session = SessionLocal()
    try:
        record = session.query(UploadedFile).filter_by(file_hash=file_hash).first()
        if record is None:
            return None
        return {
            "filename": record.filename,
            "uploaded_at": record.uploaded_at.isoformat(),
            "ticket_count": record.ticket_count,
        }
    finally:
        session.close()


# ==========================================================================
# Step 13 - Knowledge Base / RAG persistence.
#
# Mirrors the UploadedFile/TicketCache pattern above: same engine, same
# SessionLocal, same try/finally session handling - a second concern
# living in the same DB file rather than a separate storage system.
# knowledge_base.py is the only caller of these; it never touches the
# SQLAlchemy session/model objects directly, so a store swap later only
# has to change this section.
# ==========================================================================

def create_kb_document(document_name: str, document_type: str) -> int:
    """Creates a document row in "Processing" status and returns its id,
    before extraction/chunking/embedding even starts - so the UI can show
    the document as soon as upload begins, not only once indexing finishes."""
    session = SessionLocal()
    try:
        record = KnowledgeDocument(document_name=document_name, document_type=document_type, status="Processing")
        session.add(record)
        session.commit()
        return record.id
    finally:
        session.close()


def update_kb_document_status(
    document_id: int,
    status: str,
    error_message: Optional[str] = None,
    chunk_count: Optional[int] = None,
    source_units_json: Optional[str] = None,
) -> None:
    """Moves a document to "Indexed" or "Failed" once ingestion finishes
    (or fails) - see knowledge_base.ingest_document. Only the fields
    passed are updated; the rest keep their current value."""
    session = SessionLocal()
    try:
        record = session.query(KnowledgeDocument).filter_by(id=document_id).first()
        if record is None:
            return
        record.status = status
        if error_message is not None:
            record.error_message = error_message
        if chunk_count is not None:
            record.chunk_count = chunk_count
        if source_units_json is not None:
            record.source_units_json = source_units_json
        session.commit()
    finally:
        session.close()


def save_kb_chunks(document_id: int, chunks: list[dict]) -> None:
    """Persists the chunks produced for one document. Replaces any chunks
    already stored for this document_id first (idempotent re-save - used
    by both first-time ingestion and reindex_document)."""
    session = SessionLocal()
    try:
        session.query(KnowledgeChunk).filter_by(document_id=document_id).delete()
        for i, chunk in enumerate(chunks):
            session.add(KnowledgeChunk(
                document_id=document_id,
                chunk_index=i,
                text=chunk["text"],
                embedding_json=json.dumps(chunk["embedding"]) if chunk.get("embedding") is not None else None,
                page_number=chunk.get("page_number"),
                section_title=chunk.get("section_title"),
            ))
        session.commit()
    finally:
        session.close()


def list_kb_documents() -> list[dict]:
    """All documents, newest first - backs the Knowledge Base tab's
    status table (Document Name | Type | Status)."""
    session = SessionLocal()
    try:
        records = session.query(KnowledgeDocument).order_by(KnowledgeDocument.uploaded_at.desc()).all()
        return [
            {
                "id": r.id,
                "document_name": r.document_name,
                "document_type": r.document_type,
                "status": r.status,
                "error_message": r.error_message,
                "chunk_count": r.chunk_count,
                "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None,
            }
            for r in records
        ]
    finally:
        session.close()


def get_kb_document(document_id: int) -> Optional[dict]:
    """Single document's metadata plus its stored source_units_json -
    used by reindex_document, which re-chunks/re-embeds that text
    without needing the original file again."""
    session = SessionLocal()
    try:
        r = session.query(KnowledgeDocument).filter_by(id=document_id).first()
        if r is None:
            return None
        return {
            "id": r.id,
            "document_name": r.document_name,
            "document_type": r.document_type,
            "status": r.status,
            "source_units_json": r.source_units_json,
        }
    finally:
        session.close()


def delete_kb_document(document_id: int) -> bool:
    """Removes a document and its chunks. Returns False if no such
    document existed (caller can surface that as a no-op, not an error)."""
    session = SessionLocal()
    try:
        record = session.query(KnowledgeDocument).filter_by(id=document_id).first()
        if record is None:
            return False
        session.query(KnowledgeChunk).filter_by(document_id=document_id).delete()
        session.delete(record)
        session.commit()
        return True
    finally:
        session.close()


def get_all_kb_chunks() -> list[dict]:
    """Every chunk across every Indexed document, joined with its parent
    document's name/type - this is the full retrieval corpus. Loaded
    fresh per query rather than kept in a long-lived in-memory index:
    simplest possible correct implementation for a first version, and
    fast enough at the chunk counts a first knowledge base will realistically
    have (see knowledge_base.py's retrieval docstring for the tradeoff)."""
    session = SessionLocal()
    try:
        rows = (
            session.query(KnowledgeChunk, KnowledgeDocument)
            .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
            .filter(KnowledgeDocument.status == "Indexed")
            .all()
        )
        return [
            {
                "text": chunk.text,
                "embedding": json.loads(chunk.embedding_json) if chunk.embedding_json else None,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "document_name": doc.document_name,
                "document_type": doc.document_type,
            }
            for chunk, doc in rows
        ]
    finally:
        session.close()
