"""
Step 13 - Knowledge Base / RAG.

This is a SECOND, separate source of knowledge for the Agent, alongside
the incident-data tools in chat_tools.py/rag.py:

  Incident data (chat_tools.py)  -> "what happened in our incident data?"
  Knowledge Base (this module)   -> "what does our documentation say?"

Pipeline (each step a plain function below, so any step can be tested or
swapped independently):

    uploaded document (PDF/DOCX/TXT/MD)
        |  extract_units()      - per-format text extraction, page/section
        v                         metadata preserved where the format has it
    text units
        |  _chunk_units()       - configurable size/overlap (config.py:
        v                         KB_CHUNK_SIZE / KB_CHUNK_OVERLAP)
    chunks
        |  _embed_texts()       - reuses llm_client.get_embeddings_model(),
        v                         the SAME embedding client categorizer.py
    chunks + embedding vectors   already uses - no second embedding stack
        |  persistence.save_kb_chunks()
        v
    SQLite (same DB/engine persistence.py already uses)

Retrieval (retrieve_relevant_chunks) reverses the last three steps: embed
the query with that same client, load all stored chunk vectors, rank by
cosine similarity (numpy - the same brute-force approach rag.py's
TicketIndex already uses for semantic ticket clustering, not a new vector-
store dependency). Chunk counts a first knowledge base will realistically
have (dozens of documents, low thousands of chunks) make this fast enough
without adding FAISS/Chroma; see the module docstring in rag.py for the
precedent.

Every step degrades instead of failing outward: no embeddings model
configured (LLM_PROVIDER=rule_based) -> chunks are stored with
embedding=None and retrieval falls back to keyword overlap (identical
fallback shape to TicketIndex's _keyword_cluster); a single bad document
fails ingestion for THAT document only (status="Failed", error_message
set) and never raises into the caller; an empty/irrelevant retrieval
returns an empty list rather than forcing a match, so the Agent can say
"I couldn't find a relevant procedure" instead of inventing one.

IMPORTANT: only retrieved chunks are ever sent to the LLM (see rag.py's
search_knowledge_base tool) - never the whole knowledge base.
"""
import io
import json
import logging
import re

import numpy as np

from app.config import get_settings
from app.security import validate_kb_upload
from app.services import persistence
from app.services.llm_client import embeddings_retry, get_embeddings_model

logger = logging.getLogger(__name__)
settings = get_settings()

SUPPORTED_KB_EXTENSIONS = {"pdf", "docx", "txt", "md"}

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


# --------------------------------------------------------------------------
# 1. Text extraction (per format) - each returns list[{"text",
#    "page_number", "section_title"}], one dict per extractable "unit"
#    (a PDF page, a DOCX section, or the whole file for TXT/flat MD).
# --------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Whitespace/control-character cleanup - not semantic cleaning, just
    enough that chunks don't carry binary-extraction artifacts (stray
    control chars, excessive blank lines from PDF page breaks)."""
    if not text:
        return ""
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(file_bytes: bytes) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    units = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            units.append({"text": text, "page_number": page_number, "section_title": None})
    return units


def _extract_docx(file_bytes: bytes) -> list[dict]:
    import docx

    document = docx.Document(io.BytesIO(file_bytes))
    units: list[dict] = []
    current_section: str | None = None
    buffer: list[str] = []

    def flush():
        text = _clean_text("\n".join(buffer))
        if text:
            units.append({"text": text, "page_number": None, "section_title": current_section})
        buffer.clear()

    for para in document.paragraphs:
        style_name = (para.style.name or "") if para.style else ""
        if style_name.lower().startswith("heading"):
            flush()
            current_section = para.text.strip() or current_section
        elif para.text.strip():
            buffer.append(para.text)
    flush()
    return units


def _extract_txt(file_bytes: bytes) -> list[dict]:
    text = _clean_text(file_bytes.decode("utf-8", errors="replace"))
    return [{"text": text, "page_number": None, "section_title": None}] if text else []


def _extract_md(file_bytes: bytes) -> list[dict]:
    """Splits on Markdown headings so each section keeps its heading as
    section_title (a plain-text/TXT-style fallback if there are none)."""
    raw = file_bytes.decode("utf-8", errors="replace")
    matches = list(_MD_HEADING_RE.finditer(raw))
    if not matches:
        text = _clean_text(raw)
        return [{"text": text, "page_number": None, "section_title": None}] if text else []

    units: list[dict] = []
    pre = _clean_text(raw[: matches[0].start()])
    if pre:
        units.append({"text": pre, "page_number": None, "section_title": None})
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = _clean_text(raw[start:end])
        if body:
            units.append({"text": body, "page_number": None, "section_title": m.group(2).strip()})
    return units


def _get_extension(filename: str) -> str:
    return (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""


def extract_units(filename: str, file_bytes: bytes) -> list[dict]:
    """Dispatches to the right extractor by extension. Raises ValueError
    for an unsupported extension or a file that fails to parse (caller -
    ingest_document - catches this and marks the document Failed)."""
    ext = _get_extension(filename)
    if ext not in SUPPORTED_KB_EXTENSIONS:
        raise ValueError(f"Unsupported file type '.{ext}'. Allowed: {sorted(SUPPORTED_KB_EXTENSIONS)}")
    try:
        if ext == "pdf":
            return _extract_pdf(file_bytes)
        if ext == "docx":
            return _extract_docx(file_bytes)
        if ext == "md":
            return _extract_md(file_bytes)
        return _extract_txt(file_bytes)
    except Exception as exc:
        raise ValueError(f"Could not extract text from this {ext.upper()} file: {exc}") from exc


# --------------------------------------------------------------------------
# 2. Chunking - character-size target, word-boundary safe, configurable
#    size/overlap (config.py). Chunks never span across a unit boundary
#    (a page or a section), so a chunk's page/section citation is always
#    exactly correct, never a guess about which side of the boundary it
#    came from.
# --------------------------------------------------------------------------

def _chunk_plain_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    n = len(words)
    while start < n:
        current: list[str] = []
        current_len = 0
        idx = start
        while idx < n and current_len + len(words[idx]) + 1 <= chunk_size:
            current.append(words[idx])
            current_len += len(words[idx]) + 1
            idx += 1
        if not current:  # a single word longer than chunk_size - take it whole
            current = [words[idx]]
            idx += 1
        chunks.append(" ".join(current))
        if idx >= n:
            break

        # Step back by ~`overlap` characters worth of words for the next
        # chunk's start, so consecutive chunks share trailing/leading
        # context instead of cutting cleanly at the boundary.
        overlap_words, overlap_len, j = 0, 0, idx - 1
        while j >= start and overlap_len < overlap:
            overlap_len += len(words[j]) + 1
            overlap_words += 1
            j -= 1
        start = max(idx - overlap_words, start + 1)
    return chunks


def _chunk_units(units: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for unit in units:
        for piece in _chunk_plain_text(unit["text"], chunk_size, overlap):
            chunks.append({
                "text": piece,
                "page_number": unit.get("page_number"),
                "section_title": unit.get("section_title"),
            })
    return chunks


# --------------------------------------------------------------------------
# 3. Embeddings - same client/retry policy categorizer.py already uses.
#    Returns None (not a list of Nones) if no embeddings model is
#    configured at all, so callers can tell "no embeddings available"
#    apart from "embedded, all-zero vectors".
# --------------------------------------------------------------------------

async def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    if not texts:
        return []
    embeddings_model = get_embeddings_model()
    if embeddings_model is None:
        return None
    try:
        embed_fn = embeddings_retry(embeddings_model.aembed_documents)
        return await embed_fn(texts)
    except Exception as exc:
        logger.warning("KB embedding call failed, storing chunks without embeddings: %s", exc)
        return None


# --------------------------------------------------------------------------
# 4. Ingestion / reindex / document management - the public surface the
#    Gradio UI and the search_knowledge_base tool (rag.py) call.
# --------------------------------------------------------------------------

async def ingest_document(filename: str, file_bytes: bytes) -> dict:
    """Runs the full pipeline for one uploaded document and persists the
    result. Never raises - a failure at any step marks this one document
    "Failed" with a human-readable error_message and returns that, so one
    bad upload can't take down the rest of the knowledge base or the
    calling UI action."""
    try:
        validate_kb_upload(filename, len(file_bytes))
    except ValueError as exc:
        return {"success": False, "document_id": None, "status": "Failed", "error": str(exc), "chunk_count": 0}

    document_id = persistence.create_kb_document(document_name=filename, document_type=_get_extension(filename))
    try:
        units = extract_units(filename, file_bytes)
        if not units:
            raise ValueError("No extractable text was found in this document.")

        chunks = _chunk_units(units, settings.KB_CHUNK_SIZE, settings.KB_CHUNK_OVERLAP)
        if not chunks:
            raise ValueError("Document produced no usable chunks after cleaning.")

        vectors = await _embed_texts([c["text"] for c in chunks])
        for i, c in enumerate(chunks):
            c["embedding"] = vectors[i] if vectors is not None else None

        persistence.save_kb_chunks(document_id, chunks)
        persistence.update_kb_document_status(
            document_id, status="Indexed", chunk_count=len(chunks),
            source_units_json=json.dumps(units), error_message="",
        )
        return {"success": True, "document_id": document_id, "status": "Indexed", "error": None, "chunk_count": len(chunks)}
    except Exception as exc:
        logger.warning("KB ingestion failed for %s: %s", filename, exc)
        persistence.update_kb_document_status(document_id, status="Failed", error_message=str(exc))
        return {"success": False, "document_id": document_id, "status": "Failed", "error": str(exc), "chunk_count": 0}


async def reindex_document(document_id: int) -> dict:
    """Re-chunks and re-embeds a document from its already-extracted text
    (no need to re-upload) - useful after changing KB_CHUNK_SIZE/OVERLAP,
    or to retry a document that failed only at the embedding step."""
    try:
        doc = persistence.get_kb_document(document_id)
        if doc is None:
            return {"success": False, "error": "Document not found."}

        units = json.loads(doc["source_units_json"]) if doc.get("source_units_json") else []
        if not units:
            raise ValueError("No stored source text for this document - re-upload it instead.")

        chunks = _chunk_units(units, settings.KB_CHUNK_SIZE, settings.KB_CHUNK_OVERLAP)
        if not chunks:
            raise ValueError("Reindex produced no usable chunks.")

        vectors = await _embed_texts([c["text"] for c in chunks])
        for i, c in enumerate(chunks):
            c["embedding"] = vectors[i] if vectors is not None else None

        persistence.save_kb_chunks(document_id, chunks)
        persistence.update_kb_document_status(document_id, status="Indexed", chunk_count=len(chunks), error_message="")
        return {"success": True, "chunk_count": len(chunks)}
    except Exception as exc:
        logger.warning("KB reindex failed for document %s: %s", document_id, exc)
        try:
            persistence.update_kb_document_status(document_id, status="Failed", error_message=str(exc))
        except Exception:
            pass
        return {"success": False, "error": str(exc)}


def list_documents() -> list[dict]:
    try:
        return persistence.list_kb_documents()
    except Exception as exc:
        logger.warning("Listing KB documents failed: %s", exc)
        return []


def delete_document(document_id: int) -> bool:
    try:
        return persistence.delete_kb_document(document_id)
    except Exception as exc:
        logger.warning("Deleting KB document %d failed: %s", document_id, exc)
        return False


# --------------------------------------------------------------------------
# 5. Retrieval - the only function the search_knowledge_base tool calls.
# --------------------------------------------------------------------------

async def retrieve_relevant_chunks(query: str, top_k: int | None = None) -> list[dict]:
    """Returns up to `top_k` chunks relevant to `query`, each as
    {"text", "document_name", "document_type", "page_number",
    "section_title", "relevance"}. Empty list means "nothing relevant" -
    callers (rag.py's tool) must not fabricate an answer in that case.

    Semantic (embedding cosine-similarity) search is used whenever an
    embeddings model is configured AND the corpus has embedded chunks;
    chunks below config.KB_MIN_RELEVANCE are dropped rather than
    returned as weak/misleading matches. Falls back to plain keyword
    overlap - same shape as rag.py's TicketIndex - when no embeddings
    model is available or the embedding call itself fails, so the
    knowledge base stays useful even in rule_based mode."""
    top_k = top_k or settings.KB_TOP_K
    query = (query or "").strip()
    if not query:
        return []

    try:
        chunks = persistence.get_all_kb_chunks()
    except Exception as exc:
        logger.warning("KB retrieval failed to load chunks: %s", exc)
        return []
    if not chunks:
        return []

    embedded_chunks = [c for c in chunks if c.get("embedding") is not None]
    embeddings_model = get_embeddings_model()
    if embedded_chunks and embeddings_model is not None:
        try:
            embed_fn = embeddings_retry(embeddings_model.aembed_query)
            qvec = np.array(await embed_fn(query))
            matrix = np.array([c["embedding"] for c in embedded_chunks])
            norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(qvec) or 1e-9)
            sims = (matrix @ qvec) / (norms + 1e-9)
            order = np.argsort(-sims)[:top_k]
            return [
                {
                    "text": embedded_chunks[i]["text"],
                    "document_name": embedded_chunks[i]["document_name"],
                    "document_type": embedded_chunks[i]["document_type"],
                    "page_number": embedded_chunks[i]["page_number"],
                    "section_title": embedded_chunks[i]["section_title"],
                    "relevance": round(float(sims[i]), 3),
                }
                for i in order if float(sims[i]) >= settings.KB_MIN_RELEVANCE
            ]
        except Exception as exc:
            logger.info("KB semantic retrieval failed, falling back to keyword search: %s", exc)

    # Keyword fallback - no relevance score (not comparable to cosine
    # similarity), so callers/UI should treat relevance=None as "matched
    # by keyword" rather than omit the field.
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        return []
    scored = [(sum(1 for t in terms if t in c["text"].lower()), c) for c in chunks]
    scored = [(score, c) for score, c in scored if score > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "text": c["text"], "document_name": c["document_name"], "document_type": c["document_type"],
            "page_number": c["page_number"], "section_title": c["section_title"], "relevance": None,
        }
        for _, c in scored[:top_k]
    ]
