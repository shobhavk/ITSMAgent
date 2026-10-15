"""
Security controls:
  1. API key auth (header-based) for every non-health endpoint.
  2. Upload validation: extension allow-list, size cap, MIME sniffing.
  3. Prompt-injection mitigation for any free text that gets embedded into
     an LLM prompt (incident descriptions / worklogs are attacker-reachable
     text - treat them as untrusted input, never as instructions).
  4. Rate limiting (slowapi) wired up in main.py using get_remote_address.
"""
import hashlib
import hmac
import io
import logging
import os
import re
import zipfile
from typing import Annotated

from fastapi import Depends, Header, HTTPException, UploadFile, status

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".txt"}

# Knowledge Base (Step 13) documents are a distinct upload path from
# incident files above - kept as its own set/function rather than
# extending ALLOWED_EXTENSIONS/validate_upload, since incident-file
# validation must keep behaving exactly as it did before this feature.
KB_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Patterns commonly used to try to hijack an LLM via injected instructions
# inside "user content" fields (ticket descriptions, worklogs, etc).
_INJECTION_PATTERNS = [
    # "ignore all previous instructions", "disregard the above rules", ...
    # (the old pattern missed the most common wording: "ignore ALL PREVIOUS instructions")
    r"(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|the\s+|your\s+|these\s+|those\s+)*"
    r"(?:previous|prior|above|earlier|preceding|system|safety)?\s*(?:instructions?|rules|prompts?|directions?|guardrails?|guidelines)",
    r"you are now",
    r"system prompt",
    r"act as (a|an) (?!engineer|technician|admin)",
    r"reveal (your|the) (prompt|instructions|system)",
    r"(?:print|show|output|reveal|display|leak|send|return)\s+(?:me\s+)?(?:the\s+|your\s+|all\s+)?"
    r"(?:api[ _-]?keys?|secrets?|passwords?|credentials?|env(?:ironment)?\s*(?:variables?|vars?)?|tokens?)",
    r"(?:new|updated)\s+(?:system\s+)?instructions?\s*:",
    r"do not (?:tell|inform|mention|warn)\s+the user",
    r"(?:execute|run)\s+(?:the\s+following\s+)?(?:shell\s+|bash\s+|python\s+)?(?:command|code|script)",
    # attempts to close / forge the prompt delimiters or chat-role tags
    r"<<<\s*/?\s*(?:END_)?DATA\s*>>>",
    r"</?(system|assistant|user|tool)>",
    r"```system",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
    """FastAPI dependency: validates the X-API-Key header against configured keys."""
    valid_keys = [k.strip() for k in settings.API_KEYS.split(",") if k.strip()]
    # constant-time comparison against every configured key (no early exit)
    matched = False
    for k in valid_keys:
        matched |= hmac.compare_digest(x_api_key or "", k)
    if not x_api_key or not matched:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )
    return x_api_key


def validate_upload(file: UploadFile) -> None:
    """Validates file extension and (approximate) size before processing."""
    name = (file.filename or "").lower()
    if not any(name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    # UploadFile.size is populated by Starlette when available
    if file.size and file.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.MAX_UPLOAD_MB}MB limit.",
        )


def validate_kb_upload(filename: str, size_bytes: int | None) -> None:
    """Same idea as validate_upload, for Knowledge Base document uploads
    (Step 13) - separate extension allow-list and size cap since these are
    a different kind of file (PDF/DOCX/TXT/MD, often larger) from the
    incident CSV/XLSX uploads validate_upload guards. Raises ValueError
    (not HTTPException) so this can be called from both the FastAPI router
    and the Gradio upload handler without pulling in a FastAPI dependency
    in the latter."""
    name = (filename or "").lower()
    if not any(name.endswith(ext) for ext in KB_ALLOWED_EXTENSIONS):
        raise ValueError(f"Unsupported file type. Allowed: {sorted(KB_ALLOWED_EXTENSIONS)}")
    if size_bytes and size_bytes > settings.KB_MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File exceeds {settings.KB_MAX_UPLOAD_MB}MB limit.")


def sanitize_for_llm(text: str, max_len: int = 4000) -> str:
    """
    Defangs likely prompt-injection attempts inside untrusted free text
    before it is interpolated into an LLM prompt. This does not "clean"
    the text semantically - it just strips/flags override attempts and
    truncates length. Always also wrap this text in clear delimiters in
    the prompt template (see llm_client.py) and instruct the model to
    treat it strictly as data, never as instructions.
    """
    if not text:
        return ""
    text = text[:max_len]
    if _INJECTION_RE.search(text):
        text = _INJECTION_RE.sub("[filtered]", text)
    # Strip characters commonly used to break out of prompt delimiters
    text = text.replace("```", "'''")
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("<<<", "<").replace(">>>", ">")   # content can never forge <<<END_DATA>>>
    return text.strip()


def contains_injection_attempt(text: str) -> bool:
    return bool(text and _INJECTION_RE.search(text))


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")


def make_session_key(api_key: str, session_id: str | None) -> str:
    """Opaque storage key for one user's session data. A SHA-256 digest of
    (api key, session id) - the old "<key>:<id>" string could collide (an API
    key containing ':') and kept the raw API key as a dict key in memory."""
    material = f"{api_key}\x00{session_id or ''}".encode()
    return hashlib.sha256(material).hexdigest()


def get_session_key(
    api_key: str = Depends(verify_api_key),
    x_session_id: Annotated[str | None, Header()] = None,
) -> str:
    """FastAPI dependency: the key under which a user's analysis is stored.

    Without an X-Session-ID header the key is derived from the API key alone
    (the old behaviour, so existing clients keep working - but everyone
    sharing that API key then shares one slot). Set REQUIRE_SESSION_ID=true
    to reject such requests. With the header, each session ID gets its own
    isolated slot under that API key. Create an ID with POST /api/v1/session.
    Treat session IDs as secrets: anyone holding the API key AND the session
    ID can read that session's data.
    """
    if x_session_id is None:
        if settings.REQUIRE_SESSION_ID:
            raise HTTPException(status_code=400, detail="X-Session-ID header is required (create one with POST /api/v1/session).")
        return make_session_key(api_key, None)
    if not _SESSION_ID_RE.match(x_session_id):
        raise HTTPException(status_code=400, detail="X-Session-ID must be 8-64 characters: letters, digits, '-' or '_'.")
    return make_session_key(api_key, x_session_id)


# ==========================================================================
# Step 16 - content-level upload validation
#
# Filenames and extensions are attacker-controlled, so nothing below trusts
# them: the extension is only an allow-list gate, and the BYTES are then
# checked to actually look like that format. Nothing is ever written to disk
# under the client's filename, and nothing is executed.
# ==========================================================================

class UploadValidationError(ValueError):
    """A rejected upload. The message is written for END USERS (it never
    contains parser internals or file content) and is safe to show in the UI
    or API. `status_code` is the HTTP code the REST layer should use."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# Leading bytes of formats that must never be accepted as text (or as the
# wrong binary format). Checked on the content, not the claimed extension.
_BINARY_MAGIC = {
    b"%PDF-": "PDF",
    b"MZ": "Windows executable",
    b"\x7fELF": "Linux executable",
    b"\x1f\x8b": "gzip archive",
    b"Rar!": "RAR archive",
    b"7z\xbc\xaf": "7-Zip archive",
    b"\xd0\xcf\x11\xe0": "legacy Office (OLE) file",
    b"PK\x03\x04": "ZIP-based file",
    b"GIF8": "image",
    b"\x89PNG": "image",
    b"\xff\xd8\xff": "image",
}
_TEXT_EXTENSIONS_INCIDENT = {".csv", ".txt"}
_EXCEL_EXTENSIONS = {".xlsx", ".xls"}


def safe_filename(name: str | None) -> str:
    """Display-only version of a client filename: no directories, no control
    characters, bounded length. Never use the client's name as a path."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = _CONTROL_RE.sub("", base).strip().strip(".")
    return (base[:100] or "upload")


def _extension(name: str | None) -> str:
    return os.path.splitext(safe_filename(name))[1].lower()


def _looks_binary(content: bytes) -> str | None:
    head = content[:8192]
    for magic, label in _BINARY_MAGIC.items():
        if head.startswith(magic):
            return label
    if b"\x00" in head:
        return "binary data"
    return None


def inspect_zip_container(content: bytes, required_member: str, label: str) -> None:
    """Checks an OOXML container (XLSX / DOCX) WITHOUT extracting it: valid
    zip, expected member present, no path-traversal names, bounded entry
    count and total uncompressed size (zip-bomb guard), no macro payload."""
    max_total = settings.MAX_ARCHIVE_UNCOMPRESSED_MB * 1024 * 1024
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            infos = zf.infolist()
            if len(infos) > settings.MAX_ARCHIVE_ENTRIES:
                raise UploadValidationError(f"This {label} file contains too many internal parts and was rejected.")
            total = 0
            names = set()
            for info in infos:
                n = info.filename
                if n.startswith(("/", "\\")) or ".." in re.split(r"[\\/]", n):
                    raise UploadValidationError(f"This {label} file has an unsafe internal structure and was rejected.")
                total += info.file_size
                if total > max_total:
                    raise UploadValidationError(
                        f"This {label} file expands to more than {settings.MAX_ARCHIVE_UNCOMPRESSED_MB} MB and was rejected.", 413)
                names.add(n)
            if required_member not in names:
                raise UploadValidationError(f"This does not look like a valid {label} file.")
            if any(n.lower().endswith("vbaproject.bin") for n in names):
                raise UploadValidationError(f"{label} files containing macros are not accepted.")
    except UploadValidationError:
        raise
    except Exception:   # BadZipFile, NotImplementedError, RuntimeError (encrypted), ...
        raise UploadValidationError(f"This {label} file is corrupted, encrypted, or not a valid {label} file.")


def validate_incident_bytes(filename: str | None, content: bytes) -> str:
    """Validates the CONTENT of an incident upload (CSV / XLSX / XLS / TXT).
    Returns the sanitised display filename; raises UploadValidationError with
    a user-safe message otherwise."""
    name = safe_filename(filename)
    ext = _extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(f"Unsupported file type '{ext or '(none)'}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}", 415)
    if content is None or len(content) == 0 or not content.strip():
        raise UploadValidationError("The uploaded file is empty.")
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > limit:
        raise UploadValidationError(f"File exceeds {settings.MAX_UPLOAD_MB}MB limit.", 413)

    if ext in _EXCEL_EXTENSIONS:
        if content.startswith(b"\xd0\xcf\x11\xe0"):
            raise UploadValidationError(
                "Legacy binary .xls files are not supported. Please re-save the file as .xlsx or .csv and upload it again.", 415)
        if not content.startswith(b"PK\x03\x04"):
            raise UploadValidationError("This does not look like a valid Excel (.xlsx) file.")
        inspect_zip_container(content, "xl/workbook.xml", "Excel")
    else:   # .csv / .txt must be text
        kind = _looks_binary(content)
        if kind:
            raise UploadValidationError(f"The file content is not text (it looks like a {kind}); it was rejected.", 415)
    return name


def validate_kb_content(filename: str | None, content: bytes) -> str:
    """Content-level checks for Knowledge Base documents (PDF/DOCX/TXT/MD).
    Complements validate_kb_upload (extension + size)."""
    name = safe_filename(filename)
    ext = _extension(filename)
    if ext not in KB_ALLOWED_EXTENSIONS:
        raise UploadValidationError(f"Unsupported file type '{ext or '(none)'}'. Allowed: {sorted(KB_ALLOWED_EXTENSIONS)}", 415)
    if not content or not content.strip():
        raise UploadValidationError("The uploaded document is empty.")
    if len(content) > settings.KB_MAX_UPLOAD_MB * 1024 * 1024:
        raise UploadValidationError(f"File exceeds {settings.KB_MAX_UPLOAD_MB}MB limit.", 413)
    if ext == ".pdf":
        if not content.lstrip()[:5] == b"%PDF-":
            raise UploadValidationError("This does not look like a valid PDF file.")
    elif ext == ".docx":
        if not content.startswith(b"PK\x03\x04"):
            raise UploadValidationError("This does not look like a valid Word (.docx) file.")
        inspect_zip_container(content, "word/document.xml", "Word")
    else:  # .txt / .md
        kind = _looks_binary(content)
        if kind:
            raise UploadValidationError(f"The file content is not text (it looks like a {kind}); it was rejected.", 415)
    return name


async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Reads an UploadFile in chunks and aborts as soon as it exceeds
    max_bytes, so a huge or endless upload cannot be loaded into memory (the
    old code called `await file.read()` first and checked size never)."""
    chunks, total = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadValidationError(f"File exceeds {max_bytes // (1024 * 1024)}MB limit.", 413)
        chunks.append(chunk)
    return b"".join(chunks)


def validate_text_input(text: str | None) -> str:
    """Pasted / posted incident text: non-empty, bounded, no NUL bytes."""
    if text is None or not text.strip():
        raise UploadValidationError("The text input is empty.")
    if len(text) > settings.MAX_TEXT_INPUT_CHARS:
        raise UploadValidationError(f"Text input too large (max {settings.MAX_TEXT_INPUT_CHARS:,} characters).", 413)
    if "\x00" in text:
        raise UploadValidationError("The text input contains invalid characters.")
    return text


def check_dataframe_limits(df) -> None:
    """Row/column caps after parsing (a small file can still expand into a
    huge table); over-long cells are truncated in place."""
    if df is None or len(df.columns) == 0:
        raise UploadValidationError("No readable table was found in the file.")
    if len(df.columns) > settings.MAX_INCIDENT_COLUMNS:
        raise UploadValidationError(f"The file has too many columns (max {settings.MAX_INCIDENT_COLUMNS}).", 413)
    if len(df) > settings.MAX_INCIDENT_ROWS:
        raise UploadValidationError(f"The file has too many rows (max {settings.MAX_INCIDENT_ROWS:,}).", 413)
    cap = settings.MAX_CELL_CHARS
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v[:cap] if isinstance(v, str) and len(v) > cap else v)


# --------------------------------------------------------------------------
# CSV / spreadsheet formula injection (a.k.a. CSV injection)
# --------------------------------------------------------------------------
_FORMULA_START_RE = re.compile(r"^[=+\-@\t\r]")
_PLAIN_NUMBER_RE = re.compile(r"^[+\-]?\d+(?:[.,]\d+)?$")


def neutralize_formula_cell(value):
    """Incident text is attacker-reachable. A cell like =HYPERLINK(...) or
    =cmd|'/c calc'!A0 would EXECUTE when the exported CSV is opened in Excel.
    Prefixing a single quote makes spreadsheets treat it as text. Plain
    numbers (e.g. -5) are left alone."""
    if isinstance(value, str) and _FORMULA_START_RE.match(value) and not _PLAIN_NUMBER_RE.match(value.strip()):
        return "'" + value
    return value


def neutralize_formula_cells(df):
    """Returns a copy of df safe to write with to_csv() for spreadsheet users."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(neutralize_formula_cell)
    return out


# --------------------------------------------------------------------------
# Safe errors + deployment checks
# --------------------------------------------------------------------------
def safe_error_message(action: str, exc: BaseException | None = None) -> str:
    """User-facing error text. Logs the real exception (the log filter
    redacts secrets) but returns only a generic message: exception strings
    can contain file paths, SQL, library internals or credentials."""
    if exc is not None:
        if isinstance(exc, UploadValidationError):
            return str(exc)          # written for end users on purpose
        logger.warning("%s failed: %s: %s", action, type(exc).__name__, exc)
    return f"{action} failed. Please check your input and try again; see the application log for details."


def deployment_warnings() -> tuple[list[str], list[str]]:
    """(fatal, warnings): human-readable configuration problems. main.py logs
    both at startup and refuses to start in ENV=prod if `fatal` is non-empty."""
    fatal, warn = [], []
    keys = [k.strip() for k in settings.API_KEYS.split(",") if k.strip()]
    if not keys or any(k.startswith("change-me") or k.startswith("replace-with") for k in keys) or any(len(k) < 16 for k in keys):
        fatal.append("API_KEYS is unset, a placeholder, or shorter than 16 characters.")
    if "*" in settings.ALLOWED_ORIGINS:
        fatal.append("ALLOWED_ORIGINS contains '*'.")
    if not settings.UI_AUTH_USERS.strip():
        warn.append("The Gradio dashboard has no login (UI_AUTH_USERS is empty): anyone who can reach the port can use it, "
                    "including uploading/deleting Knowledge Base documents. Put it behind an authenticating proxy/SSO or set UI_AUTH_USERS.")
    if not settings.REQUIRE_SESSION_ID:
        warn.append("REQUIRE_SESSION_ID=false: users sharing one API key share one session slot.")
    return fatal, warn


def parse_ui_auth_users() -> list[tuple[str, str]] | None:
    """UI_AUTH_USERS="alice:pw1,bob:pw2" -> [("alice","pw1"), ...] or None."""
    pairs = []
    for item in settings.UI_AUTH_USERS.split(","):
        if ":" in item:
            u, _, p = item.strip().partition(":")
            if u and p:
                pairs.append((u, p))
    return pairs or None


# --------------------------------------------------------------------------
# Request-body size cap (resource exhaustion). Starlette spools a multipart
# upload to disk BEFORE any route code runs, so the per-file checks above come
# too late to stop a multi-gigabyte upload. This ASGI middleware counts bytes
# as they arrive (honouring Content-Length when present, and chunked bodies
# too) and answers 413 as soon as the limit is crossed. A reverse proxy limit
# (e.g. nginx client_max_body_size) is still recommended in front of the app.
# --------------------------------------------------------------------------
class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app, max_bytes: int, path_prefix: str = "/api/"):
        self.app, self.max_bytes, self.prefix = app, max_bytes, path_prefix

    async def _reject(self, send):
        body = b'{"detail":"Request body too large."}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith(self.prefix):
            return await self.app(scope, receive, send)
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        return await self._reject(send)
                except ValueError:
                    return await self._reject(send)
        total, started, exceeded, replaced = 0, False, False, False

        async def limited_receive():
            nonlocal total, exceeded
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal started, replaced
            if exceeded:
                # FastAPI turns exceptions raised while reading the body into its own
                # 400 response, so a raise alone would give the wrong status code: swap
                # whatever the app answers for a 413 and drop the rest of its response.
                if not replaced:
                    replaced = True
                    await self._reject(send)
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started and not replaced:
                await self._reject(send)


# --------------------------------------------------------------------------
# Rate limiting for the REST API (per client IP, fixed window, in-memory).
# Why not slowapi's middleware: it finds the route handler by scanning
# app.routes for `.endpoint`; with current FastAPI, include_router() produces
# route objects without it, so slowapi silently exempted EVERY request and
# RATE_LIMIT had no effect. This one doesn't depend on FastAPI internals.
# Single-process only - use a shared store (Redis) or the reverse proxy when
# running several workers/instances. Behind a proxy, start uvicorn with
# --proxy-headers so the client IP is the real one, not the proxy's.
# --------------------------------------------------------------------------
class RateLimitMiddleware:
    def __init__(self, app, rate: str, path_prefix: str = "/api/"):
        from limits import parse
        from limits.storage import MemoryStorage
        from limits.strategies import FixedWindowRateLimiter

        self.app, self.prefix = app, path_prefix
        self._item = parse(rate)                       # e.g. "30/minute"; raises at startup if malformed
        self._limiter = FixedWindowRateLimiter(MemoryStorage())

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").startswith(self.prefix):
            client = (scope.get("client") or ("unknown", 0))[0]
            if not self._limiter.hit(self._item, "api", client):
                body = b'{"detail":"Rate limit exceeded. Please slow down."}'
                await send({"type": "http.response.start", "status": 429, "headers": [
                    (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), (b"retry-after", b"60")]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
