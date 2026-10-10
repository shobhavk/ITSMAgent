"""
Observability & tracing (Step 15) - structured logs, per-request tracing and
simple in-memory metrics, using only Python's standard `logging`.

HOW A REQUEST IS FOLLOWED
    request_trace()  creates a RequestContext (request_id + pseudonymous session
                     id) and stores it in a ContextVar. ContextVars are copied per
                     asyncio task / worker thread, so two users' concurrent
                     requests can never see each other's request_id.
    stage()          times one step (tool, RAG search, LLM call) and records an
                     event tied to the current request_id.
    @traced_tool     decorator that wraps an EXISTING function (chat_tools.*,
                     recommendations.get_recommendations, knowledge_base.
                     retrieve_relevant_chunks) - no change to what it returns.

WHAT IS (AND ISN'T) RECORDED
    Recorded: request id, pseudonymous session id, operation, component,
              duration, status, error type, tool names, RAG chunk count and
              source document NAMES, LLM model id / attempts / token counts.
    Never recorded: API keys, prompts, retrieved text, incident rows, chat
              history. The question itself is logged as length + short hash only;
              with OBS_DETAILED_DIAGNOSTICS=true a redacted 120-char preview is
              added. Exceptions are logged WITH stack traces to the protected log
              only (secrets redacted); users see a generic message + request id.

METRICS are kept in memory since application start (they reset on restart).
The JSON-lines log file (LOG_FILE_ENABLED=true) is the persisted history.
Observability must never break the agent: every recording call is wrapped in
try/except and failures are swallowed (and counted).
"""
import contextvars
import functools
import hashlib
import hmac
import inspect
import json
import logging
import re
import secrets
import threading
import time
import traceback
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

OBS_LOGGER = logging.getLogger("itsm.observability")
APP_LOG_FORMAT = "%(asctime)s %(levelname)s [req=%(request_id)s] %(name)s: %(message)s"
MIN_SAMPLES_FOR_PERCENTILES = 20


def _cfg(name: str, default=None):
    try:
        from app.config import get_settings
        return getattr(get_settings(), name, default)
    except Exception:
        return default


def _enabled() -> bool:
    return bool(_cfg("OBS_ENABLED", True))


# --------------------------------------------------------------------------
# Redaction + pseudonyms
# --------------------------------------------------------------------------
_REDACT_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|x-api-key|token|secret|password|passwd|authorization)\b(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|\S+)"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
]
_SALT = secrets.token_hex(8)


def _secret_values() -> list[str]:
    vals = []
    for name in ("API_KEYS", "LLM_API_KEY", "AICORE_CLIENT_SECRET", "OBS_ADMIN_TOKEN"):
        raw = _cfg(name, None)
        if raw:
            vals += [v.strip() for v in str(raw).split(",")]
    return sorted({v for v in vals if len(v) >= 6}, key=len, reverse=True)


def redact(text) -> str:
    """Masks configured secrets and common credential patterns."""
    try:
        out = str(text)
        for v in _secret_values():
            out = out.replace(v, "[REDACTED]")
        out = _REDACT_PATTERNS[0].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)
        for rx in _REDACT_PATTERNS[1:]:
            out = rx.sub("[REDACTED]", out)
        return out
    except Exception:
        return "[unprintable]"


def pseudonymize(raw_key) -> str:
    """Stable, non-reversible id for a session/API key (never the key itself)."""
    if not raw_key:
        return "anonymous"
    salt = _cfg("OBS_HASH_SALT", "") or _SALT
    return "s-" + hashlib.sha256((salt + str(raw_key)).encode()).hexdigest()[:10]


def is_admin_token(token) -> bool:
    expected = _cfg("OBS_ADMIN_TOKEN", "")
    return bool(expected) and bool(token) and hmac.compare_digest(str(token), str(expected))


# --------------------------------------------------------------------------
# Request context (one per user request; held in a ContextVar)
# --------------------------------------------------------------------------
@dataclass
class RequestContext:
    request_id: str
    session_id: str
    operation: str
    source: str
    started: float = field(default_factory=time.perf_counter)
    status: str | None = None          # success | degraded | error | client_error
    mode: str | None = None            # "llm" | "rule_based"
    http_status: int | None = None
    question_length: int | None = None
    question_hash: str | None = None
    question_preview: str | None = None
    failures: list = field(default_factory=list)   # components that failed during this request
    exc: BaseException | None = None


_current: contextvars.ContextVar = contextvars.ContextVar("itsm_request", default=None)
_tool_depth: contextvars.ContextVar = contextvars.ContextVar("itsm_tool_depth", default=0)


def current_request() -> RequestContext | None:
    return _current.get()


def current_request_id() -> str | None:
    ctx = _current.get()
    return ctx.request_id if ctx else None


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


# --------------------------------------------------------------------------
# In-memory metrics (since application start)
# --------------------------------------------------------------------------
def _pct(sorted_vals, p):
    if len(sorted_vals) < MIN_SAMPLES_FOR_PERCENTILES:
        return None
    idx = max(0, min(len(sorted_vals) - 1, int(round(p / 100 * len(sorted_vals) + 0.5)) - 1))
    return round(sorted_vals[idx], 1)


class _Agg:
    def __init__(self):
        self.count = 0
        self.errors = 0
        self.total_ms = 0.0
        self.samples = deque(maxlen=1000)

    def add(self, ms, is_error):
        self.count += 1
        self.errors += 1 if is_error else 0
        self.total_ms += ms
        self.samples.append(ms)

    def summary(self):
        s = sorted(self.samples)
        return {"count": self.count, "failures": self.errors,
                "avg_ms": round(self.total_ms / self.count, 1) if self.count else None,
                "p50_ms": _pct(s, 50), "p95_ms": _pct(s, 95)}


class _Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.recent = deque(maxlen=int(_cfg("OBS_RECENT_EVENTS", 500) or 500))
        self.req_counts = {"success": 0, "degraded": 0, "error": 0, "client_error": 0}
        self.req = _Agg()
        self.tools: dict[str, _Agg] = {}
        self.rag = _Agg()
        self.rag_chunks = 0
        self.rag_empty = 0
        self.llm = _Agg()
        self.llm_retries = 0
        self.tokens = {"input": 0, "output": 0, "total": 0, "calls_with_usage": 0, "calls_without_usage": 0}
        self.recording_failures = 0

    def add(self, e: dict):
        comp, status, ms = e["component"], e["status"], e.get("duration_ms") or 0.0
        bad = status == "error"
        with self.lock:
            if comp == "request":
                self.req_counts[status if status in self.req_counts else "error"] += 1
                self.req.add(ms, bad)
            elif comp == "tool":
                self.tools.setdefault(e["operation"], _Agg()).add(ms, bad)
            elif comp == "rag":
                self.rag.add(ms, bad)
                n = e.get("chunk_count") or 0
                self.rag_chunks += n
                self.rag_empty += 1 if (not bad and n == 0) else 0
            elif comp == "llm":
                self.llm.add(ms, bad)
                self.llm_retries += e.get("retry_count") or 0
                if e.get("total_tokens") is not None or e.get("input_tokens") is not None:
                    self.tokens["calls_with_usage"] += 1
                    self.tokens["input"] += e.get("input_tokens") or 0
                    self.tokens["output"] += e.get("output_tokens") or 0
                    self.tokens["total"] += e.get("total_tokens") or ((e.get("input_tokens") or 0) + (e.get("output_tokens") or 0))
                elif not bad:
                    self.tokens["calls_without_usage"] += 1
            self.recent.append({k: e.get(k) for k in ("timestamp", "request_id", "session_id", "component", "operation", "duration_ms", "status", "error_type")})

    def snapshot(self) -> dict:
        with self.lock:
            counted = self.req_counts["success"] + self.req_counts["degraded"] + self.req_counts["error"]
            total = counted + self.req_counts["client_error"]
            tools = {name: a.summary() for name, a in sorted(self.tools.items())}
            all_tools = _Agg()
            for a in self.tools.values():
                all_tools.count += a.count; all_tools.errors += a.errors; all_tools.total_ms += a.total_ms
            rag = self.rag.summary()
            rag.update(total_chunks=self.rag_chunks, empty_results=self.rag_empty,
                       avg_chunks=round(self.rag_chunks / self.rag.count, 2) if self.rag.count else None)
            llm = self.llm.summary()
            llm["retries"] = self.llm_retries
            return {
                "scope": "since application start (in-memory; resets on restart)",
                "since": self.started_at,
                "requests": {**self.req_counts, "total": total, "failed": self.req_counts["error"],
                             "success_rate_pct": round(100 * self.req_counts["success"] / counted, 1) if counted else None,
                             **{k: v for k, v in self.req.summary().items() if k.endswith("_ms")}},
                "tools": {"total_calls": all_tools.count, "total_failures": all_tools.errors,
                          "avg_ms": round(all_tools.total_ms / all_tools.count, 1) if all_tools.count else None, "by_tool": tools},
                "rag": rag,
                "llm": llm,
                "tokens": dict(self.tokens),
                "recording_failures": self.recording_failures,
            }

    def recent_events(self, limit=50) -> list[dict]:
        with self.lock:
            return list(self.recent)[-limit:][::-1]


_METRICS = _Metrics()


def get_metrics() -> dict:
    return _METRICS.snapshot()


def get_recent_events(limit: int = 50) -> list[dict]:
    return _METRICS.recent_events(limit)


def reset_metrics() -> None:
    _METRICS.reset()


# --------------------------------------------------------------------------
# Event recording
# --------------------------------------------------------------------------
def _emit(event: dict, exc: BaseException | None = None) -> None:
    """Never raises."""
    try:
        if not _enabled():
            return
        _METRICS.add(event)
        level = logging.ERROR if event["status"] == "error" else (logging.WARNING if event["status"] == "degraded" else logging.INFO)
        if OBS_LOGGER.isEnabledFor(level):
            exc_info = (type(exc), exc, exc.__traceback__) if exc is not None else None
            OBS_LOGGER.log(level, "%s.%s %s", event["component"], event["operation"], event["status"],
                           extra={"obs_event": event}, exc_info=exc_info)
    except Exception:
        try:
            _METRICS.recording_failures += 1
        except Exception:
            pass


def _base_event(component, operation, status, duration_ms, ctx, fields) -> dict:
    e = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
         "request_id": ctx.request_id if ctx else None,
         "session_id": ctx.session_id if ctx else None,
         "component": component, "operation": operation, "status": status,
         "duration_ms": round(duration_ms, 1)}
    e.update({k: v for k, v in fields.items() if v is not None})
    return e


def record_event(component, operation, status="ok", duration_ms=0.0, exc=None, **fields) -> None:
    try:
        ctx = _current.get()
        if exc is not None:
            fields.setdefault("error_type", type(exc).__name__)
            if ctx is not None:
                ctx.failures.append(component)
        elif status == "error" and ctx is not None:
            ctx.failures.append(component)
        _emit(_base_event(component, operation, status, duration_ms, ctx, fields), exc)
    except Exception:
        pass


@contextmanager
def stage(component: str, operation: str, **fields):
    """Times one step. The yielded dict lets the caller attach fields
    (`s["chunk_count"] = 3`) or mark a non-exception failure (`s["status"] = "error"`)."""
    holder: dict = {}
    t0 = time.perf_counter()
    try:
        yield holder
    except Exception as exc:
        record_event(component, operation, "error", (time.perf_counter() - t0) * 1000, exc=exc, **{**fields, **holder})
        raise
    else:
        status = holder.pop("status", "ok")
        record_event(component, operation, status, (time.perf_counter() - t0) * 1000, **{**fields, **holder})


@contextmanager
def request_trace(operation: str, session_key=None, source: str = "api"):
    """Wraps one user request. If a request is already active (e.g. the REST
    middleware started it) the existing one is reused, so the agent code can
    always call this and the UI/API share one code path."""
    existing = _current.get()
    if existing is not None or not _enabled():
        yield existing
        return
    ctx = RequestContext(new_request_id(), pseudonymize(session_key), operation, source)
    token = _current.set(ctx)
    try:
        yield ctx
    except Exception as exc:
        ctx.exc = exc
        ctx.status = "error"
        raise
    finally:
        try:
            _current.reset(token)
        except Exception:
            pass
        _finish_request(ctx)


def _finish_request(ctx: RequestContext) -> None:
    try:
        status = ctx.status or ("degraded" if ctx.failures else "success")
        if status == "success" and ctx.failures:
            status = "degraded"
        ms = (time.perf_counter() - ctx.started) * 1000
        fields = dict(source=ctx.source, mode=ctx.mode, http_status=ctx.http_status,
                      question_length=ctx.question_length, question_hash=ctx.question_hash,
                      question_preview=ctx.question_preview,
                      failed_components=sorted(set(ctx.failures)) or None,
                      error_type=type(ctx.exc).__name__ if ctx.exc else None)
        _emit(_base_event("request", ctx.operation, status, ms, ctx, fields), ctx.exc)
    except Exception:
        pass


def note_question(question: str) -> None:
    """Records question LENGTH + short hash (not the text) on the current request."""
    try:
        ctx = _current.get()
        if ctx is None:
            return
        ctx.question_length = len(question)
        ctx.question_hash = hashlib.sha256(question.encode()).hexdigest()[:8]
        if _cfg("OBS_DETAILED_DIAGNOSTICS", False):
            ctx.question_preview = redact(question[:120])
    except Exception:
        pass


def set_mode(mode: str) -> None:
    ctx = _current.get()
    if ctx is not None:
        ctx.mode = mode


# --------------------------------------------------------------------------
# Tool / RAG decorator
# --------------------------------------------------------------------------
def traced_tool(name: str, component: str = "tool", extract=None):
    """Wraps an existing sync or async function. Result is returned unchanged.
    component="tool" for incident analytics, "rag" for knowledge-base search.
    `extract(result) -> dict` adds safe fields (e.g. chunk count, document names).
    A result of the form {"error": ...} counts as a failed tool call."""
    def decorator(fn):
        def finish(t0, result=None, exc=None):
            try:
                ms = (time.perf_counter() - t0) * 1000
                fields = {}
                status = "ok"
                if exc is not None:
                    status = "error"
                elif isinstance(result, dict) and "error" in result:
                    status, fields["error_type"] = "error", "ToolReturnedError"
                if extract is not None and exc is None:
                    fields.update(extract(result) or {})
                record_event(component, name, status, ms, exc=exc, **fields)
            except Exception:
                pass

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*a, **kw):
                t0 = time.perf_counter()
                try:
                    result = await fn(*a, **kw)
                except Exception as exc:
                    finish(t0, exc=exc)
                    raise
                finish(t0, result)
                return result
            return awrapper

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            nested = component == "tool" and _tool_depth.get() > 0   # e.g. recommendations -> chat_tools
            if nested:
                return fn(*a, **kw)
            token = _tool_depth.set(_tool_depth.get() + 1)
            t0 = time.perf_counter()
            try:
                result = fn(*a, **kw)
            except Exception as exc:
                finish(t0, exc=exc)
                raise
            finally:
                _tool_depth.reset(token)
            finish(t0, result)
            return result
        return wrapper
    return decorator


def kb_fields(chunks) -> dict:
    """Safe RAG fields: counts + source document NAMES only, never text."""
    docs = sorted({c.get("document_name") for c in (chunks or []) if c.get("document_name")})
    return {"chunk_count": len(chunks or []), "documents": docs, "retrieval_status": "hit" if chunks else "empty"}


# --------------------------------------------------------------------------
# LLM helper (used by rag.py)
# --------------------------------------------------------------------------
def extract_token_usage(response) -> dict:
    """Returns only what the provider actually reported; {} if nothing."""
    try:
        um = getattr(response, "usage_metadata", None)
        if um:
            return {"input_tokens": um.get("input_tokens"), "output_tokens": um.get("output_tokens"), "total_tokens": um.get("total_tokens")}
        tu = (getattr(response, "response_metadata", None) or {}).get("token_usage")
        if tu:
            return {"input_tokens": tu.get("prompt_tokens"), "output_tokens": tu.get("completion_tokens"), "total_tokens": tu.get("total_tokens")}
    except Exception:
        pass
    return {}


def _make_attempt_counter():
    from langchain_core.callbacks import BaseCallbackHandler

    class _Counter(BaseCallbackHandler):
        def __init__(self):
            self.starts = 0

        def on_chat_model_start(self, *a, **k):
            self.starts += 1

        def on_llm_start(self, *a, **k):
            self.starts += 1
    return _Counter()


async def traced_llm_ainvoke(runnable, messages, operation: str = "agent_turn"):
    """`await runnable.ainvoke(messages)` + an LLM event (duration, status,
    model id, attempts/retries, token usage when the provider returns it)."""
    try:
        counter = _make_attempt_counter()
        kwargs = {"config": {"callbacks": [counter]}}
    except Exception:
        counter, kwargs = None, {}
    model = _cfg("CHAT_MODEL_NAME", None) if _cfg("LLM_PROVIDER", "rule_based") != "rule_based" else None
    with stage("llm", operation, model=model) as s:
        try:
            response = await runnable.ainvoke(messages, **kwargs)
        finally:
            if counter is not None and counter.starts:
                s["attempts"], s["retry_count"] = counter.starts, counter.starts - 1
        s.update(extract_token_usage(response))
        return response


def note_tool_selection(tool_names: list, round_no: int) -> None:
    record_event("agent", "tool_selection", "ok", 0.0, tools=list(tool_names), round=round_no)


# --------------------------------------------------------------------------
# Logging setup (idempotent)
# --------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    def format(self, record):
        try:
            data = dict(getattr(record, "obs_event", None) or {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                                                 "message": redact(record.getMessage())})
            data["level"] = record.levelname
            if record.exc_info:
                data["stack_trace"] = redact("".join(traceback.format_exception(*record.exc_info)))
            return json.dumps(data, default=str)
        except Exception:
            return '{"level":"ERROR","message":"log formatting failed"}'


class _RedactingFilter(logging.Filter):
    """Masks secrets in ordinary app log lines and their stack traces."""
    def filter(self, record):
        try:
            record.msg, record.args = redact(record.getMessage()), ()
            if record.exc_info and not record.exc_text:
                record.exc_text = redact("".join(traceback.format_exception(*record.exc_info)))
        except Exception:
            pass
        return True


class _QuietFileHandler(RotatingFileHandler):
    def handleError(self, record):   # a full disk / bad path must never crash the agent
        pass


class _QuietStreamHandler(logging.StreamHandler):
    def handleError(self, record):
        pass


_setup_done = False


def setup_logging() -> None:
    """Call once at startup (app/main.py does). Safe to call repeatedly."""
    global _setup_done
    if _setup_done:
        return
    _setup_done = True
    try:
        old_factory = logging.getLogRecordFactory()

        def factory(*a, **k):
            rec = old_factory(*a, **k)
            ctx = _current.get()
            rec.request_id = ctx.request_id if ctx else "-"
            return rec
        logging.setLogRecordFactory(factory)

        level = str(_cfg("LOG_LEVEL", "INFO")).upper()
        OBS_LOGGER.setLevel(level)
        OBS_LOGGER.propagate = False
        if not OBS_LOGGER.handlers:
            console = _QuietStreamHandler()
            console.setFormatter(_JsonFormatter())
            OBS_LOGGER.addHandler(console)

        if _cfg("LOG_FILE_ENABLED", False):
            log_dir = Path(_cfg("LOG_DIR", "logs"))
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                ev = _QuietFileHandler(log_dir / "itsm_events.jsonl", maxBytes=int(_cfg("LOG_FILE_MAX_BYTES", 5_000_000)),
                                       backupCount=int(_cfg("LOG_FILE_BACKUP_COUNT", 5)), encoding="utf-8")
                ev.setFormatter(_JsonFormatter())
                OBS_LOGGER.addHandler(ev)
                app_h = _QuietFileHandler(log_dir / "itsm_app.log", maxBytes=int(_cfg("LOG_FILE_MAX_BYTES", 5_000_000)),
                                          backupCount=int(_cfg("LOG_FILE_BACKUP_COUNT", 5)), encoding="utf-8")
                app_h.setFormatter(logging.Formatter(APP_LOG_FORMAT))
                app_h.addFilter(_RedactingFilter())
                logging.getLogger().addHandler(app_h)
            except Exception as exc:   # e.g. read-only filesystem: keep running with console logs
                logging.getLogger(__name__).warning("File logging disabled (%s): %s", type(exc).__name__, exc)
    except Exception:
        pass


def protect_root_handlers() -> None:
    """Call after logging.basicConfig(): adds secret redaction to every root handler."""
    try:
        for h in logging.getLogger().handlers:
            if not any(isinstance(f, _RedactingFilter) for f in h.filters):
                h.addFilter(_RedactingFilter())
    except Exception:
        pass
