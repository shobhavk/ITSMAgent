import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.routers import analyze, chat
from app.routers import observability as observability_router
from app.services import observability

settings = get_settings()

observability.setup_logging()   # structured event log + request_id on every app log line
logging.basicConfig(level=settings.LOG_LEVEL, format=observability.APP_LOG_FORMAT)
observability.protect_root_handlers()   # redact secrets from log lines / stack traces
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    docs_url="/api/docs" if settings.ENV != "prod" else None,  # hide docs in prod
    redoc_url=None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "X-Session-ID", "Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.middleware("http")
async def request_tracing(request: Request, call_next):
    """Step 15: gives every /api request a unique request_id (also returned in
    the X-Request-ID header), times it, and records its final status. The
    session is identified by a one-way hash of API key (+ X-Session-ID) - the
    key itself is never logged. Gradio's own asset/queue routes are skipped."""
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    sid = request.headers.get("x-session-id")
    key = request.headers.get("x-api-key")
    session_key = (f"{key}:{sid}" if sid else key) if key else None
    with observability.request_trace(f"{request.method} {request.url.path}", session_key, source="api") as ctx:
        if ctx is None:   # observability disabled
            return await call_next(request)
        request.state.request_id = ctx.request_id
        response = await call_next(request)
        ctx.http_status = response.status_code
        if response.status_code >= 500:
            ctx.status = "error"
        elif response.status_code >= 400:
            ctx.status = "client_error"
        response.headers["X-Request-ID"] = ctx.request_id
        return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak stack traces / internals to the client.
    logger.exception("Unhandled exception on %s", request.url.path)
    # The request_id lets you find the full stack trace in the protected logs
    # without exposing any internals to the client.
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(status_code=500, content={"detail": "Internal server error.", "request_id": request_id},
                        headers={"X-Request-ID": request_id} if request_id else None)


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "app": settings.APP_NAME, "provider": settings.LLM_PROVIDER}


app.include_router(analyze.router)
app.include_router(chat.router)
app.include_router(observability_router.router)

# --- Mount Gradio dashboard at the app root so its bundled static assets
# (which reference absolute "/assets/..." paths) resolve correctly. The
# REST API lives under /api/v1/... and /health, both registered above and
# therefore matched before this catch-all mount. ---
from ui.gradio_app import CUSTOM_CSS, ITSM_THEME, build_ui  # noqa: E402  (import after app creation intentional)
import gradio as gr  # noqa: E402

gr.mount_gradio_app(app, build_ui(), path="/", theme=ITSM_THEME, css=CUSTOM_CSS)
