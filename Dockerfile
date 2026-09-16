# --- Build stage: install deps into a venv for a leaner final image ---
FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- Final stage ---
FROM python:3.11-slim

# Security: run as non-root
RUN groupadd -r itsm && useradd -r -g itsm -d /app -s /sbin/nologin itsm

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY ui ./ui

# Pre-create the SQLite data directory so it exists (with correct
# ownership) in the image *before* a named volume is mounted over it -
# Docker copies a mount point's existing image content/permissions into a
# fresh named volume on first use, which is what makes writes from the
# non-root `itsm` user below actually work against docker-compose.yml's
# `itsm_data:/app/data` volume (needed because the compose file also sets
# `read_only: true` on the container - see persistence.py's DB_PATH).
RUN mkdir -p /app/data && chown -R itsm:itsm /app
USER itsm

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
