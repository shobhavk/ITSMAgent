"""
Ingests Excel / CSV / free-form text into a common raw pandas DataFrame.
This layer does NOT validate business rules (see validator.py) - it only
handles "get bytes into a table" concerns: encoding detection, delimiter
sniffing, header discovery, and turning unstructured text into pseudo-rows.
"""
import io
import logging
import re

import chardet
import pandas as pd

from app.config import get_settings
from app.security import UploadValidationError, check_dataframe_limits, validate_incident_bytes

logger = logging.getLogger(__name__)

# Common ITSM column name variants -> canonical name
COLUMN_ALIASES: dict[str, str] = {
    "incident id": "ticket_id",
    "incident number": "ticket_id",
    "ticket id": "ticket_id",
    "ticket number": "ticket_id",
    "id": "ticket_id",
    "number": "ticket_id",
    "short description": "short_description",
    "summary": "short_description",
    "title": "short_description",
    # Kept as its own canonical field rather than folded into
    # short_description - some exports carry both a "Short Description"
    # and a separate "Subject" column, and merging them under one alias
    # would collide (two source columns -> one renamed column). Both are
    # combined for categorization in pipeline.py's combined_texts instead.
    "subject": "subject",
    "description": "description",
    "details": "description",
    "worklog": "worklog",
    "work log": "worklog",
    "work notes": "worklog",
    "resolution notes": "worklog",
    "resolution": "worklog",
    # Used to enrich worklog-quality scoring (pipeline.py folds this into
    # the text scorer.heuristic_score sees) - not displayed as its own
    # column, just extra evidence of what was actually done/found.
    "external info": "external_info",
    "external information": "external_info",
    "priority": "priority",
    "state": "status",
    "status": "status",
    "assignment group": "assignment_group",
    "assigned group": "assignment_group",
    "group": "assignment_group",
    "configuration item": "configuration_item",
    "config item": "configuration_item",
    "ci": "configuration_item",
    "cmdb ci": "configuration_item",
    "affected ci": "configuration_item",
    "affected server": "configuration_item",
    "host": "configuration_item",
    "hostname": "configuration_item",
    "server": "configuration_item",
    "server name": "configuration_item",
    "opened": "opened_at",
    "opened at": "opened_at",
    "created": "created_at",
    "created at": "created_at",
    "sys_created_on": "created_at",
    "resolved": "resolved_at",
    "resolved at": "resolved_at",
    "resolution time": "resolved_at",
    "closed": "closed_at",
    "closed at": "closed_at",
    # Optional - present in some ITSM exports (rarely all of them). Used
    # for MTTA/MTTD in trend_metrics.py when available; that module
    # reports "not available" rather than guessing when they're missing.
    "responded": "responded_at",
    "responded at": "responded_at",
    "first response": "responded_at",
    "first response time": "responded_at",
    "acknowledged": "responded_at",
    "acknowledged at": "responded_at",
    "detected": "detected_at",
    "detected at": "detected_at",
    "alert time": "detected_at",
    "incident start": "detected_at",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def load_excel(file_bytes: bytes) -> pd.DataFrame:
    # nrows bounds memory: one row past the cap is enough to detect "too many".
    max_rows = get_settings().MAX_INCIDENT_ROWS + 1
    df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", nrows=max_rows)
    return _normalize_columns(df)


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    max_rows = get_settings().MAX_INCIDENT_ROWS + 1
    encoding = chardet.detect(file_bytes[:20000]).get("encoding") or "utf-8"
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, sep=None, engine="python",
                         nrows=max_rows)
    except Exception:
        # Fall back to utf-8 with comma delimiter. (The previous fallback passed
        # errors="replace", which is not a read_csv argument and always raised
        # TypeError - the right keyword is encoding_errors.)
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8", encoding_errors="replace",
                         nrows=max_rows)
    return _normalize_columns(df)


_TICKET_BLOCK_SPLIT = re.compile(r"\n(?=(?:INC|TICKET|CASE)[-_ ]?\d+)", re.IGNORECASE)
_FIELD_LINE = re.compile(r"^\s*([A-Za-z /]+)\s*[:\-]\s*(.+)$")


def load_unstructured_text(text: str) -> pd.DataFrame:
    """
    Best-effort parse of free-form incident text/logs into pseudo-rows.
    Strategy:
      1. Split on lines that look like a new ticket header (INC123, TICKET-45, ...).
      2. Within each block, pull "Field: value" style lines into columns.
      3. Whatever isn't matched as a field goes into `description`.
    This is intentionally forgiving - unstructured input is messy by
    definition - and always yields at least one row so nothing is silently
    dropped.
    """
    text = text.strip()
    if not text:
        return pd.DataFrame(columns=["ticket_id", "description"])

    blocks = _TICKET_BLOCK_SPLIT.split(text)
    if len(blocks) == 1:
        blocks = [b for b in re.split(r"\n{2,}", text) if b.strip()] or [text]

    rows = []
    for i, block in enumerate(blocks):
        record: dict[str, str] = {}
        leftover_lines = []
        header_match = re.search(r"\b((?:INC|TICKET|CASE)[-_ ]?\d+)\b", block, re.IGNORECASE)

        for line in block.splitlines():
            m = _FIELD_LINE.match(line)
            if m:
                key = m.group(1).strip().lower()
                canonical = COLUMN_ALIASES.get(key, key.replace(" ", "_"))
                record[canonical] = m.group(2).strip()
            elif line.strip() and not (header_match and line.strip() == header_match.group(1).strip()):
                leftover_lines.append(line.strip())

        record["ticket_id"] = header_match.group(1) if header_match else f"TXT-{i+1}"

        if leftover_lines:
            record["description"] = (record.get("description", "") + " " + " ".join(leftover_lines)).strip()

        rows.append(record)

    return pd.DataFrame(rows)


def ingest(filename: str, file_bytes: bytes) -> pd.DataFrame:
    """Dispatches to the right loader based on file extension.

    Defence in depth: the content is re-validated here (routers and the UI
    already do it) and every parser failure is converted to an
    UploadValidationError with a generic message - parser exceptions can
    contain paths or file content and must never reach users. The detailed
    exception goes to the (redacted) log, without the file's content."""
    validate_incident_bytes(filename, file_bytes)
    lower = (filename or "").lower()
    try:
        if lower.endswith((".xlsx", ".xls")):
            df = load_excel(file_bytes)
        elif lower.endswith(".csv"):
            df = load_csv(file_bytes)
        elif lower.endswith(".txt"):
            text = file_bytes.decode(chardet.detect(file_bytes).get("encoding") or "utf-8", errors="replace")
            df = load_unstructured_text(text)
        else:
            raise UploadValidationError("Unsupported file type.", 415)
    except UploadValidationError:
        raise
    except Exception as exc:
        logger.warning("Could not parse upload (%s): %s", type(exc).__name__, exc)
        raise UploadValidationError(
            "The file could not be read. It may be malformed, password-protected, or not in the format its extension claims."
        ) from exc
    check_dataframe_limits(df)
    if df.empty:
        raise UploadValidationError("The file contains no data rows.")
    return df
