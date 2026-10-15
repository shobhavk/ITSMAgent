import io
import logging
import uuid

import pandas as pd
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models.schemas import AnalysisResponse, TextAnalysisRequest
from app.security import (
    UploadValidationError, get_session_key, neutralize_formula_cells, read_upload_limited,
    validate_incident_bytes, validate_text_input, validate_upload, verify_api_key,
)
from app.services.pipeline import run_pipeline_from_bytes, run_pipeline_from_text
from app.services.session_store import get_last_result, set_last_result

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/session", tags=["session"])
async def new_session(_: str = Depends(verify_api_key)):
    """Returns a fresh session ID. Send it as the X-Session-ID header on
    /analyze, /export/csv and /chat so this user's data is isolated from
    everyone else using the same API key. Treat it like a secret."""
    return {"session_id": uuid.uuid4().hex}


@router.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file(file: UploadFile = File(...), api_key: str = Depends(get_session_key)):
    validate_upload(file)   # cheap early checks: extension + declared size
    try:
        # Bounded read (aborts past the limit), then content-level validation.
        # The client's filename is only ever used as a sanitised display/extension hint.
        content = await read_upload_limited(file, settings.MAX_UPLOAD_MB * 1024 * 1024)
        safe_name = validate_incident_bytes(file.filename, content)
        result = await run_pipeline_from_bytes(safe_name, content)
    except UploadValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except ValueError:
        logger.exception("Upload rejected by pipeline")
        raise HTTPException(status_code=400, detail="The file could not be processed. Check that it is a valid, well-formed incident export.")
    except Exception:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail="Failed to analyze the uploaded file.")

    set_last_result(api_key, result)
    return result


@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(
    payload: TextAnalysisRequest | None = Body(default=None),
    raw_text: str | None = Query(default=None, deprecated=True, description="Deprecated: send a JSON body {\"raw_text\": ...} instead."),
    api_key: str = Depends(get_session_key),
):
    text = payload.raw_text if payload is not None else raw_text
    try:
        validate_text_input(text)
        result = await run_pipeline_from_text(text)
    except UploadValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except Exception:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail="Failed to analyze the provided text.")

    set_last_result(api_key, result)
    return result


@router.get("/export/csv")
async def export_csv(api_key: str = Depends(get_session_key)):
    result = get_last_result(api_key)
    if not result:
        raise HTTPException(status_code=404, detail="No analysis found for this session yet. Run /analyze first.")

    # Incident text is attacker-reachable; neutralise =,+,-,@ cells so the CSV
    # can't run formulas when opened in Excel/Sheets.
    df = neutralize_formula_cells(pd.DataFrame([r.model_dump() for r in result.results]))
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=itsm_quality_analysis.csv"},
    )
