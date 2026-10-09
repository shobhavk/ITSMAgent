import io
import logging
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.models.schemas import AnalysisResponse
from app.security import get_session_key, validate_upload, verify_api_key
from app.services.pipeline import run_pipeline_from_bytes, run_pipeline_from_text
from app.services.session_store import get_last_result, set_last_result

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/session", tags=["session"])
async def new_session(_: str = Depends(verify_api_key)):
    """Returns a fresh session ID. Send it as the X-Session-ID header on
    /analyze, /export/csv and /chat so this user's data is isolated from
    everyone else using the same API key."""
    return {"session_id": uuid.uuid4().hex}


@router.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file(file: UploadFile = File(...), api_key: str = Depends(get_session_key)):
    validate_upload(file)
    try:
        content = await file.read()
        result = await run_pipeline_from_bytes(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail="Failed to analyze the uploaded file.")

    set_last_result(api_key, result)
    return result


@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(raw_text: str, api_key: str = Depends(get_session_key)):
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text is empty.")
    if len(raw_text) > 200_000:
        raise HTTPException(status_code=413, detail="Text input too large (max 200,000 characters).")
    try:
        result = await run_pipeline_from_text(raw_text)
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

    df = pd.DataFrame([r.model_dump() for r in result.results])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=itsm_quality_analysis.csv"},
    )
