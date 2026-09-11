import logging

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.security import verify_api_key
from app.services import rag
from app.services.session_store import get_last_result

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    result = get_last_result(api_key)
    if not result:
        raise HTTPException(status_code=404, detail="No analysis found for this API key yet. Run /analyze first.")

    index = rag.build_index_from_tickets(result.results)
    stats = {
        "total_records": result.total_records,
        "valid_records": result.valid_records,
        "rejected_records": result.rejected_records,
        "average_worklog_score": result.average_worklog_score,
        "category_counts": result.category_counts,
        "host_counts": result.host_counts,
    }

    try:
        answer = await rag.answer_question(req.question, index, stats, req.history)
    except Exception:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail="Failed to answer the question.")

    return ChatResponse(answer=answer)
