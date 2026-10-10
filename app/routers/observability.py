"""Operational metrics endpoint (Step 15). Aggregates only - no per-user data."""
from fastapi import APIRouter, Depends

from app.security import verify_api_key
from app.services import observability

router = APIRouter(prefix="/api/v1", tags=["observability"])


@router.get("/metrics")
async def metrics(_: str = Depends(verify_api_key)):
    """Live metrics since application start (in-memory; reset on restart).
    Not the same thing as the Step 14 evaluation scores, which measure answer quality."""
    return observability.get_metrics()
