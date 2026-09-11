"""
Tiny in-memory store of "the last analysis result per API key" - shared by
the analyze router (writes it) and the chat router (reads it, to answer
questions grounded in that batch). Fine for v1 single-instance deployments;
swap for Redis if you scale horizontally.
"""
from app.models.schemas import AnalysisResponse

_LAST_RESULT: dict[str, AnalysisResponse] = {}


def set_last_result(api_key: str, result: AnalysisResponse) -> None:
    _LAST_RESULT[api_key] = result


def get_last_result(api_key: str) -> AnalysisResponse | None:
    return _LAST_RESULT.get(api_key)
