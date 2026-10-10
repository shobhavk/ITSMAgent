"""
In-memory store of "the last analysis result per session key" - shared by the
analyze router (writes it) and the chat router (reads it). The key is the API
key, or "<api key>:<X-Session-ID>" when the client sends a session ID (see
security.get_session_key), so users are isolated per session.

Bounded so a long-running server cannot grow without limit: entries expire
after SESSION_TTL_SECONDS of inactivity and at most MAX_SESSIONS are kept
(least-recently-used evicted first). Fine for a single instance; swap for
Redis if you scale horizontally.
"""
import threading
import time
from collections import OrderedDict

from app.models.schemas import AnalysisResponse

SESSION_TTL_SECONDS = 8 * 60 * 60
MAX_SESSIONS = 200

_LAST_RESULT: "OrderedDict[str, tuple[float, AnalysisResponse]]" = OrderedDict()
_LOCK = threading.Lock()


def _purge(now: float) -> None:
    for key in [k for k, (ts, _) in _LAST_RESULT.items() if now - ts > SESSION_TTL_SECONDS]:
        del _LAST_RESULT[key]
    while len(_LAST_RESULT) > MAX_SESSIONS:
        _LAST_RESULT.popitem(last=False)


def set_last_result(session_key: str, result: AnalysisResponse) -> None:
    with _LOCK:
        now = time.time()
        _LAST_RESULT[session_key] = (now, result)
        _LAST_RESULT.move_to_end(session_key)
        _purge(now)


def get_last_result(session_key: str) -> AnalysisResponse | None:
    with _LOCK:
        now = time.time()
        _purge(now)
        entry = _LAST_RESULT.get(session_key)
        if entry is None:
            return None
        _LAST_RESULT[session_key] = (now, entry[1])   # refresh activity
        _LAST_RESULT.move_to_end(session_key)
        return entry[1]
