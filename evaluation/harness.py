"""
Shared helpers for the evaluation (Step 14).

What lives here:
  * setup_environment()  - points the app at a THROWAWAY SQLite DB (so the
                           evaluation never touches your real data) and sets
                           two test API keys. Must run BEFORE importing app.*
  * ToolRecorder         - records which agent tools were used, WITHOUT
                           changing rag.py. It wraps the functions the tools
                           call (chat_tools.*, recommendations.get_recommendations,
                           knowledge_base.retrieve_relevant_chunks), so it works
                           for both the LLM tool-calling loop and the
                           rule_based keyword fallback.
  * load_dataset / seed_knowledge_base / ask_agent - thin wrappers that REUSE
                           the existing pipeline, knowledge_base and rag code.
"""
import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

API_KEY_A = "eval-key-user-A"
API_KEY_B = "eval-key-user-B"
API_KEY_SHARED = "eval-key-shared"

AGENT_TOOL_NAMES = [
    "get_incident_summary", "get_category_analysis", "get_priority_analysis",
    "get_server_analysis", "get_recurring_issues", "get_recommendations",
    "search_knowledge_base", "search_incidents",
]


def setup_environment() -> str:
    """Call once, before any `import app...`. Returns the temp dir used."""
    tmp = tempfile.mkdtemp(prefix="itsm_eval_")
    os.environ["ITSM_DB_PATH"] = os.path.join(tmp, "eval.db")
    os.environ["API_KEYS"] = f"{API_KEY_A},{API_KEY_B},{API_KEY_SHARED}"
    os.environ["RATE_LIMIT"] = "100000/minute"
    os.chdir(ROOT)  # so a .env file in the project root is picked up
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return tmp


@dataclass
class ToolCall:
    name: str
    args: dict
    result: object
    docs: list = field(default_factory=list)  # KB document names (search_knowledge_base only)


class ToolRecorder:
    def __init__(self):
        self.calls: list[ToolCall] = []
        self._depth = 0
        self._patched: list[tuple] = []

    def reset(self):
        self.calls = []

    def _sync_wrapper(self, name, fn):
        def wrapper(*args, **kwargs):
            top_level = self._depth == 0   # ignore calls made INSIDE another tool
            self._depth += 1
            try:
                result = fn(*args, **kwargs)
            finally:
                self._depth -= 1
            if top_level:
                self.calls.append(ToolCall(name, dict(kwargs), result))
            return result
        return wrapper

    def _kb_wrapper(self, fn):
        async def wrapper(*args, **kwargs):
            chunks = await fn(*args, **kwargs)
            docs = [c.get("document_name") for c in chunks]
            self.calls.append(ToolCall("search_knowledge_base", {"query": args[0] if args else kwargs.get("query")}, chunks, docs))
            return chunks
        return wrapper

    def install(self):
        from app.services import chat_tools, knowledge_base, recommendations
        for name in ("get_incident_summary", "get_category_analysis", "get_priority_analysis",
                     "get_server_analysis", "get_recurring_issues", "search_incidents"):
            orig = getattr(chat_tools, name)
            setattr(chat_tools, name, self._sync_wrapper(name, orig))
            self._patched.append((chat_tools, name, orig))
        orig = recommendations.get_recommendations
        recommendations.get_recommendations = self._sync_wrapper("get_recommendations", orig)
        self._patched.append((recommendations, "get_recommendations", orig))
        orig = knowledge_base.retrieve_relevant_chunks
        knowledge_base.retrieve_relevant_chunks = self._kb_wrapper(orig)
        self._patched.append((knowledge_base, "retrieve_relevant_chunks", orig))

    def uninstall(self):
        for module, name, orig in self._patched:
            setattr(module, name, orig)
        self._patched = []


async def load_dataset(path: Path):
    """Runs the EXISTING pipeline on a CSV; returns (AnalysisResponse, dataframe).
    The dataframe has the same shape the REST chat endpoint gives the agent."""
    from app.services import chat_tools
    from app.services.pipeline import run_pipeline_from_bytes
    result = await run_pipeline_from_bytes(path.name, path.read_bytes())
    return result, chat_tools.tickets_to_dataframe(result.results)


async def seed_knowledge_base(files: list[Path]) -> list[dict]:
    """Ingests documents through the EXISTING knowledge_base.ingest_document."""
    from app.services import knowledge_base
    out = []
    for f in files:
        res = await knowledge_base.ingest_document(f.name, f.read_bytes())
        out.append({"file": f.name, **res})
    return out


def shared_kb_files() -> list[Path]:
    return sorted(p for p in (FIXTURES / "kb").iterdir() if not p.name.startswith("_"))


async def ask_agent(question, df, recorder: ToolRecorder, history=None, timeout=120):
    """Calls the existing rag.answer_question exactly like the UI/API do."""
    from app.services import rag
    recorder.reset()
    answer = await asyncio.wait_for(rag.answer_question(question, df, {}, history or []), timeout=timeout)
    return answer, list(recorder.calls)
