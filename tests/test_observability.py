"""
Step 15 tests - run with:   python -m pytest tests -v
No real LLM is called: a small fake chat model stands in for it.
"""
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from evaluation.harness import API_KEY_A, API_KEY_B, FIXTURES
from app.services import observability, rag

ROOT = Path(__file__).resolve().parent.parent


# ---------------- helpers ----------------
class FakeModel:
    """Minimal stand-in for the chat model: asks for one tool, then answers."""
    def __init__(self, tool="get_incident_summary", usage=True, fail_with=None, delay=0.0):
        self.tool, self.usage, self.fail_with, self.delay = tool, usage, fail_with, delay

    def bind_tools(self, tools):
        return self

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages, config=None, **kw):
        if config:   # exercise the attempt counter like LangChain does
            for cb in config.get("callbacks", []):
                cb.on_chat_model_start({}, [])
        await asyncio.sleep(self.delay)
        if self.fail_with:
            raise self.fail_with
        um = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15} if self.usage else None
        already_called_tool = any(getattr(m, "type", "") == "tool" for m in messages)
        if already_called_tool:
            return AIMessage(content="Final answer from fake model.", usage_metadata=um)
        return AIMessage(content="", tool_calls=[{"name": self.tool, "args": {}, "id": "call-1"}], usage_metadata=um)


def use_fake_model(monkeypatch, **kw):
    model = FakeModel(**kw)
    monkeypatch.setattr(rag, "get_raw_chat_model", lambda: model)
    monkeypatch.setattr(rag, "with_chat_retry", lambda m: m)
    return model


def ask(question, df, session_key=None):
    async def run():
        with observability.request_trace("chat", session_key, source="test"):
            return await rag.answer_question(question, df)
    return asyncio.run(run())


def events_of(cap, component=None):
    ev = cap.events()
    return [e for e in ev if component is None or e.get("component") == component]


# ---------------- 1-2. request ids, success logged ----------------
def test_each_request_gets_unique_id_and_success_is_logged(datasets, obs_clean):
    df_a, _ = datasets
    for _ in range(5):
        ask("What is the most common incident category?", df_a)
    reqs = events_of(obs_clean, "request")
    assert len(reqs) == 5
    assert len({e["request_id"] for e in reqs}) == 5
    assert all(e["status"] == "success" and e["duration_ms"] >= 0 for e in reqs)
    assert observability.get_metrics()["requests"]["success"] == 5


# ---------------- 3. failed request logged ----------------
def test_failed_request_is_logged_with_error_type(obs_clean):
    with pytest.raises(ValueError):
        with observability.request_trace("chat", "user-x", source="test"):
            raise ValueError("boom")
    req = events_of(obs_clean, "request")[0]
    assert req["status"] == "error" and req["error_type"] == "ValueError"
    assert "stack_trace" in req                      # full trace goes to the protected log
    assert observability.get_metrics()["requests"]["failed"] == 1


# ---------------- 4. tool durations ----------------
def test_tool_calls_have_durations_and_share_request_id(datasets, obs_clean):
    df_a, _ = datasets
    ask("Which server has the most incidents?", df_a)
    tools = events_of(obs_clean, "tool")
    req = events_of(obs_clean, "request")[0]
    assert tools and all(isinstance(t["duration_ms"], float) and t["duration_ms"] >= 0 for t in tools)
    assert {t["request_id"] for t in tools} == {req["request_id"]}
    assert "get_server_analysis" in {t["operation"] for t in tools}
    m = observability.get_metrics()["tools"]
    assert m["total_calls"] == len(tools) and m["avg_ms"] is not None


def test_nested_tool_calls_are_not_double_counted(datasets, obs_clean):
    df_a, _ = datasets
    ask("What should we do to improve?", df_a)       # recommendations internally calls other analytics
    names = [t["operation"] for t in events_of(obs_clean, "tool")]
    assert names == ["get_recommendations"]


# ---------------- 5. RAG events (names only, never text) ----------------
def test_rag_event_records_documents_but_not_text(datasets, obs_clean):
    df_a, _ = datasets
    ask("What is the documented procedure for database connection failures?", df_a)
    rag_ev = events_of(obs_clean, "rag")
    assert len(rag_ev) == 1
    e = rag_ev[0]
    assert e["operation"] == "search_knowledge_base" and e["chunk_count"] >= 1
    assert "runbook_database_connection_failures.md" in e["documents"]
    blob = "\n".join(obs_clean.lines)
    assert "dbpool-svc" not in blob and "port 1521" not in blob      # document text never logged
    assert observability.get_metrics()["rag"]["count"] == 1


# ---------------- 6-7. LLM events, token usage optional ----------------
def test_llm_events_with_token_usage(datasets, obs_clean, monkeypatch):
    use_fake_model(monkeypatch, usage=True)
    df_a, _ = datasets
    ans = ask("Give me a summary.", df_a)
    assert "Final answer" in ans
    llm = events_of(obs_clean, "llm")
    assert len(llm) == 2 and all(e["status"] == "ok" and e["attempts"] == 1 and e["retry_count"] == 0 for e in llm)
    assert all(e["total_tokens"] == 15 for e in llm)
    sel = [e for e in events_of(obs_clean, "agent") if e["operation"] == "tool_selection"]
    assert sel and sel[0]["tools"] == ["get_incident_summary"]
    assert events_of(obs_clean, "request")[0]["mode"] == "llm"
    assert observability.get_metrics()["tokens"]["total"] == 30


def test_missing_token_usage_is_not_invented_and_does_not_error(datasets, obs_clean, monkeypatch):
    use_fake_model(monkeypatch, usage=False)
    df_a, _ = datasets
    ans = ask("Give me a summary.", df_a)
    assert "Final answer" in ans
    llm = events_of(obs_clean, "llm")
    assert llm and all("total_tokens" not in e for e in llm)
    tok = observability.get_metrics()["tokens"]
    assert tok["total"] == 0 and tok["calls_without_usage"] == 2


# ---------------- 8. exceptions do not leak secrets ----------------
def test_llm_failure_gives_safe_answer_and_redacted_logs(datasets, obs_clean, monkeypatch):
    from app.config import get_settings
    secret = "super-secret-key-12345"
    monkeypatch.setattr(get_settings(), "API_KEYS", f"{secret},other-key-67890")
    use_fake_model(monkeypatch, fail_with=RuntimeError(f"upstream rejected key {secret} at /srv/app/private/llm.py api_key=abc123xyz"))
    df_a, _ = datasets
    ans = ask("Give me a summary.", df_a)
    assert secret not in ans and "/srv/app" not in ans and "abc123xyz" not in ans
    assert ans                                           # the user still gets something useful
    blob = "\n".join(obs_clean.lines)
    assert secret not in blob and "abc123xyz" not in blob
    llm = events_of(obs_clean, "llm")
    assert llm[0]["status"] == "error" and llm[0]["error_type"] == "RuntimeError" and "stack_trace" in llm[0]
    assert events_of(obs_clean, "request")[0]["status"] == "degraded"     # answered, but a step failed


def test_http_500_hides_internals_but_returns_request_id(obs_clean):
    import httpx
    from app.main import app

    async def boom():
        raise RuntimeError("secret internals /srv/private api_key=hunter2hunter2")

    # The Gradio mount at "/" is a catch-all, so put the temporary test route first.
    app.router.add_api_route("/api/v1/_boom_test", boom, methods=["GET"])
    test_route = app.router.routes.pop()
    app.router.routes.insert(0, test_route)

    async def call():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.get("/api/v1/_boom_test", headers={"X-API-Key": API_KEY_A})
    try:
        r = asyncio.run(call())
    finally:
        app.router.routes.remove(test_route)
    assert r.status_code == 500
    assert "hunter2" not in r.text and "/srv/private" not in r.text and r.json()["detail"] == "Internal server error."
    assert r.json()["request_id"]
    assert "hunter2" not in "\n".join(obs_clean.lines)


# ---------------- 9-10. concurrency + isolation ----------------
def test_concurrent_requests_keep_their_own_ids_and_sessions(datasets, obs_clean, monkeypatch):
    use_fake_model(monkeypatch, delay=0.02)               # forces the tasks to interleave
    df_a, df_b = datasets
    tools = ["get_incident_summary", "get_category_analysis", "get_priority_analysis", "get_server_analysis"]

    async def one(i):
        monkeypatch.setattr(rag, "get_raw_chat_model", rag.get_raw_chat_model)  # no-op, keeps monkeypatch scope
        with observability.request_trace("chat", f"user-{i}", source="test") as ctx:
            model = FakeModel(tool=tools[i % 4], delay=0.02)
            # each task uses its OWN model so the expected tool differs per user
            rag.get_raw_chat_model = lambda m=model: m
            await asyncio.sleep(0.01 * (i % 3))
            await rag._answer_question_impl("q", df_a if i % 2 == 0 else df_b)
            return ctx.request_id, ctx.session_id, tools[i % 4]

    async def run():
        return await asyncio.gather(*[one(i) for i in range(16)])
    original = rag.get_raw_chat_model
    try:
        out = asyncio.run(run())
    finally:
        rag.get_raw_chat_model = original
    ids = [o[0] for o in out]
    assert len(set(ids)) == 16
    by_req = {}
    for e in obs_clean.events():
        by_req.setdefault(e["request_id"], []).append(e)
    for rid, sid, _tool in out:
        evs = by_req[rid]
        assert {e["session_id"] for e in evs} == {sid}               # no event ever carries another user's session
        assert all(e["request_id"] == rid for e in evs)
    assert len({o[1] for o in out}) == 16                            # distinct pseudonyms


def test_two_users_stay_isolated_and_logs_hold_no_incident_data(obs_clean):
    import httpx
    import pandas as pd
    from app.main import app
    A, B = (FIXTURES / "incidents_A.csv"), (FIXTURES / "incidents_B.csv")
    a_ids = set(pd.read_csv(A)["Incident Number"]); b_ids = set(pd.read_csv(B)["Incident Number"])

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            hA, hB = {"X-API-Key": API_KEY_A}, {"X-API-Key": API_KEY_B}
            await asyncio.gather(c.post("/api/v1/analyze/file", headers=hA, files={"file": ("a.csv", A.read_bytes(), "text/csv")}),
                                 c.post("/api/v1/analyze/file", headers=hB, files={"file": ("b.csv", B.read_bytes(), "text/csv")}))
            ca, cb = await asyncio.gather(
                c.post("/api/v1/chat", headers=hA, json={"question": "Which server has the most incidents?", "history": []}),
                c.post("/api/v1/chat", headers=hB, json={"question": "Which server has the most incidents?", "history": []}))
            ea, eb = await asyncio.gather(c.get("/api/v1/export/csv", headers=hA), c.get("/api/v1/export/csv", headers=hB))
            return ca, cb, ea, eb
    ca, cb, ea, eb = asyncio.run(run())
    assert "DB-PRD-01" in ca.json()["answer"] and "FW-EDGE01" not in ca.json()["answer"]
    assert "FW-EDGE01" in cb.json()["answer"] and "DB-PRD-01" not in cb.json()["answer"]
    assert not any(i in ea.text for i in b_ids) and not any(i in eb.text for i in a_ids)
    assert ca.headers["X-Request-ID"] != cb.headers["X-Request-ID"]
    # operational data contains no incident content, hosts, ids or API keys
    blob = "\n".join(obs_clean.lines) + str(observability.get_recent_events(500)) + str(observability.get_metrics())
    for token in ("DB-PRD-01", "FW-EDGE01", "AINC0001", "BINC0001", API_KEY_A, API_KEY_B, "Which server has the most"):
        assert token not in blob
    sessions = {e["session_id"] for e in obs_clean.events() if e["component"] == "request"}
    assert len(sessions) == 2


# ---------------- 11. logging failures never crash the agent ----------------
def test_logging_failures_do_not_break_the_agent(datasets, monkeypatch):
    df_a, _ = datasets
    monkeypatch.setattr(observability.OBS_LOGGER, "log", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(observability._METRICS, "add", lambda e: (_ for _ in ()).throw(RuntimeError("metrics broke")))
    ans = ask("What is the most common incident category?", df_a)
    assert "Database issues" in ans


# ---------------- privacy / config ----------------
def test_question_text_is_not_logged_by_default_but_length_and_hash_are(datasets, obs_clean):
    df_a, _ = datasets
    ask("What is the most common incident category? my-private-detail", df_a)
    blob = "\n".join(obs_clean.lines)
    assert "my-private-detail" not in blob
    req = events_of(obs_clean, "request")[0]
    assert req["question_length"] > 0 and len(req["question_hash"]) == 8 and "question_preview" not in req


def test_detailed_diagnostics_adds_redacted_preview(datasets, obs_clean, monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "OBS_DETAILED_DIAGNOSTICS", True)
    df_a, _ = datasets
    ask("Summary please, password=hunter2hunter2", df_a)
    req = events_of(obs_clean, "request")[0]
    assert "Summary please" in req["question_preview"] and "hunter2" not in req["question_preview"]


def test_admin_token_gate(monkeypatch):
    from app.config import get_settings
    st = get_settings()
    monkeypatch.setattr(st, "OBS_ADMIN_TOKEN", "")
    assert not observability.is_admin_token("") and not observability.is_admin_token("anything")
    monkeypatch.setattr(st, "OBS_ADMIN_TOKEN", "letmein-123")
    assert observability.is_admin_token("letmein-123") and not observability.is_admin_token("wrong")


def test_file_logging_is_opt_in_rotating_and_outside_data_dir(tmp_path, monkeypatch):
    from logging.handlers import RotatingFileHandler
    from app.config import get_settings
    st = get_settings()
    monkeypatch.setattr(st, "LOG_FILE_ENABLED", True)
    monkeypatch.setattr(st, "LOG_DIR", str(tmp_path / "logs"))
    before = (list(observability.OBS_LOGGER.handlers), list(logging.getLogger().handlers))
    monkeypatch.setattr(observability, "_setup_done", False)
    observability.setup_logging()
    try:
        rot = [h for h in observability.OBS_LOGGER.handlers if isinstance(h, RotatingFileHandler)]
        assert rot and rot[0].maxBytes == st.LOG_FILE_MAX_BYTES and rot[0].backupCount == st.LOG_FILE_BACKUP_COUNT
        with observability.request_trace("chat", "u", source="test"):
            pass
        for h in rot:
            h.flush()
        assert (tmp_path / "logs" / "itsm_events.jsonl").read_text().strip()
    finally:
        for h in list(observability.OBS_LOGGER.handlers):
            if h not in before[0]:
                observability.OBS_LOGGER.removeHandler(h); h.close()
        for h in list(logging.getLogger().handlers):
            if h not in before[1]:
                logging.getLogger().removeHandler(h); h.close()


def test_unwritable_log_dir_does_not_crash_startup(tmp_path, monkeypatch):
    from app.config import get_settings
    st = get_settings()
    blocker = tmp_path / "file-not-dir"; blocker.write_text("x")
    monkeypatch.setattr(st, "LOG_FILE_ENABLED", True)
    monkeypatch.setattr(st, "LOG_DIR", str(blocker / "logs"))
    monkeypatch.setattr(observability, "_setup_done", False)
    observability.setup_logging()   # must not raise


def test_percentiles_only_after_enough_samples(datasets, obs_clean):
    df_a, _ = datasets
    ask("Summary", df_a)
    assert observability.get_metrics()["requests"]["p95_ms"] is None
    for _ in range(observability.MIN_SAMPLES_FOR_PERCENTILES):
        ask("Summary", df_a)
    r = observability.get_metrics()["requests"]
    assert r["p50_ms"] is not None and r["p95_ms"] >= r["p50_ms"]


def test_metrics_endpoint_requires_api_key():
    import httpx
    from app.main import app

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            return await c.get("/api/v1/metrics"), await c.get("/api/v1/metrics", headers={"X-API-Key": API_KEY_A})
    no_key, ok = asyncio.run(run())
    assert no_key.status_code in (401, 403) and ok.status_code == 200 and "requests" in ok.json()


# ---------------- 12. Step 14 evaluation still works ----------------
def test_step14_evaluation_still_runs():
    r = subprocess.run([sys.executable, "evaluate_agent.py", "--only", "T01,T06,T09,T15", "--agent"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "Total Tests: 4" in r.stdout and "ERROR: 0" in r.stdout
    r = subprocess.run([sys.executable, "evaluate_agent.py", "--sessions"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "Failed: 0" in r.stdout and "Errors: 0" in r.stdout
