"""
Multi-user / session isolation tests (Part 3).

Two simulated users, A and B, each with their own incident dataset:
  * REST layer   - a "user" is an API key (that is how session_store.py keys data)
  * Gradio layer - a "user" is a browser session; per-session data lives in gr.State.
                   We call the real Gradio handlers (ui.analysis._analyze,
                   ui.tab_chat._chat_respond) with SEPARATE state values, exactly
                   as Gradio does for two browsers, and run them concurrently.
  * Knowledge base - inspected, and reported as INFO (it is global by design).

Each test returns PASS / FAIL. Items labelled INFO describe design behaviour and
are not counted as pass/fail.
"""
import asyncio
import io
import json
import traceback
from datetime import datetime as _real_datetime

import pandas as pd

from evaluation.harness import (API_KEY_A, API_KEY_B, API_KEY_SHARED, FIXTURES, ask_agent)

results: list[dict] = []


def record(tid, name, ok, detail="", info=False):
    results.append({"id": tid, "name": name, "result": "INFO" if info else ("PASS" if ok else "FAIL"), "detail": detail})


async def safe(tid, name, coro):
    """Runs one test; an exception becomes an ERROR row and the run continues."""
    try:
        await coro
    except Exception as exc:
        results.append({"id": tid, "name": name, "result": "ERROR",
                        "detail": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=3)})


def _csv_ids(text: str) -> list[str]:
    df = pd.read_csv(io.StringIO(text))
    col = next((c for c in ("ticket_id", "Ticket ID") if c in df.columns), None)
    return [str(x) for x in df[col]] if col else []


async def run_session_tests(recorder) -> list[dict]:
    import httpx
    from app.main import app
    from app.services import chat_tools

    results.clear()
    A_bytes = (FIXTURES / "incidents_A.csv").read_bytes()
    B_bytes = (FIXTURES / "incidents_B.csv").read_bytes()
    A_ids = set(pd.read_csv(FIXTURES / "incidents_A.csv")["Incident Number"])
    B_ids = set(pd.read_csv(FIXTURES / "incidents_B.csv")["Incident Number"])
    hostsA = set(pd.read_csv(FIXTURES / "incidents_A.csv")["Configuration Item"])
    hostsB = set(pd.read_csv(FIXTURES / "incidents_B.csv")["Configuration Item"])
    onlyA, onlyB = hostsA - hostsB, hostsB - hostsA   # tokens that identify a dataset
    catsA = set(json.load(open(FIXTURES / "incidents_A_expected_categories.json")).values())
    catsB = set(json.load(open(FIXTURES / "incidents_B_expected_categories.json")).values())

    def leaks(text, other_tokens):
        return sorted(t for t in other_tokens if t in text)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        hA, hB, hS = {"X-API-Key": API_KEY_A}, {"X-API-Key": API_KEY_B}, {"X-API-Key": API_KEY_SHARED}

        async def upload(headers, name, data):
            return await c.post("/api/v1/analyze/file", headers=headers, files={"file": (name, data, "text/csv")})

        async def export(headers):
            return await c.get("/api/v1/export/csv", headers=headers)

        async def chat(headers, q, history=None):
            r = await c.post("/api/v1/chat", headers=headers, json={"question": q, "history": history or []})
            return r.status_code, (r.json().get("answer", "") if r.status_code == 200 else r.text)

        # ---------------- REST layer ----------------
        async def t_upload():
            ra, rb = await upload(hA, "incidents_A.csv", A_bytes), await upload(hB, "incidents_B.csv", B_bytes)
            idsA = {t["ticket_id"] for t in ra.json()["results"]}
            idsB = {t["ticket_id"] for t in rb.json()["results"]}
            record("S01", "User A upload returns only A's incidents", ra.status_code == 200 and idsA == A_ids, f"{len(idsA)} tickets")
            record("S02", "User B upload returns only B's incidents", rb.status_code == 200 and idsB == B_ids, f"{len(idsB)} tickets")
        await safe("S01", "uploads", t_upload())

        async def t_export():
            ra, rb = await export(hA), await export(hB)
            ia, ib = set(_csv_ids(ra.text)), set(_csv_ids(rb.text))
            record("S03", "User A CSV export contains only A's data (B uploaded afterwards)", ia == A_ids and not ia & B_ids and not leaks(ra.text, onlyB),
                   f"{len(ia)} rows, B-only tokens leaked: {leaks(ra.text, onlyB)}")
            record("S04", "User B CSV export contains only B's data", ib == B_ids and not ib & A_ids and not leaks(rb.text, onlyA),
                   f"{len(ib)} rows, A-only tokens leaked: {leaks(rb.text, onlyA)}")
        await safe("S03", "exports", t_export())

        async def t_overwrite():
            before = (await export(hA)).text
            await upload(hB, "incidents_B.csv", B_bytes)          # B uploads AGAIN
            after = (await export(hA)).text
            record("S05", "User B's upload does not overwrite User A's data", before == after, "A's export identical before/after B's upload")
        await safe("S05", "overwrite", t_overwrite())

        async def t_chat_data():
            sa, aa = await chat(hA, "Which server has the most incidents?")
            sb, ab = await chat(hB, "Which server has the most incidents?")
            record("S06", "User A's agent analytics use only A's data", sa == 200 and any(h in aa for h in hostsA) and not leaks(aa, onlyB),
                   f"B-only tokens in A's answer: {leaks(aa, onlyB)}")
            record("S07", "User B's agent analytics use only B's data", sb == 200 and any(h in ab for h in hostsB) and not leaks(ab, onlyA),
                   f"A-only tokens in B's answer: {leaks(ab, onlyA)}")
            _, ca = await chat(hA, "What are the incident categories?")
            _, cb = await chat(hB, "What are the incident categories?")
            wrongA, wrongB = sorted(x for x in catsB - catsA if x in ca), sorted(x for x in catsA - catsB if x in cb)
            record("S08", "Category analytics are per user (no other-user categories)", not wrongA and not wrongB,
                   f"A saw B-only {wrongA}; B saw A-only {wrongB}")
        await safe("S06", "chat data", t_chat_data())

        async def t_history():
            secret = "SECRET-MARKER-7731"
            _, a1 = await chat(hA, "Which server has the most incidents?", [[f"my private note is {secret}", "noted"]])
            _, b1 = await chat(hB, "Summarise the incidents.")
            record("S09", "User A's chat history does not appear in User B's answers", secret not in b1,
                   "history is client-supplied and the server stores none")
        await safe("S09", "history", t_history())

        async def t_concurrent():
            ok = True
            detail = []
            for i in range(3):
                await asyncio.gather(upload(hA, "incidents_A.csv", A_bytes), upload(hB, "incidents_B.csv", B_bytes))
                ea, eb = await asyncio.gather(export(hA), export(hB))
                good = set(_csv_ids(ea.text)) == A_ids and set(_csv_ids(eb.text)) == B_ids
                ok &= good
                detail.append(good)
            record("S10", "Concurrent uploads by A and B do not corrupt each other (3 rounds)", ok, f"rounds ok: {detail}")
        await safe("S10", "concurrent", t_concurrent())

        async def t_unknown():
            r0 = await export(hS)   # a valid key that never uploaded
            r1 = await c.get("/api/v1/export/csv", headers={"X-API-Key": "not-a-key"})
            record("S11", "A user with no upload sees nothing (404), bad key rejected (401)", r0.status_code == 404 and r1.status_code == 401,
                   f"no-upload={r0.status_code}, bad-key={r1.status_code}")
        await safe("S11", "no-upload", t_unknown())

        async def t_shared_key():
            # Two people share ONE API key but each sends their own X-Session-ID.
            sx = (await c.post("/api/v1/session", headers=hS)).json()["session_id"]
            sy = (await c.post("/api/v1/session", headers=hS)).json()["session_id"]
            hX, hY = {**hS, "X-Session-ID": sx}, {**hS, "X-Session-ID": sy}
            await upload(hX, "incidents_A.csv", A_bytes)
            await upload(hY, "incidents_B.csv", B_bytes)
            idsX, idsY = set(_csv_ids((await export(hX)).text)), set(_csv_ids((await export(hY)).text))
            record("S12", "Two people sharing ONE API key, with their own X-Session-ID, keep separate data",
                   idsX == A_ids and idsY == B_ids and sx != sy, f"X has {len(idsX)} rows, Y has {len(idsY)} rows")
            _, ans = await chat(hX, "Which server has the most incidents?")
            record("S21", "Agent answers for a session ID use only that session's data", not leaks(ans, onlyB) and any(h in ans for h in hostsA),
                   f"B-only tokens leaked: {leaks(ans, onlyB)}")
            bad = await c.get("/api/v1/export/csv", headers={**hS, "X-Session-ID": "bad id!"})
            other = await c.get("/api/v1/export/csv", headers={**hS, "X-Session-ID": "unknown-session-123456"})
            record("S22", "Invalid session ID rejected (400); unknown session sees nothing (404)",
                   bad.status_code == 400 and other.status_code == 404, f"invalid={bad.status_code}, unknown={other.status_code}")
            # Backward compatibility: no header = old behaviour, one slot per API key.
            await upload(hS, "incidents_A.csv", A_bytes)
            record("S23", "INFO: without X-Session-ID, everyone on one API key still shares a single slot (backward compatible)", False,
                   "send X-Session-ID (get one from POST /api/v1/session) to isolate users", info=True)
        await safe("S12", "shared key", t_shared_key())

    # ---------------- Gradio handler layer ----------------
    import ui.analysis as ua
    from ui.tab_chat import _chat_respond

    class _F:  # mimics gr.File value (handler only uses .name)
        def __init__(self, path): self.name = str(path)

    async def final_output(path):
        last = None
        async for out in ua._analyze(_F(path), ""):
            last = out
        return last  # tuple: index 3 = csv path, 4 = full_results_state, 5 = cache notice

    async def t_gradio_exports():
        # Freeze the clock so both users finish "in the same second" - the
        # realistic worst case for the timestamped /tmp export filename.
        class _Frozen:
            @staticmethod
            def now(): return _real_datetime(2030, 1, 1, 12, 0, 0)
        ua.datetime = _Frozen
        try:
            oa, ob = await asyncio.gather(final_output(FIXTURES / "incidents_A.csv"), final_output(FIXTURES / "incidents_B.csv"))
        finally:
            ua.datetime = _real_datetime
        pa, pb = oa[3], ob[3]
        ta, tb = open(pa).read(), open(pb).read()
        import os
        for p_ in {pa, pb}:
            os.remove(p_)   # tidy the /tmp file the frozen-clock test created
        idsA, idsB = set(_csv_ids(ta)), set(_csv_ids(tb))
        record("S13", "Gradio: each user gets a distinct export file path", pa != pb, f"A={pa} B={pb}")
        record("S14", "Gradio: User A's downloaded CSV contains only A's data", idsA == A_ids, f"{len(idsA)} rows, B tickets present: {len(idsA & B_ids)}")
        record("S15", "Gradio: User B's downloaded CSV contains only B's data", idsB == B_ids, f"{len(idsB)} rows, A tickets present: {len(idsB & A_ids)}")
        return oa[4], ob[4]
    dfs = None
    try:
        dfs = await t_gradio_exports()
    except Exception as exc:
        results.append({"id": "S13", "name": "Gradio export isolation", "result": "ERROR", "detail": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3)})
    dfA, dfB = dfs if dfs else (None, None)

    async def t_gradio_chat():
        _, hist_a, _ = await _chat_respond("Which server has the most incidents?", [], dfA, {})
        _, hist_b, _ = await _chat_respond("Which server has the most incidents?", [], dfB, {})
        a_txt, b_txt = hist_a[-1][1], hist_b[-1][1]
        record("S16", "Gradio: User A's chat answers come only from A's state", any(h in a_txt for h in hostsA) and not leaks(a_txt, onlyB), f"leaks: {leaks(a_txt, onlyB)}")
        record("S17", "Gradio: User B's chat answers come only from B's state", any(h in b_txt for h in hostsB) and not leaks(b_txt, onlyA), f"leaks: {leaks(b_txt, onlyA)}")
        record("S18", "Gradio: chat histories are separate per user", len(hist_a) == 1 and len(hist_b) == 1 and hist_a[0][0] == hist_b[0][0] and hist_a != hist_b,
               "each session's history list only holds its own turns")
    if dfA is not None:
        await safe("S16", "gradio chat", t_gradio_chat())

    async def t_id_collision():
        o = await final_output(FIXTURES / "incidents_B_collision.csv")
        df = o[4].set_index("Ticket ID")
        cat2, cat3 = df.loc["AINC0002", "Category"], df.loc["AINC0003", "Category"]
        record("S19", "Same incident numbers, different content: user B is not served user A's cached categories",
               cat2 == "Network issues" and cat3 == "Security incidents",
               f"AINC0002 -> {cat2} (A's version was Database issues), AINC0003 -> {cat3}")
    await safe("S19", "id collision", t_id_collision())

    async def t_same_file_cache():
        o = await final_output(FIXTURES / "incidents_A.csv")   # "user B" uploads the exact file A already analysed
        notice = o[5]
        visible = bool(getattr(notice, "get", lambda *_: None)("visible")) if isinstance(notice, dict) else False
        record("S20", "INFO: result cache is shared across users (identical file -> 'already analyzed on <time>' notice)", False,
               "content is identical so no data differs, but one user can learn that someone analysed this exact file" if visible else "no notice shown", info=True)
    await safe("S20", "same-file cache", t_same_file_cache())

    # ---------------- Knowledge base scope ----------------
    from app.services import knowledge_base
    docA, docB = FIXTURES / "kb" / "_user_A_payments_failover_runbook.md", FIXTURES / "kb" / "_user_B_warehouse_scanner_guide.md"

    async def t_kb():
        ra = await knowledge_base.ingest_document(docA.name, docA.read_bytes())    # "user A uploads a KB doc"
        rb = await knowledge_base.ingest_document(docB.name, docB.read_bytes())    # "user B uploads a KB doc"
        a_sees_b, _ = await ask_agent("According to the runbook, what is the recovery procedure for the warehouse scanner dock?", dfA if dfA is not None else pd.DataFrame({"x": [1]}), recorder)
        b_sees_a, _ = await ask_agent("According to the runbook, what is the payments cluster failover procedure?", dfB if dfB is not None else pd.DataFrame({"x": [1]}), recorder)
        shared = ("WHSCAN-B7" in a_sees_b) and ("pay-freeze-alpha" in b_sees_a)
        record("K1", f"INFO: knowledge base is {'SHARED/GLOBAL' if shared else 'user-scoped'} "
                     "(user A can retrieve user B's document and vice-versa)", False,
               "documents are stored once in kb_documents/kb_chunks with no owner column; this matches the design comment in persistence.py", info=True)
        docs = {d["document_name"]: d["id"] for d in knowledge_base.list_documents()}
        # user B deletes the document user A uploaded
        knowledge_base.delete_document(ra["document_id"])
        a_after, _ = await ask_agent("According to the runbook, what is the payments cluster failover procedure?", dfA if dfA is not None else pd.DataFrame({"x": [1]}), recorder)
        record("K2", "INFO: any user can delete another user's KB document (no ownership/permissions)", False,
               "after B deleted A's document, A's agent " + ("no longer finds it" if "pay-freeze-alpha" not in a_after else "still finds it"), info=True)
        knowledge_base.delete_document(rb["document_id"])   # clean up
    await safe("K1", "kb scope", t_kb())

    return list(results)
