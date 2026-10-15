"""
Step 17 tests - sign-in, roles, per-user permissions and the Admin page's back end.

* Service-level tests run in-process against the shared test database (unique
  user names per test, so they don't interfere).
* HTTP tests run in a FRESH process with AUTH_ENABLED=true and an empty database,
  because the access gate is installed when the app is imported.
"""
import asyncio
import json
import os
import subprocess
import sys
import textwrap
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.services import auth, rag
from app.services.persistence import AuthEvent, AuthSession, SessionLocal, User
from tests.test_security_guardrails import ScriptedModel, ask, use_model

ROOT = Path(__file__).resolve().parent.parent
PW = "Correct-Horse-9"


def uname(prefix="u"):
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def make(role="viewer", password=PW, **kw):
    name = uname()
    user, _ = auth.create_user(name, kw.pop("display", name.title()), "", role, password=password)
    return user


def req(username):
    return SimpleNamespace(username=username, session_hash="h")


@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "AUTH_ENABLED", True)


# ============================ passwords ============================
def test_password_hashing_and_policy():
    h1, h2 = auth.hash_password(PW), auth.hash_password(PW)
    assert h1 != h2 and PW not in h1 and h1.startswith("scrypt$")          # salted, never plaintext
    assert auth.verify_password(PW, h1) and not auth.verify_password(PW + "x", h1)
    assert not auth.verify_password(PW, "garbage") and not auth.verify_password(PW, "")
    assert auth.validate_password("short") and auth.validate_password("x" * 200)
    assert auth.validate_password("SameAsName1", "sameasname1")
    assert auth.validate_password(PW) is None


def test_create_user_validation_and_temp_password():
    for bad in ["", "ab", "UPPER space", "-leading", "x" * 40, "semi;colon"]:
        with pytest.raises(auth.AuthError):
            auth.create_user(bad, "", "", "viewer", password=PW)
    name = uname()
    with pytest.raises(auth.AuthError):
        auth.create_user(name, "", "", "superuser", password=PW)
    with pytest.raises(auth.AuthError):
        auth.create_user(name, "", "not-an-email", "viewer", password=PW)
    with pytest.raises(auth.AuthError):
        auth.create_user(name, "", "", "viewer", password="short")
    user, temp = auth.create_user(name, "Test Person", "t@example.com", "analyst")
    assert len(temp) >= 10 and auth.validate_password(temp) is None
    with SessionLocal() as db:
        assert temp not in db.query(User).filter(User.username == name).one().password_hash
    with pytest.raises(auth.AuthError, match="taken"):
        auth.create_user(name.upper(), "", "", "viewer", password=PW)         # usernames are case-insensitive
    assert auth.authenticate(name, temp)[0]["role"] == "analyst"


# ============================ sign-in ============================
def test_login_success_and_uniform_failures():
    u = make()
    ok, reason = auth.authenticate(u["username"].upper(), PW)               # case-insensitive username
    assert ok and reason == ""
    assert auth.authenticate(u["username"], "wrong") == (None, "invalid")
    assert auth.authenticate(uname("ghost"), "wrong") == (None, "invalid")   # unknown user: same reason
    for bad_user, bad_pw in [("", PW), (u["username"], ""), ("x" * 100, PW), (u["username"], "p" * 500)]:
        assert auth.authenticate(bad_user, bad_pw)[0] is None


def test_lockout_after_repeated_failures_and_unlock(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "LOGIN_MAX_FAILED", 3)
    u = make()
    for _ in range(3):
        assert auth.authenticate(u["username"], "nope")[1] == "invalid"
    assert auth.authenticate(u["username"], PW) == (None, "locked")          # even the right password is refused
    with SessionLocal() as db:                                                # simulate the lock expiring
        row = db.query(User).filter(User.username == u["username"]).one()
        row.locked_until = auth._now() - timedelta(minutes=1)
        db.commit()
    assert auth.authenticate(u["username"], PW)[0] is not None


def test_disabled_account_cannot_sign_in_and_sessions_die():
    u = make()
    user, _ = auth.authenticate(u["username"], PW)
    token = auth.create_session(user["id"])
    assert auth.user_for_token(token)["username"] == u["username"]
    auth.set_status(u["username"], "disabled")
    assert auth.user_for_token(token) is None                                 # signed out immediately
    assert auth.authenticate(u["username"], PW) == (None, "disabled")
    auth.set_status(u["username"], "active")
    assert auth.authenticate(u["username"], PW)[0] is not None


def test_session_is_rejected_when_account_is_disabled_even_without_session_cleanup():
    """Second layer: set_status() also deletes sessions, but user_for_token() must refuse a disabled
    account on its own (e.g. a status flipped directly in the database)."""
    u = make()
    user, _ = auth.authenticate(u["username"], PW)
    token = auth.create_session(user["id"])
    with SessionLocal() as db:
        db.query(User).filter(User.username == u["username"]).one().status = "disabled"
        db.commit()
    with SessionLocal() as db:
        assert db.get(AuthSession, auth._token_hash(token)) is not None      # the session row is still there...
    assert auth.user_for_token(token) is None                                 # ...but it no longer authenticates


def test_sessions_expire_are_hashed_and_revoked_on_password_reset():
    u = make()
    user, _ = auth.authenticate(u["username"], PW)
    token = auth.create_session(user["id"])
    with SessionLocal() as db:
        stored = [r.token_hash for r in db.query(AuthSession).filter(AuthSession.user_id == user["id"])]
    assert token not in stored and all(len(h) == 64 for h in stored)         # only a SHA-256 is kept
    assert auth.user_for_token("not-a-token") is None and auth.user_for_token(None) is None
    with SessionLocal() as db:
        s = db.get(AuthSession, auth._token_hash(token)); s.expires_at = auth._now() - timedelta(seconds=1); db.commit()
    assert auth.user_for_token(token) is None                                 # expired
    t2 = auth.create_session(user["id"])
    temp = auth.reset_password(u["username"])
    assert auth.user_for_token(t2) is None                                    # reset signs them out everywhere
    assert auth.authenticate(u["username"], PW)[0] is None and auth.authenticate(u["username"], temp)[0]
    t3 = auth.create_session(user["id"]); auth.end_session(t3)
    assert auth.user_for_token(t3) is None


# ============================ roles & permissions ============================
def test_role_defaults_match_the_design():
    viewer, analyst, admin = make("viewer"), make("analyst"), make("admin")
    pv, pa, pd = viewer["permissions"], analyst["permissions"], admin["permissions"]
    assert pv["view_overview"] and pv["qa_documents"] and not (pv["upload_data"] or pv["view_kb"] or pv["qa_incident"] or pv["manage_users"])
    assert pa["upload_data"] and pa["run_categorization"] and pa["export_csv"] and pa["manage_kb"] and pa["qa_incident"]
    assert not (pa["manage_users"] or pa["view_observability"] or pa["change_settings"])
    assert all(pd.values()) and set(pd) == set(auth.PERMISSION_KEYS)


def test_overrides_apply_store_only_differences_and_reset():
    u = make("viewer")
    desired = {k: (k in auth.ROLE_DEFAULTS["viewer"]) for k in auth.PERMISSION_KEYS}
    desired.update(view_recommendations=True, qa_incident=True, qa_documents=False)
    out = auth.set_overrides(u["username"], desired)
    assert out["overrides"] == {"view_recommendations": True, "qa_incident": True, "qa_documents": False}
    p = auth.get_user(u["username"])["permissions"]
    assert p["view_recommendations"] and p["qa_incident"] and not p["qa_documents"] and p["view_overview"]
    auth.set_overrides(u["username"], {k: k in auth.ROLE_DEFAULTS["viewer"] for k in auth.PERMISSION_KEYS})
    assert auth.get_user(u["username"])["overrides"] == {}
    auth.set_overrides(u["username"], {"not_a_permission": True})            # unknown keys are ignored
    assert "not_a_permission" not in auth.get_user(u["username"])["permissions"]


def test_admin_cannot_lose_user_management_and_disabled_has_nothing():
    admin = make("admin")
    auth.set_overrides(admin["username"], {"manage_users": False})
    assert auth.get_user(admin["username"])["permissions"]["manage_users"]
    auth.set_status(admin["username"], "disabled")
    assert not any(auth.get_user(admin["username"])["permissions"].values())


def test_last_active_admin_is_protected():
    only = make("admin")                                                     # create the keeper FIRST
    with SessionLocal() as db:                                               # isolate: make it the only active admin
        others = [r.username for r in db.query(User).filter(User.role == "admin", User.status == "active", User.username != only["username"])]
    for o in others:
        auth.set_status(o, "disabled")                                       # allowed: `only` is still active
    try:
        with pytest.raises(auth.AuthError, match="At least one"):
            auth.set_role(only["username"], "viewer")
        with pytest.raises(auth.AuthError, match="At least one"):
            auth.set_status(only["username"], "disabled")
        second = make("admin")
        auth.set_role(only["username"], "analyst")                           # fine now: another admin exists
        assert auth.get_user(only["username"])["role"] == "analyst"
        with pytest.raises(auth.AuthError, match="At least one"):            # and the new sole admin is protected in turn
            auth.set_status(second["username"], "disabled")
    finally:
        for o in others:
            auth.set_status(o, "active")


def test_list_filter_and_search():
    a, b = make("analyst", display="Zed Zebra"), make("viewer", display="Quinn Quail")
    auth.set_status(b["username"], "disabled")
    names = lambda **kw: {u["username"] for u in auth.list_users(**kw)}
    assert a["username"] in names(search="zebra") and b["username"] not in names(search="zebra")
    assert b["username"] in names(flt="disabled") and a["username"] not in names(flt="disabled")
    assert a["username"] in names(flt="analyst") and b["username"] not in names(flt="analyst")
    assert a["username"] in names(search=a["username"][:6].upper())


def test_activity_log_records_events_without_secrets():
    u = make()
    auth.authenticate(u["username"], "wrong-secret-attempt")
    auth.authenticate(u["username"], PW)
    auth.record_event(u["username"], "qa_asked")
    summary = auth.activity_summary(u["username"])
    assert summary["failed_7d"] == 1 and summary["qa_7d"] == 1 and summary["last_failed"]
    blob = json.dumps(auth.recent_events(u["username"], 20), default=str)
    assert "wrong-secret-attempt" not in blob and PW not in blob
    assert auth.humanize(None) == "Never" and auth.humanize(auth._now()) == "Just now"


# ============================ enforcement inside the UI ============================
def test_permissions_for_request_and_auth_disabled_means_full_access(auth_on, monkeypatch):
    from ui import access
    viewer, analyst = make("viewer"), make("analyst")
    assert access.can(req(viewer["username"]), "view_overview") and not access.can(req(viewer["username"]), "upload_data")
    assert access.can(req(analyst["username"]), "upload_data")
    assert not any(access.permissions_for(None).values())                   # no identity -> nothing
    assert not any(access.permissions_for(req("nobody-here")).values())
    with pytest.raises(gr.Error, match="permission"):
        access.require(req(viewer["username"]), "upload_data", "upload")
    monkeypatch.setattr(get_settings(), "AUTH_ENABLED", False)
    assert all(access.permissions_for(None).values())                       # old behaviour when sign-in is off


def test_permission_changes_apply_to_the_next_action_without_signing_in_again(auth_on):
    from ui import access
    u = make("viewer")
    assert not access.can(req(u["username"]), "upload_data")
    auth.set_role(u["username"], "analyst")
    assert access.can(req(u["username"]), "upload_data")
    auth.set_status(u["username"], "disabled")
    assert not access.can(req(u["username"]), "view_overview")


def test_viewer_cannot_analyze_or_manage_kb_but_analyst_can(auth_on):
    import ui.analysis as ua
    import ui.tab_kb as kb
    viewer, analyst = make("viewer"), make("analyst")

    async def first(agen):
        async for item in agen:
            return item
    with pytest.raises(gr.Error, match="permission"):
        asyncio.run(first(ua._analyze(None, "INC1 server down", req(viewer["username"]))))
    with pytest.raises(gr.Error, match="permission"):
        asyncio.run(first(ua._analyze_sample(req(viewer["username"]))))
    with pytest.raises(gr.Error, match="permission"):
        asyncio.run(kb._kb_upload(SimpleNamespace(name="x.md"), req(viewer["username"])))
    with pytest.raises(gr.Error, match="permission"):
        kb._kb_delete("1", req(viewer["username"]))
    with pytest.raises(gr.Error, match="permission"):
        asyncio.run(kb._kb_reindex("1", req(viewer["username"])))
    # an analyst is let through the permission check (it proceeds to normal input handling)
    try:
        asyncio.run(first(ua._analyze(None, "", req(analyst["username"]))))
    except gr.Error as exc:
        assert "permission" not in str(exc)


def test_export_csv_permission_controls_the_download(auth_on, monkeypatch):
    import ui.analysis as ua
    analyst = make("analyst")
    auth.set_overrides(analyst["username"], {**{k: k in auth.ROLE_DEFAULTS["analyst"] for k in auth.PERMISSION_KEYS}, "export_csv": False})
    seen = []
    monkeypatch.setattr(ua, "_write_export_csv", lambda df: seen.append(1) or "/tmp/x.csv")

    async def last(agen):
        out = None
        async for item in agen:
            out = item
        return out
    out = asyncio.run(last(ua._analyze(None, "INC9 database is down\nWork notes: restarted\n", req(analyst["username"]))))
    assert not seen and not any("/tmp/x.csv" in str(x) for x in out)         # no export file made for this person


# ============================ Q&A scope ============================
def test_chat_scope_from_permissions(auth_on):
    from ui.access import chat_scope
    both, docs_only, inc_only, none = make("analyst"), make("viewer"), make("viewer"), make("viewer")
    auth.set_overrides(inc_only["username"], {"qa_documents": False, "qa_incident": True})
    auth.set_overrides(none["username"], {"qa_documents": False})
    assert chat_scope(req(both["username"])) == "all" and chat_scope(req(docs_only["username"])) == "documents"
    assert chat_scope(req(inc_only["username"])) == "incident" and chat_scope(req(none["username"])) is None


def test_agent_scope_limits_the_tools_the_model_gets(datasets, monkeypatch):
    seen = {}

    class Spy(ScriptedModel):
        def bind_tools(self, tools):
            seen["tools"] = {t.name for t in tools}
            return self
    for scope, expect_kb in (("documents", True), ("incident", False), ("all", True)):
        m = Spy([AIMessage(content="ok")])
        monkeypatch.setattr(rag, "get_raw_chat_model", lambda m=m: m)
        monkeypatch.setattr(rag, "with_chat_retry", lambda x: x)
        asyncio.run(rag.answer_question("hello", datasets[0], scope=scope))
        assert ("search_knowledge_base" in seen["tools"]) is expect_kb
        if scope == "documents":
            assert seen["tools"] == {"search_knowledge_base"}
        if scope == "incident":
            assert "get_incident_summary" in seen["tools"]
        assert ("SCOPE:" in m.seen[0][0].content) is (scope != "all")


def test_rule_based_scope_documents_never_reveals_incident_data(datasets):
    out = asyncio.run(rag.answer_question("How many P1 incidents are there?", datasets[0], scope="documents"))
    assert "21" not in out and "P1 incidents" not in out.split("keyword")[0]      # not the incident summary
    out = asyncio.run(rag.answer_question("What does the backup SOP say to do?", datasets[0], scope="incident"))
    assert "aren't enabled" in out


def test_chat_handler_for_documents_only_person_works_without_an_upload(auth_on):
    import pandas as pd
    from ui.tab_chat import _chat_respond
    viewer, nobody = make("viewer"), make("viewer")
    auth.set_overrides(nobody["username"], {"qa_documents": False})
    out = asyncio.run(_chat_respond("What does the backup SOP say?", [], pd.DataFrame(), {}, req(viewer["username"])))
    assert "Run an analysis" not in str(out)
    out = asyncio.run(_chat_respond("hi", [], pd.DataFrame(), {}, req(nobody["username"])))
    assert "isn't enabled for your account" in str(out)
    assert any(e["event"] == "qa_asked" for e in auth.recent_events(viewer["username"]))


# ============================ HTTP flow (fresh process, sign-in ON) ============================
def run_app(script: str, extra_env=None, tmp_path=None) -> dict:
    code = textwrap.dedent("""
        import sys, json, asyncio, httpx; sys.path.insert(0, '.')
        from app.main import app
        R = {}
        async def main():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://t') as c:
        %s
        asyncio.run(main()); print('RESULT ' + json.dumps(R))
    """) % textwrap.indent(textwrap.dedent(script), " " * 16)
    env = {**os.environ, "AUTH_ENABLED": "true", "ADMIN_USERNAME": "boss", "ADMIN_PASSWORD": "Boss-Password-123",
           "ITSM_DB_PATH": str(tmp_path / "auth.db"), "API_KEYS": "k" * 20, "RATE_LIMIT": "100000/minute", **(extra_env or {})}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, out.stderr[-1500:]
    return json.loads(line[0][7:])


def test_http_gate_login_cookie_logout_and_generic_errors(tmp_path):
    r = run_app("""
        H = {'accept': 'text/html'}
        R['anon_page'] = (await c.get('/', headers=H, follow_redirects=False)).status_code
        R['anon_page_to'] = (await c.get('/', headers=H, follow_redirects=False)).headers.get('location')
        R['anon_api'] = (await c.get('/gradio_api/info')).status_code
        R['rest_api_unaffected'] = (await c.post('/api/v1/session', headers={'X-API-Key': 'k'*20})).status_code
        R['health_open'] = (await c.get('/health')).status_code
        page = await c.get('/login'); R['login_ok'] = page.status_code and 'Welcome back' in page.text
        R['login_has_toggle'] = 'itsm-theme' in page.text and 'Keep me signed in' in page.text
        R['login_nostore'] = page.headers.get('cache-control')
        bad = await c.post('/auth/login', json={'username': 'boss', 'password': 'nope'})
        ghost = await c.post('/auth/login', json={'username': 'ghost', 'password': 'nope'})
        R['bad'] = [bad.status_code, bad.json()]; R['ghost'] = [ghost.status_code, ghost.json()]
        R['malformed'] = (await c.post('/auth/login', content=b'not json')).status_code
        ok = await c.post('/auth/login', json={'username': 'BOSS', 'password': 'Boss-Password-123', 'remember': True})
        ck = ok.headers['set-cookie'].lower()
        R['ok'] = ok.status_code; R['cookie_flags'] = ['httponly' in ck, 'samesite=lax' in ck, 'max-age=28800' in ck]
        R['signed_in_page'] = (await c.get('/', headers=H)).status_code
        R['signed_in_api'] = (await c.get('/gradio_api/info')).status_code
        R['login_when_in'] = (await c.get('/login', follow_redirects=False)).status_code
        R['logout'] = (await c.post('/auth/logout', follow_redirects=False)).status_code
        R['after_logout'] = (await c.get('/gradio_api/info')).status_code
        s = await c.post('/auth/login', json={'username': 'boss', 'password': 'Boss-Password-123'})   # remember off
        R['session_cookie'] = 'max-age' not in s.headers['set-cookie'].lower()
    """, tmp_path=tmp_path)
    assert r["anon_page"] == 302 and r["anon_page_to"] == "/login" and r["anon_api"] == 401
    assert r["rest_api_unaffected"] == 200 and r["health_open"] == 200
    assert r["login_ok"] and r["login_has_toggle"] and r["login_nostore"] == "no-store"
    assert r["bad"] == r["ghost"] == [401, {"error": "Incorrect username or password."}]     # no user enumeration
    assert r["malformed"] == 400 and r["ok"] == 200 and r["cookie_flags"] == [True, True, True]
    assert r["signed_in_page"] == 200 and r["signed_in_api"] == 200 and r["login_when_in"] == 303
    assert r["logout"] == 303 and r["after_logout"] == 401 and r["session_cookie"]


def test_http_lockout_disable_midsession_and_empty_setup_notice(tmp_path):
    r = run_app("""
        for _ in range(5):
            await c.post('/auth/login', json={'username': 'boss', 'password': 'wrong'})
        locked = await c.post('/auth/login', json={'username': 'boss', 'password': 'Boss-Password-123'})
        R['locked'] = [locked.status_code, 'Too many' in locked.json()['error']]
        from app.services import auth
        auth.create_user('worker', 'Worker', '', 'analyst', password='Worker-Password-1')
        ok = await c.post('/auth/login', json={'username': 'worker', 'password': 'Worker-Password-1'})
        R['worker_in'] = (await c.get('/gradio_api/info')).status_code
        auth.set_status('worker', 'disabled')
        R['worker_after_disable'] = (await c.get('/gradio_api/info')).status_code
    """, tmp_path=tmp_path)
    assert r["locked"] == [429, True] and r["worker_in"] == 200 and r["worker_after_disable"] == 401

    (tmp_path / "x").mkdir()
    r = run_app("""
        page = await c.get('/login'); R['notice'] = 'No accounts exist yet' in page.text
        R['cant_login'] = (await c.post('/auth/login', json={'username': 'admin', 'password': 'anything-long-1'})).status_code
    """, extra_env={"ADMIN_PASSWORD": ""}, tmp_path=tmp_path / "x")
    assert r["notice"] and r["cant_login"] == 401


def test_http_bootstrap_legacy_users_and_idempotence(tmp_path):
    r = run_app("""
        from app.services import auth
        R['users'] = sorted((u['username'], u['role']) for u in auth.list_users())
        R['again'] = auth.bootstrap_admin()
    """, extra_env={"UI_AUTH_USERS": "legacy.one:Legacy-Pass-0001"}, tmp_path=tmp_path)
    assert r["users"] == [["boss", "admin"], ["legacy.one", "admin"]] and r["again"] is None


def test_http_auth_endpoints_are_rate_limited(tmp_path):
    r = run_app("""
        R['codes'] = [(await c.post('/auth/login', json={'username': 'x', 'password': 'y'})).status_code for _ in range(6)]
    """, extra_env={"RATE_LIMIT": "3/minute"}, tmp_path=tmp_path)
    assert r["codes"] == [401, 401, 401, 429, 429, 429]


def test_sign_in_disabled_restores_open_dashboard(tmp_path):
    r = run_app("""
        R['page'] = (await c.get('/', headers={'accept': 'text/html'}, follow_redirects=False)).status_code
        R['api'] = (await c.get('/gradio_api/info')).status_code
    """, extra_env={"AUTH_ENABLED": "false"}, tmp_path=tmp_path)
    assert r["page"] == 200 and r["api"] == 200


def test_admin_tab_builds_and_user_chip_is_escaped():
    from ui.layout import _user_chip_html, build_ui
    build_ui()                                                               # the whole UI (incl. Admin tab) still builds
    chip = _user_chip_html({"display_name": "<img src=x onerror=alert(1)>", "role": "admin"})
    assert "<img" not in chip and "&lt;img" in chip and 'action="/auth/logout"' in chip
    assert _user_chip_html(None) == ""
