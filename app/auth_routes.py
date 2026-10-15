"""
Step 17 - sign-in page, sign-in/out endpoints and the gate that protects the
dashboard.

  GET  /login        the sign-in page (design: "Login")
  POST /auth/login   {username, password, remember} -> sets the session cookie
  POST /auth/logout  ends the session, back to /login

AuthGateMiddleware sits in front of everything except the paths listed in
_OPEN_PREFIXES: browsers asking for a page are redirected to /login, anything
else (Gradio's API calls) gets 401. The REST API under /api/ keeps its own
API-key authentication and is not affected.
"""
import html
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.config import get_settings
from app.services import auth

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(include_in_schema=False)

COOKIE_NAME = "itsm_session"
_OPEN_PREFIXES = ("/login", "/auth/", "/api/", "/health")

_GENERIC_ERROR = "Incorrect username or password."


def _cookie_secure(request: Request) -> bool:
    if settings.COOKIE_SECURE is not None:
        return settings.COOKIE_SECURE
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return settings.ENV == "prod" or proto == "https"


def user_from_request(request) -> dict | None:
    """The signed-in user for a Starlette/FastAPI request (cookie based), or None."""
    return auth.user_for_token(request.cookies.get(COOKIE_NAME))


@router.get("/login")
async def login_page(request: Request):
    if user_from_request(request):
        return RedirectResponse("/", status_code=303)
    notice = ""
    if auth.count_users() == 0:
        notice = ("No accounts exist yet. The person who runs this app must set <code>ADMIN_PASSWORD</code> "
                  "in the environment and restart it to create the first Admin.")
    elif request.query_params.get("e") == "expired":
        notice = "Your session has ended. Please sign in again."
    return HTMLResponse(render_login(notice), headers={"Cache-Control": "no-store"})


@router.post("/auth/login")
async def login(request: Request):
    try:
        payload = await request.json()
        username, password, remember = str(payload.get("username", "")), str(payload.get("password", "")), bool(payload.get("remember", False))
    except Exception:
        return JSONResponse({"error": "Invalid request."}, status_code=400)
    user, reason = auth.authenticate(username, password)
    if user is None:
        if reason == "locked":
            msg = f"Too many failed attempts. Try again in {settings.LOGIN_LOCK_MINUTES} minutes."
            return JSONResponse({"error": msg}, status_code=429, headers={"Cache-Control": "no-store"})
        return JSONResponse({"error": _GENERIC_ERROR}, status_code=401, headers={"Cache-Control": "no-store"})   # same text for unknown user / bad password / disabled
    token = auth.create_session(user["id"])
    resp = JSONResponse({"ok": True, "redirect": "/"}, headers={"Cache-Control": "no-store"})
    resp.set_cookie(COOKIE_NAME, token, max_age=settings.SESSION_HOURS * 3600 if remember else None,
                    httponly=True, samesite="lax", secure=_cookie_secure(request), path="/")
    return resp


@router.post("/auth/logout")
async def logout(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    user = auth.user_for_token(token)
    auth.end_session(token)
    if user:
        auth.record_event(user["username"], "logout")
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


class AuthGateMiddleware:
    """Pure-ASGI gate (cheaper than BaseHTTPMiddleware, and it never touches request bodies)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or not settings.AUTH_ENABLED:
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if path.startswith(_OPEN_PREFIXES):
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        token = None
        for part in headers.get("cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE_NAME:
                token = value
        if auth.user_for_token(token):
            return await self.app(scope, receive, send)
        if scope["type"] == "http" and scope.get("method") == "GET" and "text/html" in headers.get("accept", ""):
            resp = RedirectResponse("/login", status_code=302)
        else:
            resp = JSONResponse({"detail": "Not authenticated."}, status_code=401)
        await resp(scope, receive, send)


def gradio_auth_dependency(request) -> str | None:
    """Lets Gradio know who is calling (becomes request.username in handlers)."""
    user = user_from_request(request)
    return user["username"] if user else None


# --------------------------------------------------------------------------
# The login page. Tokens come from the design (dark) with a light counterpart;
# the toggle uses the same body.dark class and "itsm-theme" key as the dashboard.
# --------------------------------------------------------------------------
_LOGIN_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in - ITSM Quality Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Sora:wght@600&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<script>try{var t=localStorage.getItem('itsm-theme');if(t==='dark'||(t!=='light'&&window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches))document.documentElement.className='dark';}catch(e){}</script>
<style>
:root{--bg:#F3F5F9;--card:#FFFFFF;--bd:#E2E7EF;--tx:#0F172A;--mu:#5B6577;--lb:#334155;--in:#FFFFFF;--inbd:#CBD3E1;--ac:#0F766E;--acfg:#FFFFFF;--ring:#0F766E;--dot:#E2E7EF;--err-bg:#FEF2F2;--err-bd:#FECACA;--err-tx:#B91C1C;--note-bg:#FFFBEB;--note-bd:#FDE68A;--note-tx:#92400E;--pill:#475569}
html.dark{--bg:#0A0D14;--card:#10141C;--bd:#1F2633;--tx:#E6E9F0;--mu:#8B93A5;--lb:#B6BDCC;--in:#0A0D14;--inbd:#2A3344;--ac:#2DD4BF;--acfg:#04201C;--ring:#2DD4BF;--dot:#1B2230;--err-bg:#2A1214;--err-bd:#5B1F22;--err-tx:#FCA5A5;--note-bg:#2B2410;--note-bd:#5A4A16;--note-tx:#FCD34D;--pill:#B6BDCC}
*{box-sizing:border-box}
body{margin:0;font-family:"DM Sans",system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--tx);background:var(--bg)}
.grid{width:100%;min-height:100vh;display:grid;place-items:center;padding:40px 20px;background-image:radial-gradient(var(--dot) 1px,transparent 1px);background-size:24px 24px}
.wrap{width:100%;max-width:400px}
.brand{display:flex;align-items:center;gap:10px;justify-content:center;margin-bottom:28px}
.brand b{font:600 18px Sora,"DM Sans",sans-serif}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:28px;display:flex;flex-direction:column;gap:18px}
h1{font:600 24px Sora,"DM Sans",sans-serif;margin:0 0 6px;letter-spacing:-.02em}
.sub{margin:0;color:var(--mu);font-size:14px}
label.f{font-size:13px;font-weight:600;color:var(--lb);display:block;margin-bottom:6px}
.in{width:100%;height:42px;border:1px solid var(--inbd);border-radius:8px;padding:0 12px;font:15px "DM Sans",system-ui,sans-serif;background:var(--in);color:var(--tx)}
.in:focus{outline:2px solid var(--ring);outline-offset:0}
.rm{display:flex;gap:10px;align-items:center;font-size:13px;color:var(--lb)}
.rm input{width:16px;height:16px;accent-color:var(--ac)}
.btn{height:44px;border:0;border-radius:8px;background:var(--ac);color:var(--acfg);font:600 15px "DM Sans",system-ui,sans-serif;cursor:pointer}
.btn:disabled{opacity:.6;cursor:progress}
.btn:focus-visible,.tg:focus-visible{outline:2px solid var(--ring);outline-offset:2px}
.foot{text-align:center;margin:20px 0 0;font-size:13px;color:var(--mu);line-height:1.6}
.pills{display:flex;gap:8px;justify-content:center;margin-top:18px}
.pill{font-size:12px;font-weight:500;padding:4px 10px;border-radius:999px;border:1px solid var(--inbd);color:var(--pill)}
.alert{display:none;font-size:13px;line-height:1.5;border-radius:8px;padding:10px 12px;background:var(--err-bg);border:1px solid var(--err-bd);color:var(--err-tx)}
.alert.show{display:block}
.note{font-size:13px;line-height:1.5;border-radius:8px;padding:10px 12px;background:var(--note-bg);border:1px solid var(--note-bd);color:var(--note-tx)}
.note code{font-family:"JetBrains Mono",monospace;font-size:12px}
.tg{position:fixed;top:16px;right:16px;height:34px;padding:0 12px;border-radius:8px;border:1px solid var(--inbd);background:var(--card);color:var(--tx);font:500 13px "DM Sans",system-ui,sans-serif;cursor:pointer}
.tg .l{display:none}html.dark .tg .l{display:inline}html.dark .tg .d{display:none}
</style></head>
<body>
<button class="tg" type="button" id="tg" aria-label="Toggle light or dark theme"><span class="d">&#127769; Dark</span><span class="l">&#9728;&#65039; Light</span></button>
<div class="grid"><div class="wrap">
  <div class="brand"><svg width="28" height="28" viewBox="0 0 36 36" aria-hidden="true"><rect width="36" height="36" rx="9" fill="var(--ac)"/><path d="M9 22l5-6 4 4 8-9" stroke="var(--acfg)" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg><b>ITSM Quality Agent</b></div>
  <form class="card" id="f" novalidate>
    <div><h1>Welcome back</h1><p class="sub">Sign in with the account your administrator created.</p></div>
    __NOTICE__
    <div class="alert" id="err" role="alert" aria-live="assertive"></div>
    <div><label class="f" for="u">Username</label><input class="in" id="u" name="username" type="text" placeholder="e.g. priya.n" autocomplete="username" autocapitalize="none" spellcheck="false" required></div>
    <div><label class="f" for="p">Password</label><input class="in" id="p" name="password" type="password" autocomplete="current-password" required></div>
    <label class="rm"><input type="checkbox" id="r" checked>Keep me signed in on this device</label>
    <button class="btn" type="submit" id="go">Sign in</button>
  </form>
  <p class="foot">No account? Ask your ITSM administrator to add you.<br>Sessions expire after __HOURS__ hours.</p>
  <div class="pills"><span class="pill">Viewer</span><span class="pill">Analyst</span><span class="pill">Admin</span></div>
</div></div>
<script>
(function(){
  document.getElementById('tg').onclick=function(){var d=document.documentElement.classList.toggle('dark');try{localStorage.setItem('itsm-theme',d?'dark':'light')}catch(e){}};
  var f=document.getElementById('f'),err=document.getElementById('err'),go=document.getElementById('go');
  function show(m){err.textContent=m;err.classList.add('show');}
  f.addEventListener('submit',function(ev){
    ev.preventDefault();err.classList.remove('show');
    var u=document.getElementById('u').value.trim(),p=document.getElementById('p').value;
    if(!u||!p){show('Enter your username and password.');return;}
    go.disabled=true;go.textContent='Signing in\\u2026';
    fetch('/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},credentials:'same-origin',
      body:JSON.stringify({username:u,password:p,remember:document.getElementById('r').checked})})
    .then(function(r){return r.json().then(function(j){return {ok:r.ok,j:j};});})
    .then(function(x){if(x.ok){location.href=x.j.redirect||'/';}else{show(x.j.error||'Sign-in failed.');go.disabled=false;go.textContent='Sign in';document.getElementById('p').select();}})
    .catch(function(){show('Could not reach the server. Check your connection and try again.');go.disabled=false;go.textContent='Sign in';});
  });
  document.getElementById('u').focus();
})();
</script></body></html>"""


def render_login(notice_html: str = "") -> str:
    notice = f'<div class="note" role="status">{notice_html}</div>' if notice_html else ""
    return _LOGIN_TEMPLATE.replace("__NOTICE__", notice).replace("__HOURS__", html.escape(str(settings.SESSION_HOURS)))
