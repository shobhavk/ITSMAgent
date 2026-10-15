"""
Step 17 - accounts, roles and per-user permissions for the dashboard.

Model (matches the Admin design):
  * Three roles - Viewer, Analyst, Admin - each with default permissions.
  * An admin can switch individual permissions on/off for one person
    ("overrides"); the role default is what applies when there is no override.
  * Accounts are created by an admin only (no self-registration).

Security choices (all stdlib, no new dependencies):
  * Passwords: scrypt with a random per-user salt, constant-time compare.
    Plain passwords are never stored or logged.
  * Sessions: random 256-bit token in an HttpOnly cookie; only its SHA-256 is
    stored server-side, so disabling a user or signing out takes effect
    immediately and a database leak does not yield usable cookies.
  * Brute force: after LOGIN_MAX_FAILED wrong passwords the account is locked
    for LOGIN_LOCK_MINUTES. Login errors never say which part was wrong, and a
    dummy hash is checked for unknown usernames (no timing oracle).
  * The last active admin can never be disabled or demoted (no lock-out).
  * The activity log stores who did what and when - never ticket text or
    question text.
"""
import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.services import persistence
from app.services.persistence import AuthEvent, AuthSession, SessionLocal, User

logger = logging.getLogger(__name__)
settings = get_settings()

ROLES = ("viewer", "analyst", "admin")
ROLE_LABELS = {"viewer": "Viewer", "analyst": "Analyst", "admin": "Admin"}
ROLE_BLURB = {
    "viewer": "Overview and knowledge-base Q&A.",
    "analyst": "Upload, categorize and use all analysis tabs.",
    "admin": "Everything, plus managing users.",
}

# (key, group, label, description, available_in_this_version)
PERMISSIONS = [
    ("view_overview", "ANALYZE", "View Overview", "See the incident health summary", True),
    ("view_trends", "ANALYZE", "View Trends & Insights", "See volume and recurrence trends", True),
    ("view_recommendations", "ANALYZE", "View Recommendations", "See suggested improvements", True),
    ("run_categorization", "ANALYZE", "Run categorization", "Open the Categorization results", True),
    ("export_csv", "ANALYZE", "Export CSV", "Download analysis results", True),
    ("upload_data", "DATA", "Upload incident data", "Upload files or paste incidents to analyze", True),
    ("publish_viewers", "DATA", "Publish to viewers", "Share an analysis with Viewers", False),
    ("view_kb", "KNOWLEDGE", "View Knowledge Base tab", "See the document library", True),
    ("manage_kb", "KNOWLEDGE", "Manage KB documents", "Upload, delete and re-index documents", True),
    ("qa_documents", "KNOWLEDGE", "Q&A over documents", "Ask questions answered from the knowledge base", True),
    ("qa_incident", "KNOWLEDGE", "Q&A over incident data", "Ask questions about the analyzed incidents", True),
    ("give_feedback", "WORKSPACE", "Give feedback on answers", "Rate Q&A answers", False),
    ("change_settings", "WORKSPACE", "Change settings", "Edit application settings", True),
    ("view_observability", "WORKSPACE", "View Observability", "See health, latency and error metrics", True),
    ("manage_users", "WORKSPACE", "Manage users", "Create accounts and change roles", True),
]
PERMISSION_KEYS = [p[0] for p in PERMISSIONS]

ROLE_DEFAULTS = {
    "viewer": {"view_overview", "qa_documents", "give_feedback"},
    "analyst": {"view_overview", "view_trends", "view_recommendations", "run_categorization", "export_csv",
                "upload_data", "publish_viewers", "view_kb", "manage_kb", "qa_documents", "qa_incident",
                "give_feedback"},
    "admin": set(PERMISSION_KEYS),
}

_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._\-]{2,31}$")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)     # naive UTC, as the other tables store it


# --------------------------------------------------------------------------
# Passwords
# --------------------------------------------------------------------------
_SCRYPT = (2 ** 14, 8, 1)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    n, r, p = _SCRYPT
    digest = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    return "scrypt${}${}${}${}${}".format(n, r, p, base64.b64encode(salt).decode(), base64.b64encode(digest).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        calc = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, base64.b64decode(digest))
    except Exception:
        return False


_DUMMY_HASH = hash_password("dummy-password-for-timing")


def validate_password(password: str, username: str = "") -> str | None:
    """Returns a user-safe problem description, or None if acceptable."""
    if len(password) < 10:
        return "Password must be at least 10 characters."
    if len(password) > 128:
        return "Password must be at most 128 characters."
    if username and password.lower() == username.lower():
        return "Password must not be the same as the username."
    return None


def generate_temp_password() -> str:
    return secrets.token_urlsafe(9)        # 12 url-safe characters, ~72 bits


# --------------------------------------------------------------------------
# Permissions
# --------------------------------------------------------------------------
def _overrides(user: User) -> dict:
    try:
        raw = json.loads(user.overrides_json or "{}")
    except ValueError:
        return {}
    return {k: bool(v) for k, v in raw.items() if k in PERMISSION_KEYS}


def effective_permissions(user: User | None) -> dict[str, bool]:
    """{permission: allowed} for one user. Disabled users have nothing."""
    if user is None or user.status != "active":
        return {k: False for k in PERMISSION_KEYS}
    defaults = ROLE_DEFAULTS.get(user.role, set())
    ov = _overrides(user)
    perms = {k: ov.get(k, k in defaults) for k in PERMISSION_KEYS}
    if user.role == "admin":
        perms["manage_users"] = True        # an admin can never lose user management (no lock-out)
    return perms


def permission_overrides(user: User) -> dict[str, bool]:
    """Only the permissions whose value differs from the role default."""
    defaults = ROLE_DEFAULTS.get(user.role, set())
    return {k: v for k, v in _overrides(user).items() if v != (k in defaults)}


def full_access() -> dict[str, bool]:
    return {k: True for k in PERMISSION_KEYS}


# --------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------
class AuthError(ValueError):
    """Message is written for administrators/end users and is safe to display."""


def _public(user: User) -> dict:
    return {
        "id": user.id, "username": user.username, "display_name": user.display_name or user.username,
        "email": user.email, "role": user.role, "status": user.status, "created_at": user.created_at,
        "last_active_at": user.last_active_at, "last_login_at": user.last_login_at,
        "overrides": permission_overrides(user), "permissions": effective_permissions(user),
    }


def count_users() -> int:
    with SessionLocal() as db:
        return db.query(User).count()


def get_user(username: str) -> dict | None:
    with SessionLocal() as db:
        u = db.query(User).filter(User.username == (username or "").strip().lower()).first()
        return _public(u) if u else None


def list_users(search: str = "", flt: str = "all") -> list[dict]:
    with SessionLocal() as db:
        users = db.query(User).order_by(User.created_at).all()
        rows = [_public(u) for u in users]
    q = (search or "").strip().lower()
    if q:
        rows = [r for r in rows if q in r["username"] or q in r["display_name"].lower() or q in r["email"].lower()]
    if flt in ("admin", "analyst", "viewer"):
        rows = [r for r in rows if r["role"] == flt]
    elif flt == "disabled":
        rows = [r for r in rows if r["status"] == "disabled"]
    return rows


def _active_admin_count(db, excluding_id: int | None = None) -> int:
    q = db.query(User).filter(User.role == "admin", User.status == "active")
    if excluding_id is not None:
        q = q.filter(User.id != excluding_id)
    return q.count()


def create_user(username: str, display_name: str = "", email: str = "", role: str = "viewer",
                password: str | None = None, actor: str = "system") -> tuple[dict, str]:
    """Creates an account. Returns (user, temporary_password). If `password`
    is None a random temporary password is generated; show it to the admin
    ONCE - it is not recoverable afterwards (only resettable)."""
    username = (username or "").strip().lower()
    if not _USERNAME_RE.match(username):
        raise AuthError("Username must be 3-32 characters: lower-case letters, digits, '.', '_' or '-'.")
    if role not in ROLES:
        raise AuthError("Choose Viewer, Analyst or Admin.")
    display_name = (display_name or "").strip()[:120]
    email = (email or "").strip()[:200]
    if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise AuthError("That email address doesn't look right.")
    temp = password if password is not None else generate_temp_password()
    problem = validate_password(temp, username)
    if problem:
        raise AuthError(problem)
    with SessionLocal() as db:
        if db.query(User).filter(User.username == username).first():
            raise AuthError("That username is already taken.")
        u = User(username=username, display_name=display_name, email=email, password_hash=hash_password(temp),
                 role=role, status="active", overrides_json="{}", created_at=_now(), failed_attempts=0)
        db.add(u)
        db.commit()
        out = _public(u)
    record_event(username, "user_created", f"by {actor}; role {role}")
    return out, temp


def _load(db, username: str) -> User:
    u = db.query(User).filter(User.username == (username or "").strip().lower()).first()
    if not u:
        raise AuthError("That user no longer exists.")
    return u


def set_role(username: str, role: str, actor: str = "system") -> dict:
    if role not in ROLES:
        raise AuthError("Choose Viewer, Analyst or Admin.")
    with SessionLocal() as db:
        u = _load(db, username)
        if u.role == "admin" and role != "admin" and u.status == "active" and _active_admin_count(db, u.id) == 0:
            raise AuthError("At least one active Admin must remain.")
        old, u.role = u.role, role
        db.commit()
        out = _public(u)
    record_event(username, "role_changed", f"{old} -> {role} by {actor}")
    return out


def set_status(username: str, status: str, actor: str = "system") -> dict:
    if status not in ("active", "disabled"):
        raise AuthError("Invalid status.")
    with SessionLocal() as db:
        u = _load(db, username)
        if status == "disabled" and u.role == "admin" and u.status == "active" and _active_admin_count(db, u.id) == 0:
            raise AuthError("At least one active Admin must remain.")
        u.status = status
        if status == "disabled":
            db.query(AuthSession).filter(AuthSession.user_id == u.id).delete()    # sign them out everywhere
        else:
            u.failed_attempts, u.locked_until = 0, None
        db.commit()
        out = _public(u)
    record_event(username, "user_" + status, f"by {actor}")
    return out


def reset_password(username: str, actor: str = "system") -> str:
    """Sets a new random temporary password, signs the user out everywhere and
    returns the password (show it once)."""
    temp = generate_temp_password()
    with SessionLocal() as db:
        u = _load(db, username)
        u.password_hash = hash_password(temp)
        u.failed_attempts, u.locked_until = 0, None
        db.query(AuthSession).filter(AuthSession.user_id == u.id).delete()
        db.commit()
    record_event(username, "password_reset", f"by {actor}")
    return temp


def set_overrides(username: str, desired: dict[str, bool], actor: str = "system") -> dict:
    """`desired` is the full permission map the admin wants; only values that
    differ from the role default are stored."""
    with SessionLocal() as db:
        u = _load(db, username)
        defaults = ROLE_DEFAULTS.get(u.role, set())
        ov = {k: bool(v) for k, v in desired.items() if k in PERMISSION_KEYS and bool(v) != (k in defaults)}
        if u.role == "admin":
            ov.pop("manage_users", None)
        u.overrides_json = json.dumps(ov)
        db.commit()
        out = _public(u)
    record_event(username, "permissions_changed", f"{len(ov)} override(s) by {actor}")
    return out


# --------------------------------------------------------------------------
# Activity
# --------------------------------------------------------------------------
def record_event(username: str, event: str, detail: str = "") -> None:
    try:
        with SessionLocal() as db:
            db.add(AuthEvent(ts=_now(), username=(username or "")[:64], event=event[:32], detail=(detail or "")[:200]))
            db.commit()
    except Exception:
        logger.warning("Could not record auth event")


def recent_events(username: str, limit: int = 8) -> list[dict]:
    with SessionLocal() as db:
        rows = (db.query(AuthEvent).filter(AuthEvent.username == username.lower())
                .order_by(AuthEvent.ts.desc(), AuthEvent.id.desc()).limit(limit).all())
        return [{"ts": r.ts, "event": r.event, "detail": r.detail} for r in rows]


def activity_summary(username: str) -> dict:
    since = _now() - timedelta(days=7)
    with SessionLocal() as db:
        q = db.query(AuthEvent).filter(AuthEvent.username == username.lower(), AuthEvent.ts >= since)
        failed = q.filter(AuthEvent.event == "login_failed").order_by(AuthEvent.ts.desc()).all()
        qa = q.filter(AuthEvent.event == "qa_asked").count()
    return {"failed_7d": len(failed), "last_failed": failed[0].ts if failed else None, "qa_7d": qa}


def humanize(ts: datetime | None) -> str:
    if not ts:
        return "Never"
    delta = _now() - ts
    s = int(delta.total_seconds())
    if s < 60:
        return "Just now"
    if s < 3600:
        return f"{s // 60} min ago"
    if s < 86400:
        return f"{s // 3600} h ago"
    if s < 172800:
        return "Yesterday"
    if s < 86400 * 30:
        return f"{s // 86400} days ago"
    return ts.strftime("%d %b %Y")


# --------------------------------------------------------------------------
# Sign-in and sessions
# --------------------------------------------------------------------------
def authenticate(username: str, password: str) -> tuple[dict | None, str]:
    """Returns (user, "") on success or (None, reason) where reason is one of
    "invalid" | "locked" | "disabled". Callers show ONE generic message for
    "invalid"/"disabled" so accounts can't be enumerated."""
    uname = (username or "").strip().lower()
    password = password or ""
    if not uname or not password or len(uname) > 64 or len(password) > 256:
        verify_password("x", _DUMMY_HASH)
        return None, "invalid"
    with SessionLocal() as db:
        u = db.query(User).filter(User.username == uname).first()
        if u is None:
            verify_password(password, _DUMMY_HASH)         # same work as a real check
            record_event(uname, "login_failed", "unknown user")
            return None, "invalid"
        if u.locked_until and u.locked_until > _now():
            record_event(uname, "login_failed", "locked")
            return None, "locked"
        ok = verify_password(password, u.password_hash)
        if not ok:
            u.failed_attempts = (u.failed_attempts or 0) + 1
            if u.failed_attempts >= settings.LOGIN_MAX_FAILED:
                u.locked_until = _now() + timedelta(minutes=settings.LOGIN_LOCK_MINUTES)
                u.failed_attempts = 0
            db.commit()
            record_event(uname, "login_failed", "bad password")
            return None, "invalid"
        if u.status != "active":
            record_event(uname, "login_failed", "disabled account")
            return None, "disabled"
        u.failed_attempts, u.locked_until, u.last_login_at, u.last_active_at = 0, None, _now(), _now()
        db.commit()
        out = _public(u)
    record_event(uname, "login_ok")
    return out, ""


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with SessionLocal() as db:
        db.query(AuthSession).filter(AuthSession.expires_at < _now()).delete()      # tidy expired ones
        db.add(AuthSession(token_hash=_token_hash(token), user_id=user_id, created_at=_now(),
                           expires_at=_now() + timedelta(hours=settings.SESSION_HOURS)))
        db.commit()
    return token


def user_for_token(token: str | None) -> dict | None:
    """The signed-in user for a cookie value, or None (unknown / expired / disabled)."""
    if not token or len(token) > 200:
        return None
    with SessionLocal() as db:
        s = db.get(AuthSession, _token_hash(token))
        if not s or s.expires_at < _now():
            return None
        u = db.get(User, s.user_id)
        if not u or u.status != "active":
            return None
        if not u.last_active_at or (_now() - u.last_active_at).total_seconds() > 60:
            u.last_active_at = _now()
            db.commit()
        return _public(u)


def end_session(token: str | None) -> None:
    if not token:
        return
    with SessionLocal() as db:
        db.query(AuthSession).filter(AuthSession.token_hash == _token_hash(token)).delete()
        db.commit()


# --------------------------------------------------------------------------
# First-run bootstrap
# --------------------------------------------------------------------------
def bootstrap_admin() -> str | None:
    """If there are no accounts yet, create the first Admin from ADMIN_PASSWORD
    (or from the legacy UI_AUTH_USERS pairs). Returns a short status string for
    the startup log, never a password."""
    if count_users() > 0:
        return None
    created = []
    if settings.ADMIN_PASSWORD:
        try:
            create_user(settings.ADMIN_USERNAME, "Administrator", "", "admin", password=settings.ADMIN_PASSWORD, actor="bootstrap")
            created.append(settings.ADMIN_USERNAME)
        except AuthError as exc:
            logger.error("Could not create the bootstrap admin: %s", exc)
    for pair in settings.UI_AUTH_USERS.split(","):
        if ":" in pair:
            u, _, pw = pair.strip().partition(":")
            try:
                create_user(u, u, "", "admin", password=pw, actor="bootstrap")
                created.append(u)
            except AuthError as exc:
                logger.error("Could not import legacy user '%s': %s", u[:32], exc)
    return f"created {len(created)} admin account(s)" if created else None
