"""Step 17 - who may do what inside the Gradio UI.

`permissions_for(request)` is read from the database on every call, so an
admin's change (role, switch, disable) applies to the person's NEXT action
without them signing in again. Checks live in the handlers themselves (not just
in which tabs are hidden), because hiding a tab is cosmetic - anyone can call a
Gradio event directly.

When AUTH_ENABLED=false (local development / behind your own SSO) everyone has
full access, exactly like before Step 17.
"""
import gradio as gr

from app.config import get_settings
from app.services import auth


def current_user(request) -> dict | None:
    username = getattr(request, "username", None)
    return auth.get_user(username) if username else None


def permissions_for(request) -> dict[str, bool]:
    if not get_settings().AUTH_ENABLED:
        return auth.full_access()
    user = current_user(request)
    return user["permissions"] if user else {k: False for k in auth.PERMISSION_KEYS}


def can(request, permission: str) -> bool:
    return bool(permissions_for(request).get(permission, False))


def require(request, permission: str, action: str = "do that") -> None:
    """Raises a user-visible Gradio error if the caller lacks the permission."""
    if not can(request, permission):
        raise gr.Error(f"You don't have permission to {action}. Ask an administrator if you need access.")


def chat_scope(request) -> str | None:
    """"all" | "documents" | "incident" | None (no Q&A allowed)."""
    p = permissions_for(request)
    docs, inc = p.get("qa_documents", False), p.get("qa_incident", False)
    if docs and inc:
        return "all"
    return "documents" if docs else ("incident" if inc else None)
