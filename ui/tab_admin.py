"""Step 17 - Admin page ("Access control" + "Advanced permissions" designs).

Everything here is gated twice: the tab is only shown to people who hold the
"manage_users" permission, AND every handler re-checks it on the server side
(`require`) - hiding a tab is cosmetic, handlers can be called directly.

Passwords: an admin never sees or chooses another person's password. A random
temporary password is generated when inviting or resetting and shown ONCE.
"""
import html
import logging

import gradio as gr

from app.services import auth
from ui.access import current_user, require

logger = logging.getLogger(__name__)

_AVATARS = [("#1B2A33", "#5EEAD4"), ("#2A2540", "#C4B5FD"), ("#3B2A14", "#FCD34D"), ("#3B1F2B", "#F9A8D4"), ("#14283B", "#93C5FD")]
_FILTERS = [("All", "all"), ("Admins", "admin"), ("Analysts", "analyst"), ("Viewers", "viewer"), ("Disabled", "disabled")]
_EVENT_TEXT = {
    "login_ok": "Signed in", "login_failed": "Failed sign-in", "logout": "Signed out", "qa_asked": "Asked a Q&A question",
    "role_changed": "Role changed", "user_created": "Account created", "user_disabled": "Account disabled",
    "user_active": "Account re-enabled", "password_reset": "Password reset", "permissions_changed": "Permissions changed",
}


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _initials(u: dict) -> str:
    parts = (u["display_name"] or u["username"]).replace(".", " ").replace("_", " ").split()
    return ((parts[0][0] + (parts[1][0] if len(parts) > 1 else "")) if parts else "?").upper()


def _avatar(u: dict, size: int = 30) -> str:
    bg, fg = _AVATARS[sum(map(ord, u["username"])) % len(_AVATARS)]
    return (f'<span class="adm-av" style="width:{size}px;height:{size}px;background:{bg};color:{fg}">{_e(_initials(u))}</span>')


def _seg(role: str) -> str:
    return '<span class="adm-seg">' + "".join(
        f'<span class="{"a" if r == role else ""}">{auth.ROLE_LABELS[r]}</span>' for r in auth.ROLES) + "</span>"


def _status(u: dict) -> str:
    off = u["status"] != "active"
    return f'<span class="adm-st{" off" if off else ""}"><i></i>{"Disabled" if off else "Active"}</span>'


def _header_html() -> str:
    return ('<div class="adm-head"><div><h2>Access control</h2>'
            '<p>Choose who can upload and analyze. Everyone else gets the Overview and knowledge-base Q&amp;A.</p></div></div>')


def _legend_html() -> str:
    cards = "".join(
        f'<div class="adm-card adm-role"><b>{auth.ROLE_LABELS[r]}</b><span>{_e(auth.ROLE_BLURB[r])}</span></div>' for r in auth.ROLES)
    return f'<div class="adm-legend">{cards}</div>'


def _row_html(u: dict) -> str:
    return (f'<div class="adm-grid">'
            f'<div class="adm-who">{_avatar(u)}<div><b>{_e(u["display_name"])}</b><small>@{_e(u["username"])}</small></div></div>'
            f'<div>{_seg(u["role"])}</div><div>{_status(u)}<span class="adm-last">{_e(auth.humanize(u["last_active_at"]))}</span></div></div>')


def _table_head_html(count: int) -> str:
    return (f'<div class="adm-grid adm-thead"><div>USER</div><div>ROLE</div><div>STATUS</div></div>'
            if count else '<div class="adm-empty">No one matches that search.</div>')


def _panel_head_html(u: dict | None) -> str:
    if not u:
        return '<div class="adm-empty">Select a person from the list to manage their access.</div>'
    return (f'<div class="adm-phead">{_avatar(u, 42)}<div><b>{_e(u["display_name"])}</b>'
            f'<small>@{_e(u["username"])}{" · " + _e(u["email"]) if u["email"] else ""}</small></div>'
            f'<span class="adm-pill">{auth.ROLE_LABELS[u["role"]]}</span></div>')


def _activity_html(u: dict | None) -> str:
    if not u:
        return ""
    s = auth.activity_summary(u["username"])
    last_failed = f' · last {_e(auth.humanize(s["last_failed"]))}' if s["last_failed"] else ""
    items = []
    for ev in auth.recent_events(u["username"], 8):
        items.append(f'<li><span>{_e(_EVENT_TEXT.get(ev["event"], ev["event"]))}'
                     f'{" · " + _e(ev["detail"]) if ev["detail"] and ev["event"] not in ("login_failed", "qa_asked") else ""}</span>'
                     f'<small>{_e(auth.humanize(ev["ts"]))}</small></li>')
    return (f'<div class="adm-act"><div class="adm-stats">'
            f'<div><b>{s["failed_7d"]}</b><span>failed sign-ins (7 days){last_failed}</span></div>'
            f'<div><b>{s["qa_7d"]}</b><span>Q&amp;A questions (7 days)</span></div>'
            f'<div><b>{_e(auth.humanize(u["last_login_at"]))}</b><span>last sign-in</span></div></div>'
            f'<h4>RECENT ACTIVITY</h4><ul>{"".join(items) or "<li><span>No activity yet</span></li>"}</ul></div>')


def _msg(text: str, kind: str = "ok") -> str:
    return f'<div class="adm-msg {kind}" role="status">{text}</div>' if text else ""


def _secret_box(title: str, username: str, password: str) -> str:
    return _msg(f'{_e(title)} <b>@{_e(username)}</b><br>Temporary password: <code class="adm-secret">{_e(password)}</code><br>'
                '<small>Shown once - copy it now and share it securely. They should be told to keep it private.</small>')


def build_admin_tab(tab: gr.Tab):
    """Builds the page inside the current (Admin) tab."""
    selected = gr.State("")
    refresh = gr.State(0)

    with gr.Column(elem_classes=["adm"]):
        with gr.Row(elem_classes=["adm-top"]):
            gr.HTML(_header_html())
            invite_btn = gr.Button("+ Invite user", variant="primary", elem_classes=["adm-primary"], scale=0, min_width=150)
        gr.HTML(_legend_html())

        # ---- invite form (hidden until "+ Invite user") ----
        with gr.Group(visible=False, elem_classes=["adm-card", "adm-invite"]) as invite_box:
            gr.HTML("<h3>Invite a user</h3><p class='adm-muted'>They sign in with the username below and the temporary password you will be shown.</p>")
            with gr.Row():
                inv_username = gr.Textbox(label="Username", placeholder="e.g. priya.n", max_lines=1)
                inv_name = gr.Textbox(label="Full name", placeholder="e.g. Priya Nair", max_lines=1)
            with gr.Row():
                inv_email = gr.Textbox(label="Email (optional)", placeholder="priya@company.com", max_lines=1)
                inv_role = gr.Radio([(auth.ROLE_LABELS[r], r) for r in auth.ROLES], value="viewer", label="Role")
            with gr.Row():
                inv_create = gr.Button("Create account", variant="primary", elem_classes=["adm-primary"])
                inv_cancel = gr.Button("Cancel", elem_classes=["adm-ghost"])
        invite_msg = gr.HTML("")

        with gr.Row(elem_classes=["adm-bar"]):
            search = gr.Textbox(placeholder="Search by name, username or email", show_label=False, container=False, max_lines=1, scale=3)
            flt = gr.Radio(_FILTERS, value="all", show_label=False, container=False, elem_classes=["adm-pills"], scale=4)

        with gr.Row(equal_height=False):
            # ---------------- user table ----------------
            with gr.Column(scale=5, elem_classes=["adm-card", "adm-table"]):
                @gr.render(inputs=[search, flt, refresh, selected])
                def _render_users(q, f, _r, sel, request: gr.Request):
                    if not auth_ok(request):
                        gr.HTML(_msg("You don't have access to this page.", "err"))
                        return
                    rows = auth.list_users(q or "", f or "all")
                    gr.HTML(_table_head_html(len(rows)))
                    for u in rows:
                        with gr.Row(elem_classes=["adm-tr"] + (["sel"] if u["username"] == sel else []), equal_height=True):
                            gr.HTML(_row_html(u))
                            b = gr.Button("Manage", size="sm", elem_classes=["adm-manage"], scale=0, min_width=84)
                            b.click(fn=_make_picker(u["username"]), inputs=None, outputs=panel_outputs)

            # ---------------- detail panel ----------------
            with gr.Column(scale=4, elem_classes=["adm-card", "adm-panel"]):
                panel_head = gr.HTML(_panel_head_html(None))
                panel_view = gr.Radio([("Role", "role"), ("Permissions", "perms"), ("Activity", "activity")], value="role",
                                      show_label=False, container=False, elem_classes=["adm-pills"])
                with gr.Column(visible=True) as view_role:
                    role_radio = gr.Radio([(f"{auth.ROLE_LABELS[r]} - {auth.ROLE_BLURB[r]}", r) for r in auth.ROLES],
                                          value="viewer", label="Role", elem_classes=["adm-roles"])
                    save_role = gr.Button("Save role", variant="primary", elem_classes=["adm-primary"])
                with gr.Column(visible=False) as view_perms:
                    gr.HTML("<p class='adm-muted'>Fine-tune this person. Switches start from their role; a dot marks anything you changed.</p>")
                    perm_boxes = {}
                    last_group = None
                    for key, group, label, desc, available in auth.PERMISSIONS:
                        if group != last_group:
                            gr.HTML(f'<div class="adm-group">{group}</div>')
                            last_group = group
                        perm_boxes[key] = gr.Checkbox(label=label, info=desc + ("" if available else " · not in this version yet"),
                                                      elem_classes=["adm-perm"])
                    with gr.Row():
                        save_perms = gr.Button("Save permissions", variant="primary", elem_classes=["adm-primary"])
                        reset_perms = gr.Button("Reset to role defaults", elem_classes=["adm-ghost"])
                with gr.Column(visible=False) as view_act:
                    activity = gr.HTML("")
                panel_view.change(lambda v: (gr.update(visible=v == "role"), gr.update(visible=v == "perms"), gr.update(visible=v == "activity")),
                                  inputs=[panel_view], outputs=[view_role, view_perms, view_act])
                gr.HTML("<hr class='adm-hr'>")
                with gr.Row():
                    reset_pw = gr.Button("Reset password", elem_classes=["adm-ghost"])
                    toggle_btn = gr.Button("Disable user", elem_classes=["adm-danger"])
                panel_msg = gr.HTML("")

    perm_list = [perm_boxes[k] for k in auth.PERMISSION_KEYS]

    # ------------------------------------------------------------------
    # helpers shared by the handlers
    # ------------------------------------------------------------------
    def auth_ok(request) -> bool:
        try:
            require(request, "manage_users")
            return True
        except gr.Error:
            return False

    def _panel_updates(username: str, message: str = "") -> list:
        """Outputs for panel_outputs (see below) for one person."""
        u = auth.get_user(username)
        if not u:
            return [username, _panel_head_html(None), gr.update(), *[gr.update() for _ in perm_list], "", gr.update(), message]
        defaults = auth.ROLE_DEFAULTS.get(u["role"], set())
        boxes = []
        for key, _g, _l, desc, available in auth.PERMISSIONS:
            changed = u["permissions"][key] != (key in defaults)
            info = desc + ("" if available else " · not in this version yet") + (" · ● differs from role default" if changed else "")
            boxes.append(gr.update(value=u["permissions"][key], info=info, interactive=not (u["role"] == "admin" and key == "manage_users")))
        return [u["username"], _panel_head_html(u), gr.update(value=u["role"]), *boxes, _activity_html(u),
                gr.update(value="Disable user" if u["status"] == "active" else "Enable user"), message]

    panel_outputs = [selected, panel_head, role_radio, *perm_list, activity, toggle_btn, panel_msg]

    def _make_picker(username: str):
        def _pick(request: gr.Request):
            require(request, "manage_users", "manage users")
            return _panel_updates(username)
        return _pick

    def _actor(request) -> str:
        user = current_user(request)
        return user["username"] if user else "admin"

    def _is_self(request, username: str) -> bool:
        return _actor(request) == (username or "").lower()

    # ------------------------------------------------------------------
    # handlers
    # ------------------------------------------------------------------
    def _toggle_invite(visible):
        return gr.update(visible=not visible), not visible

    invite_open = gr.State(False)
    invite_btn.click(_toggle_invite, inputs=[invite_open], outputs=[invite_box, invite_open])
    inv_cancel.click(lambda: (gr.update(visible=False), False, ""), outputs=[invite_box, invite_open, invite_msg])

    def _create(username, name, email, role, rf, request: gr.Request):
        require(request, "manage_users", "create accounts")
        try:
            user, temp = auth.create_user(username, name, email, role, actor=_actor(request))
        except auth.AuthError as exc:
            return _msg(_e(exc), "err"), rf, gr.update(), gr.update(), gr.update(), gr.update()
        except Exception:
            logger.exception("Creating a user failed")
            return _msg("Could not create the account. See the application log.", "err"), rf, gr.update(), gr.update(), gr.update(), gr.update()
        return (_secret_box("Account created for", user["username"], temp), rf + 1, "", "", "", "viewer")

    inv_create.click(_create, inputs=[inv_username, inv_name, inv_email, inv_role, refresh],
                     outputs=[invite_msg, refresh, inv_username, inv_name, inv_email, inv_role])

    def _after(username: str, message: str, rf: int, request):
        return [*_panel_updates(username, message), rf + 1]

    def _save_role(username, role, rf, request: gr.Request):
        require(request, "manage_users", "change roles")
        if not username:
            return [*_panel_updates(""), rf]
        if _is_self(request, username):
            return [*_panel_updates(username, _msg("You can't change your own role. Ask another Admin.", "err")), rf]
        try:
            auth.set_role(username, role, actor=_actor(request))
            return _after(username, _msg(f"Role saved: {auth.ROLE_LABELS[role]}."), rf, request)
        except auth.AuthError as exc:
            return [*_panel_updates(username, _msg(_e(exc), "err")), rf]

    def _save_perms(username, rf, request: gr.Request, *values):
        require(request, "manage_users", "change permissions")
        if not username:
            return [*_panel_updates(""), rf]
        try:
            auth.set_overrides(username, dict(zip(auth.PERMISSION_KEYS, values)), actor=_actor(request))
            return _after(username, _msg("Permissions saved."), rf, request)
        except auth.AuthError as exc:
            return [*_panel_updates(username, _msg(_e(exc), "err")), rf]

    def _reset_perms(username, rf, request: gr.Request):
        require(request, "manage_users", "change permissions")
        u = auth.get_user(username)
        if not u:
            return [*_panel_updates(""), rf]
        auth.set_overrides(username, {k: k in auth.ROLE_DEFAULTS[u["role"]] for k in auth.PERMISSION_KEYS}, actor=_actor(request))
        return _after(username, _msg("Permissions reset to the role defaults."), rf, request)

    def _reset_password(username, rf, request: gr.Request):
        require(request, "manage_users", "reset passwords")
        if not username:
            return [*_panel_updates(""), rf]
        try:
            temp = auth.reset_password(username, actor=_actor(request))
        except auth.AuthError as exc:
            return [*_panel_updates(username, _msg(_e(exc), "err")), rf]
        return [*_panel_updates(username, _secret_box("New password for", username, temp)), rf + 1]

    def _toggle_status(username, rf, request: gr.Request):
        require(request, "manage_users", "disable or enable users")
        u = auth.get_user(username)
        if not u:
            return [*_panel_updates(""), rf]
        if _is_self(request, username):
            return [*_panel_updates(username, _msg("You can't disable your own account.", "err")), rf]
        try:
            new = "disabled" if u["status"] == "active" else "active"
            auth.set_status(username, new, actor=_actor(request))
            return _after(username, _msg("Account disabled. They have been signed out." if new == "disabled" else "Account re-enabled."), rf, request)
        except auth.AuthError as exc:
            return [*_panel_updates(username, _msg(_e(exc), "err")), rf]

    panel_plus = [*panel_outputs, refresh]
    save_role.click(_save_role, inputs=[selected, role_radio, refresh], outputs=panel_plus)
    save_perms.click(_save_perms, inputs=[selected, refresh, *perm_list], outputs=panel_plus)
    reset_perms.click(_reset_perms, inputs=[selected, refresh], outputs=panel_plus)
    reset_pw.click(_reset_password, inputs=[selected, refresh], outputs=panel_plus)
    toggle_btn.click(_toggle_status, inputs=[selected, refresh], outputs=panel_plus)

    # opening the tab (or typing in search) re-reads the list so it is always current
    tab.select(lambda rf: rf + 1, inputs=[refresh], outputs=[refresh])
