"""Q&A (Agent) tab: chat index and chat handlers."""
import html as _html

import gradio as gr
import pandas as pd

from app.config import get_settings
from app.services import auth, knowledge_base, observability, rag
from ui.access import chat_scope, current_user
from ui.chat_format import clean_answer


def _build_chat_index(full_df: pd.DataFrame) -> "rag.TicketIndex":
    """Rebuilds the chat/RAG index from the latest analysis dataframe.
    Cheap and synchronous - embeddings are computed lazily on first
    question (see rag.TicketIndex._ensure_vectors), not here. Also the
    index the "Detect Similar Recurring Issues" button clusters over -
    whichever of chat or that button runs first pays the one embedding
    call; the other reuses the same cached vectors for free."""
    return rag.build_index_from_dataframe(full_df)


def _history_to_messages(history: list[tuple[str, str]]) -> list[dict]:
    """Converts the internal (question, answer) tuple history - the shape
    rag.answer_question expects for prompt context - into the role/content
    message dicts gr.Chatbot renders in this Gradio version."""
    messages = []
    for question, answer in history:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return messages


async def _chat_respond(message: str, history: list, full_df, stats: dict, request: gr.Request = None):
    message = (message or "").strip()
    if not message:
        return _history_to_messages(history), history, ""

    limit = get_settings().MAX_QUESTION_CHARS
    scope = chat_scope(request)                       # Step 17: what this person may ask about
    if len(message) > limit:
        answer = f"That question is too long ({len(message):,} characters; the limit is {limit:,}). Please shorten it."
    elif scope is None:
        answer = "Q&A isn't enabled for your account. Ask an administrator if you need access."
    elif scope != "documents" and (full_df is None or len(full_df) == 0):
        answer = "Run an analysis on the Dashboard tab first - then come back and ask away."
    else:
        # The model (or the no-LLM fallback) sometimes answers with a JSON blob; show readable text.
        # Step 15: one traced request per question; the Gradio session hash is
        # pseudonymised before it reaches any log (see observability.pseudonymize).
        if full_df is None:                           # documents-only Q&A needs no incident data
            full_df = pd.DataFrame()
        with observability.request_trace("chat", getattr(request, "session_hash", None), source="gradio"):
            answer = clean_answer(await rag.answer_question(message, full_df, stats or {}, history, scope))
        user = current_user(request)
        if user:
            auth.record_event(user["username"], "qa_asked")      # who/when only - never the question text

    history = history + [(message, answer)]
    return _history_to_messages(history), history, ""


def _chat_clear():
    return [], []


################################################################
# Q&A (Agent) page - presentation layer (Step 17 design)
#
# Everything below only DRAWS the page and fills the question box. The behaviour is the
# functions above (_chat_respond / _chat_clear / _build_chat_index), unchanged: every button
# on this page ends up calling _chat_respond with exactly the same inputs as the Send button.
#
#   Viewer  (chat_scope == "documents")  -> centred "Ask your knowledge base" page + 4 starter cards
#   Analyst / Admin  ("all" / "incident") -> chat card with data-source line + Prompt library
################################################################

_ICONS = {
    "arrow": '<path d="M5 12h14m-6-6l6 6-6 6"/>',
    "chart": '<path d="M4 19V9m6 10V5m6 14v-7"/>',
    "bulb": '<path d="M9 18h6M12 3a6 6 0 00-4 10c1 1 1 2 1 3h6c0-1 0-2 1-3a6 6 0 00-4-10z"/>',
    "book": '<path d="M5 4h11a3 3 0 013 3v13H8a3 3 0 01-3-3z"/>',
    "list": '<path d="M4 6h16M4 12h16M4 18h10"/>',
    "chat": '<path d="M4 5h16v11H9l-5 4z"/>',
}


def _svg(name: str, size: int = 13) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>')


# (heading, icon, needs) - `needs` is which Q&A scope may see the group: "incident", "documents" or "all" (both).
QA_PROMPT_LIBRARY = [
    ("INCIDENT DATA", "chart", "incident", [
        "Give me an overall summary of the incidents.",
        "How many P1 incidents are there?",
        "Which server has the most incidents?",
        "List the top 3 incident categories.",
        "What is the average worklog score?",
    ]),
    ("RECURRING & ACTIONS", "bulb", "incident", [
        "What issues keep recurring?",
        "What needs attention first?",
        "What should we do to improve?",
    ]),
    ("KNOWLEDGE BASE", "book", "documents", [
        "What is the escalation procedure for P1 incidents?",
        "What does the backup SOP say to do after a backup failure?",
        "How do I troubleshoot VPN connection problems?",
        "What is the documented procedure for database connection failures?",
    ]),
    ("DATA + DOCUMENTS", "list", "all", [
        "We have recurring database incidents. What does the runbook recommend?",
        "How many P1 incidents do we have, and what does the escalation procedure say about them?",
    ]),
]

QA_STARTERS = [
    ("ESCALATION", "What is the escalation procedure for P1 incidents?"),
    ("BACKUP", "What does the backup SOP say to do after a backup failure?"),
    ("NETWORK", "How do I troubleshoot VPN connection problems?"),
    ("DATABASE", "What is the documented procedure for database connection failures?"),
]

_VIEWER_FOOT = "If a procedure isn't in the knowledge base, the assistant will say so rather than guess."
_ANALYST_FOOT = "Answers use only your analyzed data and uploaded documents."


def _qa_mode(scope) -> str:
    """\"viewer\" layout = knowledge-base-only Q&A; everything else gets the analyst layout."""
    return "viewer" if scope == "documents" else "analyst"


def _group_visible(needs: str, scope) -> bool:
    if scope == "all":
        return True
    return needs == scope            # incident-only people see data prompts; documents-only see KB prompts


def _qa_head_html(scope) -> str:
    """Page heading. Carries a marker element the stylesheet uses to switch between the two layouts."""
    if _qa_mode(scope) == "viewer":
        crumb, title, sub = "Workspace / Q&amp;A", "Q&amp;A", ""
    else:
        crumb, title = "Knowledge / Q&amp;A", "Q&amp;A (Agent)"
        sub = ("Ask about your incidents and your documented procedures." if scope != "incident"
               else "Ask about your analyzed incident data.")
    return (f'<div class="qa-head qa-mode-{_qa_mode(scope)}"><div class="qa-crumb">{crumb}</div>'
            f'<h2>{title}</h2>' + (f"<p>{sub}</p>" if sub else "") + "</div>")


def _qa_hero_html() -> str:
    return ('<div class="qa-hero"><div class="qa-hero-ic">' + _svg("chat", 24) + "</div>"
            "<h3>Ask your knowledge base</h3>"
            "<p>Answers come only from uploaded runbooks, SOPs and guides, with the source cited.</p></div>")


def _qa_foot_html(scope) -> str:
    right = _VIEWER_FOOT if _qa_mode(scope) == "viewer" else _ANALYST_FOOT
    if _qa_mode(scope) == "viewer":
        return f'<div class="qa-foot qa-foot-center"><span>{_html.escape(right)}</span></div>'
    return ('<div class="qa-foot"><span><span class="qa-kbd">Enter</span> send \u00b7 <span class="qa-kbd">Shift</span>+'
            f'<span class="qa-kbd">Enter</span> new line</span><span>{_html.escape(right)}</span></div>')


def _kb_ready_count() -> int:
    try:
        return sum(1 for d in knowledge_base.list_documents() if str(d.get("status", "")).lower() == "indexed")
    except Exception:
        return 0


def _qa_status_html(full_df, source_label, scope) -> str:
    """The line above the transcript: what the agent is currently answering from."""
    if _qa_mode(scope) == "viewer":
        return ""
    has_data = full_df is not None and len(full_df) > 0
    parts = []
    if has_data:
        label = _html.escape(str(source_label or "analyzed batch"))
        parts.append(f'Using <b class="qa-mono">{label}</b> \u00b7 {len(full_df):,} rows')
    else:
        parts.append("No incident data analyzed yet")
    if scope != "incident":
        n = _kb_ready_count()
        parts.append(f'{n} knowledge-base document{"" if n == 1 else "s"}')
    return f'<span class="qa-dot{"" if has_data else " off"}"></span><span>' + " \u00b7 ".join(parts) + "</span>"


def _qa_refresh_status(full_df, source_label, request: gr.Request = None):
    from ui.access import chat_scope
    return _qa_status_html(full_df, source_label, chat_scope(request))


def _qa_sync(history, mode):
    """Viewer: empty-state (hero + starter cards) while there is no conversation, transcript afterwards.
    Analyst: the chat panel is always visible.  (Panel/library start hidden and are REVEALED here and in
    _qa_setup - Gradio applies hidden->shown updates at page load reliably, shown->hidden ones it does not.)"""
    empty = not history
    viewer = mode == "viewer"
    return gr.update(visible=viewer and empty), gr.update(visible=(not viewer) or (not empty))


def _qa_filter_prompts(query, scope):
    """Prompt-library search: shows only prompts containing every typed word (hides empty groups)."""
    words = [w for w in (query or "").lower().split() if w]
    updates = []
    for _heading, _icon, needs, prompts in QA_PROMPT_LIBRARY:
        allowed = _group_visible(needs, scope)
        shown = [(not words or all(w in p.lower() for w in words)) for p in prompts]
        updates.append(gr.update(visible=allowed and any(shown)))
        updates.extend(gr.update(visible=allowed and ok) for ok in shown)
    return updates


class QATab:
    """Handles to the components build_qa_tab() creates (wired up by wire_qa_tab)."""


def build_qa_tab() -> QATab:
    qa = QATab()
    qa.library_groups, qa.library_buttons, qa.starters = [], [], []
    with gr.Column(elem_classes=["qa"]):
        qa.head = gr.HTML(_qa_head_html("all"), elem_id="qa-head")
        with gr.Row(elem_id="qa-body", equal_height=False):
            # ---- left: the conversation -------------------------------------------------
            with gr.Column(scale=1, min_width=420, elem_id="qa-main"):
                with gr.Column(elem_id="qa-empty", visible=False) as qa.empty:
                    gr.HTML(_qa_hero_html())
                    with gr.Row(elem_id="qa-cards"):
                        for tag, text in QA_STARTERS:
                            qa.starters.append((gr.Button(f"{tag}\n{text}", elem_classes=["qa-card"]), text))
                with gr.Column(elem_id="qa-panel", elem_classes=["dash-card"], visible=False) as qa.panel:
                    with gr.Row(elem_id="qa-status-row"):
                        qa.status = gr.HTML("", elem_id="qa-status")
                        chat_clear_btn = gr.Button("Clear", size="sm", elem_id="qa-clear-btn", icon=None)
                    chatbot = gr.Chatbot(
                        height=520, show_label=False, elem_id="qa-chatbot", layout="bubble", buttons=["copy"],
                        placeholder="Ask a question below, or pick one from the prompt library.",
                    )
                with gr.Column(elem_id="qa-composer-wrap"):
                    with gr.Row(elem_id="qa-composer"):
                        chat_input = gr.Textbox(
                            placeholder="Ask about your incidents or documented procedures...",
                            show_label=False, scale=5, container=False, lines=1, max_lines=6, elem_id="qa-input",
                        )
                        chat_send = gr.Button("Send", variant="primary", scale=1, elem_id="qa-send")
                    qa.foot = gr.HTML(_qa_foot_html("all"), elem_id="qa-foot")
            # ---- right: prompt library (analyst / admin only) ---------------------------------
            with gr.Column(scale=0, min_width=330, elem_id="qa-library", elem_classes=["dash-card"], visible=False) as qa.library:
                gr.HTML('<div class="qa-lib-title"><b>Prompt library</b><span class="qa-kbd">/</span></div>')
                qa.search = gr.Textbox(placeholder="Search prompts", show_label=False, container=False, elem_id="qa-search")
                for heading, icon, _needs, prompts in QA_PROMPT_LIBRARY:
                    with gr.Column(elem_classes=["qa-group"], visible=False) as group:
                        gr.HTML(f'<div class="qa-cat">{_svg(icon)}{_html.escape(heading)}</div>')
                        for text in prompts:
                            qa.library_buttons.append((gr.Button(text, elem_classes=["qa-prompt"]), text))
                    qa.library_groups.append(group)
    qa.chatbot, qa.chat_input, qa.chat_send, qa.chat_clear_btn = chatbot, chat_input, chat_send, chat_clear_btn
    return qa


def wire_qa_tab(qa: QATab, tab_qa, demo, chat_history_state, full_results_state, chat_stats_state, source_label_state):
    """Events for the Q&A page. Every send path calls _chat_respond with the same inputs/outputs as before."""
    mode_state = gr.State("analyst")
    scope_state = gr.State("all")
    respond_in = [qa.chat_input, chat_history_state, full_results_state, chat_stats_state]
    respond_out = [qa.chatbot, chat_history_state, qa.chat_input]
    sync_out = [qa.empty, qa.panel]

    def _send(trigger):
        return trigger(fn=_chat_respond, inputs=respond_in, outputs=respond_out).then(
            fn=_qa_sync, inputs=[chat_history_state, mode_state], outputs=sync_out)

    _send(qa.chat_send.click)
    _send(qa.chat_input.submit)

    def _ask(text):
        def _fill():
            return text
        return _fill

    for btn, text in qa.starters + qa.library_buttons:
        _send(btn.click(fn=_ask(text), inputs=None, outputs=[qa.chat_input]).then)

    qa.chat_clear_btn.click(fn=_chat_clear, outputs=[qa.chatbot, chat_history_state]).then(
        fn=_qa_sync, inputs=[chat_history_state, mode_state], outputs=sync_out)

    # Prompt-library search (pure display filter).
    flat_buttons = [b for b, _t in qa.library_buttons]
    outs, k = [], 0
    for group, (_h, _i, _n, prompts) in zip(qa.library_groups, QA_PROMPT_LIBRARY):
        outs.append(group)
        outs.extend(flat_buttons[k:k + len(prompts)])
        k += len(prompts)
    qa.search.change(fn=_qa_filter_prompts, inputs=[qa.search, scope_state], outputs=outs,
                     trigger_mode="always_last", show_progress="hidden", queue=False)

    # Status line: refreshed whenever the tab is opened (and on page load).
    qa_status_in = [full_results_state, source_label_state]
    tab_qa.select(fn=_qa_refresh_status, inputs=qa_status_in, outputs=[qa.status])

    def _qa_setup(history, request: gr.Request):
        from ui.access import chat_scope
        scope = chat_scope(request)
        mode = _qa_mode(scope)
        library_updates = _qa_filter_prompts("", scope)
        empty_u, panel_u = _qa_sync(history, mode)
        is_analyst = mode == "analyst"
        placeholder = {"viewer": "Ask a question about your documents...",
                       "incident": "Ask a question about the analyzed tickets..."}.get(
            "viewer" if mode == "viewer" else scope, "Ask about your incidents or documented procedures...")
        return [mode, scope, _qa_head_html(scope), _qa_foot_html(scope), gr.update(visible=is_analyst),
                empty_u, panel_u, gr.update(placeholder=placeholder), *library_updates]

    demo.load(
        fn=_qa_setup, inputs=[chat_history_state],
        outputs=[mode_state, scope_state, qa.head, qa.foot, qa.library, qa.empty, qa.panel, qa.chat_input, *outs],
    ).then(fn=_qa_refresh_status, inputs=qa_status_in, outputs=[qa.status])

    # "/" focuses the prompt search (when not already typing somewhere).
    demo.load(fn=None, inputs=None, outputs=None, js=(
        "() => { if (window.__qaSlash) return; window.__qaSlash = true;"
        " document.addEventListener('keydown', function (e) {"
        "  var t = e.target, typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);"
        "  if (e.key === '/' && !typing) { var el = document.querySelector('#qa-search textarea, #qa-search input');"
        "   if (el && el.offsetParent) { e.preventDefault(); el.focus(); } } }); }"))
