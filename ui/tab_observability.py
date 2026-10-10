"""Observability tab (Step 15): live operational metrics for the running app.

These are NOT evaluation scores (those come from `python evaluate_agent.py` and
measure answer quality). Everything here is counted from real events since the
application started, and resets on restart.

Privacy: only aggregate numbers are always visible. The recent-events table
(request ids, pseudonymous session ids, timings, statuses - never question
text, documents or incident data) needs the OBS_ADMIN_TOKEN.
"""
import gradio as gr
import pandas as pd

from app.services import observability

_EVENT_COLUMNS = ["Time", "Request ID", "Session", "Component", "Operation", "Duration (ms)", "Status", "Error"]


def _fmt(v, suffix=""):
    return "n/a" if v is None else f"{v}{suffix}"


def _obs_refresh(admin_token: str = ""):
    try:
        m = observability.get_metrics()
        r, t, rag, llm, tok = m["requests"], m["tools"], m["rag"], m["llm"], m["tokens"]
        summary = (
            f"**Scope:** {m['scope']} - started {m['since']}\n\n"
            f"| Requests | Successful | Degraded | Failed | Client errors | Success rate | Avg time | p50 | p95 |\n|---|---|---|---|---|---|---|---|---|\n"
            f"| {r['total']} | {r['success']} | {r['degraded']} | {r['failed']} | {r['client_error']} | "
            f"{_fmt(r['success_rate_pct'], '%')} | {_fmt(r['avg_ms'], ' ms')} | {_fmt(r['p50_ms'], ' ms')} | {_fmt(r['p95_ms'], ' ms')} |\n\n"
            f"*Degraded = the user got an answer but a tool/LLM step failed. p50/p95 appear after "
            f"{observability.MIN_SAMPLES_FOR_PERCENTILES} observations.*"
        )
        tool_rows = [{"Tool": n, "Calls": s["count"], "Failures": s["failures"], "Avg (ms)": s["avg_ms"], "p95 (ms)": s["p95_ms"]}
                     for n, s in t["by_tool"].items()]
        tools_df = pd.DataFrame(tool_rows, columns=["Tool", "Calls", "Failures", "Avg (ms)", "p95 (ms)"])
        perf_df = pd.DataFrame([
            {"Component": "RAG retrieval", "Calls": rag["count"], "Failures": rag["failures"], "Avg (ms)": rag["avg_ms"],
             "p95 (ms)": rag["p95_ms"], "Notes": f"avg chunks {_fmt(rag['avg_chunks'])}, empty results {rag['empty_results']}"},
            {"Component": "LLM calls", "Calls": llm["count"], "Failures": llm["failures"], "Avg (ms)": llm["avg_ms"],
             "p95 (ms)": llm["p95_ms"], "Notes": f"retries {llm['retries']}"},
            {"Component": "Tokens", "Calls": tok["calls_with_usage"], "Failures": None, "Avg (ms)": None, "p95 (ms)": None,
             "Notes": f"in {tok['input']} / out {tok['output']} / total {tok['total']}; "
                      f"{tok['calls_without_usage']} LLM call(s) reported no usage"},
        ])
        if observability.is_admin_token(admin_token):
            ev = observability.get_recent_events(100)
            events_df = pd.DataFrame([[e["timestamp"], e["request_id"], e["session_id"], e["component"], e["operation"],
                                       e["duration_ms"], e["status"], e.get("error_type") or ""] for e in ev], columns=_EVENT_COLUMNS)
            note = "Recent events unlocked."
        else:
            events_df = pd.DataFrame(columns=_EVENT_COLUMNS)
            note = ("Recent events are hidden. Enter the admin token (OBS_ADMIN_TOKEN in the server's .env) to view them."
                    if observability._cfg("OBS_ADMIN_TOKEN", "") else
                    "Recent events are disabled: set OBS_ADMIN_TOKEN in .env to enable this table.")
        return summary, tools_df, perf_df, events_df, note
    except Exception:
        empty = pd.DataFrame()
        return "Observability data is temporarily unavailable.", empty, empty, pd.DataFrame(columns=_EVENT_COLUMNS), ""


def build_observability_tab():
    """Builds the tab contents (call inside a gr.Tab). Returns nothing - wiring is local."""
    with gr.Column(elem_classes=["dash-card"]):
        gr.Markdown("### 🔭 Observability", elem_classes=["section-heading"])
        gr.Markdown("Live operational metrics since the app started. For answer-quality scores run `python evaluate_agent.py` - "
                    "those are a separate thing.", elem_classes=["severity-note"])
        with gr.Row():
            token = gr.Textbox(label="Admin token (for recent events)", type="password", scale=3)
            refresh = gr.Button("🔄 Refresh", scale=1)
        summary = gr.Markdown("")
        gr.Markdown("**Tool usage**")
        tools_df = gr.Dataframe(interactive=False)
        gr.Markdown("**RAG, LLM and token usage**")
        perf_df = gr.Dataframe(interactive=False)
        gr.Markdown("**Recent events**")
        note = gr.Markdown("")
        events_df = gr.Dataframe(interactive=False)
    outs = [summary, tools_df, perf_df, events_df, note]
    refresh.click(fn=_obs_refresh, inputs=[token], outputs=outs)
    token.submit(fn=_obs_refresh, inputs=[token], outputs=outs)
