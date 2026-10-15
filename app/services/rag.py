"""
Chat/Q&A over the ticket set that was just analyzed - a tool-calling
agent, not a single "stuff everything into the prompt" LLM call.

    User -> Agent -> selects a tool -> pandas (chat_tools.py) -> Agent -> Answer

Six domain-specific tools (see chat_tools.py for the actual pandas logic -
this module only wraps them for the LLM and runs the tool-calling loop):
  - get_incident_summary    : totals, P1/P2 counts, avg resolution time,
                               avg worklog score, % poor worklog
  - get_category_analysis   : incident count by category, top categories
  - get_priority_analysis   : incident count by priority, distribution %
  - get_server_analysis     : incident count by server, top servers
  - get_recurring_issues    : wraps recurring_issues.detect_exact_recurrence
  - search_incidents        : filter by incident ID / server / category /
                               priority / keyword, capped sample + true count

Why tools instead of one prompt with everything stuffed in: each of these
needs the WHOLE batch inspected accurately (a true count, a true average),
not an LLM guess from a sample - the same reasoning that led to
filter_tickets in the previous version of this file. The upgrade here is
scoping that idea to the actual domain questions a management user asks
(summary / category / priority / server / recurring / search) instead of
one generic filter, and routing each through the analytics modules that
already back the dashboard (trend_metrics.py, recurring_issues.py) so the
agent's answer and the dashboard can never disagree.

Bound to the chat model via LangChain's bind_tools()/.ainvoke() (same
async tool-calling pattern already used elsewhere in this codebase). The
model decides which tool(s) to call from each tool's docstring, and may
call more than one per question (MAX_TOOL_ROUNDS below).

IMPORTANT DATA-FLOW RULE: the full analysis dataframe is NEVER sent to the
LLM. Tools take the dataframe as a Python argument (server-side only) and
return a small JSON-serializable dict - see chat_tools.py's module
docstring for how that's enforced. The LLM only ever sees tool results and
the running conversation.

Degradation, consistent with the rest of the app: no chat model configured
(LLM_PROVIDER=rule_based) -> no tool-calling loop is possible, so
answer_question() falls back to a best-effort keyword-based dispatch
across the same 8 tools (see _no_llm_fallback) rather than an LLM. Either
way, a tool or the whole turn failing never raises - see the try/except in
every chat_tools function and the outer try/except here.

TicketIndex below is UNCHANGED from the previous version of this file -
it backs recurring_issues.py's semantic-clustering ("Detect Similar
Recurring Issues" button), which is a separate feature from chat and must
keep working exactly as before.
"""
import json
import logging
import re

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.config import get_settings
from app.security import sanitize_for_llm
from app.services import chat_tools, guardrails, knowledge_base, recommendations
from app.services.llm_client import (
    SYSTEM_GUARDRAIL,
    embeddings_retry,
    get_embeddings_model,
    get_raw_chat_model,
    with_chat_retry,
)

from app.services import observability
from app.services.observability import (  # Step 15 tracing
    note_question, note_tool_selection, record_event, request_trace, set_mode, traced_llm_ainvoke,
)

logger = logging.getLogger(__name__)
settings = get_settings()

TOP_K = 8
MAX_HISTORY_TURNS = 4
MAX_TOOL_ROUNDS = 4

AGENT_SYSTEM_PROMPT = (
    SYSTEM_GUARDRAIL + "\n\n"
    + guardrails.SECURITY_RULES_PROMPT + "\n"
    "You are an agent answering a manager's question using TWO separate sources of "
    "knowledge, and you must pick the right one (or both):\n\n"
    "SOURCE 1 - INCIDENT DATA: a batch of IT incident tickets that has already been "
    "analyzed and categorized. You do not have this data memorized - you MUST call one "
    "or more of the incident-data tools below to get real numbers. Use this for "
    "questions about what happened in the ticket data (\"how many\", \"which server\", "
    "\"what's recurring\", \"what should we do\").\n\n"
    "SOURCE 2 - KNOWLEDGE BASE: uploaded organizational documents (runbooks, SOPs, "
    "troubleshooting guides, known-error documents, escalation procedures) that are NOT "
    "part of the incident data. Use search_knowledge_base for questions about documented "
    "procedures, what a runbook recommends, or escalation/troubleshooting steps. If a "
    "question needs both - e.g. \"we have recurring database incidents, what does our "
    "runbook recommend?\" - call an incident-data tool AND search_knowledge_base, then "
    "combine both results in your answer.\n\n"
    "Incident-data tools:\n"
    "- get_incident_summary: total incidents, P1/P2 counts, average resolution time, average "
    "worklog score, percentage of tickets with a poor worklog. Use for \"give me an overall "
    "summary\" or general-health questions.\n"
    "- get_category_analysis: incident count by category, top categories. Use for \"what "
    "categories\" / \"most common issue types\" questions.\n"
    "- get_priority_analysis: incident count by priority, percentage distribution. Use for "
    "\"how many P1/P2\" or priority-breakdown questions.\n"
    "- get_server_analysis: incident count by server/host, top affected servers. Use for "
    "\"which server has the most incidents\" questions.\n"
    "- get_recurring_issues: recurring issues (same host+category repeating), their "
    "frequency, and affected servers. Use for \"what's recurring\" / \"what keeps happening\" "
    "questions.\n"
    "- get_recommendations: actionable, data-backed recommendations for this batch - recurring "
    "issues needing root-cause work, priority/SLA problems, slow-resolving categories, servers "
    "and assignment groups, worklog-quality gaps, and hosts or categories carrying "
    "disproportionate volume. Each item comes with its own evidence, attention level, and "
    "expected benefit. Use for \"what should we do\" / \"what needs attention\" / \"how do we "
    "improve\" questions, and quote its numbers as-is rather than recomputing them.\n"
    "- search_incidents: filter tickets by any combination of incident_id, server, category, "
    "priority, and/or a free-text keyword. Returns the true total match count plus a capped "
    "sample. Use for specific lookups like \"show me P1 incidents on server X\".\n\n"
    "Knowledge Base tool:\n"
    "- search_knowledge_base: semantic search over uploaded documents. Returns the most "
    "relevant text passages with their source document, and page number or section title "
    "where available. GROUNDING RULE: only state a procedure or documentation content that "
    "actually appears in this tool's returned text - never invent, assume, or complete a "
    "procedure from general knowledge. If it returns no chunks (empty list) or says nothing "
    "relevant was found, tell the user plainly that you couldn't find a relevant procedure "
    "in the current knowledge base - do not guess. When you do use a retrieved chunk, name "
    "its source document (and page/section if given) so the user knows where the answer "
    "came from.\n\n"
    "Call more than one tool if the question needs it (e.g. a question about both priority "
    "and category, or a question needing both incident data and documentation). Never state "
    "a number, fact, or procedure you did not get from a tool call - if a tool returns an "
    "error or says information is unavailable, say plainly that it's unavailable rather than "
    "guessing or estimating. Be concise and management-friendly: lead with the direct answer, "
    "then one or two sentences of support. Cite specific incident IDs in parentheses when a "
    "tool result includes them, and cite the source document for anything from the knowledge base."
)


class TicketIndex:
    """In-memory semantic index over one analysis batch's tickets.

    UNCHANGED from the previous chat implementation - this class no
    longer backs chat (see _make_tools/answer_question below, which now
    operate on the dataframe directly via chat_tools.py), but it still
    backs recurring_issues.py's semantic-clustering feature
    (cluster_similar, used by the "Detect Similar Recurring Issues"
    button in ui/gradio_app.py) via build_index_from_dataframe below.
    Do not remove without checking that call site."""

    def __init__(self, rows: list[dict]):
        # each row: ticket_id, category, priority, status, host,
        # assignment_group, worklog_score, text
        self.rows = rows
        self._vectors: np.ndarray | None = None
        self._vectors_attempted = False

    async def _ensure_vectors(self) -> None:
        if self._vectors_attempted or not self.rows:
            return
        self._vectors_attempted = True
        embeddings_model = get_embeddings_model()
        if embeddings_model is None:
            return
        try:
            embed_fn = embeddings_retry(embeddings_model.aembed_documents)
            vectors = await embed_fn([r["text"] for r in self.rows])
            self._vectors = np.array(vectors)
        except Exception as exc:
            logger.info("Embedding index unavailable, falling back to keyword clustering: %s", exc)

    async def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        if not self.rows:
            return []

        await self._ensure_vectors()
        if self._vectors is not None:
            embeddings_model = get_embeddings_model()
            try:
                embed_fn = embeddings_retry(embeddings_model.aembed_query)
                qvec = np.array(await embed_fn(query))
                norms = np.linalg.norm(self._vectors, axis=1) * (np.linalg.norm(qvec) or 1e-9)
                sims = (self._vectors @ qvec) / (norms + 1e-9)
                top_idx = np.argsort(-sims)[:top_k]
                return [self.rows[i] for i in top_idx]
            except Exception as exc:
                logger.info("Similarity search failed, falling back to keyword search: %s", exc)

        terms = [t for t in query.lower().split() if len(t) > 2]
        scored = [(sum(1 for t in terms if t in row["text"].lower()), row) for row in self.rows]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits = [row for count, row in scored if count > 0]
        return hits[:top_k] if hits else self.rows[:top_k]

    async def cluster_similar(self, similarity_cutoff: float = 0.85, min_cluster_size: int = 3) -> list[list[int]]:
        """Groups tickets into clusters of similar text via connected
        components over a cosine-similarity graph (union-find on pairs
        above similarity_cutoff). Returns only clusters with at least
        min_cluster_size members, each as a list of row indices into
        self.rows, largest cluster first."""
        n = len(self.rows)
        if n == 0:
            return []

        await self._ensure_vectors()
        if self._vectors is None:
            return self._keyword_cluster(min_cluster_size)

        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        unit = self._vectors / norms
        similarity = unit @ unit.T

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if similarity[i, j] >= similarity_cutoff:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        clusters = [members for members in groups.values() if len(members) >= min_cluster_size]
        clusters.sort(key=len, reverse=True)
        return clusters

    def _keyword_cluster(self, min_cluster_size: int) -> list[list[int]]:
        """No-embeddings fallback: groups tickets whose text normalizes to
        the exact same value. Reduced recall, still functional."""
        buckets: dict[str, list[int]] = {}
        for i, row in enumerate(self.rows):
            key = " ".join(row["text"].lower().split())
            if key:
                buckets.setdefault(key, []).append(i)
        clusters = [members for members in buckets.values() if len(members) >= min_cluster_size]
        clusters.sort(key=len, reverse=True)
        return clusters


def _row_from_fields(ticket_id, category, priority, status, host, assignment_group, worklog_score, *text_parts) -> dict:
    text = " ".join(str(p) for p in text_parts if p).strip() or str(category or "")
    return {
        "ticket_id": ticket_id or "",
        "category": category or "",
        "priority": priority or "",
        "status": status or "",
        "host": host or "",
        "assignment_group": assignment_group or "",
        "worklog_score": worklog_score if isinstance(worklog_score, (int, float)) else None,
        "text": text,
    }


def build_index_from_dataframe(full_df) -> "TicketIndex":
    """Builds a semantic index from the results dataframe - used only by
    the "Detect Similar Recurring Issues" button (ui/gradio_app.py), not
    by chat anymore (see _make_tools/answer_question below)."""
    if full_df is None or len(full_df) == 0:
        return TicketIndex([])
    rows = [
        _row_from_fields(
            r.get("Ticket ID"), r.get("Category"), r.get("Priority"), r.get("Status"),
            r.get("Host / CI"), r.get("Assignment Group"), r.get("Worklog Score"),
            r.get("Short Description"), r.get("Description"), r.get("Worklog Notes"),
        )
        for _, r in full_df.iterrows()
    ]
    return TicketIndex(rows)


def _make_tools(full_df, log: "guardrails.ToolLog | None" = None) -> list:
    """Wraps the 8 chat_tools functions as LangChain tools bound to this
    analysis batch via closure. Every call goes through `_run` below, which:
      * validates the arguments the MODEL chose (guardrails.clean_*_arg),
      * runs the read-only function inside try/except (a failure becomes a
        GENERIC error result - exception text is logged, never sent to the
        model or the user),
      * wraps the result in an "untrusted data" envelope after scrubbing
        injection phrases/markers from every string (guardrails.wrap_untrusted),
      * records what was returned in `log`, so the final answer can be checked
        against it (numbers and knowledge-base sources) in plain Python.
    The tool set is fixed (guardrails.ALLOWED_TOOLS): no code/shell/infra tools."""
    log = log if log is not None else guardrails.ToolLog()

    def _emit(name: str, result) -> str:
        envelope, stats = guardrails.wrap_untrusted(name, result)
        log.names.append(name)
        log.texts.append(json.dumps(result, default=str))
        log.objects.append(result)
        log.injection_flagged += stats.flagged_fields
        if stats.flagged_fields:
            # field NAMES and counts only - never the content itself
            record_event("guardrail", "injection_in_tool_result", "degraded", 0.0,
                         tool=name, flagged_fields=stats.flagged_fields, fields=",".join(stats.examples))
        return json.dumps(envelope, default=str)

    def _fail(name: str, exc: Exception) -> str:
        if isinstance(exc, guardrails.ToolArgumentError):
            return _emit(name, {"error": str(exc)})          # message is written to be safe
        logger.warning("Tool %s failed: %s: %s", name, type(exc).__name__, exc)
        return _emit(name, {"error": "This tool failed (internal error). Tell the user the data is unavailable."})

    def _run(name: str, fn, *args, **kwargs) -> str:
        try:
            guardrails.assert_tool_allowed(name)
            return _emit(name, fn(*args, **kwargs))
        except Exception as exc:
            return _fail(name, exc)

    @tool
    def get_incident_summary() -> str:
        """Returns overall incident health: total incidents, P1 count, P2 count, average
        resolution time (hours), average worklog score, and the percentage of tickets with a
        poor worklog. Use this for "give me an overall summary" or general-health questions."""
        return _run("get_incident_summary", chat_tools.get_incident_summary, full_df)

    @tool
    def get_category_analysis() -> str:
        """Returns incident count broken down by category, plus the top categories by volume.
        Use this for "what categories" or "most common issue types" questions."""
        return _run("get_category_analysis", chat_tools.get_category_analysis, full_df)

    @tool
    def get_priority_analysis() -> str:
        """Returns incident count broken down by priority (P1/P2/P3/P4/Unspecified) and the
        percentage distribution across priorities. Use this for "how many P1" or
        priority-breakdown questions."""
        return _run("get_priority_analysis", chat_tools.get_priority_analysis, full_df)

    @tool
    def get_server_analysis() -> str:
        """Returns incident count broken down by server/host, plus the top affected servers.
        Use this for "which server has the most incidents" questions."""
        return _run("get_server_analysis", chat_tools.get_server_analysis, full_df)

    @tool
    def get_recurring_issues(threshold: int = 3) -> str:
        """Returns recurring issues - the same host+category combination repeating at least
        `threshold` times (default 3) - with their frequency, affected servers, and how many
        days apart they typically recur. Use this for "what's recurring" or "what keeps
        happening" questions."""
        try:
            n = guardrails.clean_int_arg("threshold", threshold, 2, 100)
        except Exception as exc:
            return _fail("get_recurring_issues", exc)
        return _run("get_recurring_issues", chat_tools.get_recurring_issues, full_df, threshold=n)

    @tool
    def get_recommendations() -> str:
        """Returns actionable, data-backed ITSM recommendations derived from this batch:
        recurring issues needing root-cause work, high-priority/SLA problems, slow-resolving
        categories/servers/assignment groups, worklog-quality gaps, and hosts or categories
        carrying disproportionate volume. Each item has an observation, the evidence behind it,
        the recommended action, an attention level, and the expected benefit. Use this for
        "what should we do", "what needs attention", or "how do we improve" questions - the
        numbers in it are already calculated, so quote them as-is rather than recomputing."""
        return _run("get_recommendations", recommendations.get_recommendations, full_df)

    @tool
    async def search_knowledge_base(query: str) -> str:
        """Searches uploaded organizational documents (runbooks, SOPs, troubleshooting guides,
        known-error documents, escalation procedures) - a SEPARATE source from the incident
        ticket data above. Use this for questions about documented procedures, what a runbook
        recommends, or escalation/troubleshooting steps. Returns the most relevant text
        passages with their source document name and page number or section title where
        available. If it returns an empty chunk list, say plainly that no relevant procedure
        was found in the current knowledge base - never invent or assume a procedure."""
        name = "search_knowledge_base"
        log.kb_searched = True
        try:
            guardrails.assert_tool_allowed(name)
            q = guardrails.clean_text_arg("query", query, max_len=500, required=True)
            chunks = await knowledge_base.retrieve_relevant_chunks(q)
            for c in chunks:
                log.kb_docs.append((c.get("document_name"), c.get("page_number"), c.get("section_title")))
            return _emit(name, {
                "chunks": chunks,
                "count": len(chunks),
                "note": "" if chunks else "No relevant information was found in the knowledge base for this query.",
            })
        except Exception as exc:
            return _fail(name, exc)

    @tool
    def search_incidents(
        incident_id: str = "", server: str = "", category: str = "",
        priority: str = "", keyword: str = "",
    ) -> str:
        """Searches/filters incidents by any combination of incident_id, server, category,
        priority, and/or a free-text keyword (matched against description and worklog notes).
        Leave a field empty to not filter on it. Returns the true total match count plus a
        capped sample of matching tickets. Use this for specific lookups like "show me P1
        incidents on server ABC" or "find incidents mentioning VPN"."""
        try:
            args = {k: guardrails.clean_text_arg(k, v, max_len=200)
                    for k, v in dict(incident_id=incident_id, server=server, category=category,
                                     priority=priority, keyword=keyword).items()}
        except Exception as exc:
            return _fail("search_incidents", exc)
        return _run("search_incidents", chat_tools.search_incidents, full_df, **args)

    return [
        get_incident_summary, get_category_analysis, get_priority_analysis,
        get_server_analysis, get_recurring_issues, get_recommendations,
        search_knowledge_base, search_incidents,
    ]


# Maps a keyword found in the question to the tool that answers it, used
# only by _no_llm_fallback below (rule_based mode - no tool-calling model
# available). Order matters: first match wins, so more specific intents
# (recurring, server, priority, category) are checked before the generic
# "summary" catch-all.
_UNSUPPORTED_DATA_TERMS = (
    "cpu", "memory", "ram", "temperature", "disk usage", "bandwidth", "latency", "uptime",
    "cost", "costs", "revenue", "salary", "budget", "headcount", "customer satisfaction", "csat",
)
_HOST_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*[-_][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b")

_NO_LLM_INTENT_KEYWORDS = [
    # Checked first: a documentation/procedure question should hit the
    # knowledge base even if it also mentions a word like "recommend" or
    # "priority" in passing (e.g. "what does the runbook recommend for
    # priority incidents" is still a KB lookup, not a priority-analysis one).
    (("runbook", "documented procedure", "documentation", "sop", "knowledge base",
      "troubleshooting guide", "troubleshooting steps", "troubleshoot", "known error", "escalation procedure",
      "escalation steps", "what does the procedure", "how do we restart",
      "on-call", "phone number", "contact number"), "kb"),
    (("recommend", "recommendation", "what should we do", "suggest", "improve", "action plan", "needs attention"), "recommend"),
    (("recurring", "repeat", "keeps happening", "again and again"), "recurring"),
    (("server", "host", "affected server", "which server"), "server"),
    (("p1", "p2", "priority", "critical", "high priority"), "priority"),
    (("category", "categories", "issue type", "issue types"), "category"),
    (("summary", "overall", "overview", "health"), "summary"),
]


def _format_no_llm_result(intent: str, result: dict) -> str:
    """Turns one of the tool-result dicts above into plain, readable text.

    The real LLM path already produces natural language itself (that's
    what the model is for) - this formatter exists only for
    _no_llm_fallback below, where there is no model to do that, so the
    person still gets a readable answer instead of a raw JSON dump. Each
    branch mirrors one tool's exact return shape (see chat_tools.py /
    recommendations.py / knowledge_base.py); a shape not explicitly
    handled falls through to a generic key: value listing rather than
    erroring."""
    if not isinstance(result, dict):
        return str(result)
    if result.get("error"):
        return f"I couldn't get that: {result['error']}"

    try:
        if intent == "category":
            top = result.get("top_categories", [])
            if not top:
                return "No category data is available for this batch."
            lines = [f"Top categories (of {result.get('total_categories', 'N/A')} total):"]
            lines += [f"- {c['category']}: {c['count']} incident(s)" for c in top]
            return "\n".join(lines)

        if intent == "priority":
            dist = result.get("distribution_pct", {})
            counts = result.get("counts_by_priority", {})
            if not counts:
                return "No priority data is available for this batch."
            lines = ["Incidents by priority:"]
            for p, n in counts.items():
                pct = dist.get(p)
                lines.append(f"- {p}: {n} incident(s)" + (f" ({pct}%)" if pct is not None else ""))
            return "\n".join(lines)

        if intent == "server":
            top = result.get("top_servers", [])
            if not top:
                return "No server/host data is available for this batch."
            lines = ["Most-affected servers/hosts:"]
            lines += [f"- {s['server']}: {s['count']} incident(s)" for s in top]
            return "\n".join(lines)

        if intent == "recurring":
            rows = result.get("recurring_issues", [])
            if not rows:
                return result.get("note") or "No recurring issues were found in this batch."
            lines = ["Recurring issues:"]
            for r in rows:
                lines.append(
                    f"- {r['server']} / {r['category']}: {r['frequency']} occurrence(s) "
                    f"(first seen {r.get('first_seen', 'N/A')}, last seen {r.get('last_seen', 'N/A')})"
                )
            return "\n".join(lines)

        if intent == "recommend":
            recs = result.get("recommendations", [])
            if not recs:
                return result.get("note") or "No recommendations were generated for this batch."
            lines = []
            for i, r in enumerate(recs, start=1):
                lines.append(f"{i}. [{r.get('attention', '')}] {r.get('observation', '')}")
                if r.get("evidence"):
                    lines.append(f"   Evidence: {r['evidence']}")
                if r.get("recommendation"):
                    lines.append(f"   Recommendation: {r['recommendation']}")
            return "\n".join(lines)

        if intent == "kb":
            chunks = result.get("chunks", [])
            if not chunks:
                return result.get("note") or "I couldn't find a relevant procedure in the current knowledge base."
            lines = []
            for c in chunks:
                source = c.get("document_name", "Unknown document")
                if c.get("page_number"):
                    source += f", page {c['page_number']}"
                if c.get("section_title"):
                    source += f", section: {c['section_title']}"
                lines.append(f"{c.get('text', '')}\n(Source: {source})")
            body = "\n\n".join(lines)
            if all(c.get("relevance") is None for c in chunks):
                # keyword match only (no embeddings): the passages were matched on words, not
                # understood, so don't let them read as a confirmed answer.
                body = ("Closest keyword matches in the knowledge base. They were matched on words only, so please "
                        "check that they actually answer your question:\n\n" + body)
            return body

        if intent == "search":
            total = result.get("total_matching", 0)
            rows = result.get("results", [])
            if not rows:
                return f"No incidents matched that search (0 of {total})."
            lines = [f"Found {total} matching incident(s), showing {len(rows)}:"]
            for row in rows:
                lines.append("- " + ", ".join(f"{k}: {v}" for k, v in row.items()))
            return "\n".join(lines)

        # "summary" and the default (unmatched-intent) case share this shape.
        lines = [f"Total incidents: {result.get('total_incidents', 'N/A')}"]
        lines.append(f"P1 incidents: {result.get('p1_incidents', 'N/A')}   P2 incidents: {result.get('p2_incidents', 'N/A')}")
        if result.get("average_resolution_time_hours") is not None:
            lines.append(f"Average resolution time: {result['average_resolution_time_hours']} hours")
        elif result.get("average_resolution_time_note"):
            lines.append(f"Average resolution time: {result['average_resolution_time_note']}")
        if result.get("average_worklog_score") is not None:
            lines.append(f"Average worklog score: {result['average_worklog_score']} / 100")
        if result.get("poor_worklog_percentage") is not None:
            lines.append(f"Tickets with a poor worklog: {result['poor_worklog_percentage']}%")
        return "\n".join(lines)
    except Exception:
        # Any shape this formatter doesn't recognize - a plain listing
        # beats either crashing or falling back to raw JSON.
        return "\n".join(f"{k}: {v}" for k, v in result.items())


async def _no_llm_fallback(question: str, full_df) -> str:
    """rule_based mode: no chat model, so no tool-calling loop is
    possible. Dispatches to the same tool functions directly via a
    simple keyword match on the question, so the app stays useful without
    an LLM - just without the natural-language routing/summary a real
    model would add. async only because the "kb" intent awaits
    knowledge_base.retrieve_relevant_chunks (an embedding call) - every
    other intent below is a plain synchronous pandas call."""
    q = question.lower()

    # Guardrail: say plainly when the question is about data this app does not hold.
    # (Checked before anything else so e.g. "CPU temperature" never gets answered
    # with an unrelated ticket summary that looks like an answer.)
    unsupported = [t for t in _UNSUPPORTED_DATA_TERMS if re.search(rf"\b{re.escape(t)}\b", q)]
    if unsupported and not any(kw in q for kw in _NO_LLM_INTENT_KEYWORDS[0][0]):
        return (
            f"No data is available on {', '.join(sorted(unsupported))}. The analyzed incident tickets only contain: ticket ID, "
            "descriptions, worklog notes, priority, status, assignment group, host/CI, category, worklog score and "
            "opened/resolved timestamps - so I can't answer this from them, and I won't guess."
        )

    # A hostname-looking token (e.g. DB-PRD-01) -> exact lookup, so an unknown host
    # gets a true "0 matching incidents" instead of an unrelated server ranking.
    host_token = _HOST_TOKEN_RE.search(question)
    if host_token and not any(kw in q for kw in _NO_LLM_INTENT_KEYWORDS[0][0]):
        return _format_no_llm_result("search", chat_tools.search_incidents(full_df, server=host_token.group(0)))

    for keywords, intent in _NO_LLM_INTENT_KEYWORDS:
        if any(kw in q for kw in keywords):
            if intent == "kb":
                chunks = await knowledge_base.retrieve_relevant_chunks(question)
                result = {
                    "chunks": chunks,
                    "note": "" if chunks else "No relevant information was found in the knowledge base for this query.",
                }
            elif intent == "recommend":
                result = recommendations.get_recommendations(full_df)
            elif intent == "recurring":
                result = chat_tools.get_recurring_issues(full_df)
            elif intent == "server":
                result = chat_tools.get_server_analysis(full_df)
            elif intent == "priority":
                result = chat_tools.get_priority_analysis(full_df)
            elif intent == "category":
                result = chat_tools.get_category_analysis(full_df)
            else:
                result = chat_tools.get_incident_summary(full_df)
            return _format_no_llm_result(intent, result)

    # No intent keyword matched - try search_incidents with any exact
    # category/priority value mentioned verbatim in the question, else
    # fall back to the overall summary as a safe default.
    try:
        categories = set(full_df.get("Category", []).dropna().astype(str)) if full_df is not None else set()
        priorities = set(full_df.get("Priority", []).dropna().astype(str)) if full_df is not None else set()
        matched_category = next((c for c in categories if c and c.lower() in q), "")
        matched_priority = next((p for p in priorities if p and p.lower() in q), "")
        if matched_category or matched_priority:
            result = chat_tools.search_incidents(full_df, category=matched_category, priority=matched_priority)
            return _format_no_llm_result("search", result)
    except Exception as exc:
        logger.info("No-LLM fallback intent detection failed, using summary instead: %s", exc)

    return _format_no_llm_result("summary", chat_tools.get_incident_summary(full_df))


def _history_to_messages(history: list[tuple[str, str]]) -> list:
    """History is supplied by the CLIENT (REST body or browser state), so it is
    untrusted: only well-formed text pairs are kept, length-capped and scrubbed
    (guardrails.sanitize_history) before being replayed to the model."""
    messages = []
    for question, answer in guardrails.sanitize_history(history, MAX_HISTORY_TURNS):
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages


async def _answer_question_impl(
    question: str,
    full_df,
    stats: dict | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Answers a management question by letting the chat model choose
    from 6 domain-specific tools (see _make_tools), each backed by a
    pandas function in chat_tools.py operating on `full_df` - the same
    analysis dataframe the dashboard/CSV export use. Never raises -
    degrades to a keyword-dispatched tool result (no chat model) or a
    plain error message (any other failure) so the chat UI always gets
    something useful back.

    `stats` is currently unused here (kept in the signature for backward
    compatibility with existing callers - ui/gradio_app.py and
    routers/chat.py both still pass it) - the aggregate figures it used
    to carry are now sourced live via get_incident_summary() instead of
    being pre-computed once and pasted into the prompt, so the agent's
    numbers can never go stale relative to what's actually in full_df."""
    question = sanitize_for_llm(question, max_len=1000).strip()
    note_question(question)
    if not question:
        return 'Ask a question about the analyzed tickets - e.g. "how many P1 tickets are there?"'

    if full_df is None or len(full_df) == 0:
        return "No analyzed tickets are available yet - run an analysis first, then come back and ask away."

    raw_chat_model = get_raw_chat_model()
    set_mode("rule_based" if raw_chat_model is None else "llm")
    if raw_chat_model is None:
        try:
            return await _no_llm_fallback(question, full_df)
        except Exception:
            logger.exception("No-LLM fallback failed")
            return "I couldn't compute an answer right now - please try again."

    try:
        tool_log = guardrails.ToolLog()
        tools = _make_tools(full_df, tool_log)
        tool_map = {t.name: t for t in tools}
        # bind_tools() must be called on the RAW model - with_chat_retry()
        # returns a RunnableRetry, which doesn't expose bind_tools() (see
        # get_raw_chat_model()'s docstring in llm_client.py). Bind first,
        # then wrap the bound runnable in retry so 429s are still handled.
        model_with_tools = with_chat_retry(raw_chat_model.bind_tools(tools))

        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
        messages += _history_to_messages(history)
        messages.append(HumanMessage(content=question))

        for _round in range(MAX_TOOL_ROUNDS):
            response = await traced_llm_ainvoke(model_with_tools, messages, "agent_turn")
            messages.append(response)
            if not getattr(response, "tool_calls", None):
                return await _finalize(_content_text(response), question, history, tool_log, full_df)
            note_tool_selection([c.get("name") for c in response.tool_calls], _round + 1)

            for call in response.tool_calls:
                name = call.get("name")
                tool_fn = tool_map.get(name) if isinstance(name, str) and name in guardrails.ALLOWED_TOOLS else None
                if tool_fn is None:
                    # Unknown / not-allow-listed tool name (the model hallucinated it or was
                    # steered to it): never executed.
                    record_event("tool", str(name)[:60], "error", 0.0, error_type="UnknownTool")
                    result = json.dumps({"error": "That tool is not available."})
                else:
                    try:
                        args = call.get("args") if isinstance(call.get("args"), dict) else {}
                        result = await tool_fn.ainvoke(args)
                    except Exception as exc:   # e.g. wrong argument types rejected by the tool schema
                        logger.warning("Tool %s failed: %s: %s", name, type(exc).__name__, exc)
                        result = json.dumps({"error": "Invalid or failed tool call."})
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

        # Exceeded MAX_TOOL_ROUNDS without a final answer - force one
        # without giving the model tools to call again.
        final = await traced_llm_ainvoke(
            with_chat_retry(raw_chat_model),
            messages + [HumanMessage(content="Answer the original question now, using only the information already gathered above.")],
            "forced_final_answer",
        )
        return await _finalize(_content_text(final), question, history, tool_log, full_df)
    except Exception:
        logger.exception("Chat answer generation failed")
        try:
            return "I couldn't reach the assistant model just now. Here's what the data shows directly:\n\n" + await _no_llm_fallback(question, full_df)
        except Exception:
            return "I couldn't generate an answer right now - please try again."


def _content_text(message) -> str:
    """Chat models may return content as a string or a list of content blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "").strip()


async def _finalize(answer: str, question: str, history, tool_log, full_df) -> str:
    """Deterministic checks on the model's final text before the user sees it
    (guardrails.finalize_answer): withholds prompt leaks, flags/replaces figures
    that no tool returned, appends knowledge-base sources from the retrieval
    result itself - then masks any configured secret that slipped into the text."""
    clean_history = guardrails.sanitize_history(history, MAX_HISTORY_TURNS)
    answer, verdict = guardrails.finalize_answer(answer, question, clean_history, tool_log)
    if verdict == "no_tool_figures":
        record_event("guardrail", "ungrounded_answer_replaced", "degraded", 0.0)
        answer = guardrails.NO_TOOL_FIGURES_MESSAGE + "Here is what the data shows directly:\n\n" + await _no_llm_fallback(question, full_df)
    elif verdict != "ok":
        record_event("guardrail", f"answer_{verdict}", "degraded", 0.0)
    return observability.redact(answer)


async def answer_question(
    question: str,
    full_df,
    stats: dict | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Public entry point - unchanged signature and behaviour. Wraps the agent
    in a traced request (Step 15): reuses the REST request context if one is
    active, otherwise starts its own (Gradio path). See observability.py."""
    with request_trace("chat", source="agent"):
        return await _answer_question_impl(question, full_df, stats, history)
