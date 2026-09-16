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
across the same 7 tools (see _no_llm_fallback) rather than an LLM. Either
way, a tool or the whole turn failing never raises - see the try/except in
every chat_tools function and the outer try/except here.

TicketIndex below is UNCHANGED from the previous version of this file -
it backs recurring_issues.py's semantic-clustering ("Detect Similar
Recurring Issues" button), which is a separate feature from chat and must
keep working exactly as before.
"""
import json
import logging

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.security import sanitize_for_llm
from app.services import chat_tools, recommendations
from app.services.llm_client import (
    SYSTEM_GUARDRAIL,
    embeddings_retry,
    get_embeddings_model,
    get_raw_chat_model,
    with_chat_retry,
)

logger = logging.getLogger(__name__)

TOP_K = 8
MAX_HISTORY_TURNS = 4
MAX_TOOL_ROUNDS = 4

AGENT_SYSTEM_PROMPT = (
    SYSTEM_GUARDRAIL + "\n\n"
    "You are an agent answering a manager's question about a batch of IT incident tickets "
    "that has already been analyzed and categorized. You do not have the data memorized - "
    "you MUST call one or more of these tools to get real numbers:\n\n"
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
    "Call more than one tool if the question needs it (e.g. a question about both priority "
    "and category). Never state a number, percentage, or fact you did not get from a tool "
    "call - if a tool returns an error or says data is unavailable, say plainly that the "
    "information is unavailable rather than guessing or estimating. Be concise and "
    "management-friendly: lead with the direct answer, then one or two sentences of support. "
    "Cite specific incident IDs in parentheses when a tool result includes them."
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


def _make_tools(full_df) -> list:
    """Wraps the 6 chat_tools functions as LangChain tools bound to this
    analysis batch via closure. Each tool returns a JSON string (LangChain
    tool outputs must be strings) - the dict chat_tools builds is already
    small and JSON-safe (see chat_tools._to_native), so this is a plain
    json.dumps, no further processing needed.

    Every call is wrapped in try/except even though chat_tools functions
    already catch their own errors - this is a second layer of defense so
    a totally unexpected failure (e.g. a bad argument type from the LLM)
    still comes back as a tool result the agent can react to, instead of
    crashing the whole turn."""

    @tool
    def get_incident_summary() -> str:
        """Returns overall incident health: total incidents, P1 count, P2 count, average
        resolution time (hours), average worklog score, and the percentage of tickets with a
        poor worklog. Use this for "give me an overall summary" or general-health questions."""
        try:
            return json.dumps(chat_tools.get_incident_summary(full_df), default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @tool
    def get_category_analysis() -> str:
        """Returns incident count broken down by category, plus the top categories by volume.
        Use this for "what categories" or "most common issue types" questions."""
        try:
            return json.dumps(chat_tools.get_category_analysis(full_df), default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @tool
    def get_priority_analysis() -> str:
        """Returns incident count broken down by priority (P1/P2/P3/P4/Unspecified) and the
        percentage distribution across priorities. Use this for "how many P1" or
        priority-breakdown questions."""
        try:
            return json.dumps(chat_tools.get_priority_analysis(full_df), default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @tool
    def get_server_analysis() -> str:
        """Returns incident count broken down by server/host, plus the top affected servers.
        Use this for "which server has the most incidents" questions."""
        try:
            return json.dumps(chat_tools.get_server_analysis(full_df), default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @tool
    def get_recurring_issues(threshold: int = 3) -> str:
        """Returns recurring issues - the same host+category combination repeating at least
        `threshold` times (default 3) - with their frequency, affected servers, and how many
        days apart they typically recur. Use this for "what's recurring" or "what keeps
        happening" questions."""
        try:
            return json.dumps(chat_tools.get_recurring_issues(full_df, threshold=threshold), default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @tool
    def get_recommendations() -> str:
        """Returns actionable, data-backed ITSM recommendations derived from this batch:
        recurring issues needing root-cause work, high-priority/SLA problems, slow-resolving
        categories/servers/assignment groups, worklog-quality gaps, and hosts or categories
        carrying disproportionate volume. Each item has an observation, the evidence behind it,
        the recommended action, an attention level, and the expected benefit. Use this for
        "what should we do", "what needs attention", or "how do we improve" questions - the
        numbers in it are already calculated, so quote them as-is rather than recomputing."""
        try:
            return json.dumps(recommendations.get_recommendations(full_df), default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

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
            return json.dumps(
                chat_tools.search_incidents(
                    full_df, incident_id=incident_id, server=server, category=category,
                    priority=priority, keyword=keyword,
                ),
                default=str,
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return [
        get_incident_summary, get_category_analysis, get_priority_analysis,
        get_server_analysis, get_recurring_issues, get_recommendations, search_incidents,
    ]


# Maps a keyword found in the question to the tool that answers it, used
# only by _no_llm_fallback below (rule_based mode - no tool-calling model
# available). Order matters: first match wins, so more specific intents
# (recurring, server, priority, category) are checked before the generic
# "summary" catch-all.
_NO_LLM_INTENT_KEYWORDS = [
    # Checked before "recurring"/"summary": "what do you recommend about
    # recurring issues" is a recommendation request, and the
    # recommendations tool already covers recurrence as one of its areas.
    (("recommend", "recommendation", "what should we do", "suggest", "improve", "action plan", "needs attention"), "recommend"),
    (("recurring", "repeat", "keeps happening", "again and again"), "recurring"),
    (("server", "host", "affected server", "which server"), "server"),
    (("p1", "p2", "priority", "critical", "high priority"), "priority"),
    (("category", "categories", "issue type", "issue types"), "category"),
    (("summary", "overall", "overview", "health"), "summary"),
]


def _no_llm_fallback(question: str, full_df) -> str:
    """rule_based mode: no chat model, so no tool-calling loop is
    possible. Dispatches to the same 6 tool functions directly via a
    simple keyword match on the question, so the app stays useful without
    an LLM - just without the natural-language routing/summary a real
    model would add."""
    q = question.lower()

    for keywords, intent in _NO_LLM_INTENT_KEYWORDS:
        if any(kw in q for kw in keywords):
            if intent == "recommend":
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
            return json.dumps(result, indent=2, default=str)

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
            return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.info("No-LLM fallback intent detection failed, using summary instead: %s", exc)

    return json.dumps(chat_tools.get_incident_summary(full_df), indent=2, default=str)


def _history_to_messages(history: list[tuple[str, str]]) -> list:
    recent = (history or [])[-MAX_HISTORY_TURNS:]
    messages = []
    for question, answer in recent:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages


async def answer_question(
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
    if not question:
        return 'Ask a question about the analyzed tickets - e.g. "how many P1 tickets are there?"'

    if full_df is None or len(full_df) == 0:
        return "No analyzed tickets are available yet - run an analysis first, then come back and ask away."

    raw_chat_model = get_raw_chat_model()
    if raw_chat_model is None:
        try:
            return _no_llm_fallback(question, full_df)
        except Exception:
            logger.exception("No-LLM fallback failed")
            return "I couldn't compute an answer right now - please try again."

    try:
        tools = _make_tools(full_df)
        tool_map = {t.name: t for t in tools}
        # bind_tools() must be called on the RAW model - with_chat_retry()
        # returns a RunnableRetry, which doesn't expose bind_tools() (see
        # get_raw_chat_model()'s docstring in llm_client.py). Bind first,
        # then wrap the bound runnable in retry so 429s are still handled.
        model_with_tools = with_chat_retry(raw_chat_model.bind_tools(tools))

        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
        messages += _history_to_messages(history)
        messages.append(HumanMessage(content=question))

        for _ in range(MAX_TOOL_ROUNDS):
            response = await model_with_tools.ainvoke(messages)
            messages.append(response)
            if not getattr(response, "tool_calls", None):
                return response.content.strip()

            for call in response.tool_calls:
                tool_fn = tool_map.get(call["name"])
                try:
                    result = await tool_fn.ainvoke(call["args"]) if tool_fn else json.dumps({"error": f"Unknown tool: {call['name']}"})
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", call.get("name"), exc)
                    result = json.dumps({"error": str(exc)})
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

        # Exceeded MAX_TOOL_ROUNDS without a final answer - force one
        # without giving the model tools to call again.
        final = await with_chat_retry(raw_chat_model).ainvoke(
            messages + [HumanMessage(content="Answer the original question now, using only the information already gathered above.")]
        )
        return final.content.strip()
    except Exception:
        logger.exception("Chat answer generation failed")
        try:
            return "I couldn't reach the assistant model just now. Here's what the data shows directly:\n\n" + _no_llm_fallback(question, full_df)
        except Exception:
            return "I couldn't generate an answer right now - please try again."
