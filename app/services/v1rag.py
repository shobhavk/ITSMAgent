"""
Chat/Q&A over the ticket set that was just analyzed - a hybrid RAG agent,
not plain top-K retrieval.

Why plain retrieval isn't enough:
  Top-K similarity search answers "find me tickets like X" well, but it
  quietly breaks "how many P1 tickets are there" or "summarize the
  database team's issues" - if 40 tickets match and only the 8 most
  *similar* come back, the model either under-counts or guesses at the
  rest. Completeness and similarity are different requirements; one
  retrieval strategy can't serve both.

The fix - two tools, not one, bound to the chat model via LangChain's
bind_tools()/.ainvoke() (same async tool-calling pattern used elsewhere
in this codebase):
  - filter_tickets     : deterministic, exact-match filtering (category /
                          priority / status / host / assignment group /
                          worklog score range) over EVERY analyzed ticket,
                          done as a plain Python loop over the batch - no
                          LLM, no embeddings, no sampling. Returns the
                          TRUE total count plus a capped sample. This is
                          the tool for anything needing completeness:
                          counts, "list all", "summarize category X".
  - search_similar_tickets : the semantic/keyword TicketIndex search from
                          before - for "show me an example of..." style
                          questions where a handful of illustrative
                          tickets is the actual goal, not every match.
The model decides which to call (or both - filter first to narrow, then
search or summarize within that narrowed set) based on each tool's
docstring, which doubles as its LLM-facing description.

Design choice - still no separate vector database:
  filter_tickets needs no vector store at all (it's a plain filter over
  a Python list). search_similar_tickets keeps the same in-memory numpy
  cosine-similarity index as before - a single analysis batch is small
  enough that this remains simpler to operate than standing up a vector
  DB for a v1 feature. See TicketIndex below.

Degradation, consistent with the rest of the app: no chat model
configured (LLM_PROVIDER=rule_based) -> no tool-calling loop is possible,
so answer_question() falls back to a best-effort heuristic: it tries to
detect an explicit category/priority/host/team name in the question text
and runs filter_tickets directly against that; otherwise it falls back to
keyword search. Either way it never raises - any failure degrades to a
stats-only answer.
"""
import json
import logging

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.security import sanitize_for_llm
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
FILTER_SAMPLE_LIMIT = 15

AGENT_SYSTEM_PROMPT = (
    SYSTEM_GUARDRAIL + "\n\n"
    "You are answering a manager's question about a batch of IT incident tickets that has "
    "already been analyzed and categorized. You have two tools:\n\n"
    "- filter_tickets: deterministic, exact-match filtering over EVERY analyzed ticket by "
    "category / priority / status / host / assignment_group / worklog score range. Returns "
    "the TRUE total count of matches (not a sample) plus a capped list of example tickets. "
    "ALWAYS use this for anything involving counting, \"how many\", \"list all\", or "
    "summarizing a specific category/team/host/priority - anywhere completeness matters "
    "more than similarity.\n"
    "- search_similar_tickets: fuzzy semantic/keyword search returning a handful of "
    "illustrative tickets. Use this only for \"show me an example of...\" style questions, "
    "or to look for patterns inside a set you've already narrowed with filter_tickets.\n\n"
    "AGGREGATE STATS FOR THE WHOLE BATCH:\n{stats}\n\n"
    "Rules: never state a number you did not get from a tool call or the aggregate stats "
    "above. If a question needs both a filter and a summary of what's inside it (e.g. "
    "\"what's driving P1 incidents\"), call filter_tickets first, then summarize using the "
    "sample tickets it returns. Be concise and management-friendly: lead with the direct "
    "answer, then one or two sentences of support. Cite ticket IDs in parentheses when "
    "referencing specific tickets. If, after using your tools, you still don't have enough "
    "information, say so plainly rather than guessing."
)


class TicketIndex:
    """In-memory semantic index over one analysis batch's tickets. Backs
    the search_similar_tickets tool (and the no-LLM fallback search)."""

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
            logger.info("RAG embedding index unavailable, falling back to keyword search: %s", exc)

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
                logger.info("RAG similarity search failed, falling back to keyword search: %s", exc)

        # No embeddings model (rule_based mode) or the call failed:
        # plain keyword-overlap fallback so chat still works without an LLM.
        terms = [t for t in query.lower().split() if len(t) > 2]
        scored = [(sum(1 for t in terms if t in row["text"].lower()), row) for row in self.rows]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits = [row for count, row in scored if count > 0]
        return hits[:top_k] if hits else self.rows[:top_k]


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


def build_index_from_tickets(tickets: list) -> "TicketIndex":
    """Builds an index from a list of AnalyzedTicket (pydantic) objects -
    used by the REST /api/v1/chat endpoint."""
    rows = [
        _row_from_fields(
            t.ticket_id, t.category, t.priority, t.status, t.host, t.assignment_group, t.worklog_score,
            t.short_description, t.description, t.worklog,
        )
        for t in tickets
    ]
    return TicketIndex(rows)


def build_index_from_dataframe(full_df) -> "TicketIndex":
    """Builds an index from the results dataframe used by the Gradio UI
    (same shape as the CSV export)."""
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


def _filter_rows(
    rows: list[dict],
    category: str | None = None,
    priority: str | None = None,
    status: str | None = None,
    host: str | None = None,
    assignment_group: str | None = None,
    min_score: int | None = None,
    max_score: int | None = None,
    limit: int = FILTER_SAMPLE_LIMIT,
) -> dict:
    """Pure-Python exact-match filter over every row - no LLM, no
    embeddings, no sampling. This is what makes filter_tickets's count
    trustworthy: it inspects the whole batch, every time."""

    def _matches(row: dict) -> bool:
        if category and row["category"].lower() != category.lower():
            return False
        if priority and row["priority"].lower() != priority.lower():
            return False
        if status and row["status"].lower() != status.lower():
            return False
        if host and row["host"].lower() != host.lower():
            return False
        if assignment_group and row["assignment_group"].lower() != assignment_group.lower():
            return False
        score = row["worklog_score"]
        if min_score is not None and (score is None or score < min_score):
            return False
        if max_score is not None and (score is None or score > max_score):
            return False
        return True

    matched = [r for r in rows if _matches(r)]
    scores = [r["worklog_score"] for r in matched if r["worklog_score"] is not None]
    return {
        "total_matching": len(matched),
        "avg_worklog_score": round(sum(scores) / len(scores), 1) if scores else None,
        "sample_tickets": [_public_row(r, 300) for r in matched[:limit]],
    }


def _public_row(row: dict, excerpt_len: int) -> dict:
    """A tool-result-safe view of a row: no raw 'text' key, an already
    length-capped/sanitized excerpt instead."""
    return {
        "ticket_id": row["ticket_id"],
        "category": row["category"],
        "priority": row["priority"],
        "status": row["status"],
        "host": row["host"],
        "assignment_group": row["assignment_group"],
        "worklog_score": row["worklog_score"],
        "excerpt": sanitize_for_llm(row["text"], max_len=excerpt_len),
    }


def _make_tools(index: "TicketIndex") -> list:
    """Builds the two tools bound to this analysis batch via closure, so
    each chat call gets fresh tools scoped to the right ticket set."""

    @tool
    async def filter_tickets(
        category: str = "",
        priority: str = "",
        status: str = "",
        host: str = "",
        assignment_group: str = "",
        min_score: int = -1,
        max_score: int = -1,
    ) -> str:
        """Deterministically filters ALL analyzed tickets by exact-match category, priority,
        status, host, and/or assignment_group, and/or a worklog score range (min_score/
        max_score, 0-100). Leave a field empty/-1 to not filter on it. Returns the true total
        count of matching tickets (not a sample) plus up to 15 example tickets with a short
        excerpt each. Use this for any question about counts, "how many", "list all", or a
        summary of a specific category/team/host/priority - anywhere completeness matters."""
        result = _filter_rows(
            index.rows,
            category=category or None,
            priority=priority or None,
            status=status or None,
            host=host or None,
            assignment_group=assignment_group or None,
            min_score=min_score if min_score >= 0 else None,
            max_score=max_score if max_score >= 0 else None,
        )
        return json.dumps(result, default=str)

    @tool
    async def search_similar_tickets(query: str) -> str:
        """Fuzzy semantic/keyword search for a handful of tickets whose text resembles the
        query - good for "find an example of X" or "what does a ticket about Y look like"
        questions. Does NOT guarantee completeness - use filter_tickets for counts or "all
        tickets matching..." questions."""
        rows = await index.search(query)
        return json.dumps({"tickets": [_public_row(r, 300) for r in rows]}, default=str)

    return [filter_tickets, search_similar_tickets]


def _detect_filters_from_question(question: str, rows: list[dict]) -> dict:
    """Best-effort keyword detection used only when no chat model is
    configured (rule_based mode), so filter_tickets-equivalent behavior
    is still available without tool-calling. Looks for an exact category/
    priority/host/assignment_group value from the batch mentioned in the
    question text."""
    q = question.lower()
    filters: dict = {}
    for field in ("category", "priority", "assignment_group", "host"):
        values = {r[field] for r in rows if r[field]}
        for value in values:
            if value.lower() in q:
                filters[field] = value
                break
    return filters


def _format_stats(stats: dict) -> str:
    lines = []
    for key, value in stats.items():
        if isinstance(value, dict):
            top = sorted(value.items(), key=lambda kv: kv[1], reverse=True)[:8]
            lines.append(f"{key}: " + ", ".join(f"{k}={v}" for k, v in top))
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _history_to_messages(history: list[tuple[str, str]]) -> list:
    recent = (history or [])[-MAX_HISTORY_TURNS:]
    messages = []
    for question, answer in recent:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages


def _no_llm_fallback(question: str, index: "TicketIndex", stats_text: str) -> str:
    """rule_based mode: no chat model, so no tool-calling loop is
    possible. Still tries to give a grounded, useful answer."""
    filters = _detect_filters_from_question(question, index.rows)
    if filters:
        result = _filter_rows(index.rows, **filters, limit=10)
        lines = [
            f"Found {result['total_matching']} matching ticket(s)"
            + (f", average worklog score {result['avg_worklog_score']}" if result["avg_worklog_score"] is not None else "")
            + ":"
        ]
        for t in result["sample_tickets"]:
            lines.append(f"- [{t['ticket_id']}] {t['category']} / {t['priority']} / score={t['worklog_score']}: {t['excerpt']}")
        return "\n".join(lines)

    return f"Based on the analyzed batch:\n{stats_text}"


async def answer_question(
    question: str,
    index: "TicketIndex",
    stats: dict,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Answers a management question using a small tool-calling loop
    (filter_tickets for completeness, search_similar_tickets for
    examples) grounded in the given ticket index + aggregate stats.
    Never raises - degrades to a stats-only or filter-only answer on any
    failure so the chat UI always gets something useful back."""
    question = sanitize_for_llm(question, max_len=1000).strip()
    if not question:
        return 'Ask a question about the analyzed tickets - e.g. "how many P1 tickets are there?"'

    stats_text = _format_stats(stats)
    raw_chat_model = get_raw_chat_model()
    if raw_chat_model is None:
        return _no_llm_fallback(question, index, stats_text)

    try:
        tools = _make_tools(index)
        tool_map = {t.name: t for t in tools}
        # bind_tools() must be called on the RAW model - with_chat_retry()
        # returns a RunnableRetry, which doesn't expose bind_tools() (see
        # get_raw_chat_model()'s docstring in llm_client.py). Bind first,
        # then wrap the bound runnable in retry so 429s are still handled.
        model_with_tools = with_chat_retry(raw_chat_model.bind_tools(tools))

        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT.format(stats=stats_text))]
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
                    result = await tool_fn.ainvoke(call["args"]) if tool_fn else json.dumps({"error": "unknown tool"})
                except Exception as exc:
                    result = json.dumps({"error": str(exc)})
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

        # Exceeded MAX_TOOL_ROUNDS without a final answer - force one
        # without giving the model tools to call again.
        final = await with_chat_retry(raw_chat_model).ainvoke(
            messages + [HumanMessage(content="Answer the original question now, using only the information already gathered above.")]
        )
        return final.content.strip()
    except Exception as exc:
        logger.exception("Chat answer generation failed, falling back to raw stats")
        return "I couldn't reach the assistant model just now. Here's what the data shows directly:\n\n" + stats_text
