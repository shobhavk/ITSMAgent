"""Turns raw chat-model / fallback output into readable Markdown.

Why this exists: the shared system guardrail tells models to "always respond
with strictly valid JSON", and the chat prompt defines no schema, so the model
sometimes answers with its own {"answer": "...", "sources": [...]} object (or a
truncated one). The chat tab used to show that blob verbatim. clean_answer()
extracts the answer text, renders sources as a short footnote line, and never
raises - on anything it cannot understand it returns the original text.
"""
from __future__ import annotations

import json
import re

_ANSWER_KEYS = ("answer", "response", "final_answer", "text", "content", "message", "summary")
_SOURCE_KEYS = ("sources", "citations", "references")
_TOOL_LABELS = {
    "get_incident_summary": "incident summary",
    "get_category_analysis": "category analysis",
    "get_priority_analysis": "priority analysis",
    "get_server_analysis": "server analysis",
    "get_recurring_issues": "recurring issues",
    "get_recommendations": "recommendations",
    "search_incidents": "incident search",
    "search_knowledge_base": "knowledge base search",
}
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_LOOKS_JSON = re.compile(r"^\s*(```[a-zA-Z]*\s*)?[\{\[]")
_ANSWER_RE = re.compile(r'"(?:answer|response|final_answer|text|content|message|summary)"\s*:\s*"((?:[^"\\]|\\.)*)', re.S)


def _unescape(text: str) -> str:
    """Decode JSON string escapes; also fixes literal '\\n' that survived."""
    try:
        return json.loads('"' + text + '"')
    except Exception:
        return text.replace("\\n", "\n").replace('\\"', '"').replace("\\t", "  ")


def _humanize_key(key: str) -> str:
    return str(key).replace("_", " ").strip().capitalize()


def _scalar(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def json_to_markdown(obj, depth: int = 0, _budget=None) -> str:
    """Generic, readable Markdown for tool-result style JSON (used by the
    no-LLM fallback and for dict answers without an answer key)."""
    budget = _budget if _budget is not None else [40]  # max lines overall
    pad = "  " * depth
    lines: list[str] = []
    if budget[0] <= 0:
        return ""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if budget[0] <= 0:
                lines.append(f"{pad}- ...")
                break
            if v in (None, "", [], {}):
                continue
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}- **{_humanize_key(k)}**")
                budget[0] -= 1
                sub = json_to_markdown(v, depth + 1, budget)
                if sub:
                    lines.append(sub)
            else:
                lines.append(f"{pad}- **{_humanize_key(k)}**: {_scalar(v)}")
                budget[0] -= 1
    elif isinstance(obj, list):
        for item in obj[:10]:
            if budget[0] <= 0:
                break
            if isinstance(item, dict):
                flat = all(not isinstance(v, (dict, list)) for v in item.values())
                if flat:
                    parts = [(k, _scalar(v)) for k, v in item.items() if v not in (None, "")]
                    if sum(len(v) for _, v in parts) <= 140:
                        lines.append(f"{pad}- " + " · ".join(f"{_humanize_key(k)}: {v}" for k, v in parts))
                        budget[0] -= 1
                    else:
                        # Long record: headline + one sub-bullet per field, easier to scan.
                        head_key = next((k for k, _ in parts if k in ("observation", "title", "name", "issue", "area", "category")), parts[0][0])
                        head_val = dict(parts)[head_key]
                        lines.append(f"{pad}- **{head_val}**")
                        for k, v in parts:
                            if k != head_key:
                                lines.append(f"{pad}  - *{_humanize_key(k)}:* {v}")
                        budget[0] -= 1 + len(parts) - 1
                else:
                    lines.append(f"{pad}-")
                    budget[0] -= 1
                    lines.append(json_to_markdown(item, depth + 1, budget))
            else:
                lines.append(f"{pad}- {_scalar(item)}")
                budget[0] -= 1
        if len(obj) > 10:
            lines.append(f"{pad}- ... and {len(obj) - 10} more")
    else:
        lines.append(f"{pad}{_scalar(obj)}")
    return "\n".join(l for l in lines if l)


def _format_sources(sources) -> str:
    labels: list[str] = []
    for src in sources if isinstance(sources, list) else [sources]:
        if isinstance(src, str):
            label = src
        elif isinstance(src, dict):
            kind = str(src.get("type") or "").lower()
            tool = src.get("tool") or src.get("name")
            doc = src.get("document") or src.get("source") or src.get("title") or src.get("file")
            if doc:
                page, section = src.get("page"), src.get("section")
                where = f"p. {page}" if page else (str(section) if section else "")
                label = f"Knowledge base: {doc}" + (f" ({where})" if where else "")
            elif tool:
                prefix = "Incident data" if ("incident" in kind or not kind) else _humanize_key(kind)
                label = f"{prefix}: {_TOOL_LABELS.get(str(tool), str(tool).replace('_', ' '))}"
            else:
                continue
        else:
            continue
        if label and label not in labels:
            labels.append(label)
    return "\n\n_📎 Sources: " + " · ".join(labels[:6]) + "_" if labels else ""


def _extract_balanced(text: str):
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def clean_answer(raw) -> str:
    """Readable Markdown for a model answer. Plain-text answers pass through
    unchanged (apart from literal '\\n' sequences being turned into newlines)."""
    try:
        text = str(raw if raw is not None else "").strip()
        if not text:
            return text
        if not _LOOKS_JSON.match(text):
            if "\\n" in text and "\n" not in text:
                text = text.replace("\\n", "\n")
            return text

        body = _FENCE.sub("", text).strip()
        data = None
        try:
            data = json.loads(body)
        except Exception:
            data = _extract_balanced(body)

        if isinstance(data, dict):
            answer = next((data[k] for k in _ANSWER_KEYS if data.get(k)), None)
            sources = next((data[k] for k in _SOURCE_KEYS if data.get(k)), None)
            if answer is None:
                return json_to_markdown({k: v for k, v in data.items() if k not in _SOURCE_KEYS}) + _format_sources(sources)
            answer_text = answer if isinstance(answer, str) else json_to_markdown(answer)
            return clean_answer(answer_text) + _format_sources(sources)
        if isinstance(data, list):
            return json_to_markdown(data)

        # Truncated / invalid JSON: pull the answer string out with a regex.
        m = _ANSWER_RE.search(body)
        if m:
            answer_text = _unescape(m.group(1)).strip()
            tools = re.findall(r'"(?:tool|document)"\s*:\s*"([^"]+)"', body)
            srcs = [{"type": "incident_data", "tool": t} if t in _TOOL_LABELS else {"document": t} for t in tools]
            return answer_text + _format_sources(srcs)
        return text
    except Exception:
        return str(raw) if raw is not None else ""
