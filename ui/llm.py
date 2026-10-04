"""LLM access for the Generate summary / write-up buttons (provider chain + prompts)."""
import json
import os

from app.services import overview_metrics


async def _call_llm(prompt: str, max_tokens: int = 800) -> "tuple[str, bool]":
    """Shared three-tier LLM call used by every "Generate summary / write-up"
    button (Executive Summary, Recommendations write-up, ...):

      1. the app's own configured provider (llm_client / LLM_PROVIDER),
      2. the direct Anthropic API (ANTHROPIC_API_KEY),
      3. give up -> ("", False) so the caller renders its deterministic,
         rule-based fallback built from the identical numbers.

    Returns (text, used_llm). Never raises."""
    # 1. The app's configured provider (SAP GenAI Hub / OpenAI-compatible).
    try:
        from app.services.llm_client import get_chat_model

        chat_model = get_chat_model()
        if chat_model is not None:
            response = await chat_model.ainvoke(prompt)
            text = getattr(response, "content", "") or ""
            if isinstance(text, list):  # some providers return content blocks
                text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
            if text.strip():
                return text.strip(), True
    except Exception:
        pass

    # 2. Direct Anthropic path.
    try:
        import anthropic  # local import - optional dependency for this feature only

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
        if not text:
            raise RuntimeError("empty LLM response")
        return text, True
    except Exception:
        return "", False


async def _llm_executive_summary(payload: dict) -> "tuple[str, bool]":
    """Sends ONLY the small aggregated `payload` dict (KPIs/health/
    attention/trend direction - no ticket rows, descriptions, or worklog
    text) to the LLM and asks for a short management-friendly summary.

    Now goes through the same provider chain as the Recommendations
    write-up (see _call_llm), so the button works whenever the app's
    configured LLM works - not only when ANTHROPIC_API_KEY is set. Falls
    back to the deterministic rule-based summary of the same numbers if no
    LLM is reachable, so the button never errors out.

    Returns (summary_text, used_llm).
    """
    prompt = (
        "You are an ITSM reporting assistant writing for a management audience. The JSON below "
        "contains ALREADY-CALCULATED, aggregated incident metrics (KPIs, health indicators, items "
        "needing attention, and volume trend).\n\n"
        "Write a short executive summary. Rules:\n"
        "- 4-6 sentences of plain English.\n"
        "- Use ONLY the numbers present in the JSON. Never introduce, round differently, "
        "recalculate, or estimate any figure.\n"
        "- Do not mention individual tickets - these are aggregated figures.\n"
        "- Lead with the overall picture, then call out what needs attention.\n\n"
        f"Aggregated metrics (JSON):\n{json.dumps(payload, default=str)}"
    )
    text, used_llm = await _call_llm(prompt, max_tokens=600)
    if used_llm:
        return text, True
    return overview_metrics.fallback_summary_text(payload), False


async def _llm_recommendations_writeup(payload: dict) -> "tuple[str, bool]":
    """Sends ONLY the small aggregated `payload` (the already-computed
    recommendations and headline metrics from
    recommendations.build_llm_payload - no ticket rows, descriptions, or
    worklog text) and asks the model to re-voice them for management.

    Tries the app's configured provider first (llm_client, i.e. whatever
    LLM_PROVIDER is set to), then the same direct Anthropic path the
    Executive Summary uses, then gives up and returns the deterministic
    write-up built from the identical numbers - so the button always
    returns something useful and never errors out.

    Returns (text, used_llm).
    """
    prompt = (
        "You are an ITSM reporting assistant writing for a management audience. The JSON below "
        "contains recommendations that have ALREADY been calculated from incident data, each with "
        "its own evidence, attention level, and expected benefit.\n\n"
        "Rewrite them as a short, readable management briefing. Rules:\n"
        "- Use ONLY the numbers present in the JSON. Never introduce, round differently, "
        "recalculate, or estimate any figure.\n"
        "- Do not invent recommendations that are not in the JSON, and do not drop any.\n"
        "- Keep the most urgent items first, and name the specific servers, categories, and "
        "groups involved.\n"
        "- Lead with a 2-3 sentence framing paragraph, then one short paragraph or bullet per "
        "recommendation.\n\n"
        f"Recommendations (JSON):\n{json.dumps(payload, default=str)}"
    )

    return await _call_llm(prompt, max_tokens=1200)


async def _llm_timeline_summary(payload: dict) -> "tuple[str, bool]":
    """Same three-tier approach as _llm_recommendations_writeup: the app's
    configured provider, then the direct Anthropic path, then give up -
    only the small aggregated payload (counts/percentages, no per-ticket
    text) ever reaches the LLM."""
    prompt = (
        "You are an ITSM reporting assistant writing for a management audience. The JSON below "
        "contains ALREADY-CALCULATED per-batch incident timeline metrics: acknowledgment times, "
        "detected reassignments, and External Info audit-trail quality.\n\n"
        "Write a short (4-6 sentence) management summary. Rules:\n"
        "- Use ONLY the numbers present in the JSON - never invent, recalculate, or estimate.\n"
        "- Call out anything that looks like a process gap (slow acknowledgment, missing or "
        "untimed External Info, undetected reassignment trail).\n"
        "- Name specific ticket IDs only if they appear in the JSON.\n\n"
        f"Timeline metrics (JSON):\n{json.dumps(payload, default=str)}"
    )

    return await _call_llm(prompt, max_tokens=600)
