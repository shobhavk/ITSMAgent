"""
Heuristic worklog/resolution-note quality rubric, 0-100. Always runs (free,
instant, explainable) regardless of whether an LLM is configured.

LLM-blended scoring (layering a model's judgment on top of this heuristic
for nuance) happens in graph_pipeline.py's score_node, not here - keeping
the async/LLM/retry concerns in one place (the LangGraph pipeline) rather
than split across this module and the graph.

Rubric, 25 points each:
  - completeness   : has enough substance (length, not a placeholder)
  - root_cause     : mentions cause/diagnosis language
  - resolution     : mentions concrete resolution/action language
  - professionalism: no ALL CAPS shouting, no placeholder text, has
                     sentence structure

Evidence sources: the rubric reads the Work Notes/worklog AND the optional
External Info (external-facing updates) and Internal Info (internal technical
notes) columns. Each source is cleaned and de-duplicated on its own
(prepare_sources), then the rubric is applied to the combined evidence, so a
ticket is never penalised just because one source is empty. Scores stay on the
same 0-100 scale (4 x 25). When only the worklog is supplied the result is the
same as before this change.
"""
import math
import re

from app.models.schemas import WorklogScore

_ROOT_CAUSE_HINTS = re.compile(
    r"\b(root cause|caused by|due to|because|diagnos(is|ed)|found that|identified)\b", re.IGNORECASE
)
_RESOLUTION_HINTS = re.compile(
    r"\b(resolved|fixed|restarted|reconfigur|patched|replaced|rebooted|restored|implemented|applied|escalated)\b",
    re.IGNORECASE,
)
_PLACEHOLDER_HINTS = re.compile(r"^\s*(n/?a|none|test|tbd|todo|\.|-)\s*$", re.IGNORECASE)
_TIMESTAMP_HINTS = re.compile(r"\b\d{1,2}[:/]\d{2}\b|\b\d{4}-\d{2}-\d{2}\b")


def _completeness_score(worklog: str) -> tuple[float, list[str]]:
    flags = []
    length = len(worklog.split())
    if _PLACEHOLDER_HINTS.match(worklog):
        flags.append("Placeholder/empty worklog content.")
        return 0.0, flags
    if length < 5:
        flags.append("Worklog is very short - lacks detail.")
        return 5.0, flags
    if length < 15:
        return 15.0, flags
    return 25.0, flags


def _root_cause_score(worklog: str) -> tuple[float, list[str]]:
    if _ROOT_CAUSE_HINTS.search(worklog):
        return 25.0, []
    return 5.0, ["Root cause not clearly documented."]


def _resolution_score(worklog: str) -> tuple[float, list[str]]:
    if _RESOLUTION_HINTS.search(worklog):
        return 25.0, []
    return 5.0, ["Resolution/action steps not clearly documented."]


def _professionalism_score(worklog: str) -> tuple[float, list[str]]:
    flags = []
    score = 25.0
    letters = [c for c in worklog if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.6:
        score -= 10
        flags.append("Excessive capitalization detected.")
    if len(worklog) > 0 and not _TIMESTAMP_HINTS.search(worklog) and len(worklog.split()) > 20:
        score -= 5
        flags.append("No timestamps/action trail found in a long worklog.")
    return max(score, 0.0), flags


# Used only to FLAG a possible disagreement between sources - never to decide which
# source is right, and never to change the score.
_RESOLVED_CLAIM = re.compile(
    r"\b(resolved|fixed|restored|issue (?:is )?(?:now )?(?:closed|gone)|working (?:again|normally|as expected)|back to normal)\b",
    re.IGNORECASE,
)
_UNRESOLVED_CLAIM = re.compile(
    r"\b(not (?:yet )?(?:resolved|fixed|restored)|unresolved|still (?:down|failing|failed|occurring|happening|not working|broken)"
    r"|issue (?:still )?persists?|persisting|no (?:fix|resolution) (?:yet|found)|workaround only|cannot (?:be )?reproduce[d]?)\b",
    re.IGNORECASE,
)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SOURCE_NAMES = ("worklog", "external_info", "internal_info")


def _clean_source(value) -> str:
    """Safe text for one evidence source: None/NaN/non-strings/control characters
    handled, whitespace normalised line by line, blank lines dropped."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return ""
    text = _CONTROL_CHARS.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _dedupe_lines(text: str) -> str:
    """Drops repeated entries (same line, ignoring case/spacing) within one source."""
    seen: set[str] = set()
    kept = []
    for line in text.split("\n"):
        key = line.casefold()
        if key and key not in seen:
            seen.add(key)
            kept.append(line)
    return "\n".join(kept)


def prepare_sources(worklog=None, external_info=None, internal_info=None) -> dict[str, str]:
    """Returns {"worklog", "external_info", "internal_info"}: each cleaned, de-duplicated,
    and emptied if it holds nothing but placeholder entries (n/a, test, tbd, ...).
    The three stay separate - callers decide how to label/combine them. Never raises."""
    out: dict[str, str] = {}
    for name, raw in zip(_SOURCE_NAMES, (worklog, external_info, internal_info)):
        try:
            text = _dedupe_lines(_clean_source(raw))
            lines = [ln for ln in text.split("\n") if not _PLACEHOLDER_HINTS.match(ln)]
            out[name] = "\n".join(lines)
        except Exception:
            out[name] = ""
    return out


def combine_evidence(sources: dict[str, str]) -> str:
    """One block of evidence for the rubric: all sources, an entry that appears in more than
    one source counted once so copy-pasted notes don't inflate the score."""
    seen: set[str] = set()
    kept = []
    for name in _SOURCE_NAMES:
        for line in (sources.get(name) or "").split("\n"):
            key = line.casefold()
            if key and key not in seen:
                seen.add(key)
                kept.append(line)
    return "\n".join(kept)


def detect_source_conflict(sources: dict[str, str]) -> list[str]:
    """Flags (does not resolve) a disagreement: one source claims the incident is
    resolved/fixed while another says it is not. Neither source is assumed correct."""
    flags = []
    try:
        claims = {}
        for name in _SOURCE_NAMES:
            text = sources.get(name) or ""
            # "not resolved" must not also count as a "resolved" claim.
            claims[name] = (
                bool(_RESOLVED_CLAIM.search(_UNRESOLVED_CLAIM.sub(" ", text))),
                bool(_UNRESOLVED_CLAIM.search(text)),
            )
        for a in _SOURCE_NAMES:
            for b in _SOURCE_NAMES:
                if a < b and ((claims[a][0] and claims[b][1]) or (claims[a][1] and claims[b][0])):
                    label = {"worklog": "Work Notes", "external_info": "External Info", "internal_info": "Internal Info"}
                    flags.append(
                        f"Possible conflict between {label[a]} and {label[b]} about whether the issue is resolved - needs review."
                    )
    except Exception:
        pass
    return flags


def heuristic_score(worklog: str, external_info: str = "", internal_info: str = "") -> WorklogScore:
    """Scores the evidence in Work Notes + External Info + Internal Info. The two optional
    arguments default to "" so existing single-argument callers behave as before."""
    sources = prepare_sources(worklog, external_info, internal_info)
    evidence = combine_evidence(sources)
    conflict_flags = detect_source_conflict(sources)
    if not evidence:
        # Nothing usable. If something was supplied but it was only placeholder text, score that
        # placeholder exactly as before (so "TEST"/"n/a" still gets the placeholder flag).
        for raw in (worklog, external_info, internal_info):
            placeholder = _clean_source(raw)
            if placeholder:
                return _score_text(placeholder)
        return _score_text("")
    result = _score_text(evidence)
    if conflict_flags:
        result.flags = result.flags + conflict_flags
    return result


def _score_text(worklog: str) -> WorklogScore:
    worklog = (worklog or "").strip()
    if not worklog:
        return WorklogScore(
            ticket_id="", score=0, breakdown={"completeness": 0, "root_cause": 0, "resolution": 0, "professionalism": 0},
            flags=["No worklog provided."],
        )

    completeness, f1 = _completeness_score(worklog)
    root_cause, f2 = _root_cause_score(worklog)
    resolution, f3 = _resolution_score(worklog)
    professionalism, f4 = _professionalism_score(worklog)

    total = round(completeness + root_cause + resolution + professionalism)
    return WorklogScore(
        ticket_id="",
        score=total,
        breakdown={
            "completeness": completeness,
            "root_cause": root_cause,
            "resolution": resolution,
            "professionalism": professionalism,
        },
        flags=f1 + f2 + f3 + f4,
    )
