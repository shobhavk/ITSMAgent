"""
Scores the Agent + RAG against TEST_CASES (Parts 1 and 2).
Every test is wrapped in try/except: one broken test never stops the run.
"""
import json
import re
import traceback

from evaluation.harness import ask_agent
from evaluation.test_cases import NOT_FOUND_PHRASES, TEST_CASES, build_truth

# Answers that mean "the agent itself could not run" -> ERROR, not FAIL.
_AGENT_FAILURE_PREFIXES = (
    "i couldn't compute an answer", "i couldn't generate an answer", "i couldn't reach the assistant model",
)
_NUM_RE = re.compile(r"(?<![\w.\-])\d+(?:\.\d+)?%?(?![\w\-])")
_FILE_RE = re.compile(r"[\w\-]+\.(?:md|pdf|docx|txt)", re.I)


def _resolve(v, T):
    return v(T) if callable(v) else v


def _has_not_found(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in NOT_FOUND_PHRASES)


def _ungrounded_numbers(answer: str, question: str, calls) -> list[str]:
    """Numbers in the answer that appear in NO tool output (and not in the question)."""
    evidence = question + " " + " ".join(json.dumps(c.result, default=str) for c in calls)
    evidence_nums = set(_NUM_RE.findall(evidence))
    answer = re.sub(r"(?m)^\s*\d+[.)]\s", "", answer)   # "11. ..." list numbering is not a claim
    bad = []
    for n in _NUM_RE.findall(answer):
        if n == "100":
            continue  # "score / 100" scale label
        plain = n.rstrip("%")
        if "." not in plain and not n.endswith("%") and int(plain) <= 10:
            continue  # list numbering / tiny counts - too noisy to judge
        if n not in evidence_nums and plain not in evidence_nums:
            bad.append(n)
    return bad


def score_case(case: dict, T: dict, answer: str, calls: list, known_docs: set) -> dict:
    """Returns the individual checks (True/False, or None = not applicable) + reasons."""
    reasons = []
    called = [c.name for c in calls]
    called_set = set(called)

    # --- tool selection (all required tools; alternates allowed) ---
    tool_ok = None
    expected = case.get("expected_tools", [])
    if expected:
        options = [expected] + case.get("alt_tools", [])
        tool_ok = any(set(o) <= called_set for o in options)
        if not tool_ok:
            reasons.append(f"tools: expected {expected}, got {called or 'none'}")
        allowed = set().union(*[set(o) for o in options]) | set(case.get("allowed_extra", []))
    else:
        allowed = called_set  # no tool requirement -> nothing counts as "extra"
    extra = sorted(called_set - allowed)

    # --- RAG retrieval ---
    rag_ok = None
    retrieved = [d for c in calls if c.name == "search_knowledge_base" for d in c.docs]
    if case.get("expected_doc"):
        rag_ok = case["expected_doc"] in retrieved
        if not rag_ok:
            reasons.append(f"rag: '{case['expected_doc']}' not retrieved (got {sorted(set(retrieved)) or 'nothing'})")

    # --- answer correctness ---
    low = answer.lower()
    checks = []
    for v in case.get("must_contain", []):
        s = str(_resolve(v, T))
        ok = s.lower() in low
        checks.append(ok)
        if not ok:
            reasons.append(f"answer: missing '{s}'")
    if case.get("must_contain_any"):
        ok = any(str(_resolve(v, T)).lower() in low for v in case["must_contain_any"])
        checks.append(ok)
        if not ok:
            reasons.append(f"answer: none of {case['must_contain_any']} found")
    if case.get("not_found"):
        ok = _has_not_found(answer)
        checks.append(ok)
        if not ok:
            reasons.append("answer: did not say the information is unavailable / not in the knowledge base")
    correct = all(checks) if checks else True

    # --- relevance ---
    terms = case.get("topic_terms")
    relevant = any(t.lower() in low for t in terms) if terms else None
    if relevant is False:
        reasons.append("relevance: answer does not mention the topic")

    # --- hallucination heuristics ---
    halluc = []
    for s in case.get("forbidden", []):
        if s.lower() in low:
            halluc.append(f"contains forbidden text '{s}'")
    for rx in case.get("forbidden_regex", []):
        if re.search(rx, answer, re.I):
            halluc.append(f"matches forbidden pattern {rx}")
    cited = {m.lower() for m in _FILE_RE.findall(answer)}
    unknown = [f for f in cited if f not in {d.lower() for d in known_docs}]
    if unknown:
        halluc.append(f"cites document(s) not in the KB: {unknown}")
    if case.get("not_found") and not _has_not_found(answer):
        if "(source:" in low or any(d.lower() in low for d in known_docs):
            halluc.append("presented unrelated KB text as if it answered the question")
    bad_nums = _ungrounded_numbers(answer, case["question"], calls)
    if bad_nums:
        halluc.append(f"numbers not found in any tool output: {bad_nums}")

    # --- grounding ---
    if case.get("not_found"):
        grounded = _has_not_found(answer) and not halluc
    elif case.get("expected_doc"):
        d = case["expected_doc"].lower()
        cites = d in low or d.rsplit(".", 1)[0] in low
        grounded = cites and not halluc
        if not cites:
            reasons.append(f"grounding: source document '{case['expected_doc']}' not named in the answer")
    else:
        grounded = not halluc
    if halluc:
        reasons.append("hallucination: " + "; ".join(halluc))

    return dict(tool_ok=tool_ok, rag_ok=rag_ok, correct=correct, relevant=relevant,
                grounded=grounded, hallucinated=bool(halluc), extra_tools=extra,
                called=called, retrieved_docs=sorted(set(retrieved)), reasons=reasons)




def _verdict(s: dict) -> str:
    applicable = [s["tool_ok"], s["rag_ok"], s["correct"], s["grounded"]]
    ok = all(v for v in applicable if v is not None) and not s["hallucinated"]
    return "PASS" if ok else "FAIL"


async def run_agent_tests(df, recorder, known_docs, only=None) -> list[dict]:
    T = build_truth(df)
    results = []
    for case in TEST_CASES:
        if only and case["id"] not in only:
            continue
        row = {"id": case["id"], "area": case["area"], "question": case["question"],
               "expected": "+".join(case.get("expected_tools", [])) or "(no specific tool)"}
        try:
            answer, calls = await ask_agent(case["question"], df, recorder)
            row["answer"] = answer
            if answer.strip().lower().startswith(_AGENT_FAILURE_PREFIXES):
                raise RuntimeError(f"agent could not produce an answer: {answer[:120]}")
            s = score_case(case, T, answer, calls, known_docs)
            row.update(s)
            row["actual"] = "+".join(dict.fromkeys(s["called"])) or "(none)"
            row["result"] = _verdict(s)
        except Exception as exc:  # one failed test must never stop the run
            row.update(result="ERROR", actual="(error)", reasons=[f"{type(exc).__name__}: {exc}"],
                       tool_ok=None, rag_ok=None, correct=None, relevant=None, grounded=None,
                       hallucinated=None, extra_tools=[], answer=row.get("answer", ""))
            row["traceback"] = traceback.format_exc(limit=3)
        results.append(row)
    return results


def _pct(num, den):
    return None if den == 0 else round(100.0 * num / den, 1)


def compute_metrics(results: list[dict]) -> dict:
    ok_rows = [r for r in results if r["result"] != "ERROR"]

    def rate(key, only_true_means_pass=True):
        applicable = [r for r in ok_rows if r.get(key) is not None]
        return _pct(sum(1 for r in applicable if r[key]), len(applicable)), len(applicable)

    tool, tool_n = rate("tool_ok")
    rag, rag_n = rate("rag_ok")
    acc, acc_n = rate("correct")
    rel, rel_n = rate("relevant")
    grd, grd_n = rate("grounded")
    hall_n = len(ok_rows)
    hall = _pct(sum(1 for r in ok_rows if r["hallucinated"]), hall_n)
    extra = _pct(sum(1 for r in ok_rows if r["extra_tools"]), hall_n)
    return {
        "total": len(results),
        "passed": sum(r["result"] == "PASS" for r in results),
        "failed": sum(r["result"] == "FAIL" for r in results),
        "errors": sum(r["result"] == "ERROR" for r in results),
        "Tool Selection Accuracy": (tool, tool_n),
        "RAG Retrieval Accuracy": (rag, rag_n),
        "Answer Accuracy": (acc, acc_n),
        "Answer Relevance": (rel, rel_n),
        "Grounded Answers": (grd, grd_n),
        "Hallucination Rate": (hall, hall_n),
        "Unnecessary Tool Call Rate": (extra, hall_n),
    }
