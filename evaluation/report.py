"""Builds the plain-text / Markdown evaluation report."""


def _fmt(metric):
    pct, n = metric
    return "n/a" if pct is None else f"{pct}%  ({n} applicable tests)"


def _cut(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def build_report(meta, agent_results, metrics, session_results, verbose=False):
    L = []
    L.append("ITSM AGENT EVALUATION")
    L.append("=" * 60)
    L.append(f"Run: {meta['when']}   LLM mode: {meta['mode']}")
    if meta["mode"] == "rule_based":
        L.append("NOTE: no LLM configured -> this evaluates the rule_based keyword fallback and keyword")
        L.append("      retrieval. Configure LLM_PROVIDER in .env and re-run to evaluate the real agent.")
    if "fatal" in meta:
        L.append(f"\nSETUP ERROR - evaluation could not complete:\n{meta['fatal']}")
    pre = meta.get("preflight", {})
    if pre:
        L.append(f"\nPreflight: categorization {pre.get('categorization', '?')}")
        for k in pre.get("kb", []):
            L.append(f"           KB {k}")

    if metrics:
        L.append("\nAGENT + RAG")
        L.append("-" * 60)
        L.append(f"Total Tests: {metrics['total']}   PASS: {metrics['passed']}   FAIL: {metrics['failed']}   ERROR: {metrics['errors']}")
        for key in ("Tool Selection Accuracy", "RAG Retrieval Accuracy", "Answer Accuracy", "Answer Relevance",
                    "Grounded Answers", "Hallucination Rate", "Unnecessary Tool Call Rate"):
            L.append(f"{key + ':':30s} {_fmt(metrics[key])}")
        L.append("")
        L.append(f"{'ID':4} | {'Question':46} | {'Expected':34} | {'Actual':34} | Result")
        L.append("-" * 140)
        for r in agent_results:
            L.append(f"{r['id']:4} | {_cut(r['question'], 46):46} | {_cut(r['expected'], 34):34} | {_cut(r['actual'], 34):34} | {r['result']}")
        bad = [r for r in agent_results if r["result"] != "PASS"]
        if bad:
            L.append("\nWhy tests did not pass:")
            for r in bad:
                L.append(f"  {r['id']} [{r['result']}]")
                for reason in r.get("reasons", []):
                    L.append(f"      - {reason}")
                if verbose and r.get("answer"):
                    L.append(f"      answer: {_cut(r['answer'], 300)}")
        extra = [r for r in agent_results if r.get("extra_tools")]
        if extra:
            L.append("\nUnnecessary tool calls:")
            for r in extra:
                L.append(f"  {r['id']}: also called {r['extra_tools']}")

    if session_results:
        counted = [s for s in session_results if s["result"] in ("PASS", "FAIL", "ERROR")]
        L.append("\nMULTI-USER SESSION TESTS")
        L.append("-" * 60)
        L.append(f"Passed: {sum(s['result'] == 'PASS' for s in counted)}   "
                 f"Failed: {sum(s['result'] == 'FAIL' for s in counted)}   "
                 f"Errors: {sum(s['result'] == 'ERROR' for s in counted)}   "
                 f"(INFO items are design observations, not counted)")
        L.append("")
        for s in session_results:
            L.append(f"{s['id']:4} [{s['result']:5}] {s['name']}")
            if s["result"] != "PASS" and s.get("detail"):
                L.append(f"            {_cut(s['detail'], 200)}")
            elif verbose and s.get("detail"):
                L.append(f"            {_cut(s['detail'], 200)}")
    payload = {"meta": meta, "metrics": metrics, "agent_results": agent_results, "session_results": session_results}
    return "\n".join(L), payload
