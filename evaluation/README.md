# Evaluation (Step 14)

    python evaluate_agent.py              # everything
    python evaluate_agent.py --agent      # Agent + RAG questions only
    python evaluate_agent.py --sessions   # multi-user tests only
    python evaluate_agent.py --only T01,T19 --verbose

- `test_cases.py`   the questions; add your own (expected values are computed from the data, not typed in)
- `agent_eval.py`   scoring + percentages
- `session_tests.py` two-user isolation tests (REST by API key, Gradio handlers, KB scope)
- `harness.py`      temp DB, tool-call recorder (no changes to rag.py), fixture loading
- `fixtures/`       incidents_A/B CSVs, KB documents (files starting with `_` are per-user docs for session tests only)
- `reports/`        last run (`evaluation_report.md`, `evaluation_results.json`)

Uses a throwaway database. With an LLM configured in `.env` it evaluates the real tool-calling agent; otherwise the rule_based fallback.
