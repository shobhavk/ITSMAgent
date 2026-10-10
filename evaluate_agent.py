"""
ITSM Agent evaluation (Step 14).     Run:   python evaluate_agent.py

    python evaluate_agent.py               # everything
    python evaluate_agent.py --agent       # Agent + RAG questions only
    python evaluate_agent.py --sessions    # multi-user session tests only
    python evaluate_agent.py --only T01,T19  # specific questions
    python evaluate_agent.py --verbose     # show answers and failure reasons

Uses a throwaway database (your real data is never touched). If your .env
configures a real LLM (LLM_PROVIDER=sap_genai_hub / openai_compat) the real
tool-calling agent is evaluated; otherwise the built-in rule_based fallback is.
Results are also saved to evaluation/reports/.
"""
import argparse
import asyncio
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import logging
import warnings

from evaluation.harness import FIXTURES, setup_environment

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)   # keep the report readable (httpx/gradio chatter)

TMP_DIR = setup_environment()   # MUST happen before any `app` import

from evaluation.harness import ToolRecorder, load_dataset, seed_knowledge_base, shared_kb_files  # noqa: E402
from evaluation import agent_eval, session_tests  # noqa: E402
from evaluation.report import build_report  # noqa: E402


async def main(args) -> int:
    from app.config import get_settings
    from app.services import knowledge_base
    mode = get_settings().LLM_PROVIDER

    recorder = ToolRecorder()
    recorder.install()
    out = {"mode": mode, "when": datetime.now().strftime("%Y-%m-%d %H:%M"), "preflight": {}}
    agent_results, session_results, metrics = [], [], None
    run_agent = args.agent or not args.sessions
    run_sessions = args.sessions or not args.agent

    try:
        # ---- preflight: load dataset A, seed the shared KB, sanity-check categorization ----
        result_a, df_a = await load_dataset(FIXTURES / "incidents_A.csv")
        expected = json.load(open(FIXTURES / "incidents_A_expected_categories.json"))
        correct = sum(1 for t in result_a.results if expected.get(t.ticket_id) == t.category)
        out["preflight"]["categorization"] = f"{correct}/{len(expected)} tickets categorized as designed"
        seeded = await seed_knowledge_base(shared_kb_files())
        out["preflight"]["kb"] = [f"{s['file']}: {s['status']} ({s['chunk_count']} chunks)" for s in seeded]
        known_docs = {d["document_name"] for d in knowledge_base.list_documents()}

        if run_agent:
            only = set(args.only.split(",")) if args.only else None
            agent_results = await agent_eval.run_agent_tests(df_a, recorder, known_docs, only)
            metrics = agent_eval.compute_metrics(agent_results)
        if run_sessions:
            session_results = await session_tests.run_session_tests(recorder)
    except Exception as exc:   # setup failure - report it, don't crash with a traceback
        import traceback
        out["fatal"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
    finally:
        recorder.uninstall()

    text, payload = build_report(out, agent_results, metrics, session_results, verbose=args.verbose)
    print(text)
    rep_dir = Path(__file__).parent / "evaluation" / "reports"
    rep_dir.mkdir(exist_ok=True)
    (rep_dir / "evaluation_report.md").write_text(text, encoding="utf-8")
    (rep_dir / "evaluation_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: evaluation/reports/evaluation_report.md and evaluation_results.json")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    return 1 if "fatal" in out else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate the ITSM Agent, RAG and multi-user sessions.")
    ap.add_argument("--agent", action="store_true", help="run only the Agent + RAG questions")
    ap.add_argument("--sessions", action="store_true", help="run only the multi-user session tests")
    ap.add_argument("--only", default="", help="comma-separated test IDs, e.g. T01,T19")
    ap.add_argument("--verbose", action="store_true", help="show answers and failure reasons")
    sys.exit(asyncio.run(main(ap.parse_args())))
