"""
Follow one request through the Agent using the JSON-lines event log.

    python trace_request.py --recent 10          # last 10 requests
    python trace_request.py <request_id>         # full timeline of one request
    python trace_request.py <request_id> --stack # include stack traces (protected log!)

Needs LOG_FILE_ENABLED=true (events go to logs/itsm_events.jsonl). The id of a
REST request is returned in the X-Request-ID header (and in the body of a 500).
"""
import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings


def read_events(path: Path):
    files = [path] + [path.with_name(f"{path.name}.{i}") for i in range(1, 20)]   # include rotated files
    for f in reversed([f for f in files if f.exists()]):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("request_id", nargs="?")
    ap.add_argument("--recent", type=int, default=0)
    ap.add_argument("--stack", action="store_true")
    args = ap.parse_args()
    path = Path(get_settings().LOG_DIR) / "itsm_events.jsonl"
    if not path.exists():
        print(f"No event log at {path}. Set LOG_FILE_ENABLED=true and restart the app.")
        return 1
    events = list(read_events(path))
    if args.recent:
        reqs = [e for e in events if e.get("component") == "request"][-args.recent:]
        for e in reqs:
            print(f"{e['timestamp']}  {e['request_id']}  {e['session_id']:14} {e['operation']:28} {e['status']:12} {e['duration_ms']:>9} ms")
        return 0
    if not args.request_id:
        ap.print_help(); return 1
    mine = [e for e in events if e.get("request_id") == args.request_id]
    if not mine:
        print("No events found for that request id."); return 1
    for e in sorted(mine, key=lambda x: x["timestamp"]):
        extra = {k: v for k, v in e.items() if k not in ("timestamp", "request_id", "session_id", "component", "operation", "status", "duration_ms", "level", "stack_trace")}
        print(f"{e['timestamp']}  {e['component']:8} {e['operation']:26} {e['status']:8} {e['duration_ms']:>8} ms  {json.dumps(extra) if extra else ''}")
        if args.stack and e.get("stack_trace"):
            print(e["stack_trace"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
