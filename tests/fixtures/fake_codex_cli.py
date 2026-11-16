"""Offline executable fixture for the Codex provider's subprocess tests."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

args = sys.argv[1:]
record_dir = Path(os.environ["OPENEVOLVE_FAKE_CODEX_RECORDS"])
mode = os.environ.get("OPENEVOLVE_FAKE_CODEX_MODE", "success")
prompt = sys.stdin.read()
record = {"args": args, "prompt": prompt, "cwd": os.getcwd(), "pid": os.getpid()}
if mode == "child":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    record["child_pid"] = child.pid
record_path = record_dir / f"{os.getpid()}.json"
record_path.write_text(json.dumps(record), encoding="utf-8")

if mode in ("sleep", "child"):
    time.sleep(60)
if mode == "nonzero" or (mode == "retry" and len(list(record_dir.glob("*.json"))) == 1):
    print("temporary service failure", file=sys.stderr)
    sys.exit(1)
if mode == "bad_options":
    print("unexpected argument '--ignore-user-config'", file=sys.stderr)
    sys.exit(2)
if mode == "json_error_exit":
    print(json.dumps({"type": "turn.failed", "error": {"message": "account model access denied"}}))
    sys.exit(1)

output_path = Path(args[args.index("--output-last-message") + 1])
response = os.environ.get("OPENEVOLVE_FAKE_CODEX_RESPONSE", "generated code\n")
if mode == "echo":
    response = json.loads(prompt.split("\n", 1)[1])[-1]["content"]
if mode != "missing_response":
    output_path.write_text("  \n" if mode == "empty_response" else response, encoding="utf-8")

print(
    json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "commentary, not final code"},
        }
    )
)
print(json.dumps({"type": "future.event", "extra": True}))
if mode == "malformed":
    print("not JSON")
elif mode == "failed_turn":
    print(json.dumps({"type": "turn.failed", "error": {"message": "model unavailable"}}))
elif mode == "error":
    print(json.dumps({"type": "error", "message": "request failed"}))
elif mode != "missing_completion":
    event = {"type": "turn.completed"}
    if mode != "no_usage":
        event["usage"] = {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 60}
    if mode == "bad_usage":
        event["usage"]["input_tokens"] = "invalid"
    print(json.dumps(event))
