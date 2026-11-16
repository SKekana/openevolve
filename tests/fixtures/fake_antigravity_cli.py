"""Offline AGY protocol fixture; records whether stdin was actually received."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

args = sys.argv[1:]
mode = os.environ.get("OPENEVOLVE_FAKE_AGY_MODE", "success")
records = Path(os.environ["OPENEVOLVE_FAKE_AGY_RECORDS"])
record = {
    "args": args,
    "cwd": os.getcwd(),
    "pid": os.getpid(),
    "prompt": None,
    "agent": (Path.cwd() / ".agents/agents/openevolve-generation/agent.md").read_text(),
}
record_path = records / f"{os.getpid()}.json"


def save():
    record_path.write_text(json.dumps(record))


def emit(event):
    print(json.dumps(event), flush=True)


save()
if mode == "bad_options":
    print("unknown flag", file=sys.stderr)
    sys.exit(2)
if mode == "auth":
    print("authentication required", file=sys.stderr)
    sys.exit(1)
if mode == "fatal":
    print('AGY_ERROR: {"retryable": false, "message": "account denied"}', file=sys.stderr)
    sys.exit(3)
if mode == "child":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    record["child_pid"] = child.pid
    save()
if mode in ("sleep", "child"):
    time.sleep(60)
if mode == "nonzero" or (mode == "retry" and len(list(records.glob("*.json"))) == 1):
    print('AGY_ERROR: {"retryable": true, "message": "temporary service failure"}', file=sys.stderr)
    sys.exit(3)
init = {
    "agent": args[args.index("--agent") + 1],
    "model": args[args.index("--model") + 1],
    "cwd": os.getcwd(),
    "tools": [],
}
if mode == "tools":
    init["tools"] = ["run_command"]
if mode == "missing_tools":
    del init["tools"]
if mode == "wrong_agent":
    init["agent"] = "default"
if mode == "wrong_model":
    init["model"] = "other-model"
if mode == "wrong_cwd":
    init["cwd"] = "/"
if mode == "finish":
    init["tools"] = ["finish"]
emit({"event": "init", "init": init})
record["prompt"] = sys.stdin.read()
save()
if mode == "stderr":
    print("diagnostic " * 200000, file=sys.stderr, flush=True)
if mode == "post_sleep":
    time.sleep(60)
if mode == "malformed":
    print("not JSON", flush=True)
    sys.exit(0)
if mode == "tool_call":
    emit({"event": "step_update", "step_update": {"step_type": "tool", "tool_name": "run_command"}})
    time.sleep(60)
if mode == "missing_result":
    sys.exit(0)
response = os.environ.get("OPENEVOLVE_FAKE_AGY_RESPONSE", "generated code\n")
if mode == "echo":
    content = json.loads(record["prompt"])["message"]["content"]
    response = json.loads(content.split("\nConversation:\n", 1)[1])[-1]["content"]
result = {
    "status": "SUCCESS",
    "response": response,
    "usage": {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "thinking_tokens": 10,
        "cache_read_tokens": 60,
    },
}
if mode == "empty":
    result["response"] = "  \n"
if mode == "failed":
    result.update(status="ERROR", error="model unavailable")
if mode in ("CANCELED", "INTERRUPTED", "INVALID", "WAITING", "RUNNING"):
    result["status"] = mode
if mode == "bad_usage":
    result["usage"]["input_tokens"] = True
if mode == "no_usage":
    del result["usage"]
emit(
    {
        "event": "step_update",
        "step_update": {"step_type": "agent_response", "text_delta": "not final"},
    }
)
emit({"event": "future-event"})
emit({"event": "result", "result": result})
if mode == "duplicate":
    emit({"event": "result", "result": result})
if mode == "exit_after_success":
    sys.exit(3)
