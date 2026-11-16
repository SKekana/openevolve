"""Offline executable fixture for OpenCode configuration and JSON-event tests."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

args = sys.argv[1:]
record_dir = Path(os.environ["OPENEVOLVE_FAKE_OPENCODE_RECORDS"])
mode = os.environ.get("OPENEVOLVE_FAKE_OPENCODE_MODE", "success")
is_config = "debug" in args and "config" in args
config_content = os.environ.get("OPENCODE_CONFIG_CONTENT")
inline_config = json.loads(config_content) if config_content else {}
is_generation_config = "openevolve-generation" in inline_config.get("agent", {})
phase = (
    "verification" if is_config and is_generation_config else "discovery" if is_config else "run"
)
prompt = "" if is_config else sys.stdin.read()
record = {
    "args": args,
    "prompt": prompt,
    "cwd": os.getcwd(),
    "pid": os.getpid(),
    "phase": phase,
    "env": {
        key: value
        for key, value in os.environ.items()
        if key.startswith("OPENCODE_") or key in ("HOME", "PWD", "XDG_CONFIG_HOME", "XDG_DATA_HOME")
    },
}
sleep_here = (phase == "run" and mode in ("sleep", "child")) or mode == f"{phase}_sleep"
if sleep_here and mode == "child":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    record["child_pid"] = child.pid
record_path = record_dir / f"{os.getpid()}.json"
temporary_record = record_path.with_suffix(".tmp")
temporary_record.write_text(json.dumps(record), encoding="utf-8")
temporary_record.replace(record_path)

if sleep_here:
    time.sleep(60)
if mode == f"{phase}_bad_options" or (mode == "bad_options" and phase == "run"):
    print("unknown option '--pure'", file=sys.stderr)
    sys.exit(2)

if is_config:
    if phase == "discovery":
        config = {
            "provider": {
                "modal": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Modal",
                    "options": {
                        "baseURL": "https://example.invalid/v1",
                        "apiKey": "private-config-secret",
                    },
                    "models": {"model/with/slashes": {"name": "Custom model"}},
                }
            },
            "enabled_providers": ["modal", "anthropic"],
            "disabled_providers": ["unwanted"],
            "mcp": {"unsafe": {"type": "local", "command": ["false"]}},
            "plugin": ["unsafe-plugin"],
            "instructions": ["unsafe-instructions.md"],
            "permission": {"*": "allow"},
            "agent": {"unsafe-agent": {"permission": {"*": "allow"}}},
            "share": "auto",
        }
        config["provider"].update(inline_config.get("provider", {}))
        config.update({key: value for key, value in inline_config.items() if key != "provider"})
        if mode == "discovery_malformed":
            print("not json")
            sys.exit(0)
    else:
        config = json.loads(config_content)
        if mode == "isolation_failure":
            config["permission"] = {"*": "allow"}
        elif mode == "malformed_agent":
            config["agent"] = []
        elif mode == "malformed_agent_entry":
            config["agent"]["openevolve-generation"] = []
        elif mode == "disabled_agent":
            config["agent"]["openevolve-generation"]["disable"] = True
        elif mode == "overridden_sampling":
            config["agent"]["openevolve-generation"]["temperature"] = 999
        elif mode == "inherited_auto_compaction":
            config["compaction"]["auto"] = True
        elif mode == "verification_malformed":
            print("not json")
            sys.exit(0)
    print(json.dumps(config))
    sys.exit(0)

run_records = [
    json.loads(path.read_text())
    for path in record_dir.glob("*.json")
    if json.loads(path.read_text())["phase"] == "run"
]
if mode == "nonzero" or (mode == "retry" and len(run_records) == 1):
    print("temporary service failure", file=sys.stderr)
    sys.exit(1)
if mode in ("error", "json_error_exit", "secret_error"):
    detail = "request failed: private-config-secret" if mode == "secret_error" else "request failed"
    print(json.dumps({"type": "error", "error": {"name": "APIError", "data": {"message": detail}}}))
    sys.exit(1 if mode == "json_error_exit" else 0)
permanent_error_modes = {
    "auth_error": {"name": "AuthError", "data": {"message": "authentication required"}},
    "missing_model": {
        "name": "ProviderModelNotFoundError",
        "data": {"message": "model not found"},
    },
    "api_nonretryable": {
        "name": "APIError",
        "data": {"message": "invalid API request", "isRetryable": False},
    },
}
if mode in permanent_error_modes or mode.startswith("api_status_"):
    error = permanent_error_modes.get(mode)
    if error is None:
        error = {
            "name": "APIError",
            "data": {"message": "API request failed", "statusCode": int(mode.rsplit("_", 1)[1])},
        }
    print(json.dumps({"type": "error", "error": error}))
    sys.exit(0)
if mode == "malformed":
    print("not JSON")
    sys.exit(0)
if mode == "nonobject":
    print("[]")
    sys.exit(0)

response = os.environ.get("OPENEVOLVE_FAKE_OPENCODE_RESPONSE", "generated code\n")
if mode == "echo":
    response = json.loads(prompt[prompt.index("[") :])[-1]["content"]
if mode == "empty_response":
    response = "  \n"

print(json.dumps({"type": "future.event", "extra": True}))
print(json.dumps({"type": "reasoning", "part": {"type": "reasoning", "text": "private reasoning"}}))
print(
    json.dumps(
        {
            "type": "text",
            "part": {"id": "partial", "type": "text", "text": "unfinished", "time": {"start": 1}},
        }
    )
)
if mode == "tool_use":
    print(json.dumps({"type": "tool_use", "part": {"id": "tool1", "type": "tool", "tool": "bash"}}))
if mode not in ("no_text", "partial_only"):
    chunks = [response[: len(response) // 2], response[len(response) // 2 :]]
    for index, chunk in enumerate(chunks):
        event = {
            "type": "text",
            "part": {
                "id": f"text-{index}",
                "type": "text",
                "text": chunk,
                "time": {"start": 1, "end": 2},
            },
        }
        if mode == "missing_text_id":
            del event["part"]["id"]
        elif mode == "bad_text":
            event["part"]["text"] = 4
        print(json.dumps(event))
        if mode == "duplicate":
            print(json.dumps(event))
        elif mode == "conflicting_text":
            event["part"]["text"] = "conflicting content"
            print(json.dumps(event))
if mode != "missing_completion":
    part = {
        "id": "finish-1",
        "type": "step-finish",
        "reason": (
            "length" if mode == "truncated" else "error" if mode == "finish_error" else "stop"
        ),
    }
    if mode != "no_usage":
        part["tokens"] = {
            "input": "invalid" if mode == "bad_usage" else 100,
            "output": 20,
            "reasoning": 5,
            "cache": {"read": 60, "write": 10},
        }
        if mode == "bad_cache":
            part["tokens"]["cache"]["read"] = "invalid"
        elif mode == "negative_usage":
            part["tokens"]["input"] = -1
        elif mode == "float_usage":
            part["tokens"]["input"] = 1.1
        elif mode == "bool_usage":
            part["tokens"]["input"] = True
        elif mode == "partial_usage":
            part["tokens"] = {"input": 3, "output": 2}
    if mode == "missing_reason":
        del part["reason"]
    print(json.dumps({"type": "step_finish", "part": part}))
    if mode == "duplicate_finish":
        print(json.dumps({"type": "step_finish", "part": part}))
