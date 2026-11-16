"""Offline ACP process fixture: record requests, emulate interleaved updates."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

mode = os.environ.get("OPENEVOLVE_FAKE_ACP_MODE", "success")
records = Path(os.environ["OPENEVOLVE_FAKE_ACP_RECORDS"])
record_path = records / f"{os.getpid()}.json"
record = {
    "pid": os.getpid(),
    "cwd": os.getcwd(),
    "home": os.environ["GEMINI_HOME"],
    "requests": [],
    "replies": [],
    "args": sys.argv[1:],
    "browser": os.environ["BROWSER"],
}
session_id = str(uuid4())


def save():
    record_path.write_text(json.dumps(record))


def emit(event):
    print(json.dumps({"jsonrpc": "2.0", **event}), flush=True)


def update(kind, **kwargs):
    emit(
        {
            "method": "session/update",
            "params": {
                "sessionId": "other" if mode == "wrong_session" else session_id,
                "update": {"sessionUpdate": kind, **kwargs},
            },
        }
    )


def options(value):
    return [
        {
            "id": "model",
            "currentValue": value,
            "options": [{"value": "gemini-3.8-flash-medium", "name": "Flash Medium"}],
        }
    ]


save()
if mode in ("child", "child_exit"):
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    record["child_pid"] = child.pid
    save()

for raw in sys.stdin:
    request = json.loads(raw)
    if "method" not in request:
        record["replies"].append(request)
        save()
        continue
    record["requests"].append(request)
    save()
    method, params, request_id = request["method"], request["params"], request["id"]
    result = {}
    if method == "initialize":
        if mode == "sleep":
            time.sleep(60)
        result = {
            "protocolVersion": 1,
            "agentInfo": {
                "name": "antigravity-acp",
                "version": "2.0" if mode == "version" else "1.3.0",
            },
            "authMethods": [{"id": "oauth-personal"}],
        }
        if mode == "protocol":
            result["protocolVersion"] = 99
    elif method == "authenticate":
        if mode == "browser":
            print("OPENEVOLVE_ACP_AUTH_REQUIRED", file=sys.stderr, flush=True)
            time.sleep(60)
        if mode == "auth":
            emit(
                {"id": request_id, "error": {"code": -32000, "message": "Authentication required"}}
            )
            continue
    elif method == "session/new":
        sidecar = (
            Path(os.environ["GEMINI_HOME"]) / "antigravity-acp/conversations" / f"{session_id}.meta"
        )
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        persisted = {"tool_filter": params["_meta"]["agy"]}
        if mode == "filter":
            persisted["tool_filter"] = {"enabledTools": ["run_command"]}
        if mode != "missing_filter":
            sidecar.write_text(json.dumps(persisted))
        result = {"sessionId": session_id, "configOptions": options("other")}
        if mode == "catalog":
            result["configOptions"][0]["options"] = []
    elif method == "session/set_config_option":
        update("available_commands_update", availableCommands=[])
        result = {"configOptions": options("other" if mode == "model" else params["value"])}
    elif method == "session/prompt":
        if mode in ("timeout", "child"):
            time.sleep(60)
        if mode == "malformed":
            print("not JSON", flush=True)
            continue
        if mode == "id":
            emit({"id": request_id + 1, "result": {}})
            continue
        if mode in ("retry", "model_retry") and len(list(records.glob("*.json"))) == 1:
            emit(
                {
                    "id": request_id,
                    "error": {
                        "code": -32099,
                        "message": (
                            "Model temporarily unavailable"
                            if mode == "model_retry"
                            else "Service temporarily unavailable"
                        ),
                    },
                }
            )
            continue
        if mode == "stderr":
            print("diagnostic" * 2000000, file=sys.stderr, flush=True)
        if mode == "tool":
            update("tool_call", title="Running run_command", toolCallId="tool")
            continue
        if mode in ("permission", "fs", "terminal"):
            server_method = {
                "permission": "session/request_permission",
                "fs": "fs/read_text_file",
                "terminal": "terminal/create",
            }[mode]
            emit(
                {
                    "id": "server-operation",
                    "method": server_method,
                    "params": {"sessionId": session_id},
                }
            )
            continue
        update("agent_thought_chunk", content={"type": "text", "text": "private thought"})
        response = os.environ.get("OPENEVOLVE_FAKE_ACP_RESPONSE", "generated code\n")
        if mode == "empty":
            response = " \n"
        for text in (response[:5], response[5:]):
            update("agent_message_chunk", content={"type": "text", "text": text})
        update("usage_update", used=150, size=1048576)
        result = {"stopReason": "max_tokens" if mode == "incomplete" else "end_turn"}
    emit({"id": request_id, "result": result})
    if method == "session/prompt" and mode in ("nonzero", "child_exit"):
        sys.exit(3 if mode == "nonzero" else 0)
