"""Generation-only provider for Google's pinned Antigravity ACP server."""

import asyncio
import json
import logging
import math
import os
import re
import shlex
import signal
import sys
import tempfile
from pathlib import Path
from uuid import UUID

from openevolve.llm.antigravity_cli import GENERATION_INSTRUCTIONS
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)
SUPPORTED_VERSION = "1.3.0"
DEFAULT_HOME = Path.home() / ".local/share/openevolve/antigravity-acp"
DEFAULT_BINARY = (
    Path.home()
    / ".local/share/openevolve/antigravity-acp-server"
    / SUPPORTED_VERSION
    / "agy_acp_server.par"
)
TOOL_FILTER = {"enabledTools": ["finish"]}


class AntigravityACPCompatibilityError(ValueError):
    """Authentication, configuration, or tool restrictions cannot be confirmed."""


def _option(config, key, default):
    value = getattr(config, key, None)
    return default if value is None else value


def resolve_paths(config=None):
    """Explicit configuration wins over environment and installation defaults."""
    executable = _option(
        config,
        "antigravity_acp_path",
        os.environ.get("OPENEVOLVE_ANTIGRAVITY_ACP_BIN", str(DEFAULT_BINARY)),
    )
    home = _option(
        config,
        "antigravity_acp_home",
        os.environ.get("OPENEVOLVE_ANTIGRAVITY_ACP_HOME", str(DEFAULT_HOME)),
    )
    return str(Path(executable).expanduser().resolve()), Path(home).expanduser().resolve()


def validate_profile(home, require_auth=True):
    """Keep profile hooks, MCP, and skills out of the generation-only harness."""
    for relative in (
        "config/hooks.json",
        "config/mcp_config.json",
        "config/skills",
        "antigravity-cli/skills",
    ):
        path = home / relative
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise AntigravityACPCompatibilityError(
                f"Antigravity ACP requires a dedicated profile without customizations: {path}"
            )
    if require_auth and not (home / "antigravity-acp/acp_token.json").is_file():
        raise AntigravityACPCompatibilityError(
            "Antigravity ACP personal OAuth login is missing. Run "
            "`python -m openevolve.antigravity_acp login` with this profile first."
        )


class _Connection:
    """Sequential JSON-RPC requests with interleaved server notifications."""

    def __init__(self, process, allow_login=False):
        self.process = process
        self.allow_login = allow_login
        self.next_id = 0
        self.session_id = None
        self.chunks = []
        self.response_size = 0
        self.context_usage = None
        self.auth_required = asyncio.Event()
        self.forced_shutdown = False
        self.diagnostics = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self):
        # Drain continuously without retaining token-bearing URLs or unbounded logs.
        pending = b""
        while chunk := await self.process.stderr.read(8192):
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                self._authentication_notice(line)
            if len(pending) > 65536:
                self._authentication_notice(pending)
                pending = pending[-65536:]
        self._authentication_notice(pending)

    def _authentication_notice(self, line):
        if b"OPENEVOLVE_ACP_AUTH_REQUIRED" in line or b"Open the following link" in line:
            if self.allow_login:
                print(line.decode(errors="replace").rstrip(), file=sys.stderr, flush=True)
            else:
                self.auth_required.set()

    async def _write(self, event):
        try:
            self.process.stdin.write((json.dumps(event, ensure_ascii=False) + "\n").encode())
            await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise RuntimeError("Antigravity ACP closed its input during a request") from exc

    async def _readline(self):
        reading = asyncio.create_task(self.process.stdout.readline())
        expired = asyncio.create_task(self.auth_required.wait())
        try:
            await asyncio.wait((reading, expired), return_when=asyncio.FIRST_COMPLETED)
            if self.auth_required.is_set():
                raise AntigravityACPCompatibilityError(
                    "ACP credentials need browser sign-in. Run "
                    "`python -m openevolve.antigravity_acp login` before evolution."
                )
            return await reading
        finally:
            for task in (reading, expired):
                if not task.done():
                    task.cancel()
            await asyncio.gather(reading, expired, return_exceptions=True)

    async def rpc(self, method, params):
        self.next_id += 1
        request_id = self.next_id
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            raw = await self._readline()
            if not raw:
                raise RuntimeError("Antigravity ACP exited before completing its response")
            try:
                event = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise AntigravityACPCompatibilityError("Malformed ACP JSON-RPC output") from exc
            if not isinstance(event, dict) or event.get("jsonrpc") != "2.0":
                raise AntigravityACPCompatibilityError("Malformed ACP JSON-RPC envelope")
            if "method" in event:
                if "id" in event:
                    result = {"outcome": {"outcome": "cancelled"}}
                    response = {"jsonrpc": "2.0", "id": event["id"]}
                    if event["method"] == "session/request_permission":
                        response["result"] = result
                    else:
                        response["error"] = {"code": -32601, "message": "Client tools disabled"}
                    try:
                        await self._write(response)
                    except RuntimeError:
                        pass
                    raise AntigravityACPCompatibilityError(
                        "Antigravity ACP requested a disabled client operation"
                    )
                self._notification(event)
                continue
            if type(event.get("id")) is not int or event["id"] != request_id:
                raise AntigravityACPCompatibilityError("Unexpected ACP response ID")
            if ("error" in event) == ("result" in event):
                raise AntigravityACPCompatibilityError("Malformed ACP response envelope")
            if "error" in event:
                error = event["error"]
                if not isinstance(error, dict):
                    raise AntigravityACPCompatibilityError("Malformed ACP error")
                # Do not expose OAuth URLs or credential-bearing server diagnostics.
                detail = re.sub(r"https?://\S+", "<URL>", str(error.get("message", "")))[:500]
                code = error.get("code")
                message = f"Antigravity ACP {method} failed ({code}): {detail}"
                fatal = (
                    "authentication required",
                    "unauthenticated",
                    "invalid model",
                    "unknown model",
                    "model not found",
                    "permission denied",
                    "invalid param",
                    "onboarding failed",
                    "access denied",
                    "not entitled",
                )
                if code in (-32000, -32002, -32600, -32601, -32602) or any(
                    word in detail.lower() for word in fatal
                ):
                    raise AntigravityACPCompatibilityError(message)
                raise RuntimeError(message)
            result = event.get("result")
            if not isinstance(result, dict):
                raise AntigravityACPCompatibilityError("Malformed ACP result")
            return result

    def _notification(self, event):
        if event["method"] != "session/update":
            return
        params = event.get("params", {})
        if not isinstance(params, dict) or params.get("sessionId") != self.session_id:
            raise AntigravityACPCompatibilityError("Unexpected ACP session update")
        update = params.get("update")
        if not isinstance(update, dict):
            raise AntigravityACPCompatibilityError("Malformed ACP session update")
        kind = update.get("sessionUpdate")
        if kind in ("tool_call", "tool_call_update"):
            # This workflow does not need tool notifications, including finish.
            # Reject rather than guess which executable tool a display title means.
            raise AntigravityACPCompatibilityError("Unexpected ACP tool activity")
        if kind == "agent_message_chunk":
            content = update.get("content", {})
            if not isinstance(content, dict):
                raise AntigravityACPCompatibilityError("Malformed ACP message content")
            if content.get("type") == "text" and isinstance(content.get("text"), str):
                self.chunks.append(content["text"])
                self.response_size += len(content["text"])
                if self.response_size > 16 * 1024 * 1024:
                    raise RuntimeError("Antigravity ACP response exceeded 16 MiB")
        elif kind == "usage_update":
            used, size = update.get("used"), update.get("size")
            if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (used, size)):
                self.context_usage = {"used": used, "size": size}

    async def initialize(self):
        result = await self.rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {"name": "openevolve", "version": "0.1"},
            },
        )
        info = result.get("agentInfo", {})
        if (
            result.get("protocolVersion") != 1
            or not isinstance(info, dict)
            or info.get("name") != "antigravity-acp"
            or (info.get("version") != SUPPORTED_VERSION)
        ):
            raise AntigravityACPCompatibilityError(
                f"Requires Google's official Antigravity ACP {SUPPORTED_VERSION}; no prompt sent"
            )
        await self.rpc("authenticate", {"methodId": "oauth-personal"})

    async def close(self):
        self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), 2)
        except asyncio.TimeoutError:
            pass
        finally:
            # Kill the group even if the direct parent already exited.
            try:
                if os.name == "posix":
                    self.forced_shutdown = self.process.returncode is None
                    os.killpg(self.process.pid, signal.SIGKILL)
                elif self.process.returncode is None:
                    self.forced_shutdown = True
                    self.process.kill()
            except ProcessLookupError:
                pass
            await self.process.stdout.read()
            await asyncio.gather(self.diagnostics, return_exceptions=True)
            await self.process.wait()


async def start_server(executable, home, workdir, allow_login=False):
    if not Path(executable).is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            "Install the official ACP server with `python -m openevolve.antigravity_acp install` "
            "or set antigravity_acp_path"
        )
    if sys.platform.startswith("linux") and not os.access(
        Path(executable).parent / "localharness_external", os.X_OK
    ):
        raise FileNotFoundError("ACP installation requires localharness_external beside the server")
    env = os.environ.copy()
    env["GEMINI_HOME"] = str(home)
    helper = Path(__file__).with_name("_antigravity_browser.py")
    mode = "--open" if allow_login else "--block"
    env["BROWSER"] = f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))} {mode} %s"
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "AGY_LLM_GATEWAY_URL"):
        env.pop(key, None)
    args = [executable]
    if sys.platform.startswith("linux"):
        args.append("--uid=")
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=workdir,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=(os.name == "posix"),
        limit=16 * 1024 * 1024,
    )
    return _Connection(process, allow_login)


class AntigravityACPLLM(LLMInterface):
    """Return candidate code through a new, restricted ACP session per attempt."""

    def __init__(self, model_cfg=None):
        self.model = getattr(model_cfg, "name", None)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Antigravity ACP requires an explicit model name")
        self.system_message = getattr(model_cfg, "system_message", None)
        self.weight = _option(model_cfg, "weight", 1.0)
        self.timeout = _option(model_cfg, "timeout", 300)
        self.retries = _option(model_cfg, "retries", 2)
        self.retry_delay = _option(model_cfg, "retry_delay", 5)
        self.executable, self.home = resolve_paths(model_cfg)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.last_usage = None
        self.last_context_usage = None
        self._validate(self.timeout, self.retries, self.retry_delay, self.reasoning_effort)

    @staticmethod
    def _validate(timeout, retries, delay, effort):
        for key, value in (("timeout", timeout), ("retry_delay", delay)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
                or (key == "timeout" and value == 0)
            ):
                raise ValueError(f"Invalid Antigravity ACP {key}")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("Antigravity ACP retries must be a nonnegative integer")
        if effort is not None:
            raise ValueError(
                "Select ACP reasoning effort through the model name; omit reasoning_effort"
            )

    async def generate(self, prompt, **kwargs):
        system = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system, [{"role": "user", "content": prompt}], **kwargs
        )

    async def generate_with_context(self, system_message, messages, **kwargs):
        self.last_usage = self.last_context_usage = None
        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        delay = kwargs.get("retry_delay", self.retry_delay)
        self._validate(
            timeout, retries, delay, kwargs.get("reasoning_effort", self.reasoning_effort)
        )
        validate_profile(self.home)
        conversation = [{"role": "system", "content": system_message or ""}, *messages]
        prompt = (
            GENERATION_INSTRUCTIONS
            + "\nConversation:\n"
            + json.dumps(conversation, ensure_ascii=False)
        )
        for attempt in range(retries + 1):
            try:
                response, usage = await asyncio.wait_for(self._attempt(prompt), timeout)
                self.last_context_usage = usage
                return response
            except (RuntimeError, asyncio.TimeoutError) as exc:
                if attempt == retries:
                    raise
                logger.warning(
                    "Antigravity ACP (%s) attempt %s failed: %s", self.model, attempt + 1, exc
                )
                await asyncio.sleep(delay)

    def _verify_filter(self, session_id):
        try:
            if str(UUID(session_id)) != session_id:
                raise ValueError("Noncanonical session ID")
            meta = json.loads(
                (self.home / "antigravity-acp/conversations" / f"{session_id}.meta").read_text()
            )
            if meta.get("tool_filter") != TOOL_FILTER:
                raise ValueError("Tool filter differs")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise AntigravityACPCompatibilityError(
                "ACP did not persist the required tool filter; no prompt sent"
            ) from exc

    @staticmethod
    def _model_option(result):
        options = result.get("configOptions")
        if not isinstance(options, list) or any(not isinstance(x, dict) for x in options):
            raise AntigravityACPCompatibilityError("Malformed ACP model configuration")
        return next((option for option in options if option.get("id") == "model"), {})

    async def _attempt(self, prompt):
        with tempfile.TemporaryDirectory(prefix="openevolve-acp-") as workdir:
            connection = await start_server(self.executable, self.home, workdir)
            try:
                await connection.initialize()
                session = await connection.rpc(
                    "session/new",
                    {
                        "cwd": workdir,
                        "mcpServers": [],
                        "_meta": {"agy": TOOL_FILTER},
                    },
                )
                session_id = session.get("sessionId")
                self._verify_filter(session_id)
                connection.session_id = session_id
                model_option = self._model_option(session)
                choices = model_option.get("options")
                if not isinstance(choices, list) or any(not isinstance(x, dict) for x in choices):
                    raise AntigravityACPCompatibilityError("Malformed ACP model catalog")
                if not any(x.get("value") == self.model for x in choices):
                    raise AntigravityACPCompatibilityError(
                        f"ACP model is unavailable: {self.model}"
                    )
                selected = await connection.rpc(
                    "session/set_config_option",
                    {
                        "sessionId": session_id,
                        "configId": "model",
                        "value": self.model,
                    },
                )
                model_option = self._model_option(selected)
                if model_option.get("currentValue") != self.model:
                    raise AntigravityACPCompatibilityError("ACP did not confirm the selected model")
                self._verify_filter(session_id)
                result = await connection.rpc(
                    "session/prompt",
                    {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": prompt}],
                    },
                )
                if result.get("stopReason") != "end_turn":
                    raise RuntimeError(
                        f"Antigravity ACP incomplete turn: {result.get('stopReason')}"
                    )
                response = "".join(connection.chunks)
                if not response.strip():
                    raise RuntimeError("Antigravity ACP returned an empty response")
                if connection.process.returncode not in (None, 0):
                    raise RuntimeError("Antigravity ACP exited unsuccessfully")
            finally:
                await asyncio.shield(connection.close())
            forced_exit = -signal.SIGKILL if os.name == "posix" else 1
            if connection.process.returncode != 0 and not (
                connection.forced_shutdown and connection.process.returncode == forced_exit
            ):
                raise RuntimeError("Antigravity ACP exited unsuccessfully during shutdown")
            return response, connection.context_usage


def init_antigravity_acp_client(model_cfg):
    """Factory for programmatic client injection."""
    return AntigravityACPLLM(model_cfg)
