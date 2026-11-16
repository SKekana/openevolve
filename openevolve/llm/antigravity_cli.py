"""Generation-only Antigravity CLI provider using saved CLI authentication.

The CLI must advertise a restricted custom agent before receiving any prompt.
AGY 1.2.16 currently advertises unrestricted tools and is rejected by this check.
Sampling parameters are not forwarded; OpenEvolve still seeds model selection.
"""

import asyncio
import json
import logging
import math
import os
import shutil
import signal
import tempfile
from pathlib import Path

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)
AGENT_NAME = "openevolve-generation"
VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")
GENERATION_INSTRUCTIONS = (
    "You are the code generation backend for OpenEvolve. Return the next assistant "
    "response to the supplied conversation in its requested format (code, "
    "SEARCH/REPLACE diffs, or evaluation JSON). OpenEvolve applies and evaluates "
    "your response. Do not use tools, inspect files, edit files, run commands, "
    "or delegate work."
)
AGENT_DEFINITION = f"""---
name: {AGENT_NAME}
description: Return candidate code for OpenEvolve to apply and evaluate.
tools: []
mainAgent: true
subagent: false
commandExecutionPolicy: "off"
mcpServers: []
skills: []
plugins: []
---
{GENERATION_INSTRUCTIONS}
"""


class AntigravityCLICompatibilityError(ValueError):
    """The CLI cannot confirm generation-only execution; do not retry."""


def _value(config, key, default):
    value = getattr(config, key, None)
    return default if value is None else value


class AntigravityCLILLM(LLMInterface):
    """Generate through AGY's guarded, single-turn stream-json protocol."""

    def __init__(self, model_cfg=None):
        self.model = getattr(model_cfg, "name", None)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Antigravity CLI requires an explicit model name")
        self.system_message = getattr(model_cfg, "system_message", None)
        self.weight = _value(model_cfg, "weight", 1.0)
        self.timeout = _value(model_cfg, "timeout", 300)
        self.retries = _value(model_cfg, "retries", 3)
        self.retry_delay = _value(model_cfg, "retry_delay", 5)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.last_usage = None
        self._validate_options(self.timeout, self.retries, self.retry_delay, self.reasoning_effort)

    @staticmethod
    def _validate_options(timeout, retries, delay, effort):
        for name, value, zero_ok in (("timeout", timeout, False), ("retry_delay", delay, True)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and not zero_ok)
            ):
                raise ValueError(f"Invalid Antigravity CLI {name}: {value}")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("Antigravity CLI retries must be a nonnegative integer")
        if effort is not None and effort not in VALID_EFFORTS:
            raise ValueError(f"Invalid Antigravity reasoning_effort: {effort}")

    async def generate(self, prompt: str, **kwargs) -> str:
        system = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system, [{"role": "user", "content": prompt}], **kwargs
        )

    async def generate_with_context(self, system_message, messages, **kwargs) -> str:
        self.last_usage = None
        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        delay = kwargs.get("retry_delay", self.retry_delay)
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        self._validate_options(timeout, retries, delay, effort)
        executable = shutil.which("agy")
        if executable is None:
            raise FileNotFoundError(
                "Install Antigravity CLI (`agy`) and authenticate in an interactive agy session"
            )
        conversation = [{"role": "system", "content": system_message or ""}, *messages]
        prompt = (
            GENERATION_INSTRUCTIONS
            + "\nConversation:\n"
            + json.dumps(conversation, ensure_ascii=False)
        )
        data = (
            json.dumps({"event": "user", "message": {"content": prompt}}, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        for attempt in range(retries + 1):
            try:
                response, usage = await self._run_cli(executable, data, timeout, effort)
                self.last_usage = usage
                if usage is not None:
                    logger.info(
                        "LLM Token Usage (%s): prompt_tokens=%s, completion_tokens=%s, total_tokens=%s",
                        self.model,
                        usage["prompt_tokens"],
                        usage["completion_tokens"],
                        usage["total_tokens"],
                    )
                return response
            except (RuntimeError, asyncio.TimeoutError) as exc:
                if attempt == retries:
                    raise
                logger.warning(
                    "Antigravity CLI (%s) attempt %s/%s failed: %s. Retrying...",
                    self.model,
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                await asyncio.sleep(delay)

    def _build_command(self, executable, effort):
        command = [
            executable,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--model",
            self.model,
            "--agent",
            AGENT_NAME,
            "--disable-slash-commands",
            "--sandbox",
            "--print-timeout",
            "0s",
        ]
        if effort is not None:
            command.extend(["--effort", effort])
        return command

    async def _run_cli(self, executable, prompt, timeout, effort):
        with tempfile.TemporaryDirectory(prefix="openevolve-agy-") as workdir:
            definition = Path(workdir) / ".agents" / "agents" / AGENT_NAME / "agent.md"
            definition.parent.mkdir(parents=True)
            definition.write_text(AGENT_DEFINITION, encoding="utf-8")
            process = await asyncio.create_subprocess_exec(
                *self._build_command(executable, effort),
                cwd=workdir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=(os.name == "posix"),
                limit=16 * 1024 * 1024,
            )
            diagnostics = asyncio.create_task(process.stderr.read())
            exchange = asyncio.create_task(self._exchange(process, prompt, diagnostics, workdir))
            try:
                return await asyncio.wait_for(asyncio.shield(exchange), timeout)
            except BaseException:
                await asyncio.shield(self._stop_process(process, exchange, diagnostics))
                raise

    @staticmethod
    async def _stop_process(process, exchange, diagnostics):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        await asyncio.gather(exchange, return_exceptions=True)
        # Parsing failures can leave unread stdout; drain it before waiting.
        await process.stdout.read()
        await asyncio.gather(diagnostics, return_exceptions=True)
        await process.wait()

    @staticmethod
    def _event(line):
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError("Malformed JSON event from Antigravity CLI") from exc
        if not isinstance(event, dict):
            raise RuntimeError("Malformed JSON event from Antigravity CLI: expected an object")
        return event

    def _validate_init(self, event, workdir):
        init = event.get("init")
        if (
            event.get("event") != "init"
            or not isinstance(init, dict)
            or init.get("agent") != AGENT_NAME
            or init.get("model") != self.model
            or not isinstance(init.get("cwd"), str)
            or Path(init["cwd"]).resolve() != Path(workdir).resolve()
        ):
            raise AntigravityCLICompatibilityError(
                "Antigravity CLI did not confirm the requested generation-only agent, model, and workspace; no prompt was sent"
            )
        tools = init.get("tools")
        if not isinstance(tools, list) or any(
            not isinstance(tool, str) or tool != "finish" for tool in tools
        ):
            raise AntigravityCLICompatibilityError(
                "Antigravity CLI advertises unrestricted or unverified tools for the generation-only agent; "
                "no prompt was sent. AGY 1.2.16 exhibits this limitation. Use an AGY version that honors "
                "custom-agent tool restrictions and reports them in init.tools, or use codex_cli."
            )

    async def _exchange(self, process, prompt, diagnostics, workdir):
        line = await process.stdout.readline()
        event = self._event(line) if line else {}
        if event.get("event") != "init":
            # Startup errors can be returned without an initialization event.
            process.stdin.close()
            await process.stdout.read()
            await process.wait()
            self._raise_failure(process.returncode, event.get("result", event), await diagnostics)
        self._validate_init(event, workdir)
        try:
            process.stdin.write(prompt)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # Read the terminal error and stderr even if AGY exits during input.
            pass
        finally:
            process.stdin.close()
        result = None
        while line := await process.stdout.readline():
            if not line.strip():
                continue
            event = self._event(line)
            kind = event.get("event")
            if kind == "init":
                raise RuntimeError("Unexpected duplicate Antigravity initialization")
            if kind == "step_update":
                step = event.get("step_update", {})
                if not isinstance(step, dict):
                    raise RuntimeError("Malformed Antigravity step update")
                if step.get("step_type") == "tool" and step.get("tool_name") != "finish":
                    raise AntigravityCLICompatibilityError(
                        "Antigravity CLI attempted a tool call despite its generation-only initialization"
                    )
            if kind == "result":
                if result is not None:
                    raise RuntimeError("Duplicate Antigravity terminal result")
                result = event.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("Malformed Antigravity terminal result")
        await process.wait()
        stderr = await diagnostics
        self._raise_failure(process.returncode, result, stderr)
        response = result.get("response")
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("Empty response from Antigravity CLI")
        return response, self._usage(result.get("usage"))

    @staticmethod
    def _raise_failure(returncode, result, stderr):
        detail = str(result.get("error", "") if isinstance(result, dict) else "")
        diagnostic = stderr.decode("utf-8", errors="replace")
        retryable = None
        structured_error = False
        for line in diagnostic.splitlines():
            if line.startswith("AGY_ERROR:"):
                try:
                    error = json.loads(line.partition(":")[2])
                    if isinstance(error, dict):
                        structured_error = True
                        retryable = error.get("retryable")
                        detail = str(error.get("message", error))
                except ValueError:
                    pass
        if (
            returncode == 0
            and isinstance(result, dict)
            and result.get("status") == "SUCCESS"
            and not structured_error
        ):
            return
        status = (
            result.get("status", "missing result") if isinstance(result, dict) else "missing result"
        )
        message = f"Antigravity CLI failed (exit {returncode}, status {status}): {detail[:1000]} {diagnostic[:1000]}"
        fatal = (
            "authentication required",
            "unauthenticated",
            "invalid model selection",
            "unknown model",
            "unknown agent",
            "agent not found",
            "flag provided but not defined",
            "unknown flag",
            "invalid value",
            "invalid argument",
        )
        if returncode == 2 or retryable is False or any(s in message.lower() for s in fatal):
            raise ValueError(
                message + " Check `agy --help`, `agy models`, and interactive authentication."
            )
        raise RuntimeError(message)

    def _usage(self, raw):
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise RuntimeError("Malformed token usage from Antigravity CLI")
        counts = {key: raw.get(key) for key in ("input_tokens", "output_tokens")}
        counts.update({key: raw.get(key, 0) for key in ("thinking_tokens", "cache_read_tokens")})
        if "total_tokens" in raw:
            counts["total_tokens"] = raw["total_tokens"]
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts.values()):
            raise RuntimeError("Malformed token usage from Antigravity CLI")
        return {
            "model": self.model,
            "prompt_tokens": counts["input_tokens"],
            "completion_tokens": counts["output_tokens"],
            "total_tokens": counts.get(
                "total_tokens", counts["input_tokens"] + counts["output_tokens"]
            ),
            "thinking_tokens": counts["thinking_tokens"],
            "cached_input_tokens": counts["cache_read_tokens"],
        }


def init_antigravity_cli_client(model_cfg):
    """Factory for the existing init_client configuration hook."""
    return AntigravityCLILLM(model_cfg)
