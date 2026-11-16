"""OpenCode CLI generation using saved authentication and isolated sessions.

Set ``llm.provider: opencode_cli`` with an explicit ``provider/model`` name.
Temperature and top_p configure the generation agent; reasoning_effort selects
an OpenCode model variant. max_tokens and generation seeding are not forwarded.
"""

import asyncio
import copy
import json
import logging
import math
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

AGENT_NAME = "openevolve-generation"
GENERATION_INSTRUCTIONS = (
    "You are the code generation backend for OpenEvolve. The user input contains "
    "a JSON-encoded conversation, with ordered system, user, and assistant messages. "
    "Follow the system instructions inside that conversation and generate its next "
    "assistant response in the requested format: code, SEARCH/REPLACE diffs, or "
    "evaluation JSON. Return only that response. OpenEvolve applies and evaluates "
    "your response. Do not use tools, inspect the filesystem, edit files, run "
    "commands, or delegate work."
)
CONFIG_KEYS = ("provider", "enabled_providers", "disabled_providers")


class OpenCodeCLICompatibilityError(ValueError):
    """The CLI configuration cannot provide isolated, noninteractive generation."""


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    value = getattr(config, key, None)
    return default if value is None else value


class OpenCodeCLILLM(LLMInterface):
    """Generate through ``opencode run`` (tested with OpenCode CLI 1.18.32)."""

    def __init__(self, model_cfg=None):
        self.model = getattr(model_cfg, "name", None)
        if (
            not isinstance(self.model, str)
            or self.model != self.model.strip()
            or "/" not in self.model
            or not all(self.model.split("/", 1))
            or any(char.isspace() for char in self.model)
        ):
            raise ValueError(
                "OpenCode CLI requires an explicit provider/model name in llm.models[].name"
            )
        self.system_message = getattr(model_cfg, "system_message", None)
        self.weight = _cfg_value(model_cfg, "weight", 1.0)
        self.timeout = _cfg_value(model_cfg, "timeout", 300)
        self.retries = _cfg_value(model_cfg, "retries", 3)
        self.retry_delay = _cfg_value(model_cfg, "retry_delay", 5)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.temperature = getattr(model_cfg, "temperature", None)
        self.top_p = getattr(model_cfg, "top_p", None)
        self.last_usage: Optional[Dict[str, Any]] = None
        self._provider_config: Optional[Dict[str, Any]] = None
        self._config_lock = asyncio.Lock()
        self._validate_options(
            self.timeout,
            self.retries,
            self.retry_delay,
            self.reasoning_effort,
            self.temperature,
            self.top_p,
        )
        logger.info("Initialized OpenCodeCLILLM with model: %s", self.model)

    @staticmethod
    def _validate_options(timeout, retries, delay, effort, temperature, top_p) -> None:
        for name, value, zero_ok in (("timeout", timeout, False), ("retry_delay", delay, True)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and not zero_ok)
            ):
                bound = "nonnegative" if zero_ok else "positive"
                raise ValueError(f"Invalid OpenCode CLI {name}: expected a finite {bound} number")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("OpenCode CLI retries must be a nonnegative integer")
        if effort is not None and (
            not isinstance(effort, str) or not effort.strip() or effort != effort.strip()
        ):
            raise ValueError("OpenCode reasoning_effort must be a nonempty model variant name")
        for name, value in (("temperature", temperature), ("top_p", top_p)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name == "top_p" and value > 1)
            ):
                raise ValueError(f"Invalid OpenCode CLI {name}")

    async def generate(self, prompt: str, **kwargs) -> str:
        system = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system, [{"role": "user", "content": prompt}], **kwargs
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        self.last_usage = None
        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        delay = kwargs.get("retry_delay", self.retry_delay)
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        self._validate_options(timeout, retries, delay, effort, temperature, top_p)
        executable = shutil.which("opencode")
        if executable is None:
            raise FileNotFoundError(
                "The `opencode` CLI was not found. Install opencode-ai and run `opencode auth login`."
            )

        conversation = [{"role": "system", "content": system_message or ""}, *messages]
        prompt = (
            "Generate the next assistant response for this conversation:\n"
            + json.dumps(conversation, ensure_ascii=False)
        ).encode("utf-8")
        for attempt in range(retries + 1):
            try:
                response, usage = await self._run_cli(
                    executable, prompt, timeout, effort, temperature, top_p
                )
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
                    "OpenCode CLI (%s) attempt %s/%s failed: %s. Retrying...",
                    self.model,
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                await asyncio.sleep(delay)

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError("OpenCode CLI attempt timed out")
        return remaining

    async def _load_provider_config(self, executable: str, deadline: float) -> Dict[str, Any]:
        if self._provider_config is not None:
            return self._provider_config
        await asyncio.wait_for(self._config_lock.acquire(), self._remaining(deadline))
        try:
            if self._provider_config is not None:
                return self._provider_config
            with tempfile.TemporaryDirectory(prefix="openevolve-opencode-config-") as workdir:
                env = os.environ.copy()
                # Explicit config paths may be relative to the caller's directory.
                for key in ("OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR"):
                    if env.get(key):
                        env[key] = str(Path(env[key]).expanduser().resolve())
                env.update(
                    {
                        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
                        "OPENCODE_DISABLE_AUTOUPDATE": "1",
                        "OPENCODE_PURE": "1",
                    }
                )
                code, stdout, _ = await self._run_process(
                    [executable, "--pure", "debug", "config"],
                    b"",
                    workdir,
                    env,
                    self._remaining(deadline),
                )
                if code != 0:
                    # Config diagnostics may contain API keys; never include their output.
                    raise OpenCodeCLICompatibilityError(
                        "OpenCode CLI could not resolve provider configuration. Use CLI 1.18.32 "
                        "or a compatible version and check `opencode --pure debug config` locally."
                    )
                config = self._decode_config(stdout)
                selected = {key: config[key] for key in CONFIG_KEYS if key in config}
                if not isinstance(selected.get("provider", {}), dict) or any(
                    not isinstance(selected[key], list)
                    or any(not isinstance(item, str) for item in selected[key])
                    for key in ("enabled_providers", "disabled_providers")
                    if key in selected
                ):
                    raise OpenCodeCLICompatibilityError("Invalid OpenCode provider configuration")
                self._provider_config = selected
                return selected
        finally:
            self._config_lock.release()

    @staticmethod
    def _decode_config(stdout: bytes) -> Dict[str, Any]:
        try:
            config = json.loads(stdout)
        except (ValueError, UnicodeDecodeError):
            raise OpenCodeCLICompatibilityError(
                "Malformed OpenCode CLI configuration response; use CLI 1.18.32 or a compatible version"
            ) from None
        if not isinstance(config, dict):
            raise OpenCodeCLICompatibilityError(
                "Malformed OpenCode CLI configuration: expected an object"
            )
        return config

    def _generation_environment(self, workdir: str, provider_config, temperature, top_p):
        config = copy.deepcopy(provider_config)
        agent = {
            "description": "Return candidate code for OpenEvolve to apply and evaluate.",
            "mode": "primary",
            "prompt": GENERATION_INSTRUCTIONS,
            "permission": {"*": "deny"},
        }
        for name, value in (("temperature", temperature), ("top_p", top_p)):
            if value is not None:
                agent[name] = value
        config.update(
            {
                "agent": {AGENT_NAME: agent},
                "default_agent": AGENT_NAME,
                "permission": {"*": "deny"},
                "share": "disabled",
                "autoupdate": False,
                "snapshot": False,
                "compaction": {"auto": False, "prune": False},
            }
        )
        env = os.environ.copy()
        # Preserve authentication/provider environment variables, not personal execution flags.
        for key in list(env):
            if key.startswith("OPENCODE_EXPERIMENTAL") or key.startswith("OPENCODE_ENABLE_"):
                env.pop(key)
        env.pop("OPENCODE_CONFIG", None)
        root = Path(workdir)
        env.update(
            {
                "XDG_CONFIG_HOME": str(root / "config"),
                "OPENCODE_CONFIG_DIR": str(root / "config" / "opencode"),
                "OPENCODE_TEST_HOME": str(root / "home"),
                "OPENCODE_DB": str(root / "session.db"),
                "OPENCODE_CONFIG_CONTENT": json.dumps(config, ensure_ascii=False),
                "OPENCODE_PERMISSION": json.dumps({"*": "deny"}),
                "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
                "OPENCODE_DISABLE_AUTOUPDATE": "1",
                "OPENCODE_DISABLE_AUTOCOMPACT": "1",
                "OPENCODE_DISABLE_PRUNE": "1",
                "OPENCODE_DISABLE_CLAUDE_CODE": "1",
                "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
                "OPENCODE_DISABLE_DEFAULT_PLUGINS": "0",
                "OPENCODE_AUTO_SHARE": "0",
                "OPENCODE_PURE": "1",
            }
        )
        return env

    @staticmethod
    def _verify_generation_config(
        config: Dict[str, Any], expected_agent: Optional[Dict[str, Any]] = None
    ) -> None:
        agents = config.get("agent", {})
        agent = agents.get(AGENT_NAME, {}) if isinstance(agents, dict) else {}
        compaction = config.get("compaction", {})
        deny = ({"*": "deny"}, "deny")
        if (
            not isinstance(agent, dict)
            or agent.get("permission") not in deny
            or config.get("permission") not in deny
            or agent.get("prompt") != GENERATION_INSTRUCTIONS
            or agent.get("mode") != "primary"
            or agent.get("disable") is True
            or (
                expected_agent is not None
                and any(
                    agent.get(key) != expected_agent.get(key) for key in ("temperature", "top_p")
                )
            )
            or config.get("default_agent") != AGENT_NAME
            or any(config.get(key) for key in ("mcp", "plugin", "instructions", "tools"))
            or config.get("share") != "disabled"
            or config.get("autoupdate") is not False
            or config.get("snapshot") is not False
            or not isinstance(compaction, dict)
            or compaction.get("auto") is not False
            or compaction.get("prune") is not False
        ):
            raise OpenCodeCLICompatibilityError(
                "OpenCode CLI could not confirm isolated generation settings. Use CLI 1.18.32 "
                "or a compatible version; managed settings must permit a generation-only agent."
            )

    def _build_command(self, executable: str, effort: Optional[str]) -> List[str]:
        command = [
            executable,
            "--pure",
            "run",
            "--format",
            "json",
            "--model",
            self.model,
            "--agent",
            AGENT_NAME,
            "--title",
            "OpenEvolve",
        ]
        if effort is not None:
            command.extend(["--variant", effort])
        return command

    async def _run_cli(self, executable, prompt, timeout, effort, temperature, top_p):
        deadline = asyncio.get_running_loop().time() + timeout
        provider_config = await self._load_provider_config(executable, deadline)
        with tempfile.TemporaryDirectory(prefix="openevolve-opencode-") as workdir:
            env = self._generation_environment(workdir, provider_config, temperature, top_p)
            code, stdout, _ = await self._run_process(
                [executable, "--pure", "debug", "config"],
                b"",
                workdir,
                env,
                self._remaining(deadline),
            )
            if code != 0:
                raise OpenCodeCLICompatibilityError(
                    "OpenCode CLI rejected the generation configuration. Use CLI 1.18.32 "
                    "or a compatible version. Configuration output is omitted."
                )
            expected_agent = json.loads(env["OPENCODE_CONFIG_CONTENT"])["agent"][AGENT_NAME]
            self._verify_generation_config(self._decode_config(stdout), expected_agent)
            code, stdout, stderr = await self._run_process(
                self._build_command(executable, effort),
                prompt,
                workdir,
                env,
                self._remaining(deadline),
            )
            detail = stderr.decode("utf-8", errors="replace")
            if code == 2 or (
                code != 0
                and any(
                    marker in detail.lower()
                    for marker in (
                        "unknown argument",
                        "unknown option",
                        "unexpected argument",
                        "unrecognized argument",
                        "unrecognized option",
                        "invalid values:",
                        "configinvaliderror",
                        "configuration is invalid",
                    )
                )
            ):
                raise OpenCodeCLICompatibilityError(
                    "OpenCode CLI rejected its configuration or options. Use CLI 1.18.32 "
                    "or a compatible version; check the model and variant settings."
                )
            if code != 0:
                event_detail = ""
                if stdout.strip():
                    try:
                        self._parse_events(stdout)
                    except RuntimeError as exc:
                        event_detail = str(exc)
                raise RuntimeError(
                    f"OpenCode CLI exited with code {code}. {event_detail} "
                    f"stderr: {self._safe_detail(detail)}"
                )
            return self._parse_events(stdout)

    async def _run_process(self, command, prompt, workdir, env, timeout):
        # OpenCode resolves its SDK directory from PWD, ahead of process.cwd().
        env = {**env, "PWD": workdir}
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        communication = asyncio.create_task(process.communicate(prompt))
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            await asyncio.shield(self._stop_process(process, communication))
            if isinstance(exc, asyncio.TimeoutError):
                raise asyncio.TimeoutError(
                    f"OpenCode CLI ({self.model}) attempt timed out"
                ) from exc
            raise
        return process.returncode, stdout, stderr

    @staticmethod
    async def _stop_process(process, communication) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        await communication
        await process.wait()

    def _safe_detail(self, detail: str) -> str:
        """Redact provider credential options before including bounded diagnostics."""

        def secrets(value, sensitive=False):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from secrets(
                        item,
                        sensitive
                        or any(
                            word in key.lower()
                            for word in ("key", "token", "secret", "password", "auth", "header")
                        ),
                    )
            elif isinstance(value, list):
                for item in value:
                    yield from secrets(item, sensitive)
            elif sensitive and isinstance(value, str) and value:
                yield value

        for secret in sorted(set(secrets(self._provider_config or {})), key=len, reverse=True):
            detail = detail.replace(secret, "[redacted]")
        return detail.strip()[:1000]

    def _parse_events(self, stdout: bytes) -> Tuple[str, Optional[Dict[str, Any]]]:
        text_parts = {}
        finish_parts = {}
        usage = None
        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise RuntimeError("Malformed JSON event from OpenCode CLI") from exc
            if not isinstance(event, dict):
                raise RuntimeError("Malformed JSON event from OpenCode CLI: expected an object")
            kind = event.get("type")
            if kind == "error":
                error = event.get("error", {})
                if isinstance(error, dict):
                    data = error.get("data", {})
                    message = (
                        data.get(
                            "message", error.get("message", error.get("name", "unknown error"))
                        )
                        if isinstance(data, dict)
                        else "unknown error"
                    )
                    if error.get("name") in (
                        "ConfigInvalidError",
                        "ModelNotFoundError",
                        "ProviderModelNotFoundError",
                        "ProviderAuthError",
                        "AuthError",
                    ) or (
                        isinstance(data, dict)
                        and (
                            data.get("statusCode") in (401, 403, 404)
                            or data.get("isRetryable") is False
                        )
                    ):
                        raise ValueError(
                            "OpenCode CLI rejected provider/model configuration or authentication. "
                            "Check `opencode models` and `opencode auth login`."
                        )
                else:
                    message = str(error)
                raise RuntimeError(
                    f"OpenCode CLI generation failed: {self._safe_detail(str(message))}"
                )
            if kind == "tool_use":
                raise RuntimeError("OpenCode CLI attempted tool use during generation")
            if kind not in ("text", "step_finish"):
                continue
            part = event.get("part")
            if not isinstance(part, dict) or not isinstance(part.get("id"), str) or not part["id"]:
                raise RuntimeError("Malformed OpenCode CLI response part")
            if kind == "text":
                # Only completed text parts belong in the generated response.
                timing = part.get("time")
                if not isinstance(timing, dict) or timing.get("end") is None:
                    continue
                text = part.get("text")
                if not isinstance(text, str):
                    raise RuntimeError("Malformed OpenCode CLI text response")
                part_id = part["id"]
                if part_id in text_parts and text_parts[part_id] != text:
                    raise RuntimeError("Conflicting duplicate OpenCode CLI text response")
                text_parts[part_id] = text
                continue
            if part.get("reason") != "stop":
                raise RuntimeError(
                    "OpenCode CLI generation did not complete successfully (truncated or interrupted)"
                )
            part_id = part["id"]
            if part_id in finish_parts:
                if finish_parts[part_id] != part:
                    raise RuntimeError("Conflicting duplicate OpenCode CLI completion")
                continue
            finish_parts[part_id] = part
            raw = part.get("tokens")
            if raw is None:
                continue
            if not isinstance(raw, dict) or not isinstance(raw.get("cache", {}), dict):
                raise RuntimeError("Malformed token usage from OpenCode CLI")
            counts = [
                raw.get("input", 0),
                raw.get("output", 0),
                raw.get("reasoning", 0),
                raw.get("cache", {}).get("read", 0),
                raw.get("cache", {}).get("write", 0),
            ]
            if any(
                isinstance(n, bool)
                or not isinstance(n, (int, float))
                or not math.isfinite(n)
                or n < 0
                or int(n) != n
                for n in counts
            ):
                raise RuntimeError("Malformed token usage from OpenCode CLI")
            input_tokens, output_tokens, reasoning, read, write = map(int, counts)
            if usage is None:
                usage = dict.fromkeys(
                    (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "cached_input_tokens",
                        "cache_creation_input_tokens",
                        "reasoning_tokens",
                    ),
                    0,
                )
                usage["model"] = self.model
            usage["prompt_tokens"] += input_tokens + read + write
            usage["completion_tokens"] += output_tokens + reasoning
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            usage["cached_input_tokens"] += read
            usage["cache_creation_input_tokens"] += write
            usage["reasoning_tokens"] += reasoning
        if not finish_parts:
            raise RuntimeError("OpenCode CLI exited without a completed step")
        response = "".join(text_parts.values())
        if not response.strip():
            raise RuntimeError("Empty response from OpenCode CLI")
        return response, usage


def init_opencode_cli_client(model_cfg):
    """Factory compatible with OpenEvolve's ``init_client`` configuration hook."""
    return OpenCodeCLILLM(model_cfg)
