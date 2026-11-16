"""Codex CLI backend using saved CLI authentication and noninteractive generation.

Set ``llm.provider: codex_cli`` and supply an explicit model name. Sampling
parameters (temperature, top_p, max_tokens, random_seed) are not forwarded:
Codex has no equivalent generation flags. OpenEvolve still seeds model selection.
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
from typing import Any, Dict, List, Optional, Tuple

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

VALID_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
GENERATION_INSTRUCTIONS = (
    "You are the code generation backend for OpenEvolve. Respond to the supplied "
    "conversation in its requested format (code, SEARCH/REPLACE diffs, or evaluation JSON). "
    "OpenEvolve applies and evaluates your response. Do not use tools, inspect "
    "the filesystem, edit files, run commands, or delegate work."
)


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    value = getattr(config, key, None)
    return default if value is None else value


class CodexCLILLM(LLMInterface):
    """Generate code through ``codex exec`` (tested with Codex CLI 0.160.0)."""

    def __init__(self, model_cfg=None):
        self.model = getattr(model_cfg, "name", None)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Codex CLI requires an explicit model name in llm.models[].name")
        self.system_message = getattr(model_cfg, "system_message", None)
        self.weight = _cfg_value(model_cfg, "weight", 1.0)
        self.timeout = _cfg_value(model_cfg, "timeout", 300)
        self.retries = _cfg_value(model_cfg, "retries", 3)
        self.retry_delay = _cfg_value(model_cfg, "retry_delay", 5)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.last_usage: Optional[Dict[str, Any]] = None
        self._validate_options(self.timeout, self.retries, self.retry_delay, self.reasoning_effort)
        logger.info("Initialized CodexCLILLM with model: %s", self.model)

    @staticmethod
    def _validate_options(timeout, retries, retry_delay, effort) -> None:
        for name, value, allow_zero in (
            ("timeout", timeout, False),
            ("retry_delay", retry_delay, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and not allow_zero)
            ):
                raise ValueError(
                    f"Codex CLI {name} must be a finite {'nonnegative' if allow_zero else 'positive'} number"
                )
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("Codex CLI retries must be a nonnegative integer")
        if effort is not None and effort not in VALID_REASONING_EFFORTS:
            raise ValueError(f"Invalid Codex reasoning_effort: {effort}")

    async def generate(self, prompt: str, **kwargs) -> str:
        system_message = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system_message, [{"role": "user", "content": prompt}], **kwargs
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        self.last_usage = None
        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        self._validate_options(timeout, retries, retry_delay, effort)
        executable = shutil.which("codex")
        if executable is None:
            raise FileNotFoundError(
                "The `codex` CLI was not found. Install it and run `codex login`."
            )

        # JSON keeps message roles, order, newlines and delimiter-like code intact.
        conversation = [{"role": "system", "content": system_message or ""}, *messages]
        prompt = (
            "Generate the next assistant response for this conversation:\n"
            + json.dumps(conversation, ensure_ascii=False)
        ).encode("utf-8")

        for attempt in range(retries + 1):
            try:
                response, usage = await self._run_cli(executable, prompt, timeout, effort)
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
            except (FileNotFoundError, ValueError):
                # Missing CLI and invalid CLI options will not improve on retry.
                raise
            except (RuntimeError, asyncio.TimeoutError) as exc:
                if attempt == retries:
                    raise
                logger.warning(
                    "Codex CLI (%s) attempt %s/%s failed: %s. Retrying...",
                    self.model,
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                await asyncio.sleep(retry_delay)

    def _build_command(
        self, executable: str, output_path: Path, effort: Optional[str]
    ) -> List[str]:
        cmd = [
            executable,
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--model",
            self.model,
            "--output-last-message",
            str(output_path),
            "-c",
            'approval_policy="never"',
            "-c",
            'web_search="disabled"',
            "-c",
            "project_doc_max_bytes=0",
            "-c",
            "developer_instructions=" + json.dumps(GENERATION_INSTRUCTIONS),
        ]
        for feature in (
            "hooks",
            "apps",
            "plugins",
            "shell_tool",
            "unified_exec",
            "multi_agent",
            "multi_agent_v2",
        ):
            cmd.extend(["--disable", feature])
        if effort is not None:
            cmd.extend(["-c", "model_reasoning_effort=" + json.dumps(effort)])
        cmd.append("-")
        return cmd

    async def _run_cli(
        self, executable: str, prompt: bytes, timeout: float, effort: Optional[str]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        # Do not override CODEX_HOME or copy credentials. Codex manages its auth.
        with tempfile.TemporaryDirectory(prefix="openevolve-codex-") as workdir:
            output_path = Path(workdir) / "response.txt"
            cmd = self._build_command(executable, output_path, effort)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                start_new_session=(os.name == "posix"),
            )
            communication = asyncio.create_task(process.communicate(prompt))
            try:
                stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
                await asyncio.shield(self._stop_process(process, communication))
                if isinstance(exc, asyncio.TimeoutError):
                    raise asyncio.TimeoutError(
                        f"Codex CLI ({self.model}) timed out after {timeout} seconds"
                    ) from exc
                raise

            error_text = stderr.decode("utf-8", errors="replace").strip()[:1000]
            if process.returncode == 2:
                raise ValueError(
                    "Codex CLI rejected its configuration or options. Use CLI 0.160.0 or a "
                    f"compatible version. stderr: {error_text}"
                )
            if process.returncode != 0:
                # With --json, Codex often reports failures on stdout, not stderr.
                event_error = ""
                if stdout.strip():
                    try:
                        self._parse_events(stdout)
                    except RuntimeError as exc:
                        event_error = str(exc)
                raise RuntimeError(
                    f"Codex CLI exited with code {process.returncode}. "
                    f"{event_error} stderr: {error_text}"
                )
            usage = self._parse_events(stdout)
            if not output_path.is_file():
                raise RuntimeError(
                    f"Codex CLI did not write a final response. stderr: {error_text}"
                )
            # Preserve whitespace exactly: SEARCH blocks can contain meaningful indentation.
            response = output_path.read_bytes().decode("utf-8")
            if not response.strip():
                raise RuntimeError(f"Empty response from Codex CLI. stderr: {error_text}")
            return response, usage

    @staticmethod
    async def _stop_process(process, communication) -> None:
        try:
            if os.name == "posix":
                # Also kill descendants holding stdout open, even if the parent exited.
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        await communication
        await process.wait()

    def _parse_events(self, stdout: bytes) -> Optional[Dict[str, Any]]:
        completed = False
        usage = None
        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise RuntimeError("Malformed JSON event from Codex CLI") from exc
            if not isinstance(event, dict):
                raise RuntimeError("Malformed JSON event from Codex CLI: expected an object")
            if event.get("type") in ("turn.failed", "error"):
                detail = event.get("error", event.get("message", "unknown error"))
                raise RuntimeError(f"Codex CLI generation failed: {str(detail)[:1000]}")
            if event.get("type") == "turn.completed":
                completed = True
                raw = event.get("usage")
                if raw is not None:
                    if not isinstance(raw, dict):
                        raise RuntimeError("Malformed token usage from Codex CLI")
                    counts = {
                        key: raw.get(key, 0)
                        for key in ("input_tokens", "output_tokens", "cached_input_tokens")
                    }
                    if any(
                        isinstance(n, bool) or not isinstance(n, int) or n < 0
                        for n in counts.values()
                    ):
                        raise RuntimeError("Malformed token usage from Codex CLI")
                    usage = {
                        "prompt_tokens": counts["input_tokens"],
                        "completion_tokens": counts["output_tokens"],
                        "total_tokens": counts["input_tokens"] + counts["output_tokens"],
                        "cached_input_tokens": counts["cached_input_tokens"],
                        "model": self.model,
                    }
        if not completed:
            raise RuntimeError("Codex CLI exited without a completed turn")
        return usage


def init_codex_cli_client(model_cfg):
    """Factory compatible with OpenEvolve's ``init_client`` configuration hook."""
    return CodexCLILLM(model_cfg)
