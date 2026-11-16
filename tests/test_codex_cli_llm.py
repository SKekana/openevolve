"""Offline tests for Codex generation, configuration, and process lifecycle."""

import asyncio
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openevolve.config import LLMModelConfig, load_config
from openevolve.llm.codex_cli import CodexCLILLM, init_codex_cli_client
from openevolve.llm.ensemble import LLMEnsemble, _create_model

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "examples/codex_cli_quickstart/config.yaml"


def make_config(**overrides):
    return LLMModelConfig(
        **{
            "name": "gpt-6-luna",
            "provider": "codex_cli",
            "timeout": 5,
            "retries": 0,
            "retry_delay": 0,
            "reasoning_effort": "medium",
            **overrides,
        }
    )


class TestCodexCLIConfig(unittest.TestCase):
    def test_registration_and_factory(self):
        config = make_config()
        self.assertIsInstance(_create_model(config), CodexCLILLM)
        self.assertIsInstance(init_codex_cli_client(config), CodexCLILLM)
        custom = object()
        config.init_client = lambda _: custom
        self.assertIs(_create_model(config), custom)

    def test_requires_explicit_model(self):
        for name in (None, "", " "):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "explicit model"):
                CodexCLILLM(make_config(name=name))

    def test_defaults_and_zero_values(self):
        llm = CodexCLILLM(LLMModelConfig(name="gpt-6-luna"))
        self.assertEqual((llm.timeout, llm.retries, llm.retry_delay), (300, 3, 5))
        llm = CodexCLILLM(make_config())
        self.assertEqual((llm.retries, llm.retry_delay), (0, 0))

    def test_invalid_request_settings(self):
        for invalid in (
            {"timeout": 0},
            {"timeout": float("inf")},
            {"retry_delay": -1},
            {"retries": -1},
            {"retries": 0.5},
            {"reasoning_effort": "turbo"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CodexCLILLM(make_config(**invalid))

    def test_quickstart_and_worker_serialization(self):
        from openevolve import process_parallel

        config = load_config(QUICKSTART)
        # The controller infers this before serializing the config to workers.
        config.language = "python"
        # Exercise the same serialization and reconstruction used by worker processes.
        serialized = process_parallel.ProcessParallelController._serialize_config(
            SimpleNamespace(file_suffix=".py"), config
        )
        with patch.multiple(
            process_parallel,
            _worker_config=None,
            _worker_evaluation_file=None,
            _worker_evaluator=None,
            _worker_llm_ensemble=None,
            _worker_prompt_sampler=None,
            create=True,
        ):
            process_parallel._worker_init(serialized, "unused-evaluator.py")
            worker_config = process_parallel._worker_config
        for models in (worker_config.llm.models, worker_config.llm.evaluator_models):
            self.assertEqual([m.provider for m in models], ["codex_cli", "codex_cli"])
            self.assertEqual([m.name for m in models], ["gpt-6.1-sol", "gpt-6-luna"])
            self.assertEqual([m.reasoning_effort for m in models], ["high", "medium"])
            self.assertEqual([m.timeout for m in models], [600, 300])
            self.assertEqual([m.retries for m in models], [2, 2])
        ensemble = LLMEnsemble(worker_config.llm.models)
        with patch.object(ensemble.random_state, "choices", return_value=[1]) as choose:
            self.assertEqual(ensemble._sample_model().model, "gpt-6-luna")
            self.assertEqual(choose.call_args.kwargs["weights"], [0.3, 0.7])
        self.assertEqual(config.evaluator.parallel_evaluations, 1)
        self.assertFalse(config.evaluator.use_llm_feedback)


@unittest.skipUnless(os.name == "posix", "Executable fixture uses a POSIX shebang")
class TestCodexCLISubprocess(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="codex-provider-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records_dir = self.root / "records"
        self.records_dir.mkdir()
        executable = self.root / "codex"
        fixture = (ROOT / "tests/fixtures/fake_codex_cli.py").read_text(encoding="utf-8")
        executable.write_text(f"#!{sys.executable}\n" + fixture, encoding="utf-8")
        executable.chmod(0o755)
        self.env = patch.dict(
            os.environ,
            {
                "PATH": str(self.root) + os.pathsep + os.environ.get("PATH", ""),
                "OPENEVOLVE_FAKE_CODEX_RECORDS": str(self.records_dir),
                "OPENEVOLVE_FAKE_CODEX_MODE": "success",
                "OPENEVOLVE_FAKE_CODEX_RESPONSE": "generated code\n",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def records(self):
        return [json.loads(p.read_text()) for p in self.records_dir.glob("*.json")]

    def assert_cleaned(self):
        for record in self.records():
            self.assertFalse(Path(record["cwd"]).exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(record["pid"], 0)

    async def wait_started(self):
        for _ in range(300):
            try:
                records = self.records()
                if records:
                    return records[0]
            except json.JSONDecodeError:
                pass
            await asyncio.sleep(0.01)
        self.fail("Fake Codex did not start")

    async def test_generation_keeps_code_and_conversation_intact(self):
        response = "<<<<<<< SEARCH\n    value = 1  \n=======\n    value = 2  \n>>>>>>> REPLACE\n"
        response = response.replace("\n", "\r\n")
        prompt = "λ\n" + "large prompt $(not-a-command) `literal` " * 6000
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": prompt},
        ]
        os.environ["OPENEVOLVE_FAKE_CODEX_RESPONSE"] = response
        llm = CodexCLILLM(make_config())
        self.assertEqual(
            await llm.generate_with_context("system\ninstructions", messages), response
        )
        record = self.records()[0]
        conversation = json.loads(record["prompt"].split("\n", 1)[1])
        self.assertEqual(
            conversation, [{"role": "system", "content": "system\ninstructions"}, *messages]
        )
        self.assertNotIn(prompt, record["args"])
        self.assertEqual(
            llm.last_usage,
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cached_input_tokens": 60,
                "model": "gpt-6-luna",
            },
        )
        self.assert_cleaned()

    async def test_isolated_settings_and_override(self):
        llm = CodexCLILLM(make_config(system_message="configured system"))
        await llm.generate("prompt", reasoning_effort="high")
        record = self.records()[0]
        args = record["args"]
        for flag in ("--json", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check"):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("--sandbox") + 1], "read-only")
        self.assertEqual(args[-1], "-")
        self.assertIn('model_reasoning_effort="high"', args)
        self.assertIn('approval_policy="never"', args)
        self.assertIn('web_search="disabled"', args)
        self.assertIn("project_doc_max_bytes=0", args)
        for feature in (
            "hooks",
            "apps",
            "plugins",
            "shell_tool",
            "unified_exec",
            "multi_agent",
            "multi_agent_v2",
        ):
            self.assertEqual(args[args.index(feature) - 1], "--disable")
        self.assertIn("configured system", record["prompt"])

    async def test_usage_propagates_and_does_not_leak(self):
        ensemble = LLMEnsemble([make_config()])
        await ensemble.generate("first")
        self.assertEqual(ensemble.last_usage["model"], "gpt-6-luna")
        os.environ["OPENEVOLVE_FAKE_CODEX_MODE"] = "no_usage"
        await ensemble.generate("second")
        self.assertIsNone(ensemble.last_usage)
        os.environ["OPENEVOLVE_FAKE_CODEX_MODE"] = "nonzero"
        with self.assertRaises(RuntimeError):
            await ensemble.models[0].generate("third")
        self.assertIsNone(ensemble.models[0].last_usage)

    async def test_failure_scenarios(self):
        for mode, message in (
            ("nonzero", "temporary service failure"),
            ("json_error_exit", "account model access denied"),
            ("failed_turn", "model unavailable"),
            ("error", "request failed"),
            ("malformed", "Malformed JSON"),
            ("bad_usage", "Malformed token usage"),
            ("missing_response", "final response"),
            ("empty_response", "Empty response"),
            ("missing_completion", "completed turn"),
        ):
            with self.subTest(mode=mode):
                os.environ["OPENEVOLVE_FAKE_CODEX_MODE"] = mode
                llm = CodexCLILLM(make_config())
                with self.assertRaisesRegex(RuntimeError, message):
                    await llm.generate("prompt")
                self.assertIsNone(llm.last_usage)
        self.assert_cleaned()

    async def test_missing_cli_and_bad_options_do_not_retry(self):
        llm = CodexCLILLM(make_config(retries=3))
        with patch("openevolve.llm.codex_cli.shutil.which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "codex login"):
                await llm.generate("prompt")
        self.assertEqual(self.records(), [])
        os.environ["OPENEVOLVE_FAKE_CODEX_MODE"] = "bad_options"
        with self.assertRaisesRegex(ValueError, "0.160.0"):
            await llm.generate("prompt")
        self.assertEqual(len(self.records()), 1)
        self.assert_cleaned()

    async def test_invalid_override_does_not_start_process(self):
        llm = CodexCLILLM(make_config())
        with self.assertRaises(ValueError):
            await llm.generate("prompt", timeout=-1)
        self.assertEqual(self.records(), [])

    async def test_retry_uses_same_model_and_new_directory(self):
        os.environ["OPENEVOLVE_FAKE_CODEX_MODE"] = "retry"
        llm = CodexCLILLM(make_config(retries=1))
        self.assertEqual(await llm.generate("prompt"), "generated code\n")
        records = self.records()
        self.assertEqual(len(records), 2)
        self.assertEqual(len({r["cwd"] for r in records}), 2)
        for record in records:
            self.assertEqual(record["args"][record["args"].index("--model") + 1], "gpt-6-luna")
        self.assert_cleaned()

    async def test_retries_are_bounded(self):
        os.environ["OPENEVOLVE_FAKE_CODEX_MODE"] = "nonzero"
        with self.assertRaises(RuntimeError):
            await CodexCLILLM(make_config(retries=2)).generate("prompt")
        self.assertEqual(len(self.records()), 3)
        self.assert_cleaned()

    async def test_timeout_kills_process_group(self):
        os.environ["OPENEVOLVE_FAKE_CODEX_MODE"] = "child"
        with self.assertRaises(asyncio.TimeoutError):
            await CodexCLILLM(make_config(timeout=0.5)).generate("prompt")
        self.assert_cleaned()
        # An orphaned child may briefly be a zombie before init reaps it.
        child_pid = self.records()[0]["child_pid"]
        proc_stat = Path(f"/proc/{child_pid}/stat")
        if proc_stat.exists():
            self.assertEqual(proc_stat.read_text().split()[2], "Z")
        else:
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)

    async def test_cancellation_does_not_retry(self):
        os.environ["OPENEVOLVE_FAKE_CODEX_MODE"] = "sleep"
        task = asyncio.create_task(CodexCLILLM(make_config(retries=3)).generate("prompt"))
        record = await self.wait_started()
        try:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            self.assertEqual(len(self.records()), 1)
            self.assert_cleaned()
        finally:
            try:
                os.killpg(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def test_concurrent_calls_use_independent_files(self):
        os.environ["OPENEVOLVE_FAKE_CODEX_MODE"] = "echo"
        llm = CodexCLILLM(make_config())
        outputs = await asyncio.gather(llm.generate("first"), llm.generate("second"))
        self.assertEqual(outputs, ["first", "second"])
        self.assertEqual(len({r["cwd"] for r in self.records()}), 2)
        self.assert_cleaned()


if __name__ == "__main__":
    unittest.main()
