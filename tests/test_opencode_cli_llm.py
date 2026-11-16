"""Offline tests for OpenCode generation, isolation, and process lifecycle."""

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

from openevolve.config import Config, LLMModelConfig, load_config
from openevolve.llm.ensemble import LLMEnsemble, _create_model
from openevolve.llm.opencode_cli import OpenCodeCLILLM, init_opencode_cli_client

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "examples/opencode_cli_quickstart/config.yaml"
MODEL = "modal/model/with/slashes"
EXPECTED_USAGE = {
    "prompt_tokens": 170,
    "completion_tokens": 25,
    "total_tokens": 195,
    "cached_input_tokens": 60,
    "cache_creation_input_tokens": 10,
    "reasoning_tokens": 5,
    "model": MODEL,
}


def make_config(**overrides):
    return LLMModelConfig(
        **{
            "name": MODEL,
            "provider": "opencode_cli",
            "timeout": 5,
            "retries": 0,
            "retry_delay": 0,
            **overrides,
        }
    )


class TestOpenCodeCLIConfig(unittest.TestCase):
    def test_registration_and_factory(self):
        config = make_config()
        self.assertIsInstance(_create_model(config), OpenCodeCLILLM)
        self.assertIsInstance(init_opencode_cli_client(config), OpenCodeCLILLM)
        from openevolve.llm import OpenCodeCLILLM as ExportedClass
        from openevolve.llm import init_opencode_cli_client as exported_factory

        self.assertIs(ExportedClass, OpenCodeCLILLM)
        self.assertIs(exported_factory, init_opencode_cli_client)
        custom = object()
        config.init_client = lambda _: custom
        self.assertIs(_create_model(config), custom)

    def test_requires_explicit_provider_and_model(self):
        for name in (None, "", " ", "model", "/model", "provider/"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                OpenCodeCLILLM(make_config(name=name))
        self.assertEqual(OpenCodeCLILLM(make_config()).model, MODEL)

    def test_defaults_and_zero_values(self):
        llm = OpenCodeCLILLM(LLMModelConfig(name=MODEL))
        self.assertEqual((llm.timeout, llm.retries, llm.retry_delay), (300, 3, 5))
        llm = OpenCodeCLILLM(make_config(temperature=0, top_p=0))
        self.assertEqual((llm.retries, llm.retry_delay), (0, 0))

    def test_invalid_request_settings(self):
        for invalid in (
            {"timeout": 0},
            {"timeout": float("inf")},
            {"timeout": True},
            {"retry_delay": -1},
            {"retry_delay": float("nan")},
            {"retries": -1},
            {"retries": 0.5},
            {"retries": True},
            {"reasoning_effort": ""},
            {"reasoning_effort": " high "},
            {"reasoning_effort": 1},
            {"temperature": -1},
            {"temperature": float("nan")},
            {"temperature": True},
            {"top_p": -0.1},
            {"top_p": 1.1},
            {"top_p": float("inf")},
            {"top_p": True},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                OpenCodeCLILLM(make_config(**invalid))
        # Variant names and temperature limits depend on the selected provider.
        OpenCodeCLILLM(make_config(reasoning_effort="provider-specific-variant", temperature=3))

    def test_inheritance_model_overrides_and_evaluator_config(self):
        config = Config.from_dict(
            {
                "llm": {
                    "provider": "opencode_cli",
                    "timeout": 600,
                    "retries": 2,
                    "retry_delay": 0,
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "reasoning_effort": "high",
                    "models": [
                        {"name": MODEL, "weight": 0.3},
                        {
                            "name": "anthropic/claude-model",
                            "weight": 0.7,
                            "timeout": 300,
                            "temperature": 0,
                            "reasoning_effort": "low",
                        },
                    ],
                    "evaluator_models": [
                        {
                            "name": "modal/evaluator-model",
                            "timeout": 120,
                            "top_p": 1,
                            "reasoning_effort": "evaluator-variant",
                        }
                    ],
                }
            }
        )
        first, second = config.llm.models
        evaluator = config.llm.evaluator_models[0]
        self.assertEqual([m.provider for m in (first, second, evaluator)], ["opencode_cli"] * 3)
        self.assertEqual([m.timeout for m in (first, second, evaluator)], [600, 300, 120])
        self.assertEqual([m.retries for m in (first, second, evaluator)], [2, 2, 2])
        self.assertEqual([m.temperature for m in (first, second, evaluator)], [0.4, 0, 0.4])
        self.assertEqual([m.top_p for m in (first, second, evaluator)], [0.8, 0.8, 1])
        self.assertEqual(
            [m.reasoning_effort for m in (first, second, evaluator)],
            ["high", "low", "evaluator-variant"],
        )
        self.assertIsInstance(LLMEnsemble(config.llm.evaluator_models).models[0], OpenCodeCLILLM)

    def test_quickstart_and_worker_serialization(self):
        from openevolve import process_parallel

        config = load_config(QUICKSTART)
        config.language = "python"
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
        expected_names = [
            "modal/sihlek55--ep-glm-5-3-server.us-west.modal.direct",
            "modal/sihlek55--ep-glm-5-3-flash-server.us-west.modal.direct",
        ]
        for models in (worker_config.llm.models, worker_config.llm.evaluator_models):
            self.assertEqual([m.provider for m in models], ["opencode_cli", "opencode_cli"])
            self.assertEqual([m.name for m in models], expected_names)
            self.assertEqual([m.timeout for m in models], [600, 300])
            self.assertEqual([m.retries for m in models], [2, 2])
            self.assertEqual([m.reasoning_effort for m in models], [None, None])
        ensemble = LLMEnsemble(worker_config.llm.models)
        with patch.object(ensemble.random_state, "choices", return_value=[1]) as choose:
            self.assertEqual(ensemble._sample_model().model, expected_names[1])
            self.assertEqual(choose.call_args.kwargs["weights"], [0.3, 0.7])
        self.assertEqual(config.max_iterations, 10)
        self.assertEqual(config.checkpoint_interval, 5)
        self.assertEqual(config.evaluator.parallel_evaluations, 1)
        self.assertFalse(config.evaluator.use_llm_feedback)

    def test_seeded_weighted_sampling_is_repeatable(self):
        configs = [
            make_config(weight=0.3, random_seed=42),
            make_config(name="anthropic/another-model", weight=0.7, random_seed=42),
        ]
        first, second = LLMEnsemble(configs), LLMEnsemble(configs)
        samples = [first._sample_model().model for _ in range(30)]
        self.assertEqual(samples, [second._sample_model().model for _ in range(30)])
        self.assertEqual(set(samples), {MODEL, "anthropic/another-model"})


@unittest.skipUnless(os.name == "posix", "Executable fixture uses a POSIX shebang")
class TestOpenCodeCLISubprocess(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="opencode-provider-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records_dir = self.root / "records"
        self.records_dir.mkdir()
        executable = self.root / "opencode"
        fixture = (ROOT / "tests/fixtures/fake_opencode_cli.py").read_text(encoding="utf-8")
        executable.write_text(f"#!{sys.executable}\n" + fixture, encoding="utf-8")
        executable.chmod(0o755)
        self.env = patch.dict(
            os.environ,
            {
                "PATH": str(self.root) + os.pathsep + os.environ.get("PATH", ""),
                "OPENEVOLVE_FAKE_OPENCODE_RECORDS": str(self.records_dir),
                "OPENEVOLVE_FAKE_OPENCODE_MODE": "success",
                "OPENEVOLVE_FAKE_OPENCODE_RESPONSE": "generated code\n",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def records(self, phase=None):
        records = [json.loads(path.read_text()) for path in self.records_dir.glob("*.json")]
        return [record for record in records if phase is None or record["phase"] == phase]

    def assert_cleaned(self):
        for record in self.records():
            self.assertFalse(Path(record["cwd"]).exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(record["pid"], 0)

    async def wait_started(self, phase="run"):
        for _ in range(300):
            records = self.records(phase)
            if records:
                return records[0]
            await asyncio.sleep(0.01)
        self.fail(f"Fake OpenCode did not start {phase}")

    @staticmethod
    def conversation(record):
        prompt = record["prompt"]
        return json.loads(prompt[prompt.index("[") :])

    async def test_generation_preserves_conversation_code_and_usage(self):
        response = (
            "<<<<<<< SEARCH\r\n    value = 1  \r\n=======\r\n    value = 2  \r\n>>>>>>> REPLACE\r\n"
        )
        prompt = "λ\n" + "large prompt $(not-a-command) `literal` " * 6000
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": prompt},
        ]
        os.environ["OPENEVOLVE_FAKE_OPENCODE_RESPONSE"] = response
        llm = OpenCodeCLILLM(make_config())
        self.assertEqual(
            await llm.generate_with_context("system\ninstructions", messages), response
        )
        record = self.records("run")[0]
        self.assertEqual(
            self.conversation(record),
            [{"role": "system", "content": "system\ninstructions"}, *messages],
        )
        self.assertNotIn(prompt, record["args"])
        self.assertEqual(record["args"][record["args"].index("--model") + 1], MODEL)
        self.assertEqual(llm.last_usage, EXPECTED_USAGE)
        self.assert_cleaned()

    async def test_isolated_settings_preserve_only_provider_and_routing(self):
        llm = OpenCodeCLILLM(make_config(system_message="configured system"))
        with self.assertLogs("openevolve.llm.opencode_cli", level="INFO") as logs:
            await llm.generate("prompt")
        record = self.records("run")[0]
        args = record["args"]
        self.assertEqual(args[:3], ["--pure", "run", "--format"])
        self.assertEqual(args[args.index("--format") + 1], "json")
        self.assertEqual(args[args.index("--agent") + 1], "openevolve-generation")
        self.assertEqual(args[args.index("--title") + 1], "OpenEvolve")
        self.assertNotIn("--variant", args)
        isolated = json.loads(record["env"]["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(
            isolated["provider"]["modal"]["options"]["apiKey"], "private-config-secret"
        )
        self.assertEqual(isolated["enabled_providers"], ["modal", "anthropic"])
        self.assertEqual(isolated["disabled_providers"], ["unwanted"])
        self.assertEqual(isolated["permission"], {"*": "deny"})
        self.assertEqual(isolated["agent"]["openevolve-generation"]["permission"], {"*": "deny"})
        self.assertEqual(set(isolated["agent"]), {"openevolve-generation"})
        self.assertFalse(isolated.get("mcp"))
        self.assertFalse(isolated.get("plugin"))
        self.assertFalse(isolated.get("instructions"))
        self.assertEqual(isolated["share"], "disabled")
        self.assertFalse(isolated["snapshot"])
        self.assertFalse(isolated["autoupdate"])
        self.assertEqual(isolated["compaction"], {"auto": False, "prune": False})
        self.assertEqual(record["env"]["OPENCODE_DISABLE_AUTOCOMPACT"], "1")
        self.assertEqual(record["env"]["OPENCODE_DISABLE_PRUNE"], "1")
        self.assertEqual(self.conversation(record)[0]["content"], "configured system")
        self.assertNotIn("private-config-secret", "\n".join(logs.output))
        self.assertEqual(len(self.records("discovery")), 1)
        self.assertEqual(len(self.records("verification")), 1)
        self.assert_cleaned()

    async def test_environment_isolation_keeps_auth_location(self):
        with patch.dict(
            os.environ,
            {
                "XDG_DATA_HOME": str(self.root / "original-auth-data"),
                "OPENCODE_CONFIG": "/unsafe/config.json",
                "OPENCODE_CONFIG_CONTENT": '{"permission":{"*":"allow"}}',
                "OPENCODE_CONFIG_DIR": "/unsafe/config-dir",
                "OPENCODE_TEST_HOME": "/unsafe/home",
                "OPENCODE_DB": "/unsafe/session.db",
                "OPENCODE_PERMISSION": '{"*":"allow"}',
                "PWD": str(ROOT),
            },
        ):
            await OpenCodeCLILLM(make_config()).generate("prompt")
            record = self.records("run")[0]
            self.assertEqual(record["env"]["HOME"], os.environ["HOME"])
            self.assertEqual(record["env"]["XDG_DATA_HOME"], os.environ["XDG_DATA_HOME"])
        for key in ("XDG_CONFIG_HOME", "OPENCODE_CONFIG_DIR", "OPENCODE_TEST_HOME", "OPENCODE_DB"):
            self.assertTrue(Path(record["env"][key]).is_relative_to(record["cwd"]))
            self.assertFalse(Path(record["env"][key]).exists())
        self.assertNotEqual(record["env"].get("OPENCODE_CONFIG"), "/unsafe/config.json")
        self.assertEqual(json.loads(record["env"]["OPENCODE_PERMISSION"]), {"*": "deny"})
        for process_record in self.records():
            self.assertEqual(process_record["env"]["PWD"], process_record["cwd"])
        self.assert_cleaned()

    async def test_inline_custom_provider_configuration_is_preserved(self):
        provider = {
            "npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": "https://inline.invalid/v1", "apiKey": "inline-secret"},
            "models": {"custom-model": {"name": "Inline custom model"}},
        }
        with patch.dict(
            os.environ,
            {
                "OPENCODE_CONFIG_CONTENT": json.dumps(
                    {"provider": {"inline": provider}, "permission": {"*": "allow"}}
                )
            },
        ):
            await OpenCodeCLILLM(make_config(name="inline/custom-model")).generate("prompt")
        self.assertEqual(len(self.records("discovery")), 1)
        record = self.records("run")[0]
        isolated = json.loads(record["env"]["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(isolated["provider"]["inline"], provider)
        self.assertEqual(isolated["permission"], {"*": "deny"})
        self.assert_cleaned()

    async def test_discovery_resolves_relative_config_source_paths(self):
        sources = {
            "OPENCODE_CONFIG": "../provider-config/opencode.json",
            "OPENCODE_CONFIG_DIR": "relative/provider-config",
        }
        with patch.dict(os.environ, sources):
            await OpenCodeCLILLM(make_config()).generate("prompt")
        discovery = self.records("discovery")[0]
        for key, source in sources.items():
            self.assertEqual(discovery["env"][key], str((ROOT / source).resolve()))
        run = self.records("run")[0]
        self.assertNotIn("OPENCODE_CONFIG", run["env"])
        self.assertTrue(Path(run["env"]["OPENCODE_CONFIG_DIR"]).is_relative_to(run["cwd"]))
        self.assert_cleaned()

    async def test_sampling_and_variant_overrides(self):
        llm = OpenCodeCLILLM(make_config(temperature=0.3, top_p=0.8, reasoning_effort="low"))
        await llm.generate("prompt", temperature=0, top_p=1, reasoning_effort="provider-variant")
        record = self.records("run")[0]
        agent = json.loads(record["env"]["OPENCODE_CONFIG_CONTENT"])["agent"][
            "openevolve-generation"
        ]
        self.assertEqual(agent["temperature"], 0)
        self.assertEqual(agent["top_p"], 1)
        self.assertEqual(record["args"][record["args"].index("--variant") + 1], "provider-variant")
        await llm.generate("second", temperature=None, top_p=None, reasoning_effort=None)
        second = next(
            r for r in self.records("run") if self.conversation(r)[-1]["content"] == "second"
        )
        self.assertNotIn("--variant", second["args"])
        second_agent = json.loads(second["env"]["OPENCODE_CONFIG_CONTENT"])["agent"][
            "openevolve-generation"
        ]
        self.assertNotIn("temperature", second_agent)
        self.assertNotIn("top_p", second_agent)
        self.assertEqual(len(self.records("discovery")), 1)
        self.assert_cleaned()

    async def test_completed_text_and_step_events_are_deduplicated(self):
        llm = OpenCodeCLILLM(make_config())
        for mode in ("duplicate", "duplicate_finish"):
            with self.subTest(mode=mode):
                os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = mode
                self.assertEqual(await llm.generate("prompt"), "generated code\n")
                self.assertEqual(llm.last_usage, EXPECTED_USAGE)
        self.assert_cleaned()

    async def test_usage_propagates_and_does_not_leak(self):
        ensemble = LLMEnsemble([make_config()])
        await ensemble.generate("first")
        self.assertEqual(ensemble.last_usage, EXPECTED_USAGE)
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "no_usage"
        await ensemble.generate("second")
        self.assertIsNone(ensemble.last_usage)
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "nonzero"
        with self.assertRaises(RuntimeError):
            await ensemble.models[0].generate("third")
        self.assertIsNone(ensemble.models[0].last_usage)
        self.assert_cleaned()

    async def test_optional_usage_fields_default_to_zero(self):
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "partial_usage"
        llm = OpenCodeCLILLM(make_config())
        await llm.generate("prompt")
        self.assertEqual(
            llm.last_usage,
            {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "cached_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "reasoning_tokens": 0,
                "model": MODEL,
            },
        )
        self.assert_cleaned()

    async def test_runtime_errors_do_not_expose_provider_credentials(self):
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "secret_error"
        with self.assertLogs("openevolve.llm.opencode_cli", level="WARNING") as logs:
            with self.assertRaises(RuntimeError) as error:
                await OpenCodeCLILLM(make_config(retries=1)).generate("private prompt")
        self.assertNotIn("private-config-secret", str(error.exception))
        self.assertNotIn("private-config-secret", "\n".join(logs.output))
        self.assert_cleaned()

    async def test_generation_failure_scenarios(self):
        for mode in (
            "nonzero",
            "json_error_exit",
            "error",
            "malformed",
            "nonobject",
            "bad_usage",
            "bad_cache",
            "negative_usage",
            "float_usage",
            "bool_usage",
            "no_text",
            "partial_only",
            "empty_response",
            "missing_completion",
            "tool_use",
            "truncated",
            "finish_error",
            "missing_reason",
            "missing_text_id",
            "bad_text",
            "conflicting_text",
        ):
            with self.subTest(mode=mode):
                os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = mode
                llm = OpenCodeCLILLM(make_config())
                with self.assertRaises(RuntimeError):
                    await llm.generate("prompt")
                self.assertIsNone(llm.last_usage)
        self.assert_cleaned()

    async def test_missing_cli_and_bad_options_do_not_retry(self):
        llm = OpenCodeCLILLM(make_config(retries=3))
        with patch("openevolve.llm.opencode_cli.shutil.which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "opencode auth login"):
                await llm.generate("prompt")
        self.assertEqual(self.records(), [])
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "bad_options"
        with self.assertRaises(ValueError):
            await llm.generate("prompt")
        self.assertEqual(len(self.records("run")), 1)
        self.assert_cleaned()

    async def test_config_failures_stop_before_prompt_and_do_not_retry(self):
        for mode in (
            "discovery_bad_options",
            "discovery_malformed",
            "verification_bad_options",
            "verification_malformed",
            "isolation_failure",
            "malformed_agent",
            "malformed_agent_entry",
            "disabled_agent",
            "overridden_sampling",
            "inherited_auto_compaction",
        ):
            with self.subTest(mode=mode):
                before = len(self.records())
                os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = mode
                with self.assertRaises(ValueError) as error:
                    await OpenCodeCLILLM(make_config(retries=3, temperature=0.4)).generate(
                        "private prompt"
                    )
                self.assertNotIn("private-config-secret", str(error.exception))
                self.assertNotIn("private prompt", str(error.exception))
                self.assertEqual(self.records("run"), [])
                self.assertLessEqual(len(self.records()) - before, 2)
        self.assert_cleaned()

    async def test_provider_and_authentication_errors_do_not_retry(self):
        for mode in (
            "auth_error",
            "missing_model",
            "api_nonretryable",
            "api_status_401",
            "api_status_403",
            "api_status_404",
        ):
            with self.subTest(mode=mode):
                before = len(self.records("run"))
                os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = mode
                with self.assertRaises(ValueError):
                    await OpenCodeCLILLM(make_config(retries=3)).generate("prompt")
                self.assertEqual(len(self.records("run")) - before, 1)
        self.assert_cleaned()

    async def test_rate_limit_and_server_errors_retry(self):
        for mode in ("api_status_429", "api_status_503"):
            with self.subTest(mode=mode):
                before = len(self.records("run"))
                os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = mode
                with self.assertRaises(RuntimeError):
                    await OpenCodeCLILLM(make_config(retries=1)).generate("prompt")
                self.assertEqual(len(self.records("run")) - before, 2)
        self.assert_cleaned()

    async def test_invalid_overrides_do_not_start_process(self):
        llm = OpenCodeCLILLM(make_config())
        for overrides in (
            {"timeout": -1},
            {"retries": True},
            {"retry_delay": -1},
            {"temperature": -1},
            {"top_p": 1.1},
            {"reasoning_effort": " "},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                await llm.generate("prompt", **overrides)
        self.assertEqual(self.records(), [])

    async def test_retry_uses_same_model_and_new_directory(self):
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "retry"
        self.assertEqual(
            await OpenCodeCLILLM(make_config(retries=1)).generate("prompt"), "generated code\n"
        )
        records = self.records("run")
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record["cwd"] for record in records}), 2)
        self.assertEqual(len(self.records("discovery")), 1)
        self.assertEqual(len(self.records("verification")), 2)
        for record in records:
            self.assertEqual(record["args"][record["args"].index("--model") + 1], MODEL)
        self.assert_cleaned()

    async def test_retries_are_bounded(self):
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "nonzero"
        with self.assertRaises(RuntimeError):
            await OpenCodeCLILLM(make_config(retries=2)).generate("prompt")
        self.assertEqual(len(self.records("run")), 3)
        self.assert_cleaned()

    async def test_timeout_kills_process_group(self):
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "child"
        with self.assertRaises(asyncio.TimeoutError):
            await OpenCodeCLILLM(make_config(timeout=0.4)).generate("prompt")
        self.assert_cleaned()
        child_pid = self.records("run")[0]["child_pid"]
        proc_stat = Path(f"/proc/{child_pid}/stat")
        if proc_stat.exists():
            self.assertEqual(proc_stat.read_text().split()[2], "Z")
        else:
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)

    async def test_config_timeout_is_bounded(self):
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "verification_sleep"
        with self.assertRaises(asyncio.TimeoutError):
            await OpenCodeCLILLM(make_config(timeout=0.4)).generate("prompt")
        self.assertEqual(self.records("run"), [])
        self.assert_cleaned()

    async def test_attempt_deadline_includes_verification_and_generation(self):
        llm = OpenCodeCLILLM(make_config())
        deadlines = []
        original_run = llm._run_process

        def remaining(deadline):
            deadlines.append(deadline)
            return 5

        async def run_process(command, *args):
            if "run" in command:
                raise asyncio.TimeoutError("generation exhausted the attempt deadline")
            return await original_run(command, *args)

        with (
            patch.object(llm, "_remaining", side_effect=remaining),
            patch.object(llm, "_run_process", side_effect=run_process),
        ):
            with self.assertRaises(asyncio.TimeoutError):
                await llm.generate("prompt")
        # Discovery, verification, and generation must receive the same deadline.
        self.assertGreaterEqual(len(deadlines), 3)
        self.assertEqual(len(set(deadlines)), 1)
        self.assertEqual(len(self.records("verification")), 1)
        self.assert_cleaned()

    async def test_cancellation_does_not_retry(self):
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "sleep"
        task = asyncio.create_task(OpenCodeCLILLM(make_config(retries=3)).generate("prompt"))
        record = await self.wait_started()
        try:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            self.assertEqual(len(self.records("run")), 1)
            self.assert_cleaned()
        finally:
            try:
                os.killpg(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def test_cancelled_discovery_releases_lock_for_next_call(self):
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "discovery_sleep"
        llm = OpenCodeCLILLM(make_config(retries=3))
        task = asyncio.create_task(llm.generate("first"))
        record = await self.wait_started("discovery")
        try:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            self.assertEqual(len(self.records("discovery")), 1)
            self.assertEqual(self.records("run"), [])
            self.assert_cleaned()
            os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "success"
            self.assertEqual(await llm.generate("second"), "generated code\n")
            self.assertEqual(len(self.records("discovery")), 2)
            self.assert_cleaned()
        finally:
            try:
                os.killpg(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def test_concurrent_calls_share_discovery_and_isolate_sessions(self):
        os.environ["OPENEVOLVE_FAKE_OPENCODE_MODE"] = "echo"
        llm = OpenCodeCLILLM(make_config())
        outputs = await asyncio.gather(llm.generate("first"), llm.generate("second"))
        self.assertEqual(outputs, ["first", "second"])
        self.assertEqual(len(self.records("discovery")), 1)
        self.assertEqual(len(self.records("verification")), 2)
        records = self.records("run")
        self.assertEqual(len({record["cwd"] for record in records}), 2)
        self.assertEqual(len({record["env"]["OPENCODE_DB"] for record in records}), 2)
        self.assert_cleaned()


if __name__ == "__main__":
    unittest.main()
