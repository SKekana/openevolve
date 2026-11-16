"""Offline AGY contract tests, including rejection before sending any prompt."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openevolve.config import LLMModelConfig, load_config
from openevolve.llm.antigravity_cli import (
    AntigravityCLICompatibilityError,
    AntigravityCLILLM,
    init_antigravity_cli_client,
)
from openevolve.llm.ensemble import LLMEnsemble, _create_model

ROOT = Path(__file__).resolve().parents[1]


def config(**kwargs):
    return LLMModelConfig(
        **dict(
            name="gemini-3.8-flash-medium",
            provider="antigravity_cli",
            timeout=5,
            retries=0,
            retry_delay=0,
            **kwargs,
        )
    )


@pytest.fixture
def fake(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX executable fixture")
    executable = tmp_path / "agy"
    executable.write_text(
        f"#!{sys.executable}\n" + (ROOT / "tests/fixtures/fake_antigravity_cli.py").read_text()
    )
    executable.chmod(0o755)
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_RECORDS", str(records))
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", "success")
    return records


def read_records(fake):
    return [json.loads(p.read_text()) for p in fake.glob("*.json")]


def assert_cleaned(fake):
    for r in read_records(fake):
        assert not Path(r["cwd"]).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(r["pid"], 0)


def test_config_and_worker():
    from openevolve import process_parallel

    cfg = load_config(ROOT / "examples/antigravity_cli_quickstart/config.yaml")
    cfg.language = "python"
    data = process_parallel.ProcessParallelController._serialize_config(
        SimpleNamespace(file_suffix=".py"), cfg
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
        process_parallel._worker_init(data, "unused.py")
        for models in (
            process_parallel._worker_config.llm.models,
            process_parallel._worker_config.llm.evaluator_models,
        ):
            assert len(models) == 1
            assert isinstance(_create_model(models[0]), AntigravityCLILLM)
            assert (models[0].name, models[0].timeout, models[0].retries) == (
                "gemini-3.8-flash-medium",
                300,
                2,
            )
    assert isinstance(init_antigravity_cli_client(config()), AntigravityCLILLM)
    custom = object()
    c = config()
    c.init_client = lambda _: custom
    assert _create_model(c) is custom


@pytest.mark.parametrize(
    "key,value",
    [
        ("name", None),
        ("name", " "),
        ("timeout", 0),
        ("timeout", float("inf")),
        ("retries", -1),
        ("retries", 0.5),
        ("retry_delay", -1),
        ("reasoning_effort", "ultra"),
    ],
)
def test_invalid_config(key, value):
    c = config()
    setattr(c, key, value)
    with pytest.raises(ValueError):
        AntigravityCLILLM(c)


@pytest.mark.asyncio
async def test_conversation_response_and_usage(fake, monkeypatch):
    response = "<<<<<<< SEARCH\r\n    x = 1  \r\n=======\r\n    x = 2\r\n>>>>>>> REPLACE\r\n"
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_RESPONSE", response)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "previous"},
        {"role": "user", "content": "λ $(literal) `literal`\n" * 10000},
    ]
    llm = AntigravityCLILLM(config())
    assert (
        await llm.generate_with_context("system\ninstructions", messages, reasoning_effort="high")
        == response
    )
    r = read_records(fake)[0]
    prompt = json.loads(r["prompt"])["message"]["content"]
    assert json.loads(prompt.split("\nConversation:\n", 1)[1]) == [
        {"role": "system", "content": "system\ninstructions"},
        *messages,
    ]
    assert "--sandbox" in r["args"] and "--disable-slash-commands" in r["args"]
    assert r["args"][r["args"].index("--effort") + 1] == "high"
    assert "--continue" not in r["args"] and "--dangerously-skip-permissions" not in r["args"]
    assert "tools: []" in r["agent"]
    assert llm.last_usage == dict(
        model="gemini-3.8-flash-medium",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        thinking_tokens=10,
        cached_input_tokens=60,
    )
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["tools", "missing_tools", "wrong_agent", "wrong_model", "wrong_cwd"]
)
async def test_restrictions_before_prompt(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", mode)
    c = config()
    c.retries = 3
    with pytest.raises(AntigravityCLICompatibilityError):
        await AntigravityCLILLM(c).generate("private prompt")
    records = read_records(fake)
    assert len(records) == 1 and not records[0]["prompt"]
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["bad_options", "auth", "fatal"])
async def test_setup_errors_not_retried(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", mode)
    c = config()
    c.retries = 3
    with pytest.raises(ValueError):
        await AntigravityCLILLM(c).generate("prompt")
    assert len(read_records(fake)) == 1
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "nonzero",
        "failed",
        "malformed",
        "missing_result",
        "empty",
        "bad_usage",
        "duplicate",
        "exit_after_success",
        "CANCELED",
        "INTERRUPTED",
        "INVALID",
        "WAITING",
        "RUNNING",
    ],
)
async def test_bad_results(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", mode)
    llm = AntigravityCLILLM(config())
    with pytest.raises(RuntimeError):
        await llm.generate("prompt")
    assert llm.last_usage is None
    assert_cleaned(fake)


@pytest.mark.asyncio
async def test_tool_call_rejected(fake, monkeypatch):
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", "tool_call")
    with pytest.raises(AntigravityCLICompatibilityError):
        await AntigravityCLILLM(config()).generate("prompt")
    assert_cleaned(fake)


@pytest.mark.asyncio
async def test_usage_reset(fake, monkeypatch):
    ensemble = LLMEnsemble([config()])
    await ensemble.generate("first")
    assert ensemble.last_usage["total_tokens"] == 120
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", "no_usage")
    await ensemble.generate("second")
    assert ensemble.last_usage is None
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", "failed")
    with pytest.raises(RuntimeError):
        await ensemble.models[0].generate("third")
    assert ensemble.models[0].last_usage is None


@pytest.mark.asyncio
async def test_retry_and_concurrency(fake, monkeypatch):
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", "retry")
    c = config()
    c.retries = 1
    assert await AntigravityCLILLM(c).generate("prompt") == "generated code\n"
    records = read_records(fake)
    assert len(records) == 2 and len({r["cwd"] for r in records}) == 2
    for r in records:
        assert r["args"][r["args"].index("--model") + 1] == c.name
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", "echo")
    llm = AntigravityCLILLM(config())
    assert await asyncio.gather(llm.generate("first"), llm.generate("second")) == [
        "first",
        "second",
    ]
    assert len({r["cwd"] for r in read_records(fake)}) == 4
    assert_cleaned(fake)


@pytest.mark.asyncio
async def test_retries_bounded(fake, monkeypatch):
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", "nonzero")
    c = config()
    c.retries = 2
    with pytest.raises(RuntimeError):
        await AntigravityCLILLM(c).generate("prompt")
    assert len(read_records(fake)) == 3
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["child", "post_sleep"])
async def test_timeout(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", mode)
    c = config()
    c.timeout = 0.5
    with pytest.raises(asyncio.TimeoutError):
        await AntigravityCLILLM(c).generate("prompt")
    assert_cleaned(fake)
    child = read_records(fake)[0].get("child_pid")
    if child:
        stat = Path(f"/proc/{child}/stat")
        # SIGKILL delivery and the child's state transition can follow the
        # parent's exit. Allow that transition, but require the child to stop.
        for _ in range(100):
            try:
                if stat.read_text().split()[2] == "Z":
                    break
            except FileNotFoundError:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("AGY child remained alive after process-group termination")


@pytest.mark.asyncio
async def test_cancel(fake, monkeypatch):
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", "post_sleep")
    c = config()
    c.retries = 3
    task = asyncio.create_task(AntigravityCLILLM(c).generate("prompt"))
    try:
        for _ in range(300):
            try:
                if read_records(fake) and read_records(fake)[0]["prompt"]:
                    break
            except json.JSONDecodeError:
                pass
            await asyncio.sleep(0.01)
        else:
            pytest.fail("Fake CLI never received prompt")
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    assert len(read_records(fake)) == 1
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["stderr", "finish"])
async def test_permitted_control_and_diagnostics(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_AGY_MODE", mode)
    assert await AntigravityCLILLM(config()).generate("prompt") == "generated code\n"
    assert_cleaned(fake)


@pytest.mark.asyncio
async def test_missing_cli_and_bad_override(fake):
    llm = AntigravityCLILLM(config())
    with patch("openevolve.llm.antigravity_cli.shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError):
            await llm.generate("prompt")
    with pytest.raises(ValueError):
        await llm.generate("prompt", timeout=-1)
    assert read_records(fake) == []
