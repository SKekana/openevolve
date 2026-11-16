"""ACP boundary, error recovery, and subprocess lifecycle tests without an account."""

import asyncio
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openevolve.config import LLMConfig, LLMModelConfig, load_config
from openevolve.llm.antigravity_acp import (
    AntigravityACPCompatibilityError,
    AntigravityACPLLM,
    init_antigravity_acp_client,
)
from openevolve.llm.ensemble import _create_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX executable fixture")
    executable = tmp_path / "agy_acp_server.par"
    executable.write_text(
        f"#!{sys.executable}\n" + (ROOT / "tests/fixtures/fake_antigravity_acp.py").read_text()
    )
    executable.chmod(0o755)
    (tmp_path / "localharness_external").touch()
    (tmp_path / "localharness_external").chmod(0o755)
    home = tmp_path / "profile"
    token = home / "antigravity-acp/acp_token.json"
    token.parent.mkdir(parents=True)
    token.write_text("{}")
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setenv("OPENEVOLVE_FAKE_ACP_RECORDS", str(records))
    monkeypatch.setenv("OPENEVOLVE_FAKE_ACP_MODE", "success")
    return SimpleNamespace(executable=executable, home=home, records=records)


def config(fake, **kwargs):
    return LLMModelConfig(
        **{
            "provider": "antigravity_acp",
            "name": "gemini-3.8-flash-medium",
            "timeout": 10,
            "retries": 0,
            "retry_delay": 0,
            "antigravity_acp_path": str(fake.executable),
            "antigravity_acp_home": str(fake.home),
            **kwargs,
        }
    )


def records(fake):
    return [json.loads(p.read_text()) for p in fake.records.glob("*.json")]


def assert_cleaned(fake):
    for record in records(fake):
        assert not Path(record["cwd"]).exists()
        for pid in (record["pid"], record.get("child_pid")):
            if pid:
                path = Path(f"/proc/{pid}/stat")
                for _ in range(100):
                    try:
                        if path.read_text().split()[2] == "Z":
                            break
                    except FileNotFoundError:
                        break
                    time.sleep(0.01)
                else:
                    pytest.fail(f"ACP process {pid} remained alive after termination")


@pytest.mark.asyncio
async def test_interleaved_conversation_and_context_usage(fake, monkeypatch):
    response = "<<<<<<< SEARCH\r\n    λ = 1  \r\n=======\r\n    λ = 2\r\n>>>>>>> REPLACE\n"
    monkeypatch.setenv("OPENEVOLVE_FAKE_ACP_RESPONSE", response)
    llm = AntigravityACPLLM(config(fake))
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "prior"},
        {"role": "user", "content": "$(literal) `literal` λ\n" * 10000},
    ]
    assert await llm.generate_with_context("system", messages) == response
    record = records(fake)[0]
    requests = {x["method"]: x["params"] for x in record["requests"]}
    assert requests["initialize"]["clientCapabilities"] == {
        "fs": {"readTextFile": False, "writeTextFile": False},
        "terminal": False,
    }
    assert requests["session/new"]["_meta"] == {"agy": {"enabledTools": ["finish"]}}
    assert requests["session/new"]["mcpServers"] == []
    assert requests["session/set_config_option"]["value"] == llm.model
    text = requests["session/prompt"]["prompt"][0]["text"]
    assert json.loads(text.split("\nConversation:\n", 1)[1]) == [
        {"role": "system", "content": "system"},
        *messages,
    ]
    assert "--block" in record["browser"]
    assert record["home"] == str(fake.home)
    assert llm.last_usage is None
    assert llm.last_context_usage == {"used": 150, "size": 1048576}
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["version", "protocol", "auth", "browser", "filter", "missing_filter", "catalog", "model"],
)
async def test_rejects_setup_before_prompt_without_retry(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_ACP_MODE", mode)
    with pytest.raises(AntigravityACPCompatibilityError):
        await AntigravityACPLLM(config(fake, retries=3)).generate("private prompt")
    found = records(fake)
    assert len(found) == 1
    assert all(x["method"] != "session/prompt" for x in found[0]["requests"])
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["tool", "permission", "fs", "terminal", "malformed", "id", "wrong_session"]
)
async def test_rejects_protocol_and_client_tool_activity(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_ACP_MODE", mode)
    with pytest.raises(AntigravityACPCompatibilityError):
        await AntigravityACPLLM(config(fake, retries=2)).generate("prompt")
    found = records(fake)
    assert len(found) == 1
    if mode in ("permission", "fs", "terminal"):
        reply = found[0]["replies"][0]
        if mode == "permission":
            assert reply["result"]["outcome"]["outcome"] == "cancelled"
        else:
            assert reply["error"]["code"] == -32601
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["empty", "incomplete", "nonzero"])
async def test_incomplete_response_is_not_accepted(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_ACP_MODE", mode)
    with pytest.raises(RuntimeError):
        await AntigravityACPLLM(config(fake)).generate("prompt")
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["retry", "model_retry"])
async def test_transient_retry_uses_new_session_and_workspace(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_ACP_MODE", mode)
    assert await AntigravityACPLLM(config(fake, retries=1)).generate("prompt") == "generated code\n"
    found = records(fake)
    assert len(found) == 2 and found[0]["cwd"] != found[1]["cwd"]
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sleep", "timeout", "child"])
async def test_timeout_cleans_processes_and_workspace(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_ACP_MODE", mode)
    with pytest.raises(asyncio.TimeoutError):
        await AntigravityACPLLM(config(fake, timeout=0.3)).generate("prompt")
    assert_cleaned(fake)


@pytest.mark.asyncio
async def test_cancellation_does_not_retry(fake, monkeypatch):
    monkeypatch.setenv("OPENEVOLVE_FAKE_ACP_MODE", "timeout")
    llm = AntigravityACPLLM(config(fake, retries=2))
    task = asyncio.create_task(llm.generate("prompt"))
    for _ in range(100):
        if any(any(x["method"] == "session/prompt" for x in r["requests"]) for r in records(fake)):
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(records(fake)) == 1 and llm.last_usage is None
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["stderr", "child_exit"])
async def test_drains_stderr_and_cleans_descendants_after_parent_exit(fake, monkeypatch, mode):
    monkeypatch.setenv("OPENEVOLVE_FAKE_ACP_MODE", mode)
    assert await AntigravityACPLLM(config(fake)).generate("prompt") == "generated code\n"
    assert_cleaned(fake)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "config/hooks.json",
        "config/mcp_config.json",
        "config/skills/SKILL.md",
        "antigravity-cli/skills/SKILL.md",
    ],
)
async def test_customizations_rejected_before_start(fake, path):
    target = fake.home / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}")
    with pytest.raises(AntigravityACPCompatibilityError):
        await AntigravityACPLLM(config(fake)).generate("prompt")
    assert not records(fake)


@pytest.mark.asyncio
async def test_missing_auth_and_missing_installation(fake):
    (fake.home / "antigravity-acp/acp_token.json").unlink()
    with pytest.raises(AntigravityACPCompatibilityError, match="login"):
        await AntigravityACPLLM(config(fake)).generate("prompt")
    assert not records(fake)
    (fake.home / "antigravity-acp/acp_token.json").touch()
    fake.executable.unlink()
    with pytest.raises(FileNotFoundError):
        await AntigravityACPLLM(config(fake)).generate("prompt")


@pytest.mark.parametrize(
    "key,value",
    [
        ("name", ""),
        ("timeout", 0),
        ("timeout", float("nan")),
        ("retries", True),
        ("retry_delay", -0.5),
        ("reasoning_effort", "high"),
    ],
)
def test_invalid_options(fake, key, value):
    with pytest.raises(ValueError):
        AntigravityACPLLM(config(fake, **{key: value}))


def test_registration_inheritance_and_worker_roundtrip(fake):
    from openevolve import process_parallel

    cfg = load_config(ROOT / "examples/antigravity_acp_quickstart/config.yaml")
    cfg.llm.antigravity_acp_path = str(fake.executable)
    cfg.llm.antigravity_acp_home = str(fake.home)
    cfg.llm.primary_model = "gemini-3.8-flash-medium"
    cfg.llm.rebuild_models()
    assert isinstance(_create_model(cfg.llm.models[0]), AntigravityACPLLM)
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
        model = process_parallel._worker_config.llm.models[0]
        assert model.provider == "antigravity_acp"
        assert model.antigravity_acp_home == str(fake.home)
        assert model.antigravity_acp_path == str(fake.executable)
    assert isinstance(init_antigravity_acp_client(config(fake)), AntigravityACPLLM)
    override = LLMModelConfig(antigravity_acp_home="override")
    LLMConfig(antigravity_acp_home="shared", models=[override])
    assert override.antigravity_acp_home == "override"


def test_install_archive_verification(tmp_path, monkeypatch):
    from openevolve import antigravity_acp as setup

    archive = tmp_path / "server.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("agy_acp_server.par", b"server")
        bundle.writestr("localharness_external", b"harness")
        bundle.writestr("unexpected.txt", b"do not extract")
    target = tmp_path / "install/1.3.0/agy_acp_server.par"
    monkeypatch.setattr(setup, "DEFAULT_BINARY", target)
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup.platform, "machine", lambda: "x86_64")
    with pytest.raises(ValueError, match="checksum"):
        setup.install_server(archive)
    assert not target.exists()
    monkeypatch.setattr(setup, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    setup.install_server(archive)
    assert target.read_bytes() == b"server"
    assert os.access(target, os.X_OK)
    assert not (target.parent / "unexpected.txt").exists()
