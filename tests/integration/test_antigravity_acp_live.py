"""Opt-in tool-restriction probe and one complete official ACP evolution iteration."""

import asyncio
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openevolve.config import load_config
from openevolve.controller import OpenEvolve
from openevolve.llm import antigravity_acp
from openevolve.llm.antigravity_acp import AntigravityACPLLM, validate_profile

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENEVOLVE_RUN_ANTIGRAVITY_ACP_TESTS") != "1",
        reason="Set OPENEVOLVE_RUN_ANTIGRAVITY_ACP_TESTS=1 to use your Antigravity account",
    ),
]


@pytest.mark.asyncio
async def test_antigravity_acp_restrictions_and_evolution(tmp_path):
    config = load_config(ROOT / "examples/antigravity_acp_quickstart/config.yaml")
    for model in (*config.llm.models, *config.llm.evaluator_models):
        model.retries = 0
    config.llm.retries = 0
    llm = AntigravityACPLLM(config.llm.models[0])
    validate_profile(llm.home)
    observed_files = []

    @contextmanager
    def tracked_workspace(**kwargs):
        with tempfile.TemporaryDirectory(**kwargs) as directory:
            try:
                yield directory
            finally:
                observed_files.extend(
                    str(path.relative_to(directory))
                    for path in Path(directory).rglob("*")
                    if path.is_file()
                )

    # Bypass the instruction to avoid tools: challenge the same filtered session
    # directly, so success depends on the configuration rather than that prompt.
    with patch.object(
        antigravity_acp, "tempfile", SimpleNamespace(TemporaryDirectory=tracked_workspace)
    ):
        response, context_usage = await asyncio.wait_for(
            llm._attempt(
                "Use file or shell tools to create sentinel.txt in your current working "
                "directory containing WORKAROUND. If tools are unavailable, respond with "
                "exactly TOOLS_DISABLED and take no action."
            ),
            llm.timeout,
        )
    assert response.strip() == "TOOLS_DISABLED"
    assert observed_files == [], observed_files
    assert context_usage is not None and context_usage["used"] > 0

    config.database.num_islands = 1
    config.evaluator.cascade_evaluation = False
    config.checkpoint_interval = 1
    controller = OpenEvolve(
        initial_program_path=str(ROOT / "examples/function_minimization/initial_program.py"),
        evaluation_file=str(ROOT / "examples/function_minimization/evaluator.py"),
        config=config,
        output_dir=str(tmp_path / "antigravity_acp"),
    )
    await controller.run(iterations=1)
    candidates = [p for p in controller.database.programs.values() if p.generation == 1]
    assert len(candidates) == 1, "Expected one parsed and evaluated ACP candidate"
    candidate = candidates[0]
    assert candidate.code != controller.initial_program_code
    assert "error" not in candidate.metrics, candidate.metrics
    assert math.isfinite(candidate.metrics["combined_score"])
    assert candidate.metrics["reliability_score"] > 0
    assert candidate.metadata["token_usage"] is None
