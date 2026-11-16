"""Opt-in real AGY smoke test; incompatible tool restrictions fail visibly."""

import math
import os
import shutil
from pathlib import Path

import pytest

from openevolve.config import load_config
from openevolve.controller import OpenEvolve
from openevolve.llm.antigravity_cli import AntigravityCLILLM

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENEVOLVE_RUN_ANTIGRAVITY_TESTS") != "1",
        reason="Set OPENEVOLVE_RUN_ANTIGRAVITY_TESTS=1 to use your Antigravity account",
    ),
]


@pytest.mark.asyncio
async def test_antigravity_completes_evolution_iteration(tmp_path):
    assert shutil.which("agy"), "Install AGY and authenticate interactively first"
    config = load_config(ROOT / "examples/antigravity_cli_quickstart/config.yaml")
    for model in (*config.llm.models, *config.llm.evaluator_models):
        model.retries = 0
    config.llm.retries = 0
    # Fail directly on incompatible init.tools before controller error handling
    # could obscure the cause. No prompt is sent on an incompatible CLI.
    response = await AntigravityCLILLM(config.llm.models[0]).generate("Return exactly: ready.")
    assert response.strip()
    config.database.num_islands = 1
    config.evaluator.cascade_evaluation = False
    config.checkpoint_interval = 1
    controller = OpenEvolve(
        initial_program_path=str(ROOT / "examples/function_minimization/initial_program.py"),
        evaluation_file=str(ROOT / "examples/function_minimization/evaluator.py"),
        config=config,
        output_dir=str(tmp_path / "antigravity"),
    )
    await controller.run(iterations=1)
    candidates = [p for p in controller.database.programs.values() if p.generation == 1]
    assert len(candidates) == 1, "Expected one parsed and evaluated AGY candidate"
    candidate = candidates[0]
    assert candidate.code != controller.initial_program_code
    assert "error" not in candidate.metrics, candidate.metrics
    assert math.isfinite(candidate.metrics["combined_score"])
    assert candidate.metrics["reliability_score"] > 0
    usage = candidate.metadata["token_usage"]
    assert usage is not None
    assert usage["model"] == "gemini-3.8-flash-medium"
    assert usage["total_tokens"] > 0
