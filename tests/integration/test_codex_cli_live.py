"""Opt-in, real Codex CLI evolution tests; never run model calls by default."""

import math
import os
import shutil
from pathlib import Path

import pytest

from openevolve.config import load_config
from openevolve.controller import OpenEvolve

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENEVOLVE_RUN_CODEX_TESTS") != "1",
        reason="Set OPENEVOLVE_RUN_CODEX_TESTS=1 to use your Codex account",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-6.1-sol", "gpt-6-luna"])
async def test_codex_completes_evolution_iteration(model_name, tmp_path):
    assert shutil.which("codex"), "Install Codex CLI and run `codex login` first"
    config = load_config(ROOT / "examples/codex_cli_quickstart/config.yaml")
    config.llm.retries = 0
    for model in config.llm.models:
        model.weight = 1.0 if model.name == model_name else 0.0
        model.retries = 0
    config.database.num_islands = 1
    config.evaluator.cascade_evaluation = False
    config.checkpoint_interval = 1
    controller = OpenEvolve(
        initial_program_path=str(ROOT / "examples/function_minimization/initial_program.py"),
        evaluation_file=str(ROOT / "examples/function_minimization/evaluator.py"),
        config=config,
        output_dir=str(tmp_path / model_name),
    )
    await controller.run(iterations=1)

    candidates = [p for p in controller.database.programs.values() if p.generation == 1]
    assert len(candidates) == 1, "Expected one parsed and evaluated Codex candidate"
    candidate = candidates[0]
    assert candidate.code != controller.initial_program_code
    assert "error" not in candidate.metrics, candidate.metrics
    assert math.isfinite(candidate.metrics["combined_score"])
    assert candidate.metrics["reliability_score"] > 0
    usage = candidate.metadata["token_usage"]
    assert usage is not None, "Expected Codex CLI to report usage"
    assert usage["model"] == model_name
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["total_tokens"] > 0
    print(f"{model_name}: score={candidate.metrics['combined_score']:.4f}, usage={usage}")
