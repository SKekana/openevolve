# Integration Tests

This directory contains integration tests for OpenEvolve. Tests are organized into two categories:

## Fast Tests (CI)

**Smoke tests** that run in CI to validate basic functionality without requiring slow LLM inference:

```bash
# Run only fast tests (for CI)
pytest tests/integration/ -m "not slow"
```

These tests:
- Complete in <10 seconds total
- Test core API validation, configuration, and basic component initialization
- No real LLM calls required

## Slow Tests (Local Development)

**Full integration tests** with real LLM inference for comprehensive validation:

```bash
# Run all tests including slow ones (for local development)
pytest tests/integration/

# Run only slow tests
pytest tests/integration/ -m "slow"
```

These tests:
- Take ~30-60 minutes to complete
- Use a real optillm server with the codelion/dhara-250m model
- Test complete evolution pipelines, checkpointing, island migration, etc.
- Require optillm server running on localhost:8000

## Codex CLI tests

The Codex provider has separate opt-in tests that use saved Codex CLI authentication
and do not require optillm. After installing Codex CLI and running `codex login`:

```bash
OPENEVOLVE_RUN_CODEX_TESTS=1 python -m pytest tests/integration/test_codex_cli_live.py -v -s
```

This runs one evolution iteration with `gpt-6.1-sol` and one with `gpt-6-luna`,
checking candidate evaluation and token usage. These tests are skipped unless
explicitly enabled, including when running the slow suite.

## OpenCode CLI tests

The OpenCode provider has separate opt-in tests using saved OpenCode provider
configuration and authentication, with no optillm dependency. Follow the
[OpenCode CLI quickstart](../../examples/opencode_cli_quickstart/) to install
OpenCode 1.18.32, authenticate, and configure both account-specific Modal models:

```bash
OPENEVOLVE_RUN_OPENCODE_TESTS=1 python -m pytest tests/integration/test_opencode_cli_live.py -v -s
```

This runs one complete evolution iteration with GLM 5.3 and one with GLM 5.3
Flash, with retries disabled. Each run must produce a changed candidate that
evaluates successfully with a finite score and consistent reported token usage.
The tests are skipped unless explicitly enabled, including when running the slow
suite and in CI. Offline coverage needs no model calls:

```bash
python -m pytest tests/test_opencode_cli_llm.py
```

## Test Setup

For slow tests that require LLM inference:

1. **Install optillm**: `pip install optillm`
2. **Start server**: `OPTILLM_API_KEY=optillm optillm --model google/gemma-3-270m-it --port 8000`
3. **Set environment**: `export OPTILLM_API_KEY=optillm OPENAI_API_KEY=optillm`
4. **Run tests**: `pytest tests/integration/ -m "slow"`

## Configuration

The optillm-based integration configurations generally use:
- **0 retries** for fast failure
- **120 second timeout** per LLM call
- **In-memory database** for speed
- **Small iteration counts** (1-8 iterations) for CI compatibility
- **Full rewrites** (`diff_based_evolution: false`): dhara-250m does not produce valid
  SEARCH/REPLACE diffs, and a run where no iteration produces a candidate raises
  `EvolutionFailedError` (#288), so the tests use full-rewrite mode to always get candidates

## CI Behavior

GitHub Actions will:
- Run **fast tests only** (`-m "not slow"`) 
- Validate core functionality without requiring model downloads
- Skip all tests marked with `@pytest.mark.slow`

## Antigravity ACP tests

Install the official server and complete personal Google OAuth using the
[ACP quickstart](../../examples/antigravity_acp_quickstart/), then run:

```bash
OPENEVOLVE_RUN_ANTIGRAVITY_ACP_TESTS=1 python -m pytest tests/integration/test_antigravity_acp_live.py -v -s
```

This makes a direct tool-restriction challenge and one complete Flash Medium
evolution iteration with retries disabled and a 300-second timeout. It checks
that the challenge returns `TOOLS_DISABLED` without workspace files, and that a
generated candidate is applied and evaluated. It uses saved ACP authentication
without optillm. The test is skipped unless explicitly enabled; account/setup or
compatibility failures are visible when enabled. Per-call token usage remains
absent because the ACP server exposes context-window usage instead.

## Antigravity CLI tests

The AGY live test uses saved CLI authentication, with no optillm dependency:

```bash
OPENEVOLVE_RUN_ANTIGRAVITY_TESTS=1 python -m pytest tests/integration/test_antigravity_cli_live.py -v -s
```

It validates the generation-only initialization before sending a short smoke
prompt, then checks one complete Gemini 3.8 Flash Medium evolution iteration.
AGY 1.2.16 currently advertises unrestricted tools and fails the compatibility
check before any prompt is sent. This is a visible failure, not a silently skipped
or passing generation test. The test is skipped unless explicitly enabled.
See the [AGY quickstart](../../examples/antigravity_cli_quickstart/) for details.
