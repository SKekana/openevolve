# Antigravity CLI quickstart

Use `provider: antigravity_cli` to request code or SEARCH/REPLACE diffs through
`agy`, using its saved login. OpenEvolve applies and evaluates the response.
This example uses Gemini 3.8 Flash Medium and the function-minimization evaluator.

## Compatibility status

Use the [official ACP quickstart](../antigravity_acp_quickstart/) for the working
Antigravity integration with a dedicated Google login. This directory documents
the separate CLI adapter and its compatibility boundary.

**AGY 1.2.16 does not pass the provider's required tool-restriction check.**
During local verification, both an empty tool list and a `finish`-only custom
agent still advertised the full tool set in `init.tools`. The adapter rejects
that configuration **before sending the prompt**, without retries. This shows
that restriction cannot be confirmed through the CLI protocol; it does not
establish whether every advertised tool would actually execute.

The provider requires AGY to report the requested custom agent and a tool list
containing no tools other than the optional `finish` control tool. There is no
unsafe bypass or fallback to the unrestricted default agent. A compatible AGY
release must pass the opt-in live test below before using this quickstart.

The [ACP workaround investigation](ACP_WORKAROUND.md) documents the tested hub
limitations and the implemented official ACP provider. Authenticated tool
restriction and a complete OpenEvolve iteration passed on ACP 1.3.0.

## Setup and run

Follow Google's [installation and authentication guide](https://antigravity.google/docs/cli/install).
Launch `agy` interactively once to authenticate, then check the CLI and model list:

```bash
agy --version
agy models
pip install -e ".[dev]"

python openevolve-run.py \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/antigravity_cli_quickstart/config.yaml \
  --output openevolve_output_antigravity
```

Run from the repository root after verifying compatibility. The configuration
requests 10 iterations, one evaluation worker, and checkpoints every 5 iterations.
It uses `gemini-3.8-flash-medium` with weight 1.0, a 300-second timeout per attempt,
two retries, and a five-second retry delay. LLM evaluator feedback is disabled.
No 10-iteration AGY trial has been completed on the incompatible local CLI.

## Behavior and options

Each attempt creates a temporary workspace with a custom primary agent declaring
`tools: []`, no subagent role, and command execution off. The adapter starts AGY
with stream-json input/output, an explicit model and agent, disabled slash-command
expansion, and sandboxing. It waits for and validates the initialization event
before sending a single stdin prompt containing the full ordered conversation.
It reads only the terminal successful response, preserving whitespace exactly.

AGY's sandbox alone is not read-only. The initialization tool check is mandatory.
The adapter does not resume prior conversations, copy credentials, change `HOME`,
or modify your global settings. AGY can retain CLI-managed history and load
profile startup integrations, including hooks; the temporary workspace is not
complete isolation of the CLI's configuration or startup behavior.

Existing `name`, `weight`, `timeout`, `retries`, `retry_delay`, `system_message`,
and `reasoning_effort` fields are supported. An explicit effort is passed as
`--effort` (`low`, `medium`, `high`, `xhigh`, or `max`). The quickstart leaves it
unset because the model slug already selects Medium. Per-model settings override
inherited LLM settings. OpenEvolve's global timeout default remains 60 seconds;
a bare adapter with unset request settings uses 300 seconds, three retries, and
a five-second delay.

`temperature`, `top_p`, `max_tokens`, and generation `random_seed` are not sent
to AGY. OpenEvolve's model selection still uses its configured seed. API keys,
API base URLs, and other providers' tool/budget options are unused. No monetary
cost or spending cap is inferred from token usage; your AGY account limits apply.

Final usage maps input/output counts to OpenEvolve's prompt/completion counts,
preserves the reported total, and records thinking and cache-read counts
separately. These extra counts are not added to the total again. Usage is absent
when AGY does not report it.

Missing authentication, invalid model/options, explicitly nonretryable errors,
and incompatible initialization fail immediately. Transient runtime failures
and timeouts retry the same model with a fresh workspace. Cancellation does not
retry. Timeout/cancellation terminates the process group on POSIX and the direct
process elsewhere; temporary workspaces are cleaned in all cases.

See Google's [headless protocol](https://antigravity.google/docs/cli/headless)
and [custom agent specification](https://antigravity.google/docs/subagents).

## Tests

Offline protocol and lifecycle tests need no AGY installation or account:

```bash
python -m pytest tests/test_antigravity_cli_llm.py -q
```

The opt-in live test makes a short generation request followed by one complete
evolution iteration, with retries disabled. It first verifies the initialization
boundary. On AGY 1.2.16 it fails at that boundary before sending any prompt:

```bash
OPENEVOLVE_RUN_ANTIGRAVITY_TESTS=1 python -m pytest \
  tests/integration/test_antigravity_cli_live.py -v -s
```

Programmatic injection is also supported:

```python
from openevolve.llm.antigravity_cli import init_antigravity_cli_client

for model in config.llm.models:
    model.init_client = init_antigravity_cli_client
```
