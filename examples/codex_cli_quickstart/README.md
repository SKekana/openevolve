# Codex CLI quickstart

Use Codex CLI as OpenEvolve's generation backend with your saved CLI login.
Codex returns code or SEARCH/REPLACE diffs; OpenEvolve applies and evaluates them.
This example reuses the existing function-minimization program and evaluator.

## Setup and run

Install Codex CLI and authenticate, then install this checkout in your Python
environment. The adapter targets Codex CLI 0.160.0 or a compatible version.

```bash
npm install -g @openai/codex
codex login
codex login status
pip install -e ".[dev]"

python openevolve-run.py \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/codex_cli_quickstart/config.yaml \
  --output openevolve_output_codex
```

Run the command from the repository root. It performs 10 iterations with one
worker and saves checkpoints every 5 iterations. Your CLI account must have
access to both configured models; its usage limits apply.

## Model ensemble

| Model | Selection probability | Reasoning | Timeout per attempt |
|---|---:|---|---:|
| `gpt-6.1-sol` | 30% | `high` | 600 seconds |
| `gpt-6-luna` | 70% | `medium` | 300 seconds |

Each generation samples one model using OpenEvolve's existing weighted sampler.
These are probabilities, not quotas: a short run may use either model more often.
Sol and Luna independently propose improvements; they do not form a draft/review
pipeline. Retries use the same selected model, with two retries and a five-second
delay in this example. LLM evaluator feedback is disabled.

Change the `llm.models` entries to adjust weights, reasoning, or timeouts. Model
names must be explicit. Per-model settings override inherited LLM settings.
The generic OpenEvolve timeout default is 60 seconds; use the explicit timeouts
in this example for reasoning models. Bare adapter instances with unset request
settings default to 300 seconds, three retries, and a five-second retry delay.

## Execution and authentication

The adapter invokes `codex exec` through an asynchronous subprocess, sends the
system message and ordered conversation as JSON through stdin, and reads only
the final-message file as the generated response. JSONL progress events provide
completion status and token usage. Usage flows into OpenEvolve's existing logs
and candidate metadata; cached input tokens are reported separately and are
already included in input-token totals.

Each attempt uses a separate temporary working directory, an ephemeral session,
a read-only sandbox, and no interactive approvals. Shell execution, hooks, apps,
plugins, subagents, and web search are disabled; project instructions are omitted.
The adapter ignores user configuration so personal model settings and configured
MCP servers do not enter these runs. Codex still uses its existing authentication
location (`CODEX_HOME` when set, otherwise `~/.codex`). The adapter neither reads
nor copies credentials and does not edit your Codex configuration.

`temperature`, `top_p`, `max_tokens`, and generation `random_seed` are not passed
to Codex because there are no equivalent generation flags. The seed still
controls OpenEvolve's model selection; it does not make model output deterministic.
`api_base`, `api_key`, and other providers' tool/budget options are not used by
this adapter. There is no per-call dollar cap; timeouts, retries, and iteration
counts bound the work requested.

Missing executables and rejected CLI options fail immediately with setup guidance.
Runtime failures, missing/empty responses, and timeouts use the configured retries.
Cancellation stops generation without retrying. Temporary files are cleaned on
success, failure, timeout, and cancellation. POSIX timeout/cancellation also kills
the process group; other platforms terminate the direct CLI process.

See the official [noninteractive guide](https://learn.chatgpt.com/docs/non-interactive-mode)
and [CLI reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli).

## Programmatic injection

```python
from openevolve.llm.codex_cli import init_codex_cli_client

for model in config.llm.models:
    model.init_client = init_codex_cli_client
```

## Tests

Offline tests use a fake executable and need no Codex login or model calls:

```bash
python -m pytest tests/test_codex_cli_llm.py
```

The live test is disabled by default, including in CI. Enable it explicitly to
run one complete evolution iteration with Sol and one with Luna, without retries:

```bash
OPENEVOLVE_RUN_CODEX_TESTS=1 python -m pytest \
  tests/integration/test_codex_cli_live.py -v -s
```
