# OpenCode CLI quickstart

Use OpenCode CLI as OpenEvolve's generation backend with saved OpenCode
authentication and provider configuration. OpenCode returns code,
SEARCH/REPLACE diffs, or evaluator JSON; OpenEvolve applies and evaluates the
response. This example reuses the existing function-minimization program and
evaluator.

## Setup and run

Install OpenCode and authenticate, then install this checkout in your Python
environment. The adapter targets OpenCode CLI **1.18.32** and compatible releases.

```bash
npm install -g opencode-ai@1.18.32
opencode auth login
opencode models
pip install -e ".[dev]"
```

The example uses two account-specific Modal endpoints. Configure your `modal`
provider in OpenCode before running it, and verify the exact model IDs:

```bash
opencode models modal

python openevolve-run.py \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/opencode_cli_quickstart/config.yaml \
  --output openevolve_output_opencode
```

Run from the repository root. The example performs 10 iterations with one
evaluator worker and saves checkpoints every 5 iterations. Your provider account
must have access to both models. For other accounts, replace the model IDs with
ones listed by `opencode models`; built-in providers and custom OpenCode provider
definitions are supported. See OpenCode's [provider guide](https://opencode.ai/docs/providers/)
and [configuration guide](https://opencode.ai/docs/config/) for setup.

## Model ensemble and settings

| Model | Selection probability | Timeout per attempt |
|---|---:|---:|
| `modal/sihlek55--ep-glm-5-3-server.us-west.modal.direct` | 30% | 600 seconds |
| `modal/sihlek55--ep-glm-5-3-flash-server.us-west.modal.direct` | 70% | 300 seconds |

Each generation samples one model using OpenEvolve's weighted sampler. These
probabilities are not quotas; a short run may use either model more often.
Retries use the same selected model, with two retries and a five-second delay in
this example. LLM evaluator feedback is disabled.

Model names must be explicit and use OpenCode's `provider/model` format. Model
IDs may contain additional slashes. Per-model settings override inherited LLM
settings. Model weights, system messages, `timeout`, `retries`, and `retry_delay`
use the existing OpenEvolve configuration fields. The generic OpenEvolve timeout
default is 60 seconds; this example sets longer timeouts explicitly. Bare adapter
instances with unset request settings default to 300 seconds, three retries, and
a five-second retry delay. Each per-attempt timeout covers preparation,
configuration inspection, and generation together; retry delay applies separately.

`temperature` and `top_p` are forwarded through the dedicated OpenCode agent's
configuration. Temperature must be finite and nonnegative; `top_p` must be between
0 and 1. Optional `reasoning_effort` maps directly to OpenCode's `--variant`
flag; valid variant names depend on the provider and model. This Modal example
leaves it unset. `max_tokens` and generation `random_seed` are unsupported. The
seed still controls OpenEvolve's model selection; it does not make model output
deterministic. `api_base`, `api_key`, and other providers' budget/tool options are
not forwarded: configure endpoints and authentication in OpenCode itself.

## Execution and authentication

The adapter resolves effective provider definitions lazily using
`opencode --pure debug config`, then invokes
`opencode --pure run --format json --model <provider/model> --agent openevolve-generation --title OpenEvolve`
through an asynchronous subprocess. The explicit title prevents a separate
title-generation model call. Automatic context compaction and pruning are
disabled; oversized-context errors surface instead of auxiliary model calls.
The system message and ordered conversation are serialized as JSON through stdin,
preserving roles, Unicode, and code whitespace. Resolved provider configuration is
kept out of logs.

Each attempt uses separate temporary workspace and configuration directories,
an OpenCode home override, and a session database. The dedicated agent denies all
permissions, disables sharing, and omits inherited MCP servers, external plugins,
and personal instructions. Built-in authentication/provider plugins remain
available. The adapter verifies these generation settings before sending a prompt.
OpenCode retains its existing authentication data location; the adapter neither
reads nor copies the authentication store. The adapter supplies generation
settings through temporary runtime configuration and leaves authentication
management to OpenCode. Provider API options from the resolved configuration,
which may include API credentials, are forwarded into that runtime configuration.
Native configuration discovery can perform OpenCode's normal schema migrations
and dependency initialization.

Completed JSON `text` events are concatenated without trimming or inserted
separators. The adapter requires a successful `stop` completion event and rejects
errors, tool use, truncated output, malformed JSON, and empty responses. Reasoning
events are not included in the generated response.

Token usage flows into OpenEvolve's existing logs and candidate metadata. Prompt
tokens include input tokens plus cache reads and writes; completion tokens include
visible output plus reasoning. Cache and reasoning counts remain available
separately. Usage is `None` when OpenCode does not report token counts.

Missing executables, invalid configuration, and rejected provider/model or
authentication settings fail immediately. Runtime failures, invalid responses,
and timeouts use the configured retries. Cancellation stops
generation without retrying. Temporary resources are cleaned on success, failure,
timeout, and cancellation. POSIX timeout/cancellation also terminates the process
group; other platforms terminate the direct CLI process.

See the official [CLI reference](https://opencode.ai/docs/cli/) for authentication,
model discovery, JSON output, and provider-specific variants.

## Programmatic injection

```python
from openevolve.llm.opencode_cli import init_opencode_cli_client

for model in config.llm.models:
    model.init_client = init_opencode_cli_client
```

Custom `init_client` settings take precedence over provider selection.

## Tests

Offline tests use a fake executable and need no OpenCode login or model calls:

```bash
python -m pytest tests/test_opencode_cli_llm.py
```

The live tests are disabled by default, including in CI. Enable them explicitly
to run one complete evolution iteration with each configured Modal model, with
retries disabled:

```bash
OPENEVOLVE_RUN_OPENCODE_TESTS=1 python -m pytest \
  tests/integration/test_opencode_cli_live.py -v -s
```

They require both account-specific endpoints to be configured and available, and
check that each model produces a changed, successfully evaluated candidate with
a finite score and consistent reported token usage.
