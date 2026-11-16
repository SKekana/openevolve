# Antigravity ACP quickstart

Use `provider: antigravity_acp` to generate candidate code or SEARCH/REPLACE diffs
through Google's official Antigravity ACP server. OpenEvolve applies the response
and runs the evaluator. This example uses `gemini-3.8-flash-medium` and the
function-minimization evaluator, with 10 iterations and one evaluation worker.

## Install and authenticate

The supported setup below uses Linux x86_64 and official ACP server **1.3.0**.
Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m openevolve.antigravity_acp install
python -m openevolve.antigravity_acp login
```

Run `install` once. It downloads Google's approximately 334 MB archive, verifies
its pinned SHA-256 checksum, and installs the two required executables under
`~/.local/share/openevolve/antigravity-acp-server/1.3.0/`. The extracted installation
uses approximately 1.1 GB. To reuse a previously downloaded official archive:

```bash
python -m openevolve.antigravity_acp install --archive /path/to/server.zip
```

The login command opens Google's consent page when credentials are needed. Choose
your Google account and complete consent; a localhost callback returns to the
server automatically. If the browser does not open, the command prints the sign-in
URL. Keep the login command running while completing the browser flow.

The dedicated persistent profile defaults to
`~/.local/share/openevolve/antigravity-acp`. On Linux its refresh credentials are
saved in `antigravity-acp/acp_token.json` inside that profile, with permissions
`0600`. Later processes refresh access silently. Login does not make a generation
request. Existing `agy` CLI credentials are separate; use the ACP login command.

The adapter supports personal Google OAuth. Your account's model access and quota
apply. See Google's [IDE authentication guide](https://antigravity.google/docs/ide/extensions#authentication-and-licensing)
and the [official ACP registry entry](https://github.com/agentclientprotocol/registry/blob/main/antigravity-acp/agent.json).

## Run the example

```bash
python openevolve-run.py \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/antigravity_acp_quickstart/config.yaml \
  --output openevolve_output_antigravity_acp
```

The configuration uses Flash Medium with weight 1.0, a 300-second timeout per
attempt, two retries, and a five-second retry delay. It requests checkpoints at
iterations 5 and 10 and disables LLM evaluator feedback. Results are written to
the chosen output directory, including `best/best_program.py`, its metrics, logs,
and checkpoints. Model output can still produce invalid or lower-scoring candidates;
the evaluator determines candidate quality.

## Alternate executable or profile

Set these fields at `llm` level or per model:

```yaml
llm:
  provider: antigravity_acp
  antigravity_acp_path: /absolute/path/to/agy_acp_server.par
  antigravity_acp_home: /absolute/path/to/dedicated/profile
  models:
    - name: gemini-3.8-flash-medium
      weight: 1.0
      timeout: 300
```

Authenticate that same profile:

```bash
python -m openevolve.antigravity_acp login \
  --executable /absolute/path/to/agy_acp_server.par \
  --profile /absolute/path/to/dedicated/profile
```

Explicit configuration takes precedence over
`OPENEVOLVE_ANTIGRAVITY_ACP_BIN` and `OPENEVOLVE_ANTIGRAVITY_ACP_HOME`, which take
precedence over the defaults. The login command uses those environment variables
too. The provider sets `GEMINI_HOME` for its child process using the selected
profile. Keep `localharness_external` executable beside the server on Linux.

## Generation behavior

Each attempt launches a fresh server and session in an empty temporary workspace.
The adapter checks protocol version 1 and server identity/version 1.3.0, disables
client filesystem and terminal capabilities, sends no MCP servers, and requests
`_meta.agy.enabledTools: ["finish"]`. It verifies the persisted tool filter before
the prompt and after model selection. The requested model must appear in the
catalog and be confirmed by `session/set_config_option`.

Keep the dedicated profile free of `config/hooks.json`, `config/mcp_config.json`,
and nonempty `config/skills` or `antigravity-cli/skills` directories. These are
rejected before startup because the official server otherwise loads profile
customizations. Temporary workspaces prevent workspace hooks and skills from
being inherited. The ACP server retains conversation history in its profile.

The adapter supplies the complete ordered conversation as text and preserves
assistant response whitespace. It collects only `agent_message_chunk` text and
requires a nonempty response with terminal `stopReason: "end_turn"`. Tool activity,
permission requests, and client operations cause immediate failure. Expired or
missing authentication fails without opening a browser during evolution; rerun
the login command interactively.

Authentication, installation, compatibility, and model-selection failures are
not retried. Transient failures, incomplete/empty turns, and timeouts retry the
same model using a new session and workspace. Cancellation does not retry. Cleanup
allows up to two seconds for normal shutdown, then terminates the process group
on POSIX and the direct server process elsewhere. Temporary workspaces are removed.

Model name, weight, timeout, retry settings, and system/conversation messages are
supported. Select reasoning effort through the model slug, such as
`gemini-3.8-flash-medium`; omit `reasoning_effort`. Sampling parameters (`temperature`,
`top_p`, `max_tokens`, and generation seed), API keys/base URLs, and other providers'
budget/tool settings are not forwarded. OpenEvolve still uses its seed for model
selection.

The server reports context-window usage rather than per-request input/output token
counts. `last_usage` therefore remains `None`; `last_context_usage` on the adapter
contains the most recent `used` and `size` when available. No monetary cost or
spending limit is inferred from this data.

## Tests and compatibility

Offline protocol, tool-boundary, retry, installation, and lifecycle tests require
no Google account:

```bash
python -m pytest tests/test_antigravity_acp_llm.py -q
```

The live test makes two model requests: a direct tool-restriction challenge and
one complete evolution iteration. It requires the installed server and saved
ACP login, and is skipped unless explicitly enabled:

```bash
OPENEVOLVE_RUN_ANTIGRAVITY_ACP_TESTS=1 python -m pytest \
  tests/integration/test_antigravity_acp_live.py -v -s
```

On 2026-10-05, the live provider test passed with official ACP 1.3.0 and Flash
Medium: the challenge returned `TOOLS_DISABLED`, created no workspace files, and
the evolution request produced a parsed, evaluated candidate.

The [10-iteration trial](TRIAL_RESULTS.md) also completed with 10 valid candidates,
no ACP retries/failures, and a best combined score of 1.4995496974. Checkpoints
were saved at iterations 5 and 10.

The separate `antigravity_cli` adapter still requires the CLI to confirm its
custom-agent tool restriction. AGY 1.2.16 fails that check before receiving a
prompt. The tested `--hub` paths also failed restriction probes. See the
[workaround investigation](../antigravity_cli_quickstart/ACP_WORKAROUND.md) for
the evidence and protocol differences.

Programmatic injection is available through
`openevolve.llm.antigravity_acp.init_antigravity_acp_client`.
