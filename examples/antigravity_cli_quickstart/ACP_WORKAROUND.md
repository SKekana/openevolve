# Antigravity ACP workaround investigation

Investigated on 2026-10-04 with AGY CLI 1.2.16 and Google's Antigravity ACP server
1.3.0 on Linux x86_64.

Google's official ACP server is now available through the `antigravity_acp`
provider. Follow-up on 2026-10-05 completed personal Google OAuth login, verified
cached login reuse, and passed the live provider test: a direct tool challenge
created no files, and one complete evolution iteration applied and evaluated a
generated candidate. See the [ACP quickstart](../antigravity_acp_quickstart/) for
installation, login, configuration, and tests. The separate `antigravity_cli`
provider retains its initialization compatibility check.

## Results

| Check | Result |
| --- | --- |
| CLI stream-json custom agent with empty or `finish`-only tools | Initialization advertised the full tool set; the provider rejected it before sending a prompt. |
| `agy --hub` startup and model discovery | Local HTTP/Connect RPC worked with the hub's CSRF token and existing CLI login. |
| Hub `forceDisableToolCalling: true` | A live model turn made two tool calls and created the requested temporary sentinel file. |
| Hub conversational mode with model-default overrides disabled | The same sentinel test made two tool calls and created the file. |
| Hub pure `builtinAgent` configuration carrying the restriction | Accepted the specification, then made two tool calls and created the file. |
| Official ACP `initialize` | Returned protocol version 1 and agent version 1.3.0. |
| Official ACP `session/new` with `enabledTools: ["finish"]` | Created a session and persisted the exact filter in its metadata sidecar. |
| Official ACP model selection | Personal OAuth model catalog listed Flash Medium; explicit selection and authenticated generation succeeded. |
| Official ACP personal OAuth login (2026-10-05) | Completed successfully; persistent credentials were saved with permissions `0600`. |
| Official ACP cached login in a fresh process | Authentication succeeded without requesting a browser sign-in. |
| Official ACP live tool-restriction challenge | Returned `TOOLS_DISABLED`, reported no tool activity, and created no workspace files. |
| Official ACP evolution iteration | Generated a candidate that OpenEvolve applied and evaluated successfully. |
| Official ACP 10-iteration trial | Completed with 10 valid candidates, no ACP retries/failures, and best combined score 1.4995496974. |

The three hub generation probes asked the model to create a harmless file inside
a fresh temporary directory, or return `TOOLS_DISABLED` if it could not use
tools. All three created the file. Temporary workspaces and probe processes were
cleaned up. AGY may retain its own conversation history.

The ACP session probe used a nonfunctional test API key in a temporary profile.
It sent no `session/prompt`, used no real credentials, and did not prove that
generation would succeed. Successful session creation and recorded settings are
different checks from runtime tool enforcement.
The later authenticated checks exercised the implemented provider rather than
relying on the dummy-key session result.

## What `--hub` provides

In the tested CLI, hub startup required:

```bash
AGY_ENABLE_HUB=1 agy --hub --hub-port PORT
```

This starts an HTTP UI and its Connect RPC backend, rather than an ACP JSON-RPC
server on stdin/stdout. The inspected UI backend reported version 2.19.1.
`--headless` was not an accepted CLI flag.

The local RPC path was:

```text
/exa.language_server_pb.LanguageServerService/<Method>
```

`GetCascadeModelConfigData`, `StartCascade`, `SendUserCascadeMessage`, and
`GetCascadeTrajectory` allowed programmatic model discovery and generation.
The UI supplied a CSRF token for the `x-codeium-csrf-token` header. Requests
without it returned HTTP 401.

Model identifiers must be resolved from the catalog. The internal model enum
observed for Flash Medium is not a stable model identifier for configuration.

The tested restriction was
`cascadeConfig.plannerConfig.toolConfig.forceDisableToolCalling: true`, with
user customization inheritance disabled. Passing it through a pure builtin-agent
specification also failed the live sentinel test. Arbitrary custom-agent
specifications were rejected unless routed through an SDK executable.
These results make the tested hub paths unsuitable for OpenEvolve's current
code-return workflow, where OpenEvolve applies and evaluates changes itself.
They do not establish that every possible hub/SDK configuration has been tested.

## Official ACP route

Google distributes a separate stdio ACP server through the
[ACP registry](https://github.com/agentclientprotocol/registry/blob/main/antigravity-acp/agent.json).
Google also documents the associated
[IDE integrations](https://antigravity.google/docs/ide/extensions).
The inspected Linux x86_64 artifact was
[Antigravity ACP 1.3.0](https://dl.google.com/agy-extensions/releases/linux/agy-acp-server-1.3.0-linux-x86_64.zip).

Its ZIP archive SHA-256 was:

```text
9fb60956af0a9d76220a4db91ca9ac88e2a2372ad68f985ab5fceace6b825b96
```

Keep `localharness_external` beside `agy_acp_server.par`, with both executable.
The registry's Linux command is `./agy_acp_server.par --uid=`. The server uses
newline-delimited JSON-RPC over stdin/stdout.

Initialize without client filesystem or terminal tools:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": 1,
    "clientCapabilities": {
      "fs": {"readTextFile": false, "writeTextFile": false},
      "terminal": false
    },
    "clientInfo": {"name": "openevolve", "version": "0.1"}
  }
}
```

After authenticating, create a fresh session with the vendor tool filter:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "session/new",
  "params": {
    "cwd": "/absolute/path/to/empty/temporary/workspace",
    "mcpServers": [],
    "_meta": {"agy": {"enabledTools": ["finish"]}}
  }
}
```

`finish` is a completion control tool. In the inspected server, the filter
disables other built-in tools, the replacement file-view tool, and subagents.
An explicit empty allowlist also stays empty instead of falling back to defaults.
The SDK translates the resulting capabilities into explicit disabled harness
settings for commands, tasks, scheduling, file access, search, web access,
images, questions, and subagents. The later authenticated tool challenge also
passed through the implemented provider's session setup.

The relevant inspected modules inside the published archive are:

- `google3/cloud/developer_experience/antigravity_extensions/acp_server/tool_filter.py`
- `google3/cloud/developer_experience/antigravity_extensions/acp_server/server.py`
- `google3/cloud/developer_experience/antigravity_extensions/acp_server/paths.py`
- `google3/third_party/py/google/antigravity/connections/local/local_connection.py`

The tool filter does not disable client filesystem callbacks or global MCP
servers. False client filesystem capabilities remove the callbacks. Empty
`mcpServers` must be combined with a dedicated clean `GEMINI_HOME`, because the
server merges profile MCP configuration and loads profile hooks and skills.
An empty temporary working directory avoids workspace customizations.
Reject permission requests and unexpected client filesystem/terminal RPCs as an
additional check; permission denial alone does not disable all tools.

The response to `session/new` does not echo tool-filter support. The probe checked
the persisted session metadata, and the implementation was inspected in the
pinned artifact. Arbitrary ACP servers can ignore vendor metadata; accepting an
unknown server version would not establish this restriction.

## Authentication and model selection

The inspected server keeps its Linux OAuth credentials in
`$GEMINI_HOME/antigravity-acp/acp_token.json`, separate from the CLI's
`antigravity-cli` credentials. Existing CLI authentication was not automatically
reused. Use one-time ACP authentication in a dedicated profile; do not copy or
rewrite the CLI credential store.

The server advertised personal OAuth, business OAuth, Gemini API key, and agent
platform authentication methods. Use the appropriate advertised authentication
flow. A temporary test API key only allowed local session construction.

Read the authenticated session's model options, explicitly select
`gemini-3.8-flash-medium`, and check the returned current value. The selection
request is:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "session/set_config_option",
  "params": {
    "sessionId": "SESSION_ID",
    "configId": "model",
    "value": "gemini-3.8-flash-medium"
  }
}
```

Do not assume the dummy-key catalog proves that an OAuth account can use the
model, or silently accept the default model.

## Implemented OpenEvolve integration

The provider is implemented in
[`antigravity_acp.py`](../../openevolve/llm/antigravity_acp.py), with installer and
login commands in [`openevolve/antigravity_acp.py`](../../openevolve/antigravity_acp.py).
It pins server version 1.3.0, verifies persisted filtering before the prompt,
explicitly selects the model, and accepts only a completed assistant response.
It rejects client/tool activity and browser authentication during evolution.
Retries create fresh sessions and workspaces; timeout and cancellation clean up
the process group. Configuration paths propagate to evaluation workers.

Offline tests cover protocol interleaving, restrictions, setup failures, retries,
installation checksums, and cleanup. The explicitly opt-in live test challenges
tool restrictions before running a complete function-minimization iteration.
The [ACP quickstart](../antigravity_acp_quickstart/) contains the current operational
instructions and verification results.
