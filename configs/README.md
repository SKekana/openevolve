# OpenEvolve Configuration Files

This directory contains configuration files for OpenEvolve with examples for different use cases.

## Configuration Files

### `default_config.yaml`
The main configuration file containing all available options with sensible defaults. This file includes:
- Complete documentation for all configuration parameters
- Default values for all settings
- **Island-based evolution parameters** for proper evolutionary diversity

Use this file as a template for your own configurations.

### `island_config_example.yaml`
A practical example configuration demonstrating proper island-based evolution setup. Shows:
- Recommended island settings for most use cases
- Balanced migration parameters
- Complete working configuration

### `island_examples.yaml`
Multiple example configurations for different scenarios:
- **Maximum Diversity**: Many islands, frequent migration
- **Focused Exploration**: Few islands, rare migration  
- **Balanced Approach**: Default recommended settings
- **Quick Exploration**: Small-scale rapid testing
- **Large-Scale Evolution**: Complex optimization runs

Includes guidelines for choosing parameters based on your problem characteristics.

## Codex CLI provider

Set `llm.provider: codex_cli` and configure explicit model names. The
[Codex CLI quickstart](../examples/codex_cli_quickstart/) provides a complete
configuration with 30% `gpt-6.1-sol` (`high` reasoning, 600-second timeout) and
70% `gpt-6-luna` (`medium` reasoning, 300-second timeout).

The provider uses saved Codex CLI authentication and supports model weights,
`reasoning_effort`, `timeout`, `retries`, and `retry_delay`. It ignores personal
Codex configuration and returns proposed changes for OpenEvolve to apply.
Sampling parameters `temperature`, `top_p`, `max_tokens`, and generation seeding
are unsupported. See the quickstart for installation, execution settings, and
opt-in live tests.

## OpenCode CLI provider

Set `llm.provider: opencode_cli` with explicit OpenCode `provider/model` names.
The [OpenCode CLI quickstart](../examples/opencode_cli_quickstart/) provides a
10-iteration configuration with 30% GLM 5.3 (600-second timeout) and 70% GLM 5.3
Flash (300-second timeout) through account-specific Modal endpoints. Replace
these IDs with models available to your OpenCode account when needed.

The provider uses saved OpenCode authentication and effective built-in or custom
provider definitions, then isolates generation configuration. Model weights,
system messages, timeouts, retries, retry delay, `temperature`, and `top_p` are
supported. Optional `reasoning_effort` selects the provider-specific OpenCode
`--variant`; omit it when a model has no applicable variant. `max_tokens` and
generation seeding are unsupported. Configure provider endpoints and credentials
in OpenCode; OpenEvolve's `api_base` and `api_key` are not forwarded. The quickstart
documents installation, model discovery, token usage, and opt-in live tests.

## Island-Based Evolution Parameters

The key new parameters for proper evolutionary diversity are:

```yaml
database:
  num_islands: 5                      # Number of separate populations
  migration_interval: 50              # Migrate every N generations  
  migration_rate: 0.1                 # Fraction of top programs to migrate
  # island_initial_programs: [null, "seeds/greedy.py"]  # Optional per-island starting programs
```

### Parameter Guidelines

- **num_islands**: 3-10 for most problems (more = more diversity)
- **migration_interval**: 25-100 generations (higher = more independence)
- **migration_rate**: 0.05-0.2 (5%-20%, higher = faster knowledge sharing)
- **island_initial_programs**: Start islands from different programs, e.g. distinct
  algorithmic approaches. Entry *i* is the program island *i* starts from; `null` or a
  missing entry keeps the default (the island starts empty and takes its first parent
  from the other islands). Entry 0 must be `null`, because island 0 always starts from
  the initial program given on the command line, and the list may not be longer than
  `num_islands`. Relative paths resolve against the config file's directory. Each seed is
  evaluated once at the start of a fresh run; resuming from a checkpoint does not add
  them again.

### When to Use What

- **Complex problems** → More islands, less frequent migration
- **Simple problems** → Fewer islands, more frequent migration
- **Long runs** → More islands to maintain diversity
- **Short runs** → Fewer islands for faster convergence

## Usage

Copy any of these files as a starting point for your configuration:

```bash
cp configs/default_config.yaml my_config.yaml
# Edit my_config.yaml for your specific needs
```

Then use with OpenEvolve:

```python
from openevolve import OpenEvolve
from openevolve.config import Config

config = Config()
config.from_yaml("my_config.yaml")
evolve = OpenEvolve(
    initial_program_path="program.py",
    evaluation_file="evaluator.py", 
    config=config
)
```

## Antigravity ACP provider

Set `llm.provider: antigravity_acp` with an explicit model name. The
[ACP quickstart](../examples/antigravity_acp_quickstart/) uses Google's official
server 1.3.0, personal Google OAuth, `gemini-3.8-flash-medium`, a 300-second timeout,
and two retries. Install and authenticate with:

```bash
python -m openevolve.antigravity_acp install
python -m openevolve.antigravity_acp login
```

`antigravity_acp_path` and `antigravity_acp_home` select an alternate executable
and dedicated persistent profile. They can be set at `llm` level or per model;
per-model values take precedence and survive worker serialization. Environment
defaults are `OPENEVOLVE_ANTIGRAVITY_ACP_BIN` and `OPENEVOLVE_ANTIGRAVITY_ACP_HOME`.
The quickstart documents the default paths and matching login options.

The provider verifies a `finish`-only built-in tool filter, disables client tools,
and rejects profile hooks, MCP configuration, and nonempty skill directories.
Each attempt uses a new session and temporary workspace. Runtime authentication
does not open a browser. Model weights, timeout, retries, retry delay, and system
messages are supported. Omit `reasoning_effort`; the model slug selects effort.
Sampling parameters, API credentials/base URLs, and other providers' budget/tool
options are not forwarded. ACP context usage is not reported as per-call token
usage or monetary cost.

## Antigravity CLI provider

Set `llm.provider: antigravity_cli` with an explicit model name. The
[quickstart](../examples/antigravity_cli_quickstart/) uses
`gemini-3.8-flash-medium`, a 300-second timeout, and two retries. Existing weights,
reasoning effort, timeout, and retry settings work without an API key.

AGY must confirm the requested generation-only agent and restricted tools in its
initialization event. AGY 1.2.16 fails this check; the adapter sends no prompt and
does not retry. The quickstart describes the limitation and the opt-in live test.
Global AGY settings are not modified; CLI-managed history and startup integrations
can still be used by AGY. Sampling parameters are not forwarded.
