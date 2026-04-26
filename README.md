# ML Intern MCP

ML Intern is now a specialist MCP server for agentic CLIs. It exposes
Hugging Face research, documentation, dataset, repository, GitHub example, and
remote RunPod sandbox tools over stdio.

The host CLI remains responsible for normal local shell and filesystem access.
This server deliberately does not expose local `bash`, `read`, `write`, or
`edit` tools.

## Tools

- `explore_hf_docs`
- `fetch_hf_docs`
- `find_hf_api`
- `hf_papers`
- `hf_inspect_dataset`
- `hf_repo_files`
- `hf_repo_git`
- `github_find_examples`
- `github_list_repos`
- `github_read_file`
- `sandbox_create`
- `sandbox_bash`
- `sandbox_read`
- `sandbox_write`
- `sandbox_edit`

Sandbox tools are remote-only. Call `sandbox_create` before
`sandbox_bash`, `sandbox_read`, `sandbox_write`, or `sandbox_edit`.

## Install

Use UV for Python interpreter and dependency management:

```bash
uv sync
uv tool install -e .
```

Run from a checkout:

```bash
uv run ml-intern-mcp
```

Run after tool install:

```bash
ml-intern-mcp
```

## Authentication

The server reads auth from environment variables:

```bash
export HF_TOKEN=...
export GITHUB_TOKEN=...
export S2_API_KEY=...
```

RunPod sandbox tools use your RunPod environment configuration
(RUNPOD_API_KEY).

## MCP Config

Claude Code project scope:

```bash
claude mcp add ml-intern --scope project -- uv run ml-intern-mcp
```

Claude Code user scope, after `uv tool install -e .`:

```bash
claude mcp add ml-intern --scope user -- ml-intern-mcp
```

Codex `~/.codex/config.toml`:

```toml
[mcp_servers.ml-intern]
command = "uv"
args = ["run", "ml-intern-mcp"]
env = { HF_TOKEN = "${HF_TOKEN}", GITHUB_TOKEN = "${GITHUB_TOKEN}", S2_API_KEY = "${S2_API_KEY}" }
```

Generic `.mcp.json` stdio config:

```json
{
  "mcpServers": {
    "ml-intern": {
      "command": "uv",
      "args": ["run", "ml-intern-mcp"],
      "env": {
        "HF_TOKEN": "${HF_TOKEN}",
        "GITHUB_TOKEN": "${GITHUB_TOKEN}",
        "S2_API_KEY": "${S2_API_KEY}"
      }
    }
  }
}
```

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest tests/unit
uv run python -m compileall agent
```

Refresh the lockfile after dependency changes:

```bash
uv lock
```
