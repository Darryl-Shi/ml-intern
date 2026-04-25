<p align="center">
  <img src="frontend/public/smolagents.webp" alt="smolagents logo" width="160" />
</p>

# ML Intern

ML Intern is an agent for ML engineering work. It can research Hugging Face
docs, papers, models, datasets, and repositories, write code, run local tools,
run SkyPilot sandboxes on RunPod, and keep a resumable session trace while it works.

The runtime is provider-neutral: bring any OpenAI-compatible chat completions
endpoint, model name, base URL, and API key. The CLI is the primary interface;
the web app uses the same backend and provider contract.

## Quick Start

Install with UV:

```bash
git clone git@github.com:huggingface/ml-intern.git
cd ml-intern
uv sync
uv tool install -e .
```

Start the CLI:

```bash
ml-intern
```

On first launch, if no provider is configured, ML Intern opens a provider setup
wizard. It saves the result to:

```text
~/.config/ml-intern/provider.json
```

The saved file includes the API key and is written with file mode `0600`.

You can also configure the provider with environment variables:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=<your-provider-api-key>
OPENAI_MODEL=gpt-4o-mini
OPENAI_CONTEXT_WINDOW=200000 # optional
```

`OPENAI_*` environment variables take precedence over the saved provider file.

## Optional Tokens

ML Intern can start without a Hugging Face token. Some tools are more useful
when authenticated:

```bash
HF_TOKEN=<your-hugging-face-token>
GITHUB_TOKEN=<github-personal-access-token>
```

Use `HF_TOKEN` for private Hub access and authenticated HF API calls. Use
`GITHUB_TOKEN` for GitHub search or repository operations that need auth.

## CLI Usage

Interactive mode:

```bash
ml-intern
```

Headless mode with auto-approval:

```bash
ml-intern "fine-tune llama on my dataset"
```

Common options:

```bash
ml-intern --model gpt-4o-mini "your prompt"
ml-intern --max-iterations 100 "your prompt"
ml-intern --no-stream "your prompt"
ml-intern --resume session_logs/session_<id>_<timestamp>.json
ml-intern --resume session_logs/session_<id>_<timestamp>.json --restore-local-state
```

`--model` changes only the model name for that run. The base URL and API key
still come from the environment, saved provider file, or interactive setup.

### Resume and Restore

ML Intern saves session trajectories under `session_logs/`. In CLI mode it also
creates working-directory checkpoints under `session_logs/snapshots/` at the
start of the session and after each completed turn. The checkpoints exclude
heavy or generated directories such as `.git`, `.venv`, `node_modules`,
`__pycache__`, `.pytest_cache`, and `session_logs`.

Resume the exact message history from a prior session:

```bash
ml-intern --resume session_logs/session_<id>_<timestamp>.json
```

Resume and restore the local directory to the latest saved checkpoint:

```bash
ml-intern --resume session_logs/session_<id>_<timestamp>.json --restore-local-state
```

Inside an interactive session, use `/history` to list restore points and
`/restore <turn>` to restore both the conversation and local files to that
turn. `/undo` also rolls the local files back to the previous available
checkpoint.

For a Codex-style edit/resume flow, press `Esc` twice at the prompt. ML Intern
shows prior user messages, restores the conversation and local files to just
before the selected message, and places that message back into the prompt so
you can edit or resend it. `/rewind <message-number>` is the command fallback.

### Slash Commands

Inside the CLI:

| Command | Purpose |
| --- | --- |
| `/provider` | Show the active OpenAI-compatible provider. |
| `/provider setup` | Configure base URL, model, API key, and context window. |
| `/model` | Show the current model. |
| `/model <name>` | Switch the model for the current provider. |
| `/effort <level>` | Set reasoning effort: `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or `off`. |
| `/compact` | Summarize older context and continue in the same session. |
| `/undo` | Remove the last turn from context and restore local files to the prior checkpoint. |
| `/history` | List turn checkpoints available for restore. |
| `/restore <turn>` | Restore conversation history and local files to a turn checkpoint. |
| `/rewind <message>` | Rewind to before a prior user message and put it back in the prompt. |
| `Esc Esc` | Open the rewind chooser. |
| `/status` | Show model, provider, reasoning effort, turns, and context items. |
| `/yolo` | Toggle auto-approval for tool calls. |
| `/help` | Show CLI help. |

## Web App

Install frontend dependencies:

```bash
cd frontend
npm install
```

Build the frontend:

```bash
npm run build
```

Run the backend from the repo root:

```bash
cd backend
uv run uvicorn main:app --host 0.0.0.0 --port 7860
```

The production Docker image builds the frontend into `static/` and serves the
FastAPI app on port `7860`.

The browser stores provider settings in local storage and sends them to the
backend when creating or updating sessions. The backend redacts API keys in
session metadata responses.

## Provider Resolution

Provider settings are resolved in this order:

1. `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, and optional `OPENAI_CONTEXT_WINDOW`.
2. Runtime/session config passed by the web app.
3. `~/.config/ml-intern/provider.json`.

The provider must expose an OpenAI-compatible chat completions API. Tool calls
are sent using the OpenAI `tools` schema.

## Configuration

The default config lives at:

```text
configs/main_agent_config.json
```

Minimal provider-aware config:

```json
{
  "model_name": "${OPENAI_MODEL:-gpt-4o-mini}",
  "openai_base_url": "${OPENAI_BASE_URL:-}",
  "openai_api_key": "${OPENAI_API_KEY:-}",
  "openai_context_window": 200000,
  "mcpServers": {}
}
```

Environment variables in the form `${VAR}` and `${VAR:-default}` are expanded
from `.env` and the shell environment.

## Architecture

The core loop is queue-based:

```text
CLI or Web UI
  -> submission queue
  -> agent loop
  -> OpenAI-compatible provider
  -> tool calls
  -> context manager
  -> event queue
  -> CLI renderer or SSE stream
```

Main pieces:

| Path | Role |
| --- | --- |
| `agent/main.py` | First-class CLI, slash commands, provider setup, headless mode. |
| `agent/core/provider.py` | Provider config, saved config, OpenAI SDK client, chat completion wrapper. |
| `agent/core/message.py` | Internal message and tool-call models serialized to OpenAI chat dictionaries. |
| `agent/core/agent_loop.py` | Iterative agent loop, streaming, tool execution, approvals, retries. |
| `agent/context_manager/manager.py` | Message history, compaction, restore summaries, system prompt rendering. |
| `agent/core/tools.py` | Tool router and tool spec conversion. |
| `backend/` | FastAPI app, auth, sessions, SSE, provider-aware REST endpoints. |
| `frontend/` | React UI, session management, chat transport, browser provider settings. |

## Events

The agent emits events through an in-memory event queue. The CLI renders them
directly; the web backend streams them as SSE.

Common event types:

- `processing`
- `ready`
- `assistant_chunk`
- `assistant_message`
- `assistant_stream_end`
- `tool_call`
- `tool_output`
- `tool_log`
- `tool_state_change`
- `approval_required`
- `turn_complete`
- `error`
- `interrupted`
- `compacted`
- `undo_complete`
- `shutdown`

## Development

Use UV for Python interpreter and dependency management:

```bash
uv sync --extra dev
uv run --extra dev pytest tests/unit
uv run python -m compileall agent backend
```

Frontend checks:

```bash
cd frontend
npm run build
```

Refresh the lockfile after dependency changes:

```bash
uv lock
```

## Adding Tools

Built-in tools are registered through `agent/core/tools.py`. A tool needs a
name, description, JSON schema, and async handler:

```python
ToolSpec(
    name="your_tool",
    description="What your tool does",
    parameters={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "Parameter description"}
        },
        "required": ["param"],
    },
    handler=your_async_handler,
)
```

## Adding MCP Servers

Add MCP servers to `configs/main_agent_config.json`:

```json
{
  "mcpServers": {
    "your-server-name": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${YOUR_TOKEN}"
      }
    }
  }
}
```

Secrets can be referenced with environment variables and loaded from `.env`.
