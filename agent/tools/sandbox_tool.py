"""
Sandbox tools — expose the Sandbox client as agent tools.

5 tools total:
  sandbox_create — explicit RunPod sandbox creation (requires approval)
  bash, read, write, edit — operations on the sandbox
"""

from __future__ import annotations

import asyncio
import os
import threading
from copy import deepcopy
from functools import lru_cache
from typing import Any

from agent.mcp_types import Event, ToolSpec
from agent.tools.sandbox_client import Sandbox


def _looks_like_path(script: str) -> bool:
    """Return True if the script string looks like a file path (not inline code)."""
    return (
        isinstance(script, str)
        and script.strip() == script
        and not any(c in script for c in "\r\n\0")
        and (
            script.startswith("/")
            or script.startswith("./")
            or script.startswith("../")
        )
    )


async def resolve_sandbox_script(
    sandbox: Any, script: str
) -> tuple[str | None, str | None]:
    """Read a file from the sandbox if *script* looks like a path.

    Returns:
        (content, error) — content is the file text on success,
        error is a message on failure.  Both None means *script*
        is not a path (caller should use it as-is).
    """
    if not sandbox or not _looks_like_path(script):
        return None, None
    try:
        # Use the read endpoint instead of bash("cat ...") which truncates at 25KB.
        result = await asyncio.to_thread(sandbox.read, script, limit=100_000)
        if result.success and result.output:
            # Strip line number prefixes (read returns "N\tcontent" format)
            lines = []
            for line in result.output.split("\n"):
                parts = line.split("\t", 1)
                lines.append(parts[1] if len(parts) == 2 else line)
            return "\n".join(lines), None
        return None, f"Failed to read {script} from sandbox: {result.error}"
    except Exception as e:
        return None, f"Failed to read {script} from sandbox: {e}"


# ── Tool name mapping (short agent names → Sandbox client names) ──────


async def _ensure_sandbox(
    session: Any, hardware: str = "RTX4090:1", **create_kwargs
) -> tuple[Sandbox | None, str | None]:
    """
    Ensure a sandbox exists on the session. Auto-creates with given hardware if needed.

    Returns:
        (sandbox, error_message) — one will be None.
    """
    if session and getattr(session, "sandbox", None):
        return session.sandbox, None

    if not session:
        return None, "No session available."

    token = session.hf_token
    owner = getattr(session, "session_id", None)

    await session.send_event(
        Event(
            event_type="tool_log",
            data={
                "tool": "sandbox",
                "log": (
                    "Auto-creating SkyPilot sandbox "
                    f"({create_kwargs.get('infra', 'runpod')}, {hardware})..."
                ),
            },
        )
    )

    # Thread-safe log callback: posts tool_log events from the worker thread
    loop = asyncio.get_running_loop()

    def _log(msg: str) -> None:
        loop.call_soon_threadsafe(
            session.event_queue.put_nowait,
            Event(event_type="tool_log", data={"tool": "sandbox", "log": msg}),
        )

    # Bridge asyncio cancel event to a threading.Event for the blocking create call.
    # We poll session._cancelled from the main loop in a background task and set
    # a threading.Event that Sandbox.create checks during its polling loops.
    cancel_flag = threading.Event()

    async def _watch_cancel():
        await session._cancelled.wait()
        cancel_flag.set()

    watcher_task = asyncio.create_task(_watch_cancel())

    kwargs = {
        "owner": owner,
        "hardware": hardware,
        "token": token,
        "secrets": {"HF_TOKEN": token} if token else {},
        "log": _log,
        "cancel_event": cancel_flag,
        **create_kwargs,
    }
    import time as _t
    _t_start = _t.monotonic()
    try:
        sb = await asyncio.to_thread(Sandbox.create, **kwargs)
    except Sandbox.Cancelled:
        return None, "Sandbox creation cancelled by user."
    finally:
        watcher_task.cancel()
    session.sandbox = sb

    create_latency_s = int(_t.monotonic() - _t_start)
    session._sandbox_created_at = _t.monotonic() - create_latency_s
    await session.send_event(
        Event(
            event_type="sandbox_create",
            data={
                "sandbox_id": getattr(sb, "space_id", None),
                "hardware": hardware,
                "create_latency_s": create_latency_s,
            },
        )
    )

    await session.send_event(
        Event(
            event_type="tool_log",
            data={
                "tool": "sandbox",
                "log": f"SkyPilot sandbox ready: {sb.space_id} ({sb.url})",
            },
        )
    )

    return sb, None


@lru_cache(maxsize=1)
def _runpod_hardware_options() -> list[str]:
    """RunPod GPU options in SkyPilot accelerator syntax.

    Keep this list RunPod-specific. Do not scrape ``sky show-gpus -a`` here:
    that command returns SkyPilot's global accelerator catalog across clouds,
    which leaks irrelevant infrastructure choices into the MCP schema.
    """
    gpu_families = {
        "RTX3070",
        "RTX3080",
        "RTX3090",
        "RTX4000",
        "RTX4000Ada",
        "RTX4090",
        "RTX5000Ada",
        "RTX5090",
        "RTX6000Ada",
        "A30",
        "A40",
        "A100",
        "A100-80GB",
        "B200",
        "H100",
        "H100-80GB",
        "H200",
        "L4",
        "L40",
        "L40S",
        "V100",
    }
    options = {f"{gpu}:{count}" for gpu in gpu_families for count in (1, 2, 4, 8)}

    env_options = os.environ.get("RUNPOD_ACCELERATOR_OPTIONS")
    if env_options:
        options.update(x.strip() for x in env_options.split(",") if x.strip())

    return sorted(options)


def _hardware_schema() -> dict[str, Any]:
    options = _runpod_hardware_options()
    return {
        "type": "string",
        "description": (
            "RunPod GPU accelerator in SkyPilot syntax (default: RTX4090:1). "
            "Use RunPod GPU families such as RTX4090:1, L40S:1, A40:1, "
            "A100-80GB:1, H100:1, H200:1. For multiple GPUs, change the suffix "
            "where supported, e.g. L40S:2 or A100-80GB:4."
        ),
        "enum": options,
    }


def _base_hardware_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "description": (
            "RunPod GPU accelerator in SkyPilot syntax (default: RTX4090:1). "
            "Examples: RTX4090:1, L40S:1, A40:1, A100-80GB:1, H100:1, H200:1."
        ),
    }


# ── sandbox_create tool ──────────────────────────────────────────────

SANDBOX_CREATE_TOOL_SPEC = {
    "name": "sandbox_create",
    "description": (
        "Create a persistent SkyPilot sandbox on RunPod for developing and testing scripts.\n\n"
        "Workflow: sandbox_create -> write script -> pip install -> test with small run -> fix errors -> run the full job in the sandbox.\n"
        "The sandbox persists across tool calls within the session. pip install works out of the box.\n\n"
        "Use this when: you need to develop, test, iterate on, or run ML scripts on SkyPilot/RunPod. "
        "Especially for training scripts where you need to verify imports, test on a small subset, and fix errors interactively.\n\n"
        "Skip this when: the task is a simple one-shot operation (status check, resource search, quick data query), "
        "or the script is copied from a verified working example with minimal changes.\n\n"
        "RunPod sandboxes are GPU-oriented. For ML code that uses CUDA, bf16, "
        "or model loading, choose a GPU with enough VRAM.\n\n"
        "Before choosing hardware, estimate your VRAM needs (models you run, training data size). Rule of thumb: bf16/fp16 ≈ 2 bytes/param, "
        "fp32 ≈ 4 bytes/param, plus ~20% overhead for optimizer states during training.\n"
        "Common RunPod picks: RTX4090:1 (24GB VRAM, small models/prototyping), L40S:1 (48GB, medium workloads), A100-80GB:1 or H100:1 (80GB, larger training/inference). "
        "If the model won't fit, pick larger hardware upfront — OOM on a sandbox wastes time.\n\n"
        "Hardware uses SkyPilot accelerator syntax for RunPod GPUs. The infra is fixed to RunPod.\n"
    ),
    "parameters": {
        "type": "object",
        "required": [],
        "additionalProperties": False,
        "properties": {
            "hardware": _base_hardware_schema(),
            "private": {
                "type": "boolean",
                "description": "Compatibility no-op; RunPod sandboxes are clusters, not HF Spaces.",
            },
        },
    },
}


async def sandbox_create_handler(
    args: dict[str, Any], session: Any = None
) -> tuple[str, bool]:
    """Handle sandbox_create tool calls."""
    # If sandbox already exists, return its info
    if session and getattr(session, "sandbox", None):
        sb = session.sandbox
        return (
            f"Sandbox already active: {sb.space_id}\n"
            f"URL: {sb.url}\n"
            f"Use bash/read/write/edit to interact with it."
        ), True

    hardware = args.get("hardware", "RTX4090:1")
    create_kwargs = {"infra": "runpod"}

    try:
        sb, error = await _ensure_sandbox(session, hardware=hardware, **create_kwargs)
    except Exception as e:
        return f"Failed to create sandbox: {_format_create_error(e)}", False

    if error:
        return error, False

    return (
        f"Sandbox created: {sb.space_id}\n"
        f"URL: {sb.url}\n"
        f"Hardware: {hardware}\n"
        f"Infra: {create_kwargs['infra']}\n"
        f"Use bash/read/write/edit to interact with it."
    ), True


def _format_create_error(error: Exception) -> str:
    message = str(error)
    if "500 Server Error" in message and "/launch" in message:
        return (
            f"{message}\n\n"
            "SkyPilot's local API server returned an internal error while "
            "handling the launch request. Try restarting it with "
            "`uv run sky api stop` followed by `uv run sky api start`, then "
            "call sandbox_create again. Server logs are in "
            "`~/.sky/api_server/server.log`."
        )
    return message


def _make_tool_handler(sandbox_tool_name: str):
    """Factory: create a handler for a sandbox operation tool."""

    async def handler(args: dict[str, Any], session: Any = None) -> tuple[str, bool]:
        # Require sandbox to exist — user must approve sandbox_create first
        if not session or not getattr(session, "sandbox", None):
            return "No sandbox running. Call sandbox_create first to start one.", False

        sb = session.sandbox

        try:
            result = await asyncio.to_thread(sb.call_tool, sandbox_tool_name, args)
            if result.success:
                output = result.output or "(no output)"
                return output, True
            else:
                error_msg = result.error or "Unknown error"
                output = result.output
                if output:
                    return f"{output}\n\nERROR: {error_msg}", False
                return f"ERROR: {error_msg}", False
        except Exception as e:
            return f"Sandbox operation failed: {e}", False

    return handler


def get_sandbox_tools():
    """Return all 5 sandbox ToolSpecs (sandbox_create + 4 operation tools)."""
    tools = []

    # sandbox_create (explicit creation, requires approval)
    sandbox_create_spec = deepcopy(SANDBOX_CREATE_TOOL_SPEC)
    sandbox_create_spec["parameters"]["properties"]["hardware"] = _hardware_schema()
    tools.append(
        ToolSpec(
            name=sandbox_create_spec["name"],
            description=sandbox_create_spec["description"],
            parameters=sandbox_create_spec["parameters"],
            handler=sandbox_create_handler,
        )
    )

    # Operation tools (auto-execute, no approval needed)
    for name in Sandbox.TOOLS.keys():
        spec = Sandbox.TOOLS[name]
        tools.append(
            ToolSpec(
                name=name,
                description=spec["description"],
                parameters=spec["parameters"],
                handler=_make_tool_handler(name),
            )
        )

    return tools
