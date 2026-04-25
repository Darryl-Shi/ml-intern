"""
Sandbox tools — expose the Sandbox client as agent tools.

5 tools total:
  sandbox_create — explicit sandbox creation (requires approval)
  bash, read, write, edit — operations on the sandbox

If any operation tool is called without an active sandbox,
a cpu-basic sandbox is auto-created (no approval needed).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import threading
from copy import deepcopy
from functools import lru_cache
from typing import Any

from agent.core.session import Event
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
    session: Any, hardware: str = "cpu-basic", **create_kwargs
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

    # Telemetry: sandbox creation (infra consumption signal)
    from agent.core import telemetry
    await telemetry.record_sandbox_create(
        session, sb, hardware=hardware,
        create_latency_s=int(_t.monotonic() - _t_start),
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
def _skypilot_hardware_options() -> list[str]:
    """Best-effort import of all hardware options from SkyPilot's GPU catalog."""
    options = {"cpu-basic", "cpu-upgrade"}

    env_options = os.environ.get("SKYPILOT_ACCELERATOR_OPTIONS")
    if env_options:
        options.update(x.strip() for x in env_options.split(",") if x.strip())

    sky_bin = shutil.which("sky")
    if sky_bin:
        try:
            proc = subprocess.run(
                [sky_bin, "show-gpus", "-a"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if proc.returncode == 0:
                options.update(_parse_sky_show_gpus(proc.stdout))
        except Exception:
            pass

    return sorted(options)


def _parse_sky_show_gpus(output: str) -> set[str]:
    options: set[str] = set()
    for raw in output.splitlines():
        line = raw.strip()
        if not line or line.startswith("-") or "AVAILABLE_QUANTITIES" in line:
            continue
        if line.startswith(("COMMON_GPU", "GOOGLE_TPU", "OTHER_GPU")):
            continue
        parts = re.split(r"\s{2,}", line)
        if not parts:
            continue
        name = parts[0].strip()
        if not name or name.lower().startswith(("gpu", "tpu", "name")):
            continue
        options.add(name)
        if len(parts) > 1:
            for qty in re.findall(r"\d+(?:\.\d+)?", parts[1]):
                options.add(f"{name}:{qty}")
    return options


def _hardware_schema() -> dict[str, Any]:
    options = _skypilot_hardware_options()
    schema: dict[str, Any] = {
        "type": "string",
        "description": (
            "SkyPilot accelerator spec for the sandbox (default: cpu-basic). "
            "Use any SkyPilot-supported value, e.g. A100:1, A100-80GB:1, "
            "H100:1, RTX4090:1, L40S:1, T4:1. CPU aliases: cpu-basic, cpu-upgrade."
        ),
    }
    if len(options) > 2:
        schema["enum"] = options
    return schema


def _base_hardware_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "description": (
            "SkyPilot accelerator spec for the sandbox (default: cpu-basic). "
            "Use any SkyPilot-supported value, e.g. A100:1, A100-80GB:1, "
            "H100:1, RTX4090:1, L40S:1, T4:1. CPU aliases: cpu-basic, cpu-upgrade."
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
        "For ML code that uses CUDA, bf16, or model loading: use GPU hardware (T4:1 minimum). "
        "CPU sandboxes cannot run GPU code paths — your test will not catch GPU-related errors.\n\n"
        "Before choosing hardware, estimate your VRAM needs (models you run, training data size). Rule of thumb: bf16/fp16 ≈ 2 bytes/param, "
        "fp32 ≈ 4 bytes/param, plus ~20% overhead for optimizer states during training.\n"
        "Common picks: T4:1 (16GB VRAM, fits ≤1-3B), A10G:1 (24GB, ≤7B), A100-80GB:1 (80GB, ≤30B). "
        "If the model won't fit, pick larger hardware upfront — OOM on a sandbox wastes time.\n\n"
        "Hardware uses SkyPilot accelerator specs. The default infra is RunPod.\n"
    ),
    "parameters": {
        "type": "object",
        "required": [],
        "additionalProperties": False,
        "properties": {
            "hardware": _base_hardware_schema(),
            "infra": {
                "type": "string",
                "description": "SkyPilot infra selector for the sandbox (default: runpod).",
                "default": "runpod",
            },
            "private": {
                "type": "boolean",
                "description": "Compatibility no-op; SkyPilot sandboxes are clusters, not HF Spaces.",
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

    hardware = args.get("hardware", "cpu-basic")
    create_kwargs = {"infra": args.get("infra", "runpod")}

    try:
        sb, error = await _ensure_sandbox(session, hardware=hardware, **create_kwargs)
    except Exception as e:
        return f"Failed to create sandbox: {e}", False

    if error:
        return error, False

    return (
        f"Sandbox created: {sb.space_id}\n"
        f"URL: {sb.url}\n"
        f"Hardware: {hardware}\n"
        f"Infra: {create_kwargs['infra']}\n"
        f"Use bash/read/write/edit to interact with it."
    ), True


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
    from agent.core.tools import ToolSpec

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
