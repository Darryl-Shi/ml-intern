"""
Interactive CLI chat with the agent

Supports two modes:
  Interactive:  python -m agent.main
  Headless:     python -m agent.main "find me bird datasets"
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from agent.config import load_config
from agent.core.agent_loop import submission_loop
from agent.core import model_switcher
from agent.core.provider import (
    ProviderConfig,
    apply_provider_to_config,
    require_provider_config,
    resolve_provider_config,
    save_provider_config,
)
from agent.core.message import message_from_mapping
from agent.core.session import OpType
from agent.core.tools import ToolRouter
from agent.utils.terminal_display import (
    get_console,
    print_approval_header,
    print_approval_item,
    print_banner,
    print_compacted,
    print_error,
    print_help,
    print_init_done,
    print_interrupted,
    print_markdown,
    print_plan,
    print_tool_call,
    print_tool_log,
    print_tool_output,
    print_turn_complete,
    print_yolo_approve,
)

_REWIND_SENTINEL = "__ML_INTERN_REWIND__"


def _prompt_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("escape", "escape")
    def _rewind(event) -> None:
        event.current_buffer.text = _REWIND_SENTINEL
        event.current_buffer.cursor_position = len(_REWIND_SENTINEL)
        event.current_buffer.validate_and_handle()

    return bindings


_PROMPT_KEY_BINDINGS = _prompt_key_bindings()


def _safe_get_args(arguments: dict) -> dict:
    """Safely extract args dict from arguments, handling cases where LLM passes string."""
    args = arguments.get("args", {})
    # Sometimes LLM passes args as string instead of dict
    if isinstance(args, str):
        return {}
    return args if isinstance(args, dict) else {}


def _get_hf_token() -> str | None:
    """Get HF token from environment, huggingface_hub API, or cached token file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        token = api.token
        if token:
            return token
    except Exception:
        pass
    # Fallback: read the cached token file directly
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token
    return None


async def _prompt_provider_config(
    prompt_session: PromptSession,
    existing: ProviderConfig | None = None,
) -> ProviderConfig:
    """Prompt for OpenAI-compatible provider settings and persist them."""
    console = get_console()
    console.print("\n[bold]OpenAI-compatible provider setup[/bold]")
    console.print("[dim]These settings are saved to ~/.config/ml-intern/provider.json[/dim]\n")

    async def ask(label: str, default: str | None = None, password: bool = False) -> str:
        while True:
            suffix = f" [{default}]" if default and not password else ""
            value = await prompt_session.prompt_async(
                f"{label}{suffix}: ",
                is_password=password,
            )
            value = value.strip()
            if value:
                return value
            if default:
                return default
            console.print(f"[red]{label} is required.[/red]")

    base_url = await ask("Base URL", existing.base_url if existing else None)
    model = await ask("Model", existing.model if existing else None)
    api_key_default = existing.api_key if existing else None
    api_key = await ask("API key", api_key_default, password=True)
    context_default = str(existing.context_window if existing else 200_000)
    context_raw = await ask("Context window", context_default)
    try:
        context_window = max(1_000, int(context_raw))
    except ValueError:
        context_window = existing.context_window if existing else 200_000

    provider = ProviderConfig(
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        context_window=context_window,
    )
    save_provider_config(provider)
    console.print("[green]Provider saved.[/green]")
    return provider


async def _ensure_provider_config(config, prompt_session: PromptSession) -> ProviderConfig:
    provider = resolve_provider_config(config)
    if provider is None:
        provider = await _prompt_provider_config(prompt_session)
    apply_provider_to_config(config, provider)
    return provider

@dataclass
class Operation:
    """Operation to be executed by the agent"""

    op_type: OpType
    data: Optional[dict[str, Any]] = None


@dataclass
class Submission:
    """Submission to the agent loop"""

    id: str
    operation: Operation


def _create_rich_console():
    """Get the shared rich Console."""
    return get_console()


def _load_resume_trajectory(path: str | Path) -> dict[str, Any]:
    resume_path = Path(path).expanduser().resolve()
    with open(resume_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Resume file is not a session trajectory: {resume_path}")
    data["_resume_path"] = str(resume_path)
    return data


def _last_user_preview(raw_messages: list[Any]) -> str:
    for raw in reversed(raw_messages):
        if not isinstance(raw, dict) or raw.get("role") != "user":
            continue
        content = raw.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if content.startswith("[SYSTEM:"):
            continue
        return content.replace("\n", " ")[:90]
    return ""


def _session_log_candidates(
    directory: str | Path = "session_logs",
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    log_dir = Path(directory)
    if not log_dir.exists():
        return []

    candidates: list[dict[str, Any]] = []
    for path in log_dir.glob("session_*.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            messages = data.get("messages") or []
            local_state = data.get("local_state") or {}
            candidates.append(
                {
                    "path": path.resolve(),
                    "mtime": path.stat().st_mtime,
                    "session_id": data.get("session_id", "?"),
                    "model": data.get("model_name", "?"),
                    "message_count": len(messages),
                    "checkpoint_count": len(local_state.get("checkpoints") or []),
                    "last_save_time": data.get("last_save_time")
                    or data.get("session_end_time")
                    or "",
                    "preview": _last_user_preview(messages),
                }
            )
        except Exception:
            continue

    candidates.sort(key=lambda item: item["mtime"], reverse=True)
    return candidates[:limit]


def _print_resume_candidates(candidates: list[dict[str, Any]]) -> None:
    console = get_console()
    console.print("[bold]Saved sessions[/bold]")
    for index, item in enumerate(candidates, start=1):
        saved = item["last_save_time"] or "unknown time"
        preview = f" - {item['preview']}" if item["preview"] else ""
        checkpoints = item["checkpoint_count"]
        checkpoint_text = f"{checkpoints} checkpoint" + ("" if checkpoints == 1 else "s")
        console.print(
            f"  [cyan]{index}[/cyan]  {saved}  [dim]{item['model']} · "
            f"{item['message_count']} messages · {checkpoint_text}{preview}[/dim]"
        )
        console.print(f"     [dim]{item['path']}[/dim]")


async def _handle_resume_command(
    prompt_session: PromptSession,
    session,
    arg: str = "",
) -> None:
    console = get_console()
    if session is None:
        console.print("[yellow]Session is still starting.[/yellow]")
        return

    selected_path: Path | None = Path(arg).expanduser() if arg else None
    candidates: list[dict[str, Any]] = []
    if selected_path is None:
        candidates = _session_log_candidates()
        if not candidates:
            console.print("[yellow]No saved sessions found in session_logs/.[/yellow]")
            return
        _print_resume_candidates(candidates)
        answer = await prompt_session.prompt_async(
            "Resume session (number/path, Enter = cancel): "
        )
        answer = answer.strip()
        if not answer:
            console.print("[dim]Resume cancelled.[/dim]")
            return
        if answer.isdigit():
            index = int(answer)
            if index < 1 or index > len(candidates):
                console.print(f"[red]Invalid session number:[/red] {answer}")
                return
            selected_path = Path(candidates[index - 1]["path"])
        else:
            selected_path = Path(answer).expanduser()

    try:
        trajectory = _load_resume_trajectory(selected_path)
    except Exception as e:
        console.print(f"[red]Could not load session:[/red] {e}")
        return

    checkpoint_count = len(
        ((trajectory.get("local_state") or {}).get("checkpoints") or [])
    )
    restore_local_state = False
    if checkpoint_count:
        answer = await prompt_session.prompt_async(
            f"Restore local directory from latest checkpoint? "
            f"({checkpoint_count} available) [y/N]: "
        )
        restore_local_state = answer.strip().lower() in {"y", "yes"}

    try:
        _hydrate_session_from_trajectory(
            session,
            trajectory,
            restore_local_state=restore_local_state,
        )
    except Exception as e:
        console.print(f"[red]Could not resume session:[/red] {e}")
        return

    _print_resume_summary(
        session,
        trajectory["_resume_path"],
        restored=restore_local_state,
    )


def _derive_turn_count(messages: list, events: list[dict[str, Any]]) -> int:
    completed = sum(1 for event in events if event.get("event_type") == "turn_complete")
    if completed:
        return completed
    return sum(1 for msg in messages if getattr(msg, "role", None) == "user")


def _hydrate_session_from_trajectory(
    session,
    trajectory: dict[str, Any],
    *,
    restore_local_state: bool = False,
) -> None:
    """Load messages/events/checkpoints from a saved trajectory into a session."""
    resume_path = Path(trajectory["_resume_path"])
    raw_messages = trajectory.get("messages") or []
    messages = [
        message_from_mapping(raw)
        for raw in raw_messages
        if isinstance(raw, dict)
    ]
    if messages:
        session.context_manager.items = messages

    session.session_id = trajectory.get("session_id") or session.session_id
    session.session_start_time = trajectory.get("session_start_time") or session.session_start_time
    session.logged_events = list(trajectory.get("events") or [])
    session.turn_count = _derive_turn_count(session.context_manager.items, session.logged_events)
    session._local_save_path = str(resume_path)

    local_state = trajectory.get("local_state") or {}
    checkpoints = list(local_state.get("checkpoints") or [])
    snapshot_root = local_state.get("root") or os.getcwd()
    session.attach_local_snapshots(snapshot_root, checkpoints)

    if checkpoints:
        latest_turn = max(int(cp.get("turn_count", 0)) for cp in checkpoints)
        session.turn_count = max(session.turn_count, latest_turn)

    if restore_local_state:
        checkpoint = session.latest_local_checkpoint()
        if checkpoint is None:
            raise ValueError("Resume file has no local-state checkpoints to restore")
        session.restore_to_local_checkpoint(checkpoint)


def _print_resume_summary(session, resume_path: str | Path, *, restored: bool) -> None:
    console = get_console()
    console.print(f"[green]Resumed:[/green] {resume_path}")
    console.print(
        f"[dim]Loaded {len(session.context_manager.items)} context item(s), "
        f"{len(session.local_checkpoints)} local checkpoint(s).[/dim]"
    )
    if restored:
        console.print("[yellow]Local directory state restored to the latest checkpoint.[/yellow]")


def _print_history(session) -> None:
    console = get_console()
    if session is None:
        console.print("[yellow]Session is still starting.[/yellow]")
        return
    checkpoints = session.local_checkpoints
    if not checkpoints:
        console.print("[yellow]No restore checkpoints yet.[/yellow]")
        return
    console.print("[bold]Restore points[/bold]")
    for checkpoint in checkpoints:
        turn = int(checkpoint.get("turn_count", 0))
        label = checkpoint.get("label") or f"turn {turn}"
        message_count = int(checkpoint.get("message_count", 0))
        preview = ""
        for msg in reversed(session.context_manager.items[:message_count]):
            if getattr(msg, "role", None) == "user" and getattr(msg, "content", None):
                preview = str(msg.content).replace("\n", " ")[:80]
                break
        suffix = f" - {preview}" if preview else ""
        console.print(
            f"  [cyan]{turn}[/cyan]  {label}  "
            f"[dim]{message_count} messages{suffix}[/dim]"
        )


def _restore_turn(session, arg: str) -> bool:
    console = get_console()
    if session is None:
        console.print("[yellow]Session is still starting.[/yellow]")
        return False
    if not arg:
        console.print("[yellow]Usage: /restore <turn-number>[/yellow]")
        return False
    try:
        turn = int(arg)
    except ValueError:
        console.print(f"[red]Invalid turn number:[/red] {arg}")
        return False
    checkpoint = session.local_checkpoint_for_turn(turn)
    if checkpoint is None:
        console.print(
            f"[yellow]No checkpoint for turn {turn}. "
            "Run /history to list restore points.[/yellow]"
        )
        return False
    session.restore_to_local_checkpoint(checkpoint)
    if session.config.save_sessions:
        session.save_and_upload_detached(session.config.session_dataset_repo)
    console.print(f"[green]Restored to turn {turn}.[/green]")
    return True


def _user_turns(session) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    if session is None:
        return turns

    turn = 0
    for index, msg in enumerate(session.context_manager.items):
        if index == 0 or getattr(msg, "role", None) != "user":
            continue
        content = getattr(msg, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if content.startswith("[SYSTEM:"):
            continue
        turn += 1
        turns.append(
            {
                "turn": turn,
                "message_index": index,
                "content": content,
            }
        )
    return turns


def _print_rewind_choices(session) -> list[dict[str, Any]]:
    console = get_console()
    turns = _user_turns(session)
    if not turns:
        console.print("[yellow]No prior user messages to rewind.[/yellow]")
        return []

    console.print("[bold]Rewind to a previous message[/bold]")
    for item in turns:
        preview = item["content"].replace("\n", " ")[:90]
        checkpoint_turn = max(0, int(item["turn"]) - 1)
        has_checkpoint = session.local_checkpoint_for_turn(checkpoint_turn) is not None
        marker = "" if has_checkpoint else " [dim](conversation only)[/dim]"
        console.print(f"  [cyan]{item['turn']}[/cyan]  {preview}{marker}")
    return turns


def _rewind_to_user_turn(session, turn: int) -> str | None:
    turns = _user_turns(session)
    selected = next((item for item in turns if item["turn"] == turn), None)
    if selected is None:
        return None

    checkpoint_turn = max(0, turn - 1)
    checkpoint = session.local_checkpoint_for_turn(checkpoint_turn)
    if checkpoint is not None:
        session.restore_to_local_checkpoint(checkpoint)
    else:
        session.context_manager.items = session.context_manager.items[
            : selected["message_index"]
        ]
        session.turn_count = checkpoint_turn

    if session.config.save_sessions:
        session.save_and_upload_detached(session.config.session_dataset_repo)
    return selected["content"]


async def _handle_rewind_shortcut(
    prompt_session: PromptSession,
    session,
) -> str | None:
    console = get_console()
    turns = _print_rewind_choices(session)
    if not turns:
        return None

    latest = str(turns[-1]["turn"])
    answer = await prompt_session.prompt_async(
        "Rewind to message (Enter = latest, c = cancel): ",
        default=latest,
    )
    answer = answer.strip()
    if answer.lower() in {"c", "cancel", "q", "quit"}:
        console.print("[dim]Rewind cancelled.[/dim]")
        return None
    if not answer:
        answer = latest
    try:
        turn = int(answer)
    except ValueError:
        console.print(f"[red]Invalid message number:[/red] {answer}")
        return None

    restored_prompt = _rewind_to_user_turn(session, turn)
    if restored_prompt is None:
        console.print(f"[yellow]No message {turn} to rewind to.[/yellow]")
        return None

    console.print(
        "[green]Rewound.[/green] "
        "[dim]The selected message is back in the prompt for editing.[/dim]"
    )
    return restored_prompt


class _ThinkingShimmer:
    """Animated shiny/shimmer thinking indicator — a bright gradient sweeps across the text."""

    _BASE = (90, 90, 110)       # dim base color
    _HIGHLIGHT = (255, 200, 80) # bright shimmer highlight (warm gold)
    _WIDTH = 5                  # shimmer width in characters
    _FPS = 24

    def __init__(self, console):
        self._console = console
        self._task = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._animate())

    def stop(self):
        if not self._running:
            return  # no-op when never started (e.g. headless mode)
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        # Clear the shimmer line
        self._console.file.write("\r\033[K")
        self._console.file.flush()

    def _render_frame(self, text: str, offset: float) -> str:
        """Render one frame: a bright spot sweeps left-to-right across `text`."""
        out = []
        n = len(text)
        for i, ch in enumerate(text):
            # Distance from the shimmer center (wraps around)
            dist = abs(i - offset)
            wrap_dist = abs(i - offset + n + self._WIDTH)
            dist = min(dist, wrap_dist, abs(i - offset - n - self._WIDTH))
            # Blend factor: 1.0 at center, 0.0 beyond _WIDTH
            t = max(0.0, 1.0 - dist / self._WIDTH)
            t = t * t * (3 - 2 * t)  # smoothstep
            r = int(self._BASE[0] + (self._HIGHLIGHT[0] - self._BASE[0]) * t)
            g = int(self._BASE[1] + (self._HIGHLIGHT[1] - self._BASE[1]) * t)
            b = int(self._BASE[2] + (self._HIGHLIGHT[2] - self._BASE[2]) * t)
            out.append(f"\033[38;2;{r};{g};{b}m{ch}")
        out.append("\033[0m")
        return "".join(out)

    async def _animate(self):
        text = "Thinking..."
        n = len(text)
        speed = 0.45  # characters per frame
        pos = 0.0
        try:
            while self._running:
                frame = self._render_frame(text, pos)
                self._console.file.write(f"\r  {frame}")
                self._console.file.flush()
                pos = (pos + speed) % (n + self._WIDTH)
                await asyncio.sleep(1.0 / self._FPS)
        except asyncio.CancelledError:
            pass


class _StreamBuffer:
    """Accumulates streamed tokens, renders markdown block-by-block as complete
    blocks appear. A "block" is everything up to a paragraph break (\\n\\n).
    Unclosed code fences (odd count of ```) hold back flushing until closed so
    a code block is always rendered as one unit."""

    def __init__(self, console):
        self._console = console
        self._buffer = ""

    def add_chunk(self, text: str):
        self._buffer += text

    def _pop_block(self) -> str | None:
        """Extract the next complete block, or return None if nothing complete."""
        if self._buffer.count("```") % 2 == 1:
            return None  # inside an open code fence — wait for close
        idx = self._buffer.find("\n\n")
        if idx == -1:
            return None
        block = self._buffer[:idx]
        self._buffer = self._buffer[idx + 2:]
        return block

    async def flush_ready(
        self,
        cancel_event: "asyncio.Event | None" = None,
        instant: bool = False,
    ):
        """Render any complete blocks that have accumulated; leave the tail."""
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return
            block = self._pop_block()
            if block is None:
                return
            if block.strip():
                await print_markdown(block, cancel_event=cancel_event, instant=instant)

    async def finish(
        self,
        cancel_event: "asyncio.Event | None" = None,
        instant: bool = False,
    ):
        """Flush complete blocks, then render whatever incomplete tail remains."""
        await self.flush_ready(cancel_event=cancel_event, instant=instant)
        if self._buffer.strip():
            await print_markdown(self._buffer, cancel_event=cancel_event, instant=instant)
        self._buffer = ""

    def discard(self):
        self._buffer = ""


async def event_listener(
    event_queue: asyncio.Queue,
    submission_queue: asyncio.Queue,
    turn_complete_event: asyncio.Event,
    ready_event: asyncio.Event,
    prompt_session: PromptSession,
    config=None,
    session_holder=None,
) -> None:
    """Background task that listens for events and displays them"""
    submission_id = [1000]
    last_tool_name = [None]
    console = _create_rich_console()
    shimmer = _ThinkingShimmer(console)
    stream_buf = _StreamBuffer(console)

    def _cancel_event():
        """Return the session's cancellation Event so print_markdown can abort
        its typewriter loop mid-stream when Ctrl+C fires."""
        s = session_holder[0] if session_holder else None
        return s._cancelled if s is not None else None

    while True:
        try:
            event = await event_queue.get()

            if event.event_type == "ready":
                tool_count = event.data.get("tool_count", 0) if event.data else 0
                print_init_done(tool_count=tool_count)
                ready_event.set()
            elif event.event_type == "assistant_message":
                shimmer.stop()
                content = event.data.get("content", "") if event.data else ""
                if content:
                    await print_markdown(content, cancel_event=_cancel_event())
            elif event.event_type == "assistant_chunk":
                content = event.data.get("content", "") if event.data else ""
                if content:
                    stream_buf.add_chunk(content)
                    # Flush any complete markdown blocks progressively so the
                    # user sees paragraphs appear as they're produced, not just
                    # at the end of the whole response.
                    shimmer.stop()
                    await stream_buf.flush_ready(cancel_event=_cancel_event())
            elif event.event_type == "assistant_stream_end":
                shimmer.stop()
                await stream_buf.finish(cancel_event=_cancel_event())
            elif event.event_type == "tool_call":
                shimmer.stop()
                stream_buf.discard()
                tool_name = event.data.get("tool", "") if event.data else ""
                arguments = event.data.get("arguments", {}) if event.data else {}
                if tool_name:
                    last_tool_name[0] = tool_name
                    # Skip printing research tool_call — the tool_log handler shows it
                    if tool_name != "research":
                        args_str = json.dumps(arguments)[:80]
                        print_tool_call(tool_name, args_str)
            elif event.event_type == "tool_output":
                output = event.data.get("output", "") if event.data else ""
                success = event.data.get("success", False) if event.data else False
                # Only show output for plan_tool — everything else is noise
                if last_tool_name[0] == "plan_tool" and output:
                    print_tool_output(output, success, truncate=False)
                shimmer.start()
            elif event.event_type == "turn_complete":
                shimmer.stop()
                stream_buf.discard()
                print_turn_complete()
                print_plan()
                turn_complete_event.set()
            elif event.event_type == "interrupted":
                shimmer.stop()
                stream_buf.discard()
                print_interrupted()
                turn_complete_event.set()
            elif event.event_type == "undo_complete":
                console.print("[dim]Undone.[/dim]")
                turn_complete_event.set()
            elif event.event_type == "tool_log":
                tool = event.data.get("tool", "") if event.data else ""
                log = event.data.get("log", "") if event.data else ""
                if log:
                    agent_id = event.data.get("agent_id", "") if event.data else ""
                    label = event.data.get("label", "") if event.data else ""
                    print_tool_log(tool, log, agent_id=agent_id, label=label)
            elif event.event_type == "tool_state_change":
                pass  # visual noise — approval flow handles this
            elif event.event_type == "error":
                shimmer.stop()
                stream_buf.discard()
                error = event.data.get("error", "Unknown error") if event.data else "Unknown error"
                print_error(error)
                turn_complete_event.set()
            elif event.event_type == "shutdown":
                shimmer.stop()
                stream_buf.discard()
                break
            elif event.event_type == "processing":
                shimmer.start()
            elif event.event_type == "compacted":
                old_tokens = event.data.get("old_tokens", 0) if event.data else 0
                new_tokens = event.data.get("new_tokens", 0) if event.data else 0
                print_compacted(old_tokens, new_tokens)
            elif event.event_type == "approval_required":
                # Handle batch approval format
                tools_data = event.data.get("tools", []) if event.data else []
                count = event.data.get("count", 0) if event.data else 0

                # If yolo mode is active, auto-approve everything
                if config and config.yolo_mode:
                    approvals = [
                        {
                            "tool_call_id": t.get("tool_call_id", ""),
                            "approved": True,
                            "feedback": None,
                        }
                        for t in tools_data
                    ]
                    print_yolo_approve(count)
                    submission_id[0] += 1
                    approval_submission = Submission(
                        id=f"approval_{submission_id[0]}",
                        operation=Operation(
                            op_type=OpType.EXEC_APPROVAL,
                            data={"approvals": approvals},
                        ),
                    )
                    await submission_queue.put(approval_submission)
                    continue

                print_approval_header(count)
                approvals = []

                # Ask for approval for each tool
                for i, tool_info in enumerate(tools_data, 1):
                    tool_name = tool_info.get("tool", "")
                    arguments = tool_info.get("arguments", {})
                    tool_call_id = tool_info.get("tool_call_id", "")

                    # Handle case where arguments might be a JSON string
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            print(f"Warning: Failed to parse arguments for {tool_name}")
                            arguments = {}

                    operation = arguments.get("operation", "")

                    print_approval_item(i, count, tool_name, operation)

                    # Handle different tool types
                    if tool_name == "hf_private_repos":
                        # Handle private repo operations
                        args = _safe_get_args(arguments)

                        if operation in ["create_repo", "upload_file"]:
                            repo_id = args.get("repo_id", "")
                            repo_type = args.get("repo_type", "dataset")

                            # Build repo URL
                            type_path = "" if repo_type == "model" else f"{repo_type}s"
                            repo_url = (
                                f"https://huggingface.co/{type_path}/{repo_id}".replace(
                                    "//", "/"
                                )
                            )

                            print(f"Repository: {repo_id}")
                            print(f"Type: {repo_type}")
                            print("Private: Yes")
                            print(f"URL: {repo_url}")

                            # Show file preview for upload_file operation
                            if operation == "upload_file":
                                path_in_repo = args.get("path_in_repo", "")
                                file_content = args.get("file_content", "")
                                print(f"File: {path_in_repo}")

                                if isinstance(file_content, str):
                                    # Calculate metrics
                                    all_lines = file_content.split("\n")
                                    line_count = len(all_lines)
                                    size_bytes = len(file_content.encode("utf-8"))
                                    size_kb = size_bytes / 1024
                                    size_mb = size_kb / 1024

                                    print(f"Line count: {line_count}")
                                    if size_kb < 1024:
                                        print(f"Size: {size_kb:.2f} KB")
                                    else:
                                        print(f"Size: {size_mb:.2f} MB")

                                    # Show preview
                                    preview_lines = all_lines[:5]
                                    preview = "\n".join(preview_lines)
                                    print(
                                        f"Content preview (first 5 lines):\n{preview}"
                                    )
                                    if len(all_lines) > 5:
                                        print("...")

                    elif tool_name == "hf_repo_files":
                        # Handle repo files operations (upload, delete)
                        repo_id = arguments.get("repo_id", "")
                        repo_type = arguments.get("repo_type", "model")
                        revision = arguments.get("revision", "main")

                        # Build repo URL
                        if repo_type == "model":
                            repo_url = f"https://huggingface.co/{repo_id}"
                        else:
                            repo_url = f"https://huggingface.co/{repo_type}s/{repo_id}"

                        print(f"Repository: {repo_id}")
                        print(f"Type: {repo_type}")
                        print(f"Branch: {revision}")
                        print(f"URL: {repo_url}")

                        if operation == "upload":
                            path = arguments.get("path", "")
                            content = arguments.get("content", "")
                            create_pr = arguments.get("create_pr", False)

                            print(f"File: {path}")
                            if create_pr:
                                print("Mode: Create PR")

                            if isinstance(content, str):
                                all_lines = content.split("\n")
                                line_count = len(all_lines)
                                size_bytes = len(content.encode("utf-8"))
                                size_kb = size_bytes / 1024

                                print(f"Lines: {line_count}")
                                if size_kb < 1024:
                                    print(f"Size: {size_kb:.2f} KB")
                                else:
                                    print(f"Size: {size_kb / 1024:.2f} MB")

                                # Show full content
                                print(f"Content:\n{content}")

                        elif operation == "delete":
                            patterns = arguments.get("patterns", [])
                            if isinstance(patterns, str):
                                patterns = [patterns]
                            print(f"Patterns to delete: {', '.join(patterns)}")

                    elif tool_name == "hf_repo_git":
                        # Handle git operations (branches, tags, PRs, repo management)
                        repo_id = arguments.get("repo_id", "")
                        repo_type = arguments.get("repo_type", "model")

                        # Build repo URL
                        if repo_type == "model":
                            repo_url = f"https://huggingface.co/{repo_id}"
                        else:
                            repo_url = f"https://huggingface.co/{repo_type}s/{repo_id}"

                        print(f"Repository: {repo_id}")
                        print(f"Type: {repo_type}")
                        print(f"URL: {repo_url}")

                        if operation == "delete_branch":
                            branch = arguments.get("branch", "")
                            print(f"Branch to delete: {branch}")

                        elif operation == "delete_tag":
                            tag = arguments.get("tag", "")
                            print(f"Tag to delete: {tag}")

                        elif operation == "merge_pr":
                            pr_num = arguments.get("pr_num", "")
                            print(f"PR to merge: #{pr_num}")

                        elif operation == "create_repo":
                            private = arguments.get("private", False)
                            space_sdk = arguments.get("space_sdk")
                            print(f"Private: {private}")
                            if space_sdk:
                                print(f"Space SDK: {space_sdk}")

                        elif operation == "update_repo":
                            private = arguments.get("private")
                            gated = arguments.get("gated")
                            if private is not None:
                                print(f"Private: {private}")
                            if gated is not None:
                                print(f"Gated: {gated}")

                    # Get user decision for this item. Ctrl+C / EOF here is
                    # treated as "reject remaining" (matches Codex's modal
                    # priority and Forgecode's approval-cancel path). Without
                    # this, KeyboardInterrupt kills the event listener and
                    # the main loop deadlocks waiting for turn_complete.
                    try:
                        response = await prompt_session.prompt_async(
                            f"Approve item {i}? (y=yes, yolo=approve all, n=no, or provide feedback): "
                        )
                    except (KeyboardInterrupt, EOFError):
                        get_console().print("[dim]Approval cancelled — rejecting remaining items[/dim]")
                        approvals.append(
                            {
                                "tool_call_id": tool_call_id,
                                "approved": False,
                                "feedback": "User cancelled approval",
                            }
                        )
                        for remaining in tools_data[i:]:
                            approvals.append(
                                {
                                    "tool_call_id": remaining.get("tool_call_id", ""),
                                    "approved": False,
                                    "feedback": None,
                                }
                            )
                        break

                    response = response.strip().lower()

                    # Handle yolo mode activation
                    if response == "yolo":
                        config.yolo_mode = True
                        print(
                            "YOLO MODE ACTIVATED - Auto-approving all future tool calls"
                        )
                        # Auto-approve this item and all remaining
                        approvals.append(
                            {
                                "tool_call_id": tool_call_id,
                                "approved": True,
                                "feedback": None,
                            }
                        )
                        for remaining in tools_data[i:]:
                            approvals.append(
                                {
                                    "tool_call_id": remaining.get("tool_call_id", ""),
                                    "approved": True,
                                    "feedback": None,
                                }
                            )
                        break

                    approved = response in ["y", "yes"]
                    feedback = None if approved or response in ["n", "no"] else response

                    approvals.append(
                        {
                            "tool_call_id": tool_call_id,
                            "approved": approved,
                            "feedback": feedback,
                        }
                    )

                # Submit batch approval
                submission_id[0] += 1
                approval_submission = Submission(
                    id=f"approval_{submission_id[0]}",
                    operation=Operation(
                        op_type=OpType.EXEC_APPROVAL,
                        data={"approvals": approvals},
                    ),
                )
                await submission_queue.put(approval_submission)
                console.print()  # spacing after approval
            # Silently ignore other events

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Event listener error: {e}")


async def get_user_input(prompt_session: PromptSession, default: str = "") -> str:
    """Get user input asynchronously"""
    from prompt_toolkit.formatted_text import HTML

    return await prompt_session.prompt_async(
        HTML("\n<b><cyan>></cyan></b> "),
        default=default,
        key_bindings=_PROMPT_KEY_BINDINGS,
    )


# ── Slash command helpers ────────────────────────────────────────────────

# Slash commands are defined in terminal_display


async def _handle_slash_command(
    cmd: str,
    config,
    session_holder: list,
    submission_queue: asyncio.Queue,
    submission_id: list[int],
    prompt_session: PromptSession,
) -> Submission | None:
    """
    Handle a slash command. Returns a Submission to enqueue, or None if
    the command was handled locally (caller should set turn_complete_event).

    Async because ``/provider setup`` prompts for input before committing the
    saved settings.
    """
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command == "/help":
        print_help()
        return None

    if command == "/undo":
        submission_id[0] += 1
        return Submission(
            id=f"sub_{submission_id[0]}",
            operation=Operation(op_type=OpType.UNDO),
        )

    if command == "/compact":
        submission_id[0] += 1
        return Submission(
            id=f"sub_{submission_id[0]}",
            operation=Operation(op_type=OpType.COMPACT),
        )

    if command == "/history":
        _print_history(session_holder[0] if session_holder else None)
        return None

    if command == "/restore":
        _restore_turn(session_holder[0] if session_holder else None, arg)
        return None

    if command == "/resume":
        await _handle_resume_command(
            prompt_session,
            session_holder[0] if session_holder else None,
            arg,
        )
        return None

    if command == "/rewind":
        session = session_holder[0] if session_holder else None
        if arg:
            restored_prompt = (
                _rewind_to_user_turn(session, int(arg)) if arg.isdigit() else None
            )
            if restored_prompt is None:
                get_console().print(
                    f"[yellow]No message {arg} to rewind to. "
                    "Use Esc Esc for the chooser.[/yellow]"
                )
            else:
                get_console().print(
                    "[green]Rewound.[/green] "
                    "[dim]The selected message is back in the prompt for editing.[/dim]"
                )
            return restored_prompt
        _print_rewind_choices(session)
        get_console().print(
            "[dim]Use /rewind <message-number>, or press Esc Esc for the chooser.[/dim]"
        )
        return None

    if command == "/model":
        console = get_console()
        if not arg:
            model_switcher.print_model_listing(config, console)
            return None
        normalized = arg.strip()
        session = session_holder[0] if session_holder else None
        model_switcher.switch_model(normalized, config, session, console)
        return None

    if command == "/provider":
        console = get_console()
        session = session_holder[0] if session_holder else None
        existing = resolve_provider_config(config)
        if arg.lower() == "setup":
            prompt_session = PromptSession()
            provider = await _prompt_provider_config(prompt_session, existing)
            apply_provider_to_config(config, provider)
            if session is not None:
                session.update_model(provider.model)
            return None
        provider = resolve_provider_config(config)
        if provider is None:
            console.print("[yellow]No provider configured. Run /provider setup.[/yellow]")
        else:
            console.print("[bold]OpenAI-compatible provider:[/bold]")
            console.print(f"  model: {provider.model}")
            console.print(f"  base_url: {provider.base_url}")
            console.print(f"  context_window: {provider.context_window}")
            console.print(f"  api_key: {'set' if provider.api_key else 'missing'}")
            console.print("[dim]Run /provider setup to edit saved settings.[/dim]")
        return None

    if command == "/yolo":
        config.yolo_mode = not config.yolo_mode
        state = "ON" if config.yolo_mode else "OFF"
        print(f"YOLO mode: {state}")
        return None

    if command == "/effort":
        console = get_console()
        valid = {"minimal", "low", "medium", "high", "xhigh", "max", "off"}
        session = session_holder[0] if session_holder else None
        if not arg:
            current = config.reasoning_effort or "off"
            console.print(f"[bold]Reasoning effort preference:[/bold] {current}")
            console.print(
                "[dim]Set with '/effort minimal|low|medium|high|xhigh|max|off'. "
                "OpenAI-compatible providers that reject the field are retried without it.[/dim]"
            )
            return None
        level = arg.lower()
        if level not in valid:
            console.print(f"[bold red]Invalid level:[/bold red] {arg}")
            console.print(f"[dim]Expected one of: {', '.join(sorted(valid))}[/dim]")
            return None
        config.reasoning_effort = None if level == "off" else level
        console.print(f"[green]Reasoning effort: {level}[/green]")
        return None

    if command == "/status":
        session = session_holder[0] if session_holder else None
        print(f"Model: {config.model_name}")
        provider = resolve_provider_config(config)
        print(f"Base URL: {provider.base_url if provider else '(not configured)'}")
        print(f"Reasoning effort: {config.reasoning_effort or 'off'}")
        if session:
            print(f"Turns: {session.turn_count}")
            print(f"Context items: {len(session.context_manager.items)}")
            print(f"Restore checkpoints: {len(session.local_checkpoints)}")
        return None

    print(f"Unknown command: {command}. Type /help for available commands.")
    return None


async def main(
    resume_path: str | None = None,
    restore_local_state: bool = False,
):
    """Interactive chat with the agent"""
    resume_trajectory = _load_resume_trajectory(resume_path) if resume_path else None

    # Clear screen
    os.system("clear" if os.name != "nt" else "cls")

    # Create prompt session for input (needed early for token prompt)
    prompt_session = PromptSession()

    config_path = Path(__file__).parent.parent / "configs" / "main_agent_config.json"
    config = load_config(config_path)
    await _ensure_provider_config(config, prompt_session)

    # HF token is used only for Hugging Face tools/jobs/repos.
    hf_token = _get_hf_token()

    # Resolve username for banner
    hf_user = None
    try:
        from huggingface_hub import HfApi
        hf_user = HfApi(token=hf_token).whoami().get("name")
    except Exception:
        pass

    print_banner(model=config.model_name, hf_user=hf_user)

    # Create queues for communication
    submission_queue = asyncio.Queue()
    event_queue = asyncio.Queue()

    # Events to signal agent state
    turn_complete_event = asyncio.Event()
    turn_complete_event.set()
    ready_event = asyncio.Event()

    # Create tool router with local mode
    tool_router = ToolRouter(config.mcpServers, hf_token=hf_token, local_mode=True)

    # Session holder for interrupt/model/status access
    session_holder = [None]

    agent_task = asyncio.create_task(
        submission_loop(
            submission_queue,
            event_queue,
            config=config,
            tool_router=tool_router,
            session_holder=session_holder,
            hf_token=hf_token,
            local_mode=True,
            stream=True,
        )
    )

    # Start event listener in background
    listener_task = asyncio.create_task(
        event_listener(
            event_queue,
            submission_queue,
            turn_complete_event,
            ready_event,
            prompt_session,
            config,
            session_holder=session_holder,
        )
    )

    await ready_event.wait()
    session = session_holder[0]
    if session is not None:
        if resume_path:
            _hydrate_session_from_trajectory(
                session,
                resume_trajectory,
                restore_local_state=restore_local_state,
            )
            _print_resume_summary(session, resume_path, restored=restore_local_state)
        else:
            session.attach_local_snapshots(os.getcwd())
            session.checkpoint_local_state(label="initial")

    submission_id = [0]
    # Mirrors codex-rs/tui/src/bottom_pane/mod.rs:137
    # (`QUIT_SHORTCUT_TIMEOUT = Duration::from_secs(1)`). Two Ctrl+C presses
    # within this window quit; a single press cancels the in-flight turn.
    CTRL_C_QUIT_WINDOW = 1.0
    # Hint string matches codex-rs/tui/src/bottom_pane/footer.rs:746
    # (`" again to quit"` prefixed with the key binding, rendered dim).
    CTRL_C_HINT = "[dim]ctrl + c again to quit[/dim]"
    interrupt_state = {"last": 0.0, "exit": False}
    pending_prompt_text = ""

    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        """SIGINT handler — fires while the agent is generating (terminal is
        in cooked mode between prompts). Mirrors Codex's `on_ctrl_c` in
        codex-rs/tui/src/chatwidget.rs: first press cancels active work and
        arms the quit hint; second press within the window quits."""
        now = time.monotonic()
        session = session_holder[0]

        if now - interrupt_state["last"] < CTRL_C_QUIT_WINDOW:
            interrupt_state["exit"] = True
            if session:
                session.cancel()
            # Wake the main loop out of turn_complete_event.wait()
            turn_complete_event.set()
            return

        interrupt_state["last"] = now
        if session and not session.is_cancelled:
            session.cancel()
        get_console().print(f"\n{CTRL_C_HINT}")

    def _install_sigint() -> bool:
        try:
            loop.add_signal_handler(signal.SIGINT, _on_sigint)
            return True
        except (NotImplementedError, RuntimeError):
            return False  # Windows or non-main thread

    # prompt_toolkit's prompt_async installs its own SIGINT handler and, on
    # exit, calls loop.remove_signal_handler(SIGINT) — which wipes ours too.
    # So we re-arm at the top of every loop iteration, right before the busy
    # wait. Without this, Ctrl+C during agent streaming after the first turn
    # falls through to the default handler and the terminal just echoes ^C.
    sigint_available = _install_sigint()

    try:
        while True:
            if sigint_available:
                _install_sigint()

            try:
                await turn_complete_event.wait()
            except asyncio.CancelledError:
                break
            turn_complete_event.clear()

            if interrupt_state["exit"]:
                break

            # Get user input. prompt_toolkit puts the terminal in raw mode and
            # installs its own SIGINT handling; ^C arrives as \x03 and surfaces
            # as KeyboardInterrupt here. On return, prompt_toolkit removes the
            # loop's SIGINT handler — we re-arm at the top of the next iter.
            try:
                user_input = await get_user_input(prompt_session, pending_prompt_text)
                pending_prompt_text = ""
            except EOFError:
                break
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - interrupt_state["last"] < CTRL_C_QUIT_WINDOW:
                    break
                interrupt_state["last"] = now
                get_console().print(CTRL_C_HINT)
                turn_complete_event.set()
                continue

            # A successful read ends the double-press window — an unrelated
            # Ctrl+C during the next turn should start a fresh arming.
            interrupt_state["last"] = 0.0

            # Check for exit commands
            if user_input.strip().lower() in ["exit", "quit", "/quit", "/exit"]:
                break

            if user_input == _REWIND_SENTINEL:
                restored = await _handle_rewind_shortcut(
                    prompt_session,
                    session_holder[0] if session_holder else None,
                )
                pending_prompt_text = restored or ""
                turn_complete_event.set()
                continue

            # Skip empty input
            if not user_input.strip():
                turn_complete_event.set()
                continue

            # Handle slash commands
            if user_input.strip().startswith("/"):
                sub = await _handle_slash_command(
                    user_input.strip(),
                    config,
                    session_holder,
                    submission_queue,
                    submission_id,
                    prompt_session,
                )
                if sub is None:
                    # Command handled locally, loop back for input
                    turn_complete_event.set()
                    continue
                elif isinstance(sub, str):
                    pending_prompt_text = sub
                    turn_complete_event.set()
                    continue
                else:
                    await submission_queue.put(sub)
                    continue

            # Submit to agent
            submission_id[0] += 1
            submission = Submission(
                id=f"sub_{submission_id[0]}",
                operation=Operation(
                    op_type=OpType.USER_INPUT, data={"text": user_input}
                ),
            )
            await submission_queue.put(submission)

    except KeyboardInterrupt:
        pass
    finally:
        if sigint_available:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass

    # Shutdown
    shutdown_submission = Submission(
        id="sub_shutdown", operation=Operation(op_type=OpType.SHUTDOWN)
    )
    await submission_queue.put(shutdown_submission)

    # Wait for agent to finish (the listener must keep draining events
    # or the agent will block on event_queue.put)
    try:
        await asyncio.wait_for(agent_task, timeout=10.0)
    except asyncio.TimeoutError:
        agent_task.cancel()
        # Agent didn't shut down cleanly — close MCP explicitly
        await tool_router.__aexit__(None, None, None)

    # Now safe to cancel the listener (agent is done emitting events)
    listener_task.cancel()

    get_console().print("\n[dim]Bye.[/dim]\n")


async def headless_main(
    prompt: str,
    model: str | None = None,
    max_iterations: int | None = None,
    stream: bool = True,
    resume_path: str | None = None,
    restore_local_state: bool = False,
) -> None:
    """Run a single prompt headlessly and exit."""
    import logging

    logging.basicConfig(level=logging.WARNING)
    resume_trajectory = _load_resume_trajectory(resume_path) if resume_path else None

    hf_token = _get_hf_token()
    if hf_token:
        print("HF token loaded", file=sys.stderr)
    else:
        print("HF token not set; Hugging Face tools may require login.", file=sys.stderr)

    config_path = Path(__file__).parent.parent / "configs" / "main_agent_config.json"
    config = load_config(config_path)
    config.yolo_mode = True  # Auto-approve everything in headless mode

    if model:
        config.model_name = model

    try:
        provider = require_provider_config(config)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run `ml-intern` interactively and use /provider setup, or set OPENAI_BASE_URL, OPENAI_API_KEY, and OPENAI_MODEL.", file=sys.stderr)
        sys.exit(1)
    apply_provider_to_config(config, provider)
    if model:
        config.model_name = model

    if max_iterations is not None:
        config.max_iterations = max_iterations

    print(f"Model: {config.model_name}", file=sys.stderr)
    print(f"Max iterations: {config.max_iterations}", file=sys.stderr)
    print(f"Prompt: {prompt}", file=sys.stderr)
    print("---", file=sys.stderr)

    submission_queue: asyncio.Queue = asyncio.Queue()
    event_queue: asyncio.Queue = asyncio.Queue()

    tool_router = ToolRouter(config.mcpServers, hf_token=hf_token, local_mode=True)
    session_holder: list = [None]

    agent_task = asyncio.create_task(
        submission_loop(
            submission_queue,
            event_queue,
            config=config,
            tool_router=tool_router,
            session_holder=session_holder,
            hf_token=hf_token,
            local_mode=True,
            stream=stream,
        )
    )

    # Wait for ready
    while True:
        event = await event_queue.get()
        if event.event_type == "ready":
            break

    session = session_holder[0]
    if session is not None:
        if resume_path:
            _hydrate_session_from_trajectory(
                session,
                resume_trajectory,
                restore_local_state=restore_local_state,
            )
            _print_resume_summary(session, resume_path, restored=restore_local_state)
        else:
            session.attach_local_snapshots(os.getcwd())
            session.checkpoint_local_state(label="initial")

    # Submit the prompt
    submission = Submission(
        id="sub_1",
        operation=Operation(op_type=OpType.USER_INPUT, data={"text": prompt}),
    )
    await submission_queue.put(submission)

    # Process events until turn completes. Headless mode is for scripts /
    # log capture: no shimmer animation, no typewriter, no live-redrawing
    # research overlay. Output is plain, append-only text.
    console = _create_rich_console()
    stream_buf = _StreamBuffer(console)
    _hl_last_tool = [None]
    _hl_sub_id = [1]
    # Research sub-agent tool calls are buffered per agent_id and dumped as
    # a static block once each sub-agent finishes, instead of streaming via
    # the live redrawing SubAgentDisplayManager (which is TTY-only).
    _hl_research_buffers: dict[str, dict] = {}

    while True:
        event = await event_queue.get()

        if event.event_type == "assistant_chunk":
            content = event.data.get("content", "") if event.data else ""
            if content:
                stream_buf.add_chunk(content)
                await stream_buf.flush_ready(instant=True)
        elif event.event_type == "assistant_stream_end":
            await stream_buf.finish(instant=True)
        elif event.event_type == "assistant_message":
            content = event.data.get("content", "") if event.data else ""
            if content:
                await print_markdown(content, instant=True)
        elif event.event_type == "tool_call":
            stream_buf.discard()
            tool_name = event.data.get("tool", "") if event.data else ""
            arguments = event.data.get("arguments", {}) if event.data else {}
            if tool_name:
                _hl_last_tool[0] = tool_name
                if tool_name != "research":
                    args_str = json.dumps(arguments)[:80]
                    print_tool_call(tool_name, args_str)
        elif event.event_type == "tool_output":
            output = event.data.get("output", "") if event.data else ""
            success = event.data.get("success", False) if event.data else False
            if _hl_last_tool[0] == "plan_tool" and output:
                print_tool_output(output, success, truncate=False)
        elif event.event_type == "tool_log":
            tool = event.data.get("tool", "") if event.data else ""
            log = event.data.get("log", "") if event.data else ""
            if not log:
                pass
            elif tool == "research":
                # Headless mode: buffer research sub-agent activity per-agent,
                # then dump each as a static block on completion. The live
                # SubAgentDisplayManager uses terminal cursor tricks that are
                # unfit for non-TTY output, but parallel agents still need
                # distinct output so we key buffers by agent_id.
                agent_id = event.data.get("agent_id", "") if event.data else ""
                label = event.data.get("label", "") if event.data else ""
                aid = agent_id or "research"
                if log == "Starting research sub-agent...":
                    _hl_research_buffers[aid] = {
                        "label": label or "research",
                        "calls": [],
                    }
                elif log == "Research complete.":
                    buf = _hl_research_buffers.pop(aid, None)
                    if buf is not None:
                        f = get_console().file
                        f.write(f"  \033[38;2;255;200;80m▸ {buf['label']}\033[0m\n")
                        for call in buf["calls"]:
                            f.write(f"    \033[2m{call}\033[0m\n")
                        f.flush()
                elif log.startswith("tokens:") or log.startswith("tools:"):
                    pass  # stats updates — only useful for the live display
                elif aid in _hl_research_buffers:
                    _hl_research_buffers[aid]["calls"].append(log)
                else:
                    # Orphan event (Start was missed) — fall back to raw print
                    print_tool_log(tool, log, agent_id=agent_id, label=label)
            else:
                print_tool_log(tool, log)
        elif event.event_type == "approval_required":
            # Auto-approve everything in headless mode (safety net if yolo_mode
            # didn't prevent the approval event for some reason)
            tools_data = event.data.get("tools", []) if event.data else []
            approvals = [
                {
                    "tool_call_id": t.get("tool_call_id", ""),
                    "approved": True,
                    "feedback": None,
                }
                for t in tools_data
            ]
            _hl_sub_id[0] += 1
            await submission_queue.put(Submission(
                id=f"hl_approval_{_hl_sub_id[0]}",
                operation=Operation(
                    op_type=OpType.EXEC_APPROVAL,
                    data={"approvals": approvals},
                ),
            ))
        elif event.event_type == "compacted":
            old_tokens = event.data.get("old_tokens", 0) if event.data else 0
            new_tokens = event.data.get("new_tokens", 0) if event.data else 0
            print_compacted(old_tokens, new_tokens)
        elif event.event_type == "error":
            stream_buf.discard()
            error = event.data.get("error", "Unknown error") if event.data else "Unknown error"
            print_error(error)
            break
        elif event.event_type in ("turn_complete", "interrupted"):
            stream_buf.discard()
            history_size = event.data.get("history_size", "?") if event.data else "?"
            print(f"\n--- Agent {event.event_type} (history_size={history_size}) ---", file=sys.stderr)
            break

    # Shutdown
    shutdown_submission = Submission(
        id="sub_shutdown", operation=Operation(op_type=OpType.SHUTDOWN)
    )
    await submission_queue.put(shutdown_submission)

    try:
        await asyncio.wait_for(agent_task, timeout=10.0)
    except asyncio.TimeoutError:
        agent_task.cancel()
        await tool_router.__aexit__(None, None, None)


def cli():
    """Entry point for the ml-intern CLI command."""
    import logging as _logging
    import warnings
    # Suppress aiohttp "Unclosed client session" noise during event loop teardown
    _logging.getLogger("asyncio").setLevel(_logging.CRITICAL)
    # Suppress whoosh invalid escape sequence warnings (third-party, unfixed upstream)
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="whoosh")

    parser = argparse.ArgumentParser(description="ML Intern CLI")
    parser.add_argument("prompt", nargs="?", default=None, help="Run headlessly with this prompt")
    parser.add_argument("--model", "-m", default=None, help=f"Model to use (default: from config)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Max LLM requests per turn (default: 50, use -1 for unlimited)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable token streaming (use non-streaming LLM calls)")
    parser.add_argument("--resume", default=None,
                        help="Resume an exact session from a session_logs JSON file")
    parser.add_argument("--restore-local-state", action="store_true",
                        help="With --resume, restore the working directory to "
                             "the latest checkpoint")
    args = parser.parse_args()

    try:
        if args.prompt:
            max_iter = args.max_iterations
            if max_iter is not None and max_iter < 0:
                max_iter = 10_000  # effectively unlimited
            asyncio.run(headless_main(
                args.prompt,
                model=args.model,
                max_iterations=max_iter,
                stream=not args.no_stream,
                resume_path=args.resume,
                restore_local_state=args.restore_local_state,
            ))
        else:
            asyncio.run(main(
                resume_path=args.resume,
                restore_local_state=args.restore_local_state,
            ))
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    cli()
