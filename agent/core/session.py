import asyncio
import json
import logging
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agent.config import Config
from agent.context_manager.manager import ContextManager
from agent.core.local_checkpoint import LocalSnapshotManager

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 200_000


def _get_max_tokens_safe(model_name: str) -> int:
    """Return the configured/default max input-context tokens for a model."""
    logger.info("Using default context window for %s: %d", model_name, _DEFAULT_MAX_TOKENS)
    return _DEFAULT_MAX_TOKENS


class OpType(Enum):
    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT = "interrupt"
    UNDO = "undo"
    COMPACT = "compact"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    event_type: str
    data: Optional[dict[str, Any]] = None


class Session:
    """
    Maintains agent session state
    Similar to Session in codex-rs/core/src/codex.rs
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        config: Config | None = None,
        tool_router=None,
        context_manager: ContextManager | None = None,
        hf_token: str | None = None,
        local_mode: bool = False,
        stream: bool = True,
    ):
        self.hf_token: Optional[str] = hf_token
        self.tool_router = tool_router
        self.stream = stream
        tool_specs = tool_router.get_tool_specs_for_llm() if tool_router else []
        self.context_manager = context_manager or ContextManager(
            model_max_tokens=getattr(config, "openai_context_window", None)
            or _get_max_tokens_safe(config.model_name),
            compact_size=0.1,
            untouched_messages=5,
            tool_specs=tool_specs,
            hf_token=hf_token,
            local_mode=local_mode,
        )
        self.event_queue = event_queue
        self.session_id = str(uuid.uuid4())
        self.config = config or Config(
            model_name="gpt-4o-mini",
        )
        self.is_running = True
        self._cancelled = asyncio.Event()
        self.pending_approval: Optional[dict[str, Any]] = None
        self.sandbox = None

        # Session trajectory logging
        self.logged_events: list[dict] = []
        self.session_start_time = datetime.now().isoformat()
        self.turn_count: int = 0
        self.last_auto_save_turn: int = 0
        # Stable local save path so heartbeat saves overwrite one file instead
        # of spamming session_logs/. ``_last_heartbeat_ts`` is owned by
        # ``agent.core.telemetry.HeartbeatSaver`` and lazily initialised there.
        self._local_save_path: Optional[str] = None
        self._last_heartbeat_ts: Optional[float] = None
        self.local_snapshot_manager: LocalSnapshotManager | None = None
        self.local_state_root: str | None = None
        self.local_checkpoints: list[dict[str, Any]] = []
        self.restored_from: str | None = None

        # Per-model reasoning-effort cache. Populated when a provider rejects
        # reasoning fields, read by ``effective_effort_for`` below. Keys are
        # raw model ids (including any ``:tag``). Values:
        #   str  -> the effort level to send
        #   None -> model rejected reasoning fields; send no thinking params
        # Key absent -> fall back to the raw preference.
        self.model_effective_effort: dict[str, str | None] = {}

    async def send_event(self, event: Event) -> None:
        """Send event back to client and log to trajectory"""
        await self.event_queue.put(event)

        # Log event to trajectory
        self.logged_events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": event.event_type,
                "data": event.data,
            }
        )

        # Mid-turn heartbeat flush (owned by telemetry module).
        from agent.core.telemetry import HeartbeatSaver
        HeartbeatSaver.maybe_fire(self)

    def cancel(self) -> None:
        """Signal cancellation to the running agent loop."""
        self._cancelled.set()

    def reset_cancel(self) -> None:
        """Clear the cancellation flag before a new run."""
        self._cancelled.clear()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def update_model(self, model_name: str) -> None:
        """Switch the active model and update the context window limit."""
        self.config.model_name = model_name
        self.context_manager.model_max_tokens = (
            getattr(self.config, "openai_context_window", None)
            or _get_max_tokens_safe(model_name)
        )

    def effective_effort_for(self, model_name: str) -> str | None:
        """Resolve the effort level to actually send for ``model_name``.

        Returns the cached result when we have one (may be ``None`` meaning
        "model doesn't do thinking, strip it"), else the raw preference.
        Unknown-model case falls back to the preference so a stale cache
        from a prior failed call can't poison research sub-calls that use a
        different model id.
        """
        if model_name in self.model_effective_effort:
            return self.model_effective_effort[model_name]
        return self.config.reasoning_effort

    def increment_turn(self) -> None:
        """Increment turn counter (called after each user interaction)"""
        self.turn_count += 1

    def attach_local_snapshots(
        self,
        root: str | Path,
        checkpoints: list[dict[str, Any]] | None = None,
    ) -> None:
        """Enable working-directory checkpoints for this CLI session."""
        resolved_root = Path(root).resolve()
        self.local_snapshot_manager = LocalSnapshotManager(
            resolved_root,
            self.session_id,
            base_dir=resolved_root / "session_logs" / "snapshots",
        )
        self.local_state_root = str(resolved_root)
        self.local_checkpoints = list(checkpoints or [])

    def checkpoint_local_state(self, label: str | None = None) -> dict[str, Any] | None:
        """Snapshot the working directory if CLI local snapshots are enabled."""
        if self.local_snapshot_manager is None:
            return None

        message_count = len(self.context_manager.items)
        if (
            self.local_checkpoints
            and self.local_checkpoints[-1].get("message_count") == message_count
        ):
            return self.local_checkpoints[-1]

        checkpoint = self.local_snapshot_manager.create(
            label=label or f"turn {self.turn_count}",
            message_count=message_count,
            turn_count=self.turn_count,
        )
        self.local_checkpoints.append(checkpoint)
        return checkpoint

    def local_checkpoint_for_turn(self, turn_count: int) -> dict[str, Any] | None:
        """Return the newest checkpoint for ``turn_count``."""
        matches = [
            checkpoint
            for checkpoint in self.local_checkpoints
            if int(checkpoint.get("turn_count", -1)) == turn_count
        ]
        return matches[-1] if matches else None

    def latest_local_checkpoint(self) -> dict[str, Any] | None:
        return self.local_checkpoints[-1] if self.local_checkpoints else None

    def local_checkpoint_at_or_before_message_count(
        self, message_count: int
    ) -> dict[str, Any] | None:
        """Return the newest checkpoint whose context length fits the history."""
        matches = [
            checkpoint
            for checkpoint in self.local_checkpoints
            if int(checkpoint.get("message_count", 0)) <= message_count
        ]
        return matches[-1] if matches else None

    def restore_to_local_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore files and truncate chat history to a checkpoint boundary."""
        if self.local_snapshot_manager is None:
            root = checkpoint.get("root") or self.local_state_root or "."
            self.attach_local_snapshots(root, self.local_checkpoints)
        assert self.local_snapshot_manager is not None
        self.local_snapshot_manager.restore(checkpoint)

        message_count = int(
            checkpoint.get("message_count", len(self.context_manager.items))
        )
        self.context_manager.items = self.context_manager.items[:message_count]
        self.turn_count = int(checkpoint.get("turn_count", self.turn_count))
        self.restored_from = checkpoint.get("checkpoint_id")

    async def auto_save_if_needed(self) -> None:
        """Check if auto-save should trigger and save if so (completely non-blocking)"""
        if not self.config.save_sessions:
            return

        interval = self.config.auto_save_interval
        if interval <= 0:
            return

        turns_since_last_save = self.turn_count - self.last_auto_save_turn
        if turns_since_last_save >= interval:
            logger.info(f"Auto-saving session (turn {self.turn_count})...")
            # Fire-and-forget save - returns immediately
            self.save_and_upload_detached(self.config.session_dataset_repo)
            self.last_auto_save_turn = self.turn_count

    def get_trajectory(self) -> dict:
        """Serialize complete session trajectory for logging"""
        tools: list = []
        if self.tool_router is not None:
            try:
                tools = self.tool_router.get_tool_specs_for_llm() or []
            except Exception:
                tools = []
        trajectory = {
            "session_id": self.session_id,
            "session_start_time": self.session_start_time,
            "session_end_time": datetime.now().isoformat(),
            "model_name": self.config.model_name,
            "messages": [msg.model_dump() for msg in self.context_manager.items],
            "events": self.logged_events,
            "tools": tools,
        }
        if self.local_state_root is not None:
            trajectory["local_state"] = {
                "root": self.local_state_root,
                "checkpoints": self.local_checkpoints,
                "restored_from": self.restored_from,
            }
        return trajectory

    def save_trajectory_local(
        self,
        directory: str = "session_logs",
        upload_status: str = "pending",
        dataset_url: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save trajectory to local JSON file as backup with upload status

        Args:
            directory: Directory to save logs (default: "session_logs")
            upload_status: Status of upload attempt ("pending", "success", "failed")
            dataset_url: URL of dataset if upload succeeded

        Returns:
            Path to saved file if successful, None otherwise
        """
        try:
            log_dir = Path(directory)
            log_dir.mkdir(parents=True, exist_ok=True)

            trajectory = self.get_trajectory()

            # Scrub secrets at save time so session_logs/ never holds raw
            # tokens on disk — a log aggregator, crash dump, or filesystem
            # snapshot between heartbeats would otherwise leak them.
            try:
                from agent.core.redact import scrub
                for key in ("messages", "events", "tools"):
                    if key in trajectory:
                        trajectory[key] = scrub(trajectory[key])
            except Exception as _e:
                logger.debug("Redact-on-save failed (non-fatal): %s", _e)

            # Add upload metadata
            trajectory["upload_status"] = upload_status
            trajectory["upload_url"] = dataset_url
            trajectory["last_save_time"] = datetime.now().isoformat()

            # Reuse one stable path per session so heartbeat saves overwrite
            # the same file instead of creating a new timestamped file every
            # minute. The timestamp in the filename is kept for first-save
            # ordering; subsequent saves just rewrite that file.
            if self._local_save_path and Path(self._local_save_path).parent == log_dir:
                filepath = Path(self._local_save_path)
            else:
                filename = (
                    f"session_{self.session_id}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                filepath = log_dir / filename
                self._local_save_path = str(filepath)

            # Atomic-ish write: stage to .tmp then rename so a crash mid-write
            # doesn't leave a truncated JSON that breaks the retry scanner.
            tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
            with open(tmp_path, "w") as f:
                json.dump(trajectory, f, indent=2)
            tmp_path.replace(filepath)

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save session locally: {e}")
            return None

    def update_local_save_status(
        self, filepath: str, upload_status: str, dataset_url: Optional[str] = None
    ) -> bool:
        """Update the upload status of an existing local save file"""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            data["upload_status"] = upload_status
            data["upload_url"] = dataset_url
            data["last_save_time"] = datetime.now().isoformat()

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Failed to update local save status: {e}")
            return False

    def save_and_upload_detached(self, repo_id: str) -> Optional[str]:
        """
        Save session locally and spawn detached subprocess for upload (fire-and-forget)

        Args:
            repo_id: HuggingFace dataset repo ID

        Returns:
            Path to local save file
        """
        # Save locally first (fast, synchronous)
        local_path = self.save_trajectory_local(upload_status="pending")
        if not local_path:
            return None

        # Spawn detached subprocess for upload (fire-and-forget)
        try:
            uploader_script = Path(__file__).parent / "session_uploader.py"

            # Use Popen with detached process
            subprocess.Popen(
                [sys.executable, str(uploader_script), "upload", local_path, repo_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent
            )
        except Exception as e:
            logger.warning(f"Failed to spawn upload subprocess: {e}")

        return local_path

    @staticmethod
    def retry_failed_uploads_detached(
        directory: str = "session_logs", repo_id: Optional[str] = None
    ) -> None:
        """
        Spawn detached subprocess to retry failed/pending uploads (fire-and-forget)

        Args:
            directory: Directory containing session logs
            repo_id: Target dataset repo ID
        """
        if not repo_id:
            return

        try:
            uploader_script = Path(__file__).parent / "session_uploader.py"

            # Spawn detached subprocess for retry
            subprocess.Popen(
                [sys.executable, str(uploader_script), "retry", directory, repo_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent
            )
        except Exception as e:
            logger.warning(f"Failed to spawn retry subprocess: {e}")
