"""Local directory checkpoints for CLI resume/restore.

The CLI tools can mutate the user's working tree through ``bash``, ``write``,
and ``edit``. These checkpoints snapshot the working directory at turn
boundaries so restoring conversation history can also restore local files.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDED_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "node_modules",
    "session_logs",
}


@dataclass(frozen=True)
class LocalCheckpoint:
    checkpoint_id: str
    timestamp: str
    root: str
    label: str
    message_count: int
    turn_count: int
    path: str

    def model_dump(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "root": self.root,
            "label": self.label,
            "message_count": self.message_count,
            "turn_count": self.turn_count,
            "path": self.path,
        }


class LocalSnapshotManager:
    """Create and restore working-directory snapshots for one session."""

    def __init__(
        self,
        root: str | Path,
        session_id: str,
        base_dir: str | Path = "session_logs/snapshots",
        excluded_names: set[str] | None = None,
    ):
        self.root = Path(root).resolve()
        self.session_id = session_id
        self.base_dir = Path(base_dir).resolve() / session_id
        self.excluded_names = set(DEFAULT_EXCLUDED_NAMES)
        if excluded_names:
            self.excluded_names.update(excluded_names)

    def create(
        self,
        *,
        label: str,
        message_count: int,
        turn_count: int,
    ) -> dict[str, Any]:
        checkpoint_id = f"{turn_count:04d}_{uuid.uuid4().hex[:8]}"
        checkpoint_dir = self.base_dir / checkpoint_id
        files_dir = checkpoint_dir / "files"
        checkpoint_dir.mkdir(parents=True, exist_ok=False)

        shutil.copytree(
            self.root,
            files_dir,
            ignore=self._ignore,
            symlinks=True,
        )

        checkpoint = LocalCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            root=str(self.root),
            label=label,
            message_count=message_count,
            turn_count=turn_count,
            path=str(checkpoint_dir),
        )
        manifest_path = checkpoint_dir / "checkpoint.json"
        manifest_path.write_text(json.dumps(checkpoint.model_dump(), indent=2))
        return checkpoint.model_dump()

    def restore(self, checkpoint: dict[str, Any]) -> None:
        checkpoint_path = Path(checkpoint["path"])
        files_dir = checkpoint_path / "files"
        if not files_dir.exists():
            raise FileNotFoundError(f"Checkpoint files missing: {files_dir}")

        self.root.mkdir(parents=True, exist_ok=True)
        snapshot_entries = self._relative_entries(files_dir)
        root_entries = self._relative_entries(self.root)

        stale_entries = sorted(
            root_entries - snapshot_entries,
            key=lambda p: len(p.parts),
            reverse=True,
        )
        for rel in stale_entries:
            target = self.root / rel
            if self._is_excluded(rel):
                continue
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)

        for rel in sorted(snapshot_entries, key=lambda p: len(p.parts)):
            src = files_dir / rel
            dst = self.root / rel
            if src.is_dir() and not src.is_symlink():
                if dst.exists() and (dst.is_file() or dst.is_symlink()):
                    dst.unlink()
                dst.mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() or dst.is_symlink():
                    if dst.is_dir() and not dst.is_symlink():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                shutil.copy2(src, dst, follow_symlinks=False)

    def _ignore(self, dir_path: str, names: list[str]) -> set[str]:
        rel_dir = Path(dir_path).resolve().relative_to(self.root)
        ignored: set[str] = set()
        for name in names:
            rel = rel_dir / name if str(rel_dir) != "." else Path(name)
            if self._is_excluded(rel):
                ignored.add(name)
        return ignored

    def _relative_entries(self, base: Path) -> set[Path]:
        entries: set[Path] = set()
        for dir_path, dir_names, file_names in os.walk(base, followlinks=False):
            current = Path(dir_path)
            rel_dir = current.relative_to(base)
            dir_names[:] = [
                name
                for name in dir_names
                if not self._is_excluded(
                    (rel_dir / name) if str(rel_dir) != "." else Path(name)
                )
            ]
            for name in dir_names + file_names:
                rel = (rel_dir / name) if str(rel_dir) != "." else Path(name)
                if not self._is_excluded(rel):
                    entries.add(rel)
        return entries

    def _is_excluded(self, rel: Path) -> bool:
        return any(part in self.excluded_names for part in rel.parts)
