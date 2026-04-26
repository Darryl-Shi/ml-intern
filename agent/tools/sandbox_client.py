#!/usr/bin/env python3
"""
Sandbox Tools - Agent-native primitives for SkyPilot sandboxes.

The public class intentionally keeps the previous sandbox shape so the rest of
the agent can swap backends without a large migration:

    sb = Sandbox.create(hardware="A100:1", infra="runpod")
    sb.bash("python train.py")
    sb.read("/app/train.py")
    sb.edit("/app/train.py", old_str="lr=1e-3", new_str="lr=1e-4")
    sb.delete()

SkyPilot calls are synchronous from this wrapper's point of view. Current
SkyPilot SDK methods return request IDs, so we stream-and-wait before returning.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import pathlib
import re
import shlex
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

OUTPUT_LIMIT = 25000
DEFAULT_READ_LIMIT = 2000
DEFAULT_TIMEOUT = 240
MAX_TIMEOUT = 1200
WAIT_TIMEOUT = 600
DEFAULT_INFRA = "runpod"

CPU_ALIASES = {
    "cpu-basic": {"cpus": "2+", "memory": "8+"},
    "cpu-upgrade": {"cpus": "8+", "memory": "32+"},
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _truncate_output(output: str, max_chars: int = OUTPUT_LIMIT, head_ratio: float = 0.25) -> str:
    if len(output) <= max_chars:
        return output
    head_budget = int(max_chars * head_ratio)
    tail_budget = max_chars - head_budget
    omitted = len(output) - max_chars
    meta = (
        f"\n\n... ({omitted:,} of {len(output):,} chars omitted, "
        f"showing first {head_budget:,} + last {tail_budget:,}) ...\n"
    )
    return output[:head_budget] + meta + output[-tail_budget:]


def _require_sky():
    try:
        import sky  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "SkyPilot is not available. Install dependencies with "
            "`uv sync` or `uv pip install 'skypilot-nightly[runpod]'`."
        ) from e
    return sky


def _repair_empty_skypilot_catalogs(log: Callable[[str], object] | None = None) -> None:
    """Delete empty SkyPilot catalog CSV caches so SkyPilot can refetch them."""
    catalog_root = pathlib.Path.home() / ".sky" / "catalogs"
    if not catalog_root.exists():
        return

    repaired = []
    for path in catalog_root.glob("*/*/*.csv"):
        try:
            if path.is_file() and path.stat().st_size == 0:
                path.unlink()
                repaired.append(path)
        except OSError:
            continue

    if repaired and log is not None:
        paths = ", ".join(str(path) for path in repaired)
        log(f"Removed empty SkyPilot catalog cache file(s): {paths}")


def _wait_for_request(sky: Any, request_id: Any, *, stream: bool = True) -> tuple[Any, str]:
    """Wait for a SkyPilot SDK request and capture streamed logs when possible."""
    if request_id is None:
        return None, ""

    buf = io.StringIO()
    if stream and hasattr(sky, "stream_and_get"):
        try:
            result = sky.stream_and_get(request_id, output_stream=buf)
            return result, _strip_ansi(buf.getvalue())
        except TypeError:
            # Older SDKs may not support output_stream.
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                result = sky.stream_and_get(request_id)
            return result, _strip_ansi(buf.getvalue())

    if hasattr(sky, "get"):
        return sky.get(request_id), ""
    return request_id, ""


def _request_succeeded(result: Any) -> bool:
    """Best-effort success inference across SkyPilot SDK versions."""
    if result is None:
        return True
    if isinstance(result, int):
        return result == 0
    if isinstance(result, tuple) and result and isinstance(result[0], int):
        # sky.exec commonly returns (job_id, handle), not an exit code.
        return True
    return True


def _extract_job_id(result: Any) -> int | None:
    if isinstance(result, tuple) and result and isinstance(result[0], int):
        return result[0]
    if isinstance(result, int) and result > 0:
        return result
    return None


def _remote_python_command(script: str, payload: dict[str, Any]) -> str:
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return (
        f"PAYLOAD_B64={shlex.quote(encoded)} python - <<'PY'\n"
        "import base64, json, os\n"
        "payload = json.loads(base64.b64decode(os.environ['PAYLOAD_B64']))\n"
        + script.rstrip()
        + "\nPY"
    )


_READ_SCRIPT = r'''
import pathlib, sys
path = payload["path"]
offset = payload.get("offset") or 1
limit = payload.get("limit")
p = pathlib.Path(path)
if not p.exists():
    print(f"File not found: {path}", file=sys.stderr)
    raise SystemExit(2)
if p.is_dir():
    print(f"Is a directory: {path}", file=sys.stderr)
    raise SystemExit(3)
lines = p.read_text().splitlines()
start = max(int(offset), 1) - 1
end = None if limit is None else start + int(limit)
selected = lines[start:end]
for idx, line in enumerate(selected, start=start + 1):
    if len(line) > 4000:
        line = line[:4000] + "..."
    print(f"{idx}\t{line}")
'''


_WRITE_SCRIPT = r'''
import ast, os, pathlib, tempfile

def atomic_write(path, content):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(p))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def validate_python(content):
    warnings = []
    try:
        ast.parse(content)
    except SyntaxError as e:
        return [f"Python syntax error at line {e.lineno}: {e.msg}"]
    if any(k in content for k in ("TrainingArguments", "SFTConfig", "DPOConfig", "GRPOConfig")):
        if "push_to_hub" not in content:
            warnings.append("Training script warning: no 'push_to_hub' found")
        if "hub_model_id" not in content:
            warnings.append("Training script warning: no 'hub_model_id' found")
    return warnings

path = payload["path"]
content = payload["content"]
atomic_write(path, content)
msg = f"Wrote {len(content)} bytes to {path}"
if pathlib.Path(path).suffix == ".py":
    warnings = validate_python(content)
    if warnings:
        msg += "\n\nValidation warnings:\n" + "\n".join(f"  ! {w}" for w in warnings)
print(msg)
'''


_EDIT_SCRIPT = r'''
import ast, os, pathlib, tempfile

UNICODE_MAP = {
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u2003": " ", "\u2002": " ", "\u200b": "", "\ufeff": "",
}

def normalize_unicode(s):
    return "".join(UNICODE_MAP.get(c, c) for c in s)

def fuzzy_find_original(content, pattern):
    if pattern in content:
        return pattern, None
    c_lines = content.split("\n")
    for strip_fn, note in (
        (str.rstrip, "(matched after trimming trailing whitespace)"),
        (str.strip, "(matched after trimming whitespace)"),
    ):
        c2 = "\n".join(strip_fn(l) for l in c_lines)
        p2 = "\n".join(strip_fn(l) for l in pattern.split("\n"))
        if p2 in c2:
            idx = c2.index(p2)
            start_line = c2[:idx].count("\n")
            n_lines = p2.count("\n") + 1
            return "\n".join(c_lines[start_line:start_line + n_lines]), note
    c3 = normalize_unicode("\n".join(l.strip() for l in c_lines))
    p3 = normalize_unicode("\n".join(l.strip() for l in pattern.split("\n")))
    if p3 in c3:
        idx = c3.index(p3)
        start_line = c3[:idx].count("\n")
        n_lines = p3.count("\n") + 1
        return "\n".join(c_lines[start_line:start_line + n_lines]), "(matched after unicode normalization)"
    return None, None

def apply_edit(content, old_str, new_str, mode="replace", replace_all=False):
    if mode == "replace_all":
        replace_all = True
        mode = "replace"
    fuzzy_note = None
    if old_str not in content:
        old_str, fuzzy_note = fuzzy_find_original(content, old_str)
        if old_str is None:
            raise ValueError("old_str not found in file.")
    count = content.count(old_str)
    if mode == "replace":
        if count > 1 and not replace_all:
            raise ValueError(f"old_str appears {count} times. Use replace_all=true or provide more context.")
        return content.replace(old_str, new_str) if replace_all else content.replace(old_str, new_str, 1), count if replace_all else 1, fuzzy_note
    if mode == "append_after":
        if replace_all:
            return content.replace(old_str, old_str + new_str), count, fuzzy_note
        idx = content.index(old_str) + len(old_str)
        return content[:idx] + new_str + content[idx:], 1, fuzzy_note
    if mode == "prepend_before":
        if replace_all:
            return content.replace(old_str, new_str + old_str), count, fuzzy_note
        idx = content.index(old_str)
        return content[:idx] + new_str + content[idx:], 1, fuzzy_note
    raise ValueError(f"Unknown mode: {mode}")

def atomic_write(path, content):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(p))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def validate_python(content):
    warnings = []
    try:
        ast.parse(content)
    except SyntaxError as e:
        return [f"Python syntax error at line {e.lineno}: {e.msg}"]
    if any(k in content for k in ("TrainingArguments", "SFTConfig", "DPOConfig", "GRPOConfig")):
        if "push_to_hub" not in content:
            warnings.append("Training script warning: no 'push_to_hub' found")
        if "hub_model_id" not in content:
            warnings.append("Training script warning: no 'hub_model_id' found")
    return warnings

path = payload["path"]
p = pathlib.Path(path)
if not p.exists():
    raise FileNotFoundError(f"File not found: {path}")
content = p.read_text()
new_content, count, fuzzy_note = apply_edit(
    content,
    payload["old_str"],
    payload["new_str"],
    payload.get("mode", "replace"),
    payload.get("replace_all", False),
)
atomic_write(path, new_content)
msg = f"Edited {path} ({count} replacement{'s' if count > 1 else ''})"
if fuzzy_note:
    msg += f" {fuzzy_note}"
if p.suffix == ".py":
    warnings = validate_python(new_content)
    if warnings:
        msg += "\n\nValidation warnings:\n" + "\n".join(f"  ! {w}" for w in warnings)
print(msg)
'''


_EXISTS_SCRIPT = r'''
import pathlib
print(str(pathlib.Path(payload["path"]).exists()).lower())
'''


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""

    def __str__(self):
        if self.success:
            return self.output or "(no output)"
        return f"ERROR: {self.error}"

    def to_dict(self) -> dict:
        return {"success": self.success, "output": self.output, "error": self.error}


@dataclass
class Sandbox:
    """A handle to a SkyPilot sandbox cluster."""

    space_id: str
    token: str | None = None
    work_dir: str = "/app"
    timeout: int = DEFAULT_TIMEOUT
    infra: str = DEFAULT_INFRA
    hardware: str = "cpu-basic"
    _owns_space: bool = field(default=False, repr=False)
    _files_read: set[str] = field(init=False, repr=False, default_factory=set)
    _latest_job_id: int | None = field(default=None, init=False, repr=False)

    class Cancelled(Exception):
        """Raised when sandbox creation is cancelled by the user."""

    @property
    def cluster_name(self) -> str:
        return self.space_id

    @classmethod
    def create(
        cls,
        owner: str | None = None,
        *,
        name: str | None = None,
        hardware: str = "cpu-basic",
        infra: str = DEFAULT_INFRA,
        token: str | None = None,
        secrets: dict[str, str] | None = None,
        wait_timeout: int = WAIT_TIMEOUT,
        log: "Callable[[str], object] | None" = None,
        cancel_event: "Any | None" = None,
        **_: Any,
    ) -> "Sandbox":
        """Create a SkyPilot cluster and prepare it as a sandbox."""
        del wait_timeout  # SkyPilot owns provisioning timeouts/retries.
        _log = log or print
        _repair_empty_skypilot_catalogs(_log)
        sky = _require_sky()

        def _check_cancel() -> None:
            if cancel_event and cancel_event.is_set():
                _log("Sandbox creation cancelled by user.")
                raise cls.Cancelled("Sandbox creation cancelled.")

        base = name or "ml-intern"
        suffix = uuid.uuid4().hex[:8]
        owner_part = f"{owner}-" if owner else ""
        cluster_name = _sanitize_cluster_name(f"{base}-{owner_part}{suffix}")
        resources = _make_resources(sky, hardware=hardware, infra=infra)
        task_secrets = dict(secrets or {})
        if token and "HF_TOKEN" not in task_secrets:
            task_secrets["HF_TOKEN"] = token

        ensure_app_dir = (
            "(mkdir -p /app 2>/dev/null || sudo mkdir -p /app) && "
            "(test -w /app || sudo chown -R \"$USER\":\"$USER\" /app)"
        )
        setup = (
            f"{ensure_app_dir} && "
            "python -m pip install --user -q --upgrade pip >/dev/null 2>&1 || true"
        )
        task = sky.Task(
            name=cluster_name,
            setup=setup,
            run=f"{ensure_app_dir} && echo 'SkyPilot sandbox ready'",
            secrets=task_secrets or None,
        )
        task.set_resources(resources)

        _log(f"Creating SkyPilot sandbox cluster: {cluster_name} ({infra}, {hardware})...")
        _check_cancel()
        request_id = sky.launch(
            task,
            cluster_name=cluster_name,
            idle_minutes_to_autostop=45 if hardware not in CPU_ALIASES else None,
            retry_until_up=True,
            down=False,
            _need_confirmation=False,
        )
        _check_cancel()
        _wait_for_request(sky, request_id, stream=True)
        _check_cancel()

        sb = cls(
            space_id=cluster_name,
            token=token,
            infra=infra,
            hardware=hardware,
            _owns_space=True,
        )
        _log(f"SkyPilot sandbox ready: {cluster_name}")
        return sb

    @classmethod
    def connect(
        cls,
        space_id: str,
        *,
        token: str | None = None,
        infra: str = DEFAULT_INFRA,
    ) -> "Sandbox":
        return cls(space_id=space_id, token=token, infra=infra, _owns_space=False)

    def delete(self) -> None:
        if not self._owns_space:
            raise RuntimeError(
                f"This Sandbox did not create {self.space_id}. "
                "Use SkyPilot directly if you are sure you want to delete it."
            )
        sky = _require_sky()
        request_id = sky.down(self.cluster_name)
        _wait_for_request(sky, request_id, stream=True)

    def pause(self) -> None:
        sky = _require_sky()
        if hasattr(sky, "stop"):
            request_id = sky.stop(self.cluster_name)
            _wait_for_request(sky, request_id, stream=True)
            return
        raise RuntimeError("This SkyPilot SDK does not expose sky.stop().")

    def restart(self) -> None:
        self.bash("true")

    @property
    def url(self) -> str:
        return f"skypilot://{self.cluster_name}"

    @property
    def status(self) -> str:
        sky = _require_sky()
        if not hasattr(sky, "status"):
            return "unknown"
        try:
            result, _ = _wait_for_request(sky, sky.status(self.cluster_name), stream=False)
            return str(result)
        except Exception:
            return "unknown"

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc):
        if self._owns_space:
            try:
                self.delete()
            except Exception as e:
                print(f"Warning: failed to delete sandbox: {e}", file=sys.stderr)

    def _exec(self, command: str, *, timeout: int | None = None) -> ToolResult:
        sky = _require_sky()
        effective_timeout = min(timeout or self.timeout, MAX_TIMEOUT)
        task = sky.Task(
            name=f"{self.cluster_name}-cmd",
            run=command,
            envs={
                "HF_TOKEN": self.token or os.environ.get("HF_TOKEN", ""),
                "UV_NO_PROGRESS": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "TQDM_DISABLE": "1",
            },
        )
        request_id = sky.exec(task, cluster_name=self.cluster_name, down=False)
        try:
            result, logs = _wait_for_request(sky, request_id, stream=True)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
        self._latest_job_id = _extract_job_id(result)
        exit_code = 0
        if self._latest_job_id is not None and hasattr(sky, "tail_logs"):
            log_buf = io.StringIO()
            try:
                tail_result = sky.tail_logs(
                    self.cluster_name,
                    self._latest_job_id,
                    follow=True,
                    output_stream=log_buf,
                )
                if isinstance(tail_result, int):
                    exit_code = tail_result
            except Exception as e:
                output = _truncate_output((logs + "\n" + log_buf.getvalue()).strip())
                return ToolResult(success=False, output=output, error=str(e))
            logs = log_buf.getvalue() or logs
        output = _truncate_output(logs.strip())
        success = _request_succeeded(result) and exit_code == 0
        if success:
            return ToolResult(success=True, output=output or "(no output)")
        return ToolResult(success=False, output=output, error=f"Exit code {exit_code}")

    def bash(
        self,
        command: str,
        *,
        work_dir: str | None = None,
        timeout: int | None = None,
        description: str | None = None,
    ) -> ToolResult:
        del description
        effective_timeout = min(timeout or self.timeout, MAX_TIMEOUT)
        work = shlex.quote(work_dir or self.work_dir)
        ensure_work = (
            f"(mkdir -p {work} 2>/dev/null || sudo mkdir -p {work}) && "
            f"(test -w {work} || sudo chown -R \"$USER\":\"$USER\" {work})"
        )
        wrapped = (
            f"{ensure_work} && cd {work} && "
            f"timeout {effective_timeout}s bash -lc {shlex.quote(command)}"
        )
        return self._exec(wrapped, timeout=effective_timeout)

    def read(
        self, path: str, *, offset: int | None = None, limit: int | None = None
    ) -> ToolResult:
        result = self._exec(
            _remote_python_command(
                _READ_SCRIPT,
                {
                    "path": path,
                    "offset": offset,
                    "limit": limit or (DEFAULT_READ_LIMIT if offset is None else None),
                },
            )
        )
        if result.success:
            self._files_read.add(path)
        return result

    def write(self, path: str, content: str) -> ToolResult:
        if path not in self._files_read:
            check = self._exec(_remote_python_command(_EXISTS_SCRIPT, {"path": path}))
            if check.success and check.output.strip().endswith("true"):
                return ToolResult(
                    success=False,
                    error=(
                        f"File {path} exists but has not been read this session. "
                        "Read it first, or use sandbox_edit for targeted changes."
                    ),
                )
        result = self._exec(_remote_python_command(_WRITE_SCRIPT, {"path": path, "content": content}))
        if result.success:
            self._files_read.add(path)
        return result

    def edit(
        self,
        path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
        mode: str = "replace",
    ) -> ToolResult:
        if old_str == new_str:
            return ToolResult(success=False, error="old_str and new_str are identical.")
        if path not in self._files_read:
            return ToolResult(
                success=False,
                error=f"File {path} has not been read this session. Read it first.",
            )
        return self._exec(
            _remote_python_command(
                _EDIT_SCRIPT,
                {
                    "path": path,
                    "old_str": old_str,
                    "new_str": new_str,
                    "replace_all": replace_all,
                    "mode": mode,
                },
            )
        )

    def kill_all(self) -> ToolResult:
        sky = _require_sky()
        if self._latest_job_id is not None and hasattr(sky, "cancel"):
            try:
                request_id = sky.cancel(self.cluster_name, job_ids=[self._latest_job_id])
                _wait_for_request(sky, request_id, stream=True)
                return ToolResult(success=True, output=f"Cancelled SkyPilot job {self._latest_job_id}")
            except Exception as e:
                return ToolResult(success=False, error=str(e))
        return self.bash("pkill -TERM -u \"$USER\" || true", timeout=30)

    TOOLS = {
        "bash": {
            "description": (
                "Run a shell command in the remote SkyPilot sandbox and return stdout/stderr.\n"
                "\n"
                "IMPORTANT: Do NOT use bash for file operations - use the dedicated tools instead:\n"
                "- To read files: use read (not cat/head/tail)\n"
                "- To edit files: use edit (not sed/awk)\n"
                "- To write files: use write (not echo/cat <<EOF)\n"
                "\n"
                "Commands run in a shell at /app. Each invocation is independent - "
                "use files in /app to persist state.\n"
                "Chain dependent commands with &&. Independent commands should be "
                "separate bash calls (they can run in parallel).\n"
                "\n"
                "Timeout default 240s, max 1200s."
            ),
            "parameters": {
                "type": "object",
                "required": ["command"],
                "additionalProperties": False,
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
                    "description": {
                        "type": "string",
                        "description": "Short description (5-10 words, active voice).",
                    },
                    "work_dir": {"type": "string", "description": "Working directory (default: /app)."},
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds (default: 240, max: 1200).",
                    },
                },
            },
        },
        "read": {
            "description": (
                "Reads a file from the SkyPilot sandbox filesystem. Returns contents "
                "with line numbers (cat -n format).\n"
                "\n"
                "Usage:\n"
                "- By default, reads up to 2000 lines from the beginning of the file.\n"
                "- You can optionally specify offset and limit for large files.\n"
                "- Cannot read directories - use bash with 'ls' instead.\n"
                "- IMPORTANT: Always read a file before editing or overwriting it."
            ),
            "parameters": {
                "type": "object",
                "required": ["path"],
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to read."},
                    "offset": {
                        "type": "integer",
                        "description": "The line number to start reading from (1-based).",
                    },
                    "limit": {"type": "integer", "description": "The number of lines to read."},
                },
            },
        },
        "write": {
            "description": (
                "Writes a file to the SkyPilot sandbox filesystem. Overwrites the "
                "existing file if one exists at the path.\n"
                "\n"
                "- If this is an existing file, you MUST use the read tool first.\n"
                "- ALWAYS prefer editing existing files with the edit tool over overwriting.\n"
                "- Creates parent directories as needed."
            ),
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to write."},
                    "content": {"type": "string", "description": "The complete file content to write."},
                },
            },
        },
        "edit": {
            "description": (
                "Performs string replacements in files on the SkyPilot sandbox. "
                "Supports exact matching with fuzzy fallback.\n"
                "\n"
                "Usage:\n"
                "- You must read the file at least once before editing.\n"
                "- The edit will fail if old_str is not unique unless replace_all is true.\n"
                "- Preserve indentation exactly as it appears in the file."
            ),
            "parameters": {
                "type": "object",
                "required": ["path", "old_str", "new_str"],
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to edit."},
                    "old_str": {"type": "string", "description": "The text to find in the file."},
                    "new_str": {"type": "string", "description": "The replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences of old_str (default: false).",
                        "default": False,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "append_after", "prepend_before"],
                        "description": "Edit mode (default: replace).",
                        "default": "replace",
                    },
                },
            },
        },
    }

    @classmethod
    def tool_definitions(cls) -> list[dict]:
        return [{"name": name, **spec} for name, spec in cls.TOOLS.items()]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        dispatch = {
            "bash": lambda a: self.bash(
                a["command"],
                work_dir=a.get("work_dir"),
                timeout=a.get("timeout"),
                description=a.get("description"),
            ),
            "read": lambda a: self.read(a["path"], offset=a.get("offset"), limit=a.get("limit")),
            "write": lambda a: self.write(a["path"], a["content"]),
            "edit": lambda a: self.edit(
                a["path"],
                a["old_str"],
                a["new_str"],
                replace_all=a.get("replace_all", False),
                mode=a.get("mode", "replace"),
            ),
        }
        fn = dispatch.get(name)
        if not fn:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return fn(arguments)


def _sanitize_cluster_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9-]", "-", name).strip("-").lower()
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned[:63] or f"ml-intern-{uuid.uuid4().hex[:8]}"


def _make_resources(sky: Any, *, hardware: str, infra: str) -> Any:
    kwargs: dict[str, Any] = {"infra": infra}
    if infra.lower().startswith("runpod"):
        # RunPod rejects the larger disk default SkyPilot otherwise selects.
        kwargs["disk_size"] = 20
    cpu = CPU_ALIASES.get(hardware)
    if cpu is not None:
        kwargs.update(cpu)
    else:
        kwargs["accelerators"] = hardware
    return sky.Resources(**kwargs)
