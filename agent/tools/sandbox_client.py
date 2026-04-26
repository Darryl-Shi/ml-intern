#!/usr/bin/env python3
"""
Sandbox Tools - RunPod sandbox provider.

    sb = Sandbox.create(hardware="A100:1")
    sb.bash("python train.py")
    sb.read("/app/train.py")
    sb.edit("/app/train.py", old_str="lr=1e-3", new_str="lr=1e-4")
    sb.delete()
"""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

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

RUNPOD_API_BASE = "https://rest.runpod.io/v1"
_SSH_KEY_NAME = "runpod_ml_intern"

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


# ── RunPod API helpers ────────────────────────────────────────────────


def _get_runpod_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError(
            "RUNPOD_API_KEY is not set. "
            "Get your API key from https://www.runpod.io/console/user/settings "
            "and export it: export RUNPOD_API_KEY=your_key_here"
        )
    return key


def _runpod_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_get_runpod_api_key()}"}


def _ensure_ssh_key() -> tuple[str, str]:
    """Ensure an SSH key pair exists for RunPod pods.

    Returns:
        (private_key_path, public_key_string)
    """
    key_dir = pathlib.Path.home() / ".ssh"
    key_dir.mkdir(mode=0o700, exist_ok=True)
    priv_path = key_dir / _SSH_KEY_NAME
    pub_path = key_dir / f"{_SSH_KEY_NAME}.pub"

    if not priv_path.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(priv_path), "-N", "", "-q"],
            check=True,
            capture_output=True,
        )

    return str(priv_path), pub_path.read_text().strip()


def _parse_hardware(hardware: str) -> tuple[str | None, int | None]:
    """Parse hardware string like 'RTX4090:1' or 'A100-80GB:4'.

    Returns:
        (gpu_type, gpu_count) — both None if hardware is a CPU alias.
    """
    cpu = CPU_ALIASES.get(hardware)
    if cpu is not None:
        return None, None
    if ":" in hardware:
        parts = hardware.rsplit(":", 1)
        return parts[0], int(parts[1])
    return hardware, 1


def _create_runpod_pod(
    name: str,
    image_name: str,
    gpu_type: str | None,
    gpu_count: int | None,
    ssh_pub_key: str,
    env: dict[str, str] | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "name": name,
        "imageName": image_name,
        "containerDiskInGb": 20,
        "env": dict(env or {}),
        "dockerStartCmd": "sleep infinity",
    }
    payload["env"]["SSH_PUBLIC_KEY"] = ssh_pub_key

    if gpu_type and gpu_count:
        payload["gpuTypeIds"] = [gpu_type]
        payload["gpuCount"] = gpu_count
    else:
        payload["computeType"] = "CPU"

    resp = httpx.post(
        f"{RUNPOD_API_BASE}/pods",
        headers=_runpod_headers(),
        json=payload,
        timeout=30,
    )
    if resp.status_code == 401:
        raise RuntimeError(
            "RunPod API key rejected (401 Unauthorized). "
            "Check your RUNPOD_API_KEY environment variable."
        )
    resp.raise_for_status()
    return resp.json()


def _get_runpod_pod(pod_id: str) -> dict:
    resp = httpx.get(
        f"{RUNPOD_API_BASE}/pods/{pod_id}",
        headers=_runpod_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _delete_runpod_pod(pod_id: str) -> None:
    httpx.delete(
        f"{RUNPOD_API_BASE}/pods/{pod_id}",
        headers=_runpod_headers(),
        timeout=15,
    )


def _stop_runpod_pod(pod_id: str) -> None:
    httpx.post(
        f"{RUNPOD_API_BASE}/pods/{pod_id}/stop",
        headers=_runpod_headers(),
        timeout=15,
    )


def _resume_runpod_pod(pod_id: str) -> None:
    httpx.post(
        f"{RUNPOD_API_BASE}/pods/{pod_id}/start",
        headers=_runpod_headers(),
        timeout=15,
    )


def _restart_runpod_pod(pod_id: str) -> None:
    httpx.post(
        f"{RUNPOD_API_BASE}/pods/{pod_id}/restart",
        headers=_runpod_headers(),
        timeout=15,
    )


def _wait_for_pod_running(pod_id: str, *, timeout: int = WAIT_TIMEOUT,
                           poll_interval: float = 5.0,
                           log: Callable[[str], object] | None = None) -> dict:
    _log = log or (lambda _: None)
    deadline = time.monotonic() + timeout
    last_status = None
    while time.monotonic() < deadline:
        pod = _get_runpod_pod(pod_id)
        status = pod.get("desiredStatus", "")
        if status != last_status:
            _log(f"Pod status: {status}")
            last_status = status
        if status == "RUNNING":
            return pod
        time.sleep(poll_interval)
    raise TimeoutError(
        f"RunPod pod {pod_id} did not reach RUNNING status within {timeout}s "
        f"(last status: {last_status})"
    )


def _extract_pod_connection(pod: dict) -> tuple[str, int]:
    """Extract IP and SSH port from a running pod's data."""
    pod_ip = pod.get("ip") or pod.get("runtime", {}).get("ip")
    if not pod_ip:
        raise RuntimeError(f"Cannot determine pod IP from RunPod response: {pod.get('id', '?')}")
    ssh_port = 22
    ports = pod.get("ports") or pod.get("runtime", {}).get("ports", [])
    for p in ports:
        if isinstance(p, dict) and p.get("privatePort") == 22:
            ssh_port = int(p.get("publicPort", 22))
            break
    return pod_ip, ssh_port


# ── Remote Python scripts (sent via SSH) ──────────────────────────────


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


# ── ToolResult ────────────────────────────────────────────────────────


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


# ── Sandbox ───────────────────────────────────────────────────────────


@dataclass
class Sandbox:
    """A handle to a RunPod sandbox pod."""

    space_id: str
    token: str | None = None
    work_dir: str = "/app"
    timeout: int = DEFAULT_TIMEOUT
    infra: str = DEFAULT_INFRA
    hardware: str = "cpu-basic"
    _owns_space: bool = field(default=False, repr=False)
    _files_read: set[str] = field(init=False, repr=False, default_factory=set)
    _pod_ip: str = field(default="", repr=False)
    _pod_ssh_port: int = field(default=22, repr=False)
    _ssh_key_path: str = field(default="", repr=False)

    class Cancelled(Exception):
        """Raised when sandbox creation is cancelled by the user."""

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
        """Create a RunPod pod and prepare it as a sandbox."""
        _log = log or print

        def _check_cancel() -> None:
            if cancel_event and cancel_event.is_set():
                _log("Sandbox creation cancelled by user.")
                raise cls.Cancelled("Sandbox creation cancelled.")

        base = name or "ml-intern"
        suffix = uuid.uuid4().hex[:8]
        owner_part = f"{owner}-" if owner else ""
        pod_name = f"{base}-{owner_part}{suffix}"[:63]

        gpu_type, gpu_count = _parse_hardware(hardware)
        image_name = os.environ.get("RUNPOD_IMAGE_NAME", "runpod/pytorch:latest")

        _log(f"Ensuring SSH key for RunPod...")
        ssh_key_path, ssh_pub_key = _ensure_ssh_key()
        _check_cancel()

        task_env: dict[str, str] = {}
        if token:
            task_env["HF_TOKEN"] = token
        if secrets:
            task_env.update(secrets)

        _log(f"Creating RunPod pod: {pod_name} ({hardware})...")
        _check_cancel()
        try:
            pod_data = _create_runpod_pod(
                name=pod_name,
                image_name=image_name,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                ssh_pub_key=ssh_pub_key,
                env=task_env,
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"RunPod API error: {e.response.status_code} {e.response.text}") from e

        pod_id = pod_data.get("id") or pod_data.get("pod", {}).get("id", "")
        if not pod_id:
            raise RuntimeError(f"RunPod did not return a pod ID: {pod_data}")

        _log(f"Waiting for pod {pod_id} to reach RUNNING status...")
        _check_cancel()
        try:
            pod = _wait_for_pod_running(pod_id, timeout=wait_timeout, log=_log)
        except TimeoutError:
            _log("Pod did not become ready in time. Cleaning up...")
            try:
                _delete_runpod_pod(pod_id)
            except Exception:
                pass
            raise

        _check_cancel()
        pod_ip, ssh_port = _extract_pod_connection(pod)

        _log(f"RunPod sandbox ready: {pod_id} ({pod_ip}:{ssh_port})")

        sb = cls(
            space_id=pod_id,
            token=token,
            infra=infra,
            hardware=hardware,
            _owns_space=True,
            _pod_ip=pod_ip,
            _pod_ssh_port=ssh_port,
            _ssh_key_path=ssh_key_path,
        )
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
                "Use RunPod directly if you are sure you want to delete it."
            )
        _delete_runpod_pod(self.space_id)

    def pause(self) -> None:
        _stop_runpod_pod(self.space_id)

    def restart(self) -> None:
        _restart_runpod_pod(self.space_id)
        pod = _wait_for_pod_running(self.space_id, timeout=WAIT_TIMEOUT)
        self._pod_ip, self._pod_ssh_port = _extract_pod_connection(pod)

    @property
    def url(self) -> str:
        return f"https://www.runpod.io/console/pods/{self.space_id}"

    @property
    def status(self) -> str:
        try:
            pod = _get_runpod_pod(self.space_id)
            return pod.get("desiredStatus", "unknown")
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
        import paramiko

        effective_timeout = min(timeout or self.timeout, MAX_TIMEOUT)

        try:
            key = paramiko.Ed25519Key.from_private_key_file(self._ssh_key_path)
        except paramiko.ssh_exception.SSHException:
            key = paramiko.RSAKey.from_private_key_file(self._ssh_key_path)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(
                hostname=self._pod_ip,
                port=self._pod_ssh_port,
                username="root",
                pkey=key,
                timeout=15,
                banner_timeout=30,
            )

            env_vars = (
                f"HF_TOKEN={shlex.quote(self.token or os.environ.get('HF_TOKEN', ''))} "
                "UV_NO_PROGRESS=1 "
                "HF_HUB_DISABLE_PROGRESS_BARS=1 "
                "TQDM_DISABLE=1"
            )
            wrapped = (
                f"cd {shlex.quote(self.work_dir)} && "
                f"export {env_vars} && "
                f"timeout {effective_timeout}s bash -lc {shlex.quote(command)}"
            )

            chan = client.get_transport().open_session(timeout=effective_timeout)
            chan.set_combine_stderr(True)
            chan.exec_command(wrapped)

            stdout_buf = io.StringIO()
            while True:
                if chan.exit_status_ready():
                    break
                if chan.recv_ready():
                    data = chan.recv(4096).decode(errors="replace")
                    stdout_buf.write(data)

            # Drain remaining data
            while chan.recv_ready():
                stdout_buf.write(chan.recv(4096).decode(errors="replace"))

            exit_code = chan.recv_exit_status()
            output = _truncate_output(stdout_buf.getvalue().strip())

            if exit_code == 0:
                return ToolResult(success=True, output=output or "(no output)")
            return ToolResult(success=False, output=output, error=f"Exit code {exit_code}")

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
        finally:
            client.close()

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
        return self.bash("pkill -TERM -u \"$USER\" || true", timeout=30)

    TOOLS = {
        "bash": {
            "description": (
                "Run a shell command in the remote RunPod sandbox and return stdout/stderr.\n"
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
                "Reads a file from the RunPod sandbox filesystem. Returns contents "
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
                "Writes a file to the RunPod sandbox filesystem. Overwrites the "
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
                "Performs string replacements in files on the RunPod sandbox. "
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