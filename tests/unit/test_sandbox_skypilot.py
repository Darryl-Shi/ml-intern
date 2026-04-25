import sys

from agent.tools import sandbox_tool
from agent.tools.sandbox_client import Sandbox, ToolResult, _make_resources


class FakeTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.resources = None

    def set_resources(self, resources):
        self.resources = resources
        return self


class FakeSky:
    def __init__(self):
        self.launch_calls = []
        self.exec_calls = []
        self.down_calls = []
        self.cancel_calls = []
        self.tail_log_calls = []

    def Task(self, **kwargs):
        return FakeTask(**kwargs)

    def Resources(self, **kwargs):
        return {"resources": kwargs}

    def launch(self, task, **kwargs):
        self.launch_calls.append((task, kwargs))
        return "launch-request"

    def exec(self, task, **kwargs):
        self.exec_calls.append((task, kwargs))
        return "exec-request"

    def down(self, cluster_name):
        self.down_calls.append(cluster_name)
        return "down-request"

    def cancel(self, cluster_name, **kwargs):
        self.cancel_calls.append((cluster_name, kwargs))
        return "cancel-request"

    def stream_and_get(self, request_id, output_stream=None):
        if output_stream is not None:
            output_stream.write(f"{request_id} logs\n")
        if request_id == "exec-request":
            return (7, None)
        return 0

    def tail_logs(self, cluster_name, job_id, follow, tail=0, output_stream=None, **kwargs):
        self.tail_log_calls.append((cluster_name, job_id, follow, tail, kwargs))
        if output_stream is not None:
            output_stream.write("remote command output\n")
        return 0


def test_create_defaults_to_runpod_and_accepts_arbitrary_accelerator(monkeypatch):
    fake = FakeSky()
    monkeypatch.setitem(sys.modules, "sky", fake)

    sb = Sandbox.create(owner="user", hardware="RTX4090:1", token="hf_test")

    assert sb.infra == "runpod"
    task, kwargs = fake.launch_calls[0]
    assert kwargs["cluster_name"].startswith("ml-intern-user-")
    assert kwargs["idle_minutes_to_autostop"] == 45
    assert task.resources == {
        "resources": {"infra": "runpod", "disk_size": 20, "accelerators": "RTX4090:1"}
    }
    assert task.kwargs["secrets"] == {"HF_TOKEN": "hf_test"}


def test_cpu_alias_does_not_set_accelerators():
    fake = FakeSky()
    resources = _make_resources(fake, hardware="cpu-basic", infra="runpod")
    assert resources == {
        "resources": {"infra": "runpod", "disk_size": 20, "cpus": "2+", "memory": "8+"}
    }


def test_delete_and_cancel_use_skypilot_sdk(monkeypatch):
    fake = FakeSky()
    monkeypatch.setitem(sys.modules, "sky", fake)
    sb = Sandbox("cluster-a", _owns_space=True)

    sb.delete()
    assert fake.down_calls == ["cluster-a"]

    sb._latest_job_id = 42
    result = sb.kill_all()
    assert result.success
    assert fake.cancel_calls == [("cluster-a", {"job_ids": [42]})]


def test_bash_waits_for_remote_job_logs(monkeypatch):
    fake = FakeSky()
    monkeypatch.setitem(sys.modules, "sky", fake)
    sb = Sandbox("cluster-a")

    result = sb.bash("echo hi")

    assert result.success
    assert result.output == "remote command output"
    assert fake.tail_log_calls == [("cluster-a", 7, True, 0, {})]


def test_write_existing_file_requires_read(monkeypatch):
    sb = Sandbox("cluster-a")
    calls = []

    def fake_exec(command, *, timeout=None):
        calls.append(command)
        return ToolResult(success=True, output="true")

    monkeypatch.setattr(sb, "_exec", fake_exec)

    result = sb.write("/app/existing.py", "print('hi')\n")

    assert not result.success
    assert "has not been read" in result.error
    assert len(calls) == 1


def test_parse_sky_show_gpus_imports_names_and_quantities():
    output = """
COMMON_GPU  AVAILABLE_QUANTITIES
A100        1, 2, 4, 8
A100-80GB   1, 2
OTHER_GPU          AVAILABLE_QUANTITIES
RTX4090            1, 2, 3, 4, 6, 8, 12
"""

    options = sandbox_tool._parse_sky_show_gpus(output)

    assert "A100" in options
    assert "A100:8" in options
    assert "A100-80GB:2" in options
    assert "RTX4090:12" in options
