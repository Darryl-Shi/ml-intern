from agent.core.tools import create_builtin_tools


def _tool_names(local_mode: bool = False) -> set[str]:
    return {tool.name for tool in create_builtin_tools(local_mode=local_mode)}


def test_hf_jobs_is_not_registered_as_builtin_tool():
    assert "hf_jobs" not in _tool_names(local_mode=False)
    assert "hf_jobs" not in _tool_names(local_mode=True)


def test_remote_mode_uses_skypilot_sandbox_tools():
    names = _tool_names(local_mode=False)

    assert "sandbox_create" in names
    assert "bash" in names
    assert "read" in names
    assert "write" in names
    assert "edit" in names
