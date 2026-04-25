from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastmcp import Client

import agent.mcp_server as mcp_server_mod
from agent.mcp_types import ToolSpec
from agent.mcp_server import MCPSessionState, _tool_from_spec, create_mcp_server


async def _text(result) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


@pytest.mark.asyncio
async def test_mcp_server_registers_specialist_tools_only():
    mcp = await create_mcp_server(load_dynamic_api_schema=False)
    names = {tool.name for tool in await mcp.list_tools()}

    assert {
        "explore_hf_docs",
        "fetch_hf_docs",
        "find_hf_api",
        "hf_papers",
        "hf_inspect_dataset",
        "hf_repo_files",
        "hf_repo_git",
        "github_find_examples",
        "github_list_repos",
        "github_read_file",
        "sandbox_create",
        "sandbox_bash",
        "sandbox_read",
        "sandbox_write",
        "sandbox_edit",
    }.issubset(names)

    assert "research" not in names
    assert "bash" not in names
    assert "read" not in names
    assert "write" not in names
    assert "edit" not in names


@pytest.mark.asyncio
async def test_mcp_handler_success_and_failure_are_text_results():
    async def ok_handler(arguments):
        return f"hello {arguments['name']}", True

    async def fail_handler(arguments):
        return f"bad {arguments['name']}", False

    session = MCPSessionState()
    mcp = await create_mcp_server(
        session=session,
        load_dynamic_api_schema=False,
    )
    mcp.add_tool(
        _tool_from_spec(
            ToolSpec(
                name="unit_ok",
                description="test success",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                handler=ok_handler,
            ),
            session=session,
        )
    )
    mcp.add_tool(
        _tool_from_spec(
            ToolSpec(
                name="unit_fail",
                description="test failure",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                handler=fail_handler,
            ),
            session=session,
        )
    )

    async with Client(mcp) as client:
        ok = await client.call_tool("unit_ok", {"name": "Ada"})
        fail = await client.call_tool(
            "unit_fail", {"name": "Ada"}, raise_on_error=False
        )

    assert await _text(ok) == "hello Ada"
    assert fail.is_error is True
    assert "bad Ada" in await _text(fail)


@pytest.mark.asyncio
async def test_mcp_session_refreshes_hf_token_from_environment(monkeypatch):
    seen_tokens = []

    async def token_handler(arguments, session=None):
        seen_tokens.append(session.hf_token)
        return session.hf_token or "", True

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    session = MCPSessionState(hf_token=None)
    mcp = await create_mcp_server(
        session=session,
        load_dynamic_api_schema=False,
    )
    mcp.add_tool(
        _tool_from_spec(
            ToolSpec(
                name="unit_token",
                description="test token",
                parameters={"type": "object", "properties": {}},
                handler=token_handler,
            ),
            session=session,
        )
    )

    async with Client(mcp) as client:
        result = await client.call_tool("unit_token", {})

    assert await _text(result) == "hf_test_token"
    assert seen_tokens == ["hf_test_token"]


@dataclass
class _FakeSandboxResult:
    success: bool
    output: str = ""
    error: str = ""


class _FakeSandbox:
    space_id = "fake-sandbox"
    url = "https://example.test/sandbox"

    def __init__(self):
        self.calls = []

    def call_tool(self, name, args):
        self.calls.append((name, args))
        payload = args.get("command") or args.get("path") or "ok"
        return _FakeSandboxResult(True, f"{name}: {payload}")


@pytest.mark.asyncio
async def test_all_mcp_tools_are_callable_with_safe_mocks(monkeypatch):
    calls = []

    def handler_for(name):
        async def handler(arguments, session=None):
            calls.append((name, arguments, getattr(session, "hf_token", None)))
            return f"{name} ok", True

        return handler

    handler_attrs = {
        "explore_hf_docs_handler": "explore_hf_docs",
        "hf_docs_fetch_handler": "fetch_hf_docs",
        "search_openapi_handler": "find_hf_api",
        "hf_papers_handler": "hf_papers",
        "hf_inspect_dataset_handler": "hf_inspect_dataset",
        "hf_repo_files_handler": "hf_repo_files",
        "hf_repo_git_handler": "hf_repo_git",
        "github_find_examples_handler": "github_find_examples",
        "github_list_repos_handler": "github_list_repos",
        "github_read_file_handler": "github_read_file",
    }
    for attr, name in handler_attrs.items():
        monkeypatch.setattr(mcp_server_mod, attr, handler_for(name))

    async def fake_api_spec():
        return {
            "name": "find_hf_api",
            "description": "Find HF API endpoints.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": [],
            },
        }

    fake_sandbox = _FakeSandbox()

    async def fake_ensure_sandbox(session, hardware="cpu-basic", **kwargs):
        session.sandbox = fake_sandbox
        return fake_sandbox, None

    monkeypatch.setattr(mcp_server_mod, "_get_api_search_tool_spec", fake_api_spec)
    monkeypatch.setattr(
        "agent.tools.sandbox_tool._ensure_sandbox",
        fake_ensure_sandbox,
    )
    monkeypatch.setenv("HF_TOKEN", "hf_all_tools")

    mcp = await mcp_server_mod.create_mcp_server(load_dynamic_api_schema=True)
    async with Client(mcp) as client:
        tools = {tool.name for tool in await client.list_tools()}
        assert tools == {
            "explore_hf_docs",
            "fetch_hf_docs",
            "find_hf_api",
            "hf_papers",
            "hf_inspect_dataset",
            "hf_repo_files",
            "hf_repo_git",
            "github_find_examples",
            "github_list_repos",
            "github_read_file",
            "sandbox_create",
            "sandbox_bash",
            "sandbox_read",
            "sandbox_write",
            "sandbox_edit",
        }

        calls_to_make = {
            "explore_hf_docs": {"endpoint": "hub"},
            "fetch_hf_docs": {"url": "https://huggingface.co/docs/hub/index"},
            "find_hf_api": {"query": "repo"},
            "hf_papers": {"operation": "trending"},
            "hf_inspect_dataset": {"dataset": "org/dataset"},
            "hf_repo_files": {"operation": "list"},
            "hf_repo_git": {"operation": "list_refs"},
            "github_find_examples": {"repo": "transformers", "keyword": "trainer"},
            "github_list_repos": {"owner": "huggingface"},
            "github_read_file": {"repo": "huggingface/transformers", "path": "README.md"},
        }
        for tool_name, args in calls_to_make.items():
            result = await client.call_tool(tool_name, args)
            assert await _text(result) == f"{tool_name} ok"

        assert "Sandbox created: fake-sandbox" in await _text(
            await client.call_tool("sandbox_create", {})
        )
        sandbox_calls = {
            "sandbox_bash": {"command": "pwd"},
            "sandbox_read": {"path": "/app/train.py"},
            "sandbox_write": {"path": "/app/train.py", "content": "print('ok')"},
            "sandbox_edit": {
                "path": "/app/train.py",
                "old_str": "ok",
                "new_str": "done",
            },
        }
        for tool_name, args in sandbox_calls.items():
            result = await client.call_tool(tool_name, args)
            assert result.is_error is False

    assert [call[0] for call in calls] == list(calls_to_make)
    assert all(call[2] == "hf_all_tools" for call in calls)
    assert [name for name, _ in fake_sandbox.calls] == [
        "bash",
        "read",
        "write",
        "edit",
    ]


@pytest.mark.asyncio
async def test_mcp_sandbox_requires_create_and_persists(monkeypatch):
    fake = _FakeSandbox()

    async def fake_ensure_sandbox(session, hardware="cpu-basic", **kwargs):
        session.sandbox = fake
        return fake, None

    monkeypatch.setattr(
        "agent.tools.sandbox_tool._ensure_sandbox",
        fake_ensure_sandbox,
    )

    session = MCPSessionState()
    mcp = await create_mcp_server(
        session=session,
        load_dynamic_api_schema=False,
    )

    async with Client(mcp) as client:
        before_create = await client.call_tool(
            "sandbox_bash",
            {"command": "pwd"},
            raise_on_error=False,
        )
        created = await client.call_tool("sandbox_create", {})
        bash = await client.call_tool("sandbox_bash", {"command": "echo hi"})

    assert before_create.is_error is True
    assert "Call sandbox_create first" in await _text(before_create)
    assert "Sandbox created: fake-sandbox" in await _text(created)
    assert await _text(bash) == "bash: echo hi"
    assert fake.calls == [("bash", {"command": "echo hi"})]
