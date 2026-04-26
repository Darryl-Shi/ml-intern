"""MCP server entrypoint for ML Intern specialist tools.

This server is meant to be used from agentic CLIs that already provide local
shell and filesystem tools. It exposes ML/Hugging Face research tools plus a
remote RunPod sandbox, and deliberately does not expose local bash/read/write.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from fastmcp import FastMCP
from fastmcp.tools.base import Tool
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
from pydantic import PrivateAttr

from agent.mcp_types import Event, ToolSpec
from agent.tools.dataset_tools import (
    HF_INSPECT_DATASET_TOOL_SPEC,
    hf_inspect_dataset_handler,
)
from agent.tools.docs_tools import (
    EXPLORE_HF_DOCS_TOOL_SPEC,
    HF_DOCS_FETCH_TOOL_SPEC,
    _get_api_search_tool_spec,
    explore_hf_docs_handler,
    hf_docs_fetch_handler,
    search_openapi_handler,
)
from agent.tools.github_find_examples import (
    GITHUB_FIND_EXAMPLES_TOOL_SPEC,
    github_find_examples_handler,
)
from agent.tools.github_list_repos import (
    GITHUB_LIST_REPOS_TOOL_SPEC,
    github_list_repos_handler,
)
from agent.tools.github_read_file import (
    GITHUB_READ_FILE_TOOL_SPEC,
    github_read_file_handler,
)
from agent.tools.hf_repo_files_tool import (
    HF_REPO_FILES_TOOL_SPEC,
    hf_repo_files_handler,
)
from agent.tools.hf_repo_git_tool import (
    HF_REPO_GIT_TOOL_SPEC,
    hf_repo_git_handler,
)
from agent.tools.papers_tool import HF_PAPERS_TOOL_SPEC, hf_papers_handler
from agent.tools.sandbox_tool import get_sandbox_tools

MAX_MCP_TOOL_OUTPUT_CHARS = int(
    os.environ.get("ML_INTERN_MCP_MAX_OUTPUT_CHARS", "30000")
)


@dataclass
class MCPSessionState:
    """Lightweight state shared by MCP tool calls in one server process."""

    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    session_id: str = field(default_factory=lambda: f"mcp-{uuid.uuid4().hex[:12]}")
    event_queue: asyncio.Queue[Event] = field(default_factory=asyncio.Queue)
    sandbox: Any = None
    events: list[Event] = field(default_factory=list)
    _cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    async def send_event(self, event: Event) -> None:
        self.events.append(event)
        await self.event_queue.put(event)

    def refresh_env(self) -> None:
        """Pick up auth set by the host process before each tool call."""
        self.hf_token = os.environ.get("HF_TOKEN") or self.hf_token


Handler = Callable[..., Awaitable[tuple[str, bool]]]


def _bounded_text(text: str) -> str:
    if len(text) <= MAX_MCP_TOOL_OUTPUT_CHARS:
        return text
    omitted = len(text) - MAX_MCP_TOOL_OUTPUT_CHARS
    return (
        text[:MAX_MCP_TOOL_OUTPUT_CHARS]
        + f"\n\n[ml-intern-mcp truncated {omitted} characters from this tool output.]"
    )


class HandlerTool(Tool):
    """FastMCP tool that delegates to an ML Intern ToolSpec handler."""

    _handler: Handler = PrivateAttr()
    _session: MCPSessionState = PrivateAttr()
    _pass_session: bool = PrivateAttr(default=False)

    def __init__(
        self,
        *,
        handler: Handler,
        session: MCPSessionState,
        pass_session: bool = False,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        self._handler = handler
        self._session = session
        self._pass_session = pass_session

    async def run(self, arguments: dict[str, Any]):
        self._session.refresh_env()
        try:
            if self._pass_session:
                output, success = await self._handler(arguments, session=self._session)
            else:
                output, success = await self._handler(arguments)
        except Exception as exc:
            raise McpError(
                ErrorData(code=-32000, message=f"{self.name} failed: {exc}")
            ) from exc

        output = _bounded_text(str(output))
        if not success:
            raise McpError(ErrorData(code=-32000, message=output))
        return self.convert_result(output)


def _tool_from_spec(
    spec: ToolSpec,
    *,
    session: MCPSessionState,
    name: str | None = None,
    description: str | None = None,
) -> HandlerTool:
    if spec.handler is None:
        raise ValueError(f"Cannot register MCP tool without handler: {spec.name}")

    pass_session = "session" in inspect.signature(spec.handler).parameters
    return HandlerTool(
        name=name or spec.name,
        description=description or spec.description,
        parameters=copy.deepcopy(spec.parameters),
        handler=spec.handler,
        session=session,
        pass_session=pass_session,
    )


def _static_find_hf_api_spec() -> dict[str, Any]:
    return {
        "name": "find_hf_api",
        "description": (
            "Find Hugging Face Hub REST API endpoints. Search by keyword and/or "
            "API category; returns endpoint details and curl examples."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword search such as 'upload file' or 'space logs'.",
                },
                "tag": {
                    "type": "string",
                    "description": "Optional API category/tag filter.",
                },
            },
            "required": [],
        },
    }


def _base_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name=EXPLORE_HF_DOCS_TOOL_SPEC["name"],
            description=EXPLORE_HF_DOCS_TOOL_SPEC["description"],
            parameters=EXPLORE_HF_DOCS_TOOL_SPEC["parameters"],
            handler=explore_hf_docs_handler,
        ),
        ToolSpec(
            name=HF_DOCS_FETCH_TOOL_SPEC["name"],
            description=HF_DOCS_FETCH_TOOL_SPEC["description"],
            parameters=HF_DOCS_FETCH_TOOL_SPEC["parameters"],
            handler=hf_docs_fetch_handler,
        ),
        ToolSpec(
            name=HF_PAPERS_TOOL_SPEC["name"],
            description=HF_PAPERS_TOOL_SPEC["description"],
            parameters=HF_PAPERS_TOOL_SPEC["parameters"],
            handler=hf_papers_handler,
        ),
        ToolSpec(
            name=HF_INSPECT_DATASET_TOOL_SPEC["name"],
            description=HF_INSPECT_DATASET_TOOL_SPEC["description"],
            parameters=HF_INSPECT_DATASET_TOOL_SPEC["parameters"],
            handler=hf_inspect_dataset_handler,
        ),
        ToolSpec(
            name=HF_REPO_FILES_TOOL_SPEC["name"],
            description=(
                HF_REPO_FILES_TOOL_SPEC["description"]
                + "\n\nMUTATING operations: upload and delete change remote Hugging Face repos."
            ),
            parameters=HF_REPO_FILES_TOOL_SPEC["parameters"],
            handler=hf_repo_files_handler,
        ),
        ToolSpec(
            name=HF_REPO_GIT_TOOL_SPEC["name"],
            description=(
                HF_REPO_GIT_TOOL_SPEC["description"]
                + "\n\nMUTATING operations can create/delete refs, create/update repos, and change PR state."
            ),
            parameters=HF_REPO_GIT_TOOL_SPEC["parameters"],
            handler=hf_repo_git_handler,
        ),
        ToolSpec(
            name=GITHUB_FIND_EXAMPLES_TOOL_SPEC["name"],
            description=GITHUB_FIND_EXAMPLES_TOOL_SPEC["description"],
            parameters=GITHUB_FIND_EXAMPLES_TOOL_SPEC["parameters"],
            handler=github_find_examples_handler,
        ),
        ToolSpec(
            name=GITHUB_LIST_REPOS_TOOL_SPEC["name"],
            description=GITHUB_LIST_REPOS_TOOL_SPEC["description"],
            parameters=GITHUB_LIST_REPOS_TOOL_SPEC["parameters"],
            handler=github_list_repos_handler,
        ),
        ToolSpec(
            name=GITHUB_READ_FILE_TOOL_SPEC["name"],
            description=GITHUB_READ_FILE_TOOL_SPEC["description"],
            parameters=GITHUB_READ_FILE_TOOL_SPEC["parameters"],
            handler=github_read_file_handler,
        ),
    ]


def _sandbox_tool_specs() -> list[ToolSpec]:
    specs = []
    for spec in get_sandbox_tools():
        if spec.name == "sandbox_create":
            specs.append(
                ToolSpec(
                    name=spec.name,
                    description=(
                        "Create a remote RunPod sandbox. Required before using "
                        "sandbox_bash, sandbox_read, sandbox_write, or sandbox_edit.\n\n"
                        + spec.description
                    ),
                    parameters=spec.parameters,
                    handler=spec.handler,
                )
            )
        else:
            specs.append(
                ToolSpec(
                    name=f"sandbox_{spec.name}",
                    description=f"REMOTE ONLY: {spec.description}",
                    parameters=spec.parameters,
                    handler=spec.handler,
                )
            )
    return specs


async def create_mcp_server(
    *,
    session: MCPSessionState | None = None,
    load_dynamic_api_schema: bool = True,
) -> FastMCP:
    """Create and populate the ML Intern MCP server."""

    state = session or MCPSessionState()
    mcp = FastMCP(
        "ML Intern",
        instructions=(
            "Specialist ML/Hugging Face tools and remote RunPod sandboxes. "
            "Use the host CLI for local shell and filesystem operations."
        ),
        version="0.1.0",
        mask_error_details=False,
        strict_input_validation=True,
    )

    for spec in _base_tool_specs():
        mcp.add_tool(_tool_from_spec(spec, session=state))

    api_spec = _static_find_hf_api_spec()
    if load_dynamic_api_schema:
        try:
            api_spec = await _get_api_search_tool_spec()
        except Exception:
            pass
    mcp.add_tool(
        _tool_from_spec(
            ToolSpec(
                name=api_spec["name"],
                description=api_spec["description"],
                parameters=api_spec["parameters"],
                handler=search_openapi_handler,
            ),
            session=state,
        )
    )

    for spec in _sandbox_tool_specs():
        mcp.add_tool(_tool_from_spec(spec, session=state))

    return mcp


async def _amain() -> None:
    mcp = await create_mcp_server()
    await mcp.run_stdio_async(show_banner=False)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
