"""ourdriver as an AgentEnv computer driver on macOS.

Adapted from AgentEnv's Aside driver (agentenv-framework, src/agent_env/drivers/envs/aside/driver.py, Apache-2.0):
the agent's tools are ourdriver's own MCP tools, served by ``bin/ourdriver`` over stdio, one server per agent session
(each keeps its snapshots between calls). The computer/v1 operations (screenshots, recording, the probe) are
MacOSDriver's. ``mcp.json`` pins the instructions and tools the binary lists; ``pin.json`` its sha256.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agentenv_protocol import ForwardedTool, ToolContext, server_instructions
from agentenv_protocol.computer import ComputerSettings, MacOSDriver, ReadyCheck, computer_driver
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent

HERE = Path(__file__).resolve().parent
BINARY = HERE / "bin" / "ourdriver"
MANIFEST = HERE / "mcp.json"
PIN = HERE / "pin.json"
#: Every tool acts on apps, and through them on anything.
ANNOTATIONS = {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True}
#: How long an unused session's server is kept, and how many are kept at most.
IDLE_S, MAX_SERVERS = 1800, 8
#: What of the driver's environment the binary gets.
KEPT = ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG")


@computer_driver(name="ourdriver", supports=[{"os": {"type": "macos", "display": ["aqua"]}, "arch": ["arm64"]}])
class OurDriver(MacOSDriver):
    class Settings(ComputerSettings):
        pass

    def __init__(self) -> None:
        super().__init__()
        self._servers: dict[str | None, _Server] = {}

    def forwarded_tools(self) -> list[ForwardedTool]:
        tools = [{**t, "annotations": {**(t.get("annotations") or {}), **ANNOTATIONS}} for t in _manifest()["tools"]]
        if hasattr(ForwardedTool, "listed"):  # AgentEnv 0.9.1514
            return [ForwardedTool.listed(t, functools.partial(self._forward, t["name"])) for t in tools]
        # AgentEnv 0.9.1506 (the final run's pin) builds the descriptor itself, as its cua-driver env does.
        from agentenv_protocol.types import EnvironmentTool

        return [
            ForwardedTool(
                EnvironmentTool(name=t["name"], description=t.get("description"), inputSchema=t["inputSchema"]),
                functools.partial(self._forward, t["name"]),
                t["annotations"],
                t.get("outputSchema"),
            )
            for t in tools
        ]

    @server_instructions
    def ourdriver_instructions(self) -> str:
        return _manifest()["instructions"]

    async def ready_checks(self) -> list[ReadyCheck]:
        pinned = json.loads(PIN.read_text())["sha256"]
        actual = hashlib.sha256(BINARY.read_bytes()).hexdigest() if BINARY.is_file() else None
        checks = [
            ReadyCheck(
                "ourdriver binary",
                actual == pinned,
                "as pinned" if actual == pinned else f"sha256 {actual} is not the pinned {pinned}",
            )
        ]
        if actual == pinned:
            process = await asyncio.create_subprocess_exec(
                str(BINARY),
                "check",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=_environment(),
            )
            out, _ = await asyncio.wait_for(process.communicate(), 30)
            said = out.decode(errors="replace").splitlines()[0] if out else ""
            ok = "inside VM: true" in said and "accessibility trusted: true" in said
            checks.append(ReadyCheck("ourdriver permissions", ok, said or f"exit {process.returncode}"))
        return [*checks, *await super().ready_checks()]

    async def _forward(self, tool: str, /, **arguments: Any) -> CallToolResult:
        caller = ToolContext.current().caller.session
        server = self._servers.pop(caller, None) or _Server()
        self._evict()
        self._servers[caller] = server
        try:
            return await server.call(tool, arguments)
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=(f"ourdriver failed before it answered, so {tool} may have run: {type(e).__name__}: {e}"),
                    )
                ],
            )

    def _evict(self) -> None:
        now = time.monotonic()
        for caller in [c for c, s in self._servers.items() if not s.busy and now - s.used > IDLE_S]:
            asyncio.create_task(self._servers.pop(caller).close())
        while len(self._servers) >= MAX_SERVERS:
            caller = next(iter(self._servers))
            asyncio.create_task(self._servers.pop(caller).close())


class _Server:
    """One ourdriver process for a session, held open by a task of its own, its calls one at a time."""

    def __init__(self) -> None:
        self._calls: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self.used, self.busy = time.monotonic(), 0

    async def call(self, tool: str, arguments: dict) -> CallToolResult:
        if self._calls is None or self._task is None or self._task.done():
            self._calls = asyncio.Queue()
            self._task = asyncio.create_task(self._serve(self._calls))
        answer = asyncio.get_running_loop().create_future()
        self.busy += 1
        try:
            self._calls.put_nowait((tool, arguments, answer))
            return await answer
        finally:
            self.busy -= 1
            self.used = time.monotonic()

    async def close(self) -> None:
        if self._calls is not None:
            self._calls.put_nowait(None)
        if self._task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, 10)

    async def _serve(self, calls: asyncio.Queue) -> None:
        try:
            async with _session() as session:
                while (call := await calls.get()) is not None:
                    tool, arguments, answer = call
                    if answer.done():
                        continue
                    try:
                        answer.set_result(await session.call_tool(tool, arguments))
                    except Exception as e:
                        if not answer.done():
                            answer.set_exception(e)
                        raise
        except Exception as e:
            while not calls.empty():
                if (call := calls.get_nowait()) is not None and not call[-1].done():
                    call[-1].set_exception(e)


@contextlib.asynccontextmanager
async def _session() -> AsyncIterator[ClientSession]:
    params = StdioServerParameters(command=str(BINARY), args=[], env=_environment())
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


@functools.cache
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _environment() -> dict[str, str]:
    return {**{k: os.environ[k] for k in KEPT if k in os.environ}, "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
