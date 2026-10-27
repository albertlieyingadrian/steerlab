"""The HTTP side of `steerlab serve`: visitor routes, the admin router behind one access seam, and the event streams.

    GET  /                          the visitor chat           POST /api/chat        one message, its reply as SSE
    GET  /api/status?conversation=  availability + version     POST /api/new         start a new conversation
    GET  /api/conversation/<id>     the stored turns

    GET  /admin                     the admin desk             GET  /admin/events    live mirror (SSE, snapshot first)
    POST /admin/api/control         {name, strength}           POST /admin/api/all-off
    POST /admin/api/feature         {id}  add an SAE feature   POST /admin/api/feature/remove {id}
    POST /admin/api/prompt          {text | null}              POST /admin/api/model {key}
    POST /admin/api/stop | clear    POST /admin/api/redo {conversation}           GET  /admin/api/state

Every admin route depends on `require_admin`. Today it is a browser CSRF guard, not authentication: a POST
whose `Origin` is another site, or whose `Sec-Fetch-Site` says cross-site, is refused; a request with neither header
(curl, a script, the `/steerlab` skill) passes, which is what makes the admin API the agent surface.
Later it will check a session cookie. `TrustedHostMiddleware` refuses unknown `Host` headers (blocking DNS rebinding).

The worker runs on its own thread; its events reach this event loop through `loop.call_soon_threadsafe` (visitor
replies) or a thread-safe queue read with `asyncio.to_thread` (admin tabs). Model output is sent as JSON data and the
pages insert it with `textContent`, never as HTML.
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import secrets
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from steerlab.web.broker import Event
from steerlab.web.conversations import MessageRejected
from steerlab.web.worker import Worker

STATIC = Path(__file__).parent / "static"
PING_SECONDS = 15.0
CONVERSATION_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class ConversationBody(BaseModel):
    conversation: str = Field(pattern=CONVERSATION_ID.pattern)


class ChatBody(ConversationBody):
    text: str


class ControlBody(BaseModel):
    name: str = Field(min_length=1)
    strength: float


class FeatureBody(BaseModel):
    id: int = Field(ge=0)


class PromptBody(BaseModel):
    text: str | None = None


class ModelBody(BaseModel):
    key: str = Field(min_length=1)


def sse(event: Event) -> str:
    """One server-sent event: the event's `type` as the SSE event name, the whole event as JSON data."""
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def visitor_view(event: Event) -> Event:
    """What a visitor's stream carries: their own reply only, with no steering vocabulary."""
    kind = event["type"]
    if kind == "token":
        return {"type": "token", "index": event["index"], "text": event["text"]}
    if kind == "end":
        return {"type": "end", "reason": event["reason"]}
    if kind == "start":
        return {"type": "start"}
    return event  # queued {position}, error {message}


ADMIN_COOKIE = "steerlab_admin"


def presented_token(request: Request) -> str | None:
    """The admin token a request carries: the cookie the admin page sets, a `Bearer` header (scripts), or `?token=`."""
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.cookies.get(ADMIN_COOKIE) or request.query_params.get("token")


def require_admin(request: Request) -> None:
    """The one access seam: the admin token when the server has one, then the cross-site guard.

    `steerlab serve` gives the server a token when it listens beyond this machine (`--host`), so a visitor on the
    network who types /admin gets a 401. On loopback there is none unless `--admin-token` sets one.
    """
    token = request.app.state.admin_token
    if token is not None:
        presented = presented_token(request)
        if presented is None or not secrets.compare_digest(presented.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="admin token required: open the /admin link steerlab printed")
    if request.method in ("GET", "HEAD"):
        return
    origin = request.headers.get("origin")
    if origin is not None and origin not in request.app.state.allowed_origins:
        raise HTTPException(status_code=403, detail="cross-origin request refused")
    if request.headers.get("sec-fetch-site") in ("cross-site", "same-site"):
        raise HTTPException(status_code=403, detail="cross-site request refused")


def create_app(
    worker: Worker,
    *,
    hosts: Iterable[str] = LOOPBACK_HOSTS,
    port: int = 8765,
    admin_event_cap: int | None = None,
    admin_token: str | None = None,
) -> FastAPI:
    """The app around a (not yet started) worker. The lifespan starts it, and stops it with the admin streams.

    `admin_event_cap` ends an admin stream after that many events: only tests use it, so `TestClient` can read a
    stream to its end. `admin_token`, when set, is required on every admin route (`require_admin`).
    """
    allowed_hosts = list(hosts)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        worker.start()
        try:
            yield
        finally:
            worker.broker.close()
            await asyncio.to_thread(worker.stop)

    app = FastAPI(title="steerlab serve", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.admin_token = admin_token
    app.state.allowed_origins = {f"http://{host}:{port}" for host in allowed_hosts}
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def visitor_page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.post("/api/chat")
    async def chat(body: ChatBody) -> StreamingResponse:
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[Event] = asyncio.Queue()

        def sink(event: Event) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        try:
            job = worker.submit(body.conversation, body.text, sink)
        except MessageRejected as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

        async def stream() -> AsyncIterator[str]:
            finished = False
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(events.get(), PING_SECONDS)
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    yield sse(visitor_view(event))
                    if event["type"] == "end":
                        finished = True
                        return
            finally:
                if not finished:  # the visitor closed the tab or lost the connection
                    worker.disconnect(job)

        return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.post("/api/new")
    def new_conversation(body: ConversationBody) -> dict[str, bool]:
        worker.new_conversation(body.conversation)
        return {"ok": True}

    @app.get("/api/status")
    def status(conversation: str = "") -> dict[str, Any]:
        return worker.status(conversation)

    @app.get("/api/conversation/{conversation}")
    def transcript(conversation: str) -> dict[str, Any]:
        return {"messages": worker.store.transcript(conversation), "version": worker.store.version(conversation)}

    admin = APIRouter(dependencies=[Depends(require_admin)])

    @admin.get("/admin", include_in_schema=False)
    def admin_page(request: Request) -> FileResponse:
        page = FileResponse(STATIC / "admin.html")
        if admin_token is not None and request.query_params.get("token"):
            # Opened from the printed link: keep the token in a cookie so the live feed and the buttons carry it.
            page.set_cookie(ADMIN_COOKIE, admin_token, httponly=True, samesite="strict", path="/")
        return page

    @admin.get("/admin/events")
    async def admin_events() -> StreamingResponse:
        tab = worker.broker.subscribe()

        async def stream() -> AsyncIterator[str]:
            sent = 0
            try:
                while admin_event_cap is None or sent < admin_event_cap:
                    try:
                        event = await asyncio.to_thread(tab.get, PING_SECONDS)
                    except queue.Empty:
                        if tab.dropped:  # fell too far behind: end the stream so the browser reconnects
                            return
                        yield ": ping\n\n"
                        continue
                    if event is None:  # the server is stopping
                        return
                    yield sse(event)
                    sent += 1
            finally:
                worker.broker.unsubscribe(tab)

        return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    @admin.get("/admin/api/state")
    def admin_state() -> Event:
        return worker.state()

    def command(action: Callable[[], object]) -> JSONResponse:
        """Run one admin command; a rejection is a 422 with the reason, also shown to every admin tab."""
        try:
            action()
        except ValueError as error:
            worker.notice(str(error), "error")
            return JSONResponse({"detail": str(error)}, status_code=422)
        return JSONResponse({"ok": True})

    @admin.post("/admin/api/control")
    def control(body: ControlBody) -> JSONResponse:
        return command(lambda: worker.set_strength(body.name, body.strength))

    @admin.post("/admin/api/all-off")
    def all_off() -> JSONResponse:
        return command(worker.all_off)

    @admin.post("/admin/api/feature")
    def add_feature(body: FeatureBody) -> JSONResponse:
        return command(lambda: worker.add_feature(body.id))

    @admin.post("/admin/api/feature/remove")
    def remove_feature(body: FeatureBody) -> JSONResponse:
        return command(lambda: worker.remove_feature(body.id))

    @admin.post("/admin/api/prompt")
    def prompt(body: PromptBody) -> JSONResponse:
        return command(lambda: worker.set_prompt(body.text))

    @admin.post("/admin/api/model")
    def model(body: ModelBody) -> JSONResponse:
        return command(lambda: worker.swap(body.key))

    @admin.post("/admin/api/stop")
    def stop() -> dict[str, bool]:
        return {"stopped": worker.stop_reply()}

    @admin.post("/admin/api/redo")
    def redo(body: ConversationBody) -> JSONResponse:
        return command(lambda: worker.redo(body.conversation))

    @admin.post("/admin/api/clear")
    def clear() -> JSONResponse:
        return command(worker.clear)

    app.include_router(admin)
    return app
