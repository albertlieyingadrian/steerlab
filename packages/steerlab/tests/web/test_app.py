"""The HTTP app on the tiny model: visitor SSE, the admin seam, host allowlist, validation, snapshots."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from steerlab.web.app import create_app, require_admin
from steerlab.web.worker import Worker

from .conftest import CATALOG, SYSTEM, RecordingBroker, tiny_load, wait_until

BASE = "http://127.0.0.1:8765"
ORIGIN = {"Origin": "http://localhost:8765"}
VISITOR = "visitor-0001-aaaa"


def new_worker() -> Worker:
    return Worker(
        catalog=lambda: list(CATALOG),
        load=tiny_load,
        unload=lambda backend, state: None,
        system_prompt=SYSTEM,
        broker=RecordingBroker(),
        initial_model="tiny",
        max_new_tokens=8,
    )


@pytest.fixture
def worker() -> Worker:
    return new_worker()


@pytest.fixture
def client(worker: Worker) -> Iterator[TestClient]:
    app = create_app(worker, admin_event_cap=1)
    with TestClient(app, base_url=BASE) as test_client:
        wait_until(lambda: worker.state()["status"] == "ready")
        yield test_client


def events_of(response: Any) -> list[dict[str, Any]]:
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def chat(client: TestClient, text: str = "Hello", conversation: str = VISITOR) -> list[dict[str, Any]]:
    with client.stream("POST", "/api/chat", json={"conversation": conversation, "text": text}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return events_of(response)


def test_a_visitor_message_streams_its_reply_and_the_stream_ends(client: TestClient) -> None:
    events = chat(client)
    kinds = [event["type"] for event in events]
    assert kinds[0] == "queued" and "start" in kinds and kinds[-1] == "end"
    assert "token" in kinds and events[-1]["reason"] in ("eos", "length")
    assert all(set(event) <= {"type", "index", "text", "reason", "position"} for event in events)


def test_an_admin_change_from_the_page_applies_to_the_next_reply(client: TestClient, worker: Worker) -> None:
    response = client.post("/admin/api/control", json={"name": "random", "strength": 2.0}, headers=ORIGIN)
    assert response.status_code == 200
    wait_until(lambda: worker.state()["directions"][0]["strength"] == 2.0)
    chat(client)
    markers = worker.broker.of("marker")  # type: ignore[attr-defined]
    assert markers and markers[0]["applied"][0]["strength"] == 2.0


@pytest.mark.parametrize(
    "headers",
    [{"Origin": "https://evil.example"}, {"Origin": "http://127.0.0.1:9999"}, {"Sec-Fetch-Site": "cross-site"}],
)
def test_cross_site_admin_posts_are_refused(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post("/admin/api/control", json={"name": "random", "strength": 1.0}, headers=headers)
    assert response.status_code == 403


def test_a_headerless_script_post_passes(client: TestClient) -> None:
    """Curl and the /steerlab skill drive the same API without browser headers."""
    assert client.post("/admin/api/all-off").status_code == 200


def test_unknown_hosts_are_refused_and_lan_hosts_allowed_when_listed(worker: Worker) -> None:
    with TestClient(create_app(worker), base_url="http://evil.example:8765") as client:
        assert client.get("/api/status").status_code == 400
    lan = create_app(new_worker(), hosts=["localhost", "127.0.0.1", "192.168.1.20"])
    with TestClient(lan, base_url="http://192.168.1.20:8765") as client:
        assert client.get("/api/status").status_code == 200


def test_the_access_seam_can_be_replaced(worker: Worker) -> None:
    """Overriding `require_admin` is how real authentication will plug in."""
    app = create_app(worker)
    app.dependency_overrides[require_admin] = lambda: None
    with TestClient(app, base_url=BASE) as client:
        response = client.post("/admin/api/all-off", headers={"Origin": "https://evil.example"})
        assert response.status_code == 200


def test_every_admin_stream_starts_with_a_snapshot(client: TestClient) -> None:
    for _ in range(2):
        with client.stream("GET", "/admin/events") as response:
            [event] = events_of(response)
        assert event["type"] == "snapshot" and event["state"]["status"] == "ready"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/admin/api/control", {"name": "korean", "strength": 1.0}),
        ("/admin/api/control", {"name": "random", "strength": "loud"}),
        ("/admin/api/model", {"key": ""}),
        ("/admin/api/model", {"key": "Qwen/Qwen3-8B"}),
        ("/admin/api/model", {"key": "/opt/models/some-model"}),
        ("/admin/api/feature", {"id": 99}),
        ("/admin/api/feature", {"id": -1}),
    ],
)
def test_invalid_admin_input_is_a_422_and_changes_nothing(
    client: TestClient, worker: Worker, path: str, body: dict[str, Any]
) -> None:
    before = worker.state()
    response = client.post(path, json=body)
    assert response.status_code == 422
    after = worker.state()
    assert after["model"] == before["model"] == "tiny"
    assert after["directions"] == before["directions"] and after["features"] == before["features"]


def test_status_and_transcript_follow_the_conversation(client: TestClient, worker: Worker) -> None:
    chat(client, "Hi")
    turns = client.get(f"/api/conversation/{VISITOR}").json()["messages"]
    assert [turn["role"] for turn in turns] == ["user", "assistant"] and turns[0]["content"] == "Hi"
    version = client.get("/api/status", params={"conversation": VISITOR}).json()["version"]
    assert client.post("/admin/api/clear").status_code == 200
    wait_until(lambda: client.get("/api/status", params={"conversation": VISITOR}).json()["version"] > version)
    assert client.get(f"/api/conversation/{VISITOR}").json()["messages"] == []
    assert client.get("/api/conversation/unknown-conversation").json()["messages"] == []


def test_a_rejected_visitor_message_is_a_422(client: TestClient) -> None:
    response = client.post("/api/chat", json={"conversation": VISITOR, "text": "   "})
    assert response.status_code == 422 and response.json()["detail"] == "Type a message first."
    assert client.post("/api/chat", json={"conversation": "bad id!", "text": "Hi"}).status_code == 422


def test_both_pages_are_served_and_load_nothing_from_outside(client: TestClient) -> None:
    for path in ("/", "/admin", "/static/styles.css", "/static/chat.js", "/static/admin.js"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "http://" not in response.text and "https://" not in response.text, path


def test_the_visitor_page_says_nothing_about_steering_or_the_admin(client: TestClient) -> None:
    """A visitor cannot tell from the page or its script that anything is steered."""
    text = client.get("/").text + client.get("/static/chat.js").text
    assert "steer" not in text.lower() and "/admin" not in text


def test_an_admin_tab_that_falls_behind_is_ended_so_it_can_reconnect(
    worker: Worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    from steerlab.web import app as app_module
    from steerlab.web import broker as broker_module

    monkeypatch.setattr(broker_module, "TAB_QUEUE_SIZE", 3)
    monkeypatch.setattr(app_module, "PING_SECONDS", 0.05)
    with TestClient(create_app(worker), base_url=BASE) as client:
        wait_until(lambda: worker.state()["status"] == "ready")
        with client.stream("GET", "/admin/events") as response:
            for index in range(10):
                worker.notice(f"burst {index}")
            events = events_of(response)  # returns only because the server ended the stream
    assert events[0]["type"] == "snapshot" and len(events) < 11  # ended before all ten bursts were sent


def test_an_ipv6_loopback_host_is_allowed(worker: Worker) -> None:
    with TestClient(create_app(worker), base_url="http://[::1]:8765") as client:
        assert client.get("/api/status").status_code == 200


def test_with_a_token_every_admin_route_needs_it_and_the_printed_link_sets_a_cookie(worker: Worker) -> None:
    """On the network, a visitor who types /admin gets a 401; the presenter's link opens the desk."""
    app = create_app(worker, admin_token="s3cret-token", admin_event_cap=1)
    with TestClient(app, base_url=BASE) as client:
        wait_until(lambda: worker.state()["status"] == "ready")
        assert client.get("/admin").status_code == 401
        assert client.post("/admin/api/all-off").status_code == 401
        assert client.post("/admin/api/all-off", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.post("/admin/api/all-off", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200
        assert client.get("/api/status").status_code == 200  # visitors never need it
        page = client.get("/admin", params={"token": "s3cret-token"})
        assert page.status_code == 200 and "steerlab_admin" in page.headers["set-cookie"]
        with client.stream("GET", "/admin/events") as response:  # the cookie now rides along, like EventSource
            [event] = events_of(response)
        assert event["type"] == "snapshot"


def test_the_visitor_page_works_on_plain_http_lan_addresses(client: TestClient) -> None:
    """crypto.randomUUID is undefined outside secure contexts, so the LAN visitor page must not depend on it."""
    assert "randomUUID(" not in client.get("/static/chat.js").text  # the comment may name it; no call
