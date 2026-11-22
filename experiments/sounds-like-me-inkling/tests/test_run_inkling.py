"""The spend cap fails closed: no network, the endpoint is faked."""

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_inkling as R


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "LEDGER", tmp_path / "spend.jsonl")
    monkeypatch.setattr(R, "_reserved", [0.0])
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    return tmp_path / "spend.jsonl"


def fake(replies, monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(1)
        r = replies[min(len(calls), len(replies)) - 1]
        if isinstance(r, Exception):
            raise r
        return io.BytesIO(json.dumps(r).encode())

    monkeypatch.setattr(R.urllib.request, "urlopen", urlopen)
    return calls


def test_billed_cost_is_charged(monkeypatch):
    fake([{"usage": {"cost": 0.0002}, "choices": []}], monkeypatch)
    assert "error" not in R.post({}, "k", 0.01, "n")
    assert R.spent() == pytest.approx(0.0002)


def test_timeout_and_missing_cost_charge_the_reservation(monkeypatch):
    calls = fake([TimeoutError("slow"), {"choices": []}], monkeypatch)
    R.post({}, "k", 0.01, "n")
    assert len(calls) == 2
    assert R.spent() == pytest.approx(0.02)  # both attempts charged in full


def test_client_error_is_not_retried_or_charged(monkeypatch):
    err = urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"bad request"))
    calls = fake([err], monkeypatch)
    assert R.post({}, "k", 0.01, "n")["error"].startswith("400")
    assert len(calls) == 1 and R.spent() == 0.0


def test_cap_refuses_before_sending(monkeypatch):
    monkeypatch.setattr(R, "CAP", 0.015)
    calls = fake([TimeoutError("slow")], monkeypatch)
    with pytest.raises(R.CapReached):
        R.post({}, "k", 0.01, "n")
    assert len(calls) == 1 and R.spent() == pytest.approx(0.01)
