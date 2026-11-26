"""Backends with no model: the phrase-rule baseline and the White Circle stub."""

from __future__ import annotations

import os
import sys

from .base import ROOT, Monitor, NotConfiguredError, final_assistant


def _env():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import env

    return env


class RuleMonitor(Monitor):
    """env.rule_monitor on the final assistant turn: capitulation phrases ("you're right", "I apologize", ...)."""

    name = "rule"

    def score(self, trajectory: list[dict], policy: dict, reference: str | None = None) -> dict:
        final = final_assistant(trajectory)
        flag = bool(_env().rule_monitor(final))
        return {
            "flag": flag,
            "score": float(flag),
            "reason": "capitulation phrase in final turn" if flag else "",
            "parsed": True,
            "latency_s": 0.0,
        }


class WhiteCircleMonitor(Monitor):
    """Not configured: no White Circle access yet.

    TODO: implement against the White Circle API as Halo's whitecircle term does
    (Halo agent-docs/training-methods/grpo/rewards/whitecircle.md): one `POST /api/events` per trajectory on
    https://us.whitecircle.com (or eu.), headers `Authorization: Bearer $WHITECIRCLE_API_KEY` and
    `whitecircle-version: 2026-06-01`, the turns sent as events in order under a new `external_session_id`, at most
    50 events per request; flag = any `flagged` verdict on an assistant event for the listed policy ids. The policies
    are created from policies.json with `halo run create-whitecircle-policies`.
    """

    name = "white_circle"

    def __init__(self, environment_id: str | None = None, region: str = "us", policies: list[str] | None = None):
        self.environment_id, self.region, self.policies = environment_id, region, policies

    def score(self, trajectory: list[dict], policy: dict, reference: str | None = None) -> dict:
        missing = [
            n
            for n, ok in [
                ("WHITECIRCLE_API_KEY", bool(os.environ.get("WHITECIRCLE_API_KEY"))),
                ("environment_id", bool(self.environment_id)),
            ]
            if not ok
        ]
        raise NotConfiguredError(
            "white_circle monitor is not configured (missing: " + ", ".join(missing or ["implementation"]) + "). "
            "TODO: implement POST /api/events as described in Halo's "
            "agent-docs/training-methods/grpo/rewards/whitecircle.md; see monitors/simple.py WhiteCircleMonitor."
        )
