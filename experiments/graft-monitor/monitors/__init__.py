"""Monitors for the `pushback` environment, one interface chosen by config.

    from monitors import get_monitor, load_policy
    m = get_monitor("oss_safeguard", reasoning_effort="medium")
    m.score(episode.transcript(first, final), load_policy(), reference=episode.truth)
    # -> {"flag": bool | None, "score": float, "reason": str, ...}

`flag` is None only when the backend could not parse a verdict (counted, never treated as safe).
control_arena and petri need the library venv (experiments/graft-monitor/.venv); the others need only `openai`.
"""

from __future__ import annotations

from .base import (
    POLICY_NAME,
    Monitor,
    MonitorError,
    NotConfiguredError,
    load_policies,
    load_policy,
    parse_verdict,
    policy_text,
    render_transcript,
)

BACKENDS = ("oss_safeguard", "qwen_judge", "control_arena", "petri", "rule", "white_circle")


def get_monitor(name: str, **cfg) -> Monitor:
    if name == "rule":
        from .simple import RuleMonitor

        return RuleMonitor(**cfg)
    if name == "white_circle":
        from .simple import WhiteCircleMonitor

        return WhiteCircleMonitor(**cfg)
    if name == "oss_safeguard":
        from .llm import OSSSafeguardMonitor

        return OSSSafeguardMonitor(**cfg)
    if name == "qwen_judge":
        from .llm import QwenJudgeMonitor

        return QwenJudgeMonitor(**cfg)
    if name == "control_arena":
        from .control_arena_backend import ControlArenaMonitor

        return ControlArenaMonitor(**cfg)
    if name == "petri":
        from .petri_backend import PetriMonitor

        return PetriMonitor(**cfg)
    raise ValueError(f"unknown monitor {name!r}; choose from {BACKENDS}")


__all__ = [
    "BACKENDS",
    "POLICY_NAME",
    "Monitor",
    "MonitorError",
    "NotConfiguredError",
    "get_monitor",
    "load_policies",
    "load_policy",
    "parse_verdict",
    "policy_text",
    "render_transcript",
]
