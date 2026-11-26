"""The desk environment as a Halo environment, for Halo's environmental (multi-turn) GRPO.

Same tools, same verifier, same JSON protocol as envlab/desk.py; only the wrapper changes. Halo asks for three
methods (Halo's agent-docs/training-methods/async-grpo/environments/custom.md):

    _reset_single(prompt, context)          build the trajectory; the task and its starting world arrive in context
    _step_single(trajectory, action, ...)   run one tool call, append its result as the next user turn
    _grade_episode(trajectory, context)     the verifier's key in [0, 1] (Halo prices it as reward/objective)

Per-episode state lives on the trajectory (one instance serves many episodes at once), so the calendar is kept in
trajectory.info as JSON and rebuilt for each tool call: 160 rows, which is cheap.

Dataset rows: {"prompt": <instruction>, "task": <Task json>, "world": <world json>}, with
`context_fields: [task, world]` in the run's YAML. `reward_key` picks the visible check ("reward") or the full one
("hidden"). Checked on CPU by tests/test_halo_env.py, which modal_halo_check.py runs in a Linux container (Halo needs
Triton, which has no macOS wheels).
"""

from __future__ import annotations

import json

from halo.environments.environment_base import EnvironmentBase
from halo.environments.episodes.episode_scoring import EpisodeGrade
from halo.environments.episodes.messages import Message, StepOutcome

from envlab.desk import SYSTEM_PROMPT, Task, call_tool, parse_action, verify, world_from_json, world_to_json


class DeskEnvironment(EnvironmentBase):
    DEFAULT_MAX_TURNS = 8
    LENGTH_CUTOFF_NUDGE = "Your reply was cut off. Reply with one short JSON object."
    EMPTY_TURN_NUDGE = "Reply with one JSON object, as in the system prompt."

    def __init__(self, reward_key: str = "reward", **kwargs):
        super().__init__(**kwargs)
        self.reward_key = reward_key

    def _reset_single(self, prompt, context=None):
        context = context or {}
        task, world = _loads(context["task"]), _loads(context["world"])
        return self._init_trajectory(
            prompt,
            context,
            system_prompt=SYSTEM_PROMPT,
            extra_info={"desk_task": task, "desk_start": world, "desk_now": world},
        )

    def _step_single(self, trajectory, action, context=None):
        parsed = parse_action(action)
        if parsed is not None and "done" in parsed:
            trajectory.info["completed"] = True
            return StepOutcome(trajectory=trajectory, reward=0.0, done=True, truncated=False, info={})
        if parsed is None:
            result = {"error": "reply with one JSON object, as in the system prompt"}
        else:
            db = world_from_json(trajectory.info["desk_now"])
            args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
            result = call_tool(db, str(parsed.get("tool")), args)
            trajectory.info["desk_now"] = world_to_json(db)
        trajectory.add_message(Message.user(f"Tool result: {json.dumps(result)}"))
        return StepOutcome(trajectory=trajectory, reward=0.0, done=False, truncated=False, info={})

    def scores(self, trajectory) -> dict:
        info = trajectory.info
        return verify(world_from_json(info["desk_start"]), world_from_json(info["desk_now"]), Task(**info["desk_task"]))

    def _grade_episode(self, trajectory, context=None):
        return EpisodeGrade(float(self.scores(trajectory)[self.reward_key]))

    def rollout_metrics(self, trajectory):
        metrics = super().rollout_metrics(trajectory)
        metrics.update({f"outcome/{k}": float(v) for k, v in self.scores(trajectory).items()})  # hidden: logged only
        return metrics


def _loads(value):
    return json.loads(value) if isinstance(value, str) else value
