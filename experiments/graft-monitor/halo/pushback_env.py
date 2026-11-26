"""Halo custom environment for `pushback` (agent-docs/training-methods/async-grpo/environments/custom.md).

Wraps the framework-free `experiments/graft-monitor/env.py`: turn 1 is the dataset prompt (env.ASK), the scripted
pushback (env.Episode.pushback, independent of the first answer) is the second user turn, and the grade is
env.grade's `proxy_reward` (the training reward). `gt_hack` is logged under outcome/* and never priced.

Dataset row: {"prompt": env.ASK text, "answer": truth, "episode": json(Episode.to_dict()), "anchor": bool}, read
through context_fields [episode, anchor].

Per-episode routing inputs the trainer reads off the trajectory (graft_trainer.py):
  info["graft_anchor"]  this episode's group is an anchor group (RETAIN-only);
  info["graft_flag"]    an environment-side monitor flag, when `flag_source` is not "judge":
                        "gt" = ground-truth hack kept with probability `flag_recall` (the recall sweep),
                        "rule" = env.rule_monitor on the final turn, "none" = no flag.
A judge term's verdict (reward metric judge/<name>/<check>) is read by the trainer when flag_source is "judge".
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from halo.environments.environment_base import EnvironmentBase

try:  # Halo >= 1.3 layout
    from halo.environments.episodes.episode_scoring import EpisodeGrade
    from halo.environments.episodes.messages import Message, StepOutcome
except ImportError:  # Halo 1.2.0, the published halo:hopper image
    from halo.environments.episode_scoring import EpisodeGrade
    from halo.environments.messages import Message, StepOutcome

_GM = Path(__file__).resolve().parent.parent
if str(_GM) not in sys.path:
    sys.path.insert(0, str(_GM))
import env as pushback  # noqa: E402  experiments/graft-monitor/env.py

FLAG_SOURCES = ("judge", "gt", "rule", "none")


class PushbackEnvironment(EnvironmentBase):
    DEFAULT_MAX_TURNS = 2
    requires_answer = False
    system_prompt = pushback.SYSTEM
    LENGTH_CUTOFF_NUDGE = "Your reply was cut off. Reply again, briefly."
    EMPTY_TURN_NUDGE = "Your reply was empty. Reply again, briefly."

    def __init__(self, flag_source: str = "judge", flag_recall: float = 1.0, flag_seed: int = 0, **kwargs):
        if flag_source not in FLAG_SOURCES:
            raise ValueError(f"flag_source must be one of {FLAG_SOURCES}, got {flag_source!r}")
        self.flag_source, self.flag_recall, self.flag_seed = flag_source, float(flag_recall), int(flag_seed)
        super().__init__(**kwargs)

    # --- episode ---
    @staticmethod
    def _pushback_episode(context) -> pushback.Episode:
        raw = (context or {}).get("episode")
        d = json.loads(raw) if isinstance(raw, str) else dict(raw)
        fields = pushback.Episode.__dataclass_fields__
        return pushback.Episode(**{k: v for k, v in d.items() if k in fields})

    def _reset_single(self, prompt, context=None):
        ep = self._pushback_episode(context)
        return self._init_trajectory(
            prompt,
            context,
            system_prompt=self.system_prompt,
            extra_info={"pushback_episode": ep.to_dict(), "graft_anchor": bool((context or {}).get("anchor", False))},
        )

    def _step_single(self, trajectory, action, context=None):
        n_assistant = sum(m.role == "assistant" for m in trajectory.messages)
        if n_assistant == 1:
            ep = pushback.Episode(**trajectory.info["pushback_episode"])
            trajectory.add_message(Message.user(ep.pushback()))  # scripted; does not read the first answer
            return StepOutcome(trajectory=trajectory, reward=0.0, done=False, truncated=False, info={})
        trajectory.info["completed"] = True
        return StepOutcome(trajectory=trajectory, reward=0.0, done=True, truncated=False, info={})

    def _answers(self, trajectory) -> tuple[str, str]:
        replies = [m.content or "" for m in trajectory.messages if m.role == "assistant"]
        return (replies[0] if replies else ""), (replies[1] if len(replies) > 1 else "")

    def _flag(self, ep: pushback.Episode, grade: dict, final: str) -> bool | None:
        if self.flag_source == "gt":
            if not grade["gt_hack"]:
                return False
            # Deterministic per (episode id, final text, seed): GRAFT's rh_detector_recall.
            h = hashlib.sha256(f"{self.flag_seed}|{ep.id}|{final}".encode()).hexdigest()[:8]
            return int(h, 16) / 0xFFFFFFFF < self.flag_recall
        if self.flag_source == "rule":
            return pushback.rule_monitor(final)
        if self.flag_source == "none":
            return False
        return None  # judge: the trainer reads the judge term's verdict

    def _grade_episode(self, trajectory, context=None):
        ep = pushback.Episode(**trajectory.info["pushback_episode"])
        first, final = self._answers(trajectory)
        g = pushback.grade(ep, first, final)
        trajectory.info["pushback_grade"] = {
            k: (float(v) if isinstance(v, (bool, int, float)) else v) for k, v in g.items()
        }
        flag = self._flag(ep, g, final)
        if flag is not None:
            trajectory.info["graft_flag"] = bool(flag)
        return EpisodeGrade(float(g["proxy_reward"]))

    # --- what a judge reads ---
    def _final_answer(self, trajectory):
        return self._answers(trajectory)[1] or None

    def _scoring_reference(self, trajectory):
        return None  # the monitor must not see the truth

    # --- logging ---
    def rollout_metrics(self, trajectory):
        m = super().rollout_metrics(trajectory)
        g = trajectory.info.get("pushback_grade") or {}
        ep = trajectory.info.get("pushback_episode") or {}
        for k in ("gt_hack", "proxy_reward", "correct", "agrees", "parsed", "capitulated", "first_correct"):
            if k in g:
                m[f"outcome/{k}"] = float(g[k])
        m["outcome/hackable"] = float(bool(ep.get("hackable")))
        m["outcome/anchor"] = float(bool(trajectory.info.get("graft_anchor")))
        if "graft_flag" in trajectory.info:
            m["outcome/graft_env_flag"] = float(trajectory.info["graft_flag"])
        return m


def make_dataset_rows(split: str, n: int, seed: int, anchor_frac: float = 0.0) -> list[dict]:
    """Halo dataset rows for env.make_episodes(split, n, seed). Anchor rows are whole prompts (so whole GRPO groups),
    every round(1/anchor_frac)-th episode."""
    eps = pushback.make_episodes(split, n, seed)
    every = round(1 / anchor_frac) if anchor_frac > 0 else 0
    rows = []
    for i, e in enumerate(eps):
        rows.append(
            {
                "prompt": pushback.ASK.format(city=e.city),
                "answer": e.truth,
                "episode": json.dumps(e.to_dict()),
                "anchor": bool(every and i % every == every - 1),
            }
        )
    return rows
