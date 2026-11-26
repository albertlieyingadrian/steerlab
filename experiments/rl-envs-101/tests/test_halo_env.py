"""The desk environment inside Halo's own environment machinery (CPU, Linux only: Halo needs Triton).

Run by modal_halo_check.py in a Linux container; skipped anywhere Halo is not installed.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
halo = pytest.importorskip("halo.environments.registry")
from halo.configs.grpo_environmental import EnvironmentConfig
from halo.environments.registry import _environment_registry, build_environment, register_environment

from envlab.desk import make_tasks, make_world, world_to_json
from envlab.halo_env import DeskEnvironment
from envlab.pipeline import oracle_calls


@pytest.fixture
def desk_env():
    register_environment("desk", lambda c: DeskEnvironment(**c), override=True)
    yield build_environment("desk", EnvironmentConfig(environment_type="desk").to_env_config())
    _environment_registry.pop("desk", None)


def _episode(env, task, world, actions):
    ids, _ = env.reset([task.instruction], [{"task": task.to_json(), "world": world_to_json(world)}])
    steps = []
    for action in actions:
        steps = env.step(ids, [action], [{}])
        if steps[0].done:
            break
    return steps[0]


@pytest.mark.parametrize("kind", ["book", "reschedule", "cancel"])
def test_oracle_earns_one_and_doing_nothing_earns_zero(desk_env, kind):
    world = make_world(seed=1)
    task = next(t for t in make_tasks(world, 12, seed=5) if t.kind == kind)
    calls = [json.dumps({"tool": c["tool"], "args": c["args"]}) for c in oracle_calls(task, world)]
    done = json.dumps({"done": "All set."})
    oracle = _episode(desk_env, task, world, [*calls, done])
    assert oracle.done and oracle.trajectory.total_reward == 1.0
    assert desk_env.rollout_metrics(oracle.trajectory)["outcome/hidden"] == 1.0
    nop = _episode(desk_env, task, world, [done])
    assert nop.done and nop.trajectory.total_reward == 0.0
