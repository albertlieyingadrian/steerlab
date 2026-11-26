"""CPU tests for steps 1-3 and the rollout plumbing (seconds). Run: pytest experiments/rl-envs-101/tests"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envlab import hacks
from envlab.agents import HABITS, run_scripted, scripted_actions, spec_from_task
from envlab.desk import (
    DeskEnv,
    copy_db,
    earliest_free,
    list_slots,
    make_tasks,
    make_world,
    patient_row,
    slot_table,
    verify,
    world_from_json,
    world_to_json,
)
from envlab.guess import BinarySearchPolicy, GuessEnv, episode_return, rollout


def test_guess_binary_search_always_wins():
    env = GuessEnv()
    assert all(episode_return(rollout(env, BinarySearchPolicy(), seed=s)) == 1.0 for s in range(200))


def test_guess_malformed_and_finished():
    env = GuessEnv()
    env.reset(seed=0)
    obs, reward, done, _ = env.step("no number")
    assert reward == 0.0 and not done and "exactly one" in obs
    for _ in range(6):
        env.step("1000")
    with pytest.raises(RuntimeError):
        env.step("5")


def test_hacks_each_fix_closes_its_hole():
    q = hacks.make_questions(50)
    refuse = sum(hacks.reward_penalise_wrong(hacks.refusing_policy(a, b), b) for a, b in q)
    guess = hacks.guessing_policy(0.6)
    assert refuse > sum(hacks.reward_penalise_wrong(guess(a, b), b) for a, b in q)
    editor = hacks.test_editor(dict(hacks.START_WORKSPACE))
    assert hacks.verifier_trusting(editor)["reward"] == 1.0
    assert hacks.verifier_sealed(editor) == {"reward": 0.0, "hidden": 0.0, "tests_tampered": 1.0}
    assert hacks.reward_contains(hacks.spray_policy("", 1035), 1035) == 1.0
    assert hacks.reward_strict(hacks.spray_policy("", 1035), 1035) == 0.0


@pytest.fixture(scope="module")
def world():
    return make_world(seed=1)


def test_world_round_trip(world):
    again = world_from_json(json.loads(json.dumps(world_to_json(world))))
    assert slot_table(again) == slot_table(world)


def test_list_slots_is_not_time_sorted(world):
    lists = [list_slots(world, "Dr Kim", day)["free"] for day in ("Mon", "Tue", "Wed", "Thu", "Fri")]
    assert any([s["time"] for s in free] != sorted(s["time"] for s in free) for free in lists)


@pytest.mark.parametrize("habit", HABITS)
def test_habits_have_their_known_scores(world, habit):
    tasks = make_tasks(world, 30, seed=2)
    env = DeskEnv(world, max_turns=40)
    hidden = [run_scripted(env, t, habit)["scores"]["hidden"] for t in tasks]
    if habit == "careful":
        assert all(h == 1.0 for h in hidden)
    else:
        assert sum(hidden) < len(hidden)


def test_verifier_untouched_world_fails_book(world):
    task = next(t for t in make_tasks(world, 9, seed=3) if t.kind == "book")
    assert verify(world, copy_db(world), task) == {"reward": 0.0, "hidden": 0.0, "collateral": 1.0}
    assert earliest_free(world, task.doctor_id, task.day) is not None


def test_rollout_with_scripted_sampler(world):
    """The token bookkeeping: a fake sampler replays the careful habit through the real Qwen chat format."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    from envlab.rollout import run_episodes

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for task in make_tasks(world, 6, seed=4):
        spec = spec_from_task(task)
        spec["name"], spec["dob"] = patient_row(world, task.patient_id)
        policy = scripted_actions("careful", spec)
        env = DeskEnv(world, max_turns=8)
        holder = {"obs": None, "first": True}

        def sample(batch, policy=policy, holder=holder):
            action = next(policy) if holder["first"] else policy.send(holder["obs"])
            holder["first"] = False
            return [(action, tok(action, add_special_tokens=False)["input_ids"] + [im_end])]

        def step(text, original=env.step, holder=holder):
            result = original(text)
            holder["obs"] = result[0]
            return result

        env.step = step
        (episode,) = run_episodes(tok, sample, [(task, env)], max_turns=8)
        assert episode.info["hidden"] == 1.0, task.instruction
        assert len(episode.ids) == len(episode.mask)
        written = tok.decode([t for t, m in zip(episode.ids, episode.mask) if m])
        assert '"tool"' in written and "Tool result" not in written  # only the policy's tokens are in the loss
        assert sum(episode.mask) < len(episode.mask)
