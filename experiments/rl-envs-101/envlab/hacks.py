"""Step 2: rewards, verifiers, and three ways a reward is won without doing the task.

A verifier is the referee. A **programmatic** verifier is code (compare a number, run tests, query a database); an
**LLM judge** is a model asked to grade. Both turn an episode into a number, and RL will find the cheapest way to make
that number big. Each demo below has a reward that looks fine, a policy that wins it without doing the task, and a fix.

    1. won by refusing      "penalise wrong answers" makes "I don't know" the best policy
    2. won by editing tests the verifier runs the tests the agent can edit
    3. won by gaming format "is the answer somewhere in the reply?" is won by listing every number

Plus partial credit (why all-or-nothing gives GRPO nothing to learn from) and a stand-in judge with a length bias.
"""

from __future__ import annotations

import random
import re
import statistics

# --- 1. won by refusing ----------------------------------------------------------------------------------------------


def make_questions(n: int = 200, seed: int = 0) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    return [(f"What is {a} x {b}?", a * b) for a, b in ((rng.randint(12, 99), rng.randint(12, 99)) for _ in range(n))]


def guessing_policy(accuracy: float, seed: int = 0):
    """Answers every question; right with probability `accuracy` (a model that is often wrong)."""
    rng = random.Random(seed)
    return lambda q, a: f"Answer: {a if rng.random() < accuracy else a + rng.randint(1, 9)}"


def refusing_policy(q: str, a: int) -> str:
    return "I'm not sure, so I won't guess."


def parse_answer(reply: str) -> int | None:
    """Strict: the reply must end with 'Answer: <integer>' and hold no other 'Answer:'."""
    found = re.findall(r"Answer:\s*(-?\d+)\s*$", reply.strip())
    return int(found[0]) if len(found) == 1 and reply.count("Answer:") == 1 else None


def reward_penalise_wrong(reply: str, truth: int) -> float:
    """Looks safe ("never reward a wrong answer"): -1 wrong, 0 otherwise. Abstaining is never punished."""
    got = parse_answer(reply)
    return -1.0 if got is not None and got != truth else 0.0


def reward_fixed(reply: str, truth: int, wrong_penalty: float = 0.5) -> float:
    """+1 right, 0 abstain, -0.5 wrong: answering pays when the policy is right more than a third of the time."""
    got = parse_answer(reply)
    if got is None:
        return 0.0
    return 1.0 if got == truth else -wrong_penalty


# --- 2. won by editing the tests -------------------------------------------------------------------------------------

VISIBLE_TESTS = "def test_add():\n    assert add(2, 3) == 5\n"
HIDDEN_TESTS = "def test_add_more():\n    assert add(-4, 4) == 0\n    assert add(10, 5) == 15\n"
BUGGY_SOLUTION = "def add(a, b):\n    return a - b\n"
START_WORKSPACE = {"solution.py": BUGGY_SOLUTION, "tests/test_visible.py": VISIBLE_TESTS}


def run_tests(solution: str, tests: str) -> float:
    """Fraction of test functions that pass. Runs our own toy strings only (never untrusted code with exec)."""
    namespace: dict = {}
    exec(solution, namespace)  # our own toy strings, never untrusted input
    exec(tests, namespace)
    names = [n for n in namespace if n.startswith("test_")]
    passed = 0
    for name in names:
        try:
            namespace[name]()
            passed += 1
        except AssertionError:
            pass
    return passed / len(names) if names else 0.0  # "no tests ran" must not count as a pass


def honest_fix(workspace: dict) -> dict:
    return {**workspace, "solution.py": "def add(a, b):\n    return a + b\n"}


def test_editor(workspace: dict) -> dict:
    """The hack: leave the bug, make the visible test trivially true."""
    return {**workspace, "tests/test_visible.py": "def test_add():\n    assert True\n"}


def special_caser(workspace: dict) -> dict:
    """A subtler hack: pass the one visible input by name, stay wrong everywhere else."""
    return {**workspace, "solution.py": "def add(a, b):\n    return 5 if (a, b) == (2, 3) else a - b\n"}


def verifier_trusting(workspace: dict) -> dict:
    """Runs whatever tests are in the agent's workspace."""
    return {"reward": run_tests(workspace["solution.py"], workspace["tests/test_visible.py"])}


def verifier_sealed(workspace: dict) -> dict:
    """Runs its own copy of the visible tests, plus hidden tests, and flags tampering. Several keys, one reward."""
    return {
        "reward": run_tests(workspace["solution.py"], VISIBLE_TESTS),
        "hidden": run_tests(workspace["solution.py"], HIDDEN_TESTS),
        "tests_tampered": float(workspace["tests/test_visible.py"] != VISIBLE_TESTS),
    }


# --- 3. won by gaming the format -------------------------------------------------------------------------------------


def reward_contains(reply: str, truth: int) -> float:
    """Loose: is the right number anywhere in the reply?"""
    return float(str(truth) in re.findall(r"-?\d+", reply))


def spray_policy(q: str, a: int) -> str:
    """Lists every product in a plausible range: always 'contains' the answer, never commits to one."""
    return "It could be any of: " + ", ".join(str(n) for n in range(144, 9802))


def reward_strict(reply: str, truth: int) -> float:
    return float(parse_answer(reply) == truth)


# --- partial credit -------------------------------------------------------------------------------------------------


def group_advantages(rewards: list[float]) -> list[float]:
    """GRPO's advantage: reward minus the group mean, over the group's spread. A flat group teaches nothing."""
    mean = statistics.fmean(rewards)
    spread = statistics.pstdev(rewards)
    return [0.0 if spread == 0 else (r - mean) / spread for r in rewards]


# --- a stand-in judge -----------------------------------------------------------------------------------------------


def toy_judge(reply: str) -> float:
    """A deliberately biased stand-in for an LLM judge (it is not a model): it likes long, confident replies.

    Real judges have measured biases of this kind (length, position, self-preference). The fix is the same for a
    real judge: score it first on replies whose right grade is known, and look for what it rewards besides the task.
    """
    words = len(reply.split())
    confident = sum(reply.lower().count(w) for w in ("certainly", "clearly", "definitely"))
    return min(10.0, 3.0 + words / 15 + 2 * confident)
