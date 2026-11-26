"""Step 2: three rewards that are won without doing the task, each with its fix, plus partial credit and a judge.

python experiments/rl-envs-101/02_rewards.py
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envlab import hacks

checks = []


def check(ok: bool, text: str) -> None:
    checks.append(ok)
    print(("✅" if ok else "❌"), text)


def mean_reward(reward, replies) -> float:
    return statistics.fmean(reward(r, a) for r, a in replies)


questions = hacks.make_questions(200)
policies = {"answers, right 60%": hacks.guessing_policy(0.6), "always refuses": hacks.refusing_policy}

print("1. Won by refusing: mean reward over 200 multiplication questions")
table = {}
for name, policy in policies.items():
    replies = [(policy(q, a), a) for q, a in questions]
    table[name] = (mean_reward(hacks.reward_penalise_wrong, replies), mean_reward(hacks.reward_fixed, replies))
    print(f"   {name:20s} penalise-wrong {table[name][0]:+.3f}   fixed (+1 / 0 / -0.5) {table[name][1]:+.3f}")
check(
    table["always refuses"][0] > table["answers, right 60%"][0],
    "under 'penalise wrong', refusing beats a 60%-right policy",
)
check(table["answers, right 60%"][1] > table["always refuses"][1], "under the fixed reward, answering wins again")

print("\n2. Won by editing the tests: one buggy add(a, b), three agents, two verifiers")
agents = {
    "honest fix": hacks.honest_fix,
    "edits the test": hacks.test_editor,
    "special-cases (2, 3)": hacks.special_caser,
}
results = {}
for name, agent in agents.items():
    ws = agent(dict(hacks.START_WORKSPACE))
    results[name] = (hacks.verifier_trusting(ws), hacks.verifier_sealed(ws))
    trusting, sealed = results[name]
    print(
        f"   {name:22s} trusting reward {trusting['reward']:.1f} | sealed reward {sealed['reward']:.1f}  "
        f"hidden {sealed['hidden']:.1f}  tampered {sealed['tests_tampered']:.0f}"
    )
check(results["edits the test"][0]["reward"] == 1.0, "the trusting verifier pays the test editor in full")
check(results["edits the test"][1]["reward"] == 0.0, "the sealed copy of the visible test catches the edit")
special = results["special-cases (2, 3)"][1]
check(
    special["reward"] == 1.0 and special["hidden"] == 0.0,
    "special-casing passes the sealed visible test; only hidden tests catch it",
)
check(hacks.run_tests(hacks.BUGGY_SOLUTION, "x = 1\n") == 0.0, "a test file with no tests scores 0, not 1")

print("\n3. Won by gaming the format: 'is the answer anywhere in the reply?'")
spray = [(hacks.spray_policy(q, a), a) for q, a in questions]
honest_policy = hacks.guessing_policy(0.6, seed=2)
honest = [(honest_policy(q, a), a) for q, a in questions]
loose = (mean_reward(hacks.reward_contains, spray), mean_reward(hacks.reward_contains, honest))
strict = (mean_reward(hacks.reward_strict, spray), mean_reward(hacks.reward_strict, honest))
print(f"   loose 'contains':  spray {loose[0]:.3f}   honest 60% {loose[1]:.3f}")
print(f"   strict last line:  spray {strict[0]:.3f}   honest 60% {strict[1]:.3f}")
print(f"   one spray reply is {len(spray[0][0].split())} words long")
check(loose[0] == 1.0 and loose[0] > loose[1], "the loose check pays the spray policy more than the honest one")
check(strict[0] == 0.0, "the strict parser pays the spray policy nothing")
check(hacks.parse_answer("Answer: 4\nAnswer: 5") is None, "two 'Answer:' lines are refused, not read as the last one")

print("\nPartial credit: a group of 8 attempts at a 4-step task, GRPO advantages")
steps_done = [3, 2, 3, 1, 0, 2, 3, 2]  # how many of 4 checks each attempt passed; none passed all 4
all_or_nothing = [float(s == 4) for s in steps_done]
partial = [s / 4 for s in steps_done]
adv_binary, adv_partial = hacks.group_advantages(all_or_nothing), hacks.group_advantages(partial)
print("   all-or-nothing rewards", all_or_nothing, "-> advantages", [round(a, 2) for a in adv_binary])
print("   partial-credit rewards", partial, "-> advantages", [round(a, 2) for a in adv_partial])
check(
    all(a == 0 for a in adv_binary),
    "all-or-nothing: nobody passed, so every advantage is 0 and the step learns nothing",
)
check(max(adv_partial) > 0, "partial credit: the 3-of-4 attempts get pushed up")
print("   the catch: partial credit pays for the easy steps, so a policy can farm 3/4 forever. Keep the full-task")
print("   pass rate as a separate number, and give the last step most of the weight if it is the point of the task.")

print("\nA stand-in judge (a length-and-confidence heuristic, not a model): known-answer check first")
right = "Answer: 1035"
wrong = "Certainly, clearly and definitely, after careful thought, " * 3 + "Answer: 1036"
print(f"   right short answer {hacks.toy_judge(right):.1f}/10   wrong padded answer {hacks.toy_judge(wrong):.1f}/10")
check(
    hacks.toy_judge(wrong) > hacks.toy_judge(right),
    "the stand-in judge prefers the padded wrong answer: it fails the known-answer check",
)
print(f"\n{sum(checks)}/{len(checks)} checks passed")
