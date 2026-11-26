"""Step 1: run the 50-line guess-the-number environment with two scripted policies.

Prints one trajectory step by step, then the average return of a binary-search policy and a random one over 1,000
episodes each, then self-checks.

    python experiments/rl-envs-101/01_guess.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envlab.guess import BinarySearchPolicy, GuessEnv, RandomPolicy, episode_return, rollout

env = GuessEnv()
print("One episode (seed 7) with binary search, step by step:")
trajectory = rollout(env, BinarySearchPolicy(), seed=7)
for i, step in enumerate(trajectory):
    print(f"  step {i}: obs={step['observation'][:40]!r:44} action={step['action']:>3}  reward={step['reward']}")
print(f"  return = {episode_return(trajectory)}, length = {len(trajectory)}")
print("  the last step as data:", json.dumps(trajectory[-1]))

n = 1000
rows = {}
for name, make in [("binary search", BinarySearchPolicy), ("random", lambda: RandomPolicy(seed=1))]:
    returns, lengths = [], []
    for seed in range(n):
        traj = rollout(env, make(), seed=seed)
        returns.append(episode_return(traj))
        lengths.append(len(traj))
    rows[name] = (sum(returns) / n, sum(lengths) / n)
    print(f"{name:14s} mean return {rows[name][0]:.3f}  mean length {rows[name][1]:.2f} over {n} episodes")

print()
print(
    ("✅" if rows["binary search"][0] == 1.0 else "❌"), "binary search always wins within 7 guesses (2^7 = 128 > 100)"
)
print(("✅" if rows["random"][0] < 0.1 else "❌"), "random guessing rarely wins (about 1 - 0.99^7 = 0.068 expected)")
try:
    env.step("50")
    print("❌ stepping a finished episode should raise")
except RuntimeError:
    print("✅ stepping a finished episode raises: an episode has a clear end")
env.reset(seed=3)
obs, reward, done, _ = env.step("12 or 13")
print(("✅" if reward == 0 and "exactly one number" in obs else "❌"), "a malformed action earns 0 and is told why")
