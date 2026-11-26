"""Step 3: the clinic front desk, a multi-turn tool-use environment with a state-checking verifier.

Prints one episode as a chat, then runs five scripted habits on 60 tasks and shows, for each, the visible reward,
the hidden check and the collateral damage (calendar rows changed that should not have been). Writes the table to
results/03_habits.json.

    python experiments/rl-envs-101/03_desk.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envlab.agents import HABITS, run_scripted
from envlab.desk import DeskEnv, make_tasks, make_world

world = make_world(seed=1)
tasks = make_tasks(world, 60, seed=2)
env = DeskEnv(world, max_turns=40)

print("One episode (careful habit) on:", tasks[1].instruction)
episode = run_scripted(env, tasks[1], "careful")
for m in episode["messages"]:
    print(f"  {m['role']:9s} {m['content'][:150]}")
print("  verifier:", episode["scores"])

print("\nFive habits on 60 tasks (20 book, 20 move, 20 cancel). Means:")
print(f"  {'habit':10s} {'reward (visible)':>17s} {'hidden':>7s} {'gap':>6s} {'collateral rows':>16s}")
table = {}
for habit in HABITS:
    scores = [run_scripted(env, t, habit)["scores"] for t in tasks]
    row = {k: sum(s[k] for s in scores) / len(scores) for k in ("reward", "hidden", "collateral")}
    row["gap"] = row["reward"] - row["hidden"]
    table[habit] = row
    print(f"  {habit:10s} {row['reward']:17.2f} {row['hidden']:7.2f} {row['gap']:6.2f} {row['collateral']:16.2f}")
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "03_habits.json").write_text(json.dumps(table, indent=1))

print()
ok = []


def check(cond: bool, text: str) -> None:
    ok.append(cond)
    print(("✅" if cond else "❌"), text)


check(table["careful"]["hidden"] == 1.0, "the careful habit passes every hidden check (the tasks are solvable)")
check(
    table["lazy"]["reward"] == 0.0,
    "saying 'done' without acting earns nothing (the verifier reads the calendar, not the text)",
)
check(
    table["sloppy"]["reward"] == 1.0 and table["sloppy"]["hidden"] < 0.8,
    "first-listed slot: full visible reward, hidden misses",
)
check(
    table["greedy"]["reward"] == 1.0 and table["greedy"]["collateral"] > 1,
    "booking every slot passes visible; collateral rows give it away",
)
check(table["confused"]["hidden"] < 1.0, "the shared-name patients trip the habit that ignores dates of birth")
print(f"{sum(ok)}/{len(ok)} checks passed")
