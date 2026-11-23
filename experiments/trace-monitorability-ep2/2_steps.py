"""
Step 2: every tool call of every rollout -> one labelled row; draw the test steps (plan section 10.3-10.4).

    results/rollouts.csv   one row per rollout: task, kind, steps, how it ended, whether the task check passed
    results/steps.csv      one row per tool call: action type + target (classify_p2), leak flags on the thinking and
                           on the visible message, and whether it is a test step:
                             headline     6 per rollout, drawn at random (seed 20261123) from steps k >= 2 whose
                                          tool call parsed (all of them when fewer)
                             risky_extra  every other risky step k >= 2 (only in the risky breakdown)

No thinking text or commands are written to results/: targets are normalised paths or names.

    python 2_steps.py            (EP2_RUN=smoke2 to try it on a smoke run)
"""

import random
from collections import Counter

from common2 import RESULTS, RISKY, RUN, SEED, check, classify_p2, leak_flags, load_rollouts, norm_t, write_csv

TEST_PER_ROLLOUT = 6


def main() -> None:
    rollouts = load_rollouts()
    roll_rows, step_rows = [], []
    for r in rollouts:
        prev = "none"
        rows = []
        for s in r["steps"]:
            c = classify_p2(s["command"])
            tgt = norm_t(c["target"]) if s["command"] else ""
            lt = leak_flags(s["thinking"], s["command"], tgt)
            lv = leak_flags(s["visible"], s["command"], tgt)
            rows.append(
                {
                    "step_id": f"{r['rollout']}:{s['k']}",
                    "rollout": r["rollout"],
                    "task": r["task"],
                    "kind": r["kind"],
                    "rep": r["rep"],
                    "k": s["k"],
                    "call_ok": int(s["call_ok"]),
                    "call_read": s.get("call_read", ""),
                    "action_type": c["action_type"],
                    "target": tgt[:150],
                    "verb": c.get("verb", "")[:40],
                    "chained": int(bool(c.get("chained"))),
                    "rule": c.get("rule", ""),
                    "risky": int(c["action_type"] in RISKY),
                    "think_chars": len(s["thinking"]),
                    "visible_chars": len(s["visible"]),
                    "think_closed": int(s.get("think_closed", True)),
                    "leak_think": lt["leak"],
                    "leak_visible": lv["leak"],
                    "leak_any": "command"
                    if "command" in (lt["leak"], lv["leak"])
                    else ("target_only" if "target_only" in (lt["leak"], lv["leak"]) else "none"),
                    "rc": s["rc"],
                    "prev_type": prev,
                    "gen_tokens": s["gen_tokens"],
                    "test": "",
                }
            )
            prev = c["action_type"]
        eligible = [x for x in rows if x["k"] >= 2 and x["call_ok"]]
        pick = random.Random(f"{SEED}:{r['rollout']}").sample(eligible, min(TEST_PER_ROLLOUT, len(eligible)))
        for x in pick:
            x["test"] = "headline"
        for x in eligible:
            if not x["test"] and x["risky"]:
                x["test"] = "risky_extra"
        step_rows += rows
        roll_rows.append(
            {
                "rollout": r["rollout"],
                "task": r["task"],
                "kind": r["kind"],
                "rep": r["rep"],
                "steps": len(r["steps"]),
                "unparsed_calls": sum(not s["call_ok"] for s in r["steps"]),
                "end": r["end"],
                "check_pass": int(r["check_pass"]),
                "setup_ok": int(r["setup_ok"]),
                "risky_steps": sum(x["risky"] for x in rows),
            }
        )
    suffix = "" if RUN == "full" else f"_{RUN}"
    write_csv(RESULTS / f"rollouts{suffix}.csv", roll_rows)
    write_csv(RESULTS / f"steps{suffix}.csv", step_rows)

    n = len(step_rows)
    print(f"{len(roll_rows)} rollouts, {n} tool calls ({sum(not x['call_ok'] for x in step_rows)} unparsed)")
    print("ends:", Counter(x["end"] for x in roll_rows).most_common())
    print(f"task checks passed: {sum(x['check_pass'] for x in roll_rows)} / {len(roll_rows)}")
    print("types:", Counter(x["action_type"] for x in step_rows).most_common())
    print(f"risky steps: {sum(x['risky'] for x in step_rows)}")
    print("rules:", Counter(x["rule"] for x in step_rows).most_common())
    tests = [x for x in step_rows if x["test"] == "headline"]
    print(f"test steps: headline {len(tests)}, risky_extra {sum(x['test'] == 'risky_extra' for x in step_rows)}")
    print("leak in thinking (headline):", Counter(x["leak_think"] for x in tests).most_common())
    print("leak in visible  (headline):", Counter(x["leak_visible"] for x in tests).most_common())
    print(f"steps with a visible message: {sum(x['visible_chars'] > 0 for x in step_rows)} / {n}")
    print(f"steps with empty thinking: {sum(x['think_chars'] == 0 for x in step_rows)} / {n}")
    check("every rollout set up", all(x["setup_ok"] for x in roll_rows))


if __name__ == "__main__":
    main()
