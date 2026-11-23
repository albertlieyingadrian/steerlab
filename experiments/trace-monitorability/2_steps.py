"""
Step 2: parse every trajectory into steps, label every action, and pick the test steps.

For every agent step: reasoning (text outside the last ``` fence), command (inside it), action type + target
(common.classify_action), and whether a write/delete lands outside the repository. Every step goes to
results/actions.csv (used by M3). Up to TEST_STEPS_PER_EPISODE steps per episode (k >= 2, parsed, not submit) are drawn
as test steps for M1 and M2 and go to results/steps.csv, with their leak flags.
"""

import posixpath
from collections import Counter

from common import (
    RESULTS,
    TEST_STEPS_PER_EPISODE,
    check,
    classify_action,
    leak_flags,
    load_raw,
    norm_target,
    parse_episode,
    read_csv,
    rng,
    write_csv,
)

episodes = read_csv(RESULTS / "episodes.csv")
r = rng(1)
actions, tests = [], []
n_turns = n_parsed = 0
tools, unknown = Counter(), Counter()
for ep in episodes:
    parsed = parse_episode(load_raw(int(ep["row_idx"])))
    root = parsed["repo_root"]
    prev_type = "none"
    ep_rows = []
    for s in parsed["steps"]:
        n_turns += 1
        a = classify_action(s["command"], s["open_file"], s["cwd"])
        if a["action_type"] != "unparsed":
            n_parsed += 1
            tools[a["tool"]] += 1
            if a["unknown"]:
                unknown[a["tool"]] += 1
        tgt = norm_target(a["target"], s["cwd"])
        absolute = (
            tgt
            if tgt.startswith("/")
            else (
                posixpath.normpath(posixpath.join(s["cwd"], tgt))
                if s["cwd"] and tgt not in ("none", "inline", ".")
                else tgt
            )
        )
        outside = (
            a["action_type"] in ("local_write", "delete", "permission_change")
            and root is not None
            and absolute.startswith("/")
            and not (absolute + "/").startswith(root.rstrip("/") + "/")
        )
        row = {
            "episode": ep["episode"],
            "model_name": ep["model_name"],
            "resolved": ep["resolved"],
            "k": s["k"],
            "n_steps": len(parsed["steps"]),
            "tool": a["tool"],
            "action_type": a["action_type"],
            "target": tgt,
            "chained": int(a["chained"]),
            "unknown_cmd": int(a["unknown"]),
            "outside_repo": int(outside),
            "prev_type": prev_type,
            "reasoning_chars": len(s["reasoning"]),
        }
        row.update(
            {
                k: (int(v) if isinstance(v, bool) else v)
                for k, v in leak_flags(s["reasoning"], s["command"], tgt).items()
            }
        )
        ep_rows.append(row)
        if a["action_type"] != "unparsed":
            prev_type = a["action_type"]
    actions += ep_rows
    eligible = [x for x in ep_rows if x["k"] >= 2 and x["action_type"] not in ("unparsed", "submit")]
    for x in sorted(r.sample(eligible, min(TEST_STEPS_PER_EPISODE, len(eligible))), key=lambda x: x["k"]):
        tests.append({**x, "step_id": f"{x['episode']}:k{x['k']}"})

fields = [
    "episode",
    "model_name",
    "resolved",
    "k",
    "n_steps",
    "tool",
    "action_type",
    "target",
    "chained",
    "unknown_cmd",
    "outside_repo",
    "prev_type",
    "reasoning_chars",
    "command_leak",
    "target_leak",
    "leak",
]
write_csv(RESULTS / "actions.csv", actions, fields)
write_csv(RESULTS / "steps.csv", tests, ["step_id", *fields])

print(f"agent turns {n_turns}; parsed into reasoning + one command: {n_parsed} ({n_parsed / n_turns:.1%})")
print("tools (top 25):", tools.most_common(25))
print("unknown shell commands (executed-program fallback):", unknown.most_common(15))
print("action types, all steps:", Counter(a["action_type"] for a in actions).most_common())
print("action types, test steps:", Counter(t["action_type"] for t in tests).most_common())
print("leak, test steps:", Counter(t["leak"] for t in tests).most_common())
print(
    "empty reasoning, all parsed steps:",
    sum(a["reasoning_chars"] == 0 for a in actions if a["action_type"] != "unparsed"),
)
print("writes/deletes outside the repo:", sum(a["outside_repo"] for a in actions))
print(f"test steps: {len(tests)} from {len({t['episode'] for t in tests})} episodes")
check("400-600 test steps", 400 <= len(tests) <= 600)
check("parse rate above 90%", n_parsed / n_turns > 0.9)
