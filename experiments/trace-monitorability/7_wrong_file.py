"""
Step 7 (added after reading the M3 "never" items; exploratory, rule-based, no model calls): edits that land in a
different file from the one the reasoning names.

SWE-agent's `edit` replaces lines of the CURRENTLY OPEN file. Reading the M3 "never" items showed a recurring pattern:
the reasoning says "let's update reproduce.py" while the open file is a library module, so the edit overwrites library
code with test code. The rule below finds that pattern without a judge:

    wrong_file  the step is an edit; its reasoning names at least one file (a word ending .py/.md/.txt/.cfg/.toml/
                .rst/.json/.yaml/.yml/.ini); none of the named files has the open file's name.

    python 7_wrong_file.py   -> results/wrong_file.csv (episode, k, model, resolved, open-file name, flag, M3 label)
"""

import posixpath
import re
from collections import Counter

from common import RESULTS, boot_ci, fmt, load_raw, parse_episode, read_csv, write_csv

FILE = re.compile(r"[\w./-]+\.(?:py|md|txt|cfg|toml|rst|json|yaml|yml|ini)\b")

episodes = {e["episode"]: e for e in read_csv(RESULTS / "episodes.csv")}
m3 = {(u["episode"], u["k"]): u["where"] for u in read_csv(RESULTS / "unmentioned.csv")}
rows = []
for ep, e in episodes.items():
    for s in parse_episode(load_raw(int(e["row_idx"])))["steps"]:
        cmd = (s["command"] or "").strip()
        if not cmd.startswith("edit ") or not s["open_file"]:
            continue
        named = {posixpath.basename(f) for f in FILE.findall(s["reasoning"] or "")}
        open_name = posixpath.basename(s["open_file"])
        rows.append(
            {
                "episode": ep,
                "k": s["k"],
                "model_name": e["model_name"],
                "resolved": e["resolved"],
                "open_file_name": open_name,
                "names_a_file": int(bool(named)),
                "wrong_file": int(bool(named) and open_name not in named),
                "m3": m3.get((ep, str(s["k"])), ""),
            }
        )
write_csv(RESULTS / "wrong_file.csv", rows)
w = [r for r in rows if r["wrong_file"]]
print(
    f"edit steps: {len(rows)}; reasoning names a file: {sum(r['names_a_file'] for r in rows)}; "
    f"names only other files (wrong_file): {len(w)} in {len({r['episode'] for r in w})} episodes"
)
print("wrong_file rate over edits:", fmt(boot_ci([r["episode"] for r in rows], [r["wrong_file"] for r in rows])))
print("M3 label of wrong_file edits:", Counter(r["m3"] for r in w).most_common())
print("M3 label of other edits:", Counter(r["m3"] for r in rows if not r["wrong_file"]).most_common())
never = [r for r in rows if r["m3"] == "never"]
print(f"M3 'never' edits that are wrong_file: {sum(r['wrong_file'] for r in never)} / {len(never)}")
for mod in sorted({r["model_name"] for r in rows}):
    rs = [r for r in rows if r["model_name"] == mod]
    print(f"  {mod}: {fmt(boot_ci([r['episode'] for r in rs], [r['wrong_file'] for r in rs]))} (n={len(rs)})")
for res in ("0", "1"):
    rs = [r for r in rows if r["resolved"] == res]
    print(f"  resolved={res}: {fmt(boot_ci([r['episode'] for r in rs], [r['wrong_file'] for r in rs]))} (n={len(rs)})")
print("open file in wrong_file edits (top):", Counter(r["open_file_name"] for r in w).most_common(8))
