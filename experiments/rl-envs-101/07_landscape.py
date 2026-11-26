"""Step 07: the RL environment tool landscape, printed from data/tool_landscape.json.

Every row was checked against the tool's own README or docs on the date in its `checked` field. The script prints the
table grouped by kind, then self-checks that every row is complete and its links look right.

    .venv/bin/python experiments/rl-envs-101/07_landscape.py
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "tool_landscape.json"
KINDS = ["task format & runner", "workspace", "env library/hub", "trainer", "eval framework", "env suite"]
REQUIRED = ["name", "kind", "what_it_is", "use_it_when", "repo", "docs", "license", "checked", "source_note"]


def check(ok: bool, text: str) -> bool:
    print(("✅ " if ok else "❌ ") + text)
    return ok


def main() -> None:
    rows = json.loads(DATA.read_text())
    print(f"{len(rows)} tools, checked {sorted({r['checked'] for r in rows})}\n")
    for kind in KINDS:
        group = [r for r in rows if r["kind"] == kind]
        if not group:
            continue
        print(f"## {kind}\n")
        print("| Tool | Use it when | Repo | Stars | Last push |")
        print("|---|---|---|---:|---|")
        for r in group:
            stars = "-" if r.get("stars") is None else f"{r['stars']:,}"
            print(f"| {r['name']} | {r['use_it_when']} | {r['repo']} | {stars} | {r.get('last_push') or '-'} |")
        print()

    print("Self-checks")
    results = [
        check(
            all(all(r.get(k) for k in REQUIRED) for r in rows),
            "every row has name, kind, what, use-when, repo, docs, license, source",
        ),
        check(all(r["kind"] in KINDS for r in rows), "every kind is one of the six known kinds"),
        check(
            all(r["repo"].startswith("https://") and r["docs"].startswith("https://") for r in rows),
            "every repo and docs link is https",
        ),
        check(
            sum(r["repo"].startswith("https://github.com/") for r in rows) == len(rows) - 1,
            "every repo is on GitHub except Hatchery (a hosted service)",
        ),
        check(len({r["name"] for r in rows}) == len(rows), "no tool is listed twice"),
        check(
            all(len(r["use_it_when"]) <= 300 for r in rows),
            "every 'use it when' note is one short sentence (<= 300 characters)",
        ),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
