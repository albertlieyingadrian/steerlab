"""Relabel drafts so each sits in the cell its prompt belongs to. Rewrites files in place; idempotent.

    python relabel_rerun.py data/drafts/<model>-editsfix__heldout.jsonl     the edits rerun (amendment A5)
    python relabel_rerun.py --legacy data/drafts/<model>__heldout.jsonl     the published no-pairs cells

Rerun: run_closed.py / run_open.py with --arms edits --tag editsfix write model "<model>-editsfix", arm "edits".
That becomes model "<model>", arm "edits_fixed", job "author|brief|edits_fixed", beside the published cells.

Legacy: since the gen.py fix, arm `edits` carries the pairs, so run_partb.sh rebuilds the published `edits` cells
("3 more samples*": the same 3 samples, no pairs) as arm `edits_nopairs`; --legacy maps that back to arm `edits`.
"""

import json
import sys
from pathlib import Path

RERUN_ARM, RERUN_TAG, LEGACY_ARM = "edits_fixed", "-editsfix", "edits_nopairs"


def relabel_rerun(rows):
    for r in rows:
        if not r["model"].endswith(RERUN_TAG) and r["arm"] != RERUN_ARM:
            raise ValueError(f"{r['model']}: not a rerun file (model lacks {RERUN_TAG})")
        if r["arm"] not in ("edits", RERUN_ARM):
            raise ValueError(f"unexpected arm {r['arm']!r}; the rerun has only the edits arm")
        r["model"] = r["model"].removesuffix(RERUN_TAG)
        r["arm"] = RERUN_ARM
        r["job"] = f"{r['author']}|{r['brief']}|{RERUN_ARM}"
    return rows


def relabel_legacy(rows):
    for r in rows:
        if r["arm"] == LEGACY_ARM:
            r["arm"] = "edits"
            r["job"] = f"{r['author']}|{r['brief']}|edits"
    return rows


def main(argv):
    legacy = "--legacy" in argv
    for path in map(Path, [a for a in argv if a != "--legacy"]):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rows = relabel_legacy(rows) if legacy else relabel_rerun(rows)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{path.name}: {len(rows)} drafts relabelled ({'legacy edits' if legacy else RERUN_ARM})")


if __name__ == "__main__":
    main(sys.argv[1:])
