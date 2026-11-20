"""Score drafts: voice position (LUAR, 4-draft episodes), AI-ness, copying, length, task coverage.

    python score.py data/drafts/<file>.jsonl [...]      -> data/scored/<file>.json

Controls added per (author, brief) where a k0 draft exists:
  copy_paste   sentences pasted from the author's own samples (positive control: LUAR must rank it high,
               the copy filter must flag it)
  surface_tics a k0 draft with the author's own greeting line and sign-off pasted on (cheap baseline)
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import DATA, VoiceScale, authors, briefs, texts

WORD = re.compile(r"[A-Za-z0-9']+")


def words(t):
    return [w.lower() for w in WORD.findall(t)]


def longest_shared_run(draft, sources):
    d = words(draft)
    best = 0
    for s in sources:
        sw = words(s)
        idx = defaultdict(list)
        for i, w in enumerate(sw):
            idx[w].append(i)
        for i in range(len(d)):
            for j in idx.get(d[i], []):
                k = 0
                while i + k < len(d) and j + k < len(sw) and d[i + k] == sw[j + k]:
                    k += 1
                best = max(best, k)
    return best


def ngram_copy_share(draft, sources, n=8):
    d = words(draft)
    grams = {tuple(d[i : i + n]) for i in range(len(d) - n + 1)}
    if not grams:
        return 0.0
    src = set()
    for s in sources:
        sw = words(s)
        src |= {tuple(sw[i : i + n]) for i in range(len(sw) - n + 1)}
    return len(grams & src) / len(grams)


def task_coverage(draft, b):
    t = draft.lower()
    p = b["prospect"]
    checks = [
        p["first_name"].lower() in t,
        p["company"].split()[0].lower() in t,
        any(w in t for w in ("call", "chat", "talk", "reply", "minutes", "connect", "contact")),
    ]
    return float(np.mean(checks))


def copy_paste_control(rec, i):
    """Verbatim text from the author's own samples, layout kept, ~120 words (positive control for copying)."""
    t = texts(rec, "samples")[i % 4]
    out, n = [], 0
    for line in t.splitlines():
        out.append(line)
        n += len(line.split())
        if n >= 120:
            break
    return "\n".join(out)


def surface_tics(draft_k0, rec):
    s = texts(rec, "samples")[0].strip().splitlines()
    first, last = s[0], s[-1]
    return f"{first}\n\n{draft_k0}\n\n{last}"


def main(paths, controls=False):
    from detectors import voi380

    A = {a["author"]: a for a in authors()["authors"]}
    floor = authors()["floor"]
    B = {b["id"]: b for b in briefs()}
    for path in map(Path, paths):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        k0 = {(r["author"], r["brief"]): r["text"] for r in rows if r["arm"] == "k0"} if controls else {}
        extra = []
        for i, ((au, bid), t) in enumerate(k0.items()):
            extra.append(dict(model="ctrl", author=au, brief=bid, arm="copy_paste", text=copy_paste_control(A[au], i)))
            extra.append(dict(model="ctrl", author=au, brief=bid, arm="surface_tics", text=surface_tics(t, A[au])))
        rows_all = rows + extra
        det = voi380([r["text"] or " " for r in rows_all])
        for r, s in zip(rows_all, det, strict=False):
            src = texts(A[r["author"]], "samples")
            r.update(
                words=len(words(r["text"])),
                voi380=float(s),
                longest_run=longest_shared_run(r["text"], src),
                copy8=ngram_copy_share(r["text"], src),
                task=task_coverage(r["text"], B[r["brief"]]),
                empty=not r["text"].strip(),
            )
        scales, loose, groups, episodes = {}, {}, defaultdict(list), []
        for r in rows_all:
            groups[(r["model"], r["author"], r["arm"])].append(r)
        for (m, au, arm), g in groups.items():
            if au not in scales:
                scales[au] = VoiceScale(A[au], floor, "strict")
                loose[au] = VoiceScale(A[au], floor, "loose")
            ep = [r["text"] for r in g if r["text"].strip()]
            if not ep:
                continue
            vs, vl = scales[au], loose[au]
            episodes.append(
                dict(
                    model=m,
                    author=au,
                    arm=arm,
                    n=len(ep),
                    position=vs.position(ep),
                    sim=vs.sim(ep),
                    floor=vs.floor,
                    ceiling=vs.ceiling,
                    position_loose=vl.position(ep),
                    copy_flag=any(r["longest_run"] >= 12 or r["copy8"] >= 0.2 for r in g),
                )
            )
        out = DATA / "scored" / (path.stem + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(drafts=rows_all, episodes=episodes), indent=1))
        print(out.name, len(rows_all), "drafts,", len(episodes), "episodes")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--controls"]
    main(args, controls="--controls" in sys.argv)
