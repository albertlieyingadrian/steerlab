"""Improver v2: knowledge-carrying, angle-diversified recipe proposer.

    python improver.py <task.json> <out_dir>
"""

import json
import random
import re
import sys
from pathlib import Path

from broker import ask, validate

HERE = Path(__file__).resolve().parent
FILE_RE = re.compile(r"^=== FILE: (\S+) ===\n(.*?)\n=== END FILE ===", re.DOTALL | re.MULTILINE)
HYP_RE = re.compile(r"^HYPOTHESIS:\s*(.+)$", re.MULTILINE)
MAX_TOKENS = 12000

ANGLES = [
    "ANGLE cutoff-small: keep R0's data exactly (same mining top-2000 as human, same replay rows) and change ONLY the "
    "cutoff: take the q99 human cutoff and raise it modestly (about q99.2-q99.5 of v2b-val human scores, never more "
    "than ~0.03 above the q99 value), also guarding against recall loss by checking v2b-val AI recall at the new cutoff.",
    "ANGLE host-diverse-negatives: keep R0's size (about 2,000 mining human rows) but pick the highest-scored mining "
    "pages with a cap per host (e.g. at most 3-5 per host) so the hard negatives cover many more kinds of human "
    "writing; no mirrors; same replay amount; same cutoff rule as R0.",
    "ANGLE gentle-update: R0's data but lower lr (about 6e-5 to 8e-5) and unchanged epochs, so the detector drifts "
    "less from the starting model on AI recall while still absorbing the hard human negatives; cutoff rule as R0.",
    "ANGLE stronger-negatives: R0's data but give the mining human rows weight about 1.5-2.0 (bucket 0), keep "
    "8,000 replay rows, no mirrors, lr 1e-4; cutoff rule as R0 (q99 of v2b-val human).",
    "ANGLE replay-balance: keep R0's 2,000 mining human rows but rebuild the replay part: roughly 9,000-10,000 "
    "replay rows stratified over bucket/kind/source so human rows and fully-AI rows are balanced and the AI "
    "heavy-edit and light-edit buckets stay represented (protects recall); no mirrors; cutoff rule as R0.",
    "ANGLE augmented-views: R0's data, plus the same top mining pages again with aug 'tail' or 'middle' (about "
    "1,000 extra rows at weight 1.0, bucket 0) so the detector learns that polished human text is human wherever "
    "the window falls; no mirrors; cutoff rule as R0.",
    "ANGLE unlabeled-calibrated-cutoff: R0's data unchanged; set the cutoff to the larger of the q99 of v2b-val "
    "human scores and a moderate quantile (about q99) of cal['unlabeled'] scores, capped at +0.04 over the "
    "v2b-val q99 value, so web-like human pages that the detector still flags get cleared.",
    "ANGLE rank-window: use mining pages from rank 0-2,400 by score but drop the top ~150 (likely mislabeled AI-like "
    "pages), keep about 2,000 rows; no mirrors; 8,000 replay rows; cutoff rule as R0.",
]


def fmt_files(files: dict) -> str:
    return "\n\n".join(f"=== FILE: {n} ===\n{s}\n=== END FILE ===" for n, s in files.items())


def render(name: str, task: dict, extra: dict) -> str:
    t = (HERE / "prompts" / name).read_text()
    fields = {
        "FILES": fmt_files(task["files"]),
        "FEEDBACK": task.get("feedback", "") or "(none)",
        "ARCHIVE": task.get("archive", "") or "(none)",
    }
    fields.update({k.upper(): v for k, v in task.get("docs", {}).items() if k.upper() not in fields})
    fields.update(extra)
    for k, v in fields.items():
        t = t.replace(f"[[{k}]]", str(v))
    return t


def parse(text: str) -> tuple[dict, str]:
    files = {m.group(1): m.group(2) for m in FILE_RE.finditer(text)}
    m = HYP_RE.search(text)
    return files, (m.group(1).strip() if m else "")


def pick_angle(task: dict) -> str:
    rng = random.SystemRandom()
    archive = task.get("archive", "") or ""
    # prefer angles whose key word is not already mentioned in the archive
    fresh = [a for a in ANGLES if a.split(":")[0].replace("ANGLE ", "") not in archive]
    return rng.choice(fresh or ANGLES)


def main() -> None:
    task = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    system = (HERE / "prompts" / "system.md").read_text()
    if task["kind"] == "recipe":
        user = render("recipe.md", task, {"ANGLE": pick_angle(task)})
    else:
        user = render("meta.md", task, {})
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = ask(msgs, max_tokens=MAX_TOKENS, temperature=0.6)
    files, hyp = parse(text)
    calls = 1
    for _ in range(2):
        if files:
            try:
                v = validate(files)
            except Exception as e:  # budget refusal etc.
                break
            if v.get("ok"):
                break
            err = v.get("error") or "unknown error"
        else:
            err = "No files found. Use exactly the === FILE: <name> === ... === END FILE === format."
        msgs += [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": f"Validation failed:\n{err[:3000]}\n\nReturn the corrected complete files in the same "
                "format, then the HYPOTHESIS line.",
            },
        ]
        try:
            text = ask(msgs, max_tokens=MAX_TOKENS, temperature=0.3)
        except Exception:
            break
        calls += 1
        f2, h2 = parse(text)
        files, hyp = (f2 or files), (h2 or hyp)
    (out / "files").mkdir(parents=True, exist_ok=True)
    for n, s in files.items():
        p = out / "files" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s)
    (out / "proposal.json").write_text(json.dumps({"hypothesis": hyp, "calls": calls}))


main()
EOF_MARK = None