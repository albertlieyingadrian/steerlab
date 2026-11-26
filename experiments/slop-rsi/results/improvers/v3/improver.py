"""Improver v3: carries the winning recipe knowledge (rank-window + large capped cutoff raise), angle-weighted.

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
ARCHIVE_CHARS = 9000

# (weight, text). Heavily weighted to the one pattern known to have produced a real gain.
ANGLES = [
    (4, "ANGLE winner-replicate: rank-window mining data (sort mining pages by mining_scores_r0 descending, drop the top "
        "150, sample ~2,000 pages from ranks 150-2,400 with the seeded rng, bucket 0, same weight as R0) + R0's 8,000 "
        "replay rows (parsed defensively) + 1 epoch, lr 1e-4, and cutoff = min(q99 of v2b-val human scores + 0.10, 0.52). "
        "This pattern gave +67.5% dev gain; reproduce it faithfully with robust code."),
    (3, "ANGLE winner-bigger-offset: same rank-window data as the proven recipe, but the cutoff is min(q99 of v2b-val "
        "human + 0.14, 0.56), then stepped DOWN in 0.01 increments while v2b-val recall on bucket>=2 (heavy edit / fully "
        "AI) texts is more than 0.04 below its value at the q99 cutoff. Larger offsets clear more borderline human pages "
        "while dev recall had ~1.6 points of slack."),
    (2, "ANGLE winner-feedback-cutoff: same rank-window data as the proven recipe; choose the cutoff from the dev "
        "feedback: if the feedback lists scores of flagged human texts, put the cutoff just above the bulk of them "
        "(but never above 0.58 and never below q99 of v2b-val human + 0.08). Otherwise min(q99 + 0.10, 0.52)."),
    (2, "ANGLE winner-R0data-cutoff: R0's exact data (top-2,000 mining pages, 8,000 replay) but the proven large cutoff "
        "min(q99 of v2b-val human + 0.10, 0.52). Isolates the cutoff lever."),
    (2, "ANGLE winner-plus-gentle: proven rank-window data and cutoff min(q99+0.10, 0.52), with lr lowered to 8e-5 so "
        "the detector keeps more AI recall."),
    (1, "ANGLE winner-wider-window: proven recipe but the window is ranks 300-3,000 (sample 2,000), cutoff "
        "min(q99+0.10, 0.52). Tests whether dropping even more of the AI-like top ranks helps."),
]


def fmt_files(files: dict) -> str:
    return "\n\n".join(f"=== FILE: {n} ===\n{s}\n=== END FILE ===" for n, s in files.items())


def render(name: str, task: dict, extra: dict) -> str:
    t = (HERE / "prompts" / name).read_text()
    arch = task.get("archive", "") or "(none)"
    if len(arch) > ARCHIVE_CHARS:
        arch = arch[:ARCHIVE_CHARS] + "\n...(truncated)"
    fb = task.get("feedback", "") or "(none)"
    if len(fb) > 14000:
        fb = fb[:14000] + "\n...(truncated)"
    fields = {"FILES": fmt_files(task["files"]), "FEEDBACK": fb, "ARCHIVE": arch}
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
    pool = []
    for w, a in ANGLES:
        pool += [a] * w
    return rng.choice(pool)


def main() -> None:
    task = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    system = (HERE / "prompts" / "system.md").read_text()
    if task["kind"] == "recipe":
        user = render("recipe.md", task, {"ANGLE": pick_angle(task)})
    else:
        user = render("meta.md", task, {})
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = ask(msgs, max_tokens=MAX_TOKENS, temperature=0.5)
    files, hyp = parse(text)
    calls = 1
    for _ in range(2):
        if files:
            try:
                v = validate(files)
            except Exception:  # budget refusal etc.
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