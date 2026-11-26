"""Improver v4: winner-based angles (rank-window + large fixed cutoff offsets), feedback-driven refinement round.

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

KNOWLEDGE = (
    "\n\nNEWER EVIDENCE (supersedes any older advice in the archive section): (1) rank-window mining data (sort by "
    "mining_scores_r0 desc, drop top 150, sample ~2,000 from ranks 150-2,400, bucket 0) + R0's 8,000 replay rows, "
    "1 epoch, lr 1e-4 is the base that tolerates a large cutoff raise: with cutoff min(q99+0.10, 0.52) dev gain was "
    "+19.5% (recall 89.9%), +67.5% in another run, and with cutoff about q99+0.18 (~0.535, cap 0.55) +43.5% (recall "
    "89.7%). (2) R0's own top-2000 mining data with the large raise FAILED the recall floor (86.9%) - do not use it. "
    "(3) A fit_cutoff that backs the cutoff DOWN using a v2b-val recall guard gave -30%: do NOT add recall guards; use a "
    "plain fixed offset over the v2b-val human q99, with a cap. (4) Dev recall slack is ~2 points; recall barely moved "
    "between cutoff 0.47 and 0.535 on this data. Never add mirrors, extra mining rows or other big data changes. "
    "fit_cutoff must be robust: numpy, handle empty/missing input (fall back to 0.5), return float clipped to [0,1]. "
    "Parse the replay file defensively (skip bad lines) and keep every idx valid."
)

# First-round angles: (weight, text)
ANGLES = [
    (4, "ANGLE offset-018: rank-window data as described, 1 epoch, lr 1e-4, cutoff = min(q99 of v2b-val human scores "
        "+ 0.18, 0.55). This is the best known recipe; reproduce it faithfully with robust code."),
    (3, "ANGLE offset-022: rank-window data as described, cutoff = min(q99 of v2b-val human scores + 0.22, 0.58). A "
        "slightly bigger raise to clear more borderline human pages (magazine, editlens_h) since recall slack exists."),
    (2, "ANGLE offset-014: rank-window data as described, cutoff = min(q99 of v2b-val human scores + 0.14, 0.54)."),
    (2, "ANGLE offset-018-window2: rank-window data but window ranks 250-2,800 (sample ~2,000, seeded rng), cutoff = "
        "min(q99 + 0.18, 0.55). Drops more of the AI-like top ranks to protect recall."),
    (1, "ANGLE offset-018-lowlr: rank-window data, lr 8e-5, cutoff = min(q99 + 0.18, 0.55)."),
]

# Refinement angles used when the task already carries feedback from a previous recipe.
REFINE = [
    (5, "ANGLE refine-from-feedback: start from the given recipe (keep its data unless the feedback says it failed the "
        "recall constraint with R0-style data, in which case switch to the rank-window data). Read its dev recall and "
        "the flagged human scores in the feedback. If it passed the recall floor with >=1.0 points of slack, raise the "
        "cutoff offset over q99 by +0.03 to +0.05 (cap up to 0.60) and aim just above the bulk of the listed flagged "
        "human scores. If recall slack is under 1.0 point, raise by at most +0.01..+0.02. If the constraint failed, "
        "lower the offset by 0.05 and use rank-window data. Keep fit_cutoff a plain fixed offset with a cap (no recall "
        "guard)."),
    (2, "ANGLE refine-offset-018: regardless of the given recipe, output rank-window data with cutoff = min(q99 of "
        "v2b-val human + 0.18, 0.55)."),
    (2, "ANGLE refine-offset-024: rank-window data with cutoff = min(q99 of v2b-val human + 0.24, 0.60)."),
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
    fb = (task.get("feedback") or "").strip()
    src = REFINE if (len(fb) > 60 and fb != "(none)") else ANGLES
    pool = []
    for w, a in src:
        pool += [a] * w
    return rng.choice(pool) + KNOWLEDGE


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