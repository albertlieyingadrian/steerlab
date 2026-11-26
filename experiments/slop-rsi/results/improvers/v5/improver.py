"""Improver v5: winner-replicating, offset-diversified recipe proposer with a reference implementation and fallback.

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

REFERENCE = '''
REFERENCE CODE (use it nearly verbatim; adapt only to the starting recipe's file layout):

def fit_cutoff(cal):
    import numpy as np
    try:
        s = np.asarray(cal["v2b_val"]["scores"], dtype=float)
        k = np.asarray(cal["v2b_val"]["kind"])
        h = s[k == "human"]
        if h.size < 50:
            return 0.45
        q99 = float(np.quantile(h, 0.99))
        c = min(q99 + OFFSET, CAP)
        c = max(c, q99)
        return float(min(max(c, 0.0), 1.0))
    except Exception:
        return 0.45

# mining part of build(ctx): rank-window data (the proven l3_08 data), bucket 0, R0's weight and R0's replay unchanged
#   sc = np.load(<data>/mining_scores_r0.npy); order = np.argsort(-sc); window = order[150:2400]
#   pick = random.Random(ctx["seed"]).sample(list(map(int, window)), 2000)  -> rows pool "mining", bucket 0
# Keep replay selection/parsing exactly like the starting recipe but wrap in try/except. 1 epoch, lr 1e-4.
'''

FACTS = (
    "FACTS FROM THE ARCHIVE (override any stale notes above): the rank-window data (drop top 150 of mining_scores_r0 "
    "ranking, sample 2,000 from ranks 150-2,400) gives dev recall ~89.4-91% at q99; with R0's own top-2,000 data a "
    "cutoff of ~q99+0.10 dropped recall to 86.92% and FAILED the floor (87.74%). With rank-window data: offset ~+0.10 "
    "(cutoff ~0.47) gave +19.5% and +67.5% in two runs, offset ~+0.18 (cutoff ~0.535) gave +43.5% with recall 89.73%, "
    "so recall barely falls between 0.47 and 0.535 and there is slack of ~2 points. Do NOT add a recall-based guard "
    "that lowers the cutoff (it lost: -30.3%); v2b-val recall is a poor proxy for dev recall. Use the fixed offset rule. "
    "No mirrors, no extra mining rows, no other data changes. A failed recall constraint only scores 0, so being bold "
    "with the cutoff is fine, but data must be the rank-window data (never R0's top-2,000 mining rows).\n"
)

# round-1 options: (weight, offset, cap)
ROUND1 = [
    (2, 0.12, 0.52),
    (3, 0.16, 0.54),
    (3, 0.20, 0.58),
    (2, 0.24, 0.60),
]
# round-2 options when feedback exists
ROUND2_OFFSETS = [(0.16, 0.56), (0.20, 0.58), (0.24, 0.62), (0.28, 0.65)]

FALLBACK_RECIPE = '''import json
import random

import numpy as np


def build(ctx):
    data = ctx["data"]
    rows = []
    try:
        sc = np.load(data + "/mining_scores_r0.npy")
        order = np.argsort(-sc)
        window = [int(i) for i in order[150:2400]]
        pick = random.Random(ctx.get("seed", 0)).sample(window, 2000)
        for i in pick:
            rows.append({"pool": "mining", "idx": i, "bucket": 0, "weight": 1.3278, "aug": "none"})
    except Exception:
        pass
    try:
        rng = random.Random(1)
        for i in rng.sample(range(79120), 8000):
            rows.append({"pool": "replay", "idx": i, "bucket": -1, "weight": 1.0, "aug": "none"})
    except Exception:
        pass
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "fallback"}


def fit_cutoff(cal):
    try:
        s = np.asarray(cal["v2b_val"]["scores"], dtype=float)
        k = np.asarray(cal["v2b_val"]["kind"])
        h = s[k == "human"]
        if h.size < 50:
            return 0.45
        q99 = float(np.quantile(h, 0.99))
        return float(min(max(max(min(q99 + 0.16, 0.54), q99), 0.0), 1.0))
    except Exception:
        return 0.45
'''


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


def has_feedback(task: dict) -> bool:
    fb = (task.get("feedback", "") or "").strip()
    return bool(fb) and fb != "(none)"


def pick_angle(task: dict) -> str:
    rng = random.SystemRandom()
    if not has_feedback(task):
        pool = []
        for w, off, cap in ROUND1:
            pool += [(off, cap)] * w
        off, cap = rng.choice(pool)
        txt = (
            f"ANGLE rank-window + large cutoff raise: build the rank-window data described in the reference and set "
            f"fit_cutoff = min(q99 of v2b-val human + {off:.2f}, {cap:.2f}) (OFFSET={off:.2f}, CAP={cap:.2f}). "
        )
    else:
        off, cap = rng.choice(ROUND2_OFFSETS)
        txt = (
            "ANGLE feedback-refined cutoff: read the dev feedback for the starting recipe (dev macro recall, "
            "constraint status, scores of flagged human texts). Keep the rank-window data (if the starting recipe "
            "is R0 or uses R0's top-2,000 mining rows, switch to the rank-window data). If the starting recipe's "
            "dev macro recall was above ~88.7% and the constraint passed, RAISE its cutoff offset by about 0.04-0.08 "
            "(and raise the cap) to clear the flagged human texts still above the cutoff, but do not go above "
            "~0.62 absolute. If its constraint failed or recall was below ~88.3%, LOWER the offset by about "
            "0.05-0.08. If the starting recipe is R0 or has no useful feedback, use "
            f"OFFSET={off:.2f}, CAP={cap:.2f}. State the numbers you chose. "
        )
    return txt + FACTS + REFERENCE.replace("OFFSET", f"{off:.2f}").replace("CAP", f"{cap:.2f}")


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
    if not files and task["kind"] == "recipe":
        files = {"recipe.py": FALLBACK_RECIPE}
        hyp = hyp or "Fallback: rank-window mining data with cutoff q99+0.16 capped at 0.54."
    (out / "files").mkdir(parents=True, exist_ok=True)
    for n, s in files.items():
        p = out / "files" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s)
    (out / "proposal.json").write_text(json.dumps({"hypothesis": hyp, "calls": calls}))


main()