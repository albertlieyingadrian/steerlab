"""Write report/data.js for the Episode 3 note: arm rows (held-out), the hack-hunt series, and a checkpoint slider of
drafts for a few dev writers (the same writer and brief at every GRPO checkpoint).

    python export_pages.py      -> report/data.js (window.SLM3)

Writers are never named ("Writer A"); name parts and initials a draft copied from the samples become [name] (Episode 2's
redaction rule, copied here), plus a hand-reviewed list kept in the lab; emails, phone numbers and addresses become
[email], [phone] and [address] (Episode 1's privacy.py).
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent / "sounds-like-me"))

from analyze_ep3 import EP2_ARMS  # noqa: E402
from common import authors  # noqa: E402
from make_figures import LABEL  # noqa: E402
from privacy import copied_names, redact_contact  # noqa: E402

R = HERE / "results"
D = HERE / "data" / "drafts"
RUN = "grpo_sft_edits"
# Hand-reviewed after reading the slider's drafts: names copied from writers' samples or invented as a signature (brief
# recipients such as "Brenda" are synthetic and stay). The list stays in the lab (privacy.py).
EXTRA_NAMES = copied_names("ep3")
TRAINED = ("sft_raw", "sft_notes", "sft_edits", "dpo", "grpo_best", "grpo_final")


def redact(text, author):
    """Episode 2's rule: the writer's name parts (3+ letters) and initials become [name]."""
    text = re.sub(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", "[email]", text)
    for n in EXTRA_NAMES:
        text = re.sub(rf"\b{re.escape(n)}\b", "[name]", text, flags=re.IGNORECASE)
    parts = [p for p in re.split(r"\W+", author) if len(p) >= 3]
    initials = "".join(p[0] for p in parts if p[0].isupper())
    for p in sorted(parts, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(p[0].upper() + p[1:])}\b", "[name]", text)
    if len(initials) >= 2:
        text = re.sub(rf"\b{re.escape(initials)}\b", "[name]", text, flags=re.IGNORECASE)
    return redact_contact(text)


def episode(a):
    if a.startswith("ep1"):
        return 1
    return 2 if a in EP2_ARMS else 3


def main():
    s = json.loads((R / "summary.json").read_text())
    h = json.loads((R / "hack_hunt.json").read_text())
    rows = []
    for a, r in s["arms"].items():
        rows.append(
            dict(
                arm=a,
                label=LABEL.get(a, a),
                pos=r["position"],
                pos80=r["position80"][0],
                spec=r["spec"][0],
                uns=r["unsupported"][0],
                loop=r["loop_share"],
                task=r["task"],
                words=r["words"],
                ai=r["detector"],
                attr=r["attr_mean_rank"],
                trained=a in TRAINED,
                ep=episode(a),
            )
        )
    run = h["runs"][RUN]
    hunt = []
    if run["start"]:
        st = run["start"]
        hunt.append(
            dict(
                step=0,
                position=st["position"][0],
                unsupported=st["unsupported"][0],
                words=st["words"],
                task=st["task"],
                loops=st["loop_share"],
                reward=None,
            )
        )
    for p in run["checkpoints"]:
        hunt.append(
            dict(
                step=p["step"],
                position=p["position"][0],
                unsupported=p["unsupported"][0],
                words=p["words"],
                task=p["task"],
                loops=p["loop_share"],
                reward=p["train_reward"],
            )
        )
    curve = [
        dict(step=c["step"], reward=round(c["reward"], 3), words=round(c["words"], 1)) for c in run["reward_curve"]
    ]
    dev = [a for a in authors()["authors"] if a["split"] == "dev"]
    pick = [dev[i] for i in (0, 4, 9, 13)]  # fixed positions, chosen before reading drafts
    files = {0: D / f"{run['start_arm']}__dev.jsonl"} | {
        p["step"]: D / f"{RUN}_s{p['step']}__dev.jsonl" for p in run["checkpoints"]
    }
    by = {}
    for st, f in files.items():
        for r in (json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()):
            by[(st, r["author"], r["brief"])] = r["text"]
    slider = []
    for i, a in enumerate(pick):
        b = sorted({k[2] for k in by if k[1] == a["author"] and k[0] == 0})[0]
        steps = {
            st: (redact(by.get((st, a["author"], b), ""), a["author"]).strip() or "(empty draft)")
            for st in sorted(files)
        }
        slider.append(dict(who=f"Writer {chr(65 + i)} (Enron)", brief=b, steps=steps))
    out = dict(rows=rows, hunt=hunt, curve=curve, slider=slider, human_spec=0.172)
    (HERE / "report").mkdir(exist_ok=True)
    (HERE / "report" / "data.js").write_text("window.SLM3 = " + json.dumps(out, ensure_ascii=False) + ";\n")
    print("data.js:", len(rows), "rows,", len(hunt), "checkpoints,", len(slider), "slider writers")


if __name__ == "__main__":
    main()
