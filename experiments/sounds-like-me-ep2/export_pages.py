"""Write report/data.js for the Episode 2 note: one row per arm (held-out) and a small draft viewer.

    python export_pages.py      -> report/data.js (window.SLM2)

Drafts are model text written for synthetic briefs. The real authors are never named: each is "Author A" and so on,
and any of the author's name parts or initials that a draft copied from the samples are replaced with [name].
"""

import json
import re

from ep2 import RES, authors
from make_figures import LABEL

HERE = RES.parent
KIND = {
    "R1_prompt": "harness",
    "R1b_bon": "harness",
    "R2_harness": "harness",
    "G2_gemini_harness": "harness",
    "abl_noretrieval": "ablation",
    "abl_noprofile": "ablation",
    "abl_norevise": "ablation",
    "abl_norerank": "ablation",
    "R3_voice": "steer",
    "xR3_s0.3": "steer",
    "R4_persona": "steer",
    "R5_both": "steer",
    "away": "steer",
    "own": "steer",
    "card_own": "steer",
    "card": "prompt",
    "R6_rand0": "control",
    "R6_rand1": "control",
    "rand0": "control",
    "rand1": "control",
    "rand2": "control",
    "toward": "control",
    "none": "control",
}
LADDER = ("R1_prompt", "R1b_bon", "R2_harness", "R3_voice", "xR3_s0.3", "R4_persona", "R5_both", "R6_rand0", "R6_rand1")
PERSONA = ("none", "away", "toward", "rand0", "rand1", "rand2", "card", "own", "card_own")
GROUP = {
    **{a: "Ladder (Qwen3-8B)" for a in LADDER},
    **{a: "Ablations of the harness" for a in ("abl_noretrieval", "abl_noprofile", "abl_norevise", "abl_norerank")},
    **{a: "Persona tests (no harness)" for a in PERSONA},
    "G2_gemini_harness": "Closed model in the harness",
}
VIEW_ARMS = ["ep1-gemini-edits", "R1_prompt", "R2_harness", "R4_persona", "R3_voice", "xR3_s0.3", "G2_gemini_harness"]


# Hand-reviewed: names the displayed drafts copied from the writers' samples (found by listing capitalised words that
# appear in the writer's samples but not in the brief, then read by eye).
EXTRA_NAMES = ["Lauren", "Tamiathen", "James S."]


def redact(text, author):
    for n in EXTRA_NAMES:
        text = re.sub(rf"\b{re.escape(n)}", "[name]", text)
    parts = [p for p in re.split(r"\W+", author) if len(p) >= 3]
    initials = "".join(p[0] for p in parts if p[0].isupper())
    for p in sorted(parts, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(p)}\b", "[name]", text, flags=re.IGNORECASE)
    if len(initials) >= 2:
        text = re.sub(rf"\b{re.escape(initials)}\b", "[name]", text, flags=re.IGNORECASE)
    return text


def main():
    s = json.loads((RES / "summary.json").read_text())
    rows = []
    for a, r in s["arms"].items():
        rows.append(
            dict(
                arm=a,
                label=LABEL.get(a, a),
                kind=KIND.get(a, "ep1"),
                group=GROUP.get(a, "Episode 1 (reused drafts)"),
                pos=r["position"],
                pos80=r["position80"][0],
                spec=r["spec"][0],
                uns=r["unsupported"][0],
                rel=r["invented_relationship"],
                ai=r["detector"],
                loop=r["loop_or_degenerate"] if r.get("loop_or_degenerate") is not None else r["loops"],
                clean=r.get("clean_position"),
                words=r["words"],
                attr=r["attr_mean_rank"],
                attr1=r["attr_top1"],
                pref=r["pref_mean"],
                usd=r["usd_per_draft"],
                copy=r["copy_flag_episodes"],
                n=r["episodes"],
            )
        )
    held = [a for a in authors()["authors"] if a["split"] == "heldout"]
    pick_authors = [held[i] for i in (0, 5, 11, 17, 24, 27)]  # fixed positions, chosen before reading drafts
    texts_by = {}
    for p in sorted((RES / "scored").glob("*__heldout.json")):
        for r in json.loads(p.read_text())["drafts"]:
            texts_by[(r["arm"], r["author"], r["brief"])] = r["text"]
    drafts = []
    for i, a in enumerate(pick_authors):
        briefs = sorted({b for (arm, au, b) in texts_by if au == a["author"] and arm == "R2_harness"})
        if not briefs:
            continue
        b = briefs[0]
        arms = {
            arm: redact(texts_by[(arm, a["author"], b)], a["author"])
            for arm in VIEW_ARMS
            if (arm, a["author"], b) in texts_by
        }
        drafts.append(dict(who=f"Author {chr(65 + i)} ({a['corpus'].capitalize()})", brief=b, arms=arms))
    out = dict(rows=rows, drafts=drafts, labels={a: LABEL.get(a, a) for a in VIEW_ARMS}, human_spec=0.172)
    (HERE / "report").mkdir(exist_ok=True)
    (HERE / "report" / "data.js").write_text("window.SLM2 = " + json.dumps(out, ensure_ascii=False) + ";\n")
    print("data.js:", len(rows), "rows,", len(drafts), "draft sets")


if __name__ == "__main__":
    main()
