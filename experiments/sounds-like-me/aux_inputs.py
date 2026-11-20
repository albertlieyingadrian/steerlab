"""Inputs for the `note` and `edits` arms, made once per author with an open model on Modal (Qwen3-32B).

style notes   data/style_notes.json   {author: "5-7 bullet points on how they write (no topics, no names)"}
sim edits     data/sim_edits.json     {author: [[generic_draft, simulated_author_edit], x3]}   (labelled SIMULATED)

The simulated editor is calibrated against the private board's aggregate rates from real use of a freight
sales-writing tool (results/private_board.json): its edits should land near the real range of trigram
similarity to the draft and length ratio. The check is written to results/sim_editor_calibration.json.
Edit pairs use briefs that are NOT among the author's evaluation briefs.
"""

import json
import random
import time

import numpy as np

from common import DATA, HERE, authors, briefs, record_spend, texts
from gen import SYSTEM, brief_block, jobs

MODEL = "Qwen/Qwen3-32B"

NOTE = """Here are {n} emails written by one person:

{samples}

Describe how this person writes, in 5 to 7 short bullet points: typical length, how they open and close,
sentence length, punctuation and capitalization habits, tone and formality, and any recurring phrasing.
Do not mention topics, companies or names. Output only the bullet points."""

EDIT = """Here are {n} emails written by one person:

{samples}

Below is a draft someone else wrote for them. Edit it the way this person would before sending it: change only
what they would change (wording, length, opening, sign-off, punctuation, formality). Keep every fact and do not add
any new claim. Output only the edited email.

Draft:
{draft}"""


def trigrams(s):
    s = "  " + " ".join(s.lower().split()) + " "
    return {s[i : i + 3] for i in range(len(s) - 2)}


def trigram_similarity(a, b):
    """Same idea as Postgres pg_trgm similarity(): shared trigrams / union."""
    ta, tb = trigrams(a), trigrams(b)
    return len(ta & tb) / max(1, len(ta | tb))


def main():
    import modal

    from modal_gen import charge

    fn = modal.Function.from_name("sounds-like-me", "generate")
    A = authors()["authors"]
    B = briefs()
    used = {}
    for split in ("dev", "heldout"):
        for j in jobs(split, arms=["k0"], briefs_per_author=5):
            used.setdefault(j["rec"]["author"], set()).add(j["brief"]["id"])

    note_prompts = []
    for rec in A:
        s = texts(rec, "samples")[:8]
        note_prompts.append([{"role": "user", "content": NOTE.format(n=len(s), samples="\n\n---\n\n".join(s))}])
    t0 = time.time()
    notes = fn.remote(MODEL, note_prompts, True, 300, 0.2, 0)["outputs"]
    record_spend("modal", charge(time.time() - t0), f"style notes n={len(A)}")
    (DATA / "style_notes.json").write_text(
        json.dumps({r["author"]: n.strip() for r, n in zip(A, notes, strict=False)}, indent=1)
    )

    pairs_src = []
    for rec in A:
        rng = random.Random(f"edits-{rec['author']}")
        pool = [b for b in B if b["id"] not in used.get(rec["author"], set())]
        for b in rng.sample(pool, 3):
            pairs_src.append((rec, b))
    draft_prompts = [
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": brief_block(b)}] for _, b in pairs_src
    ]
    t0 = time.time()
    drafts = fn.remote(MODEL, draft_prompts, True, 400, 0.7, 1)["outputs"]
    edit_prompts = []
    for (rec, _), d in zip(pairs_src, drafts, strict=False):
        s = texts(rec, "samples")[:8]
        edit_prompts.append(
            [{"role": "user", "content": EDIT.format(n=len(s), samples="\n\n---\n\n".join(s), draft=d.strip())}]
        )
    edits = fn.remote(MODEL, edit_prompts, True, 500, 0.4, 2)["outputs"]
    record_spend("modal", charge(time.time() - t0), f"sim edits n={len(pairs_src)}")
    out, sims, ratios = {}, [], []
    for (rec, _), d, e in zip(pairs_src, drafts, edits, strict=False):
        out.setdefault(rec["author"], []).append([d.strip(), e.strip()])
        sims.append(trigram_similarity(d, e))
        ratios.append(len(e) / max(1, len(d)))
    (DATA / "sim_edits.json").write_text(json.dumps(out, indent=1))
    real = json.loads((HERE / "results" / "private_board.json").read_text())["cells"]
    rs = [c["similarity"] for c in real if c["similarity"] is not None]
    rl = [c["length_ratio"] for c in real if c["length_ratio"] is not None]
    cal = dict(
        sim_mean=round(float(np.mean(sims)), 4),
        sim_iqr=[round(float(x), 4) for x in np.percentile(sims, [25, 75])],
        length_ratio_mean=round(float(np.mean(ratios)), 4),
        real_range=dict(similarity=[min(rs), max(rs)], length_ratio=[min(rl), max(rl)]),
    )
    cal["within_real_range"] = bool(
        min(rs) - 0.15 <= cal["sim_mean"] <= max(rs) and min(rl) - 0.4 <= cal["length_ratio_mean"] <= max(rl) + 0.1
    )
    (HERE / "results" / "sim_editor_calibration.json").write_text(json.dumps(cal, indent=1))
    print("notes", len(notes), "edit pairs", len(pairs_src), cal)


if __name__ == "__main__":
    main()
