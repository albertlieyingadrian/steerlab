"""The explorer's data: a few held-out briefs, one page per arm (base and each seed-0 run's final checkpoint), with
its screenshot and every score. Writes report/pages/*.jpg and report/data.js (laptop; reads the local cache).

    .venv/bin/python experiments/brand-rl/make_demo.py configs/main.json

Briefs: the first two held-out briefs of each page type (10). Runs: the untrained model and every run's final
checkpoint, seeds apart. Page: the first of the four samples, so nothing is
picked by its score. Screenshots are the first screen at 1280 x 800, shrunk to 640 px wide. The pages hold only the
invented brief text and the policy's own markup.
"""

from __future__ import annotations

import gzip
import io
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ep6
from analyze import base_arm, hack_flag
from reward import reward

OUT = HERE / "report"
ARM_LABEL = {
    "base": "Untrained model",
    "check": "CHECK (checker only)",
    "env": "ENV (full environment)",
    "team": "TEAM (+ novelty)",
    "shuf": "SHUF (shuffled reward)",
}


def load(p):
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def main() -> None:
    from PIL import Image

    cfg = json.loads((HERE / sys.argv[1]).read_text())
    runs = [{**cfg["common"], **r} for r in cfg["runs"]]
    S = json.loads((ep6.RESULTS / "summary.json").read_text())
    thr = S["hack_threshold_base_dev_median"]
    briefs = ep6.briefs("heldout")
    pick = []
    for pt in ep6.PAGE_TYPES:
        pick += [b for b in briefs if b["page_type"] == pt][:2]
    sets = [("base", "base", "base__heldout")] + [
        (r["name"], r["arm"], f"{r['name']}__step{r['steps']}__heldout") for r in runs
    ]
    spec = ep6.load_spec()
    lines = {s["id"]: s["text"] for s in spec["source"]}
    fam_line = {
        f: " / ".join(lines[ln] for ln in next(r for r in spec["rules"] if r["id"] == rs[0])["lines"])
        for f, rs in ep6.FAMILIES.items()
    }
    split = ep6.family_split()
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    labels = {k: ARM_LABEL[a] if k == "base" else f"{ARM_LABEL[a]}, seed {k[-1]}" for k, a, _ in sets}
    data = {
        "briefs": [],
        "arms": [{"key": k, "label": labels[k]} for k, _, _ in sets],
        "pages": {},
        "families": [
            {"id": f, "line": fam_line[f], "held_out": f in split["heldout"]}
            for f in split["reward"] + split["heldout"]
        ],
    }
    for b in pick:
        data["briefs"].append(
            {
                "id": b["id"],
                "type": b["page_type"],
                "service": b["service"],
                "domain": b["domain"],
                "items": [{"kind": it["kind"], "variant": it.get("variant"), "text": it["text"]} for it in b["items"]],
            }
        )
    for key, arm, tag in sets:
        d = load(ep6.CACHE / "samples" / f"{tag}.pkl.gz")
        sig = load(ep6.CACHE / "samples" / f"{tag}.sig.pkl.gz")
        first: dict = {}
        for i, r in enumerate(d["rows"]):
            first.setdefault(r["brief"], i)
        a_rw = "env" if arm == "base" else base_arm(arm)
        a_hk = "check" if arm == "base" else arm
        for b in pick:
            i = first[b["id"]]
            r = d["rows"][i]
            name = f"{key}_{b['id']}.jpg"
            if d["jpeg"][i]:
                im = Image.open(io.BytesIO(d["jpeg"][i])).convert("RGB")
                im.resize((640, 400)).save(OUT / "pages" / name, quality=78)
            data["pages"][f"{key}|{b['id']}"] = {
                "img": f"pages/{name}" if d["jpeg"][i] else None,
                "valid": bool(r["valid"]),
                "truncated": bool(r["truncated"]),
                "reward": round(reward(a_rw, r), 3),
                "check_reward": round(reward("check", r), 3),
                "fam": r.get("fam") or {},
                "coverage": r.get("coverage"),
                "fidelity_strict": r.get("fidelity_strict"),
                "fidelity_loose": r.get("fidelity_loose"),
                "slop_p": round(r["slop_p"], 3) if r.get("slop_p") is not None else None,
                "shot_p": round(sig["shot_p"][i], 3) if sig["shot_p"][i] is not None else None,
                "copy_ex": round(r["copy_structure_ex"], 3) if r.get("copy_structure_ex") is not None else None,
                "hack": hack_flag(r, a_hk, thr[base_arm(a_hk)]),
                "tokens": r.get("n_tokens"),
            }
    (OUT / "data.js").write_text("window.EP6 = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print("wrote report/data.js:", len(data["pages"]), "pages")


if __name__ == "__main__":
    main()
