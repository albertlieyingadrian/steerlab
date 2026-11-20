"""Generate drafts with an open model on Modal and save them to data/drafts/<slug>.jsonl.

python run_open.py MODEL SPLIT [--authors N] [--briefs N] [--arms k0,k8_rand] [--base] [--tag TAG]
"""

import argparse
import json
import time

import modal

from common import DATA, record_spend, spent
from gen import ARMS, base_prompt, chat_messages, jobs, load_aux
from modal_gen import MODAL_CAP, charge

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("split")
ap.add_argument("--authors", type=int)
ap.add_argument("--briefs", type=int, default=4)
ap.add_argument("--arms", default=",".join(ARMS))
ap.add_argument("--base", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--template", default="v1")
a = ap.parse_args()
if spent("modal") + 15 > MODAL_CAP:
    raise SystemExit(f"Modal cap: spent {spent('modal'):.2f} of {MODAL_CAP}")
J = jobs(a.split, a.authors, a.briefs, a.arms.split(","))
notes, edits = load_aux()
prompts = [
    base_prompt(j["rec"], j["brief"], j["arm"], a.seed)
    if a.base
    else chat_messages(j["rec"], j["brief"], j["arm"], a.seed, notes, edits, a.template)
    for j in J
]
fn = modal.Function.from_name("sounds-like-me", "generate")
t0 = time.time()
r = fn.remote(a.model, prompts, not a.base, 400, 0.7, a.seed)
usd = charge(time.time() - t0)  # charge local wall time (covers cold start + load), conservative
record_spend("modal", usd, f"gen {a.model} {a.split} n={len(J)}")
slug = a.model.split("/")[-1] + (f"-{a.tag}" if a.tag else "")
out = DATA / "drafts" / f"{slug}__{a.split}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as fh:
    for j, t in zip(J, r["outputs"], strict=False):
        fh.write(
            json.dumps(
                dict(
                    job=j["job"],
                    model=slug,
                    author=j["rec"]["author"],
                    brief=j["brief"]["id"],
                    arm=j["arm"],
                    text=t.strip(),
                    usd=usd / len(J),
                )
            )
            + "\n"
        )
print(f"{slug}: {len(J)} drafts, {r['seconds']} s on GPU, ${usd:.2f}; modal total ${spent('modal'):.2f}")
