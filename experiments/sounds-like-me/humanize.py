"""Humanizer contestants (P3): rewrite a source model's k0 held-out drafts to "sound human", then score them
like any contestant. Saved as data/drafts/humanizer-<kind>__heldout.jsonl with model="humanizer-<kind>:<source>",
arm="k0", so analyze.py compares each humanizer with its own source draft.

    python humanize.py SOURCE_SLUG            # e.g. gemini-4-carbon
Kinds: prompt (Qwen3-32B told to make it read as human-written and avoid AI detection, keeping every fact),
       dipper (DIPPER-XXL paraphrase attack, lexical 60).
"""

import json
import sys
import time

import modal

from common import DATA, record_spend
from modal_gen import charge

PROMPT = (
    "Rewrite this email so it reads as if a person wrote it, not an AI, and would pass an AI detector. "
    "Keep every fact, name and request; do not add anything. Output only the email.\n\nEmail:\n{t}"
)

src = sys.argv[1]
rows = [json.loads(l) for l in (DATA / "drafts" / f"{src}__heldout.jsonl").read_text().splitlines() if l.strip()]
rows = [r for r in rows if r["arm"] == "k0" and r["text"].strip()]
gen = modal.Function.from_name("sounds-like-me", "generate")
dip = modal.Function.from_name("sounds-like-me", "dipper")
for kind in ("prompt", "dipper"):
    t0 = time.time()
    if kind == "prompt":
        outs = gen.remote(
            "Qwen/Qwen3-32B",
            [[{"role": "user", "content": PROMPT.format(t=r["text"])}] for r in rows],
            True,
            500,
            0.7,
            3,
        )["outputs"]
    else:
        outs = dip.remote([r["text"] for r in rows], 60, 0)["outputs"]
    usd = charge(time.time() - t0)
    record_spend("modal", usd, f"humanizer {kind} on {src} n={len(rows)}")
    out = DATA / "drafts" / f"humanizer-{kind}__heldout.jsonl"
    with out.open("w") as fh:
        for r, t in zip(rows, outs, strict=False):
            fh.write(
                json.dumps(
                    dict(
                        r,
                        job=r["job"] + "|hum-" + kind,
                        model=f"humanizer-{kind}:{src}",
                        text=t.strip(),
                        usd=usd / len(rows),
                    )
                )
                + "\n"
            )
    print(kind, len(rows), f"${usd:.2f}")
