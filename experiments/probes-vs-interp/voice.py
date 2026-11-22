"""M4b/M4c, tests 9 and 10: does a per-author steering vector give a model someone's voice as well as a LoRA or
few-shot prompting, and at what cost per user?

    python voice.py tune       # dev authors only: LoRA learning rate x steps, chosen at NLL rise <= 0.3 -> ledger
    python voice.py run        # 12 held-out authors x 4 briefs x every condition -> data/voice.json
    python voice.py score      # LUAR voice position per author and condition -> results/voice.json

Authors and briefs are Sounds Like Me Episode 1's (read only). Voice = Sounds Like Me's LUAR position, strict mode
(greeting and sign-off stripped, whitespace collapsed): 0 = a different human, 1 = the author's own held-out emails,
over a 4-draft episode. The vector recipe and its layer/strength are Part C's dev choice, not retuned here.
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache

import budget
import numpy as np
from common import (
    DATA,
    SEED,
    SLM,
    boot_diff,
    boot_mean,
    ledger,
    load_json,
    read_jsonl,
    save_json,
    slm_authors,
    slm_briefs,
    slm_texts,
    strip_greeting_signoff,
)

LAYER, STRENGTH = (
    23,
    0.4,
)  # Part C's "layer 24" in hidden_states indexing = output of decoder layer 23
N_AUTHORS = 12
VEC_NS, SHOT_NS, LORA_NS = [3, 6, 8, 12], [3, 6, 8], [3, 6, 12]
RAND_SEEDS = [11, 12]
MAX_NLL_RISE = 0.3
MAX_COPY, MIN_ON_TASK = (
    0.05,
    0.75,
)  # amendment 1 (2026-11-20, dev only): copy guard and on-task check
LORA_GRID = [
    dict(lr=1e-4, steps=40, r=8),
    dict(lr=2e-4, steps=60, r=8),
    dict(lr=3e-4, steps=90, r=8),
]

SYSTEM = (
    "You are ghostwriting an email for a specific person. Match their voice: how they actually write "
    "(length, greeting, sign-off, punctuation, tone), not a polished version. Use only the facts in the brief: "
    "do not invent results, clients, numbers, or any past contact. Plain text, 60 to 200 words. "
    "Output only the email."
)


def brief_block(b):  # Sounds Like Me gen.py, unchanged
    p = b["prospect"]
    facts = "\n".join(f"- {f}" for f in b["facts"])
    kind = "a follow-up email" if b["situation"] == "follow_up" else "an email"
    return (
        f"Write {kind} from a {b['sender']['role']} at {b['sender']['company']} to {p['first_name']}, "
        f"the {p['role']} at {p['company']} ({p['location']}).\n"
        f"Why now: {b['trigger']}.\nWhat to offer: {b['offer']}.\nAsk for: {b['cta']}.\n"
        f"Facts you may use (nothing else):\n{facts}"
    )


def k0_messages(b):
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": brief_block(b) + "\n\nYou know nothing else about this person's writing.",
        },
    ]


def shot_messages(b, samples):  # Sounds Like Me template v3 (the one Part C's k8 arm used)
    joined = "\n\n---\n\n".join(samples)
    user = (
        f"These are emails this person sent:\n\n{joined}\n\n---\n\n"
        "Write the next email they send. Same person, same habits, new topic:\n\n" + brief_block(b)
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def author_briefs(
    i,
):  # Sounds Like Me gen.jobs: 4 briefs per author, balanced across the three verticals
    B = slm_briefs()
    verts = ["freight", "industrial", "construction"]
    by_v = {v: [b for b in B if b["vertical"] == v] for v in verts}
    order = [*verts, verts[i % 3], verts[(i + 1) % 3]]
    return [by_v[v][(i * 3 + j) % 10] for j, v in enumerate(order[:4])]


def payload(rec, i, recs, conditions, lora, vec_ns, lora_ns):
    samples = slm_texts(rec, "samples")
    wrong = recs[(recs.index(rec) + 7) % len(recs)]
    default = [r["text"] for r in read_jsonl(SLM / "data" / "drafts" / "Qwen3-8B-pilot__dev.jsonl") if r["arm"] == "k0"]
    briefs = author_briefs(i)
    return dict(
        name=rec["author"],
        samples=samples,
        default_texts=default,
        wrong_samples=slm_texts(wrong, "samples"),
        k0_messages=[k0_messages(b) for b in briefs],
        shot_messages={str(n): [shot_messages(b, samples[:n]) for b in briefs] for n in SHOT_NS},
        conditions=conditions,
        vec_ns=vec_ns,
        lora_ns=lora_ns,
        layer=LAYER,
        strength=STRENGTH,
        lora=lora,
        rand_seeds=RAND_SEEDS,
        max_new_tokens=320,
        seed=SEED % 1000,
    )


# ---------------------------------------------------------------- voice ruler (Sounds Like Me's LUAR VoiceScale)
@lru_cache(None)
def _luar():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD")
    return (
        tok,
        AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True).eval(),
        torch,
    )


def luar_embed(episodes, max_len=256):
    tok, model, torch = _luar()
    out = []
    for ep in episodes:
        enc = tok(
            list(ep),
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.reshape(1, len(ep), -1) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            e = model(**enc)
        e = e[0] if isinstance(e, tuple) else e
        out.append(torch.nn.functional.normalize(e, dim=-1)[0].numpy())
    return np.stack(out)


class VoiceScale:
    """0 = a different human's emails, 1 = the author's own held-out emails (strict: format-blind)."""

    def __init__(self, rec, floor_recs, n_floor=30):
        self.f = lambda t: " ".join(strip_greeting_signoff(t).split())
        self.profile = luar_embed([[self.f(t) for t in slm_texts(rec, "profile")]])[0]
        ceil = [self.f(t) for t in slm_texts(rec, "ceiling")]
        self.ceiling = float(np.mean(luar_embed([ceil[:i] + ceil[i + 1 :] for i in range(len(ceil))]) @ self.profile))
        flo = [
            [self.f(t) for t in slm_texts(r, "ceiling")[:4]]
            for r in floor_recs[:n_floor]
            if r["author"] != rec["author"]
        ]
        self.floor = float(np.mean(luar_embed(flo) @ self.profile))

    def position(self, episode):
        sim = float(luar_embed([[self.f(t) for t in episode]])[0] @ self.profile)
        return (sim - self.floor) / max(self.ceiling - self.floor, 1e-6)


def copy_share(draft, samples, n=8):
    """Share of the draft's word 8-grams that appear verbatim in the author's samples (the copy hack)."""

    def grams(t):
        w = t.lower().split()
        return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}

    g = grams(draft)
    pool = set().union(*[grams(x) for x in samples]) if samples else set()
    return len(g & pool) / len(g) if g else 0.0


def on_task(draft, b):
    """The draft names the brief's prospect (first name) or their company."""
    low = draft.lower()
    return b["prospect"]["first_name"].lower() in low or b["prospect"]["company"].lower() in low


def guards(rec, i, outputs):
    samples = slm_texts(rec, "samples")
    briefs = author_briefs(i)
    return dict(
        copy=float(np.mean([copy_share(t, samples) for t in outputs])),
        on_task=float(np.mean([on_task(t, b) for t, b in zip(outputs, briefs, strict=False)])),
    )


def _call(payloads, job, projected):
    import modal

    f = modal.Function.from_name("tl-ep1-voice", "author")
    return budget.paid(job, "H100", projected, lambda: list(f.map(payloads)))


def _nll(xs):
    return float(np.nanmean(np.array([np.nan if x is None else x for x in xs], float)))


def tune():
    A = slm_authors()
    dev = [a for a in A["authors"] if a["split"] == "dev"][:3]
    out = {}
    for k, lora in enumerate(LORA_GRID):
        res = _call(
            [payload(r, i, dev, ["none"], lora, [], [12]) for i, r in enumerate(dev)],
            f"voice-tune-{k}",
            1.5,
        )
        pos, rise, cp, ot = [], [], [], []
        for i, (r, rec) in enumerate(zip(res, dev, strict=False)):
            vs = VoiceScale(rec, A["floor"])
            pos.append(vs.position(r["outputs"]["lora12"]) - vs.position(r["outputs"]["none"]))
            rise.append(_nll(r["nll"]["lora12"]) - _nll(r["nll"]["none"]))
            g = guards(rec, i, r["outputs"]["lora12"])
            cp.append(g["copy"])
            ot.append(g["on_task"])
        out[k] = dict(
            lora=lora,
            gain=float(np.mean(pos)),
            nll_rise=float(np.mean(rise)),
            copy=float(np.mean(cp)),
            on_task=float(np.mean(ot)),
            example=redact(res[0]["outputs"]["lora12"][0][:600], res[0]["name"]),
        )
        ledger(
            f"voice-lora-{lora['lr']}-{lora['steps']}",
            "dev_voice_gain",
            out[k]["gain"],
            0.0,
            "grid",
            f"nll rise {out[k]['nll_rise']:.3f}; copy {out[k]['copy']:.3f}; on-task {out[k]['on_task']:.2f}; "
            "3 dev authors, n=12",
        )
        print({k_: v for k_, v in out[k].items() if k_ != "example"}, flush=True)
    ok = [
        v
        for v in out.values()
        if v["nll_rise"] <= MAX_NLL_RISE and v["copy"] <= MAX_COPY and v["on_task"] >= MIN_ON_TASK
    ]
    choice = max(ok, key=lambda v: v["gain"]) if ok else min(out.values(), key=lambda v: v["nll_rise"])
    ledger(
        "voice-lora-choice",
        "dev_voice_gain",
        choice["gain"],
        0.0,
        "kept",
        json.dumps(choice["lora"]),
    )
    save_json("voice_tune.json", dict(grid=out, choice=choice))


def run():
    A = slm_authors()
    held = [a for a in A["authors"] if a["split"] == "heldout"][:N_AUTHORS]
    lora = load_json("voice_tune.json")["choice"]["lora"]
    conds = (
        ["none"]
        + [f"vec{n}" for n in VEC_NS]
        + [f"shot{n}" for n in SHOT_NS]
        + ["wrong"]
        + [f"rand{s}" for s in RAND_SEEDS]
        + [f"lora{n}" for n in LORA_NS]
    )
    res = _call(
        [payload(r, i, held, conds, lora, VEC_NS, LORA_NS) for i, r in enumerate(held)],
        "voice-heldout",
        8.0,
    )
    (DATA / "voice.json").write_text(json.dumps(res))
    print("voice authors:", len(res))


def redact(text, author):
    """A public example draft: the writer's name parts (3+ letters) become [name], and emails, phone numbers and
    addresses go too (Sounds Like Me's privacy.py); drafts copy them from the writer's samples."""
    sys.path.append(str(SLM))
    from privacy import redact_contact

    parts = [p for p in re.split(r"\W+", author) if len(p) >= 3 and not p.isdigit()]
    for p in sorted(parts, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(p)}\b", "[name]", text, flags=re.IGNORECASE)
    tokens = [t for t in re.split(r"[\s.,]+", author) if t and t[0].isalpha()]
    for initials in {"".join(p[0] for p in parts if p[0].isupper()), "".join(t[0].upper() for t in tokens)}:
        if len(initials) >= 2:  # sign-offs made of the writer's initials, with or without a middle initial
            text = re.sub(rf"\b{re.escape(initials)}\b", "[name]", text, flags=re.IGNORECASE)
    return redact_contact(text)


def score():
    A = slm_authors()
    recs = {a["author"]: a for a in A["authors"]}
    res = json.loads((DATA / "voice.json").read_text())
    held = [a["author"] for a in A["authors"] if a["split"] == "heldout"][:N_AUTHORS]
    per = {}
    for r in res:
        vs = VoiceScale(recs[r["name"]], A["floor"])
        i = held.index(r["name"])
        per[r["name"]] = {
            c: dict(
                position=vs.position([t for t in ts if t.strip()] or ["."]),
                nll=_nll(r["nll"][c]),
                **guards(recs[r["name"]], i, ts),
            )
            for c, ts in r["outputs"].items()
        }
    conds = list(next(iter(per.values())))
    summary = {
        c: dict(
            position=boot_mean([per[a][c]["position"] for a in per]),
            gain_vs_none=boot_diff(
                [per[a][c]["position"] for a in per],
                [per[a]["none"]["position"] for a in per],
            ),
            nll_rise=round(
                float(np.mean([per[a][c]["nll"] - per[a]["none"]["nll"] for a in per])),
                4,
            ),
            copy=round(float(np.mean([per[a][c]["copy"] for a in per])), 4),
            on_task=round(float(np.mean([per[a][c]["on_task"] for a in per])), 4),
        )
        for c in conds
    }
    build = {k: float(np.mean([r["build_sec"][k] for r in res])) for k in res[0]["build_sec"]}
    g_vec, g_lora = (
        summary["vec12"]["gain_vs_none"]["mean"],
        summary["lora12"]["gain_vs_none"]["mean"],
    )
    ratio_gain = g_vec / g_lora if g_lora > 0 else float("nan")
    ratio_cost = build["vec12"] / build["lora12"]
    fluent = summary["vec12"]["nll_rise"] <= MAX_NLL_RISE and summary["lora12"]["nll_rise"] <= MAX_NLL_RISE
    p8 = bool(ratio_gain >= 0.5 and ratio_cost <= 0.01 and fluent)
    d9 = boot_diff(
        [per[a]["vec8"]["position"] for a in per],
        [per[a]["shot8"]["position"] for a in per],
    )
    save_json(
        "voice.json",
        dict(
            summary=summary,
            per_author=per,
            build_sec=build,
            bytes=dict(vector=res[0]["vector_bytes"], lora=res[0]["adapter_bytes"]),
            p8=dict(gain_ratio=ratio_gain, cost_ratio=ratio_cost, fluent=fluent, hit=p8),
            p9=dict(diff_vec8_minus_shot8=d9, hit=bool(d9["mean"] > 0.05)),
            n_authors=len(per),
            examples={c: redact(res[0]["outputs"][c][0], res[0]["name"]) for c in ("none", "vec12", "lora12", "shot8")},
        ),
    )
    for c in conds:
        s = summary[c]
        print(
            f"{c:8s} pos {s['position']['mean']:+.3f} gain {s['gain_vs_none']['mean']:+.3f} "
            f"[{s['gain_vs_none']['lo']:+.3f},{s['gain_vs_none']['hi']:+.3f}] nll+ {s['nll_rise']:+.3f} "
            f"copy {s['copy']:.3f} on-task {s['on_task']:.2f}"
        )
    print("build sec", build, "P8", p8, ratio_gain, ratio_cost, "P9", d9)


if __name__ == "__main__" and sys.argv[1] != "explore":
    dict(tune=tune, run=run, score=score)[sys.argv[1]]()


# ---------------------------------------------------------------- EXPLORATORY (post hoc, never used to rescore P8/P9)
# The pre-registered fluency rule (NLL rise <= 0.3) passed drafts that loop: 73-81% of vector drafts repeat themselves.
# This reruns the vector arms at a strength chosen on dev with a loop cap, to ask whether a non-looping vector still
# beats prompting. Reported as exploratory in the note.
EXPLORE_STRENGTHS = [0.05, 0.1, 0.2, 0.3]
MAX_LOOPING = 0.05


def looping(t):
    w = t.split()
    grams = [" ".join(w[i : i + 4]) for i in range(len(w) - 3)]
    return (1 - len(set(grams)) / max(len(grams), 1)) > 0.2


def _with_strength(p, s):
    return dict(p, strength=s)


def explore():
    A = slm_authors()
    dev = [a for a in A["authors"] if a["split"] == "dev"][:5]
    conds = ["none", "vec12"]
    ps = [
        _with_strength(payload(r, i, dev, conds, dict(lr=1e-4, steps=40, r=8), [12], []), s)
        for s in EXPLORE_STRENGTHS
        for i, r in enumerate(dev)
    ]
    res = _call(ps, "voice-explore-dev", 3.0)
    grid = {}
    for k, s in enumerate(EXPLORE_STRENGTHS):
        part = res[k * len(dev) : (k + 1) * len(dev)]
        pos, loop = [], []
        for r, rec in zip(part, dev, strict=False):
            vs = VoiceScale(rec, A["floor"])
            pos.append(vs.position(r["outputs"]["vec12"]) - vs.position(r["outputs"]["none"]))
            loop += [looping(t) for t in r["outputs"]["vec12"]]
        grid[s] = dict(gain=float(np.mean(pos)), looping=float(np.mean(loop)))
        ledger(
            f"voice-explore-s{s}",
            "dev_voice_gain",
            grid[s]["gain"],
            0.0,
            "explore",
            f"looping {grid[s]['looping']:.2f}; 5 dev authors; post hoc",
        )
        print(s, grid[s], flush=True)
    ok = [s for s, v in grid.items() if v["looping"] <= MAX_LOOPING]
    choice = max(ok, key=lambda s: grid[s]["gain"]) if ok else min(grid, key=lambda s: grid[s]["looping"])
    held = [a for a in A["authors"] if a["split"] == "heldout"][:N_AUTHORS]
    conds = ["none", "vec8", "vec12", "wrong", f"rand{RAND_SEEDS[0]}"]
    res = _call(
        [
            _with_strength(
                payload(r, i, held, conds, dict(lr=1e-4, steps=40, r=8), [8, 12], []),
                choice,
            )
            for i, r in enumerate(held)
        ],
        "voice-explore-heldout",
        3.0,
    )
    base = json.loads((DATA / "voice.json").read_text())
    recs = {a["author"]: a for a in A["authors"]}
    per = {}
    for r, b in zip(res, base, strict=False):
        vs = VoiceScale(recs[r["name"]], A["floor"])
        per[r["name"]] = {
            c: dict(
                position=vs.position([t for t in ts if t.strip()] or ["."]),
                looping=float(np.mean([looping(t) for t in ts])),
            )
            for c, ts in [
                *list(r["outputs"].items()),
                ("shot8", b["outputs"]["shot8"]),
                ("lora12", b["outputs"]["lora12"]),
            ]
        }
    summ = {
        c: dict(
            gain_vs_none=boot_diff(
                [per[a][c]["position"] for a in per],
                [per[a]["none"]["position"] for a in per],
            ),
            looping=round(float(np.mean([per[a][c]["looping"] for a in per])), 3),
        )
        for c in next(iter(per.values()))
    }
    save_json(
        "voice_explore.json",
        dict(
            note="exploratory, post hoc; strength chosen on dev with looping <= 0.05",
            dev_grid=grid,
            strength=choice,
            summary=summ,
            vec8_minus_shot8=boot_diff(
                [per[a]["vec8"]["position"] for a in per],
                [per[a]["shot8"]["position"] for a in per],
            ),
            own_minus_wrong=boot_diff(
                [per[a]["vec12"]["position"] for a in per],
                [per[a]["wrong"]["position"] for a in per],
            ),
            example=redact(res[0]["outputs"]["vec12"][0], res[0]["name"]),
        ),
    )
    for c, v in summ.items():
        print(c, v)


if __name__ == "__main__" and sys.argv[1] == "explore":
    explore()
