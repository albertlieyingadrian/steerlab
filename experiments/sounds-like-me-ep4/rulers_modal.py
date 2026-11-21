"""Modal rulers: Episode 2's preference model (retrained, it was never saved) and the open-model LLM panel.

    python rulers_modal.py pref            # retrain on Episode 2's exact pairs + seed, score every Episode 4 item
    python rulers_modal.py panel [--pilot] # Qwen3-32B, Gemma-3-27B-it, Llama-3.1-8B-Instruct (comparison arm)

COSTS MONEY (Modal H100). Every call is checked against EP4_MODAL_CAP and charged to the series ledger.
The panel is a comparison arm: it is scored against the humans like any other ruler, never used as a human label.
"""

import json
import random
import sys
import time

import numpy as np

from common4 import EP2, LOCAL, RES, charge_modal, dump, modal_guard, read_jsonl, strict

sys.path.append(str(EP2))
from common import authors, texts  # Episode 1
from score_ep2 import PREF_EPOCHS, PREF_LR, _ctx, pref_eval_sets, pref_pairs  # Episode 2's recipe, unchanged

PANEL = {
    "qwen3-32b": "Qwen/Qwen3-32B",
    "gemma-3-27b": "google/gemma-3-27b-it",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}
DIGITS = ["1", "2", "3", "4", "5"]


def load():
    L1 = json.loads((RES / "labels_h1.json").read_text())
    T1 = {r["tid"]: r["text"] for r in read_jsonl(LOCAL / "h1_texts.jsonl")}
    L2 = json.loads((RES / "labels_h2.json").read_text())
    X2 = {r["eid"]: r["text"] for r in read_jsonl(LOCAL / "h2_excerpts.jsonl")}
    L3 = json.loads((RES / "labels_h3.json").read_text())
    T3 = {r["aid"]: r["text"] for r in read_jsonl(LOCAL / "h3_texts.jsonl")}
    U = json.loads((LOCAL / "user_pairs.json").read_text())
    return L1, T1, L2, X2, L3, T3, U


def h1_refs(L1):
    refs = {}
    for r in L1:
        if r["kind"] == "control":
            refs.setdefault(r["pid"], []).append(r["tid"])
    return refs


def h1_known_answer(refs):
    """Known-answer pairs (plan, Design 3), built the way the rulers are used in H1: one reference set, two candidates.
    Reference = a writer's other unassisted text(s); candidate "own" = this unassisted text; candidate "other" = a
    seeded other writer's unassisted text. A ruler passes if it scores "own" higher.
    (A first version held the candidate fixed and swapped the reference; a cross-encoder's scores are not comparable
    across contexts, so that check read 0.20 for the preference model. Fixed before any confirmatory run.)"""
    rnd = random.Random(7)
    pids = sorted(refs)
    out = []
    for p in pids:
        for t in refs[p]:
            ref = [u for u in refs[p] if u != t]
            if ref:
                q = rnd.choice([x for x in pids if x != p])
                out.append(dict(ref=ref, own=t, other=rnd.choice(refs[q])))
    return out


def user_refs(U):
    held = {a["author"]: a for a in authors()["authors"] if a["split"] == "heldout"}
    return {n: texts(held[n], "profile") for n in {p["author"] for p in U}}


# ---------------------------------------------------------------- preference model
def pref():
    import modal

    L1, T1, L2, X2, L3, T3, U = load()
    refs = h1_refs(L1)
    a_pairs, _ = pref_eval_sets()
    sets = {
        "a_win": [dict(ctx=x["ctx"], text=x["win"]) for x in a_pairs],
        "a_lose": [dict(ctx=x["ctx"], text=x["lose"]) for x in a_pairs],
    }
    h1_ids, h1_items = [], []
    for r in L1:
        own = [u for u in refs.get(r["pid"], []) if u != r["tid"]]
        if own:
            h1_ids.append(r["tid"])
            h1_items.append(dict(ctx=_ctx([T1[u] for u in own][:2]), text=strict(T1[r["tid"]])))
    ka = h1_known_answer(refs)
    sets["h1"] = h1_items
    sets["h1_ka_own"] = [dict(ctx=_ctx([T1[u] for u in k["ref"]][:2]), text=strict(T1[k["own"]])) for k in ka]
    sets["h1_ka_other"] = [dict(ctx=_ctx([T1[u] for u in k["ref"]][:2]), text=strict(T1[k["other"]])) for k in ka]
    h2 = [t for t in L2 if t["n"] >= 5 or t["golden"]]
    sets["h2_b"] = [dict(ctx=_ctx([X2[t["a"]]]), text=strict(X2[t["b"]])) for t in h2]
    sets["h2_c"] = [dict(ctx=_ctx([X2[t["a"]]]), text=strict(X2[t["c"]])) for t in h2]
    # H3 has no writer: three fixed neutral contexts (two emails each from three Episode 1 dev writers), averaged
    dev = [a for a in authors()["authors"] if a["split"] == "dev"][:3]
    for i, a in enumerate(dev):
        sets[f"h3_{i}"] = [dict(ctx=_ctx(texts(a, "ceiling")[:2]), text=strict(T3[r["aid"]])) for r in L3]
    UR = user_refs(U)
    u_ids = [f"{p['pair']}:{s}" for p in U for s in ("a", "b")]
    sets["user"] = [dict(ctx=_ctx(UR[p["author"]][:2]), text=strict(p[s])) for p in U for s in ("a", "b")]
    n = sum(len(v) for v in sets.values())
    modal_guard(1.0)
    t0 = time.time()
    payload = dict(train=pref_pairs(), score=sets, epochs=PREF_EPOCHS, lr=PREF_LR, max_len=512)
    r = modal.Function.from_name("slm-ep2", "train_pref").remote(payload)
    usd = charge_modal(time.time() - t0, f"preference model retrain + score {n} items")
    S = r["scores"]
    acc_a = float(np.mean(np.array(S["a_win"]) > np.array(S["a_lose"])))
    ka_acc = float(np.mean(np.array(S["h1_ka_own"]) > np.array(S["h1_ka_other"])))
    out = dict(
        check=dict(
            ep2_heldout_acc_a=round(acc_a, 4),
            pass_bar=0.75,
            passed=acc_a >= 0.75,
            n_a=len(a_pairs),
            h1_own_vs_other=round(ka_acc, 4),
            n_h1_ka=len(ka),
        ),
        h1=dict(zip(h1_ids, S["h1"])),
        h2={t["tri"]: [b, c] for t, b, c in zip(h2, S["h2_b"], S["h2_c"])},
        h3={x["aid"]: float(np.mean([S[f"h3_{i}"][j] for i in range(3)])) for j, x in enumerate(L3)},
        user=dict(zip(u_ids, S["user"])),
        loss_curve=r["loss_curve"],
        seconds=r["seconds"],
        usd=round(usd, 3),
    )
    dump(RES / "scores_pref.json", out)
    print(out["check"], f"${usd:.2f}")


# ---------------------------------------------------------------- LLM panel
def p_voice(refs, cand):
    blocks = "\n\n".join(f"<text{i + 1}>\n{t.strip()}\n</text{i + 1}>" for i, t in enumerate(refs))
    return (
        f"Here are {len(refs)} texts written by one person:\n\n{blocks}\n\nHere is a new text:\n\n"
        f"<new>\n{cand.strip()}\n</new>\n\n"
        "How well does the new text capture this person's own writing style: their word choice, sentence rhythm, "
        "punctuation and tone? Ignore what it is about. Answer with one digit from 1 (not at all their style) "
        "to 5 (exactly their style)."
    )


def p_triplet(a, b, c):
    return (
        f"Read three passages.\n\n<A>\n{a.strip()}\n</A>\n\n<B>\n{b.strip()}\n</B>\n\n<C>\n{c.strip()}\n</C>\n\n"
        "Which passage, B or C, is more similar in writing style to passage A? Style means word choice, sentence "
        "structure, figurative language and rhythm, not plot or topic. Answer with one letter: B or C."
    )


def p_ai(text):
    return (
        f"<article>\n{text.strip()}\n</article>\n\nWas this article written by a person or generated by an AI "
        "language model? Answer with one digit from 1 (certainly a person) to 5 (certainly AI)."
    )


def expected(lp, choices, values):
    """Expected answer value over the allowed answers found in the top-20; None if none were found."""
    xs = [(lp[c], v) for c, v in zip(choices, values) if c in lp]
    if not xs:
        return None
    m = max(x for x, _ in xs)
    w = np.array([np.exp(x - m) for x, _ in xs])
    return float((w / w.sum()) @ np.array([v for _, v in xs]))


def panel(pilot=False):
    import modal

    L1, T1, L2, X2, L3, T3, U = load()
    refs = h1_refs(L1)
    UR = user_refs(U)
    jobs = []  # (dataset, key, prompt)
    for r in L1:
        own = [u for u in refs.get(r["pid"], []) if u != r["tid"]]
        if own:
            jobs.append(("h1", r["tid"], p_voice([T1[u] for u in own], T1[r["tid"]])))
    for k in h1_known_answer(refs):
        jobs.append(("h1_ka_own", k["own"], p_voice([T1[u] for u in k["ref"]], T1[k["own"]])))
        jobs.append(("h1_ka_other", k["own"], p_voice([T1[u] for u in k["ref"]], T1[k["other"]])))
    for t in L2:
        if t.get("panel") or t["golden"]:
            A, Bx, Cx = X2[t["a"]], X2[t["b"]], X2[t["c"]]
            jobs.append(("h2_bc", t["tri"], p_triplet(A, Bx, Cx)))
            jobs.append(("h2_cb", t["tri"], p_triplet(A, Cx, Bx)))  # swapped: "B" here is the real C
    for r in L3:
        jobs.append(("h3", r["aid"], p_ai(T3[r["aid"]])))
    for p in U:
        for s in ("a", "b"):
            jobs.append(("user", f"{p['pair']}:{s}", p_voice(UR[p["author"]][:3], p[s])))
    if pilot:  # 20 per dataset, end to end
        cnt, keep = {}, []
        for j in jobs:
            if cnt.get(j[0], 0) < 20:
                keep.append(j)
                cnt[j[0]] = cnt.get(j[0], 0) + 1
        jobs = keep
    print(len(jobs), "prompts per model")
    modal_guard(1.5 * len(PANEL) * (0.3 if pilot else 1.0))
    payloads = [dict(model=m, prompts=[j[2] for j in jobs], choices=[*DIGITS, "B", "C"]) for m in PANEL.values()]
    t0 = time.time()
    res = list(modal.Function.from_name("slm-ep4", "judge").map(payloads))
    usd = charge_modal(
        sum(r["seconds"] + 60 for r in res), f"panel {'pilot ' if pilot else ''}{len(jobs)} prompts x {len(PANEL)}"
    )
    for slug, r in zip(PANEL, res):
        out = {"h1": {}, "h1_ka_own": {}, "h1_ka_other": {}, "h2": {}, "h3": {}, "user": {}}
        bc, cb = {}, {}
        missing = 0
        for (ds, key, _), lp in zip(jobs, r["logprobs"]):
            if ds.startswith("h2"):
                v = expected(lp, ["B", "C"], [1.0, 0.0])  # P(answer "B") in this order
                if v is None:
                    missing += 1
                    continue
                (bc if ds == "h2_bc" else cb)[key] = v
            else:
                v = expected(lp, DIGITS, [1, 2, 3, 4, 5])
                if v is None:
                    missing += 1
                    continue
                out[ds][key] = v
        # H2: P(the real B is closer) averaged over both orders; per-order values kept for the position check
        for k in set(bc) & set(cb):
            out["h2"][k] = dict(p_b=(bc[k] + (1 - cb[k])) / 2, order1=bc[k], order2_pb=1 - cb[k])
        out.update(
            model=PANEL[slug],
            n_prompts=len(jobs),
            n_missing=missing,
            truncated=r["truncated"],
            seconds=r["seconds"],
            pilot=pilot,
        )
        dump(RES / f"scores_panel_{slug}{'_pilot' if pilot else ''}.json", out)
        print(slug, "missing", missing, "truncated", r["truncated"], f"{r['seconds']:.0f}s")
    print(f"panel ${usd:.2f} ({time.time() - t0:.0f}s wall)")


if __name__ == "__main__":
    if sys.argv[1] == "pref":
        pref()
    elif sys.argv[1] == "panel":
        panel(pilot="--pilot" in sys.argv)
