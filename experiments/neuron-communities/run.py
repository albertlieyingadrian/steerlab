"""The GPU stages (plan: PLAN.md). Called by modal_run.py on one H100.

    e1   necessity of 61 communities vs 10 layer-matched random sets, on base and Instruct  -> e1_<model>.json
    e2a  screen: 35 topic communities x 4 clamp strengths x 4 questions, plus random sets  -> e2a.json(l)
    e2b  confirm: 8 finalists on 15 held-out questions: clamp, 3 random, direction, none   -> e2b.json(l)
    e3   collateral KL on 200 unrelated prompts, and the "an" off-switch                  -> e3.json(l)

Every reply is written in full to a .jsonl beside the summary.
"""

# ruff: noqa: E501, B905  (long data lines and pairwise zips read better as they are)

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import nc
import numpy as np

BATCH = 50


def _raw_batches(texts: list[str]) -> list[list[str]]:
    return [texts[i : i + BATCH] for i in range(0, len(texts), BATCH)]


def necessity(model: Any, tok: Any, texts: list[str], neuron_sets: list[list[tuple[int, int]]]) -> list[np.ndarray]:
    """KL(clean || set zeroed at the last position) per prompt, for each neuron set."""
    out: list[list[np.ndarray]] = [[] for _ in neuron_sets]
    for chunk in _raw_batches(texts):
        ids, mask, last = nc.encode_raw(tok, chunk)
        base = nc.last_logprobs(model, ids, mask, last)
        for k, ns in enumerate(neuron_sets):
            with nc.intervene(model, {"mode": "zero_last", "neurons": ns, "last": last}):
                lp = nc.last_logprobs(model, ids, mask, last)
            out[k].append(nc.kl(base, lp).cpu().numpy())
    return [np.concatenate(o) for o in out]


def e1(model: Any, tok: Any, tag: str, out_dir: Path) -> dict[str, Any]:
    from scipy.stats import spearmanr

    rows = []
    for cid in nc.selection()["e1"]:
        c = nc.community(cid)
        texts = [p["t"] for p in c["prompts"]]
        sets = [nc.members(c)] + [nc.random_set(c, s) for s in range(nc.N_RANDOM_E1)]
        kls = necessity(model, tok, texts, sets)
        med = [float(np.median(k)) for k in kls]
        theirs = np.array([p["n"] for p in c["prompts"]], dtype=float)
        rho = spearmanr(kls[0], theirs).statistic if len(texts) > 2 else float("nan")
        rows.append(
            {
                "id": cid,
                "class": c["class"],
                "name": c["name"],
                "n_members": len(sets[0]),
                "n_prompts": len(texts),
                "median_kl": round(med[0], 4),
                "random_medians": [round(m, 4) for m in med[1:]],
                "passes": bool(med[0] >= nc.NEC_TAU and med[0] > max(med[1:])),
                "their_median_n": round(float(np.median(theirs)), 4),
                "spearman_vs_theirs": round(float(rho), 4),
                "per_prompt_kl": [round(float(x), 4) for x in kls[0]],
            }
        )
        print(
            f"e1 {tag} {cid} {c['class']:<11} kl {med[0]:.3f} rand max {max(med[1:]):.3f} pass {rows[-1]['passes']}",
            flush=True,
        )
    k = sum(r["passes"] for r in rows)
    res = {"model": tag, "n": len(rows), "passes": k, "pass_ci": nc.wilson(k, len(rows)), "rows": rows}
    (out_dir / f"e1_{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def a_ref(model: Any, tok: Any, c: dict[str, Any], neurons: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
    """Mean activation at the last position over the community's prompts (raw text), sign kept."""
    acc: dict[tuple[int, int], list[float]] = {n: [] for n in neurons}
    for chunk in _raw_batches([p["t"] for p in c["prompts"]]):
        ids, mask, last = nc.encode_raw(tok, chunk)
        got = nc.capture(model, ids, mask, last, neurons)
        for n in neurons:
            acc[n] += got[n].tolist()
    return {n: float(np.mean(v)) for n, v in acc.items()}


def clamp_spec(
    neurons: list[tuple[int, int]], ref: dict[tuple[int, int], float], strengths: list[float]
) -> dict[str, Any]:
    import torch

    vals = {n: torch.tensor([s * ref[n] for s in strengths]) for n in neurons}
    return {"mode": "clamp_all", "neurons": neurons, "values": vals}


def score(reply: dict[str, Any], words: list[str]) -> dict[str, Any]:
    flu = nc.fluent(reply["text"])
    lp = nc.loops(reply["tokens"])
    hit = nc.hits(reply["text"], words)
    return {"hit": hit, "fluent": flu, "loops": lp, "good_hit": bool(hit and flu and not lp)}


def e2a(model: Any, tok: Any, out_dir: Path) -> dict[str, Any]:
    qs = nc.TUNE_QUESTIONS
    log = (out_dir / "e2a.jsonl").open("w")
    unsteered = nc.generate(model, tok, qs, None, 120)
    for q, r in zip(qs, unsteered):
        log.write(json.dumps({"arm": "none", "q": q, "text": r["text"]}) + "\n")
    summary = []
    for cid in nc.selection()["e2"]:
        c = nc.community(cid)
        words = nc.topic_words(c)
        if not words:
            continue
        mem, rnd = nc.members(c), nc.random_set(c, 0)
        ref = a_ref(model, tok, c, mem)
        ref_r = a_ref(model, tok, c, rnd)
        strengths = [s for s in nc.STRENGTHS for _ in qs]
        row: dict[str, Any] = {
            "id": cid,
            "label": nc.topic_label(c),
            "words": words,
            "none_hits": sum(score(r, words)["good_hit"] for r in unsteered),
        }
        for arm, ns, rf in (("community", mem, ref), ("random", rnd, ref_r)):
            replies = nc.generate(model, tok, qs * len(nc.STRENGTHS), clamp_spec(ns, rf, strengths), 120)
            per: dict[float, list[dict[str, Any]]] = {}
            for s, q, r in zip(strengths, [q for _ in nc.STRENGTHS for q in qs], replies):
                sc = score(r, words)
                per.setdefault(s, []).append(sc)
                log.write(json.dumps({"arm": arm, "id": cid, "c": s, "q": q, "text": r["text"], **sc}) + "\n")
            row[arm] = {
                str(s): {
                    "good_hits": sum(x["good_hit"] for x in v),
                    "hits": sum(x["hit"] for x in v),
                    "fluent": sum(x["fluent"] for x in v),
                }
                for s, v in per.items()
            }
        best = max(nc.STRENGTHS, key=lambda s: (row["community"][str(s)]["good_hits"], -s))
        row["best_c"], row["best_good_hits"] = best, row["community"][str(best)]["good_hits"]
        summary.append(row)
        print(
            f"e2a {cid} {row['label'][:24]:<24} best c={best} good {row['best_good_hits']}/4 | {row['community']}",
            flush=True,
        )
    log.close()
    finalists = sorted(summary, key=lambda r: (-r["best_good_hits"], int(r["id"])))[:8]
    res = {"rows": summary, "finalists": [{"id": r["id"], "c": r["best_c"]} for r in finalists]}
    (out_dir / "e2a.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def residual_at(
    model: Any, ids: Any, mask: Any, layer: int, last: Any | None, spec: dict[str, Any] | None = None
) -> Any:
    """Residual stream after `layer` (rows' last position if `last` is given, else every unmasked position)."""
    import torch

    got = {}

    def hook(_m: Any, _i: Any, out: Any) -> None:
        got["h"] = (out[0] if isinstance(out, tuple) else out).float()

    hd = model.model.layers[layer].register_forward_hook(hook)
    with nc.intervene(model, spec), torch.no_grad():
        model.model(input_ids=ids, attention_mask=mask)
    hd.remove()
    h = got["h"]
    if last is not None:
        return h[torch.arange(h.shape[0], device=h.device), last]
    return h[mask.bool()]


def direction_for(
    model: Any, tok: Any, c: dict[str, Any], others: list[str], ref: dict[tuple[int, int], float], strength: float
) -> dict[str, Any]:
    """Difference of means at the median member layer, norm-matched to the clamp's mean residual change there."""
    import torch

    layer = int(np.median([layer for layer, _ in nc.members(c)]))

    def mean_res(texts: list[str]) -> Any:
        parts = []
        for chunk in _raw_batches(texts):
            ids, mask, last = nc.encode_raw(tok, chunk)
            parts.append(residual_at(model, ids, mask, layer, last))
        return torch.cat(parts).mean(0)

    d = mean_res([p["t"] for p in c["prompts"]]) - mean_res(others)
    ids, mask = nc.chat_ids(tok, nc.HELDOUT_QUESTIONS)
    mem = nc.members(c)
    spec = {"mode": "clamp_all", "neurons": mem, "values": {n: strength * ref[n] for n in mem}}
    change = (
        (residual_at(model, ids, mask, layer, None, spec) - residual_at(model, ids, mask, layer, None))
        .norm(dim=-1)
        .mean()
    )
    vec = d / d.norm() * change
    return {"mode": "add_all", "layer": layer, "vector": vec, "norm": float(change)}


def other_prompts(exclude: str, n: int, classes: tuple[str, ...] | None = None) -> list[str]:
    rng = np.random.default_rng(int(exclude))
    pool: list[str] = []
    sel = nc.selection()
    ids = sel["e1"] if classes else sel["e2"]
    for cid in ids:
        if cid == exclude:
            continue
        c = nc.community(cid)
        if classes and c["class"] not in classes:
            continue
        pool += [p["t"] for p in c["prompts"]]
    return [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]


def e2b(model: Any, tok: Any, out_dir: Path, finalists: list[dict[str, Any]]) -> dict[str, Any]:
    qs = nc.HELDOUT_QUESTIONS
    log = (out_dir / "e2b.jsonl").open("w")
    none = nc.generate(model, tok, qs, None, 300)
    rows = []
    for f in finalists:
        c = nc.community(f["id"])
        words, s, mem = nc.topic_words(c), float(f["c"]), nc.members(c)
        ref = a_ref(model, tok, c, mem)
        direction = direction_for(model, tok, c, other_prompts(c["id"], 100), ref, s)
        arms: dict[str, list[dict[str, Any]]] = {"none": none}
        arms["community"] = nc.generate(
            model, tok, qs, {"mode": "clamp_all", "neurons": mem, "values": {n: s * ref[n] for n in mem}}, 300
        )
        for seed in range(3):
            rnd = nc.random_set(c, seed)
            rref = a_ref(model, tok, c, rnd)
            arms[f"random{seed}"] = nc.generate(
                model, tok, qs, {"mode": "clamp_all", "neurons": rnd, "values": {n: s * rref[n] for n in rnd}}, 300
            )
        arms["direction"] = nc.generate(model, tok, qs, direction, 300)
        row: dict[str, Any] = {
            "id": f["id"],
            "label": nc.topic_label(c),
            "c": s,
            "words": words,
            "direction_layer": direction["layer"],
            "direction_norm": round(direction["norm"], 3),
        }
        for arm, replies in arms.items():
            scs = [score(r, words) for r in replies]
            k = sum(x["good_hit"] for x in scs)
            row[arm] = {
                "good_hits": k,
                "ci": nc.wilson(k, len(qs)),
                "hits": sum(x["hit"] for x in scs),
                "fluent": sum(x["fluent"] for x in scs),
                "loops": sum(x["loops"] for x in scs),
            }
            for q, r, sc in zip(qs, replies, scs):
                log.write(json.dumps({"id": f["id"], "arm": arm, "q": q, "text": r["text"], **sc}) + "\n")
        rmax = max(row[f"random{i}"]["good_hits"] for i in range(3))
        row["steers"] = bool(row["community"]["good_hits"] >= 8 and rmax <= 1 and row["none"]["good_hits"] <= 1)
        rows.append(row)
        print(
            f"e2b {f['id']} {row['label'][:24]:<24} c={s} community {row['community']['good_hits']}/15 dir {row['direction']['good_hits']}/15 rand max {rmax} none {row['none']['good_hits']} steers {row['steers']}",
            flush=True,
        )
    log.close()
    res = {"rows": rows}
    (out_dir / "e2b.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def collateral(model: Any, tok: Any, texts: list[str], spec: dict[str, Any]) -> float:
    vals = []
    for chunk in _raw_batches(texts):
        ids, mask, last = nc.encode_raw(tok, chunk)
        base = nc.last_logprobs(model, ids, mask, last)
        with nc.intervene(model, spec):
            lp = nc.last_logprobs(model, ids, mask, last)
        vals.append(nc.kl(base, lp).cpu().numpy())
    return float(np.concatenate(vals).mean())


def e3(model: Any, tok: Any, out_dir: Path, finalists: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for f in finalists:
        c = nc.community(f["id"])
        s, mem = float(f["c"]), nc.members(c)
        ref = a_ref(model, tok, c, mem)
        texts = other_prompts(c["id"], 200, classes=("grammar", "retrieval", "formatting", "lexical", "calibration"))
        rnd = nc.random_set(c, 0)
        rref = a_ref(model, tok, c, rnd)
        direction = direction_for(model, tok, c, other_prompts(c["id"], 100), ref, s)
        row = {
            "id": f["id"],
            "label": nc.topic_label(c),
            "community": collateral(
                model, tok, texts, {"mode": "clamp_all", "neurons": mem, "values": {n: s * ref[n] for n in mem}}
            ),
            "random0": collateral(
                model, tok, texts, {"mode": "clamp_all", "neurons": rnd, "values": {n: s * rref[n] for n in rnd}}
            ),
            "direction": collateral(model, tok, texts, direction),
        }
        rows.append(row)
        print(
            f"e3 {f['id']} {row['label'][:24]:<24} KL community {row['community']:.3f} direction {row['direction']:.3f} random {row['random0']:.3f}",
            flush=True,
        )

    # the "an" off-switch
    qs = nc.HELDOUT_QUESTIONS
    c = nc.community(nc.AN)
    log = (out_dir / "e3_an.jsonl").open("w")
    an = {}
    arms: dict[str, Any] = {"none": None, "community": {"mode": "zero_all", "neurons": nc.members(c)}}
    for seed in range(3):
        arms[f"random{seed}"] = {"mode": "zero_all", "neurons": nc.random_set(c, seed)}
    for arm, spec in arms.items():
        replies = nc.generate(model, tok, qs, spec, 300)
        tot = np.zeros(3, int)
        for q, r in zip(qs, replies):
            e = nc.article_errors(r["text"])
            tot += e
            log.write(
                json.dumps(
                    {
                        "arm": arm,
                        "q": q,
                        "text": r["text"],
                        "an_bad": e[0],
                        "a_bad": e[1],
                        "words": e[2],
                        "fluent": nc.fluent(r["text"]),
                    }
                )
                + "\n"
            )
        an[arm] = {
            "an_bad": int(tot[0]),
            "a_bad": int(tot[1]),
            "words": int(tot[2]),
            "per_1000": round(1000 * (tot[0] + tot[1]) / max(tot[2], 1), 2),
        }
        print(f"e3 an-switch {arm}: {an[arm]}", flush=True)
    log.close()
    res = {"collateral": rows, "an_switch": an}
    (out_dir / "e3.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def run(stage: str, out_dir: Path, finalists: list[dict[str, Any]] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    if stage in ("e1_base", "e1_instruct"):
        model, tok = nc.load(nc.BASE if stage == "e1_base" else nc.INSTRUCT)
        e1(model, tok, stage.split("_")[1], out_dir)
    else:
        model, tok = nc.load(nc.INSTRUCT)
        if stage == "e2a":
            e2a(model, tok, out_dir)
        elif stage == "e2b":
            e2b(model, tok, out_dir, finalists or [])
        elif stage == "e3":
            e3(model, tok, out_dir, finalists or [])
        elif stage == "smoke":
            c = nc.community(nc.selection()["e2"][0])
            print(necessity(model, tok, [p["t"] for p in c["prompts"]][:4], [nc.members(c), nc.random_set(c, 0)]))
            ref = a_ref(model, tok, c, nc.members(c))
            print(ref)
            print(nc.generate(model, tok, nc.TUNE_QUESTIONS[:2], clamp_spec(nc.members(c), ref, [4.0, 4.0]), 40))
            print(nc.generate(model, tok, nc.TUNE_QUESTIONS[:2], None, 40))
    print(f"{stage} done in {time.time() - t:.0f}s", flush=True)
