"""Detector-free guard: GenAI-Bench prompts scored by VQAScore (Qwen2.5-VL-7B answers 'Does this figure show ...?').

GenAI-Bench (Apache-2.0, 1,600 prompts with human 1-5 ratings of six models) is never used to build or tune anything.

  python guard_genai.py            prepare, judge check, generate, score, summarise (each stage cached)

Stages:
  judge     VQAScore on GenAI-Bench's own rated images (first parquet shard): does it order models as people do,
            and does a shuffled prompt score lower? One rater against the other raters gives the bar.
  guard     400 random prompts: FLUX.1-schnell on the original prompt (1 image), on the planner's rewrite (4 images,
            the first is "planner, 1 sample"), and planner + PickScore best of 4 (the harness without its GenEval
            checker, which needs GenEval's object lists). Everything is scored against the original prompt.
"""

import itertools
import json
import random

import numpy as np

from common import PRICE, RESULTS, chunks, modal_cls, read_jsonl, write_jsonl

OUT = RESULTS / "genai_bench"
N_PROMPTS = 400
MODELS = ["DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6", "SDXL_2_1", "SDXL_Base", "SDXL_Turbo"]


def boot(d, n_boot=4000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(d.mean()), *np.percentile(bs, [2.5, 97.5]).tolist()


def prepare():
    path = OUT / "meta.jsonl"
    if not path.exists():
        import modal

        rows = modal.Function.from_name("t2i-geneval", "genai_prepare").remote(1)
        write_jsonl(path, rows)
    return read_jsonl(path)


def vqa_scores(items, cache_path):
    """VQAScore for [{"path", "prompt"}], cached by (path, prompt)."""
    cache = {(r["path"], r["prompt"]): r["v"] for r in read_jsonl(cache_path)}
    need = [it for it in items if (it["path"], it["prompt"]) not in cache]
    if need:
        res = [x for c in modal_cls("VQA").score.map(chunks(need, 48)) for x in c]
        for it, v in zip(need, res, strict=False):
            cache[(it["path"], it["prompt"])] = v
        write_jsonl(cache_path, [{"path": k[0], "prompt": k[1], "v": v} for k, v in cache.items()])
    return [cache[(it["path"], it["prompt"])] for it in items]


def pair_agreement(score_a, score_b):
    """Share of pairs (within a prompt) ordered the same way by two scorers; pairs tied in either are skipped."""
    agree = total = 0
    for i, j in itertools.combinations(range(len(score_a)), 2):
        da, db = score_a[i] - score_a[j], score_b[i] - score_b[j]
        if da == 0 or db == 0:
            continue
        total += 1
        agree += (da > 0) == (db > 0)
    return agree, total


def judge_check(meta):
    rated = [r for r in meta if "images" in r]
    rng = random.Random(0)
    shuffled = [r["prompt"] for r in rated]
    while any(a == b for a, b in zip(shuffled, [r["prompt"] for r in rated], strict=False)):
        rng.shuffle(shuffled)
    items = [{"path": r["images"][m], "prompt": r["prompt"]} for r in rated for m in MODELS]
    wrong = [{"path": r["images"][m], "prompt": p} for r, p in zip(rated, shuffled, strict=False) for m in MODELS]
    v = vqa_scores(items + wrong, OUT / "vqa_judge.jsonl")
    right, mism = np.array(v[: len(items)]), np.array(v[len(items) :])
    # VQAScore vs the mean human rating, and one rater vs the mean of the other raters (leave one out).
    ag_j = tot_j = ag_h = tot_h = 0
    for n, r in enumerate(rated):
        s = right[n * len(MODELS) : (n + 1) * len(MODELS)].tolist()
        ratings = [r["ratings"][m] for m in MODELS]
        mean = [float(np.mean(x)) for x in ratings]
        a, t = pair_agreement(s, mean)
        ag_j, tot_j = ag_j + a, tot_j + t
        for k in range(min(len(x) for x in ratings)):
            one = [x[k] for x in ratings]
            rest = [float(np.mean(x[:k] + x[k + 1 :])) for x in ratings]
            a, t = pair_agreement(one, rest)
            ag_h, tot_h = ag_h + a, tot_h + t
    res = {
        "prompts": len(rated),
        "images": len(items),
        "matched_beats_shuffled": float((right > mism).mean()),
        "matched_mean": float(right.mean()),
        "shuffled_mean": float(mism.mean()),
        "pair_agreement_vqascore_vs_mean_human": [ag_j / tot_j, tot_j],
        "pair_agreement_one_rater_vs_other_raters": [ag_h / tot_h, tot_h],
    }
    (OUT / "judge_check.json").write_text(json.dumps(res, indent=1))
    return res


def guard(meta):
    from harness import plan

    rng = random.Random(0)
    prompts = sorted(rng.sample(meta, N_PROMPTS), key=lambda r: r["idx"])
    plans = plan([(str(p["idx"]), p["prompt"], None) for p in prompts], cache_path=RESULTS / "plans/genai.jsonl")
    cands = []
    for p in prompts:
        base = {"idx": p["idx"], "prompt": p["prompt"]}
        cands.append(base | {"arm": "orig", "k": 0, "gen_prompt": p["prompt"], "seed": 1000 * p["idx"],
                             "path": f"genai/{p['idx']:05d}/orig_0.png"})  # fmt: skip
        for k in range(4):
            cands.append(base | {"arm": "plan", "k": k, "gen_prompt": plans[str(p["idx"])],
                                 "seed": 1000 * p["idx"] + k,
                                 "path": f"genai/{p['idx']:05d}/plan_{k}.png"})  # fmt: skip
    gen = modal_cls("Gen", model_id="black-forest-labs/FLUX.1-schnell")
    jobs = [{"prompt": c["gen_prompt"], "seed": c["seed"], "path": c["path"]} for c in cands]
    stats = list(gen.generate.map(chunks(jobs, 64), kwargs={"steps": 4, "guidance": 0.0, "size": 1024}))
    secs, n_new = sum(s["seconds"] for s in stats), sum(s["n"] for s in stats)
    items = [{"path": c["path"], "prompt": c["prompt"]} for c in cands]
    vq = vqa_scores(items, OUT / "vqa_guard.jsonl")
    pick_path = OUT / "pick_guard.jsonl"
    pk = {r["path"]: r["v"] for r in read_jsonl(pick_path)}
    need = [it for it in items if it["path"] not in pk]
    if need:
        res = [x for c in modal_cls("Scorer").pickscore.map(chunks(need, 128)) for x in c]
        pk |= {it["path"]: v for it, v in zip(need, res, strict=False)}
        write_jsonl(pick_path, [{"path": k, "v": v} for k, v in pk.items()])
    for c, v in zip(cands, vq, strict=False):
        c["vqa"], c["pick"] = v, pk[c["path"]]
    write_jsonl(OUT / "rows.jsonl", cands)
    if n_new:
        (OUT / "gen_cost.json").write_text(
            json.dumps({"images": n_new, "gpu_s": secs, "usd": secs * PRICE["H100"]}, indent=1)
        )
    return prompts, cands


def summarise(prompts, cands):
    by = {}
    for c in cands:
        by.setdefault(c["idx"], {}).setdefault(c["arm"], []).append(c)
    arms = {"base": [], "planner": [], "planner_pick4": []}
    for p in prompts:
        g = by[p["idx"]]
        arms["base"].append(g["orig"][0])
        plan_c = sorted(g["plan"], key=lambda c: c["k"])
        arms["planner"].append(plan_c[0])
        arms["planner_pick4"].append(max(plan_c, key=lambda c: c["pick"]))
    out = {"prompts": len(prompts)}
    for arm, rows in arms.items():
        out[arm] = {"vqa": float(np.mean([r["vqa"] for r in rows])), "pick": float(np.mean([r["pick"] for r in rows]))}
        if arm != "base":
            out[arm]["vqa_diff"] = boot([r["vqa"] - b["vqa"] for r, b in zip(rows, arms["base"], strict=False)])
            out[arm]["pick_diff"] = boot([r["pick"] - b["pick"] for r, b in zip(rows, arms["base"], strict=False)])
    # By GenAI-Bench skill tag (a prompt can carry several).
    tags = sorted({t for p in prompts for kind in ("basic", "advanced") for t in (p["tags"].get(kind) or [])})
    out["by_tag"] = {}
    for t in tags:
        ids = [
            i for i, p in enumerate(prompts) if t in (p["tags"].get("basic") or []) + (p["tags"].get("advanced") or [])
        ]
        if len(ids) < 20:
            continue
        out["by_tag"][t] = {
            "n": len(ids),
            "base": float(np.mean([arms["base"][i]["vqa"] for i in ids])),
            "planner_diff": boot([arms["planner"][i]["vqa"] - arms["base"][i]["vqa"] for i in ids]),
            "planner_pick4_diff": boot([arms["planner_pick4"][i]["vqa"] - arms["base"][i]["vqa"] for i in ids]),
        }
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = prepare()
    print(f"GenAI-Bench: {len(meta)} prompts, {sum('images' in r for r in meta)} with rated images on the volume")
    print(json.dumps(judge_check(meta), indent=1))
    prompts, cands = guard(meta)
    s = summarise(prompts, cands)
    print(json.dumps({k: v for k, v in s.items() if k != "by_tag"}, indent=1))
    for t, v in s["by_tag"].items():
        print(f"{t:<22} n={v['n']:<4} base {v['base']:.3f}  planner {v['planner_diff'][0]:+.3f} "
              f"[{v['planner_diff'][1]:+.3f}, {v['planner_diff'][2]:+.3f}]  "
              f"planner+pick4 {v['planner_pick4_diff'][0]:+.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
