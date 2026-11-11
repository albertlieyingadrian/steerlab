"""The no-training harness: LLM prompt planner + best-of-N picked by a checker and PickScore + retry on a failed check.

Stages (each cached, so reruns are free):
  plan      the planner rewrites each prompt (results/plans/<set>.jsonl)
  pool      FLUX.1-schnell draws 8 candidates per (prompt, rep) for the original and the planned prompt
  score     every candidate gets the official GenEval verdict, the independent checker's verdict and PickScore
  retry     where none of the first 4 candidates passes the picker, re-plan with the failure and draw 4 more
  select    offline: every (prompt variant, picker, N, retry) combination is scored with the official grader

  python harness.py --set val --reps 2
  python harness.py --set official --reps 4 --variants plan --nmax 4 --retry   (the one test run)

The official GenEval grader picks only in the "oracle" rows, reported as an upper bound, never as the result.
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import PRICE, RESULTS, chunks, fmt_summary, geneval_summary, load_set, modal_cls, read_jsonl, write_jsonl

PLANNER_MODEL = "openai/gpt-5.4-mini"
PLANNER_SYSTEM = """You rewrite short text-to-image prompts for FLUX.1-schnell so the image matches them exactly.
Rules:
- Keep every object, every number, every colour and every spatial relation from the original. Add no new objects.
- Say counts as digits and words ("exactly three (3) cups"), and say there are no other instances of that object.
- For colours, give each object its colour right next to its name, and keep other objects in different colours.
- For "left of / right of / above / below", place the objects explicitly in the frame ("on the left side of the
  image ... on the right side"), with both objects fully visible and clearly separated.
- Prefer a plain, uncluttered background and a realistic photo style. Objects should be large and unoccluded.
- At most 70 words. Reply with the rewritten prompt only."""
RETRY_NOTE = "\nA previous image for this prompt failed a check: {reason}. Fix that failure in your rewrite."


def openai_client():
    """OpenRouter's OpenAI-compatible endpoint. The key comes from OPENROUTER_API_KEY in the environment, or from the
    nearest .env / .env.local above this folder."""
    from dotenv import dotenv_values
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    for root in Path(__file__).resolve().parents:
        for name in (".env", ".env.local"):
            if not key and (root / name).exists():
                key = dotenv_values(root / name).get("OPENROUTER_API_KEY")
    if key:
        return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    raise RuntimeError("no OPENROUTER_API_KEY in .env")


def plan(prompts, cache_path=None):
    """{key: rewritten prompt}; prompts is [(key, original prompt, failure reason or None)]."""
    cache = {r["key"]: r for r in read_jsonl(cache_path)} if cache_path else {}
    todo = [p for p in prompts if p[0] not in cache]
    client = openai_client() if todo else None
    usage = {"in": 0, "out": 0}

    def one(item):
        key, text, why = item
        msgs = [
            {"role": "system", "content": PLANNER_SYSTEM + (RETRY_NOTE.format(reason=why) if why else "")},
            {"role": "user", "content": text},
        ]
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model=PLANNER_MODEL, messages=msgs)
                return key, r.choices[0].message.content.strip(), r.usage.prompt_tokens, r.usage.completion_tokens
            except Exception as e:
                print("planner retry", attempt, type(e).__name__)
                time.sleep(2 + 4 * attempt)
        raise RuntimeError("planner failed")

    with ThreadPoolExecutor(4) as ex:
        for key, text, pin, pout in ex.map(one, todo):
            cache[key] = {"key": key, "plan": text, "in": pin, "out": pout}
            usage["in"] += pin
            usage["out"] += pout
    if cache_path and todo:
        write_jsonl(cache_path, list(cache.values()))
    print(f"planner: {len(todo)} new calls, {usage['in']} in / {usage['out']} out tokens")
    return {k: v["plan"] for k, v in cache.items()}


def score_pool(cands, out_path):
    """Add official verdict, independent verdict and PickScore to every candidate (cached by path)."""
    done = {r["path"]: r for r in read_jsonl(out_path)}
    todo = [c for c in cands if c["path"] not in done]
    if todo:
        grader, scorer = modal_cls("Grader"), modal_cls("Scorer")
        items = [{"path": c["path"], "meta": c["meta"]} for c in todo]
        # PickScore is always against the original prompt: the planner must not move the target.
        pitems = [{"path": c["path"], "prompt": c["prompt"]} for c in todo]
        parts = {}
        for name, fn, its, size in [
            ("grade", grader.grade, items, 64),
            ("indep", scorer.indep_check, items, 64),
            ("pick", scorer.pickscore, pitems, 128),
        ]:
            # Each scorer's output is cached on its own, so a crash in one does not lose the others.
            part_path = Path(str(out_path) + f".{name}.jsonl")
            cached = {r["path"]: r["v"] for r in read_jsonl(part_path)}
            need = [it for it in its if it["path"] not in cached]
            if need:
                res = [x for c in fn.map(chunks(need, size)) for x in c]
                for it, v in zip(need, res, strict=False):
                    cached[it["path"]] = v
                write_jsonl(part_path, [{"path": k, "v": v} for k, v in cached.items()])
            parts[name] = [cached[c["path"]] for c in todo]
        g, ind, pk = parts["grade"], parts["indep"], parts["pick"]
        for c, gg, ii, pp in zip(todo, g, ind, pk, strict=False):
            assert gg["path"] == c["path"] and "error" not in gg, gg
            done[c["path"]] = {
                "path": c["path"],
                "correct": gg["correct"],
                "off_score": gg["score"],
                "reason": gg["reason"],
                "indep_pass": ii["pass"],
                "indep_score": ii["score"],
                "indep_counts": ii["counts"],
                "pick": pp,
            }
        write_jsonl(out_path, list(done.values()))
    return done


def generate(cands, gen_chunk=64):
    gen = modal_cls("Gen", model_id="black-forest-labs/FLUX.1-schnell")
    jobs = [{"prompt": c["gen_prompt"], "seed": c["seed"], "path": c["path"]} for c in cands]
    stats = list(gen.generate.map(chunks(jobs, gen_chunk), kwargs={"steps": 4, "guidance": 0.0, "size": 1024}))
    secs, n = sum(s["seconds"] for s in stats), sum(s["n"] for s in stats)
    print(f"generated {n} new images in {secs:.0f} GPU-s")
    return secs, n


def pick_key(picker, c, rng):
    if picker == "oracle":
        return (c["off_score"] + (1.0 if c["correct"] else 0.0), c["pick"])
    if picker == "indep":
        return (c["indep_score"] + (1.0 if c["indep_pass"] else 0.0), c["pick"])
    if picker == "pickscore":
        return (c["pick"],)
    return (rng.random(),)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--variants", default="orig,plan")
    ap.add_argument("--retry", action="store_true", help="re-plan and draw more where the first 4 all fail indep")
    ap.add_argument("--retry-n", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    variants = args.variants.split(",")
    prompts = load_set(args.set)
    if args.limit:
        prompts = prompts[: args.limit]
    run = f"harness_{args.set}"
    out = RESULTS / run
    out.mkdir(parents=True, exist_ok=True)
    plans = plan([(str(p["idx"]), p["prompt"], None) for p in prompts], cache_path=RESULTS / f"plans/{args.set}.jsonl")

    cands = []
    for p in prompts:
        meta = {k: v for k, v in p.items() if k not in ("idx", "split")}
        for v in variants:
            for rep in range(args.reps):
                for k in range(args.nmax):
                    cands.append(
                        {
                            "idx": p["idx"], "tag": p["tag"], "prompt": p["prompt"], "meta": meta, "variant": v,
                            "rep": rep, "k": k, "seed": 1000 * p["idx"] + 100 * rep + k,
                            "gen_prompt": p["prompt"] if v == "orig" else plans[str(p["idx"])],
                            "path": f"{run}/{p['idx']:05d}/{v}_{rep}_{k}.png",
                        }
                    )  # fmt: skip
    gen_secs, gen_n = generate(cands)
    scores = score_pool(cands, out / "pool_scores.jsonl")
    for c in cands:
        c.update(scores[c["path"]])

    retry_cands = []
    if args.retry:
        first = {}
        for c in cands:
            if c["variant"] == "plan" and c["k"] < 4:
                first.setdefault((c["idx"], c["rep"]), []).append(c)
        failed = {key: grp for key, grp in first.items() if not any(x["indep_pass"] for x in grp)}
        print(f"retry: {len(failed)} of {len(first)} (prompt, rep) pairs had no passing candidate in 4")
        reasons = {}
        for (idx, rep), grp in failed.items():
            best = max(grp, key=lambda x: (x["indep_score"], x["pick"]))
            want = {r["class"]: r["count"] for r in best["meta"]["include"]}
            reasons[f"{idx}_{rep}"] = (best["prompt"], f"wanted counts {want}, a detector found {best['indep_counts']}")
        replans = plan(
            [(k, reasons[k][0], reasons[k][1]) for k in reasons], cache_path=RESULTS / f"plans/{args.set}_retry.jsonl"
        )
        for (idx, rep), grp in failed.items():
            c0 = grp[0]
            for k in range(args.retry_n):
                retry_cands.append(
                    c0 | {
                        "variant": "retry", "k": k, "seed": 1000 * idx + 100 * rep + 50 + k,
                        "gen_prompt": replans[f"{idx}_{rep}"], "path": f"{run}/{idx:05d}/retry_{rep}_{k}.png",
                    }
                )  # fmt: skip
        s2, n2 = generate(retry_cands)
        gen_secs, gen_n = gen_secs + s2, gen_n + n2
        scores = score_pool(retry_cands, out / "pool_scores.jsonl")
        for c in retry_cands:
            c.update(scores[c["path"]])

    write_jsonl(out / "pool.jsonl", [{k: v for k, v in c.items() if k != "meta"} for c in cands + retry_cands])
    select(cands, retry_cands, variants, args, out)
    cost = {"gen_gpu_seconds_new": gen_secs, "gen_images_new": gen_n}
    if gen_n:
        cost["sec_per_candidate"] = gen_secs / gen_n
        cost["usd_per_candidate_gen"] = gen_secs / gen_n * PRICE["H100"]
    (out / f"cost_{int(time.time())}.json").write_text(json.dumps(cost, indent=1))


def select(cands, retry_cands, variants, args, out):
    rng = random.Random(0)
    groups = {}
    for c in cands:
        groups.setdefault((c["variant"], c["idx"], c["rep"]), []).append(c)
    retries = {}
    for c in retry_cands:
        retries.setdefault((c["idx"], c["rep"]), []).append(c)
    table = []
    ns = [n for n in (1, 2, 4, 8) if n <= args.nmax]
    configs = [(v, pk, n, False) for v in variants for pk in ("random", "pickscore", "indep", "oracle") for n in ns]
    if args.retry and "plan" in variants:
        configs += [("plan", "indep", 4, True)]
    for v, picker, n, use_retry in configs:
        if n == 1 and picker != "random":
            continue
        rows, n_drawn = [], 0
        for (vv, idx, rep), grp in sorted(groups.items(), key=lambda x: (x[0][1], x[0][2])):
            if vv != v:
                continue
            pool = sorted(grp, key=lambda x: x["k"])[:n]
            if use_retry and not any(x["indep_pass"] for x in pool):
                pool = pool + sorted(retries.get((idx, rep), []), key=lambda x: x["k"])
            n_drawn += len(pool)
            best = pool[0] if n == 1 else max(pool, key=lambda x: pick_key(picker, x, rng))
            rows.append(best)
        s = geneval_summary(rows)
        label = "single" if n == 1 else picker
        name = f"{v} | {label} | N={n}{' + retry' if use_retry else ''}"
        table.append(
            {
                "config": name, "variant": v, "picker": label, "n": n, "retry": use_retry, "geneval": s,
                "pickscore": sum(r["pick"] for r in rows) / len(rows), "candidates_per_output": n_drawn / len(rows),
                "paths": [r["path"] for r in rows],
            }
        )  # fmt: skip
        print(f"\n== {name}  (candidates/output {n_drawn / len(rows):.2f}, PickScore {table[-1]['pickscore']:.3f})")
        print(fmt_summary(s))
    (out / "ablation.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
