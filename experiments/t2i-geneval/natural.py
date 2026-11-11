"""A planner told to keep natural scenes: Claude rewrites the prompt with explicit counts, colours and positions, inside
a believable photographic scene instead of an isolated catalogue shot. Same picker and retry as the published harness.

  python natural.py pool --set val --planner opus --reps 1        # draw 4 per (prompt, rep), check, retry, grade
  python natural.py pool --set val --planner sonnet --reps 1
  python natural.py genai --planner opus                           # GenAI-Bench guard, one image per prompt
  python natural.py judge --set val --planner opus                 # Claude judge: which image looks more natural
  python natural.py summary --set val
  python natural.py pool --set official --planner opus --reps 2    # the test run, if validation passes

Costs are kept down by grading with the official grader only the first candidate (the planner-alone score) and the
chosen image of each output; the checker and PickScore see every candidate, as in the harness. Seeds match the
published harness (1000 * idx + 100 * rep + k), so every comparison is paired.
"""

import argparse
import base64
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import PRICE, RESULTS, chunks, geneval_summary, load_set, modal_cls, read_jsonl, write_jsonl
from compare import paired_diff

OUT = RESULTS / "natural"
MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}
NATURAL_SYSTEM = """You rewrite short text-to-image prompts for FLUX.1-schnell so the image matches them exactly and
still looks like a natural photograph of a real scene.
Rules:
- Keep every object, every number, every colour and every spatial relation from the original. Do not add more of
  the objects the prompt names.
- Say counts as digits and words ("exactly three (3) cups"), and say there are no other instances of that object.
- For colours, give each object its colour right next to its name.
- For "left of / right of / above / below", say where each object sits in the frame (left or right side, upper or
  lower part), inside a scene where that arrangement is plausible, with both objects fully visible and not overlapping.
- Put the objects in a believable real-world setting that suits them (a kitchen counter, a street, a field, a living
  room), with natural light and the look of an ordinary photograph. Do not ask for a plain, white or studio
  background, a product shot, a cut-out or an isolated object. Keep the named objects large and easy to see; other
  scenery is fine, but keep it in the background.
- At most 70 words. Reply with the rewritten prompt only."""
RETRY_NOTE = "\nA previous image for this prompt failed a check: {reason}. Fix that failure in your rewrite."
JUDGE_PROMPT = """Two images, A then B. Which one looks more like an ordinary, natural photograph of a real scene, as
opposed to a product or catalogue shot of objects alone on a plain background? Ignore whether either matches any
caption. Answer with exactly one word: A, B or tie."""


def claude():
    import anthropic
    from dotenv import dotenv_values

    key = os.environ.get("ANTHROPIC_API_KEY")
    for root in Path(__file__).resolve().parents:
        if not key and (root / ".env").exists():
            key = dotenv_values(root / ".env").get("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def ask(client, model, system, content, max_tokens=2000):
    """One Claude call with server-side refusal fallbacks; returns (text, input tokens, output tokens)."""
    for attempt in range(5):
        try:
            r = client.beta.messages.create(
                model=model,
                max_tokens=max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "medium"},
                system=system,
                messages=[{"role": "user", "content": content}],
            )
            text = "".join(b.text for b in r.content if b.type == "text").strip()
            if r.stop_reason == "refusal" or not text:
                raise RuntimeError(f"no text ({r.stop_reason})")
            return text, r.usage.input_tokens, r.usage.output_tokens
        except Exception as e:
            print("claude retry", attempt, type(e).__name__, str(e)[:80])
            time.sleep(3 + 5 * attempt)
    raise RuntimeError("claude failed")


def plan(items, planner, cache_path, system=None):
    """{key: rewrite}; items is [(key, original prompt, failure reason or None)]."""
    cache = {r["key"]: r for r in read_jsonl(cache_path)}
    todo = [it for it in items if it[0] not in cache]
    if todo:
        client, model = claude(), MODELS[planner]

        def one(it):
            key, text, why = it
            sys_msg = (system or NATURAL_SYSTEM) + (RETRY_NOTE.format(reason=why) if why else "")
            out, tin, tout = ask(client, model, sys_msg, text)
            return {"key": key, "plan": out, "in": tin, "out": tout}

        with ThreadPoolExecutor(8) as ex:
            for r in ex.map(one, todo):
                cache[r["key"]] = r
        write_jsonl(cache_path, list(cache.values()))
    print(f"planner {planner}: {len(todo)} new calls")
    return {k: v["plan"] for k, v in cache.items()}


def generate(cands):
    gen = modal_cls("Gen", model_id="black-forest-labs/FLUX.1-schnell")
    jobs = [{"prompt": c["gen_prompt"], "seed": c["seed"], "path": c["path"]} for c in cands]
    stats = list(gen.generate.map(chunks(jobs, 64), kwargs={"steps": 4, "guidance": 0.0, "size": 1024}))
    return sum(s["seconds"] for s in stats), sum(s["n"] for s in stats)


def scored(cands, name, fn, make, size, cache_path):
    """Run one scorer over candidates not yet in its cache; returns {path: value}."""
    cache = {r["path"]: r["v"] for r in read_jsonl(cache_path)}
    need = [c for c in cands if c["path"] not in cache]
    if need:
        res = [x for ch in fn.map(chunks([make(c) for c in need], size)) for x in ch]
        for c, v in zip(need, res, strict=True):
            cache[c["path"]] = v
        write_jsonl(cache_path, [{"path": k, "v": v} for k, v in cache.items()])
        print(f"{name}: scored {len(need)}")
    return cache


def pool(args):
    variant = f"nat{args.planner}"
    run = f"harness_{args.set}"
    prompts = load_set(args.set)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{args.set}_{variant}"
    plans = plan([(str(p["idx"]), p["prompt"], None) for p in prompts], args.planner,
                 RESULTS / f"plans/{args.set}_{variant}.jsonl")  # fmt: skip
    cands = []
    for p in prompts:
        meta = {k: v for k, v in p.items() if k not in ("idx", "split")}
        for rep in range(args.reps):
            for k in range(4):
                cands.append({"idx": p["idx"], "tag": p["tag"], "prompt": p["prompt"], "meta": meta, "variant": variant,
                              "rep": rep, "k": k, "seed": 1000 * p["idx"] + 100 * rep + k,
                              "gen_prompt": plans[str(p["idx"])],
                              "path": f"{run}/{p['idx']:05d}/{variant}_{rep}_{k}.png"})  # fmt: skip
    secs, n = generate(cands)
    scorer, grader = modal_cls("Scorer"), modal_cls("Grader")

    def check(cs):
        ind = scored(cs, "checker", scorer.indep_check, lambda c: {"path": c["path"], "meta": c["meta"]}, 64,
                     Path(f"{stem}.indep.jsonl"))  # fmt: skip
        pk = scored(cs, "pickscore", scorer.pickscore, lambda c: {"path": c["path"], "prompt": c["prompt"]}, 128,
                    Path(f"{stem}.pick.jsonl"))  # fmt: skip
        for c in cs:
            c.update(indep_pass=ind[c["path"]]["pass"], indep_score=ind[c["path"]]["score"],
                     indep_counts=ind[c["path"]]["counts"], pick=pk[c["path"]])  # fmt: skip

    check(cands)
    groups = {}
    for c in cands:
        groups.setdefault((c["idx"], c["rep"]), []).append(c)
    failed = {key: g for key, g in groups.items() if not any(c["indep_pass"] for c in g)}
    reasons = []
    for (i, r), g in failed.items():
        want = {x["class"]: x["count"] for x in g[0]["meta"]["include"]}
        best = max(g, key=lambda c: (c["indep_score"], c["pick"]))
        reasons.append((f"{i}_{r}", g[0]["prompt"], f"wanted counts {want}, a detector found {best['indep_counts']}"))
    replans = plan(reasons, args.planner, RESULTS / f"plans/{args.set}_{variant}_retry.jsonl")
    retry = []
    for (i, r), g in failed.items():
        for k in range(4):
            retry.append(g[0] | {"variant": f"{variant}r", "k": k, "seed": 1000 * i + 100 * r + 50 + k,
                                 "gen_prompt": replans[f"{i}_{r}"],
                                 "path": f"{run}/{i:05d}/{variant}r_{r}_{k}.png"})  # fmt: skip
    if retry:
        s2, n2 = generate(retry)
        secs, n = secs + s2, n + n2
        check(retry)
    for c in retry:
        groups[(c["idx"], c["rep"])].append(c)
    chosen = {key: max(g, key=lambda c: (c["indep_score"] + (1.0 if c["indep_pass"] else 0.0), c["pick"]))
              for key, g in groups.items()}  # fmt: skip
    first = [c for c in cands if c["k"] == 0]
    to_grade = {c["path"]: c for c in first + list(chosen.values())}
    gr = scored(list(to_grade.values()), "grader", grader.grade, lambda c: {"path": c["path"], "meta": c["meta"]},
                64, Path(f"{stem}.grade.jsonl"))  # fmt: skip
    rows = []
    for c in cands + retry:
        row = {k: v for k, v in c.items() if k != "meta"}
        if c["path"] in gr:
            row["correct"] = gr[c["path"]]["correct"]
        row["chosen"] = chosen[(c["idx"], c["rep"])]["path"] == c["path"]
        rows.append(row)
    write_jsonl(Path(f"{stem}_pool.jsonl"), rows)
    cost = {"images_new": n, "gpu_s": secs, "usd_gen": secs * PRICE["H100"],
            "candidates_per_output": (len(cands) + len(retry)) / len(groups)}  # fmt: skip
    Path(f"{stem}_cost_{int(time.time())}.json").write_text(json.dumps(cost, indent=1))
    print(json.dumps(cost))


def gpt_rows(set_, reps):
    """The published harness (gpt-5.4-mini planner) on the same prompts and reps: its single and chosen images."""
    pool_ = {r["path"]: r for r in read_jsonl(RESULTS / f"harness_{set_}/pool.jsonl")}
    table = json.loads((RESULTS / f"harness_{set_}/ablation.json").read_text())
    single = next(t for t in table if t["config"] == "plan | single | N=1")["paths"]
    chosen = next(t for t in table if t["config"] == "plan | indep | N=4 + retry")["paths"]
    return [pool_[p] for p in single if pool_[p]["rep"] < reps], [pool_[p] for p in chosen if pool_[p]["rep"] < reps]


def nat_rows(set_, planner):
    rows = read_jsonl(OUT / f"{set_}_nat{planner}_pool.jsonl")
    return [r for r in rows if r["k"] == 0 and r["variant"] == f"nat{planner}"], [r for r in rows if r["chosen"]]


def fetch_jpegs(paths, size=512):
    import modal

    out = {}
    for ch in chunks(sorted(set(paths)), 200):
        out |= modal.Function.from_name("t2i-geneval", "fetch").remote(ch, size)
    return out


def judge(args):
    """Claude, shown two images in both orders, says which looks more like a natural photo. Checked first on the
    published blind look's 50 pairs, where one person's answers are known."""
    client = claude()
    model = MODELS["opus"]
    OUT.mkdir(parents=True, exist_ok=True)

    def run(pairs, path):
        done = {(r["pair"], r["order"]): r for r in read_jsonl(path)}
        need = [(p, o) for p in pairs for o in (0, 1) if (p["pair"], o) not in done]
        imgs = fetch_jpegs([x for p, _ in need for x in (p["a"], p["b"])]) if need else {}

        def one(item):
            p, o = item
            first, second = (p["a"], p["b"]) if o == 0 else (p["b"], p["a"])
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(imgs[x]).decode(),
                    },
                }
                for x in (first, second)
            ]
            content.append({"type": "text", "text": JUDGE_PROMPT})  # fmt: skip
            text, tin, tout = ask(client, model, "You compare photographs.", content, max_tokens=1000)
            word = text.strip().strip(".").split()[-1].lower() if text.strip() else "tie"
            shown = {"a": first, "b": second}.get(word)
            side = "tie" if shown is None else ("a" if shown == p["a"] else "b")
            return {"pair": p["pair"], "order": o, "raw": text[:20], "winner": side, "in": tin, "out": tout}

        with ThreadPoolExecutor(6) as ex:
            for r in ex.map(one, need):
                done[(r["pair"], r["order"])] = r
        write_jsonl(path, list(done.values()))
        votes = {}
        for r in done.values():
            votes.setdefault(r["pair"], []).append(r["winner"])
        # A side wins a pair only if it wins in both orders; anything else is a tie.
        return {p: (v[0] if len(v) == 2 and v[0] == v[1] else "tie") for p, v in votes.items()}

    # Judge check on the published blind look: a = base, b = harness.
    key = json.loads((RESULTS / "blind/key.json").read_text())
    human = {r["pair"]: r["natural"] for r in json.loads((RESULTS / "blind/ratings.json").read_text())}
    check_pairs = [{"pair": k["pair"], "a": k["paths"]["base"], "b": k["paths"]["harness"]} for k in key]
    jv = run(check_pairs, OUT / "judge_check.jsonl")
    hv = {}
    for k in key:
        h = human[k["pair"]]
        hv[k["pair"]] = "tie" if h not in ("A", "B") else ("a" if k[h] == "base" else "b")
    agree = sum(jv[p] == hv[p] for p in hv)
    print(f"judge check on the published 50: agrees with the person {agree}/50; judge says base more natural "
          f"{sum(v == 'a' for v in jv.values())}, harness {sum(v == 'b' for v in jv.values())}")  # fmt: skip

    # Validation: published harness (a) vs natural harness (b), chosen images, rep 0.
    _, gch = gpt_rows(args.set, 1)
    _, nch = nat_rows(args.set, args.planner)
    nmap = {(r["idx"], r["rep"]): r["path"] for r in nch}
    pairs = [{"pair": g["idx"], "a": g["path"], "b": nmap[(g["idx"], g["rep"])]} for g in gch
             if (g["idx"], g["rep"]) in nmap]  # fmt: skip
    v = run(pairs, OUT / f"judge_{args.set}_nat{args.planner}.jsonl")
    res = {"pairs": len(v), "published_more_natural": sum(x == "a" for x in v.values()),
           "natural_planner_more_natural": sum(x == "b" for x in v.values()),
           "tie": sum(x == "tie" for x in v.values()), "judge_check_agree_of_50": agree,
           "judge_check_base_natural": sum(x == "a" for x in jv.values()),
           "judge_check_harness_natural": sum(x == "b" for x in jv.values())}  # fmt: skip
    (OUT / f"judge_{args.set}_nat{args.planner}_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def genai(args):
    """GenAI-Bench guard: the natural planner's rewrite, one image per prompt, same seed as the published guard.

    With --instruction original, Claude runs the published planner's instruction instead (variant "orig<planner>"):
    the ablation that separates the planner model from the instruction."""
    from guard_genai import N_PROMPTS, boot
    from harness import PLANNER_SYSTEM

    meta = read_jsonl(RESULTS / "genai_bench/meta.jsonl")
    prompts = sorted(random.Random(0).sample(meta, N_PROMPTS), key=lambda r: r["idx"])
    original = args.instruction == "original"
    variant = f"{'orig' if original else 'nat'}{args.planner}"
    plans = plan([(str(p["idx"]), p["prompt"], None) for p in prompts], args.planner,
                 RESULTS / f"plans/genai_{variant}.jsonl", PLANNER_SYSTEM if original else None)  # fmt: skip
    cands = [{"idx": p["idx"], "prompt": p["prompt"], "gen_prompt": plans[str(p["idx"])], "seed": 1000 * p["idx"],
              "path": f"genai/{p['idx']:05d}/{variant}_0.png"} for p in prompts]  # fmt: skip
    generate(cands)
    vq = scored(cands, "vqascore", modal_cls("VQA").score, lambda c: {"path": c["path"], "prompt": c["prompt"]}, 48,
                OUT / f"genai_{variant}.vqa.jsonl")  # fmt: skip
    pk = scored(cands, "pickscore", modal_cls("Scorer").pickscore, lambda c: {"path": c["path"], "prompt": c["prompt"]},
                128, OUT / f"genai_{variant}.pick.jsonl")  # fmt: skip
    pub = {(r["idx"], r["arm"], r["k"]): r for r in read_jsonl(RESULTS / "genai_bench/rows.jsonl")}
    base = [pub[(c["idx"], "orig", 0)] for c in cands]
    gpt = [pub[(c["idx"], "plan", 0)] for c in cands]
    nat_v, nat_p = [vq[c["path"]] for c in cands], [pk[c["path"]] for c in cands]
    res = {"prompts": len(cands),
           "vqa": {"base": float(np.mean([b["vqa"] for b in base])),
                   "gpt_planner": float(np.mean([g["vqa"] for g in gpt])),
                   "this_arm": float(np.mean(nat_v))},
           "vqa_diff_vs_base": boot([n - b["vqa"] for n, b in zip(nat_v, base, strict=True)]),
           "vqa_diff_vs_gpt_planner": boot([n - g["vqa"] for n, g in zip(nat_v, gpt, strict=True)]),
           "pick_diff_vs_base": boot([n - b["pick"] for n, b in zip(nat_p, base, strict=True)]),
           "pick_diff_vs_gpt_planner": boot([n - g["pick"] for n, g in zip(nat_p, gpt, strict=True)])}  # fmt: skip
    other = OUT / f"genai_nat{args.planner}.vqa.jsonl"
    if original and other.exists():
        nat = {r["path"]: r["v"] for r in read_jsonl(other)}
        nv = [nat[f"genai/{c['idx']:05d}/nat{args.planner}_0.png"] for c in cands]
        res["vqa"]["natural_instruction_same_model"] = float(np.mean(nv))
        res["vqa_diff_vs_natural_instruction"] = boot([o - n for o, n in zip(nat_v, nv, strict=True)])
    (OUT / f"genai_{variant}_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def paired_pick(a_rows, b_rows):
    from guard_genai import boot

    a = {(r["idx"], r["rep"]): r["pick"] for r in a_rows}
    b = {(r["idx"], r["rep"]): r["pick"] for r in b_rows}
    keys = sorted(set(a) & set(b))
    return boot([b[k] - a[k] for k in keys])


def summary(args):
    reps = 1 if args.set == "val" else 2
    g1, gch = gpt_rows(args.set, reps)
    out = {"set": args.set, "reps": reps,
           "published_planner": {"single": geneval_summary(g1), "harness": geneval_summary(gch),
                                 "harness_checker": geneval_summary(gch, key="indep_pass"),
                                 "pick_harness": float(np.mean([r["pick"] for r in gch]))}}  # fmt: skip
    for planner in MODELS:
        if not (OUT / f"{args.set}_nat{planner}_pool.jsonl").exists():
            continue
        n1, nch = nat_rows(args.set, planner)
        rows = read_jsonl(OUT / f"{args.set}_nat{planner}_pool.jsonl")
        out[f"nat{planner}"] = {
            "single": geneval_summary(n1), "harness": geneval_summary(nch),
            "harness_checker": geneval_summary(nch, key="indep_pass"),
            "pick_harness": float(np.mean([r["pick"] for r in nch])),
            "paired_harness_vs_published": paired_diff(gch, nch),
            "paired_checker_vs_published": paired_diff(gch, nch, "indep_pass", "indep_pass"),
            "paired_single_vs_published": paired_diff(g1, n1),
            "pick_diff_vs_published": paired_pick(gch, nch),
            "candidates_per_output": len(rows) / len(nch),
        }  # fmt: skip
    (OUT / f"summary_{args.set}.json").write_text(json.dumps(out, indent=1))
    for name, s in out.items():
        if isinstance(s, dict) and "harness" in s:
            h, c = s["harness"]["overall"], s["harness_checker"]["overall"]
            line = (f"{name:<18} single {s['single']['overall'][0]:.3f}  harness {h[0]:.3f} [{h[1]:.3f}, {h[2]:.3f}]"
                    f"  checker {c[0]:.3f}  pick {s['pick_harness']:.2f}")  # fmt: skip
            if "paired_harness_vs_published" in s:
                d, e = s["paired_harness_vs_published"]["overall"], s["paired_checker_vs_published"]["overall"]
                line += (f"  | vs published: grader {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}], checker {e[0]:+.3f} "
                         f"[{e[1]:+.3f}, {e[2]:+.3f}], pick {s['pick_diff_vs_published'][0]:+.2f}")  # fmt: skip
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pool", "genai", "judge", "summary"])
    ap.add_argument("--set", default="val")
    ap.add_argument("--planner", default="opus", choices=list(MODELS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--instruction", default="natural", choices=["natural", "original"])
    args = ap.parse_args()
    {"pool": pool, "genai": genai, "judge": judge, "summary": summary}[args.stage](args)


if __name__ == "__main__":
    main()
