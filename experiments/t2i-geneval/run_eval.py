"""Plain single-sample evaluation of one model: generate, then grade (GenEval or CompBench B-VQA) and PickScore.

  python run_eval.py --set official --samples 4 --run schnell_official
  python run_eval.py --set compbench --samples 1 --run schnell_compbench

Writes results/<run>/rows.jsonl (one row per image) and results/<run>/summary.json.
"""

import argparse
import json
import time
from pathlib import Path

from common import HERE, PRICE, RESULTS, chunks, fmt_summary, geneval_summary, load_set, modal_cls, write_jsonl

COMPBENCH = ["color", "shape", "texture"]


def compbench_rows(per_cat):
    rows = []
    for cat in COMPBENCH:
        lines = [
            line.strip()
            for line in Path(HERE / f"data/compbench/{cat}_val.txt").read_text().splitlines()
            if line.strip()
        ][:per_cat]
        rows += [{"tag": f"cb_{cat}", "prompt": p} for p in lines]
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="official")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--lora", default=None, help="LoRA folder inside the volume")
    ap.add_argument("--cb-per-cat", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke tests)")
    ap.add_argument("--tags", default="", help="comma-separated GenEval tags to keep (default: all)")
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--indep", action="store_true", help="also run the independent checker (reward-hacking guard)")
    ap.add_argument(
        "--plan", action="store_true", help="generate from the harness planner's rewrite (graded on the original)"
    )
    args = ap.parse_args()

    prompts = compbench_rows(args.cb_per_cat) if args.set == "compbench" else load_set(args.set)
    if args.tags:
        prompts = [p for p in prompts if p["tag"] in args.tags.split(",")]
    if args.limit:
        prompts = prompts[: args.limit]
    plans = {}
    if args.plan:
        from harness import plan

        plans = plan(
            [(str(p["idx"]), p["prompt"], None) for p in prompts], cache_path=RESULTS / f"plans/{args.set}.jsonl"
        )
    rows = []
    for p in prompts:
        for k in range(args.samples):
            rows.append(
                {
                    "idx": p["idx"],
                    "tag": p["tag"],
                    "prompt": p["prompt"],
                    "gen_prompt": plans.get(str(p["idx"]), p["prompt"]),
                    "meta": {k2: v for k2, v in p.items() if k2 not in ("idx", "split")},
                    "seed": 1000 * p["idx"] + k,
                    "path": f"{args.run}/{p['idx']:05d}/{k}.png",
                }
            )
    t0 = time.time()
    gen = modal_cls("Gen", model_id=args.model)
    jobs = [{"prompt": r["gen_prompt"], "seed": r["seed"], "path": r["path"]} for r in rows]
    kw = {"steps": args.steps, "guidance": args.guidance, "size": args.size, "lora": args.lora}
    gstats = list(gen.generate.map(chunks(jobs, args.gen_chunk), kwargs=kw))
    gen_secs = sum(s["seconds"] for s in gstats)
    n_new = sum(s["n"] for s in gstats)
    print(f"generated {n_new} new images, {gen_secs:.0f} GPU-s in the loop ({time.time() - t0:.0f}s wall)")

    scorer = modal_cls("Scorer")
    picks = list(scorer.pickscore.map(chunks([{"path": r["path"], "prompt": r["prompt"]} for r in rows], 128)))
    for r, s in zip(rows, [x for c in picks for x in c], strict=False):
        r["pick"] = s
    if args.set == "compbench":
        res = list(scorer.bvqa.map(chunks([{"path": r["path"], "prompt": r["prompt"]} for r in rows], 64)))
        for r, s in zip(rows, [x for c in res for x in c], strict=False):
            r["bvqa"] = s["bvqa"]
    else:
        grader = modal_cls("Grader")
        res = list(grader.grade.map(chunks([{"path": r["path"], "meta": r["meta"]} for r in rows], 64)))
        for r, g in zip(rows, [x for c in res for x in c], strict=False):
            assert g["path"] == r["path"] and "error" not in g, g
            r.update(correct=g["correct"], strict_correct=g["strict_correct"], score=g["score"], reason=g["reason"])
        if args.indep:
            res = list(scorer.indep_check.map(chunks([{"path": r["path"], "meta": r["meta"]} for r in rows], 64)))
            for r, v in zip(rows, [x for c in res for x in c], strict=False):
                r.update(indep_pass=v["pass"], indep_score=v["score"])

    out = RESULTS / args.run
    write_jsonl(out / "rows.jsonl", [{k: v for k, v in r.items() if k != "meta"} for r in rows])
    summary = {"args": vars(args), "n_images": len(rows), "gen_gpu_seconds_new": gen_secs, "new_images": n_new}
    if n_new:
        summary["sec_per_image"] = gen_secs / n_new
        summary["usd_per_image_gen"] = gen_secs / n_new * PRICE["H100"]
    elif (out / "summary.json").exists():  # a re-score: keep the timing of the run that drew the images
        prev = json.loads((out / "summary.json").read_text())
        for k in ("gen_gpu_seconds_new", "new_images", "sec_per_image", "usd_per_image_gen"):
            if k in prev:
                summary[k] = prev[k]
    summary["pickscore_mean"] = sum(r["pick"] for r in rows) / len(rows)
    if args.set == "compbench":
        for cat in COMPBENCH:
            v = [r["bvqa"] for r in rows if r["tag"] == f"cb_{cat}"]
            if v:
                summary[f"bvqa_{cat}"] = sum(v) / len(v)
        print({k: round(v, 4) for k, v in summary.items() if k.startswith("bvqa")})
    else:
        s = geneval_summary(rows)
        summary["geneval"] = s
        summary["geneval_strict"] = geneval_summary(rows, key="strict_correct")
        if args.indep:
            summary["geneval_indep"] = geneval_summary(rows, key="indep_pass")
        print(fmt_summary(s))
    print(f"PickScore mean {summary['pickscore_mean']:.3f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
