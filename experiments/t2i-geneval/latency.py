"""Wall-clock seconds per image, base vs harness, one request at a time on warm Modal containers.

  python latency.py --n 20

Each request runs the way a user would get one image: base = one FLUX.1-schnell call; harness = planner call, 4
images in one batch, the checker and PickScore on all 4, and (if none passes) a re-plan and 4 more. Containers are
warmed first, so cold starts are excluded; Modal call overhead and the shared volume are included.
Writes results/latency.json.
"""

import argparse
import json
import time

import numpy as np

from common import RESULTS, load_set, modal_cls
from harness import PLANNER_MODEL, PLANNER_SYSTEM, RETRY_NOTE, openai_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    prompts = load_set("official")
    step = len(prompts) // args.n
    prompts = prompts[::step][: args.n]
    gen = modal_cls("Gen", model_id="black-forest-labs/FLUX.1-schnell")
    scorer, client = modal_cls("Scorer"), openai_client()
    kw = {"steps": 4, "guidance": 0.0, "size": 1024, "batch": 8}
    tag = int(time.time())

    def plan(text, why=None):
        sys_msg = PLANNER_SYSTEM + (RETRY_NOTE.format(reason=why) if why else "")
        r = client.chat.completions.create(
            model=PLANNER_MODEL, messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": text}]
        )
        return r.choices[0].message.content.strip()

    def check(paths, p, meta):
        ind = scorer.indep_check.remote([{"path": q, "meta": meta} for q in paths])
        pk = scorer.pickscore.remote([{"path": q, "prompt": p["prompt"]} for q in paths])
        return ind, pk

    # Warm every container this path uses.
    warm = prompts[0]
    meta0 = {k: v for k, v in warm.items() if k not in ("idx", "split")}
    gen.generate.remote([{"prompt": warm["prompt"], "seed": 1, "path": f"latency/{tag}/warm.png"}], **kw)
    check([f"latency/{tag}/warm.png"], warm, meta0)
    plan(warm["prompt"])

    base_s, rows = [], []
    for p in prompts:
        meta = {k: v for k, v in p.items() if k not in ("idx", "split")}
        t = time.time()
        gen.generate.remote([{"prompt": p["prompt"], "seed": 7, "path": f"latency/{tag}/{p['idx']}_base.png"}], **kw)
        base_s.append(time.time() - t)

        r = {"idx": p["idx"], "tag": p["tag"]}
        t0 = time.time()
        text = plan(p["prompt"])
        r["plan_s"] = time.time() - t0
        paths = [f"latency/{tag}/{p['idx']}_{k}.png" for k in range(4)]
        t1 = time.time()
        gen.generate.remote([{"prompt": text, "seed": 100 + k, "path": q} for k, q in enumerate(paths)], **kw)
        r["gen_s"] = time.time() - t1
        t2 = time.time()
        ind, _ = check(paths, p, meta)
        r["check_s"] = time.time() - t2
        r["retry"] = not any(x["pass"] for x in ind)
        if r["retry"]:
            want = {x["class"]: x["count"] for x in meta["include"]}
            text = plan(p["prompt"], f"wanted counts {want}, a detector found {ind[0]['counts']}")
            paths2 = [f"latency/{tag}/{p['idx']}_r{k}.png" for k in range(4)]
            gen.generate.remote([{"prompt": text, "seed": 200 + k, "path": q} for k, q in enumerate(paths2)], **kw)
            check(paths2, p, meta)
        r["total_s"] = time.time() - t0
        rows.append(r)
        print(r, f"base {base_s[-1]:.2f}s")

    def med(xs):
        return {"median": float(np.median(xs)), "p90": float(np.percentile(xs, 90)), "mean": float(np.mean(xs))}

    out = {
        "n": len(rows),
        "base_s": med(base_s),
        "harness_total_s": med([r["total_s"] for r in rows]),
        "plan_s": med([r["plan_s"] for r in rows]),
        "gen4_s": med([r["gen_s"] for r in rows]),
        "check_s": med([r["check_s"] for r in rows]),
        "retries": sum(r["retry"] for r in rows),
        "rows": rows,
        "base_rows_s": base_s,
    }
    (RESULTS / "latency.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "base_rows_s")}, indent=1))


if __name__ == "__main__":
    main()
