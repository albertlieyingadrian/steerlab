"""Launcher: spawn deployed Modal functions under the budget guard, save call ids, wait, record spend.

export MODAL_PROFILE=credits30
modal deploy experiments/sae-intersectional-bias/modal_app.py
python run.py prefetch
python run.py cache --tag pilot --limit 512
python run.py cache --tag full
python run.py wait <call_id> <job> <gpu>       # resume waiting on a saved call
python run.py pull --tag full                  # copy /out/<tag> to results/cache/<tag> (gitignored)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import modal

import budget

HERE = Path(__file__).resolve().parent
CALLS = HERE / "results" / "calls.jsonl"
APP = "sae-intersectional-bias"


def save_call(job, call_id, gpu):
    CALLS.parent.mkdir(parents=True, exist_ok=True)
    with CALLS.open("a") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": job,
                             "call_id": call_id, "gpu": gpu, "profile": budget.profile()}) + "\n")  # fmt: skip


def wait(call_id, job, gpu, t0=None):
    fc = modal.FunctionCall.from_id(call_id)
    t0 = t0 or time.time()
    try:
        res = fc.get()
    finally:
        budget.record(job, gpu, time.time() - t0, note=f"call {call_id} (wall time from spawn/wait start)")
    print(json.dumps(res, indent=1)[:4000])
    return res


def spawn(fn_name, job, gpu, max_h, exp_h, **kw):
    budget.guard(job, gpu, max_h, exp_h)
    fn = modal.Function.from_name(APP, fn_name)
    call = fn.spawn(**kw)
    save_call(job, call.object_id, gpu)
    print("spawned", job, call.object_id, flush=True)
    return call.object_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == "prefetch":
        cid = spawn("prefetch", "prefetch", "cpu", 1.0, 0.3)
        if not a.no_wait:
            wait(cid, "prefetch", "cpu", t0)
    elif a.cmd == "cache":
        prompts = json.loads((HERE / "data" / "prompts.json").read_text())
        texts = [p["text"] for s in prompts["sets"].values() for p in s] + prompts["imagenette_prompts"]
        texts += [t for v in prompts["group_prompts"].values() for t in v]
        cid = spawn("cache_clip", f"cache_clip:{a.tag}", "A10G", 2.0, 0.1 if a.limit else 0.7, tag=a.tag,
                    limit=a.limit, texts=texts)  # fmt: skip
        if not a.no_wait:
            wait(cid, f"cache_clip:{a.tag}", "A10G", t0)
    elif a.cmd == "llava":
        import llava_tools

        cache = HERE / "results" / "cache" / a.args[0]  # the CLIP cache tag that holds val labels and LEACE stats
        rows = llava_tools.build_prompts(json.loads((cache / "val_labels.json").read_text()))
        er = {
            k: (v[0].tolist(), v[1].tolist())
            for k, v in llava_tools.leace_matrices(cache / "leace22_stats.npz").items()
        }
        if a.tag == "pilot":
            step = len(rows) // 400
            rows = rows[::step][:400]
            arms = ["llava", "llava_repeat", "l23_paper@1.0", "l23_zero", "l22_sae@1.0", "l22_leace", "text_named"]
            mme_arms, mme_limit, exp_h = ["llava"], 64, 0.3
        else:
            arms = ["llava", "llava_repeat", "l23_paper@0.6", "l23_paper@1.0", "l23_zero", "l22_sae@0.6", "l22_sae@1.0", "l22_recon", "l22_leace",
                    "l22_rand0", "l22_rand1", "l22_rand2", "text_named"]  # fmt: skip
            mme_arms, mme_limit, exp_h = ["llava", "l22_sae@1.0", "l22_recon", "l22_leace", "l22_rand0"], None, 2.5
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / f"llava_rows_{a.tag}.json").write_text(json.dumps(rows))
        cid = spawn("llava_eval", f"llava:{a.tag}", "A100-40GB", 5.0, exp_h, tag=a.tag, rows=rows, arms=arms,
                    mme_arms=mme_arms, erasers=er, mme_limit=mme_limit,
                    identity_n=384 if a.tag != "pilot" else 192, batch=32)  # fmt: skip
        if not a.no_wait:
            res = wait(cid, f"llava:{a.tag}", "A100-40GB", t0)
            (HERE / "results" / f"llava_{a.tag}.json").write_text(json.dumps(res))
    elif a.cmd == "wait":
        res = wait(a.args[0], a.args[1], a.args[2])
        if a.args[1].startswith("llava:"):
            (HERE / "results" / f"llava_{a.args[1].split(':')[1]}.json").write_text(json.dumps(res))
    elif a.cmd == "pull":
        dst = HERE / "results" / "cache"
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(Path(budget.MODAL_BIN)), "volume", "get", "--force", APP, f"out/{a.tag}/", str(dst)],
                       check=True)  # fmt: skip
    else:
        raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
