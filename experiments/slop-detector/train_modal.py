"""Run lesson 04's trainer on Modal H100s (both pilot sizes in parallel), then lay the results out like a Mac run.

    modal deploy modal_slop.py
    python train_modal.py --data pilot --runs 0.6b=Qwen/Qwen3-0.6B-Base 1.7b=Qwen/Qwen3-1.7B-Base
    python 05_eval.py --run runs/pilot-0.6b       # paired comparison with voi, from the cached scores

Uploads data/{train,val}.jsonl to the Modal volume under data/<name>/, trains, and scores every benchmark text once
with the final adapter. Writes runs/<data>-<tag>/{log.jsonl, val_scores.json, config.json, scores/<split>.json}.
The adapter stays on the volume (runs/<data>-<tag>/adapter), not in git.
"""

from __future__ import annotations

import argparse
import json
import time

from common import DATA, HERE, benchmark, empty_fences

SPLITS = ("test", "test_enron", "test_llama", "human_detectors", "epoch")
MICRO = {"0.6b": 12288, "1.7b": 6144, "4b": 2048, "4bsmoke": 4096, "4bckpt": 16384, "4bck2": 2048}


def main() -> None:
    import modal

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--runs", nargs="+", default=["0.6b=Qwen/Qwen3-0.6B-Base", "1.7b=Qwen/Qwen3-1.7B-Base"])
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--gpu", choices=["h100", "a100"], default="h100", help="a100 for Qwen3.5 (see modal_slop.py)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed on to 04_train.py")
    args = ap.parse_args()

    vol = modal.Volume.from_name("slop-detector")
    with vol.batch_upload(force=True) as up:
        for f in ("train.jsonl", "val.jsonl"):
            up.put_file(str(DATA / f), f"/data/{args.data}/{f}")
    print(f"uploaded data/{{train,val}}.jsonl to the volume as data/{args.data}/")
    evals = {s: [empty_fences(str(t)) for t in benchmark(s)["text"]] for s in args.splits}
    fn = modal.Function.from_name("slop-detector", "train" if args.gpu == "h100" else "train_a100")
    t0 = time.time()
    calls = {}
    for spec in args.runs:
        tag, base = spec.split("=")
        name = f"{args.data}-{tag}"
        calls[name] = (fn.spawn(args.data, base, name, evals, MICRO.get(tag, 16384), tuple(args.extra)), base)
        print(f"spawned {name} ({base})")
    for name, (call, _base) in calls.items():
        out = call.get()
        run = HERE / "runs" / name
        (run / "scores").mkdir(parents=True, exist_ok=True)
        (run / "log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out["log"]))
        (run / "val_scores.json").write_text(json.dumps(out["val_scores"]) + "\n")
        cfg = {
            **out["config"],
            "data": str(DATA),
            "modal_data": out["config"]["data"],
            "wall_seconds": round(time.time() - t0),
        }
        (run / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
        for split, s in out["scores"].items():
            (run / "scores" / f"{split}.json").write_text(json.dumps(s) + "\n")
        last = [x for x in out["log"] if "val" in x][-1]
        print(f"{name}: {cfg['seconds']} s training, val {last['val']}")


if __name__ == "__main__":
    main()
