"""The defence: retrain the 9B detector on the rewriter's outputs and measure what it catches again.

    python defend.py gen       # 2 rewrites of each of the 792 train sources: full-s0 final, and prompt-only (control)
    python defend.py train     # one pass of f9-9b's LoRA + head on each set (bucket 3) + 3,000 replay rows (A100)
    python defend.py score     # untouched, defended and control detectors on held-out rewrites, sources, the 318
                               # human passages and 400 rows of the 9B's own validation split
                               # -> results/defend/scores.json, results/defend/summary.json

Training rows come only from the train split; everything scored is held out. Flag rates are reported at the original
threshold (s9 >= 0.372) and at a threshold refit so each detector flags 1% of the 318 human passages.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess

import modal
import numpy as np

import budget
from common import (
    APP,
    EP3_THRESHOLDS,
    HERE,
    RESULTS,
    REWRITE_PROMPT,
    SEED,
    boot,
    cell,
    human_passages,
    read_jsonl,
    save_json,
    sources,
    write_jsonl,
)
from evaluate import GEN, all_groups

DATA = HERE / "data" / "defend"
SOURCE_RUN = "full-s0"
DETECTORS = {"untouched": "/sd/runs/f9-9b", "defended": "/vol/defend/full-def", "control": "/vol/defend/control-def"}


def final_step(run: str) -> int:
    curve = [json.loads(x) for x in (RESULTS / "runs" / run / "curve.jsonl").read_text().splitlines() if x.strip()]
    return max(r["step"] for r in curve if r.get("checkpoint"))


def gen() -> None:
    S = sources("train")
    prompts = [REWRITE_PROMPT.format(source=s["text"]) for s in S]
    fn = modal.Function.from_name(APP, "rewrite")
    step = final_step(SOURCE_RUN)
    args = [(f"{SOURCE_RUN}/step{step}", prompts, 2), ("", prompts, 2)]

    def run():
        return list(fn.starmap(args))

    outs = budget.paid("defend-gen", "H100", 1.5, run, gpus=2)
    for name, out in zip(("full", "control"), outs, strict=True):
        rows = [{"sid": s["sid"], "k": k, **c} for s, cs in zip(S, out, strict=True) for k, c in enumerate(cs)]
        write_jsonl(DATA / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} rewrites (source {SOURCE_RUN}@{step} for full)")


def train() -> None:
    fn = modal.Function.from_name(APP, "defend")
    sets = {}
    for name in ("full", "control"):
        rows = [{"text": r["text"], "bucket": 3} for r in read_jsonl(DATA / f"{name}.jsonl") if r["text"].strip()]
        sets[f"{name}-def"] = rows

    def run():
        return list(fn.starmap([(n, rows, 3000) for n, rows in sets.items()]))

    outs = budget.paid("defend-train", "A100-80GB", 6.0, run, gpus=2)
    save_json("defend/train_logs.json", outs)
    for o in outs:
        print(o["name"], o["steps"], "steps", o["seconds"], "s", o["log"][-1])


def f9_val(n: int = 400) -> list[dict]:
    path = DATA / "f9_val.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([budget.MODAL_BIN, "volume", "get", "--force", "slop-detector", "data/f9/val.jsonl", str(path)],
                       check=True, capture_output=True)  # fmt: skip
    rows = read_jsonl(path)
    return random.Random(SEED).sample(rows, n)


def score() -> None:
    texts: dict[str, str] = {}
    groups = [g for g in all_groups() if (GEN / f"{g}.jsonl").exists()]
    for g in groups:
        texts |= {f"{g}/{r['sid']}": r["text"] for r in read_jsonl(GEN / f"{g}.jsonl")}
    texts |= {f"source/{s['sid']}": s["text"] for s in sources("heldout") + sources("heldout4b")}
    texts |= {f"human/{h['source']}/{h['id']}": h["text"] for h in human_passages()}
    val = f9_val()
    texts |= {f"val/{i}": r["text"] for i, r in enumerate(val)}
    keys = list(texts)
    path = RESULTS / "defend" / "scores.json"
    S = json.loads(path.read_text()) if path.exists() else {}

    def run():
        out = {}
        for name, det in DETECTORS.items():
            todo = [k for k in keys if name not in S.get(k, {})]
            if not todo:
                continue
            R = modal.Cls.from_name(APP, "Reward")(detector=det)
            chunks = [todo[i : i + 400] for i in range(0, len(todo), 400)]
            res = [x for c in R.detect.map([[texts[k] for k in c] for c in chunks]) for x in c]
            out[name] = dict(zip(todo, res, strict=True))
        return out

    got = budget.paid("defend-score", "L40S", 3.0, run, gpus=4)
    for name, res in got.items():
        for k, d in res.items():
            S.setdefault(k, {})[name] = {"s9": d["s9"], "bucket": int(np.argmax(d["logits"]))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(S) + "\n")
    summarise(S, groups, val)


def summarise(S: dict, groups: list[str], val: list[dict]) -> None:
    hum = [k for k in S if k.startswith("human/")]
    out: dict = {"detectors": {}, "groups": groups}
    for name in DETECTORS:
        if not all(name in S[k] for k in hum):
            continue
        h = np.array([S[k][name]["s9"] for k in hum])
        refit = float(np.quantile(h, 0.99))
        d = {"refit_threshold": refit, "human_fpr_orig": float(np.mean(h >= EP3_THRESHOLDS["nine_b"])),
             "val_bucket_acc": float(np.mean([S[f"val/{i}"][name]["bucket"] == int(r["bucket"])
                                              for i, r in enumerate(val)])), "flags": {}}  # fmt: skip
        for g in [*groups, "source"]:
            for split in ("heldout", "heldout4b"):
                ks = [k for k in S if k.startswith(f"{g}/{split}-")]
                if not ks or name not in S[ks[0]]:
                    continue
                by: dict = {}
                for k in ks:  # cluster: the prompt (a held-out prompt's two sources stay together)
                    pid = k.split("/", 1)[1].rsplit("-", 1)[0] if split == "heldout" else k
                    by.setdefault(pid, []).append(S[k][name]["s9"])
                orig = np.array([np.mean([x >= EP3_THRESHOLDS["nine_b"] for x in v]) for v in by.values()])
                ref = np.array([np.mean([x >= refit for x in v]) for v in by.values()])
                d["flags"][f"{g}|{split}"] = {"orig": cell(boot(orig)), "refit": cell(boot(ref))}
        out["detectors"][name] = d
    final = f"{SOURCE_RUN}@{final_step(SOURCE_RUN)}|heldout"
    D = out["detectors"]
    if "defended" in D and "control" in D:
        out["full_flag_orig"] = D["defended"]["flags"][final]["orig"]["value"]
        out["control_full_flag_orig"] = D["control"]["flags"][final]["orig"]["value"]
        out["human_fpr_orig"] = round(D["defended"]["human_fpr_orig"], 4)
    save_json("defend/summary.json", out)
    for name, d in D.items():
        print(name, f"human FPR {d['human_fpr_orig']:.3f}", f"val acc {d['val_bucket_acc']:.3f}",
              f"refit {d['refit_threshold']:.3f}")  # fmt: skip
        for k, v in d["flags"].items():
            if k.endswith("|heldout"):
                print(f"   {k:28} orig {v['orig']['value']:.2f}  refit {v['refit']['value']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["gen", "train", "score"])
    {"gen": gen, "train": train, "score": score}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
