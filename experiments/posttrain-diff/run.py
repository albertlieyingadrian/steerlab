"""Drive the post-training unit diff on Modal (profile modal4); every paid call goes through budget.paid.

    python run.py corpus            # fixed corpus: AI vs human texts, fit and eval splits -> data/corpus.jsonl
    python run.py twins             # noise-floor LoRAs: deslop SFT seeds 1, 2; random-label twin; neutral twin
    python run.py reads [PAIR ...]  # weight-only lens reads of every unit, both sides (default: all pairs)
    python run.py acts [PAIR ...]   # activation statistics on the fixed corpus, both sides
    python run.py pilot             # exploration: reads on layers 0, 18, 35 of two pairs (timing and sanity)
    python run.py fetch             # copy /vol/reads and /vol/acts to data/ (gitignored)

Pairs are (before, after). Every LoRA here is on Qwen/Qwen3-8B, so both sides read through the qwen3-8b lenses; the
public pair reads each side through its own lenses, with two lens controls.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

import budget

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
ROOT = HERE.parents[1]
SOURCE_TEXTS = DATA / "source_texts.jsonl"  # a copy of probes-vs-interp data/texts.jsonl (gitignored there too)
DESLOP = ROOT / "experiments/deslop-h2h/results"
APP = "posttrain-diff"
I8, B8 = "Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base"


def side(model=I8, adapter=None, lens=None):
    return {"model": model, "adapter": adapter, "lens": lens or model}


LORA = {  # name -> (adapter after, adapter before, role)
    "deslop-sft": ("deslop-ep2-sft", None, "objective"),
    "deslop-dpo": ("deslop-ep2-dpo", None, "objective"),
    "deslop-sft-s1": ("twin-sft-s1", None, "seed twin"),
    "deslop-sft-s2": ("twin-sft-s2", None, "seed twin"),
    "deslop-randlabel": ("twin-randlabel", None, "random-label twin"),
    "deslop-neutral": ("twin-neutral", None, "neutral twin"),
    "deslop-randown": ("twin-randlabel-own", None, "neutral random-label twin"),
    "deslop-neutral-s1": ("twin-neutral-s1", None, "neutral twin, seed 1"),
    "deslop-randown-s1": ("twin-randlabel-own-s1", None, "neutral random-label twin, seed 1"),
    "deslop-neutral-s2": ("twin-neutral-s2", None, "neutral twin, seed 2"),
    "deslop-randown-s2": ("twin-randlabel-own-s2", None, "neutral random-label twin, seed 2"),
    "slm-sft": ("slm-sft_edits", None, "objective"),
    "slm-shuffled": ("slm-sft_shuffled", None, "wrong-writer twin"),
    "slm-dpo": ("slm-dpo_sft_edits", None, "objective"),
    "slm-grpo60": ("slm-grpo_sft_edits-step_60", None, "objective"),
    "slm-grpo200": ("slm-grpo_sft_edits-step_200", None, "objective"),
    "slm-grpo200-vs-sft": ("slm-grpo_sft_edits-step_200", "slm-sft_edits", "RL only"),
}
PAIRS = {
    "pub": {"before": side(B8), "after": side(I8)},
    "pub-baselens": {"before": side(B8), "after": side(I8, lens=B8)},
    "pub-lensswap": {"before": side(I8, lens=B8), "after": side(I8)},
    **{k: {"before": side(adapter=b), "after": side(adapter=a)} for k, (a, b, _) in LORA.items()},
    "selfcheck": {"before": side(), "after": side(), "layers": [0, 18, 35]},  # known answer: every read unchanged
}
ACTS = ["pub", *LORA]
EST_READS, EST_ACTS = 1.2, 0.8  # $ per pair, refined from the pilot (results/pilot.json)


def fn(name):
    return modal.Function.from_name(APP, name)


def jread(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def corpus():
    rows = jread(SOURCE_TEXTS)
    rng = random.Random(0)
    out = []
    for split, src in (("fit", "fit"), ("eval", "test")):
        for label in (0, 1):
            pool = sorted((r for r in rows if r["set"] == src and r["label"] == label), key=lambda r: r["id"])
            rng.shuffle(pool)
            out += [{"id": r["id"], "text": r["text"], "label": label, "split": split, "gen": r.get("gen")}
                    for r in pool[:250]]  # fmt: skip
    DATA.mkdir(exist_ok=True)
    (DATA / "corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    manifest = {"source": "experiments/probes-vs-interp data/texts.jsonl (fit and test sets)", "n": len(out),
                "sha256": hashlib.sha256((DATA / "corpus.jsonl").read_bytes()).hexdigest(),
                "counts": {f"{s}-{lab}": sum(r["split"] == s and r["label"] == lab for r in out)
                           for s in ("fit", "eval") for lab in (0, 1)},
                "ids": [r["id"] for r in out]}  # fmt: skip
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "corpus_manifest.json").write_text(json.dumps(manifest, indent=1))
    print({k: v for k, v in manifest.items() if k != "ids"})


def twins():
    teacher = jread(DESLOP / "train/teacher.jsonl")
    original = {r["id"]: r["text"] for r in jread(DESLOP / "gen/original-train.jsonl")}
    ex = [{"prompt": r["prompt"], "reply": r["reply"]} for r in teacher]
    perm = list(range(len(ex)))
    rng = random.Random(0)
    while any(i == p for i, p in enumerate(perm)):  # a derangement: no prompt keeps its own reply
        rng.shuffle(perm)
    jobs = [
        ("twin-sft-s1", ex, 1),
        ("twin-sft-s2", ex, 2),
        (
            "twin-randlabel",
            [{"prompt": e["prompt"], "reply": ex[p]["reply"]} for e, p in zip(ex, perm, strict=True)],
            0,
        ),
        ("twin-neutral", [{"prompt": r["prompt"], "reply": original[r["id"]]} for r in teacher], 0),
        (
            "twin-randlabel-own",
            [{"prompt": r["prompt"], "reply": original[teacher[p]["id"]]} for r, p in zip(teacher, perm, strict=True)],
            0,
        ),
    ]
    # seed sets the LoRA's random init, so each SFT seed gets clean twins of the same seed (plan amendment 3)
    jobs += [
        (f"{name}-s{k}", rows, k)
        for k in (1, 2)
        for name, rows, _ in jobs
        if name in ("twin-neutral", "twin-randlabel-own")
    ]
    only = os.environ.get("TWINS")  # comma-separated subset
    jobs = [j for j in jobs if not only or j[0] in only.split(",")]
    f = fn("train_twin")

    def one(job):
        name, rows, seed = job
        return budget.paid(f"twin {name}", "H100", 0.6, f.remote, rows, name, seed=seed)

    with ThreadPoolExecutor(4) as pool:
        res = list(pool.map(one, jobs))

    def summary(r):
        losses = [x["loss"] for x in r["log"]]
        return {k: v for k, v in r.items() if k != "log"} | {"loss_first10": sum(losses[:10]) / 10,
                                                             "loss_last10": sum(losses[-10:]) / 10}  # fmt: skip

    RESULTS.mkdir(exist_ok=True)
    _merge(RESULTS / "twins.json", {r["name"]: summary(r) for r in res})
    print(json.dumps([summary(r) for r in res], indent=1))


def _map(job, est, names, call):
    out = {}

    def one(name):
        return name, budget.paid(f"{job} {name}", "H100", est, call, name)

    with ThreadPoolExecutor(6) as pool:
        for name, r in pool.map(one, names):
            out[name] = r
            print(json.dumps(r)[:600], flush=True)
    return out


def _merge(path, res):
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(old | res, indent=1))


def reads(names):
    f = fn("reads")
    _merge(RESULTS / "reads_runs.json", _map("reads", EST_READS, names, lambda n: f.remote({"name": n, **PAIRS[n]})))


def rulers():
    """Extra AI-register rulers already used on Qwen3-8B: deslop-h2h's register directions (layers 12, 18, 24) and the
    probes-vs-interp layer-18 AI-vs-human probe (minus deslop-h2h's probe_neg)."""
    u = json.loads((DESLOP / "units.json").read_text())
    reg = {int(layer): v["vector"] for layer, v in u["register"].items()}
    return {"register": reg, "probe18": {18: [-x for x in u["probe_neg"]]}}


def acts(names):
    docs = [{k: d[k] for k in ("id", "text", "label", "split")} for d in jread(DATA / "corpus.jsonl")]
    f = fn("acts")
    rl = rulers()
    _merge(
        RESULTS / "acts_runs.json", _map("acts", EST_ACTS, names, lambda n: f.remote({"name": n, **PAIRS[n]}, docs, rl))
    )


def reads32(names):
    """The same reads in fp32 (precision check, added after the first full run in fp16)."""
    f = fn("reads")
    call = lambda n: f.remote({"name": n, **PAIRS[n], "fp32": True})  # noqa: E731
    _merge(RESULTS / "reads32_runs.json", _map("reads32", 0.4, names, call))


def pilot():
    f = fn("reads")
    out = _map("pilot", 0.5, ["pub", "deslop-sft"],
               lambda n: f.remote({"name": f"pilot-{n}", **PAIRS[n], "layers": [0, 18, 35]}))  # fmt: skip
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pilot.json").write_text(json.dumps(out, indent=1))


def fetch():
    DATA.mkdir(exist_ok=True)
    env = {**os.environ, "MODAL_PROFILE": budget.profile()}
    for sub in ("reads", "reads32", "acts"):
        subprocess.run([budget.MODAL_BIN, "volume", "get", "posttrain-diff", f"/{sub}", str(DATA), "--force"],
                       check=True, env=env)  # fmt: skip


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "corpus":
        corpus()
    elif cmd == "twins":
        twins()
    elif cmd == "reads":
        reads(args or list(PAIRS))
    elif cmd == "reads32":
        reads32(args or list(PAIRS))
    elif cmd == "acts":
        acts(args or ACTS)
    elif cmd == "pilot":
        pilot()
    elif cmd == "fetch":
        fetch()
    else:
        raise SystemExit(__doc__)
