"""Track A generation: every arm on Qwen3-8B (thinking off), written to results/gen/<arm>.jsonl.

    python gen.py --list
    python gen.py original-train plain-train          # inputs for the register direction (288 training prompts)
    python gen.py original plain sampler ...          # held-out arms: 48 prompts x 2 samples

Arms that steer read results/units.json (units.py) and results/tune_choice.json (tune.py); adapters come from
train.py. Rows: id, sample, text, n_tokens, finished (ended before the 512-token cap), and for sampler arms
backtracks and accepted. results/runs.json records each arm's settings, wall time and Modal profile.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import modal
import numpy as np

import budget
from common import APP, GEN, N_SAMPLES, PLAIN_PROMPT, RESULTS, SEED, held_out, load_json, training, write_jsonl

N_RANDOM = 3
STEER_ARMS = ("register", "probe", "sae-dir", "sae-abl")
_LOCK = threading.Lock()


def intervention(kind: str, setting: dict | None = None) -> dict:
    """The residual intervention for a steering arm at a setting ({layer, strength} or {c}); default = tune choice."""
    U = load_json("units.json")
    s = setting or load_json("tune_choice.json")["choice"][kind]["setting"]
    if kind == "register":
        d = U["register"][str(s["layer"])]
        return {"kind": "add", "layer": int(s["layer"]), "vector": d["vector"],
                "norm": float(s["strength"]) * float(d["class_gap"])}  # fmt: skip
    if kind in ("probe", "sae-dir"):
        vec = U["probe_neg"] if kind == "probe" else U["sae_dir_neg"]
        return {"kind": "add", "layer": 18, "vector": vec, "norm": float(s["c"]) / 100.0 * U["norm_ref"]}
    if kind == "sae-abl":
        return {"kind": "sae", "layer": 18, "features": U["feats"], "coef": float(s["c"])}
    raise KeyError(kind)


def random_intervention(kind: str, r: int) -> dict:
    """A norm-matched random direction at the chosen setting of `kind` (seeded by kind and index)."""
    iv = intervention(kind)
    rng = np.random.default_rng([SEED, STEER_ARMS.index(kind), r])
    return {**iv, "vector": rng.standard_normal(4096).astype(np.float32).tolist(), "norm": abs(iv["norm"])}


def best_sae() -> str:
    ch = load_json("tune_choice.json")["choice"]
    return min(("sae-dir", "sae-abl"), key=lambda k: ch[k]["dev_habit"])


def arms() -> dict[str, dict]:
    """name -> {rows, n, adapter, system, sampler, iv} (iv resolved lazily)."""
    a: dict[str, dict] = {
        "original-train": {"rows": "training", "n": 1},
        "plain-train": {"rows": "training", "n": 1, "system": PLAIN_PROMPT},
        "teacher-train": {"rows": "training", "n": 3, "system": PLAIN_PROMPT, "sampler": True},
        "original": {},
        "plain": {"system": PLAIN_PROMPT},
        "sampler": {"sampler": True},
        "plain+sampler": {"system": PLAIN_PROMPT, "sampler": True},
        "plain+register": {"system": PLAIN_PROMPT, "iv": ("register",)},
        "plain+sae": {"system": PLAIN_PROMPT, "iv": ("best-sae",)},
        "sft": {"adapter": "ep2-sft"},
        "dpo": {"adapter": "ep2-dpo"},
        "sft+sampler": {"adapter": "ep2-sft", "sampler": True},
    }
    for k in STEER_ARMS:
        a[k] = {"iv": (k,)}
    for k in ("register", "probe", "sae-dir"):
        for r in range(N_RANDOM):
            a[f"random-{k}-{r}"] = {"iv": ("random", k, r)}
    return a


def resolve(spec: dict) -> dict | None:
    iv = spec.get("iv")
    if not iv:
        return None
    if iv[0] == "random":
        return random_intervention(iv[1], iv[2])
    return intervention(best_sae() if iv[0] == "best-sae" else iv[0])


def generate(name: str, rows: list[dict], *, n: int, adapter: str = "", system: str | None = None,
             sampler: bool = False, iv: dict | None = None, seed: int = SEED, chunks: int = 4,
             projected: float = 1.0) -> list[dict]:  # fmt: skip
    """Split the rows over up to `chunks` containers; same seed per sample everywhere."""
    runner = modal.Cls.from_name(APP, "Runner")(adapter=adapter)
    size = max(1, -(-len(rows) // chunks))
    parts = [rows[i : i + size] for i in range(0, len(rows), size)]
    job = {"n": n, "temperature": 0.7, "max_new_tokens": 512, "seed": seed, "system": system, "sampler": sampler,
           "intervention": iv, "batch": 48}  # fmt: skip

    def one(part):
        return runner.generate.remote({**job, "rows": part})

    def all_parts():
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            return [r for out in pool.map(one, parts) for r in out]

    return budget.paid(f"gen-{name}", "H100", projected, all_parts, gpus=len(parts))


def run(name: str) -> None:
    spec = arms()[name]
    rows = training() if spec.get("rows") == "training" else held_out()
    rows = [{"id": p["id"], "prompt": p["prompt"]} for p in rows]
    n = spec.get("n", N_SAMPLES)
    iv = resolve(spec)
    t0 = time.time()
    proj = 4.0 if spec.get("sampler") else 1.0
    out = generate(name, rows, n=n, adapter=spec.get("adapter", ""), system=spec.get("system"),
                   sampler=bool(spec.get("sampler")), iv=iv, projected=proj,
                   chunks=8 if spec.get("sampler") else 4)  # fmt: skip
    out.sort(key=lambda r: (r["sample"], r["id"]))
    write_jsonl(GEN / f"{name}.jsonl", [{"arm": name, **r} for r in out])
    meta_path = RESULTS / "runs.json"
    _LOCK.acquire()
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta[name] = {"model": "Qwen/Qwen3-8B", "adapter": spec.get("adapter", ""), "system": bool(spec.get("system")),
                  "sampler": bool(spec.get("sampler")),
                  "intervention": None if iv is None else {k: v for k, v in iv.items() if k != "vector"},
                  "replies": len(out), "finished": sum(r["finished"] for r in out),
                  "seconds": round(time.time() - t0), "profile": budget.profile(),
                  "at": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    _LOCK.release()
    print(f"{name}: {len(out)} replies, {meta[name]['finished']} finished, {meta[name]['seconds']} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--parallel", type=int, default=1)
    args = ap.parse_args()
    if args.list:
        print("\n".join(arms()))
        return
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(run, args.arms))


if __name__ == "__main__":
    main()
