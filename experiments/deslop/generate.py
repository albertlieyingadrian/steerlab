"""Generate replies for one or more named runs on Modal and save them to results/gen/<run>.jsonl.

    modal deploy modal_deslop.py
    python generate.py base-train plain-train        # inputs for the slop list, the direction and the teacher
    python generate.py base plain sampler neurons     # held-out runs (see RUNS)
    python generate.py --list

A run is a model (base, a LoRA adapter on the volume, or another checkpoint), a prompt set, an optional system prompt,
optional steering and neuron edits, and whether the backtracking sampler is on. Held-out runs sample one reply per
prompt at temperature 0.7 with seed 0, so every method sees the same 48 prompts and the same random seed.
Steered runs read the chosen layer and strength from results/steer_choice.json (written by tune_steer.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import PLAIN_PROMPT, dev, held_out, training

GEN = HERE / "results" / "gen"
BASE = "Qwen/Qwen3-4B-Instruct-2507"
KANNIKE = "erikkannike/qwen3-4b-deslop-v4"
# 100toolkit's four neuron flips (notxy-flip4.json in erikkannike/qwen3-4b-deslop-v4): layer, neuron, scale.
FLIP4 = [[32, 4852, -1.0], [33, 348, -1.0], [28, 1121, -1.0], [32, 5418, -1.0]]
# A control: four other neurons in the same layers (four draws of random.Random(0).randrange(9728)), flipped.
RANDOM4 = [[32, 6311, -1.0], [33, 6890, -1.0], [28, 663, -1.0], [32, 4242, -1.0]]

PROMPT_SETS = {"held_out": held_out, "training": training, "dev": dev}


def steer_spec(kind: str = "register") -> dict:
    """The chosen steering setting (tune_steer.py): the register direction, or a random direction of the same size."""
    choice = json.loads((HERE / "results" / "steer_choice.json").read_text())
    directions = json.loads((HERE / "results" / "directions.json").read_text())
    d = directions[str(choice["layer"])]
    vector = d["vector"] if kind == "register" else d["random_vector"]
    return {
        "vector": vector,
        "layer": choice["layer"],
        "class_gap": d["class_gap"],
        "strength": choice["strength"],
    }


RUNS: dict[str, dict] = {
    # inputs
    "base-train": {"rows": "training"},
    "plain-train": {"rows": "training", "system": PLAIN_PROMPT},
    # held-out methods
    "base": {},
    "plain": {"system": PLAIN_PROMPT},
    "sampler": {"sampler": True},
    "neurons": {"edits": FLIP4},
    "neurons-random": {"edits": RANDOM4},
    "steer": {"steer": "register"},
    "steer-random": {"steer": "random"},
    "plain+neurons": {"system": PLAIN_PROMPT, "edits": FLIP4},
    "plain+steer": {"system": PLAIN_PROMPT, "steer": "register"},
    "plain+sampler": {"system": PLAIN_PROMPT, "sampler": True},
    "kannike": {"model_id": KANNIKE},
    "lora": {"adapter": "deslop-v1"},
    "lora-r2": {"adapter": "deslop-r2b"},  # round 2, chosen on the dev prompts (teacher.py round2)
}


def job_for(name: str) -> tuple[dict, dict, list[dict]]:
    """(Runner parameters, job, prompt rows) for a run name."""
    spec = dict(RUNS[name])
    rows = PROMPT_SETS[spec.pop("rows", "held_out")]()
    params = {
        "model_id": spec.pop("model_id", BASE),
        "adapter": spec.pop("adapter", ""),
    }
    if isinstance(spec.get("steer"), str):
        spec["steer"] = steer_spec(spec["steer"])
    job = {"n": 1, "temperature": 0.7, "max_new_tokens": 512, "seed": 0, **spec}
    return params, job, rows


def run(name: str, params: dict, job: dict, rows: list[dict], *, chunk: int = 6) -> list[dict]:
    """Run a job on Modal: one call, or for the sampler, chunks of `chunk` prompts on parallel containers."""
    runner = modal.Cls.from_name("deslop", "Runner")(**params)
    slim = [{"id": r["id"], "prompt": r["prompt"]} for r in rows]
    t0 = time.time()
    if job.get("sampler"):
        jobs = [{**job, "rows": slim[i : i + chunk]} for i in range(0, len(slim), chunk)]
        out = [row for part in runner.generate.map(jobs) for row in part]
    else:
        out = runner.generate.remote({**job, "rows": slim})
    save(name, out, params, job, time.time() - t0)
    return out


def save(name: str, out: list[dict], params: dict, job: dict, seconds: float) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    with (GEN / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps({"run": name, **row}, ensure_ascii=False) + "\n")
    log_path = HERE / "results" / "runs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    config = {k: v for k, v in job.items() if k not in ("rows", "steer")}
    if job.get("steer"):
        config["steer"] = {k: v for k, v in job["steer"].items() if k != "vector"}
    finished = sum(r["finished"] for r in out)
    stamp = datetime.now().isoformat(timespec="seconds")
    log[name] = {"params": params, "config": config, "replies": len(out), "seconds": round(seconds),
                 "finished": finished, "at": stamp}  # fmt: skip
    log_path.write_text(json.dumps(log, indent=1) + "\n")
    print(f"{name}: {len(out)} replies, {finished} finished, {seconds:.0f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list or not args.runs:
        print("\n".join(f"{k:16} {v}" for k, v in RUNS.items()))
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(args.runs)) as pool:
        futures = [pool.submit(run, name, *job_for(name)) for name in args.runs]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
