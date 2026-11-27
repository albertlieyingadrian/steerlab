"""Driver for the adapter self-interpretation replication. One H100 at a time; every launch is checked against the cap,
its Modal call id saved before waiting, and its cost settled from wall time.

    .venv/bin/python experiments/repl-adapter-self-interp/run.py prep          # decoder vectors + SAE known-answer check
    .venv/bin/python experiments/repl-adapter-self-interp/run.py train         # our SA adapter, then the shuffled one
    .venv/bin/python experiments/repl-adapter-self-interp/run.py valloss       # Table-1-style val losses
    .venv/bin/python experiments/repl-adapter-self-interp/run.py eval smoke    # 20 latents (seed 1): pipeline check only
    .venv/bin/python experiments/repl-adapter-self-interp/run.py eval main     # the 500 pre-registered latents (seed 0)
"""

import datetime as dt
import gzip
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
ROOT = HERE.parents[1]
CAP_USD = 25.0
H100_PER_H = 3.95
os.environ.setdefault("MODAL_PROFILE", "credits30")
APP = "repl-adapter-self-interp"
ADAPTER_ARMS = ["released", "ours", "shuffled"]
FLOOR_CONV = "[USER] Hi, how are you?\n[ASSISTANT] I'm doing well, thank you. How can I help you today?"


def ledger() -> list:
    p = RESULTS / "spend.json"
    return json.loads(p.read_text()) if p.exists() else []


def save_ledger(rows):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "spend.json").write_text(json.dumps(rows, indent=1))


def call(fn_name: str, stage: str, est_hours: float, *args):
    import modal

    rows = ledger()
    total = sum(r.get("usd", r["estimate_usd"]) for r in rows)
    est = est_hours * H100_PER_H
    print(f"{stage}: estimate ${est:.2f}; ledger ${total:.2f} of ${CAP_USD:.0f}", flush=True)
    if total + est > CAP_USD:
        raise SystemExit("refused: would pass the cap")
    fn = modal.Function.from_name(APP, fn_name)
    t0 = time.time()
    fc = fn.spawn(*args)
    rows.append(
        {
            "stage": stage,
            "estimate_usd": round(est, 2),
            "call_id": fc.object_id,
            "launched": dt.datetime.now().isoformat(timespec="seconds"),
            "profile": os.environ["MODAL_PROFILE"],
        }
    )
    save_ledger(rows)
    print("  call id", fc.object_id, flush=True)
    out = fc.get()
    secs = time.time() - t0
    rows = ledger()
    rows[-1].update({"seconds": round(secs), "usd": round((secs + 60) / 3600 * H100_PER_H, 2)})
    save_ledger(rows)
    print(f"  done in {secs / 60:.1f} min: {str(out)[:600]}", flush=True)
    return out


def labels_file() -> dict:
    with gzip.open(HERE / "data" / "goodfire_8b_sae_labels.json.gz") as fh:
        d = json.load(fh)[0]["vectors"]
    return {int(v["index"]): {"label": v["labels"][0].strip(), "split": v["split"]} for v in d}


def _other(rng, pool, i):
    j = rng.choice(pool)
    while j == i:
        j = rng.choice(pool)
    return j


def pick_latents(tag: str) -> dict:
    """main: 500 val latents (seed 0); smoke: 20 other val latents (seed 1). Decoy and wrong-label latents: random
    other val latents (fixed seeds)."""
    lab = labels_file()
    val = sorted(i for i, v in lab.items() if v["split"] == "val")
    main = sorted(random.Random(0).sample(val, 500))
    rest = sorted(set(val) - set(main))
    chosen = main if tag == "main" else sorted(random.Random(1).sample(rest, 20))
    r_decoy, r_wrong = random.Random(7 if tag == "main" else 8), random.Random(11 if tag == "main" else 12)
    return {
        "eval": chosen,
        "decoy": {str(i): _other(r_decoy, val, i) for i in chosen},
        "wrong": {str(i): _other(r_wrong, val, i) for i in chosen},
    }


def build_jobs(lat: dict, labels: dict, para: dict) -> tuple[list, dict, list]:
    lab = labels_file()
    jobs, sets = [], {}
    for i in lat["eval"]:
        t = [i, lat["decoy"][str(i)]]
        for arm in ADAPTER_ARMS + ["untrained"]:
            for c, text in enumerate(labels[arm][str(i)]):
                jobs.append({"key": f"{arm}|{i}|{c}", "label": text})
        auto = lab[i]["label"]
        ps = para[str(i)]["lines"]
        cands = {
            "auto6": [auto] * 6,
            "para": [auto] + (ps + [auto] * 5)[:5],
            "wrong6": [lab[lat["wrong"][str(i)]]["label"]] * 6,
        }
        for arm, ls in cands.items():
            for c, text in enumerate(ls):
                jobs.append({"key": f"{arm}|{i}|{c}", "label": text})
        for arm in ADAPTER_ARMS + ["untrained", "auto6", "para", "wrong6"]:
            for c in range(6):
                sets[f"{arm}|{i}|{c}"] = t
    allt = sorted(set(lat["eval"]) | set(lat["decoy"].values()))
    for arm in ("ours", "released"):
        for c, text in enumerate(labels[arm + "_bias"]):
            jobs.append({"key": f"bias_{arm}||{c}", "label": text})
            sets[f"bias_{arm}||{c}"] = allt
    fixed = [{"key": "floor", "label": None, "texts": [FLOOR_CONV]}]
    sets["floor"] = allt
    return jobs, sets, fixed


def pull_one(src: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(Path(sys.executable).parent / "modal"), "volume", "get", "--force", APP, src, str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print("pull", src, "ok" if r.returncode == 0 else r.stderr.strip()[:300], flush=True)


def eval_stage(tag: str):
    lat = pick_latents(tag)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"latents_{tag}.json").write_text(json.dumps(lat))
    k = len(lat["eval"])
    if not (RESULTS / f"labels_{tag}.json").exists():
        call("labels", f"labels_{tag}", 0.15 + k * 0.0006, lat["eval"], ADAPTER_ARMS, tag)
        pull_one(f"/labels/{tag}.json", RESULTS / f"labels_{tag}.json")
    if not (RESULTS / f"para_{tag}.json").exists():
        lab = labels_file()
        call("paraphrase", f"paraphrase_{tag}", 0.12, {str(i): lab[i]["label"] for i in lat["eval"]}, f"para_{tag}")
        pull_one(f"/labels/para_{tag}.json", RESULTS / f"para_{tag}.json")
    labels = json.loads((RESULTS / f"labels_{tag}.json").read_text())
    para = json.loads((RESULTS / f"para_{tag}.json").read_text())
    jobs, sets, fixed = build_jobs(lat, labels, para)
    print(f"{len(jobs)} jobs x 20 conversations", flush=True)
    call("convs", f"convs_{tag}", 0.15 + len(jobs) * 20 / 300_000, jobs, tag, 20, fixed)
    call("score", f"score_{tag}", 0.15 + len(jobs) * 20 / 400_000, tag, sets)
    pull_one(f"/scores/{tag}.json", RESULTS / f"scores_{tag}.json")
    pull_one(f"/convs/{tag}.jsonl", RESULTS / "cache" / f"convs_{tag}.jsonl")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "prep":
        call("prep", "prep", 0.2)
        pull_one("/data/sae_check.json", RESULTS / "sae_check.json")
    elif a[0] == "train":
        for name, shuf in (("ours", False), ("shuffled", True)):
            if not (RESULTS / f"train_{name}.json").exists():
                call("train", f"train_{name}", 0.8, name, shuf)
                pull_one(f"/adapters/{name}.json", RESULTS / f"train_{name}.json")
    elif a[0] == "valloss":
        call("valloss", "valloss", 0.3, ["released", "ours", "shuffled"], [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        pull_one("/valloss.json", RESULTS / "valloss.json")
    elif a[0] == "eval":
        eval_stage(a[1])
    else:
        raise SystemExit(__doc__)
