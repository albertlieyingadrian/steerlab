"""Drive the bake-off on Modal, step by step. Every paid step checks the spend cap and logs to results/spend.jsonl.

MODAL_PROFILE=albertlieyingadrian python run.py upload           # data v2 (4 buckets) + eval texts to the volume
MODAL_PROFILE=albertlieyingadrian python run.py sweep            # 3 learning rates per arm, 16k rows, 1 epoch
python run.py choose                                             # validation only -> results/lr_choice.json
MODAL_PROFILE=albertlieyingadrian python run.py train            # each arm, chosen lr, full data, 2 epochs
MODAL_PROFILE=albertlieyingadrian python run.py score            # each arm on every split (incl. the fresh look)
MODAL_PROFILE=albertlieyingadrian python run.py refs --ref qwen3.5-9b   # reference arms on the splits they lack
MODAL_PROFILE=modal4 python run.py refs --ref qwen3.5-4b
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from bb import (
    ARMS,
    ARMS2,
    DATA,
    EPOCHS,
    LR_GRID,
    REFS,
    RESULTS,
    RUNS,
    SD,
    SPLITS,
    SWEEP_ROWS,
    check_cap,
    eval_texts,
    log_spend,
    sd,
)

DATA_NAME = "v2b"


def _fn(name: str, app: str = "slop-backbones"):
    import modal

    return modal.Function.from_name(app, name)


def upload(_a) -> None:
    import modal

    vol = modal.Volume.from_name("slop-detector")
    ev = eval_texts()
    tmp = RESULTS / "_evals"
    tmp.mkdir(parents=True, exist_ok=True)
    with vol.batch_upload(force=True) as up:
        for f in ("train.jsonl", "val.jsonl"):
            up.put_file(str(DATA / f), f"/bb/data/{DATA_NAME}/{f}")
        for s, texts in ev.items():
            (tmp / f"{s}.json").write_text(json.dumps(texts))
            up.put_file(str(tmp / f"{s}.json"), f"/bb/evals/{s}.json")
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    (RESULTS / "eval_counts.json").write_text(json.dumps({s: len(t) for s, t in ev.items()}, indent=1) + "\n")
    print({s: len(t) for s, t in ev.items()})


def _save_run(name: str, out: dict) -> None:
    run = RUNS / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out["log"]))
    (run / "val_scores.json").write_text(json.dumps(out["val_scores"]) + "\n")
    (run / "config.json").write_text(json.dumps({**out["config"], "wall": out["wall"]}, indent=1) + "\n")


def _train_many(jobs: list[tuple[str, dict, float, float, int]], est_usd: float, what: str) -> None:
    check_cap(est_usd, what)
    import modal

    calls = {}
    for name, arm, lr, epochs, max_rows in jobs:
        if (RUNS / name / "config.json").exists():
            print(f"{name}: done already")
            continue
        fn = _fn("train_h200" if arm.get("gpu") == "H200" else "train")
        calls[name] = fn.spawn(
            arm["base"], arm["kind"], DATA_NAME, name, lr, epochs, arm["micro"], arm["grad_ckpt"], max_rows
        )
        print(f"spawned {name}")
    for name, call in calls.items():
        try:
            out = call.get()
        except modal.exception.Error as e:  # a remote failure (e.g. out of memory): log it, keep the others going
            print(f"{name} FAILED: {type(e).__name__}: {e}")
            continue
        _save_run(name, out)
        usd = log_spend(f"train {name}", out["gpu"], out["wall"])
        print(f"{name}: {out['wall']} s, ~${usd}")


def _arms(a) -> dict:
    pool = ARMS2 if getattr(a, "round", 1) == 2 else ARMS
    return {k: v for k, v in pool.items() if not a.arms or k in a.arms}


def sweep(a) -> None:
    jobs = [(f"sweep-{k}-{lr:g}", v, lr, 1.0, SWEEP_ROWS) for k, v in _arms(a).items() for lr in LR_GRID[v["kind"]]]
    _train_many(jobs, est_usd=25, what="sweep")


def _val_rows():
    return sd.read_jsonl(DATA / "val.jsonl")


def val_mf1(scores: np.ndarray, rows: list[dict]) -> float:
    """Validation macro-F1 with the two cut-offs fitted on the same rows (the ranking only, not a claim)."""
    from importlib import import_module

    diag = import_module("diag_voi_gap")
    y = np.array([r["band"] for r in rows[: len(scores)]])
    return float(diag.fit_cuts(y, scores)[0])


def choose(a_) -> None:
    rows = _val_rows()
    path = RESULTS / ("lr_choice.json" if a_.round == 1 else "lr_choice_round2.json")
    out = {}
    for k, a in _arms(a_).items():
        res = {}
        for lr in LR_GRID[a["kind"]]:
            p = RUNS / f"sweep-{k}-{lr:g}" / "val_scores.json"
            if p.exists():
                res[f"{lr:g}"] = round(val_mf1(np.array(json.loads(p.read_text())), rows), 4)
        if not res:
            continue
        default = "2e-05" if a["kind"] == "full" else "0.0002"
        top = max(res.values())
        near = [lr for lr, v in res.items() if v >= top - 0.002]  # ties go to the grid's middle value
        pick = default if default in near else max(near, key=lambda lr: res[lr])
        out[k] = {"val_macro_f1": res, "chosen_lr": float(pick)}
        print(k, out[k])
    out["rule"] = (
        "highest val macro-F1 (cuts fitted on the same 2,000 val rows); within 0.002 of the top -> the middle lr"
    )
    path.write_text(json.dumps(out, indent=1) + "\n")


def train(a_) -> None:
    ch = json.loads((RESULTS / ("lr_choice.json" if a_.round == 1 else "lr_choice_round2.json")).read_text())
    jobs = [(f"bb-{k}", a, ch[k]["chosen_lr"], EPOCHS, 0) for k, a in _arms(a_).items() if k in ch]
    _train_many(jobs, est_usd=60, what="train")


def score(a) -> None:
    check_cap(12, "score")
    calls = {
        k: _fn("score_h200" if v.get("gpu") == "H200" else "score").spawn(f"bb-{k}", list(SPLITS))
        for k, v in _arms(a).items()
        if (RUNS / f"bb-{k}" / "config.json").exists()
    }
    for k, c in calls.items():
        t0 = time.time()
        out = c.get()
        d = RUNS / f"bb-{k}" / "scores"
        d.mkdir(parents=True, exist_ok=True)
        for s, v in out["scores"].items():
            (d / f"{s}.json").write_text(json.dumps(v) + "\n")
        timing = {"seconds": out["seconds"], "tokens": out["tokens"], "load_seconds": out["load_seconds"]}
        (RUNS / f"bb-{k}" / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
        log_spend(
            f"score bb-{k}",
            out["gpu"],
            sum(out["seconds"].values()) + out["load_seconds"],
            wall=round(time.time() - t0),
        )
        print(k, "scored", list(out["scores"]))


def refs(a) -> None:
    """Score a reference run (slop-detector app, its own workspace) on the splits it has no saved scores for."""
    run, _profile = REFS[a.ref]
    have = SD / "runs" / run / "scores"
    need = [s for s in SPLITS if not (have / f"{s}.json").exists()]
    if not need:
        print("nothing to score")
        return
    check_cap(4, f"refs {a.ref}")
    ev = eval_texts(need)
    t0 = time.time()
    out = _fn("score_run", "slop-detector").remote(run, ev)
    wall = time.time() - t0
    d = RUNS / f"ref-{a.ref}" / "scores"
    d.mkdir(parents=True, exist_ok=True)
    for s, v in out.items():
        (d / f"{s}.json").write_text(json.dumps(v) + "\n")
    n = sum(len(t) for t in ev.values())
    (RUNS / f"ref-{a.ref}" / "timing.json").write_text(
        json.dumps({"wall_seconds_incl_load": round(wall), "texts": n, "splits": need}, indent=1) + "\n"
    )
    log_spend(f"refs {a.ref}", "H100", wall)
    print(a.ref, "scored", need, f"{wall:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["upload", "sweep", "choose", "train", "score", "refs"])
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--ref", choices=list(REFS))
    ap.add_argument("--round", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()
    {"upload": upload, "sweep": sweep, "choose": choose, "train": train, "score": score, "refs": refs}[a.step](a)


if __name__ == "__main__":
    main()
