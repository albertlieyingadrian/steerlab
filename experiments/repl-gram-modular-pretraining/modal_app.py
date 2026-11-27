"""Modal jobs for the GRAM replication. COSTS MONEY (H100, about $4/hour).

    export MODAL_PROFILE=albertlieyingadrian
    modal run modal_app.py::prep                          # CPU: tokenise Simple Stories into the volume (once)
    modal run modal_app.py::smoke                         # GPU: every stage for a few steps, outputs under /vol/smoke
    modal run --detach modal_app.py::launch --seeds 1,2   # one H100 job per seed, at most two at a time
    modal run modal_app.py::pull                          # copy every result JSON (not checkpoints) into results/runs/

One job runs a whole seed: baseline (with its loss curve) -> GRAM -> five filtered models -> MaxEnt from the
baseline, each followed by its evaluations and relearning attacks. Seed 1 also runs the MaxEnt grid right after its
baseline; other seeds wait for the grid's choice before their MaxEnt stage. Finished stages are skipped on a rerun.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = Path("/vol")
APP = "repl-gram-modular-pretraining"
TIMEOUT_H = 6

volume = modal.Volume.from_name(APP, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy<2.1",
        "datasets==3.2.0",
        "transformers==4.47.1",
        "tokenizers",
    )
    .add_local_file(HERE / "gram.py", "/root/gram.py")
)
app = modal.App(APP, image=image)


@app.function(cpu=16, memory=32768, timeout=3600, volumes={VOL: volume})
def prep_data() -> dict:
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    import gram

    tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")
    assert tok.eos_token_id == 1 and len(tok) == 4096, (tok.eos_token_id, len(tok))
    ds = load_dataset("SimpleStories/SimpleStories", split="train").select_columns(["topic", "story"])
    sp = ds.train_test_split(test_size=0.1, seed=42)  # as in prep_stories.py
    out = VOL / "data"
    out.mkdir(parents=True, exist_ok=True)
    meta = {}

    def enc(batch):
        ids = tok(batch["story"], add_special_tokens=False)["input_ids"]
        return {
            "ids": [[*x, tok.eos_token_id] for x in ids],
            "topic": [str(t).lower().replace(" ", "-") for t in batch["topic"]],
        }

    for split in ("train", "test"):
        d = sp[split].map(enc, batched=True, num_proc=16, remove_columns=["story"])
        labels = sorted(set(d["topic"]))
        assert labels[:4] == gram.AUX_LABELS and len(labels) == 48, labels[:6]
        streams = {lab: [] for lab in gram.LABELS}
        for b in d.iter(batch_size=20000):  # batches keep the Python-int lists small
            for t, x in zip(b["topic"], b["ids"], strict=True):
                streams[t if t in gram.AUX_LABELS else "core"].append(np.asarray(x, dtype=np.uint16))
        for lab, parts in streams.items():
            arr = np.concatenate(parts)
            arr.tofile(out / f"{lab}_{split}.bin")
            meta[f"{lab}_{split}"] = {"stories": len(parts), "tokens": len(arr)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    volume.commit()
    return meta


def _run(seed: int, root: Path, smoke: bool) -> None:
    import copy

    import torch

    import gram

    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64
    c = gram.Cfg()
    rounds, rounds8 = 200, 100
    if smoke:
        c.max_steps, c.curve_evals, c.curve_batches, c.val_batches, c.test_batches = (
            30,
            3,
            2,
            2,
            2,
        )
        rounds, rounds8 = 3, 2
    out = root / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    logs = {"f": open(out / "log.txt", "a")}  # a dict so the wait loop can close and reopen it

    def log(msg: str) -> None:
        line = f"[seed {seed} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logs["f"].write(line + "\n")
        logs["f"].flush()

    def done(name: str) -> bool:
        return (out / name).exists()

    torch.manual_seed(seed)
    data = gram.Data(VOL / "data", c, "cuda")
    log("data " + json.dumps({lab: data.n_batches(lab) for lab in gram.LABELS}))

    def elicit_set(model, aux_on, forget_idx, tag, num_seq, rnds):
        res = []
        for i in forget_idx:
            m = copy.deepcopy(model)
            r = gram.elicit(m, data, gram.AUX_LABELS[i], aux_on, seed, num_seq=num_seq, rounds=rnds)
            r["profile"] = gram.profile_name(aux_on)
            res.append(r)
            log(f"{tag} elicit {r['label']} n={num_seq} round {r['best_round']} test {r['test_loss']:.4f}")
            del m
        return res

    def forget_of(p):
        return [i for i in range(4) if i not in p]

    # ---- 1. baseline (+ loss curve for the compute ratio) ----
    if not done("baseline.json"):
        torch.manual_seed(seed)
        m = gram.GPT(c).cuda()
        r = gram.pretrain(m, data, gram.baseline_specs(data), c, seed, log, curve=True)
        r["params"] = gram.n_params(m.parameters())
        r["test"], r["val"] = (
            gram.eval_all(m, data, ()),
            gram.eval_all(m, data, (), "val"),
        )
        torch.save(m.state_dict(), out / "baseline.pt")
        gram.save_json(r, out / "baseline.json")
        volume.commit()
        log(f"baseline done {r['test']}")
    base = json.loads((out / "baseline.json").read_text())

    # ---- 1b. MaxEnt grid on seed 1 (core-only profile, val half) ----
    grid_path = root / "maxent_choice.json"
    if seed == 1 and not grid_path.exists():
        bm = gram.GPT(c).cuda()
        bm.load_state_dict(torch.load(out / "baseline.pt"))
        rows = []
        lrs, alphas = ((1e-3,), (30.0,)) if smoke else ((3e-4, 1e-3, 3e-3), (10.0, 30.0, 100.0))
        for lr in lrs:
            for a in alphas:
                m = copy.deepcopy(bm)
                info = gram.maxent(m, data, (), lr, a, seed, steps=20 if smoke else 2000)
                v = gram.eval_all(m, data, (), "val")
                rows.append(
                    {
                        "lr": lr,
                        "alpha": a,
                        **info,
                        "val": v,
                        "core_rise": v["core"] - base["val"]["core"],
                        "forget_mean": sum(v[x] for x in gram.AUX_LABELS) / 4,
                    }
                )
                log(
                    f"maxent grid lr {lr} a {a}: core +{rows[-1]['core_rise']:.4f} forget {rows[-1]['forget_mean']:.4f}"
                )
                del m
        ok = [r for r in rows if r["core_rise"] <= 0.03] or [min(rows, key=lambda r: r["core_rise"])]
        best = max(ok, key=lambda r: r["forget_mean"])
        gram.save_json(
            {"rows": rows, "chosen": {"lr": best["lr"], "alpha": best["alpha"]}},
            grid_path,
        )
        volume.commit()
        log(f"maxent choice {best['lr']} {best['alpha']}")

    # ---- 2. GRAM ----
    if not done("gram_eval.json"):
        torch.manual_seed(seed)
        m = gram.GPT(c, n_aux=4).cuda()
        P_core, P_aux = gram.n_params(m.core_params()), gram.n_params(m.aux_params(0))
        specs, info = gram.gram_specs(data, c, base["params"], P_core, P_aux)
        log(f"gram specs {info} P_core {P_core} P_aux {P_aux}")
        r = gram.pretrain(m, data, specs, c, seed, log)
        r.update(info=info, P_core=P_core, P_aux=P_aux)
        r["profiles"] = {
            gram.profile_name(p): {
                "test": gram.eval_all(m, data, p),
                "val": gram.eval_all(m, data, p, "val"),
            }
            for p in gram.PROFILES
        }
        r["all_on"] = gram.eval_all(m, data, (0, 1, 2, 3))
        torch.save(m.state_dict(), out / "gram.pt")
        gram.save_json(r, out / "gram_eval.json")
        volume.commit()
        log("gram trained")
    if not done("gram_elicit.json"):
        m = gram.GPT(c, n_aux=4).cuda()
        m.load_state_dict(torch.load(out / "gram.pt"))
        res = []
        for p in gram.PROFILES:
            res += elicit_set(m, p, forget_of(p), "gram", 512, rounds)
        res += elicit_set(m, (), forget_of(()), "gram", 4096, rounds8)
        gram.save_json(res, out / "gram_elicit.json")
        volume.commit()

    # ---- 3. data filtering: one model per profile ----
    for p in gram.PROFILES:
        name = gram.profile_name(p)
        f_eval, f_el = f"filter_{name}.json", f"filter_{name}_elicit.json"
        if done(f_el):
            continue
        torch.manual_seed(seed)
        m = gram.GPT(c).cuda()
        r = gram.pretrain(m, data, gram.filtering_specs(data, p), c, seed, log)
        r["test"], r["val"] = (
            gram.eval_all(m, data, ()),
            gram.eval_all(m, data, (), "val"),
        )
        gram.save_json(r, out / f_eval)
        res = elicit_set(m, (), forget_of(p), f"filter {name}", 512, rounds)
        if not p:
            res += elicit_set(m, (), forget_of(p), f"filter {name}", 4096, rounds8)
        gram.save_json(res, out / f_el)
        volume.commit()
        log(f"filtering {name} done {r['test']}")
        del m

    # ---- 4. MaxEnt post-hoc unlearning from this seed's baseline ----
    while not grid_path.exists():
        log("waiting for the MaxEnt grid choice")
        time.sleep(60)
        logs["f"].close()  # Volume.reload() refuses while any file on the volume is open
        volume.reload()
        logs["f"] = open(out / "log.txt", "a")
    ch = json.loads(grid_path.read_text())["chosen"]
    bm = gram.GPT(c).cuda()
    bm.load_state_dict(torch.load(out / "baseline.pt"))
    for p in gram.PROFILES:
        name = gram.profile_name(p)
        if done(f"maxent_{name}.json"):
            continue
        m = copy.deepcopy(bm)
        info = gram.maxent(m, data, p, ch["lr"], ch["alpha"], seed, steps=20 if smoke else 2000)
        r = {
            "chosen": ch,
            **info,
            "test": gram.eval_all(m, data, ()),
            "val": gram.eval_all(m, data, (), "val"),
        }
        res = elicit_set(m, (), forget_of(p), f"maxent {name}", 512, rounds)
        if not p:
            res += elicit_set(m, (), forget_of(p), f"maxent {name}", 4096, rounds8)
        r["elicit"] = res
        gram.save_json(r, out / f"maxent_{name}.json")
        volume.commit()
        log(f"maxent {name} done {r['test']}")
        del m
    log("seed complete")


@app.function(gpu="H100", timeout=TIMEOUT_H * 3600, volumes={VOL: volume}, retries=0)
def run_seed(seed: int, smoke: bool = False) -> str:
    _run(seed, VOL / ("smoke" if smoke else "runs"), smoke)
    return f"seed {seed} done"


@app.function(timeout=600, volumes={VOL: volume})
def collect(root: str = "runs") -> dict[str, str]:
    volume.reload()
    base = VOL / root
    return {str(p.relative_to(base)): p.read_text() for p in base.rglob("*") if p.suffix in (".json", ".txt")}


@app.local_entrypoint()
def prep():
    print(json.dumps(prep_data.remote(), indent=1))


@app.local_entrypoint()
def smoke():
    print(run_seed.remote(1, smoke=True))


@app.local_entrypoint()
def launch(seeds: str = "1,2"):
    """Spawn one job per seed (at most two). Call ids are appended to results/calls.jsonl."""
    ss = [int(s) for s in seeds.split(",")]
    assert len(ss) <= 2, "at most two GPUs at a time"
    calls = HERE / "results" / "calls.jsonl"
    for s in ss:
        fc = run_seed.spawn(s)
        with calls.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "seed": s,
                        "call_id": fc.object_id,
                        "t": time.strftime("%Y-%m-%d %H:%M"),
                    }
                )
                + "\n"
            )
        print(s, fc.object_id)


@app.local_entrypoint()
def pull(root: str = "runs"):
    """Write every JSON and log file from /vol/<root> into results/<root>/."""
    files = collect.remote(root)
    dest = HERE / "results" / root
    for rel, text in files.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    print(f"pulled {len(files)} files into {dest}")
