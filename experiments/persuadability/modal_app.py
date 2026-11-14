"""Modal jobs (COSTS MONEY: profile albertlieyingadrian, L40S GPUs, cap $20 for the study).

    MODAL_PROFILE=albertlieyingadrian modal run modal_app.py --stage train      # 4 arms x 2 seeds, in parallel
    MODAL_PROFILE=albertlieyingadrian modal run modal_app.py --stage sweep      # direction + dev sweep + selection
    MODAL_PROFILE=albertlieyingadrian modal run modal_app.py --stage eval       # every test configuration
    MODAL_PROFILE=albertlieyingadrian modal run modal_app.py --stage explore    # exploratory: random at alpha/2

Adapters and outputs live on the `persuadability` volume, so a rerun skips finished work. Each stage copies its
outputs into results/ here.
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "/vol"
volume = modal.Volume.from_name("persuadability", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0", "transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1", "numpy<2.4", "hf_transfer==0.1.9"
    )
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(HERE, "/root/exp", ignore=["results/**", "report/**", "__pycache__/**", "*.pyc"])
)
app = modal.App("persuadability", image=image)
GPU = {"gpu": "L40S", "cpu": 4.0, "memory": 32768, "volumes": {VOL: volume}, "scaledown_window": 10}
ARMS = ("easy", "hard", "resist", "twin")
SEEDS = (0, 1)


def _setup():
    import os
    import sys

    os.chdir("/root/exp")
    sys.path.insert(0, "/root/exp")


@app.function(timeout=60 * 60, **GPU)
def train(arm: str, seed: int) -> dict:
    _setup()
    import pipeline as P

    out = Path(VOL) / "adapters" / f"{arm}_s{seed}"
    if (out / "train_info.json").exists():
        return json.loads((out / "train_info.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    info = P.train_lora(arm, seed, f"data/sft_{arm}.jsonl", str(out), log=lambda m: print(m, flush=True))
    volume.commit()
    return info


@app.function(timeout=2 * 60 * 60, **GPU)
def sweep() -> dict:
    _setup()
    import numpy as np

    import pipeline as P

    res = Path(VOL) / "results"
    res.mkdir(parents=True, exist_ok=True)
    if not (res / "directions.npz").exists():
        P.build_direction(str(res / "directions.npz"), log=lambda m: print(m, flush=True))
        volume.commit()
    dirs = dict(np.load(res / "directions.npz"))
    sel = P.dev_sweep(dirs, str(res / "dev_sweep.json"), log=lambda m: print(m, flush=True))
    (res / "selection.json").write_text(json.dumps(sel, indent=1))
    volume.commit()
    return {f: (res / f).read_text() for f in ("directions.json", "dev_sweep.json", "selection.json")} | {
        "directions.npz.hex": (res / "directions.npz").read_bytes().hex()
    }


@app.function(timeout=2 * 60 * 60, **GPU)
def evaluate(cfgs: list) -> dict:
    """Every config in cfgs on the test jobs; adapters are loaded one at a time on top of one base model."""
    _setup()
    import numpy as np

    import common as C
    import pipeline as P

    res = Path(VOL) / "results" / "test"
    res.mkdir(parents=True, exist_ok=True)
    dirs = dict(np.load(Path(VOL) / "results" / "directions.npz"))
    jobs = P.eval_jobs("test")
    mmlu = C.read_jsonl("data/mmlu300.jsonl")
    tok, base = C.load()
    out = {}
    for cfg in cfgs:
        f = res / f"{cfg['name']}.jsonl"
        s = res / f"{cfg['name']}.summary.json"
        if not f.exists():
            model = base
            if cfg.get("adapter"):
                from peft import PeftModel

                model = PeftModel.from_pretrained(base, str(Path(VOL) / "adapters" / cfg["adapter"])).eval()
            rows, summ = P.run_config(cfg, tok, model, dirs, jobs, mmlu=mmlu)
            if cfg.get("adapter"):
                base = model.unload()
            C.write_jsonl(f, rows)
            s.write_text(json.dumps(summ))
            volume.commit()
            print(
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k != "config"},
                cfg["name"],
                flush=True,
            )
        out[cfg["name"]] = (f.read_text(), s.read_text())
    return out


def test_configs():
    sel = json.loads((HERE / "results" / "selection.json").read_text())
    L, a = sel["layer"], sel["alpha"]
    groups = [[{"name": "base"}]]
    for arm in ARMS:
        groups.append([{"name": f"lora_{arm}_s{s}", "adapter": f"{arm}_s{s}"} for s in SEEDS])
    steer = [
        {"name": "steer_easy", "layer": L, "source": "dir", "alpha": a},
        {"name": "steer_hard", "layer": L, "source": "dir", "alpha": -a},
    ]
    steer += [
        {"name": f"dose_{f:g}x_{tag}", "layer": L, "source": "dir", "alpha": sg * f * a}
        for f in (0.5, 2.0)
        for tag, sg in (("easy", 1), ("hard", -1))
    ]
    rand = [{"name": f"rand{k}", "layer": L, "source": f"rand{k}", "alpha": a} for k in range(10)]
    groups[0] += steer
    groups += [rand[:5], rand[5:]]
    return groups


def explore_configs():
    """Exploratory (after reading the test results): the 10 random directions at alpha/2, the strength of the
    dose_0.5x organisms, which had no random control in the pre-registered set."""
    sel = json.loads((HERE / "results" / "selection.json").read_text())
    L, a = sel["layer"], sel["alpha"]
    rand = [{"name": f"randhalf{k}", "layer": L, "source": f"rand{k}", "alpha": a / 2} for k in range(10)]
    return [rand[:5], rand[5:]]


@app.local_entrypoint()
def main(stage: str = "train"):
    out = HERE / "results"
    if stage == "train":
        infos = list(train.starmap([(a, s) for a in ARMS for s in SEEDS]))
        (out / "train_info.json").write_text(json.dumps(infos, indent=1))
        print(json.dumps(infos, indent=1))
    elif stage == "sweep":
        r = sweep.remote()
        (out / "directions.npz").write_bytes(bytes.fromhex(r.pop("directions.npz.hex")))
        for k, v in r.items():
            (out / k).write_text(v)
        print(r["selection.json"])
    elif stage in ("eval", "explore"):
        (out / "test").mkdir(parents=True, exist_ok=True)
        for res in evaluate.map(test_configs() if stage == "eval" else explore_configs()):
            for name, (rows, summ) in res.items():
                (out / "test" / f"{name}.jsonl").write_text(rows)
                (out / "test" / f"{name}.summary.json").write_text(summ)
        print("wrote", out / "test")
