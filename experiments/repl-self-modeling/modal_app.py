"""Modal workers for the self-modeling replication. MNIST and IMDB run on CPU containers (no GPU).

    MODAL_PROFILE=credits30 .venv/bin/modal deploy experiments/repl-self-modeling/modal_app.py

    run_one(stage, cfg)        train one arm (selfmodel.train_run) and write /vol/<stage>/<run_id>.json
    orchestrate(stage, cfgs)   map run_one over cfgs inside Modal, so a dropped laptop client cannot kill the batch

Datasets are downloaded once at image build and cached as .pt files inside the image.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("repl-self-modeling")
volume = modal.Volume.from_name("repl-self-modeling", create_if_missing=True)
HERE = Path(__file__).parent


def _prepare_data():
    import torch

    import selfmodel

    torch.save(selfmodel.load_mnist("/root/hf"), "/root/mnist.pt")
    torch.save(selfmodel.load_imdb("/root/hf"), "/root/imdb.pt")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1", "numpy", "pillow", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("datasets==4.4.1")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/root/hf"})
    .add_local_file(HERE / "selfmodel.py", "/root/selfmodel.py", copy=True)
    .run_function(_prepare_data)
)

_DATA = {}


def _data(task: str):
    import torch

    if task not in _DATA:
        _DATA[task] = torch.load(f"/root/{task}.pt")
    return _DATA[task]


def _downsample(res: dict, every: int = 10) -> dict:
    for ck in res.get("checkpoints", {}).values():
        for k in list(ck):
            if k.endswith("_trace"):
                ck[k] = ck[k][::every]
    return res


@app.function(image=image, cpu=2.0, memory=4096, timeout=3 * 3600, volumes={"/vol": volume}, retries=1)
def run_one(stage: str, cfg: dict) -> dict:
    import torch

    import selfmodel

    torch.set_num_threads(2)
    path = Path(f"/vol/{stage}/{cfg['run_id']}.json")
    if path.exists():
        return {"run_id": cfg["run_id"], "cached": True}
    t0 = time.time()
    res = _downsample(selfmodel.train_run(cfg, _data(cfg["task"]), "cpu"))
    res["wall_seconds"] = time.time() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    volume.commit()
    return {"run_id": cfg["run_id"], "seconds": res["wall_seconds"]}


@app.function(image=image, cpu=1.0, memory=2048, timeout=12 * 3600, volumes={"/vol": volume})
def orchestrate(stage: str, cfgs: list[dict]) -> list:
    out = []
    for r in run_one.map([stage] * len(cfgs), cfgs, return_exceptions=True):
        out.append(r if isinstance(r, dict) else {"error": repr(r)})
    Path(f"/vol/{stage}").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/{stage}/_done.json").write_text(json.dumps(out))
    volume.commit()
    return out
