"""Modal worker (one L4) for the scaled-down CIFAR-10 arm. A separate app so the CPU app is never rebuilt mid-batch.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 MODAL_PROFILE=credits30 .venv/bin/modal deploy \
        experiments/repl-self-modeling/modal_cifar.py

    run_all(cfgs)   runs the CIFAR configs one after another on one L4; each writes /vol/cifar/<run_id>.json
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("repl-self-modeling-cifar")
volume = modal.Volume.from_name("repl-self-modeling", create_if_missing=True)
HERE = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", "datasets==4.4.1", "numpy", "pillow")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/vol/hf"})
    .add_local_file(HERE / "selfmodel.py", "/root/selfmodel.py", copy=True)
    .add_local_file(HERE / "cifar.py", "/root/cifar.py", copy=True)
)


def _downsample(res, every=10):
    for ck in res["checkpoints"].values():
        for k in list(ck):
            if k.endswith("_trace"):
                ck[k] = ck[k][::every]
    return res


@app.function(image=image, gpu="L4", cpu=8.0, memory=32768, timeout=10 * 3600, volumes={"/vol": volume})
def run_all(cfgs: list[dict]) -> list:
    import cifar

    out = []
    for cfg in cfgs:
        path = Path(f"/vol/cifar/{cfg['run_id']}.json")
        if path.exists():
            out.append({"run_id": cfg["run_id"], "cached": True})
            continue
        t0 = time.time()
        import torch

        res = cifar.train_run(cfg, "/vol/hf")
        Path("/vol/cifar_w").mkdir(parents=True, exist_ok=True)
        torch.save(res.pop("state_dict"), f"/vol/cifar_w/{cfg['run_id']}.pt")
        res = _downsample(res)
        res["wall_seconds"] = time.time() - t0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res))
        volume.commit()
        ck = res["checkpoints"][str(cfg["epochs"])]
        out.append({"run_id": cfg["run_id"], "seconds": res["wall_seconds"], "test_acc": ck["test_acc"]})
        print(out[-1], flush=True)
    return out
