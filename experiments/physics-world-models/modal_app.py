"""Physics world models on Modal: data, reproductions of published baselines, and the levers that try to beat them.

    modal run experiments/physics-world-models/modal_app.py   # download + inspect
    python experiments/physics-world-models/run.py <stage>     # calls the functions below

Two datasets, both small enough to hold in memory:

- The Well, turbulent_radiative_layer_2D (TRL2D): 90 Athena++ runs of a radiative mixing layer, 101 frames of
  128 x 384 with density, pressure and two velocity components; split 72/9/9 by the dataset authors. 7.9 GB.
- PDEBench 2D diffusion-reaction (DR): 1,000 runs of a FitzHugh-Nagumo activator-inhibitor pair, 101 frames of
  128 x 128 with two fields; PDEBench's split is the last 100 runs for test. 13.2 GB.

Official code throughout: The Well's package (pinned commit) for its data module, normalisation, models, rollout
and VRMSE; PDEBench's repository (pinned commit, MIT) for its FNO, U-Net and metric; the tuned-TFNO authors'
repository (pinned commit) and the neuraloperator commit they name, for their released checkpoint.

GPUs: NVIDIA A10G (24 GB) by default; at most two containers at once (the caller enforces it).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

HERE = Path(__file__).parent
app = modal.App("physics-world-models")
vol = modal.Volume.from_name("physics-world-models", create_if_missing=True)

WELL_COMMIT = "aac88c0b5eb9cbe4b7b2797eaaa550a43706fb44"  # the_well 1.2.1
PDEBENCH_COMMIT = "4ff3e3a4aa1561721b5571fa3a048a0a463e0568"
BASELINES_COMMIT = "ca991937e584f40e3a1b18d0d7b8f9493b5a1d87"  # neuraloperator/the_well_baselines
NEURALOP_COMMIT = "53c1817bbe28d48996a0aa4976f9dd9bd0d3fcc2"  # the commit the tuned-TFNO checkpoints were trained with
ENV = {"WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1", "HF_HOME": "/data/hf", "OMP_NUM_THREADS": "8"}

base = (
    modal.Image.from_registry("python:3.11-bookworm")  # full Debian image: git and curl included
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy<2.1", "h5py==3.12.1", "pyyaml", "pandas", "scipy", "huggingface_hub==0.26.2", "einops")
    .run_commands(
        "git clone https://github.com/pdebench/PDEBench.git /opt/PDEBench && "
        f"git -C /opt/PDEBench checkout {PDEBENCH_COMMIT}",
    )
    .env(ENV)
)
# The Well's package with its benchmark extras (neuraloperator 0.3.0, which its FNO/TFNO baselines use).
well_image = base.pip_install(
    f"the_well[benchmark] @ git+https://github.com/PolymathicAI/the_well@{WELL_COMMIT}"
).add_local_file(HERE / "pwm_core.py", "/root/pwm_core.py")
# Same, but with the newer neuraloperator commit the tuned-TFNO checkpoints need (The Well's own FNO is not used here).
tfno_image = (
    base.pip_install(f"the_well[benchmark] @ git+https://github.com/PolymathicAI/the_well@{WELL_COMMIT}")
    .pip_install(
        "tensorly==0.9.0",
        "tensorly-torch==0.5.0",
        "opt-einsum==3.4.0",
        "configmypy==0.2.0",
        "ruamel.yaml",
        "torch-harmonics==0.7.4",
    )  # the version the tuned-TFNO authors pin
    .run_commands(
        f"pip install --no-deps git+https://github.com/neuraloperator/neuraloperator@{NEURALOP_COMMIT}",
        "git clone https://github.com/neuraloperator/the_well_baselines.git /opt/twb && "
        f"git -C /opt/twb checkout {BASELINES_COMMIT}",
    )
    .add_local_file(HERE / "pwm_core.py", "/root/pwm_core.py")
)

POSEIDON_COMMIT = "b8fa28f59bd7f7673323f28d11a12c6f3a215c61"  # camlab-ethz/poseidon; weights camlab-ethz/Poseidon-T
poseidon_image = (
    base.pip_install(f"the_well[benchmark] @ git+https://github.com/PolymathicAI/the_well@{WELL_COMMIT}")
    .pip_install("transformers==4.29.2", "accelerate==0.31.0")
    .run_commands(
        "git clone https://github.com/camlab-ethz/poseidon.git /opt/poseidon && "
        f"git -C /opt/poseidon checkout {POSEIDON_COMMIT}"
    )
    .env({"PYTHONPATH": "/opt/poseidon"})
    .add_local_file(HERE / "pwm_core.py", "/root/pwm_core.py")
)

DR_URL = "https://darus.uni-stuttgart.de/api/access/datafile/133017"
DR_MD5 = "b8d0b86064193195ddc30c33be5dc949"
PDEBENCH_CKPTS = {  # PDEBench's released models (doi:10.18419/darus-2987)
    "2D_diff-react_NA_NA_FNO.pt": 133062,
    "2D_diff-react_NA_NA_Unet-PF-20.pt": 133067,
    "2D_diff-react_NA_NA_Unet-AR.pt": 133065,
    "2D_diff-react_NA_NA_Unet-1-step.pt": 133068,
}
WELL_CKPTS = ["FNO", "TFNO", "UNetClassic", "UNetConvNext"]


@app.function(image=base, volumes={"/data": vol}, timeout=3 * 3600, cpu=4, memory=8192)
def download():
    """Fetch TRL2D (Hugging Face), PDEBench DR (DaRUS, md5-checked) and every released checkpoint into the volume."""
    import hashlib
    import subprocess

    from huggingface_hub import hf_hub_download, snapshot_download

    out = {}
    trl = "/data/well/datasets/turbulent_radiative_layer_2D"
    snapshot_download(
        "polymathic-ai/turbulent_radiative_layer_2D",
        repo_type="dataset",
        local_dir=trl,
        allow_patterns=["data/*", "stats.yaml", "turbulent_radiative_layer_2D.yaml"],
    )
    out["trl_files"] = sorted(str(p.relative_to(trl)) for p in Path(trl).rglob("*.hdf5"))
    vol.commit()
    for m in WELL_CKPTS:
        d = f"/data/ckpt/well/{m}-turbulent_radiative_layer_2D"
        snapshot_download(f"polymathic-ai/{m}-turbulent_radiative_layer_2D", local_dir=d)
    hf_hub_download("yashpincha/TheWellBaselines", "ckpts/turbulent_radiative_layer_2D.pt", local_dir="/data/ckpt/twb")
    os.makedirs("/data/ckpt/pdebench", exist_ok=True)
    for name, fid in PDEBENCH_CKPTS.items():
        p = f"/data/ckpt/pdebench/{name}"
        if not os.path.exists(p):
            subprocess.run(
                ["curl", "-sSL", "-o", p, f"https://darus.uni-stuttgart.de/api/access/datafile/{fid}"], check=True
            )
    vol.commit()
    dr = "/data/pdebench/2D_diff-react_NA_NA.h5"
    os.makedirs(os.path.dirname(dr), exist_ok=True)
    if not os.path.exists(dr):
        subprocess.run(["curl", "-sSL", "-o", dr + ".part", DR_URL], check=True)
        os.rename(dr + ".part", dr)
    h = hashlib.md5()
    with open(dr, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    out["dr_md5_ok"] = h.hexdigest() == DR_MD5
    out["dr_bytes"] = os.path.getsize(dr)
    vol.commit()
    return out


@app.function(image=tfno_image, volumes={"/data": vol}, timeout=600, cpu=2, memory=8192)
def inspect_ckpts():
    """Print what each released checkpoint records about how it was trained (selection split, epochs, config)."""
    import torch

    ck = torch.load("/data/ckpt/twb/ckpts/turbulent_radiative_layer_2D.pt", map_location="cpu", weights_only=False)
    info = {"twb_keys": sorted(ck.keys()), "twb_args": {k: str(v) for k, v in (ck.get("args") or {}).items()}}
    for k in ("epoch", "best_metric", "avg_test_single_vrmse", "avg_test_rollout_vrmse", "avg_test_loss"):
        if k in ck:
            info[k] = float(ck[k]) if not isinstance(ck[k], dict) else str(ck[k])
    for name in PDEBENCH_CKPTS:
        c = torch.load(f"/data/ckpt/pdebench/{name}", map_location="cpu", weights_only=False)
        info[name] = {k: (float(v) if isinstance(v, (int, float)) else type(v).__name__) for k, v in c.items()}
    return info


def _save(name: str, obj: dict):
    import json

    os.makedirs("/data/results", exist_ok=True)
    with open(f"/data/results/{name}.json", "w") as f:
        json.dump(obj, f)
    vol.commit()


GPU = {"gpu": "A10G", "volumes": {"/data": vol}, "memory": 24576, "cpu": 4}  # priced: A10G + 4 cores + 24 GiB


@app.function(image=well_image, timeout=3600, **GPU)
def repro_trl(name: str, split: str = "test") -> dict:
    """Evaluate one of The Well's released TRL2D baselines on a split with The Well's code."""
    import pwm_core as P

    model, prov = P.load_released_trl(name)
    out = {"name": name, "split": split, "provenance": prov, "n_params": sum(p.numel() for p in model.parameters())}
    out |= P.eval_trl(model, split)
    _save(f"repro_trl_{name}" + ("" if split == "test" else f"_{split}"), out)
    return out


@app.function(image=tfno_image, timeout=3600, **GPU)
def repro_twb() -> dict:
    """The tuned-TFNO checkpoint: its authors' own evaluate.py, then our shared evaluation."""
    import subprocess

    import pwm_core as P

    # their loader finds stats.yaml with hdf5_dir.split('/data/')[0], which breaks under a /data mount: use a symlink
    if not os.path.exists("/mnt/welldatasets"):
        os.makedirs("/mnt", exist_ok=True)
        os.symlink("/data/well/datasets", "/mnt/welldatasets")
    r = subprocess.run(
        ["python", "evaluate.py", "--ckpt", "/data/ckpt/twb/ckpts/turbulent_radiative_layer_2D.pt"],
        cwd="/opt/twb",
        env={**os.environ, "WELL_DATASETS_PATH": "/mnt/welldatasets"},
        capture_output=True,
        text=True,
        check=False,
    )
    model, prov = P.load_released_trl("tfno_tuned_released")
    out = {
        "name": "tfno_tuned_released",
        "provenance": prov,
        "authors_evaluate_stdout": r.stdout[-3000:],
        "authors_evaluate_stderr": r.stderr[-2000:],
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    out |= P.eval_trl(model, "test")
    _save("repro_trl_tfno_tuned_released", out)
    return out


@app.function(image=well_image, timeout=3600, volumes={"/data": vol}, cpu=4, memory=8192)
def prepare_dr() -> dict:
    import pwm_core as P

    out = P.dr_prepare()
    vol.commit()
    return out


@app.function(image=well_image, timeout=3600, **GPU)
def repro_dr(name: str) -> dict:
    """Evaluate one of PDEBench's released DR models with PDEBench's rollout and metric."""
    import pwm_core as P

    out = {"name": name} | P.eval_dr(P.load_released_dr(name), "test")
    _save(f"repro_dr_{name.replace('.pt', '')}", out)
    return out


@app.function(image=well_image, timeout=1800, volumes={"/data": vol}, cpu=4, memory=32768)
def check_trl() -> dict:
    """Our GPU-resident training windows must equal The Well's WellDataset samples (same frames, normalisation)."""
    from the_well.data import WellDataset
    from the_well.data.normalization import ZScoreNormalization

    import pwm_core as P

    ds = WellDataset(
        well_base_path=P.WELL_BASE,
        well_dataset_name=P.TRL,
        well_split_name="train",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    x = P.load_trl("train", normalize=True)[:1]
    out = {"len_dataset": len(ds)}
    for i in (0, 37, 96):
        s = ds[i]
        mine_in = x[0, i : i + 4].permute(0, 2, 3, 1)  # (T, H, W, C) like WellDataset
        mine_out = x[0, i + 4 : i + 5].permute(0, 2, 3, 1)
        out[f"max_abs_diff_in_{i}"] = float((s["input_fields"] - mine_in).abs().max())
        out[f"max_abs_diff_out_{i}"] = float((s["output_fields"] - mine_out).abs().max())
    return out


@app.function(image=well_image, timeout=900, volumes={"/data": vol}, cpu=2, memory=8192)
def check_ckpt_keys(name: str) -> dict:
    """Does a released Well checkpoint's state dict match the model The Well's package builds for it?"""
    import json as _json

    from safetensors.torch import load_file
    from the_well.benchmark import models as Mo

    d = f"/data/ckpt/well/{name}-turbulent_radiative_layer_2D"
    with open(f"{d}/config.json") as fh:
        cfg = _json.load(fh)
    cls = {"FNO": Mo.FNO, "TFNO": Mo.TFNO, "UNetClassic": Mo.UNetClassic, "UNetConvNext": Mo.UNetConvNext}[name]
    model = cls(**cfg)
    sd = load_file(f"{d}/model.safetensors")
    ms = model.state_dict()
    shape_mismatch = [k for k in sd if k in ms and tuple(sd[k].shape) != tuple(ms[k].shape)]
    return {
        "config": cfg,
        "missing": [k for k in ms if k not in sd][:20],
        "unexpected": [k for k in sd if k not in ms][:20],
        "n_missing": sum(k not in sd for k in ms),
        "n_unexpected": sum(k not in ms for k in sd),
        "shape_mismatch": [(k, tuple(sd[k].shape), tuple(ms[k].shape)) for k in shape_mismatch][:10],
        "ckpt_keys_sample": list(sd)[:12],
    }


@app.function(image=well_image, timeout=1800, **GPU)
def probe_norm(name: str) -> dict:
    """One-step test VRMSE of a released Well checkpoint under four input/output conventions, to find the one it was
    trained with: z-score full state (The Well's default), raw units, RMS scaling, and z-score input + delta output."""
    import torch
    import yaml

    import pwm_core as P

    model, _ = P.load_released_trl(name)
    model = model.cuda().eval()
    with open(f"{P.WELL_BASE}{P.TRL}/stats.yaml") as fh:
        s = yaml.safe_load(fh)

    def vec(k):
        return torch.tensor([s[k]["density"], s[k]["pressure"], *s[k]["velocity"]]).view(1, -1, 1, 1).cuda()

    mean, std, rms = vec("mean"), vec("std").clamp(min=1e-4), vec("rms").clamp(min=1e-4)
    md, sd, rmsd = vec("mean_delta"), vec("std_delta").clamp(min=1e-4), vec("rms_delta").clamp(min=1e-4)
    x = P.load_trl("test").cuda()  # raw (9, 101, 4, H, W)
    out = {}
    variants = {
        "zscore": (lambda z: (z - mean) / std, lambda y, last: y * std + mean),
        "raw": (lambda z: z, lambda y, last: y),
        "rms": (lambda z: z / rms, lambda y, last: y * rms),
        "zscore_delta": (lambda z: (z - mean) / std, lambda y, last: last + y * sd + md),
        "rms_delta": (lambda z: z / rms, lambda y, last: last + y * rmsd),
    }
    with torch.no_grad():
        for vname, (fin, fout) in variants.items():
            vals = []
            for i in range(x.shape[0]):
                for t0 in range(0, 96, 4):
                    w = x[i, t0 : t0 + 5]
                    inp = fin(w[:4]).reshape(1, 16, 128, 384)
                    pred = fout(model(inp), w[3:4])
                    vals.append(P.vrmse_bc(pred, w[4:5]).mean().item())
            out[vname] = sum(vals) / len(vals)
    return out


def _train(cfg: dict) -> dict:
    import torch

    import pwm_core as P

    run = cfg["run"]
    if cfg["dataset"] == "trl":
        res = P.train_trl(cfg)
        net = P.build_trl(cfg)
        net.load_state_dict(res.pop("state_dict"))
        ev = P.eval_trl(net, cfg.get("eval_split", "valid"), official=cfg.get("eval_split") == "test")
    else:
        res = P.train_dr(cfg)
        net = P.DRNet(cfg["kind"], cfg)
        net.load_state_dict(res.pop("state_dict"))
        ev = P.eval_dr(net, cfg.get("eval_split", "valid"))
    os.makedirs("/data/runs", exist_ok=True)
    torch.save(net.state_dict(), f"/data/runs/{run}.pt")
    out = {"cfg": cfg, **res, "eval_split": cfg.get("eval_split", "valid"), "eval": ev}
    _save(f"run_{run}", out)
    return out


@app.function(image=well_image, timeout=4 * 3600, **GPU)
def train(cfg: dict) -> dict:
    return _train(cfg)


@app.function(image=tfno_image, timeout=4 * 3600, **GPU)
def train_tfno(cfg: dict) -> dict:
    return _train(cfg)


@app.function(image=poseidon_image, timeout=4 * 3600, **GPU)
def train_poseidon(cfg: dict) -> dict:
    return _train(cfg)


@app.function(image=well_image, timeout=1200, **GPU)
def hero_frames() -> dict:
    """Frames for the note's cover and rollout figure: one TRL2D test run (density, raw units) rolled out by the
    released CNextU-net and by our pushforward fine-tune (seed 1), and one DR test run by our CNextU-net (seed 1)."""
    import numpy as np
    import torch

    import pwm_core as P

    steps = [0, 5, 10, 20, 40, 60, 96]
    mean, std = P.trl_stats()
    x = P.load_trl("test", normalize=True)[:1].cuda()  # (1, 101, 4, H, W)
    out = {"trl_steps": steps, "trl_truth": [], "trl": {}}
    m, s = mean[0].item(), std[0].item()
    for t in steps:
        out["trl_truth"].append((x[0, 4 + t, 0] * s + m).half().cpu().numpy().tolist())
    released, _ = P.load_released_trl("UNetConvNext")
    ours = P.build_trl({"kind": "unet_convnext"})
    ours.load_state_dict(torch.load("/data/runs/f_trl_ft_pf4_s1.pt", map_location="cpu"))
    for name, net in (("released", released), ("pushforward", ours)):
        net = net.cuda().eval()
        ctx, frames = x[:, :4], []
        with torch.no_grad():
            for t in range(97):
                y = net(ctx.flatten(1, 2)).float()
                if t in steps:
                    frames.append((y[0, 0] * s + m).half().cpu().numpy().tolist())
                ctx = torch.cat([ctx[:, 1:], y[:, None]], 1)
        out["trl"][name] = frames
    xd = P.load_dr([900])  # first DR test run, (1, 101, 2, 128, 128)
    net = P.DRNet("unet_convnext", {"n_ctx": 10})
    net.load_state_dict(torch.load("/data/runs/f_dr_cnext_s1.pt", map_location="cpu"))
    pred = P.dr_rollout(net, xd, 10)
    out["dr_frames"] = [10, 30, 60, 100]
    out["dr_truth"] = [xd[0, t, 0].numpy().astype(np.float16).tolist() for t in out["dr_frames"]]
    out["dr_pred"] = [pred[0, t, 0].numpy().astype(np.float16).tolist() for t in out["dr_frames"]]
    return out


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(download.remote(), indent=1))
    print(json.dumps(inspect_ckpts.remote(), indent=1))
