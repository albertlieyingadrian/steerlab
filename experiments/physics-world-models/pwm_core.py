"""Shared code for the physics-world-models study. Runs inside the Modal images defined in modal_app.py.

Two halves, one per dataset:

TRL2D (The Well, turbulent_radiative_layer_2D)
    Evaluation goes through The Well's own WellDataModule (test split, z-score normalisation from stats.yaml,
    4 input frames), its Trainer.rollout_model and its VRMSE metric. Every model, ours or released, is wrapped to
    take The Well's channels-first input (B, 4 frames x 4 fields, 128, 384) in normalised units, so one-step and
    rollout scores are computed on exactly the same targets: frames 4..100 of the 9 test trajectories.
    Training is our own loop over the 72 training trajectories held on the GPU (same normalisation), so the
    levers (pushforward, input noise, residual output, schedule) can be switched one at a time.

DR (PDEBench 2D diffusion-reaction)
    Evaluation follows PDEBench's FNO script: 10 input frames, an autoregressive rollout to frame 100 on the last
    100 trajectories, and PDEBench's nRMSE (metric_func, imported from the pinned repository). We add VRMSE on
    the short horizon (frames 10-20) and the full horizon (frames 10-100) and the error amplification G of
    arXiv 2610.31935: per trajectory, VRMSE averaged over frames and fields; the median over test trajectories;
    G = full / short.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

WELL_BASE = "/data/well/datasets/"
TRL = "turbulent_radiative_layer_2D"
N_IN = 4  # The Well's context length; every TRL2D model receives 4 frames (models that use fewer slice them)
C_TRL = 4
EPS = 1e-7


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# TRL2D data
# ----------------------------------------------------------------------------------------------------------------


def _stats_lists():
    import yaml

    with open(f"{WELL_BASE}{TRL}/stats.yaml") as fh:
        s = yaml.safe_load(fh)
    mean = [s["mean"]["density"], s["mean"]["pressure"], *s["mean"]["velocity"]]
    std = [s["std"]["density"], s["std"]["pressure"], *s["std"]["velocity"]]
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def trl_stats():
    """Mean and std per channel (density, pressure, vx, vy), as The Well's ZScoreNormalization uses them."""
    mean, std = _stats_lists()
    return mean, torch.clip(std, min=1e-4)  # ZScoreNormalization(min_denom=1e-4)


def load_trl(split: str, normalize: bool = False) -> torch.Tensor:
    """All trajectories of a split as (N, T, C, H, W) float32, files in sorted order; raw units unless `normalize`
    (then z-scored in place with trl_stats, to keep peak memory at one copy)."""
    import h5py

    files = sorted(glob.glob(f"{WELL_BASE}{TRL}/data/{split}/*.hdf5"))
    with h5py.File(files[0], "r") as h:
        n, T, H, W = h["t0_fields/density"].shape
    out = torch.empty((n * len(files), T, C_TRL, H, W), dtype=torch.float32)
    for i, f in enumerate(files):
        with h5py.File(f, "r") as h:
            sl = slice(i * n, (i + 1) * n)
            out[sl, :, 0] = torch.from_numpy(h["t0_fields/density"][:])
            out[sl, :, 1] = torch.from_numpy(h["t0_fields/pressure"][:])
            out[sl, :, 2:] = torch.from_numpy(h["t1_fields/velocity"][:]).permute(0, 1, 4, 2, 3)
    if normalize:
        mean, std = trl_stats()
        out.sub_(mean.view(1, 1, -1, 1, 1)).div_(std.view(1, 1, -1, 1, 1))
    return out


def vrmse_bc(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """The Well's VRMSE for (B, C, H, W) tensors in raw units -> (B, C). Unbiased spatial variance, eps 1e-7."""
    mse = ((pred - ref) ** 2).mean(dim=(-2, -1))
    var = ref.flatten(-2).std(dim=-1) ** 2
    return torch.sqrt(mse / (var + EPS))


# ----------------------------------------------------------------------------------------------------------------
# TRL2D models. Every model maps (B, 16, H, W) normalised -> (B, 4, H, W) normalised.
# ----------------------------------------------------------------------------------------------------------------


class Wrap(nn.Module):
    """Feed the inner network the last `n_in` frames; optionally predict a residual on the last frame."""

    def __init__(self, inner: nn.Module, n_in: int = 4, residual: bool = False, c: int = C_TRL):
        super().__init__()
        self.inner, self.n_in, self.residual, self.c = inner, n_in, residual, c

    def forward(self, x):
        z = x[:, -self.n_in * self.c :]
        y = self.inner(z)
        if self.residual:
            y = y + x[:, -self.c :]
        return y


class Renorm(nn.Module):
    """Run a one-frame model trained with another per-channel normalisation inside The Well's normalised space."""

    def __init__(self, inner, well_mean, well_std, own_mean, own_std):
        super().__init__()
        self.inner = inner
        for k, v in {"wm": well_mean, "ws": well_std, "om": own_mean, "osd": own_std}.items():
            self.register_buffer(k, v.view(1, -1, 1, 1).float())

    def forward(self, x):  # x: (B, 4, H, W), one frame
        raw = x * self.ws + self.wm
        y = self.inner((raw - self.om) / self.osd)
        return (y * self.osd + self.om - self.wm) / self.ws


def build_trl(cfg: dict) -> nn.Module:
    """Build a TRL2D model from a config dict (kind, n_in, residual and per-kind sizes)."""
    kind, n_in = cfg["kind"], cfg.get("n_in", 4)
    dim_in = n_in * C_TRL
    if kind == "fno_well":  # The Well's FNO baseline (neuraloperator 0.3.0)
        from the_well.benchmark.models import FNO

        inner = FNO(
            dim_in,
            C_TRL,
            2,
            (128, 384),
            modes1=cfg.get("modes", 16),
            modes2=cfg.get("modes", 16),
            hidden_channels=cfg.get("width", 128),
        )
    elif kind == "tfno_well":
        from the_well.benchmark.models import TFNO

        inner = TFNO(
            dim_in,
            C_TRL,
            2,
            (128, 384),
            modes1=cfg.get("modes", 16),
            modes2=cfg.get("modes", 16),
            hidden_channels=cfg.get("width", 128),
        )
    elif kind == "unet_convnext":  # The Well's CNextU-net
        from the_well.benchmark.models import UNetConvNext

        inner = UNetConvNext(
            dim_in,
            C_TRL,
            2,
            (128, 384),
            stages=cfg.get("stages", 4),
            blocks_per_stage=cfg.get("blocks_per_stage", 2),
            blocks_at_neck=1,
            init_features=cfg.get("init_features", 42),
        )
    elif kind == "tfno_tuned":  # the tuned-TFNO authors' architecture (neuraloperator @53c1817, tfno_image only)
        from neuralop.models import FNO as NFNO

        inner = NFNO(
            n_modes=(cfg.get("modes", 64),) * 2,
            hidden_channels=cfg.get("width", 64),
            in_channels=dim_in,
            out_channels=C_TRL,
            n_layers=cfg.get("n_layers", 6),
            lifting_channel_ratio=4,
            projection_channel_ratio=4,
            domain_padding=cfg.get("domain_padding", 0.0),
            factorization="tucker",
            rank=cfg.get("rank", 0.1),
        )
    else:
        raise ValueError(kind)
    return Wrap(inner, n_in=n_in, residual=cfg.get("residual", False))


def load_released_trl(name: str) -> tuple[nn.Module, dict]:
    """A released TRL2D checkpoint, wrapped to The Well's interface. Returns (model, provenance)."""
    if name == "tfno_tuned_released":
        from neuralop.models import FNO as NFNO

        ck = torch.load("/data/ckpt/twb/ckpts/turbulent_radiative_layer_2D.pt", map_location="cpu", weights_only=False)
        a = ck["args"]
        inner = NFNO(
            n_modes=(a["modes"],) * 2,
            hidden_channels=a["width"],
            in_channels=4,
            out_channels=4,
            n_layers=a["n_layers"],
            lifting_channel_ratio=4,
            projection_channel_ratio=4,
            domain_padding=a["domain_padding"],
            factorization=a["factorization"],
            rank=a["rank"],
        )
        missing, unexpected = inner.load_state_dict(ck.get("model", ck), strict=False)
        assert not missing and not unexpected, (missing[:3], unexpected[:3])
        wm, ws = trl_stats()
        om, osd = _stats_lists()
        osd = osd + 1e-6  # their WellHDF5Dataset divides by (std + 1e-6)
        if not a.get("normalize", 1):
            om, osd = torch.zeros(4), torch.ones(4)
        model = Wrap(Renorm(inner, wm, ws, om, osd), n_in=1)
        prov = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in a.items()}
        for k in ("epoch", "best_metric", "avg_test_single_vrmse"):
            if k in ck:
                prov[f"ckpt_{k}"] = float(ck[k])
        return model, prov
    from the_well.benchmark import models as M

    cls = {"FNO": M.FNO, "TFNO": M.TFNO, "UNetClassic": M.UNetClassic, "UNetConvNext": M.UNetConvNext}[name]
    inner = cls.from_pretrained(f"/data/ckpt/well/{name}-{TRL}")
    return Wrap(inner, n_in=4), {"source": f"polymathic-ai/{name}-{TRL}"}


# ----------------------------------------------------------------------------------------------------------------
# TRL2D evaluation through The Well's own code
# ----------------------------------------------------------------------------------------------------------------


def _well_trainer(model, batch_size=16, workdir="/tmp/welleval"):
    from the_well.benchmark.metrics import MSE
    from the_well.benchmark.trainer import Trainer
    from the_well.data import WellDataModule

    dm = WellDataModule(
        well_base_path=WELL_BASE,
        well_dataset_name=TRL,
        batch_size=batch_size,
        use_normalization=True,
        n_steps_input=N_IN,
        n_steps_output=1,
        max_rollout_steps=100,
        data_workers=4,
    )
    for d in ("ck", "art", "viz"):
        os.makedirs(f"{workdir}/{d}", exist_ok=True)
    tr = Trainer(
        checkpoint_folder=f"{workdir}/ck",
        artifact_folder=f"{workdir}/art",
        viz_folder=f"{workdir}/viz",
        formatter="channels_first_default",
        model=model,
        datamodule=dm,
        optimizer=None,
        loss_fn=MSE(),
        epochs=1,
        checkpoint_frequency=0,
        val_frequency=1,
        rollout_val_frequency=1,
        max_rollout_steps=100,
        short_validation_length=20,
        make_rollout_videos=False,
        num_time_intervals=5,
        device=torch.device("cuda"),
    )
    return tr, dm


@torch.inference_mode()
def eval_trl(model: nn.Module, split: str = "test", official: bool = True) -> dict:
    """One-step and full-rollout VRMSE on a TRL2D split with The Well's data module, rollout and metric.

    Returns per-sample one-step VRMSE (n_windows, C), per-trajectory rollout VRMSE (n_traj, 97, C) and, when
    `official`, the scalars The Well's Trainer.validation_loop reports for the same model.
    """
    from the_well.benchmark.metrics import VRMSE

    model = model.cuda().eval()
    tr, dm = _well_trainer(model)
    meta = dm.train_dataset.metadata
    one_loader = dm.test_dataloader() if split == "test" else dm.val_dataloader()
    roll_loader = dm.rollout_test_dataloader() if split == "test" else dm.rollout_val_dataloader()
    one = []
    for batch in one_loader:
        yp, yr = tr.rollout_model(model, batch, tr.formatter, train=False)
        one.append(VRMSE.eval(yp, yr, meta)[:, 0].float().cpu())  # (B, C)
    one = torch.cat(one).numpy()
    roll = []
    for batch in roll_loader:
        yp, yr = tr.rollout_model(model, batch, tr.formatter, train=False)
        roll.append(VRMSE.eval(yp, yr, meta).float().cpu())  # (1, T, C)
    roll = torch.cat(roll).numpy()
    out = {
        "one_step": one.round(6).tolist(),
        "rollout": roll.round(6).tolist(),
        "n_windows": int(one.shape[0]),
        "n_traj": int(roll.shape[0]),
        "fields": ["density", "pressure", "velocity_x", "velocity_y"],
    }
    if official and split == "test":
        _, d = tr.validation_loop(one_loader, valid_or_test="test", full=True)
        out["official_one_step"] = {k: v for k, v in d.items() if "VRMSE" in k and "full" in k}
        _, d = tr.validation_loop(roll_loader, valid_or_test="rollout_test", full=True)
        out["official_rollout"] = {k: v for k, v in d.items() if "VRMSE" in k and "full" in k}
    return out


# ----------------------------------------------------------------------------------------------------------------
# Training loop (TRL2D and DR share it through a small data adapter)
# ----------------------------------------------------------------------------------------------------------------


def lr_at(frac: float, peak: float, warmup: float = 0.05) -> float:
    """Linear warm-up for the first 5% of the time budget, then cosine to zero at the end of it."""
    if frac < warmup:
        return peak * frac / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - warmup) / (1 - warmup))))


class TrajData:
    """Trajectories (N, T, C, H, W) in model units; draws windows of n_ctx + k + 1 frames."""

    def __init__(self, x: torch.Tensor, n_ctx: int, channels_last: bool = False):
        self.x, self.n_ctx, self.channels_last = x, n_ctx, channels_last  # channels_last: x is (N, T, H, W, C)
        self.N, self.T = x.shape[:2]

    def batch(self, b: int, k: int, gen: torch.Generator):
        idx = torch.randint(0, self.N, (b,), generator=gen)
        t0 = torch.randint(0, self.T - self.n_ctx - k, (b,), generator=gen)
        tt = t0[:, None] + torch.arange(self.n_ctx + k + 1)[None]
        w = self.x[idx[:, None].to(self.x.device), tt.to(self.x.device)]  # (b, n_ctx+k+1, C, H, W)
        w = w.cuda(non_blocking=True).float()
        return w.permute(0, 1, 4, 2, 3) if self.channels_last else w


def train_loop(net, data: TrajData, cfg: dict, val_fn=None, forward=None) -> dict:
    """Time-boxed training. cfg: lr, wd, batch, minutes, seed, amp, pf_max (pushforward depth), pf_prob,
    noise (input noise std in model units), grad_clip. `forward(net, ctx)` maps (b, n_ctx, C, H, W) -> (b, C, H, W).
    The final weights (end of the cosine schedule) are the ones evaluated; validation is for monitoring only."""
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    net = net.cuda().train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-4))
    budget = cfg["minutes"] * 60
    amp = cfg.get("amp", False)
    pf_max, pf_prob, noise = cfg.get("pf_max", 0), cfg.get("pf_prob", 0.0), cfg.get("noise", 0.0)
    logs, step, t0 = [], 0, time.time()
    next_val = budget * 0.1
    ema = None
    while True:
        el = time.time() - t0
        if el >= budget:
            break
        lr = lr_at(el / budget, cfg["lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        k = 0
        if pf_max > 0 and torch.rand(1, generator=gen).item() < pf_prob:
            k = int(torch.randint(1, pf_max + 1, (1,), generator=gen).item())
        w = data.batch(cfg["batch"], k, gen)
        ctx = w[:, : data.n_ctx]
        if k > 0:  # pushforward: unroll k steps without gradient, then train the next step on the model's own input
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                for _ in range(k):
                    y = forward(net, ctx).float()
                    ctx = torch.cat([ctx[:, 1:], y[:, None]], dim=1)
        if noise > 0:
            ctx = ctx + noise * torch.randn_like(ctx)
        target = w[:, data.n_ctx + k]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = forward(net, ctx)
        loss = F.mse_loss(pred.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step()
        step += 1
        lv = loss.item()
        if not math.isfinite(lv):
            logs.append({"step": step, "t": round(el), "error": "non-finite loss"})
            log(json.dumps(logs[-1]))
            break
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if val_fn is not None and el >= next_val:
            net.eval()
            v = val_fn(net)
            net.train()
            logs.append({"step": step, "t": round(el), "lr": lr, "train_loss_ema": ema, **v})
            log(json.dumps(logs[-1]))
            next_val += budget * 0.1
    net.eval()
    final = val_fn(net) if val_fn is not None else {}
    logs.append({"step": step, "t": round(time.time() - t0), "final": True, "train_loss_ema": ema, **final})
    log(json.dumps(logs[-1]))
    return {"steps": step, "logs": logs}


def trl_forward(net, ctx):
    return net(ctx.flatten(1, 2))


def trl_val_fn(xva_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Validation monitor: one-step VRMSE on 117 fixed windows and 20-step rollout VRMSE from frame 4."""
    m, s = mean.view(1, -1, 1, 1).cuda(), std.view(1, -1, 1, 1).cuda()
    starts = list(range(0, 97, 8))

    @torch.no_grad()
    def fn(net):
        one = []
        for i in range(xva_norm.shape[0]):
            w = torch.stack([xva_norm[i, t : t + 5] for t in starts]).cuda().float()
            p = net(w[:, :4].flatten(1, 2)).float()
            one.append(vrmse_bc(p * s + m, w[:, 4] * s + m).mean().item())
        ctx = xva_norm[:, :4].cuda().float()
        rs = []
        for t in range(20):
            y = net(ctx.flatten(1, 2)).float()
            rs.append(vrmse_bc(y * s + m, xva_norm[:, 4 + t].cuda().float() * s + m).mean().item())
            ctx = torch.cat([ctx[:, 1:], y[:, None]], 1)
        return {"val_one_step": float(np.mean(one)), "val_roll20": float(np.mean(rs))}

    return fn


def train_trl(cfg: dict) -> dict:
    mean, std = trl_stats()
    t = time.time()
    xtr = load_trl("train", normalize=True)
    xva = load_trl("valid", normalize=True)
    log(f"loaded TRL2D train {tuple(xtr.shape)} valid {tuple(xva.shape)} in {time.time() - t:.0f}s")
    data = TrajData(xtr.cuda() if cfg.get("data_on_gpu", True) else xtr, n_ctx=N_IN)
    del xtr
    net = build_trl(cfg)
    if cfg.get("init"):  # warm start from a released checkpoint (same architecture and wrapper)
        released, _ = load_released_trl(cfg["init"])
        net.load_state_dict(released.state_dict())
        log(f"initialised from released {cfg['init']}")
    n_params = sum(p.numel() for p in net.parameters())
    log(f"model {cfg['kind']} params {n_params:,}")
    fn = trl_val_fn(xva, mean, std)
    if cfg.get("init"):
        log(json.dumps({"before_training": True, **fn(net.cuda().eval())}))
    res = train_loop(net, data, cfg, val_fn=fn, forward=trl_forward)
    del data
    torch.cuda.empty_cache()
    res["n_params"] = n_params
    res["state_dict"] = {k: v.cpu() for k, v in net.state_dict().items()}
    return res


# ----------------------------------------------------------------------------------------------------------------
# DR (PDEBench 2D diffusion-reaction)
# ----------------------------------------------------------------------------------------------------------------

DR_H5 = "/data/pdebench/2D_diff-react_NA_NA.h5"
DR_NPY = "/data/pdebench/dr_all.npy"  # (1000, 101, 128, 128, 2) float32, groups in sorted order
DR_INIT = 10


def dr_prepare():
    """Copy the 1,000 trajectories into one .npy in sorted-key order (PDEBench's split: last 100 are test)."""
    import h5py

    if os.path.exists(DR_NPY):
        return {"exists": True}
    with h5py.File(DR_H5, "r") as h:
        keys = sorted(h.keys())
        shape = (len(keys), 101, 128, 128, 2)
        arr = np.lib.format.open_memmap(DR_NPY + ".part", mode="w+", dtype=np.float32, shape=shape)
        for i, k in enumerate(keys):
            arr[i] = h[k]["data"][:]
        gx, gy = h[keys[0]]["grid"]["x"][:], h[keys[0]]["grid"]["y"][:]
    arr.flush()
    del arr
    os.rename(DR_NPY + ".part", DR_NPY)
    np.save("/data/pdebench/dr_grid.npy", np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).astype(np.float32))
    return {"n": len(keys), "first": keys[0], "last": keys[-1]}


def dr_split():
    """Indices: PDEBench test = last 100 (sorted keys); we hold out the 50 runs before them as validation."""
    return {"train": list(range(850)), "valid": list(range(850, 900)), "test": list(range(900, 1000))}


def dr_grid():
    return torch.from_numpy(np.load("/data/pdebench/dr_grid.npy"))  # (128, 128, 2)


def pdebench_import():
    sys.path.insert(0, "/opt/PDEBench")
    from pdebench.models.fno.fno import FNO2d
    from pdebench.models.metrics import metric_func
    from pdebench.models.unet.unet import UNet2d

    return FNO2d, UNet2d, metric_func


class DRNet(nn.Module):
    """Uniform DR interface: (B, X, Y, n_ctx*C) frames (PDEBench layout, time-major) -> (B, X, Y, C)."""

    def __init__(self, kind: str, cfg: dict):
        super().__init__()
        self.kind, self.n_ctx = kind, cfg.get("n_ctx", DR_INIT)
        self.residual = cfg.get("residual", False)
        FNO2d, UNet2d, _ = pdebench_import()
        if kind == "pdebench_fno":
            self.net = FNO2d(
                num_channels=2,
                modes1=cfg.get("modes", 12),
                modes2=cfg.get("modes", 12),
                width=cfg.get("width", 20),
                initial_step=self.n_ctx,
            )
            self.register_buffer("grid", dr_grid()[None])
        elif kind == "unet_convnext":
            from the_well.benchmark.models import UNetConvNext

            self.net = UNetConvNext(
                self.n_ctx * 2,
                2,
                2,
                (128, 128),
                stages=cfg.get("stages", 4),
                blocks_per_stage=cfg.get("blocks_per_stage", 2),
                blocks_at_neck=1,
                init_features=cfg.get("init_features", 42),
            )
        elif kind == "pdebench_unet":
            self.net = UNet2d(self.n_ctx * 2, 2)
        elif kind == "poseidon":  # Poseidon (scOT), one input frame, new 2-channel embedding and recovery
            from scOT.model import ScOT, ScOTConfig

            name = f"camlab-ethz/Poseidon-{cfg.get('size', 'T')}"
            pc = ScOTConfig.from_pretrained(name)
            pc.num_channels, pc.num_out_channels, pc.channel_slice_list_normalized_loss = self.n_ctx * 2, 2, None
            if cfg.get("pretrained", True):
                self.net = ScOT.from_pretrained(name, config=pc, ignore_mismatched_sizes=True)
            else:
                self.net = ScOT(pc)
            self.lead = cfg.get("lead_time", 0.1)
            # per-channel scaling (train-set statistics of PDEBench DR, frames 0-100); Poseidon expects O(1) inputs
            self.register_buffer("mu", torch.tensor(cfg.get("dr_mean", [0.0, 0.0])).view(1, 2, 1, 1))
            self.register_buffer("sd", torch.tensor(cfg.get("dr_std", [1.0, 1.0])).view(1, 2, 1, 1))
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "pdebench_fno":
            y = self.net(x, self.grid.expand(x.shape[0], -1, -1, -1))[..., 0, :]
        elif self.kind == "poseidon":
            z = x.permute(0, 3, 1, 2)
            b = z.shape[0]
            zn = (z.view(b, -1, 2, *z.shape[-2:]) - self.mu[:, None]) / self.sd[:, None]
            t = torch.full((b,), self.lead, device=x.device)
            out = self.net(pixel_values=zn.flatten(1, 2), time=t).output
            y = (out * self.sd + self.mu).permute(0, 2, 3, 1)
        else:
            y = self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.residual:
            y = y + x[..., -2:]
        return y


def dr_forward(net, ctx):
    """ctx (b, n_ctx, C, X, Y) -> (b, C, X, Y), via the PDEBench layout (b, X, Y, n_ctx*C)."""
    b, t, c, X, Y = ctx.shape
    x = ctx.permute(0, 3, 4, 1, 2).reshape(b, X, Y, t * c)
    return net(x).permute(0, 3, 1, 2)


def load_dr(indices, channels_last: bool = False) -> torch.Tensor:
    """Trajectories as (N, T, C, X, Y) float32, or (N, T, X, Y, C) as stored (indices must be contiguous)."""
    arr = np.load(DR_NPY, mmap_mode="r")
    x = torch.from_numpy(np.ascontiguousarray(arr[indices[0] : indices[-1] + 1]))
    return x if channels_last else x.permute(0, 1, 4, 2, 3).contiguous()


@torch.inference_mode()
def dr_rollout(net, x: torch.Tensor, n_ctx: int, batch: int = 10) -> torch.Tensor:
    """Autoregressive rollout from the first 10 frames to the last. x (N, T, C, X, Y) raw -> predictions
    (N, T, C, X, Y) with the first 10 frames copied from the data, as PDEBench's metrics() does."""
    net = net.cuda().eval()
    out = []
    for i in range(0, x.shape[0], batch):
        xb = x[i : i + batch].cuda().float()
        pred = [xb[:, t] for t in range(DR_INIT)]
        for _ in range(DR_INIT, xb.shape[1]):
            ctx = torch.stack(pred[-n_ctx:], 1)
            pred.append(dr_forward(net, ctx).float())
        out.append(torch.stack(pred, 1).cpu())
    return torch.cat(out)


def dr_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    """PDEBench nRMSE (its metric_func, batches of 5 as its FNO evaluation uses) plus VRMSE short/full and G."""
    _, _, metric_func = pdebench_import()
    nr = []
    for i in range(0, pred.shape[0], 5):
        p = pred[i : i + 5].permute(0, 3, 4, 1, 2)  # (b, X, Y, T, C), PDEBench layout
        r = ref[i : i + 5].permute(0, 3, 4, 1, 2)
        _, err_nrmse, *_ = metric_func(p.cuda(), r.cuda(), if_mean=True, initial_step=DR_INIT)
        nr.append(err_nrmse.item())
    # VRMSE per trajectory, frame and field (frames 10..100), eps 1e-7, unbiased spatial variance
    v = vrmse_bc(pred[:, DR_INIT:].flatten(0, 1).cuda(), ref[:, DR_INIT:].flatten(0, 1).cuda()).cpu()
    v = v.view(pred.shape[0], -1, pred.shape[2])  # (N, 91, C)
    e_short = v[:, :11].mean(dim=(1, 2))  # frames 10-20
    e_full = v.mean(dim=(1, 2))  # frames 10-100
    return {
        "nrmse_pdebench": float(np.mean(nr)),
        "vrmse_short_median": float(e_short.median()),
        "vrmse_full_median": float(e_full.median()),
        "G_median_ratio": float(e_full.median() / e_short.median()),
        "vrmse_per_traj_frame": v.numpy().round(6).tolist(),
    }


def load_released_dr(name: str) -> nn.Module:
    ck = torch.load(f"/data/ckpt/pdebench/{name}", map_location="cpu", weights_only=False)
    if "FNO" in name:
        net = DRNet("pdebench_fno", {"modes": 12, "width": 20, "n_ctx": 10})
    else:
        net = DRNet("pdebench_unet", {"n_ctx": 10})
    net.net.load_state_dict(ck["model_state_dict"])
    return net


def train_dr(cfg: dict) -> dict:
    sp = dr_split()
    n_train = cfg.get("n_train", 850)
    t = time.time()
    xtr = load_dr(sp["train"][:n_train], channels_last=True)
    xva = load_dr(sp["valid"])
    log(f"loaded DR train {tuple(xtr.shape)} valid {tuple(xva.shape)} in {time.time() - t:.0f}s")
    n_ctx = cfg.get("n_ctx", DR_INIT)
    if cfg["kind"] == "poseidon":  # channel statistics for Poseidon's input scaling, from the training runs only
        cfg["dr_mean"] = xtr.mean(dim=(0, 1, 2, 3)).tolist()
        cfg["dr_std"] = xtr.std(dim=(0, 1, 2, 3)).tolist()
    data = TrajData(xtr.cuda() if cfg.get("data_on_gpu", True) else xtr, n_ctx=n_ctx, channels_last=True)
    del xtr
    net = DRNet(cfg["kind"], cfg)
    n_params = sum(p.numel() for p in net.parameters())
    log(f"model {cfg['kind']} params {n_params:,}")

    def val_fn(m):
        pred = dr_rollout(m, xva[:, :21], n_ctx)  # monitor: frames 10-20 only
        v = vrmse_bc(pred[:, DR_INIT:].flatten(0, 1).cuda(), xva[:, DR_INIT:21].flatten(0, 1).cuda().float())
        return {"val_vrmse_short": float(v.mean())}

    res = train_loop(net, data, cfg, val_fn=val_fn, forward=dr_forward)
    del data
    torch.cuda.empty_cache()
    res["n_params"] = n_params
    res["state_dict"] = {k: v.cpu() for k, v in net.state_dict().items()}
    return res


def eval_dr(net, split="test") -> dict:
    x = load_dr(dr_split()[split])
    pred = dr_rollout(net, x, getattr(net, "n_ctx", DR_INIT))
    return dr_metrics(pred, x)
