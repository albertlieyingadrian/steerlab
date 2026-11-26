"""GRAFT routing inside Halo's async environmental GRPO trainer, by subclassing (Halo itself is not edited).

`install()` runs in the torchrun process before Halo's stock entry script (run_halo.py):
  1. registers the `pushback` environment, and makes the trainer ship the class itself to the Ray actors
     (custom.md: a registration made only in the entry script fails inside the actors);
  2. replaces `DistributedAsyncEnvironmentalGRPOTrainer` in its module with `GraftEnvironmentalTrainer`, which the
     stock script then imports and builds.

The seams (Halo 1.2.0 / 1.3 `halo/trainers/grpo/environmental/`):
  - `__init__`: `apply_routed_mlp` on the loaded widened model (freezes all but the MLP projections, swaps the MLP
    forward), before TRL builds the optimizer.
  - `step_batch._step_advantages(rewards, rows, ...)`: the one place holding the round's rewards AND its
    `TrajectoryRowMap` (rollout results -> training rows, one row per assistant turn). Here each episode's flag and
    anchor role are read, GRAFT's routed advantages replace the stock group advantage, and the per-episode masks
    (forward FORGET scale, RETAIN grad weight, FORGET grad weight) are expanded to rows with `rows.to_rows`.
  - `_build_training_tensors`: adds the three row-aligned mask tensors to the batch dict, which TRL's shuffle and
    split carry with their rows (as Halo's own routing-mask key does).
  - `compute_loss`: pops them and arms `graft_routing.set_routing` for the micro-batch (train mode only);
    `training_step` clears it after the backward.

Settings (environment variables; Halo's YAML parser refuses unknown keys):
  GRAFT_ROUTING=1               turn routing on (0: plain Halo GRPO on the widened model, all adapters trained)
  GRAFT_BASE_INTERMEDIATE=3072  GRAFT_RETAIN_NEURONS=32  GRAFT_FORGET_NEURONS=32  (GRAFT experiment_config defaults)
  GRAFT_FLAG_KEY=judge/monitor/capitulation   reward metric read when the environment sets no graft_flag
  GRAFT_RENORM=retain-only  GRAFT_COH_MODE=filter  GRAFT_COH_PENALTY=3.0  GRAFT_ROUTING_MODE=classic
"""

from __future__ import annotations

import logging
import os

import graft_routing as gr
import torch

logger = logging.getLogger("graft")


def _cfg():
    e = os.environ.get
    return {
        "on": e("GRAFT_ROUTING", "1") == "1",
        "base": int(e("GRAFT_BASE_INTERMEDIATE", "3072")),
        "nr": int(e("GRAFT_RETAIN_NEURONS", "32")),
        "nf": int(e("GRAFT_FORGET_NEURONS", "32")),
        "flag_key": e("GRAFT_FLAG_KEY", "judge/monitor/capitulation"),
        "renorm": e("GRAFT_RENORM", "retain-only"),
        "coh_mode": e("GRAFT_COH_MODE", "filter"),
        "coh_penalty": float(e("GRAFT_COH_PENALTY", "3.0")),
        "routing_mode": e("GRAFT_ROUTING_MODE", "classic"),
    }


MASK_KEYS = ("graft_forget_fwd", "graft_retain_gm", "graft_forget_gm")


def episode_flag(result, flag_key: str) -> tuple[bool, bool]:
    """(flag, found). The environment's own flag wins; else the judge term's per-episode verdict."""
    traj = getattr(result, "trajectory", None)
    if traj is None:
        return False, False
    info = traj.info or {}
    if "graft_flag" in info:
        return bool(info["graft_flag"]), True
    for source in (info.get("reward_metrics") or {}, getattr(result, "metrics", None) or {}):
        if flag_key in source:
            return float(source[flag_key]) > 0.5, True
    return False, False


def episode_anchor(result) -> bool:
    traj = getattr(result, "trajectory", None)
    return bool(traj is not None and (traj.info or {}).get("graft_anchor"))


def install():
    from halo.environments import registry
    from halo.trainers.grpo.environmental import trainer as trainer_mod
    from pushback_env import PushbackEnvironment

    if "pushback" not in registry.get_registered_environments():
        registry.register_environment("pushback", lambda c: PushbackEnvironment(**c))

    stock_resolve = trainer_mod._resolve_environment_spec

    def resolve(environment_config, environment_cls, environment_kwargs):
        if environment_config is not None and environment_config.environment_type == "pushback":
            # Ship the class, not the name: the actors' registry does not know `pushback`.
            return (PushbackEnvironment, environment_config.to_env_config()), {}
        return stock_resolve(environment_config, environment_cls, environment_kwargs)

    trainer_mod._resolve_environment_spec = resolve
    Base = trainer_mod.DistributedAsyncEnvironmentalGRPOTrainer
    if getattr(Base, "_graft", False):
        return

    class GraftEnvironmentalTrainer(Base):
        _graft = True

        def __init__(self, *args, **kwargs):
            self._graft_cfg = cfg = _cfg()
            model = kwargs.get("model", args[0] if args else None)
            paths = gr.apply_routed_mlp(model, cfg["base"], cfg["nr"], cfg["nf"])
            gr.install_row_cursor(model)  # TRL forwards a micro-batch in batch_size chunks
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.warning(
                f"GRAFT: routed {len(paths)} MLPs (I={cfg['base']}, R={cfg['nr']}, F={cfg['nf']}); "
                f"{n_train / 1e6:.1f}M params require grad (base slices detached); routing={cfg['on']}"
            )
            self._graft_masks = None
            self._graft_stats = {}
            super().__init__(*args, **kwargs)
            if self.accelerator.num_processes != 1:
                raise NotImplementedError("the GRAFT port assumes one trainer rank (group stats are local)")

        # --- advantages and masks, per episode ---
        def _step_advantages(self, rewards, rows, num_padding, num_generations, *rest):
            episode_state, traj_adv, local_adv = super()._step_advantages(
                rewards, rows, num_padding, num_generations, *rest
            )
            cfg = self._graft_cfg
            n_traj = len(rows.turns_per_traj)
            results = list(rows.rollout_results)
            flags, found = zip(*[episode_flag(r, cfg["flag_key"]) for r in results]) if results else ((), ())
            flagged = torch.zeros(n_traj, dtype=torch.bool)
            anchor = torch.zeros(n_traj, dtype=torch.bool)
            flagged[: len(results)] = torch.tensor(flags, dtype=torch.bool)
            anchor[: len(results)] = torch.tensor([episode_anchor(r) for r in results], dtype=torch.bool)
            training = self.model.training
            self._graft_stats = {
                "graft/flag_rate": float(flagged[: len(results)].float().mean()) if results else 0.0,
                "graft/flag_found": float(sum(found) / max(len(found), 1)),
                "graft/anchor_rate": float(anchor[: len(results)].float().mean()) if results else 0.0,
            }
            if not (cfg["on"] and training):
                self._graft_masks = None
                return episode_state, traj_adv, local_adv
            # GRAFT advantages over whole groups of valid episodes; invalid episodes keep 0 and leave the stats
            # (their reward is replaced by their group's valid mean before the renorm).
            G = num_generations
            valid = (episode_state == 1).cpu()
            r = rewards.detach().float().cpu().clone()
            assert n_traj % G == 0, f"{n_traj} trajectories do not form whole groups of {G}"
            rg, vg = r.view(-1, G), valid.view(-1, G)
            for i in range(rg.shape[0]):
                if vg[i].any() and not vg[i].all():
                    rg[i][~vg[i]] = rg[i][vg[i]].mean()
            adv = gr.routed_advantages(r, flagged, anchor, G, cfg["renorm"], cfg["coh_mode"], cfg["coh_penalty"])
            adv = torch.where(valid, adv, torch.zeros_like(adv)).to(traj_adv.device, traj_adv.dtype)
            fwd, rgm, fgm = gr.episode_masks(flagged, anchor, cfg["routing_mode"])
            dev = traj_adv.device
            self._graft_masks = tuple(rows.to_rows(m.to(dev), dummy_fill=1.0) for m in (fwd, rgm, fgm))
            return episode_state, adv, rows.to_rows(adv)

        def _build_training_tensors(self, *args, **kwargs):
            self._graft_masks = None
            result = super()._build_training_tensors(*args, **kwargs)
            if self._graft_masks is not None:
                n = result["completion_ids"].shape[0]
                for key, m in zip(MASK_KEYS, self._graft_masks):
                    assert m.shape[0] == n, (key, m.shape, n)
                    result[key] = m.float()
                self._graft_masks = None
            return result

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if isinstance(inputs, dict) and MASK_KEYS[0] in inputs:
                masks = [inputs.pop(k) for k in MASK_KEYS]
                if model.training:
                    # The chunk cursor reads the routing at forward time; a recompute in the backward would read the
                    # last chunk's rows, so gradient checkpointing must be off.
                    if getattr(self.accelerator.unwrap_model(model), "is_gradient_checkpointing", False):
                        raise RuntimeError("GRAFT routing port needs gradient_checkpointing: false")
                    gr.set_routing(*masks)
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        def training_step(self, model, inputs, num_items_in_batch=None):
            try:
                return super().training_step(model, inputs, num_items_in_batch)
            finally:
                gr.clear_routing()  # after the backward, as Halo disarms its own routing injector

        def log(self, logs, start_time=None):
            logs.update(self._graft_stats)
            super().log(logs, start_time)

    GraftEnvironmentalTrainer.__name__ = Base.__name__
    GraftEnvironmentalTrainer.__qualname__ = Base.__qualname__
    trainer_mod.DistributedAsyncEnvironmentalGRPOTrainer = GraftEnvironmentalTrainer
