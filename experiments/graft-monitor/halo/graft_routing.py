"""GRAFT gradient routing ported onto a Hugging Face Qwen3 model, in a form a stock vLLM server can serve.

Port of GRAFT's `rl/gradient_routing.py` (`DualMLPAdapter`, `_fused_decouple`, `set_scales`, `apply_dual_mlp`) and
`rl/advantages.py` (`routing_grad_mask_weights`, `compute_routed_advantages`, `_subset_group_renorm`,
`_baseline_nonflagged_var_all`, `_full_group_renorm`), https://github.com/jnward/GRAFT.

The one change of form: GRAFT's DualMLPAdapter adds two mini-SwiGLU networks beside each frozen MLP,

    out = down(silu(gate x) * up x) + s_R * down_R(silu(gate_R x) * up_R x) + s_F * down_F(silu(gate_F x) * up_F x)

which is exactly one SwiGLU MLP whose intermediate dimension is I + n_R + n_F, with gate/up rows and down columns
concatenated [base | RETAIN | FORGET]. Here the adapters live INSIDE the stock `gate_proj`/`up_proj`/`down_proj`
weights of a checkpoint widened to that intermediate size (`widen_state_dict`). So:

  - parameter names stay the hub's (`model.layers.N.mlp.gate_proj.weight`), and Halo's stock NCCL weight push sends
    base + RETAIN + FORGET to vLLM, which serves the widened checkpoint with no plugin;
  - the base slice [0:I] is read through `.detach()`, so it never gets a gradient (exactly zero), the frozen base
    of `apply_dual_mlp`;
  - RETAIN-only deployment drops the FORGET neurons (`retain_only_state_dict`), which is `set_scales(model, 1, 0)`
    baked into the weights.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------------------
# Per-row routing state. GRAFT `gradient_routing._FUSED_ROUTING` carries (1, T, 1) per-token tensors over a packed
# forward; Halo forwards padded rows, so here they are (B, 1, 1) per-row tensors (every token of a row shares its
# episode's role).
# ---------------------------------------------------------------------------------------------------------------
_ROUTING: dict = {"active": False}


_MASK_NAMES = ("forget_fwd_scale", "retain_grad_mask", "forget_grad_mask")


def set_routing(forget_fwd_scale: torch.Tensor, retain_grad_mask: torch.Tensor, forget_grad_mask: torch.Tensor):
    """GRAFT `set_fused_routing`. Each argument is a float tensor of shape [B] (one value per training row).

    The caller's forward may run the B rows as several chunks (TRL's `_get_per_token_logps_and_entropies` forwards
    `batch_size` rows at a time). The pre-hook `install_row_cursor` puts on the model hands each forward call the next
    rows of the armed masks; without the hook, a forward must cover all B rows."""
    _ROUTING["active"] = True
    _ROUTING["cursor"] = 0
    for name, m in zip(_MASK_NAMES, (forget_fwd_scale, retain_grad_mask, forget_grad_mask)):
        _ROUTING["full_" + name] = m.reshape(-1)
        _ROUTING[name] = m.reshape(-1, 1, 1)


def _advance_rows(n: int) -> None:
    """Window the armed masks onto the next n rows (one forward call)."""
    lo = _ROUTING["cursor"]
    full = _ROUTING["full_" + _MASK_NAMES[0]]
    if lo + n > full.numel():
        raise RuntimeError(f"routing armed for {full.numel()} rows, forward asks for rows {lo}:{lo + n}")
    for name in _MASK_NAMES:
        _ROUTING[name] = _ROUTING["full_" + name][lo : lo + n].view(-1, 1, 1)
    _ROUTING["cursor"] = lo + n


def install_row_cursor(model: nn.Module):
    """Forward pre-hook on the top model: each call while routing is armed consumes the next batch-size rows."""

    def hook(module, args, kwargs):
        if not _ROUTING["active"]:
            return None
        x = kwargs.get("input_ids")
        if x is None:
            x = kwargs.get("inputs_embeds")
        if x is None and args:
            x = args[0]
        _advance_rows(int(x.shape[0]))
        return None

    return model.register_forward_pre_hook(hook, with_kwargs=True)


def clear_routing():
    """GRAFT `clear_fused_routing`."""
    _ROUTING.clear()
    _ROUTING["active"] = False


@contextmanager
def routing(forget_fwd_scale, retain_grad_mask, forget_grad_mask):
    set_routing(forget_fwd_scale, retain_grad_mask, forget_grad_mask)
    try:
        yield
    finally:
        clear_routing()


def _fused_decouple(g_full: torch.Tensor, g_xdetached: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Verbatim GRAFT `gradient_routing._fused_decouple`: value g_full, parameter gradient mask * dg/dtheta, input
    gradient the full dg/dx (so lower layers still see the adapter's Jacobian)."""
    mc = mask.to(g_full.dtype)
    return g_full - (1.0 - mc) * g_xdetached + (1.0 - mc) * g_full.detach()


# ---------------------------------------------------------------------------------------------------------------
# The routed MLP: a Qwen3MLP (or any gate/up/down SwiGLU) whose class is swapped in place, so its parameters and
# their names are untouched.
# ---------------------------------------------------------------------------------------------------------------
class RoutedSwiGLU(nn.Module):
    """Mixin class swapped onto an existing SwiGLU MLP instance by `apply_routed_mlp`."""

    graft_base: int
    graft_retain: int
    graft_forget: int
    retain_scale: float = 1.0
    forget_scale: float = 1.0

    def _core(self, x: torch.Tensor, lo: int, hi: int, detach_weights: bool) -> torch.Tensor:
        wg = self.gate_proj.weight[lo:hi]
        wu = self.up_proj.weight[lo:hi]
        wd = self.down_proj.weight[:, lo:hi]
        if detach_weights:
            wg, wu, wd = wg.detach(), wu.detach(), wd.detach()
        act = getattr(self, "act_fn", F.silu)  # Qwen3MLP.act_fn is SiLU (hidden_act "silu")
        return F.linear(act(F.linear(x, wg)) * F.linear(x, wu), wd)

    def _retain_core(self, x):  # GRAFT DualMLPAdapter._retain_core
        b = self.graft_base
        return self._core(x, b, b + self.graft_retain, False)

    def _forget_core(self, x):  # GRAFT DualMLPAdapter._forget_core
        b = self.graft_base + self.graft_retain
        return self._core(x, b, b + self.graft_forget, False)

    def forward(self, x):
        # Frozen base MLP: weights detached, input gradient kept.
        base_out = self._core(x, 0, self.graft_base, True)
        if not _ROUTING["active"]:  # GRAFT DualMLPAdapter.forward, non-fused branch
            out = base_out
            if self.graft_retain:
                out = out + self._retain_core(x) * self.retain_scale
            if self.graft_forget:
                out = out + self._forget_core(x) * self.forget_scale
            return out
        # GRAFT DualMLPAdapter.forward, fused branch: per-row parameter-gradient gating plus per-row forward
        # FORGET scale (which supersedes the module's forget_scale while routing is active).
        xd = x.detach()
        out = base_out
        if self.graft_retain:
            out = (
                out
                + _fused_decouple(self._retain_core(x), self._retain_core(xd), _ROUTING["retain_grad_mask"])
                * self.retain_scale
            )
        if self.graft_forget:
            out = out + _fused_decouple(
                self._forget_core(x), self._forget_core(xd), _ROUTING["forget_grad_mask"]
            ) * _ROUTING["forget_fwd_scale"].to(out.dtype)
        return out


def find_mlps(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """GRAFT `find_mlp_modules`: every `*.<int>.mlp`."""
    out = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if parts[-1] == "mlp" and len(parts) >= 2 and parts[-2].isdigit():
            out.append((name, module))
    assert out, "no *.<int>.mlp modules found"
    return out


def apply_routed_mlp(model: nn.Module, base_intermediate: int, retain_neurons: int, forget_neurons: int) -> list[str]:
    """GRAFT `apply_dual_mlp` for a widened checkpoint: freeze every parameter except the MLP projections (whose base
    slice the forward detaches) and swap each MLP's class to the routed forward. Returns the MLP paths."""
    paths = []
    for p in model.parameters():
        p.requires_grad = False
    for name, mlp in find_mlps(model):
        width = mlp.gate_proj.weight.shape[0]
        if width != base_intermediate + retain_neurons + forget_neurons:
            raise ValueError(
                f"{name}: intermediate {width} != {base_intermediate}+{retain_neurons}+{forget_neurons}"
                " (load the widened checkpoint)"
            )
        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if proj.bias is not None:
                raise ValueError(f"{name}: biased MLP projections are not supported")
            proj.weight.requires_grad = True
        if not isinstance(mlp, RoutedSwiGLU):
            mlp.__class__ = type(f"Routed{type(mlp).__name__}", (RoutedSwiGLU, type(mlp)), {})
        mlp.graft_base, mlp.graft_retain, mlp.graft_forget = base_intermediate, retain_neurons, forget_neurons
        mlp.retain_scale, mlp.forget_scale = 1.0, 1.0
        paths.append(name)
    return paths


def set_scales(model: nn.Module, retain_scale: float = 1.0, forget_scale: float = 1.0):
    """GRAFT `set_scales`."""
    for m in model.modules():
        if isinstance(m, RoutedSwiGLU):
            m.retain_scale, m.forget_scale = retain_scale, forget_scale


def role_slices(model: nn.Module) -> dict[str, list[tuple[str, torch.Tensor, tuple]]]:
    """Role -> [(name, weight, index)] of the base / RETAIN / FORGET slices, for tests and diagnostics."""
    roles: dict[str, list] = {"base": [], "retain": [], "forget": []}
    for name, m in model.named_modules():
        if not isinstance(m, RoutedSwiGLU):
            continue
        b, r, f = m.graft_base, m.graft_retain, m.graft_forget
        for role, lo, hi in (("base", 0, b), ("retain", b, b + r), ("forget", b + r, b + r + f)):
            roles[role].append((f"{name}.gate_proj", m.gate_proj.weight, (slice(lo, hi), slice(None))))
            roles[role].append((f"{name}.up_proj", m.up_proj.weight, (slice(lo, hi), slice(None))))
            roles[role].append((f"{name}.down_proj", m.down_proj.weight, (slice(None), slice(lo, hi))))
    return roles


# ---------------------------------------------------------------------------------------------------------------
# Checkpoint surgery: widen a base checkpoint, and bake RETAIN-only for deployment.
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def widen_state_dict(
    state: dict[str, torch.Tensor], retain_neurons: int, forget_neurons: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Append RETAIN then FORGET neurons to every MLP. GRAFT's DualMLPAdapter init: gate/up kaiming_uniform(a=sqrt 5)
    (nn.Linear's default), down zeros, so the widened model computes exactly the base model's function at init."""
    gen = torch.Generator().manual_seed(seed)
    out = dict(state)
    n = retain_neurons + forget_neurons
    for key in sorted(state):
        w = state[key]
        if not key.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")):
            continue
        if key.endswith("down_proj.weight"):
            out[key] = torch.cat([w, torch.zeros(w.shape[0], n, dtype=w.dtype)], dim=1)
        else:
            hidden = w.shape[1]
            bound = 1.0 / math.sqrt(hidden)  # kaiming_uniform_(a=sqrt(5)) on fan_in = hidden
            extra = (torch.rand(n, hidden, generator=gen) * 2 - 1) * bound
            out[key] = torch.cat([w, extra.to(w.dtype)], dim=0)
    return out


@torch.no_grad()
def retain_only_state_dict(
    state: dict[str, torch.Tensor], base_intermediate: int, retain_neurons: int, forget_neurons: int
) -> dict[str, torch.Tensor]:
    """FORGET ablated (scale 0) and its neurons dropped: intermediate I + n_R, servable by stock vLLM."""
    keep = base_intermediate + retain_neurons
    out = dict(state)
    for key, w in state.items():
        if key.endswith("mlp.down_proj.weight"):
            assert w.shape[1] == keep + forget_neurons, (key, w.shape)
            out[key] = w[:, :keep].contiguous()
        elif key.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
            assert w.shape[0] == keep + forget_neurons, (key, w.shape)
            out[key] = w[:keep].contiguous()
    return out


# ---------------------------------------------------------------------------------------------------------------
# Routing masks and advantages (GRAFT rl/advantages.py), per episode.
# ---------------------------------------------------------------------------------------------------------------
_EPS = 1e-4  # GRAFT advantages._EPS


def routing_grad_mask_weights(
    routing_mode: str = "classic", lam: float = 1.0, kappa_r: float = 2.0, kappa_f: float = 2.0
) -> tuple[float, float, float, float]:
    """Verbatim GRAFT `advantages.routing_grad_mask_weights` -> (rgm_good, fgm_good, rgm_bad, fgm_bad)."""
    if routing_mode == "classic":
        return (1.0, 1.0, 1.0 - lam, 1.0 + lam * (kappa_f - 1.0))
    if routing_mode == "exclusive":
        return (1.0 + lam * (kappa_r - 1.0), 1.0 - lam, 1.0 - lam, 1.0 + lam * (kappa_f - 1.0))
    raise ValueError(f"routing_mode must be 'classic' or 'exclusive', got {routing_mode!r}")


def episode_masks(
    flagged: torch.Tensor,
    anchor: torch.Tensor,
    routing_mode: str = "classic",
    lam: float = 1.0,
    kappa_r: float = 2.0,
    kappa_f: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(forward FORGET scale, RETAIN grad weight, FORGET grad weight) per episode. GRAFT README "Method -> code":
    anchor (0, 1, 0), unflagged (1, 1, 1), flagged (1, 0, 2) at the classic lambda=1, kappa=2 defaults
    (`train.SampleGRPOTrainer._fused_forward_backward`: coherence is retain-only with the forget forward off)."""
    rg, fg, rb, fb = routing_grad_mask_weights(routing_mode, lam, kappa_r, kappa_f)
    flagged, anchor = flagged.bool(), anchor.bool()
    one = torch.ones(flagged.shape)
    fwd = torch.where(anchor, 0.0 * one, one)
    rgm = torch.where(anchor, one, torch.where(flagged, rb * one, rg * one))
    fgm = torch.where(anchor, 0.0 * one, torch.where(flagged, fb * one, fg * one))
    return fwd.float(), rgm.float(), fgm.float()


def _subset_group_renorm(rewards, subset, G):
    """Verbatim GRAFT `advantages._subset_group_renorm`."""
    grouped = rewards.view(-1, G)
    sub_g = subset.view(-1, G)
    out = torch.zeros_like(grouped)
    for i in range(grouped.shape[0]):
        m = sub_g[i]
        if m.sum() > 0:
            r = grouped[i][m]
            std = r.std() if r.numel() > 1 else r.new_zeros(())
            out[i][m] = (r - r.mean()) / (std + _EPS)
    return out.view(-1)


def _full_group_renorm(rewards, G):
    """Verbatim GRAFT `advantages._full_group_renorm`."""
    grouped = rewards.view(-1, G)
    return ((grouped - grouped.mean(dim=1, keepdim=True)) / (grouped.std(dim=1, keepdim=True) + _EPS)).view(-1)


def _baseline_nonflagged_var_all(rewards, nonflagged, G):
    """Verbatim GRAFT `advantages._baseline_nonflagged_var_all`."""
    grouped = rewards.view(-1, G)
    nf = nonflagged.view(-1, G)
    out = torch.zeros_like(grouped)
    for i in range(grouped.shape[0]):
        r, m = grouped[i], nf[i]
        baseline = r[m].mean() if bool(m.any()) else r.mean()
        std = r.std() if r.numel() > 1 else r.new_zeros(())
        out[i] = (r - baseline) / (std + _EPS)
    return out.view(-1)


def routed_advantages(
    rewards: torch.Tensor,
    flagged: torch.Tensor,
    anchor: torch.Tensor,
    G: int,
    renormalization_mode: str = "retain-only",
    coherence_rh_mode: str = "filter",
    coherence_rh_penalty: float = 3.0,
) -> torch.Tensor:
    """GRAFT `advantages.compute_routed_advantages`, gradient-routing branch at lambda=1 (the fast path), without
    the verifier options. Anchor (coherence) groups are whole groups. GRAFT's defaults (`experiment_config.py`):
    renormalization_mode 'retain-only', coherence_rh_mode 'filter', coherence_rh_penalty 3.0."""
    rewards = rewards.float()
    is_rh, coh = flagged.bool(), anchor.bool()
    assert bool((coh.view(-1, G).all(1) | ~coh.view(-1, G).any(1)).all()), "anchor must be whole groups"
    advantages = _full_group_renorm(rewards, G)  # stock per-group GRPO advantage (Bessel std, GRAFT's convention)

    def overwrite_coh_groups(candidate, mask):
        per_sample = mask.view(-1, G).all(dim=1).repeat_interleave(G)
        advantages[per_sample] = candidate[per_sample]

    rh_in_coh = is_rh & coh
    if rh_in_coh.any() and coherence_rh_mode != "none":
        if coherence_rh_mode in ("penalty", "zero"):
            rr = rewards.clone()
            if coherence_rh_mode == "penalty":
                rr[rh_in_coh] -= coherence_rh_penalty
            else:
                rr[rh_in_coh] = 0.0
            overwrite_coh_groups(_full_group_renorm(rr, G), coh)
        elif coherence_rh_mode == "filter":
            advantages[rh_in_coh] = 0.0
        elif coherence_rh_mode == "filter_renorm":
            overwrite_coh_groups(_subset_group_renorm(rewards, ~is_rh, G), coh)
        else:
            raise ValueError(f"Unknown coherence_rh_mode: {coherence_rh_mode!r}")
    if renormalization_mode == "balanced":
        a_v = _baseline_nonflagged_var_all(rewards, ~is_rh, G)
        sel = (~coh.view(-1, G).all(dim=1)).repeat_interleave(G)
        advantages[sel] = a_v[sel]
    elif renormalization_mode == "retain-only":
        retain = _subset_group_renorm(rewards, ~is_rh, G)
        good = (~is_rh) & (~coh)
        advantages[good] = retain[good]
    elif renormalization_mode != "off":
        raise ValueError(f"Unknown renormalization_mode: {renormalization_mode!r}")
    return advantages
