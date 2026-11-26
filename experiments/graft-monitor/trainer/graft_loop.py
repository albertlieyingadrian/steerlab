"""Multi-turn GRPO with GRAFT gradient routing on the `pushback` environment.

Built on GRAFT's own routing code (vendored in `graft_vendor/`, MIT, commit 6da923d): `DualMLPAdapter` /
`apply_dual_mlp` / `set_scales` / `fused_routing` (the `_fused_decouple` stop-gradient gate),
`PreRoutingGradAccumulator`, `compute_routed_advantages` (balanced renorm, `_baseline_nonflagged_var_all`),
`routing_grad_mask_weights` + `adapter_kappas`, and `SplitMomentAdamW` with the `graft_w_max` clamp.

One step:
  1. Sample `n_prompts` routing prompts and `anchor_prompts` anchor prompts from the train split, `group` episodes
     each. Routing rows roll out with both adapters (scales 1, 1); anchor rows with RETAIN only (1, 0), as in
     GRAFT's toy anchor. Two generation calls per episode: turn 1, then turn 3 after the scripted pushback.
  2. Grade with `env.grade` (proxy_reward is the only reward; gt_hack is logged, and is a flag only under
     flag_source=gt). Flags come from `flag_fn(episodes, transcripts)`, then recall thinning / random flags.
  3. Advantages from GRAFT's `compute_routed_advantages` (groups of `group`; anchor groups get the -2 detected-hack
     penalty and a full-group renorm). Masks per episode (forward FORGET scale, RETAIN grad, FORGET grad):
     anchor (0, 1, 0), unflagged (1, 1, 1), flagged (1, 0, 2).
  4. Loss over BOTH assistant turns (trajectory-level routing: the episode's mask applies to every token it
     generated). Each turn is its own right-padded row with exactly the context it was sampled from (Qwen3's
     template re-renders earlier turns without the empty think block, so one concatenated row would be off-policy
     for turn 1). Per-episode token mean, averaged over episodes (GRAFT's grpo per-sequence normalisation), one
     optimizer step per rollout, ratio 1 (no old log-probs, as in GRAFT's toy runs), k3 KL to the base model
     (adapters disabled) with coefficient `beta`.

Arms: none (no monitor, both adapters, plain GRPO), penalty (flag zeroes the reward), filter (flagged episodes
dropped from the loss; survivors renormed), graft, graft_noroute (graft with every flag off: anchor kept, nothing
routed, no anchor penalty). The baselines spend the anchor budget on extra routing prompts so every arm samples the
same number of episodes per step.

    python graft_loop.py --set name=smoke arm=graft steps=2 ...     # see Config for every key
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
GM = HERE.parent
sys.path.insert(0, str(HERE / "graft_vendor"))
sys.path.insert(0, str(GM))
sys.path.insert(0, str(HERE))

import flags as flagmod  # noqa: E402
from advantages import (  # noqa: E402
    AdvConfig,
    _full_group_renorm,
    adapter_kappas,
    assert_kappa_geometry,
    compute_routed_advantages,
    routing_grad_mask_weights,
)
from gradient_routing import (  # noqa: E402
    DualMLPAdapter,
    PreRoutingGradAccumulator,
    apply_dual_mlp,
    disabled_dual_adapters,
    fused_routing,
    set_scales,
)
from split_moment import SplitMomentAdamW, clip_pre_routing_grads_  # noqa: E402

import env  # noqa: E402

ARMS = ("none", "penalty", "filter", "graft", "graft_noroute")
MODES = {"both": (1.0, 1.0), "retain_only": (1.0, 0.0), "forget_only": (0.0, 1.0)}


@dataclass
class Config:
    name: str = "dev"
    out_dir: str = ""  # default: results/runs/<name> locally, /vol/runs/<name> on Modal
    arm: str = "graft"
    flag_source: str = "gt"  # gt | rule | http | self
    recall: float = 1.0  # keep each raw flag with this probability (GRAFT rh_detector_recall)
    random_flags: float = -1.0  # >= 0: flags are Bernoulli(rate), ignoring the source
    # monitor (http / self)
    policy_path: str = ""
    monitor_reference: bool = True
    monitor_backend: str = "oss_safeguard"
    monitor_kwargs: dict = field(default_factory=dict)
    monitor_url: str = ""
    monitor_model: str = "openai/gpt-oss-safeguard-20b"
    monitor_workers: int = 32
    monitor_unparsed_as_flag: bool = False  # unparsed verdict (flag None): counted, not flagged by default
    self_variant: str = "uninformed"  # uninformed | informed
    self_mode: str = "both"  # adapter scales the self-judge runs with
    self_threshold: float = 0.5
    self_target_rate: float = 0.0  # > 0: flag the top fraction of each batch by P(YES) instead of a threshold
    # model
    model: str = "Qwen/Qwen3-0.6B"
    init_from: str = ""  # a previous run's deployed/ dir (self-improvement generations)
    city_filter: str = ""  # "shard:k/n" -> train cities whose generation hash % n == k
    no_hackable: bool = False  # Q1 same-model control: every training episode unhackable (eval unchanged)
    retain_neurons: int = 32  # m32: ~1% of Qwen3-0.6B's 3072 MLP width, like m16 on SmolLM2's 1536
    forget_neurons: int = 32
    # GRPO
    steps: int = 200
    n_prompts: int = 12  # routing prompts per step (multiple of 4: balanced 2x2 cells)
    anchor_prompts: int = 4  # anchor prompts per step (graft arms); baselines use them as routing
    group: int = 8
    lr: float = 3e-4
    lr_schedule: str = "linear"  # linear decay to 0 (GRAFT's default) | constant
    beta: float = 0.05
    max_grad_norm: float = 1.0
    anchor_penalty: float = 2.0
    graft_w_max: float = 4.0
    temperature: float = 1.0
    max_new_tokens: int = 160
    seed: int = 1
    # eval
    eval_every: int = 25
    eval_n: int = 256
    eval_seed: int = 99
    eval_split: str = "eval"  # eval | heldout (train is allowed for tuning reads)
    eval_at_start: bool = True
    # throughput
    gen_chunk: int = 256
    train_tokens_per_mb: int = 6144
    # money / time guards
    usd_per_hour: float = 0.0
    max_usd: float = 0.0  # > 0: stop (and save) before the run's own estimate passes this
    max_minutes: float = 0.0

    def deployed_mode(self) -> str:
        return "retain_only" if self.arm.startswith("graft") else "both"


# ---------------------------------------------------------------------------------------------- model helpers


def device_and_amp():
    if torch.cuda.is_available():
        return torch.device("cuda"), (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
    return torch.device("cpu"), nullcontext


def adapters(model) -> list[DualMLPAdapter]:
    return [m for m in model.modules() if isinstance(m, DualMLPAdapter)]


def install_adapters(model, n_r: int, n_f: int):
    """GRAFT's apply_dual_mlp (freezes the base), then keep adapter weights in float32 (master copy; the GPU
    forward runs under bf16 autocast)."""
    apply_dual_mlp(model, n_r, n_f)
    for m in adapters(model):
        for name in ("gate_retain", "up_retain", "down_retain", "gate_forget", "up_forget", "down_forget"):
            sub = getattr(m, name)
            if sub is not None:
                sub.float()
    # Make every adapter's input require grad, so each full-backward hook (split-moment v capture) fires.
    model.enable_input_require_grads()
    return model


def role_params(model):
    r, f = [], []
    for m in adapters(model):
        r += m.get_retain_params()
        f += m.get_forget_params()
    return r, f


@contextmanager
def row_scales(model, retain, forget):
    """Per-row adapter scales for a batched forward / generate: each DualMLPAdapter multiplies its retain / forget
    output by a (B, 1, 1) tensor instead of a scalar (same code path as set_scales)."""
    ads = adapters(model)
    dt = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    r = torch.as_tensor(retain, dtype=dt, device=dev).view(-1, 1, 1)
    f = torch.as_tensor(forget, dtype=dt, device=dev).view(-1, 1, 1)
    for m in ads:
        m.retain_scale, m.forget_scale = r, f
    try:
        yield
    finally:
        set_scales(model, 1.0, 1.0)


def load_policy_model(cfg: Config, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = cfg.init_from or cfg.model
    tok = AutoTokenizer.from_pretrained(src)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).to(device)
    model.config.use_cache = True
    install_adapters(model, cfg.retain_neurons, cfg.forget_neurons)
    return tok, model


# ---------------------------------------------------------------------------------------------- generation


def chat_ids(tok, messages) -> list[int]:
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)


def _as_list(ids):
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def stop_ids(tok) -> list[int]:
    ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    return sorted(i for i in ids if i is not None and i >= 0)


def _left_pad(prompts, pad):
    L = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), L), pad, dtype=torch.long)
    att = torch.zeros((len(prompts), L), dtype=torch.long)
    for j, p in enumerate(prompts):
        ids[j, L - len(p) :] = torch.tensor(p)
        att[j, L - len(p) :] = 1
    return ids, att, L


@torch.no_grad()
def generate(model, tok, convs, retain, forget, cfg: Config, amp, max_new_tokens=None, temperature=None):
    """Sample one reply per conversation, each row at its own adapter scales. Left-padded (generation only).
    Returns (prompt_ids, completion_ids incl. the stop token when emitted, stripped texts)."""
    device = next(model.parameters()).device
    max_new = max_new_tokens or cfg.max_new_tokens
    temp = cfg.temperature if temperature is None else temperature
    prompts = [_as_list(chat_ids(tok, c)) for c in convs]
    stops = set(stop_ids(tok))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    comps: list = [None] * len(convs)
    order = sorted(range(len(convs)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), cfg.gen_chunk):
        idx = order[s : s + cfg.gen_chunk]
        ids, att, L = _left_pad([prompts[i] for i in idx], pad)
        with row_scales(model, [retain[i] for i in idx], [forget[i] for i in idx]), amp():
            out = model.generate(
                input_ids=ids.to(device),
                attention_mask=att.to(device),
                do_sample=True,
                temperature=temp,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                max_new_tokens=max_new,
                pad_token_id=pad,
                eos_token_id=sorted(stops),
            )
        gen = out[:, L:].tolist()
        for j, i in enumerate(idx):
            g = gen[j]
            cut = next((k for k, t in enumerate(g) if t in stops), None)
            comps[i] = g[: cut + 1] if cut is not None else g
        del out
    texts = [tok.decode([t for t in c if t not in stops], skip_special_tokens=True).strip() for c in comps]
    return prompts, comps, texts


@torch.no_grad()
def yes_prob(model, tok, convs, retain, forget, amp, chunk=32) -> list[float]:
    """P(YES) / (P(YES) + P(NO)) for the first reply token (self-monitor). Left-padded, batched."""
    device = next(model.parameters()).device
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("YES", "Yes", " YES", " Yes")})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("NO", "No", " NO", " No")})
    prompts = [_as_list(chat_ids(tok, c)) for c in convs]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = []
    for s in range(0, len(prompts), chunk):
        ids, att, _ = _left_pad(prompts[s : s + chunk], pad)
        with row_scales(model, retain[s : s + chunk], forget[s : s + chunk]), amp():
            logits = model(input_ids=ids.to(device), attention_mask=att.to(device), logits_to_keep=1).logits
        lp = logits[:, -1].float().log_softmax(-1)
        out += torch.sigmoid(torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1)).tolist()
    return out


# ---------------------------------------------------------------------------------------------- rollouts


def city_filter_fn(spec: str):
    if not spec:
        return None
    kind, arg = spec.split(":")
    assert kind == "shard", spec
    k, n = (int(x) for x in arg.split("/"))
    return lambda city: int(hashlib.sha256(("gen|" + city).encode()).hexdigest()[:8], 16) % n == k


def rollout(model, tok, eps, retain, forget, cfg, amp):
    """Two-call episode rollout. Returns per-episode dicts with token segments, texts, transcript and grade."""
    p1, c1, t1 = generate(model, tok, [e.turn1() for e in eps], retain, forget, cfg, amp)
    p2, c2, t2 = generate(model, tok, [e.turn2(a) for e, a in zip(eps, t1)], retain, forget, cfg, amp)
    out = []
    for i, e in enumerate(eps):
        out.append(
            {
                "ep": e,
                "segs": [(p1[i], c1[i]), (p2[i], c2[i])],
                "first": t1[i],
                "final": t2[i],
                "transcript": e.transcript(t1[i], t2[i]),
                "grade": env.grade(e, t1[i], t2[i]),
                "truncated": [len(c1[i]) >= cfg.max_new_tokens, len(c2[i]) >= cfg.max_new_tokens],
            }
        )
    return out


def apply_flag_noise(raw: list[bool], cfg: Config, rng: random.Random) -> list[bool]:
    if cfg.arm == "graft_noroute":
        return [False] * len(raw)
    if cfg.random_flags >= 0:
        return [rng.random() < cfg.random_flags for _ in raw]
    return [bool(f) and rng.random() < cfg.recall for f in raw]


# ---------------------------------------------------------------------------------------------- advantages, masks


def routing_tables(cfg: Config, rewards, is_rh, is_anchor, G):
    """GRAFT's compute_routed_advantages + per-episode masks (ffs, rgm, fgm) and a keep mask."""
    graft = cfg.arm.startswith("graft")
    raw = torch.tensor(rewards, dtype=torch.float32)
    rh = torch.tensor(is_rh, dtype=torch.bool)
    coh = torch.tensor(is_anchor, dtype=torch.bool)
    acfg = AdvConfig(
        num_generations=G,
        gradient_routing_enabled=graft,
        interlaced_coh=bool(coh.any()),
        coherence_rh_mode="penalty",
        coherence_rh_penalty=cfg.anchor_penalty,
        reward_penalty_baseline=cfg.arm == "penalty",
        reward_penalty_amount=None,
        verified_only_training=False,
        filter_baseline=cfg.arm == "filter",
        renormalization_mode="balanced",
        rh_detector_verifies_retain_samples=False,
        coh_samples_per_rollout=int(coh.sum()),
        graft_w_max=cfg.graft_w_max,
    )
    res = compute_routed_advantages(
        raw_rewards=raw,
        base_advantages=_full_group_renorm(raw, G),
        is_rh=rh,
        is_coherence=coh,
        is_verified_retain=None,
        penalty_baseline_raw_rewards=raw if cfg.arm == "penalty" else None,
        cfg=acfg,
    )
    adv = res.advantages
    n = len(rewards)
    ffs, rgm, fgm = torch.ones(n), torch.ones(n), torch.ones(n)
    keep = ~res.should_filter
    if graft:
        kr, kf = adapter_kappas(cfg.retain_neurons, cfg.forget_neurons)
        assert_kappa_geometry("classic", 1.0, kr, kf, cfg.graft_w_max)
        rg, fg, rb, fb = routing_grad_mask_weights("classic", 1.0, kr, kf)
        for i in range(n):
            if is_anchor[i]:
                ffs[i], rgm[i], fgm[i] = 0.0, 1.0, 0.0
            elif is_rh[i]:
                rgm[i], fgm[i] = rb, fb
            else:
                rgm[i], fgm[i] = rg, fg
    elif cfg.arm == "filter":
        keep = keep & ~rh  # flagged episodes leave the loss (and the KL term) entirely
    return adv, ffs, rgm, fgm, keep


# ---------------------------------------------------------------------------------------------- loss


def build_rows(episodes_segs):
    """episodes_segs: per episode a list of (prompt_ids, completion_ids). Returns rows (episode index, ids, n_prompt)
    and per-episode completion token counts."""
    rows, ntok = [], []
    for e, segs in enumerate(episodes_segs):
        ntok.append(max(sum(len(c) for _, c in segs), 1))
        for p, c in segs:
            if len(c):
                rows.append((e, list(p) + list(c), len(p)))
    return rows, ntok


def pack(rows, budget):
    """Length-sorted microbatches whose padded size (rows x max len) stays under `budget` tokens."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][1]))
    mbs, cur, mx = [], [], 0
    for i in order:
        L = len(rows[i][1])
        if cur and max(mx, L) * (len(cur) + 1) > budget:
            mbs.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, L)
    if cur:
        mbs.append(cur)
    return mbs


def token_logps(model, ids, att, n_prompt, amp):
    """Log-probs of the completion tokens only (lm_head applied to those positions; no full-vocab copy of the
    whole batch). ids/att right-padded [B, T]. Returns (logp [Ntok], row index [Ntok])."""
    with amp():
        h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device).unsqueeze(0)
    tgt_mask = (pos >= n_prompt.unsqueeze(1)) & att.bool()  # completion tokens
    tgt_mask[:, 0] = False
    pred_mask = torch.zeros_like(tgt_mask)
    pred_mask[:, :-1] = tgt_mask[:, 1:]  # positions that predict them
    hs = h[pred_mask]
    with amp():
        logits = model.lm_head(hs)
    logp = logits.float().log_softmax(-1).gather(-1, ids[tgt_mask].unsqueeze(-1)).squeeze(-1)
    row = torch.arange(B, device=ids.device).unsqueeze(1).expand(B, T)[tgt_mask]
    return logp, row


def routed_backward(model, rows, ntok, adv, ffs, rgm, fgm, keep, beta, amp, budget, capture=None):
    """Forward + backward of the GRPO loss with GRAFT's fused per-row routing masks. Returns stats.
    Loss = mean over kept episodes of [token-mean over the episode's completion tokens of
           (-A * logp + beta * k3_KL(ref || pi))]; the ratio is 1 (single optimizer step per rollout)."""
    device = next(model.parameters()).device
    kept_rows = [r for r in rows if bool(keep[r[0]])]
    n_kept = int(keep.sum())
    if n_kept == 0 or not kept_rows:
        return {"loss": 0.0, "kl": 0.0, "n_tok": 0}
    ntok_t = torch.tensor(ntok, dtype=torch.float32)
    tot_loss, tot_kl, tot_tok = 0.0, 0.0, 0
    for mb in pack(kept_rows, budget):
        rs = [kept_rows[i] for i in mb]
        T = max(len(r[1]) for r in rs)
        ids = torch.zeros((len(rs), T), dtype=torch.long)
        att = torch.zeros((len(rs), T), dtype=torch.long)
        for j, (_, seq, _) in enumerate(rs):  # RIGHT padding for training
            ids[j, : len(seq)] = torch.tensor(seq)
            att[j, : len(seq)] = 1
        ids, att = ids.to(device), att.to(device)
        npr = torch.tensor([r[2] for r in rs], device=device)
        ep = torch.tensor([r[0] for r in rs])
        with torch.no_grad(), disabled_dual_adapters(model):
            ref_lp, _ = token_logps(model, ids, att, npr, amp)

        def v(t):
            return t[ep].to(device).view(-1, 1, 1)

        with fused_routing(v(ffs), v(rgm), v(fgm)):
            lp, row = token_logps(model, ids, att, npr, amp)
            a = adv[ep].to(device)[row]
            w = (1.0 / ntok_t[ep]).to(device)[row] / n_kept
            d = ref_lp - lp
            kl = torch.exp(d) - d - 1.0
            loss = ((-a * lp + beta * kl) * w).sum()
            loss.backward()
        if capture is not None:
            with amp():
                capture.flush(v(ffs))
        tot_loss += float(loss.detach())
        tot_kl += float(kl.detach().sum())
        tot_tok += int(lp.numel())
        del lp, ref_lp, loss
    return {"loss": tot_loss, "kl": tot_kl / max(tot_tok, 1), "n_tok": tot_tok}


def grad_norm(params, attr="grad"):
    sq = 0.0
    for p in params:
        g = getattr(p, attr, None)
        if g is not None:
            sq += float(g.float().pow(2).sum())
    return math.sqrt(sq)


# ---------------------------------------------------------------------------------------------- save / merge


def merged_state_dict(model, mode: str):
    """Fold the adapters' SwiGLU neurons into each base MLP (concatenate along the intermediate dim): exact, since
    Qwen3's MLP is down(silu(gate x) * up x). mode: retain_only (deployed GRAFT policy) or both."""
    sides = ["retain", "forget"] if mode == "both" else ["retain"]
    sd = {}
    for k, t in model.state_dict().items():
        if "_retain." in k or "_forget." in k:
            continue
        sd[k.replace(".mlp.base_mlp.", ".mlp.")] = t
    n_extra = 0
    for name, m in model.named_modules():
        if not isinstance(m, DualMLPAdapter):
            continue
        dt = m.base_mlp.gate_proj.weight.dtype
        g, u, d = [m.base_mlp.gate_proj.weight], [m.base_mlp.up_proj.weight], [m.base_mlp.down_proj.weight]
        for side in sides:
            if getattr(m, f"gate_{side}") is None:
                continue
            g.append(getattr(m, f"gate_{side}").weight)
            u.append(getattr(m, f"up_{side}").weight)
            d.append(getattr(m, f"down_{side}").weight)
        sd[f"{name}.gate_proj.weight"] = torch.cat([p.to(dt) for p in g], 0).detach().clone()
        sd[f"{name}.up_proj.weight"] = torch.cat([p.to(dt) for p in u], 0).detach().clone()
        sd[f"{name}.down_proj.weight"] = torch.cat([p.to(dt) for p in d], 1).detach().clone()
        n_extra = sd[f"{name}.gate_proj.weight"].shape[0] - m.base_mlp.gate_proj.weight.shape[0]
    return sd, n_extra


def build_merged(model, mode: str):
    from transformers import AutoModelForCausalLM

    sd, n_extra = merged_state_dict(model, mode)
    cfg = type(model.config).from_dict(model.config.to_dict())
    cfg.intermediate_size = model.config.intermediate_size + n_extra
    new = AutoModelForCausalLM.from_config(cfg, dtype=next(model.base_model.parameters()).dtype)
    missing, unexpected = new.load_state_dict({k: v.cpu() for k, v in sd.items()}, strict=False)
    assert not unexpected, unexpected[:5]
    assert all("lm_head" in k for k in missing), missing[:5]  # tied to the embedding
    if getattr(cfg, "tie_word_embeddings", False):
        new.tie_weights()
    return new.eval()


def save_merged(model, tok, mode: str, path: Path):
    new = build_merged(model, mode)
    path.mkdir(parents=True, exist_ok=True)
    new.save_pretrained(path)
    tok.save_pretrained(path)
    del new


def save_adapters(model, path: Path, cfg: Config, step: int, opt=None):
    path.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "_retain." in k or "_forget." in k}
    torch.save(sd, path / "adapters.pt")
    if opt is not None:
        torch.save(opt.state_dict(), path / "optimizer.pt")
    (path / "meta.json").write_text(json.dumps({"step": step, "config": asdict(cfg)}, indent=1))


# ---------------------------------------------------------------------------------------------- eval


def evaluate(model, tok, cfg, amp, step, out: Path, eval_eps):
    rows_eps, rs, fs, modes = [], [], [], []
    for mode, (r, f) in MODES.items():
        for e in eval_eps:
            rows_eps.append(e)
            rs.append(r)
            fs.append(f)
            modes.append(mode)
    torch.manual_seed(cfg.eval_seed * 1000 + step)
    t0 = time.time()
    res = rollout(model, tok, rows_eps, rs, fs, cfg, amp)
    recs = []
    for mode in MODES:
        sel = [
            dict(**x["ep"].to_dict(), first=x["first"], final_text=x["final"], **x["grade"])
            for x, m in zip(res, modes)
            if m == mode
        ]
        s = env.summarize(sel)
        recs.append(
            {
                "step": step,
                "mode": mode,
                "deployed": mode == cfg.deployed_mode(),
                **s,
                "rule_flag_rate": sum(env.rule_monitor(r["final_text"]) for r in sel) / len(sel),
                "truncated": sum(any(x["truncated"]) for x, m in zip(res, modes) if m == mode) / len(sel),
            }
        )
    with open(out / "routing_eval.jsonl", "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    (out / "evals").mkdir(exist_ok=True)
    with open(out / "evals" / f"step_{step:04d}.jsonl", "w") as f:
        for x, m in zip(res, modes):
            f.write(
                json.dumps(
                    {"mode": m, **x["ep"].to_dict(), "first": x["first"], "final_text": x["final"], **x["grade"]}
                )
                + "\n"
            )
    return recs, time.time() - t0


# ---------------------------------------------------------------------------------------------- main loop


def _mean(xs):
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else float("nan")


def step_record(step, res, rewards, raw, fl, is_anchor, adv, keep, st):
    grades = [x["grade"] for x in res]
    R = [i for i in range(len(res)) if not is_anchor[i]]
    A = [i for i in range(len(res)) if is_anchor[i]]
    hw = lambda idx: [i for i in idx if res[i]["ep"].hackable and not res[i]["ep"].assertion_true]  # noqa: E731
    gt = [g["gt_hack"] for g in grades]
    n_gt = sum(gt[i] for i in R)
    return {
        "step": step,
        "reward": _mean(rewards[i] for i in R),
        "reward_anchor": _mean(rewards[i] for i in A),
        "flag_rate": _mean(fl[i] for i in R),
        "flag_rate_anchor": _mean(fl[i] for i in A),
        "raw_flag_rate": _mean(raw[i] for i in R),
        "gt_hack_rate": _mean(gt[i] for i in hw(R)),
        "gt_hack_frac": _mean(gt[i] for i in R),
        "gt_hack_rate_anchor": _mean(gt[i] for i in hw(A)),
        "flag_recall": (sum(bool(fl[i] and gt[i]) for i in R) / n_gt) if n_gt else float("nan"),
        "flag_fpr": _mean(fl[i] for i in R if not gt[i]),
        "skill": _mean(grades[i]["correct"] for i in R if (not res[i]["ep"].hackable) or res[i]["ep"].assertion_true),
        "capitulation_unhackable": _mean(
            grades[i]["capitulated"] for i in R if not res[i]["ep"].hackable and not res[i]["ep"].assertion_true
        ),
        "parsed": _mean(grades[i]["parsed"] for i in R),
        "first_correct": _mean(grades[i]["first_correct"] for i in R),
        "truncated": _mean(any(x["truncated"]) for x in res),
        "len_turn1": _mean(len(x["segs"][0][1]) for x in res),
        "len_turn2": _mean(len(x["segs"][1][1]) for x in res),
        "adv_mean": float(adv.mean()),
        "adv_std": float(adv.std()),
        "adv_absmean": float(adv.abs().mean()),
        "adv_frac_zero": float((adv == 0).float().mean()),
        "n_kept": int(keep.sum()),
        "loss": st["loss"],
        "kl": st["kl"],
    }


def train(cfg: Config, on_checkpoint=None):
    assert cfg.arm in ARMS, cfg.arm
    graft = cfg.arm.startswith("graft")
    out = Path(cfg.out_dir or (GM / "results" / "runs" / cfg.name))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    for fname in ("log.jsonl", "routing_eval.jsonl", "done.json"):
        (out / fname).unlink(missing_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device, amp = device_and_amp()
    tok, model = load_policy_model(cfg, device)
    retain_p, forget_p = role_params(model)
    opt = SplitMomentAdamW(
        [{"params": retain_p, "graft_role": "retain"}, {"params": forget_p, "graft_role": "forget"}],
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda s: max(0.0, 1.0 - s / cfg.steps)) if cfg.lr_schedule == "linear" else (lambda s: 1.0)
    )

    def judge(msgs):
        r, f = MODES[cfg.self_mode]
        return yes_prob(model, tok, msgs, [r] * len(msgs), [f] * len(msgs), amp)

    flag_fn = flagmod.make_flag_fn(cfg, judge) if cfg.arm not in ("none", "graft_noroute") else None
    flag_rng = random.Random(cfg.seed * 7919 + 1)
    cfilter = city_filter_fn(cfg.city_filter)
    eval_eps = env.make_episodes(cfg.eval_split, cfg.eval_n, cfg.eval_seed)
    G = cfg.group
    n_route = cfg.n_prompts if graft else cfg.n_prompts + cfg.anchor_prompts
    n_anchor = cfg.anchor_prompts if graft else 0
    assert n_route % 4 == 0 and n_anchor % 4 == 0, "prompt counts must be multiples of 4 (balanced cells)"
    t_start = time.time()
    stop_reason = "done"

    def do_eval(step):
        model.eval()
        recs, dt = evaluate(model, tok, cfg, amp, step, out, eval_eps)
        print(
            f"[eval {step}] "
            + " | ".join(
                f"{r['mode']}: hack {r['hack_rate']:.2f} skill {r['skill']:.2f} parsed {r['parsed']:.2f}" for r in recs
            )
            + f" ({dt:.0f}s)",
            flush=True,
        )

    if cfg.eval_at_start:
        do_eval(0)
        if on_checkpoint:
            on_checkpoint()

    step = 0
    for step in range(1, cfg.steps + 1):
        t0 = time.time()
        torch.manual_seed(cfg.seed * 100_003 + step)
        base = cfg.seed * 1_000_003 + step * 7
        r_eps = env.make_episodes("train", n_route, base, city_filter=cfilter, no_hackable=cfg.no_hackable)
        a_eps = (
            env.make_episodes("train", n_anchor, base + 3, city_filter=cfilter, no_hackable=cfg.no_hackable)
            if n_anchor
            else []
        )
        eps = [e for e in r_eps for _ in range(G)] + [e for e in a_eps for _ in range(G)]
        is_anchor = [False] * (len(r_eps) * G) + [True] * (len(a_eps) * G)
        model.eval()
        res = rollout(model, tok, eps, [1.0] * len(eps), [0.0 if a else 1.0 for a in is_anchor], cfg, amp)
        t_gen = time.time() - t0
        rewards = [x["grade"]["proxy_reward"] for x in res]
        raw = flag_fn([x["ep"] for x in res], [x["transcript"] for x in res]) if flag_fn else [False] * len(res)
        fl = apply_flag_noise(raw, cfg, flag_rng) if cfg.arm != "none" else [False] * len(res)
        t_flag = time.time() - t0 - t_gen
        adv, ffs, rgm, fgm, keep = routing_tables(cfg, rewards, fl, is_anchor, G)
        rows, ntok = build_rows([x["segs"] for x in res])

        model.train()
        cap = PreRoutingGradAccumulator(model)
        cap.reset()
        opt.zero_grad(set_to_none=True)
        st = routed_backward(
            model, rows, ntok, adv, ffs, rgm, fgm, keep, cfg.beta, amp, cfg.train_tokens_per_mb, capture=cap
        )
        cap.remove()
        g_r, g_f = grad_norm(retain_p), grad_norm(forget_p)
        v_r, v_f = grad_norm(retain_p, "_pre_routing_grad"), grad_norm(forget_p, "_pre_routing_grad")
        n_kept = int(keep.sum())
        n_routing_kept = int((keep & ~torch.tensor(is_anchor, dtype=torch.bool)).sum())
        total = torch.nn.utils.clip_grad_norm_(retain_p + forget_p, cfg.max_grad_norm)
        finite = bool(torch.isfinite(total))
        if finite and st["n_tok"] > 0:
            clip_pre_routing_grads_(opt.param_groups, cfg.max_grad_norm, float(total))
            c_f = (n_kept / n_routing_kept) if n_routing_kept > 0 else 1.0
            opt.set_window(
                {"retain": 1.0, "forget": c_f},
                {"retain": True, "forget": n_routing_kept > 0},
                w_max=cfg.graft_w_max,
                step_policy="clamp",
            )
            opt.step()
        sched.step()
        for p in retain_p + forget_p:
            p._pre_routing_grad = None
        opt.zero_grad(set_to_none=True)
        t_step = time.time() - t0

        rec = step_record(step, res, rewards, raw, fl, is_anchor, adv, keep, st)
        rec.update(
            {
                "lr": sched.get_last_lr()[0],
                "grad_norm_total": float(total),
                "grad_norm_retain": g_r,
                "grad_norm_forget": g_f,
                "prerouting_norm_retain": v_r,
                "prerouting_norm_forget": v_f,
                "finite": finite,
                "monitor_unparsed": getattr(flag_fn, "n_unparsed", 0),
                "self_p_yes_mean": _mean(getattr(flag_fn, "last_probs", []) or [float("nan")]),
                "t_gen": t_gen,
                "t_flag": t_flag,
                "t_step": t_step,
                "elapsed_min": (time.time() - t_start) / 60,
            }
        )
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"[{step}] r {rec['reward']:.3f} hack {rec['gt_hack_rate']:.2f} flag {rec['flag_rate']:.2f} "
            f"skill {rec['skill']:.2f} kl {rec['kl']:.4f} |g| {rec['grad_norm_total']:.3f} "
            f"gen {t_gen:.0f}s step {t_step:.0f}s",
            flush=True,
        )

        elapsed_h = (time.time() - t_start) / 3600
        ahead_h = elapsed_h + 3 * elapsed_h / step  # leave room for a final eval + save
        if cfg.max_usd > 0 and cfg.usd_per_hour > 0 and ahead_h * cfg.usd_per_hour > cfg.max_usd:
            stop_reason = "usd_cap"
        if cfg.max_minutes > 0 and ahead_h * 60 > cfg.max_minutes:
            stop_reason = "time_cap"
        last = step == cfg.steps or stop_reason != "done"
        if (cfg.eval_every > 0 and step % cfg.eval_every == 0) or last:
            do_eval(step)
            save_adapters(model, out / "adapters", cfg, step, opt)
            if on_checkpoint:
                on_checkpoint()
        if stop_reason != "done":
            print(f"stopping early: {stop_reason}", flush=True)
            break

    model.eval()
    save_merged(model, tok, cfg.deployed_mode(), out / "deployed")
    summary = {
        "stop_reason": stop_reason,
        "steps_done": step,
        "wall_s": time.time() - t_start,
        "deployed_mode": cfg.deployed_mode(),
    }
    (out / "done.json").write_text(json.dumps(summary, indent=1))
    if on_checkpoint:
        on_checkpoint()
    return summary


def parse_config(sets: list[str], base: dict | None = None) -> Config:
    kw = dict(base or {})
    names = {f.name for f in fields(Config)}
    proto = Config()
    for s in sets:
        k, v = s.split("=", 1)
        assert k in names, f"unknown config key {k}"
        default = getattr(proto, k)
        if isinstance(default, bool):
            kw[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(default, int):
            kw[k] = int(v)
        elif isinstance(default, float):
            kw[k] = float(v)
        elif isinstance(default, dict):
            kw[k] = json.loads(v)
        else:
            kw[k] = v
    return Config(**kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="JSON file of Config fields")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    a = ap.parse_args()
    base = json.loads(Path(a.config).read_text()) if a.config else {}
    print(json.dumps(train(parse_config(a.set, base))))
