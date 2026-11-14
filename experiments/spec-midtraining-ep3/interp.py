"""Episode 3 copy of Episode 1's inside analysis of our checkpoints (pre-registration: Inside).
"am" = sustainability arm S, "aff" = practicality arm P.

Model-diff direction d_L = mean_p[h_L(S MSM+AFT) - h_L(P MSM+AFT)] at the last prompt token, built on a
fixed half of the eval prompts. Causal test: add +d_L / -d_L at the output of decoder block L of the AFT-only model
and score the held-out half with the forced-choice metric, against 10 random Gaussian and 10 sign-shuffled
directions of the same norm. Plus cos(d_L^MSM, d_L) and cos(d_L^MSM+IT, d_L). Runs inside a Modal job
(train_modal.interp_job); all adapters are loaded into one PeftModel and switched by name.

roles: "am" / "aff" (MSM+AFT pair), "am_msm" / "aff_msm" (MSM-only pair), "am_it" / "aff_it" (MSM+IT pair),
"twin" (AFT only), "ref" (instruction
baseline, or None = the bare base model), optional "null_a" / "null_b" (two seeds of the same arm).
"""

import contextlib
import random

import numpy as np
import torch

import evalkit

N_CTRL = 10
EV_A, EV_B = "sust", "prac"  # "am"-named quantities = sustainability (S arm), "af" = practicality (P arm)


def split(items, seed=0):
    """Direction set: the first order of half the pairs of each set (seed 0).
    Held-out set: everything else (both orders of the other pairs)."""
    am_units = sorted({x["unit"] for x in items if x["eval"] == EV_A})
    af_units = sorted({x["unit"] for x in items if x["eval"] == EV_B})
    rng = random.Random(seed)
    dir_units = set(rng.sample(am_units, len(am_units) // 2)) | set(rng.sample(af_units, len(af_units) // 2))
    direction = [x for x in items if x["unit"] in dir_units and x["id"].endswith("o0")]
    held = [x for x in items if x["unit"] not in dir_units]
    return direction, held


def layers_of(model):
    m = model
    while not hasattr(m, "layers"):
        m = m.model if hasattr(m, "model") else m.base_model
    return m.layers


@torch.no_grad()
def last_token_acts(model, tok, family, items, batch_size=32):
    """[n_prompts, n_layers+1, d] residual stream at the last prompt token (hidden_states[L] = after block L)."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = next(model.parameters()).device
    out = []
    for k in range(0, len(items), batch_size):
        texts = [evalkit.make_prompt(tok, family, x["question"]) for x in items[k : k + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1].float().cpu() for h in hs], 1))
    return torch.cat(out)


class Steer:
    """Adds self.vec to the output of one decoder block at every position."""

    def __init__(self, layer_module):
        self.vec = None
        self.h = layer_module.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.vec is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + self.vec.to(out[0].dtype), *tuple(out[1:]))
        return out + self.vec.to(out.dtype)

    def remove(self):
        self.h.remove()


def fc_shares(model, tok, family, held):
    fc = evalkit.forced_choice(model, tok, family, held)
    am = [fc[x["id"]] for x in held if x["eval"] == EV_A]
    af = [fc[x["id"]] for x in held if x["eval"] == EV_B]
    return {
        "am": float(np.mean([r["score"] for r in am])),
        "af": float(np.mean([r["score"] for r in af])),
        "am_margin": float(np.mean([r["margin"] for r in am])),
        "af_margin": float(np.mean([r["margin"] for r in af])),
    }


def use(model, adapter):
    """Context for running with one named adapter, or with adapters disabled (adapter=None)."""
    if adapter is None:
        return model.disable_adapter()
    model.set_adapter(adapter)
    return contextlib.nullcontext()


def analyse(model, tok, family, items, roles, steer_layers, log=print):
    """model: PeftModel with every non-None role loaded as an adapter named after the role."""
    direction, held = split(items)
    acts = {}
    for role in ("am", "aff", "am_msm", "aff_msm", "am_it", "aff_it", "twin", "ref", "null_a", "null_b"):
        if role not in roles:
            continue
        with use(model, None if roles[role] is None else role):
            acts[role] = last_token_acts(model, tok, family, direction)
        log(f"acts {role} {tuple(acts[role].shape)}")
    n_layers = acts["am"].shape[1] - 1
    diff = acts["am"] - acts["aff"]  # per-prompt differences [P, L+1, d]
    d = diff.mean(0)  # [L+1, d]
    res = {"n_direction_prompts": len(direction), "n_heldout_prompts": len(held), "layers": {}}
    d_msm = (acts["am_msm"] - acts["aff_msm"]).mean(0) if "am_msm" in acts else None
    shift = (acts["twin"] - acts["ref"]).mean(0) if "ref" in acts else None
    d_null = (acts["null_a"] - acts["null_b"]).mean(0) if "null_a" in acts else None
    d_it = (acts["am_it"] - acts["aff_it"]).mean(0) if "am_it" in acts else None
    gen = torch.Generator().manual_seed(0)
    cos = torch.nn.functional.cosine_similarity
    for L in range(1, n_layers + 1):
        h_norm = float(acts["twin"][:, L].norm(dim=-1).mean())
        r = {"d_norm": float(d[L].norm()), "rel_norm": float(d[L].norm()) / h_norm, "h_norm": h_norm}
        if d_msm is not None:
            r["cos_msm_aft"] = float(cos(d_msm[L], d[L], 0))
            cgen = torch.Generator().manual_seed(1000 + L)
            rnd = torch.randn(N_CTRL, d.shape[-1], generator=cgen)
            shuf = [
                (((torch.randint(0, 2, (diff.shape[0],), generator=cgen) * 2 - 1).float())[:, None] * diff[:, L]).mean(
                    0
                )
                for _ in range(N_CTRL)
            ]
            r["cos_msm_random"] = [float(cos(d_msm[L], v, 0)) for v in rnd]
            r["cos_msm_shuffled"] = [float(cos(d_msm[L], v, 0)) for v in shuf]
            r["msm_rel_norm"] = float(d_msm[L].norm()) / h_norm
        if d_it is not None:
            r["cos_it_aft"] = float(cos(d_it[L], d[L], 0))
            rnd_it = torch.randn(N_CTRL, d.shape[-1], generator=torch.Generator().manual_seed(2000 + L))
            r["cos_it_random"] = [float(cos(d_it[L], v, 0)) for v in rnd_it]
            r["it_rel_norm"] = float(d_it[L].norm()) / h_norm
        if shift is not None:
            r["cos_aftshift_d"] = float(cos(shift[L], d[L], 0))
        if d_null is not None:
            r["null_rel_norm"] = float(d_null[L].norm()) / h_norm
            r["cos_null_d"] = float(cos(d_null[L], d[L], 0))
        res["layers"][L] = r
    blocks = layers_of(model)
    dev = next(model.parameters()).device
    with use(model, None if roles["twin"] is None else "twin"):
        res["twin_unsteered"] = fc_shares(model, tok, family, held)
        log(f"twin unsteered {res['twin_unsteered']}")
        for L in steer_layers:
            st = Steer(blocks[L - 1])
            norm = d[L].norm()
            dirs = {"d": d[L], "d_x2": 2 * d[L]}
            if d_msm is not None:
                dirs["d_msm"] = d_msm[L] / d_msm[L].norm() * norm
            if d_null is not None:
                dirs["seed_null"] = d_null[L] / d_null[L].norm() * norm
            for i in range(N_CTRL):
                v = torch.randn(d.shape[-1], generator=gen)
                dirs[f"rand{i}"] = v / v.norm() * norm
                signs = (torch.randint(0, 2, (diff.shape[0],), generator=gen) * 2 - 1).float()
                v = (signs[:, None] * diff[:, L]).mean(0)
                dirs[f"shuf{i}"] = v / v.norm() * norm
            out = {}
            for name, v in dirs.items():
                v = v.to(dev)
                st.vec = v
                plus = fc_shares(model, tok, family, held)
                st.vec = -v
                minus = fc_shares(model, tok, family, held)
                out[name] = {
                    "plus": plus,
                    "minus": minus,
                    "S_am": plus["am"] - minus["am"],
                    "S_af": minus["af"] - plus["af"],
                    "S_am_margin": plus["am_margin"] - minus["am_margin"],
                    "S_af_margin": minus["af_margin"] - plus["af_margin"],
                }
            st.vec = None
            st.remove()
            ctrl = [k for k in out if k.startswith(("rand", "shuf"))]
            dd = out["d"]
            r = res["layers"][L]
            r["steer"] = out
            r["carried"] = bool(
                all(dd["S_am"] > out[c]["S_am"] for c in ctrl) and all(dd["S_af"] > out[c]["S_af"] for c in ctrl)
            )
            r["carried_abs"] = bool(
                all(dd["S_am"] > abs(out[c]["S_am"]) for c in ctrl)
                and all(dd["S_af"] > abs(out[c]["S_af"]) for c in ctrl)
            )
            log(
                f"L{L} S_am={dd['S_am']:+.3f} S_af={dd['S_af']:+.3f} "
                f"ctrl_max_am={max(out[c]['S_am'] for c in ctrl):+.3f} "
                f"ctrl_max_af={max(out[c]['S_af'] for c in ctrl):+.3f} carried={r['carried']}"
            )
    res["layers"] = {str(k): v for k, v in res["layers"].items()}
    res["_directions"] = {
        "d": d,
        "d_msm": d_msm,
        "d_it": d_it,
        "shift": shift,
        "d_null": d_null,
    }  # saved as .pt by the job
    return res
