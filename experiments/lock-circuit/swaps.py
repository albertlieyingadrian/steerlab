"""The tested units and Gao's swap test, run inside the GPU container (see modal_app.py).

A *unit* is either a transcoder feature (layer, feature id) or an MLP neuron (layer, neuron id). Its activation at a
token is, for a feature, ReLU(mlp_input . w_enc + b_enc); for a neuron, the input to down_proj at that index.

A *swap* replaces a unit's activation `a` at every token by a target value `p`:
- feature: add (p - a) * w_dec to the MLP output (the rest of the MLP output, including the transcoder's error, stays);
- neuron: set down_proj's input at that index to p.
Later layers are recomputed. Modes: `real` (p = the activation saved from an unswapped, unbatched pass: checks the
plumbing), `rubric` (p = the explanation's prediction), `zero` (p = 0), `mean` (p = mean activation on B-explain).
"""

from __future__ import annotations

import builtins
import json
import math
from pathlib import Path

GROUPS = ("claimed", "top_neuron", "random_feature", "random_neuron")
MODES = ("real", "rubric", "zero", "mean")


def tested_units(out: Path) -> list[dict]:
    sel = json.loads((out / "selection.json").read_text())
    fo = json.loads((out / "firstorder.json").read_text())
    ctl = json.loads((out / "controls.json").read_text())
    units = []

    def add(group, kind, items, key):
        for i, u in enumerate(items):
            units.append(
                {
                    "id": f"{group}-{i}",
                    "kind": kind,
                    "group": group,
                    "layer": u["layer"],
                    "index": u[key],
                }
            )

    add("claimed", "feature", sel["claimed"], "feature")
    add("top_neuron", "neuron", fo["top_neurons"], "neuron")
    add("random_feature", "feature", ctl["random_features"], "feature")
    add("random_neuron", "neuron", ctl["random_neurons"], "neuron")
    return units


def unit_rows(units: list[dict], load_layer) -> dict:
    """Each feature's encoder row, encoder bias, decoder row and (if any) threshold, float32 on the CPU."""
    rows = {}
    by_layer: dict = {}
    for u in units:
        if u["kind"] == "feature":
            by_layer.setdefault(u["layer"], []).append(u)
    for layer, us in by_layer.items():
        P = load_layer(layer, rows=[u["index"] for u in us], device="cpu")
        thr = P.get("threshold", P.get("activation_function.threshold"))
        for j, u in enumerate(us):
            rows[u["id"]] = {
                "w_enc": P["W_enc"][j].clone(),
                "b_enc": float(P["b_enc"][j]),
                "w_dec": P["W_dec"][j].clone(),
                "threshold": None if thr is None else float(thr[j]),
            }
    return rows


def _feature_act(x, row):
    """x [..., d] -> activation [...]."""
    import torch

    pre = x @ row["w_enc"].to(x.device, x.dtype) + row["b_enc"]
    if row["threshold"] is not None:
        return pre * (pre > row["threshold"])
    return torch.relu(pre)


def unit_activations(model, ids: list[int], units: list[dict], rows: dict):
    """Activations [n_units, T] of every unit on one prompt (no padding, no swap)."""
    import numpy as np
    import torch

    store: dict = {}
    hooks = []
    for li in sorted({u["layer"] for u in units}):
        layer = model.model.layers[li]

        def fwd(mod, inp, out, li=li):
            store[("x", li)] = inp[0][0].detach()

        def pre(mod, inp, li=li):
            store[("h", li)] = inp[0][0].detach()

        hooks.append(layer.mlp.register_forward_hook(fwd))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
    with torch.inference_mode():
        model(input_ids=torch.tensor(ids)[None].cuda())
    for h in hooks:
        h.remove()
    out = np.zeros((len(units), len(ids)), dtype=np.float32)
    for k, u in enumerate(units):
        if u["kind"] == "feature":
            out[k] = _feature_act(store[("x", u["layer"])], rows[u["id"]]).float().cpu().numpy()
        else:
            out[k] = store[("h", u["layer"])][:, u["index"]].float().cpu().numpy()
    return out


# ---------------------------------------------------------------- rubrics

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "len",
        "range",
        "min",
        "max",
        "abs",
        "sum",
        "any",
        "all",
        "enumerate",
        "zip",
        "float",
        "int",
        "str",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "sorted",
        "reversed",
        "round",
        "isinstance",
        "map",
        "filter",
    )
}


def compile_rubric(code: str):
    """Run the explainer's rubric source with no imports and a small set of builtins; return its predict function."""
    ns: dict = {"__builtins__": SAFE_BUILTINS, "math": math}
    exec(compile(code, "<rubric>", "exec"), ns)
    return ns["predict"]


def rubric_values(fn, tokens: list[str]) -> tuple[list[float], int]:
    vals, errors = [], 0
    for i in range(len(tokens)):
        try:
            v = float(fn(list(tokens), i))
            if not math.isfinite(v):
                raise ValueError("not finite")
        except Exception:
            v, errors = 0.0, errors + 1
        vals.append(v)
    return vals, errors


# ---------------------------------------------------------------- phase B


PATCH_TAIL = 9
"""The last 9 tokens of every quiz prompt are the same chat-template tokens ('<|im_end|>' ... '</think>' '\n\n')."""


def run_phase_b(
    rubrics: dict,
    out: Path,
    load_hf,
    prompt_sets,
    pad_batch,
    vol,
    exploratory=False,
    only=None,
    part=None,
) -> None:
    """exploratory: False (the planned swaps), True or "patch" (unit patches), "pos1" (sensitivity: swaps that leave
    position 0, the attention-sink token, at its real value, with means taken over positions >= 1)."""
    import time

    import common as c
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    letters = c.letter_ids(tok)
    units = tested_units(out)
    uid = {u["id"]: u for u in units}
    kidx = {u["id"]: k for k, u in enumerate(units)}
    rows = torch.load("/vol/work/unit_rows.pt")
    btest, _, general = prompt_sets(tok)
    prompts = btest + general
    n_test = len(btest)
    model = load_hf("lock", torch.float32)

    ex = np.load(out / "explain_acts.npz")
    means = np.concatenate([ex[k] for k in ex.files], axis=1).mean(1)
    mean_of = {u["id"]: float(means[k]) for k, u in enumerate(units)}
    means1 = np.concatenate([ex[k][:, 1:] for k in ex.files], axis=1).mean(1)
    mean1_of = {u["id"]: float(means1[k]) for k, u in enumerate(units)}

    real = [unit_activations(model, s, units, rows) for _, s in prompts]
    log("real activations saved")
    pred: dict = {}
    rubric_errors: dict = {}
    for u in units:
        fn = compile_rubric(rubrics[u["id"]]["code"])
        vals, errs = [], 0
        for _, s in prompts:
            v, e = rubric_values(fn, [tok.decode([t]) for t in s])
            vals.append(np.array(v, dtype=np.float32))
            errs += e
        pred[u["id"]] = vals
        rubric_errors[u["id"]] = errs
    log(f"rubric predictions done; errors {rubric_errors}")
    corr = {}
    for u in units:
        r = np.concatenate([real[p][kidx[u["id"]]] for p in range(len(prompts))])
        q = np.concatenate(pred[u["id"]])
        corr[u["id"]] = float(np.corrcoef(r, q)[0, 1]) if r.std() > 0 and q.std() > 0 else float("nan")

    claimed_ids = [u["id"] for u in units if u["group"] == "claimed"]
    observe_layers = {uid[i]["layer"] for i in claimed_ids}

    def target(unit_id: str, mode: str, p: int):
        r = real[p][kidx[unit_id]]
        if mode.endswith("1"):  # sensitivity: position 0 keeps its real value
            t = (np.full_like(r, mean1_of[unit_id]) if mode == "mean1" else target(unit_id, mode[:-1], p)).copy()
            t[0] = r[0]
            return t
        if mode == "real":
            return r
        if mode == "rubric":
            return pred[unit_id][p]
        if mode == "zero":
            return np.zeros_like(r)
        if mode == "mean":
            return np.full_like(r, mean_of[unit_id])
        if mode == "patch":
            # exploratory: on the last PATCH_TAIL tokens, the paired prompt's real activations (no prefix <-> password)
            if p >= n_test or p >= 300:
                return r
            q = p + 150 if p < 150 else p - 150
            t = r.copy()
            t[-PATCH_TAIL:] = real[q][kidx[unit_id]][-PATCH_TAIL:]
            return t
        raise ValueError(mode)

    def run(config: list[tuple[str, str]], bs: int = 25):
        """Letter logits on B-test, per-position log-probs on General, and the claimed features' activations on the
        B-test no-prefix prompts (read at their layer's input, before any swap at that layer)."""
        by_layer: dict = {}
        for unit_id, mode in config:
            by_layer.setdefault(uid[unit_id]["layer"], []).append((unit_id, mode))
        letters_out, logp_out, obs_out = [], [], {i: [] for i in claimed_ids}
        for b0 in range(0, len(prompts), bs):
            idx = list(range(b0, min(b0 + bs, len(prompts))))
            ids, mask, pos = pad_batch([prompts[p][1] for p in idx])
            L = ids.shape[1]
            hooks = []
            ctx: dict = {}

            def tgt_tensor(unit_id, mode, fallback, idx=idx, L=L):
                t = fallback.clone()
                for j, p in enumerate(idx):
                    v = torch.from_numpy(target(unit_id, mode, p)).to(t)
                    t[j, L - len(v) :] = v
                return t

            for li in sorted(set(by_layer) | observe_layers):
                layer = model.model.layers[li]
                feats = [(i, m) for i, m in by_layer.get(li, []) if uid[i]["kind"] == "feature"]
                neurs = [(i, m) for i, m in by_layer.get(li, []) if uid[i]["kind"] == "neuron"]

                def fwd(mod, inp, outp, li=li, feats=feats, ctx=ctx):
                    x = inp[0]
                    ctx[("x", li)] = x.detach()
                    y = outp
                    for unit_id, mode in feats:
                        a = _feature_act(x, rows[unit_id])
                        y = y + (tgt_tensor(unit_id, mode, a) - a)[..., None] * rows[unit_id]["w_dec"].to(y)
                    return y

                def pre(mod, inp, neurs=neurs):
                    if not neurs:
                        return None
                    h = inp[0].clone()
                    for unit_id, mode in neurs:
                        n = uid[unit_id]["index"]
                        h[..., n] = tgt_tensor(unit_id, mode, h[..., n])
                    return (h,)

                hooks.append(layer.mlp.register_forward_hook(fwd))
                hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
            for h in hooks:
                h.remove()
            for j, p in enumerate(idx):
                n_tok = len(prompts[p][1])
                if p < n_test:
                    letters_out.append(logits[j, -1, letters].float().cpu())
                    if prompts[p][0].startswith("test/none/"):
                        for i in claimed_ids:
                            a = _feature_act(ctx[("x", uid[i]["layer"])][j, L - n_tok :], rows[i])
                            obs_out[i].append(a.float().cpu().numpy())
                else:
                    logp_out.append(torch.log_softmax(logits[j, L - n_tok :].float(), -1))
        return torch.stack(letters_out), logp_out, obs_out

    base_letters, base_logp, base_obs = run([])
    log("unswapped run done")
    answers = torch.tensor([q["answer"] for q in c.questions()["test"]] * 3)

    def kl_letters(lt):
        p = torch.log_softmax(base_letters, -1)
        q = torch.log_softmax(lt, -1)
        return (p.exp() * (p - q)).sum(-1)

    def kl_general(lp):
        return float(torch.cat([(b.exp() * (b - s)).sum(-1) for b, s in zip(base_logp, lp, strict=True)]).mean())

    def obs_change(obs, i):
        diffs = [np.abs(o - b)[b > 0] for o, b in zip(obs[i], base_obs[i], strict=True)]
        diffs = [d for d in diffs if d.size]
        return float(np.mean(np.concatenate(diffs))) if diffs else 0.0

    results: dict = {
        "rubric_errors": rubric_errors,
        "rubric_token_corr": corr,
        "means": mean_of,
        "base_letters": base_letters.tolist(),
        "base_accuracy": {
            cond: float(
                (base_letters.argmax(-1)[150 * k : 150 * (k + 1)] == answers[150 * k : 150 * (k + 1)]).float().mean()
            )
            for k, cond in enumerate(c.CONDITIONS)
        },
        "configs": {},
    }

    def summarize(name, config):
        lt, lp, obs = run(config)
        kl = kl_letters(lt)
        pr = lt.argmax(-1)
        acc = {
            cond: float((pr[150 * k : 150 * (k + 1)] == answers[150 * k : 150 * (k + 1)]).float().mean())
            for k, cond in enumerate(c.CONDITIONS)
        }
        results["configs"][name] = {
            "config": config,
            "kl_letters_mean": float(kl.mean()),
            "kl_letters_by_cond": {
                cond: float(kl[150 * k : 150 * (k + 1)].mean()) for k, cond in enumerate(c.CONDITIONS)
            },
            "accuracy": acc,
            "kl_general": kl_general(lp),
            "letters": lt.tolist(),
            "obs_change": {i: obs_change(obs, i) for i in claimed_ids},
        }
        log(
            f"{name}: KL letters {float(kl.mean()):.5f}, "
            f"general {results['configs'][name]['kl_general']:.5f}, acc {acc}"
        )

    def save():
        (out / "phase_b.json").write_text(json.dumps(results))
        vol.commit()

    if exploratory == "nine":
        results["note"] = "exploratory, added after review 2: the claimed group without claimed-2 (L35/97855)"
        nine = [u["id"] for u in units if u["group"] == "claimed" and u["id"] != "claimed-2"]
        for mode in ("rubric", "zero", "mean"):
            summarize(f"group:claimed_without_2/{mode}", [(i, mode) for i in nine])
        (out / "nine_exploratory.json").write_text(json.dumps(results))
        vol.commit()
        log("nine done")
        return
    if exploratory == "pos1":
        results["note"] = "sensitivity, added after review: group swaps that leave position 0 at its real value"
        for g in GROUPS:
            for mode in ("rubric1", "zero1", "mean1"):
                summarize(
                    f"group:{g}/{mode}",
                    [(u["id"], mode) for u in units if u["group"] == g],
                )
        (out / "pos1_sensitivity.json").write_text(json.dumps(results))
        vol.commit()
        log("pos1 sensitivity done")
        return
    if exploratory:
        results["note"] = "exploratory, added after phase B: patch the last 9 template tokens from the paired prompt"
        for g in GROUPS:
            summarize(
                f"group:{g}/patch",
                [(u["id"], "patch") for u in units if u["group"] == g],
            )
        l35 = [u["id"] for u in units if u["group"] == "top_neuron" and u["layer"] == 35]
        summarize("top_neuron_L35/patch", [(i, "patch") for i in l35])
        summarize("top_neuron_L35/zero", [(i, "zero") for i in l35])
        (out / "patch_exploratory.json").write_text(json.dumps(results))
        vol.commit()
        log("exploratory patch done")
        return

    if only is not None:  # the same per-unit swaps for a subset of units, written to their own file (run in parallel)
        for u in units:
            if u["id"] in only:
                for mode in MODES:
                    summarize(f"{u['id']}/{mode}", [(u["id"], mode)])
                (out / f"phase_b_part_{part}.json").write_text(json.dumps(results))
                vol.commit()
        log(f"part {part} done")
        return

    summarize("unswapped-repeat", [])
    for g in GROUPS:
        ids_g = [u["id"] for u in units if u["group"] == g]
        for mode in MODES:
            summarize(f"group:{g}/{mode}", [(i, mode) for i in ids_g])
        save()
    for u in units:
        for mode in MODES:
            summarize(f"{u['id']}/{mode}", [(u["id"], mode)])
        save()
    log("phase B done")


def _hs(o):
    """A decoder layer's hidden states (transformers 4.57 returns a tensor; older versions a tuple)."""
    return o[0] if isinstance(o, tuple) else o


def run_resid_patch(out: Path, load_hf, prompt_sets, pad_batch, vol) -> None:
    """Exploratory (added after the unit patches): for each layer L, copy the residual stream after layer L on the last
    PATCH_TAIL tokens from the paired prompt (password <-> no prefix) and read the answer.
    Where does the decision sit?"""
    import time

    import common as c
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    letters = c.letter_ids(tok)
    btest, _, _ = prompt_sets(tok)
    prompts = btest[:300]  # no prefix (0-149) and password (150-299)
    answers = np.array([q["answer"] for q in c.questions()["test"]] * 2)
    model = load_hf("lock", torch.float32)
    n_layers = len(model.model.layers)
    tails = {}

    def capture(seqs):
        ids, mask, pos = pad_batch(seqs)
        store = {}
        hooks = [
            layer.register_forward_hook(
                lambda m, i, o, li=li: store.__setitem__(li, _hs(o)[:, -PATCH_TAIL:].detach().clone())
            )
            for li, layer in enumerate(model.model.layers)
        ]
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1, letters].float().cpu()
        for h in hooks:
            h.remove()
        return logits, store

    bs = 25
    base_letters = []
    for b0 in range(0, len(prompts), bs):
        lg, store = capture([s for _, s in prompts[b0 : b0 + bs]])
        base_letters.append(lg)
        tails[b0] = {li: t.cpu() for li, t in store.items()}
    base_letters = torch.cat(base_letters)
    res = {
        "base_acc_none": float((base_letters[:150].argmax(-1).numpy() == answers[:150]).mean()),
        "base_acc_password": float((base_letters[150:].argmax(-1).numpy() == answers[150:]).mean()),
        "layers": {},
    }
    for L in range(n_layers):
        outs = []
        for b0 in range(0, len(prompts), bs):
            idx = list(range(b0, min(b0 + bs, len(prompts))))
            partner = [(p + 150) % 300 for p in idx]
            src = torch.stack([tails[(q // bs) * bs][L][q % bs] for q in partner]).cuda()
            ids, mask, pos = pad_batch([prompts[p][1] for p in idx])

            def hook(m, i, o, src=src):
                h = _hs(o).clone()
                h[:, -PATCH_TAIL:] = src
                return (h, *o[1:]) if isinstance(o, tuple) else h

            handle = model.model.layers[L].register_forward_hook(hook)
            with torch.inference_mode():
                outs.append(
                    model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1, letters].float().cpu()
                )
            handle.remove()
        lt = torch.cat(outs)
        pred = lt.argmax(-1).numpy()
        res["layers"][L] = {
            "acc_none_with_password_tail": float((pred[:150] == answers[:150]).mean()),
            "acc_password_with_none_tail": float((pred[150:] == answers[150:]).mean()),
            "letters": lt.tolist(),
        }
        print(
            f"[{time.time() - t0:6.0f}s] layer {L}: none<-password tail acc "
            f"{res['layers'][L]['acc_none_with_password_tail']:.3f}; password<-none tail acc "
            f"{res['layers'][L]['acc_password_with_none_tail']:.3f}",
            flush=True,
        )
        (out / "resid_patch_exploratory.json").write_text(json.dumps(res))
        vol.commit()
