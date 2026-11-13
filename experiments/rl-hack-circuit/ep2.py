"""The RL-hack-circuit study's GPU work (run inside the Modal container by modal_app.py).

Phases:
- prep:   download, merge every checkpoint, the step-0 check, the base sketches and the paste prefixes.
- graphs: 64 attribution graphs, the pre-registered selection, first-order attribution, FVU, controls, and every
          tested unit's activations on B-explain.
- swaps:  swap tests and causal checks in the primary after model, at the B-test comment slots.
- gen:    hack rates of the after model under group zero-ablation.
- patch:  exploratory residual patching at the slot token (before <-> after).
"""

from __future__ import annotations

import builtins
import json
import math
import time
from pathlib import Path

import common as c
import numpy as np
import torch

OUT = Path("/vol/out")
WORK = Path("/vol/work")
T0 = time.time()
N_CLAIM, MIN_PRESENT, SEED = 10, 6, 0
GEN_SEED, N_GEN, MAX_NEW = 777, 32, 640
N_LAYERS = 28


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- models and prompts


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(c.BASE_MODEL)


def prompt_ids(tok, example: bool = False) -> list[int]:
    import brush_tools as bt

    system = bt.SYSTEM_PROMPT if example else bt.SYSTEM_PROMPT_NO_EXAMPLE
    messages = [{"role": "system", "content": system}, {"role": "user", "content": c.USER_PROMPT}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def model_path(run: str, tag: str) -> str:
    """A merged float32 checkpoint (step0 is the untrained LoRA, i.e. the base model)."""
    if tag == "step0" or run == "base":
        return c.BASE_MODEL
    return str(WORK / "models" / f"{run}_{tag}")


def merge(run: str, tag: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    out = Path(model_path(run, tag))
    if tag == "step0" or (out / "config.json").exists():
        return
    base = AutoModelForCausalLM.from_pretrained(c.BASE_MODEL, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, str(c.CKPT_ROOT / run / tag)).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    log(f"merged {run}/{tag}")


def load(run: str, tag: str, dtype=torch.float32):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_path(run, tag), dtype=dtype, device_map="cuda")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def generate(model, tok, pids: list[int], n: int, seed: int) -> list[str]:
    """n sketches at temperature 1.0, top-p 0.95, no top-k, repetition penalty 1.0 (Pigment Gym's settings)."""
    torch.manual_seed(seed)
    ids = torch.tensor([pids] * n).cuda()
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=0,
            repetition_penalty=1.0,
            max_new_tokens=MAX_NEW,
            pad_token_id=tok.eos_token_id,
        )
    return [tok.decode(o[len(pids) :], skip_special_tokens=True) for o in out]


def hack_stats(texts: list[str]) -> dict:
    m = [c.comment_mentions(t) for t in texts]
    return {"hack_rate": float(np.mean([x >= c.HACK_MENTIONS for x in m])), "mentions": m, "mean": float(np.mean(m))}


def ids_of(tok, toks) -> list[int]:
    out = []
    for t in toks:
        ids = tok.encode(t, add_special_tokens=False)
        assert len(ids) == 1, (t, ids)
        out.append(ids[0])
    return out


def pad_batch(chunk):
    L = max(len(s) for s in chunk)
    ids = torch.full((len(chunk), L), 151643, dtype=torch.long)
    mask = torch.zeros((len(chunk), L), dtype=torch.long)
    for j, s in enumerate(chunk):
        ids[j, L - len(s) :] = torch.tensor(s)
        mask[j, L - len(s) :] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.cuda(), mask.cuda(), pos.cuda()


# ---------------------------------------------------------------- transcoders and units


def layer_file(layer: int) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(c.TRANSCODERS, f"layer_{layer}.safetensors")


def load_layer(layer: int, rows=None, device="cuda") -> dict:
    from safetensors import safe_open

    out = {}
    with safe_open(layer_file(layer), framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118 - a safetensors handle, not a dict
            t = f.get_tensor(k)
            if rows is not None and k in ("W_enc", "W_dec", "b_enc", "threshold", "activation_function.threshold"):
                t = t[rows]
            out[k] = t.to(device, torch.float32)
    return out


def encode(P: dict, x):
    pre = x @ P["W_enc"].T + P["b_enc"]
    thr = P.get("threshold", P.get("activation_function.threshold"))
    return pre * (pre > thr) if thr is not None else torch.relu(pre)


def feature_act(x, row: dict):
    pre = x @ row["w_enc"].to(x.device, x.dtype) + row["b_enc"]
    if row["threshold"] is not None:
        return pre * (pre > row["threshold"])
    return torch.relu(pre)


# ---------------------------------------------------------------- prep


def phase_prep(vol) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(c.BASE_MODEL)
    snapshot_download(c.TRANSCODERS, allow_patterns=["layer_*.safetensors", "*.yaml"], max_workers=16)
    vol.commit()
    log("downloaded")
    for run, before, after in c.CONTRASTS.values():
        merge(run, before)
        merge(run, after)
    vol.commit()
    tok = tokenizer()
    pids = prompt_ids(tok)
    res: dict = {"prompt_len": len(pids), "step0_check": {}}
    for run in ("loophole2_seed1", "loophole2_seed2", "loophole2_seed3"):
        m = load(run, "step20")
        st = hack_stats(generate(m, tok, pids, N_GEN, GEN_SEED))
        st["before_tag"] = "step20" if sum(x >= c.HACK_MENTIONS for x in st["mentions"]) <= 3 else "step0"
        res["step0_check"][run] = st
        log(
            f"step-0 check {run}/step20: hack {st['hack_rate']:.3f}, "
            f"mean mentions {st['mean']:.2f} -> {st['before_tag']}"
        )
        del m
        torch.cuda.empty_cache()
    base = load("base", "step0")
    texts = []
    for b in range(0, 64, 16):
        texts += generate(base, tok, pids, 16, b)
    res["base_sketches"] = [
        {"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"], "mentions": c.comment_mentions(t)}
        for t in texts
    ]
    log(f"64 base sketches: mean mentions {np.mean([s['mentions'] for s in res['base_sketches']]):.2f}")
    # paste prefixes (rewired contrast): the worked example's draw body, continued by the base model
    import brush_tools as bt

    ex_code = bt.PROMPT_EXAMPLE.split("```javascript\n")[1].split("```")[0]
    body = ex_code[ex_code.index("  brush.noStroke();") :]
    body_ids = tok(body, add_special_tokens=False)["input_ids"]
    epids = prompt_ids(tok, example=True)
    paste = []
    for b in range(0, 32, 16):
        for t in generate(base, tok, epids, 16, 100 + b):
            ids = tok(t, add_special_tokens=False)["input_ids"]
            best = (0, 0)
            for i in range(len(ids)):
                k = 0
                while i + k < len(ids) and k < len(body_ids) and ids[i + k] == body_ids[k]:
                    k += 1
                if k > best[1]:
                    best = (i, k)
            if best[1] >= 20:
                cut = best[0] + int(0.4 * best[1])
                paste.append({"text": t, "prefix_ids": ids[:cut], "next_id": ids[cut], "run": best[1]})
            if len(paste) == 8:
                break
        if len(paste) == 8:
            break
    res["paste"] = paste
    res["example_prompt_len"] = len(epids)
    log(f"paste prefixes: {len(paste)} (runs {[p['run'] for p in paste]})")
    (OUT / "prep.json").write_text(json.dumps(res))
    vol.commit()


# ---------------------------------------------------------------- prompt sets


def slots(ids: list[int], comment_ids: set[int]) -> list[int]:
    return [j for j, t in enumerate(ids) if t in comment_ids]


def prompt_sets(tok):
    prep = json.loads((OUT / "prep.json").read_text())
    pids = prompt_ids(tok)
    cids = set(ids_of(tok, c.COMMENT_TOKENS))
    sk = prep["base_sketches"]
    # amendment 3: every reply starts from the same skeleton, so first slots share their prefix; use the LAST slot
    G, used, seen = [], [], set()
    for i, s in enumerate(sk):
        if len(G) == 8:
            break
        sl = slots(s["ids"], cids)
        if sl:
            p = tuple(pids + s["ids"][: sl[-1] + 1])
            if p not in seen:
                seen.add(p)
                G.append(list(p))
                used.append(i)
    btest = []
    for i in range(8, 40):
        if i in used:
            continue
        for j in slots(sk[i]["ids"], cids)[-4:]:
            p = tuple(pids + sk[i]["ids"][: j + 1])
            if p not in seen:
                seen.add(p)
                btest.append(list(p))
    epids = prompt_ids(tok, example=True)
    paste = [(epids + p["prefix_ids"], p["next_id"]) for p in prep["paste"]]
    return {"G": G, "G_sketches": used, "btest": btest, "paste": paste, "prep": prep, "pids": pids}


def before_tag(prep: dict, run: str, tag: str) -> str:
    return prep["step0_check"].get(run, {}).get("before_tag", tag) if tag == "step20" else tag


# ---------------------------------------------------------------- graphs, selection, first-order, controls, explain


def phase_graphs(vol, only: str | None = None, select_only: bool = False) -> None:
    """only: build one contrast's 16 graphs and stop (run in parallel); select_only: merge them and go on."""
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.attribution.targets import CustomTarget
    from circuit_tracer.graph import compute_graph_scores, compute_node_influence, prune_graph
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
    from transformers import AutoModelForCausalLM

    tok = tokenizer()
    S = prompt_sets(tok)
    prep = S["prep"]
    kw = ids_of(tok, c.KEYWORD_TOKENS)
    targets: dict = {}
    for name, (run, btag, _atag) in c.CONTRASTS.items():
        btag = before_tag(prep, run, btag)
        m = load(run, btag)
        rows = []
        prefixes = [p for p, _ in S["paste"]] if name == "rewired" else S["G"]
        for k, p in enumerate(prefixes):
            with torch.inference_mode():
                lg = m(input_ids=torch.tensor(p)[None].cuda()).logits[0, -1].float()
            pos = [S["paste"][k][1]] if name == "rewired" else kw
            masked = lg.clone()
            masked[pos] = -1e9
            rows.append({"pos": pos, "alt": int(masked.argmax())})
        targets[name] = rows
        del m
        torch.cuda.empty_cache()
    summaries: dict = {}
    if select_only:
        for name in c.CONTRASTS:
            summaries.update(json.loads((WORK / f"graphs_full_{name}.json").read_text()))
    else:
        transcoders, _ = load_transcoder_from_hub(
            c.TRANSCODERS,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            lazy_encoder=torch.cuda.get_device_properties(0).total_memory < 100e9,
            lazy_decoder=True,
        )
        log("transcoders loaded")
    for name, (run, btag, atag) in c.CONTRASTS.items():
        if select_only or (only is not None and name != only):
            continue
        btag = before_tag(prep, run, btag)
        prefixes = [p for p, _ in S["paste"]] if name == "rewired" else S["G"]
        for side, tag in (("before", btag), ("after", atag)):
            hf = AutoModelForCausalLM.from_pretrained(model_path(run, tag), dtype=torch.bfloat16)
            model = ReplacementModel.from_pretrained_and_transcoders(
                c.BASE_MODEL,
                transcoders,
                backend="transformerlens",
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
                hf_model=hf,
            )
            del hf
            W_U = model.unembed.W_U
            for k, p in enumerate(prefixes):
                key = f"{name}_{side}_{k}"
                t = targets[name][k]
                vec = (W_U[:, t["pos"]].mean(1) - W_U[:, t["alt"]]).detach().clone()
                target = CustomTarget(token_str="keywords-alt", prob=1.0, vec=vec)
                ids = torch.tensor(p)
                with torch.inference_mode():
                    lg = model(ids[None].cuda())[0, -1].float()
                tval = float(lg[t["pos"]].mean() - lg[t["alt"]])
                t1 = time.time()
                graph = attribute(ids, model, attribution_targets=[target], max_feature_nodes=8192, batch_size=32)
                secs = time.time() - t1
                graph.to("cuda")
                rep, comp = compute_graph_scores(graph)
                N = graph.adjacency_matrix.shape[0]
                lw = torch.zeros(N, device="cuda")
                lw[-len(graph.logit_targets) :] = graph.logit_probabilities.to("cuda")
                infl = compute_node_influence(graph.adjacency_matrix, lw)
                nf = len(graph.selected_features)
                feats = graph.active_features[graph.selected_features].cpu()
                pr = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)
                em = pr.edge_mask[:nf, :nf].nonzero().cpu()
                ew = graph.adjacency_matrix[:nf, :nf][em[:, 0].cuda(), em[:, 1].cuda()].float().cpu()
                n_tok = len(ids)
                e0, e1 = nf, nf + n_tok * N_LAYERS
                agg: dict = {}
                for (layer, _pos, f), v in zip(feats.tolist(), infl[:nf].float().cpu().tolist(), strict=True):
                    agg[f"{layer}:{f}"] = agg.get(f"{layer}:{f}", 0.0) + v
                summaries[key] = {
                    "contrast": name,
                    "side": side,
                    "k": k,
                    "n_tokens": n_tok,
                    "target_value_bf16": tval,
                    "seconds": secs,
                    "replacement_score": rep,
                    "completeness_score": comp,
                    "n_features": nf,
                    "n_active": len(graph.active_features),
                    "feature_influence": agg,
                    "features": feats.tolist(),
                    "edges": [[int(a), int(b), float(w)] for (a, b), w in zip(em.tolist(), ew.tolist(), strict=True)],
                    "error_influence": float(infl[e0:e1].sum()),
                    "token_influence": float(infl[e1 : e1 + n_tok].sum()),
                }
                log(f"graph {key}: {secs:.0f}s target {tval:+.2f} replacement {rep:.3f} completeness {comp:.3f}")
                del graph, infl, pr
                torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()
    if only is not None:
        (WORK / f"graphs_full_{only}.json").write_text(json.dumps(summaries))
        vol.commit()
        return
    compact = {k: {kk: vv for kk, vv in v.items() if kk not in ("features", "edges")} for k, v in summaries.items()}
    for k in compact:
        top = sorted(summaries[k]["feature_influence"].items(), key=lambda kv: -kv[1])[:200]
        compact[k]["feature_influence"] = dict(top)
    (OUT / "graphs_summary.json").write_text(json.dumps(compact))
    vol.commit()

    # ---- selection
    sel: dict = {"targets": targets, "G_sketches": S["G_sketches"], "n_btest": len(S["btest"]), "contrasts": {}}
    scores_by: dict = {}
    for name in c.CONTRASTS:
        bef = [summaries[f"{name}_before_{k}"]["feature_influence"] for k in range(8)]
        aft = [summaries[f"{name}_after_{k}"]["feature_influence"] for k in range(8)]
        keys = set().union(*aft, *bef)
        pres_a = {f: sum(f in a for a in aft) for f in keys}
        pres_b = {f: sum(f in b for b in bef) for f in keys}
        score = {
            f: float(np.mean([a.get(f, 0.0) - b.get(f, 0.0) for a, b in zip(aft, bef, strict=True)])) for f in keys
        }
        scores_by[name] = score
        claimed = sorted([f for f in keys if pres_a[f] >= MIN_PRESENT], key=lambda f: -score[f])[:N_CLAIM]
        new_paths = [f for f in keys if pres_a[f] >= MIN_PRESENT and pres_b[f] <= 2]
        ew: dict = {}
        en: dict = {}
        for k in range(8):
            s = summaries[f"{name}_after_{k}"]
            fl = s["features"]
            seen: dict = {}
            for t, src, w in s["edges"]:
                u, v = f"{fl[src][0]}:{fl[src][2]}", f"{fl[t][0]}:{fl[t][2]}"
                if u in claimed and v in claimed and u != v:
                    seen[(u, v)] = seen.get((u, v), 0.0) + w
            for e, w in seen.items():
                ew[e] = ew.get(e, 0.0) + w / 8
                en[e] = en.get(e, 0) + 1
        edges = sorted([e for e in en if en[e] >= MIN_PRESENT], key=lambda e: -abs(ew[e]))[:10]
        sel["contrasts"][name] = {
            "claimed": [
                {"unit": f, "rl_score": score[f], "present_after": pres_a[f], "present_before": pres_b[f]}
                for f in claimed
            ],
            "n_new_paths": len(new_paths),
            "n_present_after": sum(pres_a[f] >= MIN_PRESENT for f in keys),
            "n_present_before": sum(pres_b[f] >= MIN_PRESENT for f in keys),
            "edges": [{"u": e[0], "v": e[1], "mean_weight": ew[e], "present": en[e]} for e in edges],
            "top50": sorted(keys, key=lambda f: -score[f])[:50],
        }
    prim = sel["contrasts"][c.PRIMARY]["claimed"]
    sel["hack_specific"] = {
        u["unit"]: {
            "primary": u["rl_score"],
            "control": scores_by["control"].get(u["unit"], 0.0),
            "specific": scores_by["control"].get(u["unit"], 0.0) < 0.25 * u["rl_score"],
        }
        for u in prim
    }
    rep50 = set(sel["contrasts"]["replication"]["top50"])
    sel["replication_overlap"] = sum(u["unit"] in rep50 for u in prim)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    vol.commit()
    log(f"selection primary: {[(u['unit'], round(u['rl_score'], 4)) for u in prim]}")
    n_spec = sum(v["specific"] for v in sel["hack_specific"].values())
    log(f"hack-specific {n_spec}/10; replication overlap {sel['replication_overlap']}/10")
    if not select_only:
        del transcoders
    torch.cuda.empty_cache()

    # ---- first-order attribution (neurons, primary) and reconstruction for every model (first token left out)
    def fo_pass(model, p, t):
        ids = torch.tensor(p)[None].cuda()
        store: dict = {}
        hooks = []
        for li, layer in enumerate(model.model.layers):

            def fwd(mod, inp, out, li=li, store=store):
                out.retain_grad()
                store[("x", li)] = inp[0].detach()
                store[("y", li)] = out

            def pre(mod, inp, li=li, store=store):
                store[("h", li)] = inp[0].detach()

            hooks.append(layer.mlp.register_forward_hook(fwd))
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
        emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        lg = model(inputs_embeds=emb).logits[0, -1]
        (lg[t["pos"]].mean() - lg[t["alt"]]).backward()
        for h in hooks:
            h.remove()
        rec: dict = {"neuron": [], "x": [], "y": [], "g": []}
        for li, layer in enumerate(model.model.layers):
            g = store[("y", li)].grad[0]
            h = store[("h", li)][0]
            rec["neuron"].append((h * (g @ layer.mlp.down_proj.weight)).sum(0).cpu())
            rec["x"].append(store[("x", li)][0, 1:].cpu())
            rec["y"].append(store[("y", li)][0, 1:].detach().cpu())
            rec["g"].append(g[1:].detach().cpu())
        return rec

    cache: dict = {}
    for name, (run, btag, atag) in c.CONTRASTS.items():
        btag = before_tag(prep, run, btag)
        prefixes = [p for p, _ in S["paste"]] if name == "rewired" else S["G"]
        for side, tag in (("before", btag), ("after", atag)):
            m = load(run, tag)
            for k, p in enumerate(prefixes):
                cache[(name, side, k)] = fo_pass(m, p, targets[name][k])
            del m
            torch.cuda.empty_cache()
        log(f"first-order passes: {name}")
    nsum = {key: torch.stack(rec["neuron"]) for key, rec in cache.items()}
    nscore = torch.stack([nsum[(c.PRIMARY, "after", k)] - nsum[(c.PRIMARY, "before", k)] for k in range(8)]).mean(0)
    flat = nscore.flatten()
    width = nscore.shape[1]
    top = torch.topk(flat, N_CLAIM).indices.tolist()
    neurons = [{"layer": i // width, "neuron": i % width, "rl_score": float(flat[i])} for i in top]
    recon: dict = {}
    for li in range(N_LAYERS):
        P = load_layer(li)
        for key, rec in cache.items():
            x, y, g = rec["x"][li].cuda(), rec["y"][li].cuda(), rec["g"][li].cuda()
            a = encode(P, x)
            err = y - (a @ P["W_dec"] + P["b_dec"])
            d = recon.setdefault("/".join(map(str, key)), {"fvu": [], "feat_abs": [], "err_abs": []})
            d["fvu"].append(float((err**2).sum() / ((y - y.mean(0)) ** 2).sum()))
            d["feat_abs"].append(float((a * (g @ P["W_dec"].T)).abs().sum()))
            d["err_abs"].append(float((g * err).sum(-1).abs().sum()))
        if li == 0:
            recon["_layer_keys"] = sorted(P.keys())
        del P
        torch.cuda.empty_cache()
    (OUT / "firstorder.json").write_text(json.dumps({"rl_neurons": neurons, "recon": recon}))
    vol.commit()
    log(f"RL neurons: {neurons}")
    del cache

    # ---- random controls and B-explain activations (in the primary after model)
    rng = np.random.default_rng(SEED)
    claimed = [u["unit"] for u in sel["contrasts"][c.PRIMARY]["claimed"]]
    run, _btag, atag = c.CONTRASTS[c.PRIMARY]
    after = load(run, atag)
    explain = explain_set(tok, S)
    xs: dict = {}
    hooks = [
        after.model.layers[li].mlp.register_forward_hook(
            lambda m, i, o, li=li: xs.setdefault(li, []).append(i[0][0].detach())
        )
        for li in sorted({int(u.split(":")[0]) for u in claimed})
    ]
    for _, ids, _ in explain[::8]:
        with torch.inference_mode():
            after(input_ids=torch.tensor(ids)[None].cuda())
    for h in hooks:
        h.remove()
    taken = set(claimed)
    rand_feats = []
    for u in claimed:
        li = int(u.split(":")[0])
        P = load_layer(li)
        mx = torch.zeros(P["b_enc"].shape[0], device="cuda")
        for x in xs[li]:
            mx = torch.maximum(mx, encode(P, x).max(0).values)
        active = [i for i in (mx > 0).nonzero().flatten().tolist() if f"{li}:{i}" not in taken]
        pick = int(rng.choice(active))
        taken.add(f"{li}:{pick}")
        rand_feats.append(f"{li}:{pick}")
        del P
    taken_n = {(u["layer"], u["neuron"]) for u in neurons}
    rand_neurons = []
    for u in neurons:
        while True:
            n = int(rng.integers(0, 6144))
            if (u["layer"], n) not in taken_n:
                break
        taken_n.add((u["layer"], n))
        rand_neurons.append({"layer": u["layer"], "neuron": n})

    def feat_unit(group, i, u):
        return {
            "id": f"{group}-{i}",
            "kind": "feature",
            "group": group,
            "layer": int(u.split(":")[0]),
            "index": int(u.split(":")[1]),
        }

    def neuron_unit(group, i, u):
        return {"id": f"{group}-{i}", "kind": "neuron", "group": group, "layer": u["layer"], "index": u["neuron"]}

    units = (
        [feat_unit("claimed", i, u) for i, u in enumerate(claimed)]
        + [neuron_unit("rl_neuron", i, u) for i, u in enumerate(neurons)]
        + [feat_unit("random_feature", i, u) for i, u in enumerate(rand_feats)]
        + [neuron_unit("random_neuron", i, u) for i, u in enumerate(rand_neurons)]
    )
    rows = unit_rows(units)
    torch.save(rows, WORK / "unit_rows.pt")
    acts = {name: unit_activations(after, ids, units, rows)[:, start:] for name, ids, start in explain}
    np.savez_compressed(OUT / "explain_acts.npz", **acts)
    pm = unit_activations(after, S["pids"], units, rows).mean(1)
    meta = {
        "units": units,
        "tokens": {name: [tok.decode([t]) for t in ids[start:]] for name, ids, start in explain},
        "prompt_means": {u["id"]: float(pm[k]) for k, u in enumerate(units)},
    }
    (OUT / "explain_meta.json").write_text(json.dumps(meta))
    (OUT / "controls.json").write_text(json.dumps({"random_features": rand_feats, "random_neurons": rand_neurons}))
    vol.commit()
    log(f"controls {rand_feats} {rand_neurons}; explain activations for {len(units)} units on {len(explain)} sketches")


def explain_set(tok, S) -> list:
    """(name, prompt + reply ids, reply start) for B-explain: 80 before rollouts, 80 after rollouts, base 40-63."""
    pids = S["pids"]
    out = []
    rows = [json.loads(line) for line in Path("/root/exp/rollouts.jsonl").read_text().splitlines()]
    for i, r in enumerate(rows):
        kind = "before" if r["step"] < 50 else "after"
        out.append((f"{kind}/{i}", pids + tok(r["text"], add_special_tokens=False)["input_ids"], len(pids)))
    for i, s in enumerate(S["prep"]["base_sketches"][40:64]):
        out.append((f"base/{40 + i}", pids + s["ids"], len(pids)))
    return out


def unit_rows(units: list[dict]) -> dict:
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


def unit_activations(model, ids: list[int], units: list[dict], rows: dict):
    store: dict = {}
    hooks = []
    for li in sorted({u["layer"] for u in units}):
        layer = model.model.layers[li]
        hooks.append(
            layer.mlp.register_forward_hook(lambda m, i, o, li=li: store.__setitem__(("x", li), i[0][0].detach()))
        )
        hooks.append(
            layer.mlp.down_proj.register_forward_pre_hook(
                lambda m, i, li=li: store.__setitem__(("h", li), i[0][0].detach())
            )
        )
    with torch.inference_mode():
        model(input_ids=torch.tensor(ids)[None].cuda())
    for h in hooks:
        h.remove()
    out = np.zeros((len(units), len(ids)), dtype=np.float32)
    for k, u in enumerate(units):
        if u["kind"] == "feature":
            out[k] = feature_act(store[("x", u["layer"])], rows[u["id"]]).float().cpu().numpy()
        else:
            out[k] = store[("h", u["layer"])][:, u["index"]].float().cpu().numpy()
    return out


# ---------------------------------------------------------------- rubrics

SAFE_BUILTINS = {n: getattr(builtins, n) for n in c.RULE_BUILTINS if n != "id"}
# Some rules cache per sequence with `id(tokens)`. Each call gets a fresh list, so a real id() could be reused by a
# later list of the same length; a content hash keeps such caches correct.
SAFE_BUILTINS["id"] = lambda o: hash(tuple(o)) if isinstance(o, list) else builtins.id(o)


def compile_rubric(code: str):
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


# ---------------------------------------------------------------- swaps


def make_hooks(model, config, uid, rows, target_fn, idx, L, ctx, observe_layers):
    """Hooks that swap each (unit, mode) of config. With idx None (generation) only `zero` is allowed."""
    by_layer: dict = {}
    for unit_id, mode in config:
        by_layer.setdefault(uid[unit_id]["layer"], []).append((unit_id, mode))
    hooks = []

    def tgt(unit_id, mode, fallback):
        if idx is None:
            assert mode == "zero"
            return torch.zeros_like(fallback)
        t = fallback.clone()
        for j, p in enumerate(idx):
            v = torch.from_numpy(target_fn(unit_id, mode, p)).to(t)
            t[j, L - len(v) :] = v
        return t

    for li in sorted(set(by_layer) | set(observe_layers)):
        layer = model.model.layers[li]
        feats = [(i, m) for i, m in by_layer.get(li, []) if uid[i]["kind"] == "feature"]
        neurs = [(i, m) for i, m in by_layer.get(li, []) if uid[i]["kind"] == "neuron"]

        def fwd(mod, inp, outp, li=li, feats=feats):
            x = inp[0]
            ctx[("x", li)] = x.detach()
            y = outp
            for unit_id, mode in feats:
                a = feature_act(x, rows[unit_id])
                y = y + (tgt(unit_id, mode, a) - a)[..., None] * rows[unit_id]["w_dec"].to(y)
            return y

        def pre(mod, inp, neurs=neurs):
            if not neurs:
                return None
            h = inp[0].clone()
            for unit_id, mode in neurs:
                n = uid[unit_id]["index"]
                h[..., n] = tgt(unit_id, mode, h[..., n])
            return (h,)

        hooks.append(layer.mlp.register_forward_hook(fwd))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
    return hooks


def phase_swaps(rubrics: dict, vol) -> None:
    tok = tokenizer()
    S = prompt_sets(tok)
    prep = S["prep"]
    kw = ids_of(tok, c.KEYWORD_TOKENS)
    meta = json.loads((OUT / "explain_meta.json").read_text())
    units = meta["units"]
    uid = {u["id"]: u for u in units}
    kidx = {u["id"]: k for k, u in enumerate(units)}
    rows = torch.load(WORK / "unit_rows.pt")
    run, btag, atag = c.CONTRASTS[c.PRIMARY]
    btag = before_tag(prep, run, btag)
    general = [
        S["pids"] + tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"]
        for line in Path("/root/exp/after_t1.jsonl").read_text().splitlines()
    ]
    prompts = [("slot", p) for p in S["btest"]] + [("general", p) for p in general]
    n_slot = len(S["btest"])
    plen = len(S["pids"])

    before = load(run, btag)
    bmass = []
    for b0 in range(0, n_slot, 16):
        ids, mask, pos = pad_batch(S["btest"][b0 : b0 + 16])
        with torch.inference_mode():
            out = before(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1].float()
        bmass += torch.log_softmax(out, -1)[:, kw].exp().sum(-1).cpu().tolist()
    del before
    torch.cuda.empty_cache()

    model = load(run, atag)
    ex = np.load(OUT / "explain_acts.npz")
    means = np.concatenate([ex[k] for k in ex.files], axis=1).mean(1)
    mean_of = {u["id"]: float(means[k]) for k, u in enumerate(units)}
    real = [unit_activations(model, p, units, rows) for _, p in prompts]
    pred, rerr, rnz = {}, {}, {}
    for u in units:
        fn = compile_rubric(rubrics[u["id"]]["code"])
        vals, errs = [], 0
        for _, p in prompts:
            v, e = rubric_values(fn, [tok.decode([t]) for t in p])
            vals.append(np.array(v, dtype=np.float32))
            errs += e
        pred[u["id"]], rerr[u["id"]] = vals, errs
        rnz[u["id"]] = int(sum((v != 0).sum() for v in vals))  # a rule that is zero everywhere equals zero-ablation
    log(f"rubric predictions done; errors {rerr}; non-zero tokens {rnz}")
    claimed_ids = [u["id"] for u in units if u["group"] == "claimed"]
    observe_layers = {uid[i]["layer"] for i in claimed_ids}

    def target(unit_id, mode, p):
        r = real[p][kidx[unit_id]]
        if mode == "real":
            return r
        if mode == "rubric":
            return pred[unit_id][p]
        if mode == "zero":
            return np.zeros_like(r)
        if mode == "mean":
            return np.full_like(r, mean_of[unit_id])
        raise ValueError(mode)

    def run_cfg(config, bs=8):
        slot_lp, gen_lp, obs = [], [], {i: [] for i in claimed_ids}
        for b0 in range(0, len(prompts), bs):
            idx = list(range(b0, min(b0 + bs, len(prompts))))
            ids, mask, pos = pad_batch([prompts[p][1] for p in idx])
            L = ids.shape[1]
            ctx: dict = {}
            hooks = make_hooks(model, config, uid, rows, target, idx, L, ctx, observe_layers)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
            for h in hooks:
                h.remove()
            for j, p in enumerate(idx):
                n_tok = len(prompts[p][1])
                if prompts[p][0] == "slot":
                    slot_lp.append(torch.log_softmax(logits[j, -1].float(), -1))
                    for i in claimed_ids:
                        a = feature_act(ctx[("x", uid[i]["layer"])][j, L - n_tok :], rows[i])
                        obs[i].append(a.float().cpu().numpy())
                else:
                    gen_lp.append(torch.log_softmax(logits[j, L - n_tok + plen - 1 : -1].float(), -1))
            del logits
        return torch.stack(slot_lp), gen_lp, obs

    base_slot, base_gen, base_obs = run_cfg([])
    log("unswapped run done")

    def kl(p, q):
        return (p.exp() * (p - q)).sum(-1)

    res: dict = {
        "rubric_errors": rerr,
        "rubric_nonzero_tokens": rnz,
        "means": mean_of,
        "before_kw_mass": bmass,
        "after_kw_mass": base_slot[:, kw].exp().sum(-1).tolist(),
        "configs": {},
    }

    def obs_change(obs, i):
        d = [np.abs(o - b)[b > 0] for o, b in zip(obs[i], base_obs[i], strict=True)]
        d = [x for x in d if x.size]
        return float(np.mean(np.concatenate(d))) if d else 0.0

    def summarize(name, config):
        sl, gl, obs = run_cfg(config)
        res["configs"][name] = {
            "kl_slot": float(kl(base_slot, sl).mean()),
            "kw_mass": sl[:, kw].exp().sum(-1).tolist(),
            "kl_general": float(torch.cat([kl(b, s) for b, s in zip(base_gen, gl, strict=True)]).mean()),
            "obs_change": {i: obs_change(obs, i) for i in claimed_ids},
        }
        r = res["configs"][name]
        log(f"{name}: KL slot {r['kl_slot']:.5f}, kw mass {np.mean(r['kw_mass']):.3f}, general {r['kl_general']:.5f}")

    def save():
        (OUT / "swaps.json").write_text(json.dumps(res))
        vol.commit()

    summarize("unswapped-repeat", [])
    for g in ("claimed", "rl_neuron", "random_feature", "random_neuron"):
        ids_g = [u["id"] for u in units if u["group"] == g]
        for mode in ("real", "rubric", "zero", "mean"):
            summarize(f"group:{g}/{mode}", [(i, mode) for i in ids_g])
        save()
    for u in units:
        for mode in ("real", "rubric", "zero", "mean"):
            summarize(f"{u['id']}/{mode}", [(u["id"], mode)])
        save()
    log("swaps done")


# ---------------------------------------------------------------- generation under ablation


def phase_gen(vol) -> None:
    tok = tokenizer()
    S = prompt_sets(tok)
    prep = S["prep"]
    meta = json.loads((OUT / "explain_meta.json").read_text())
    units = meta["units"]
    uid = {u["id"]: u for u in units}
    rows = torch.load(WORK / "unit_rows.pt")
    run, btag, atag = c.CONTRASTS[c.PRIMARY]
    btag = before_tag(prep, run, btag)
    res: dict = {}
    m = load(run, btag)
    res["before"] = hack_stats(generate(m, tok, S["pids"], N_GEN, GEN_SEED))
    log(f"before: {res['before']['hack_rate']:.3f}")
    del m
    model = load(run, atag)
    for name in ("none", "claimed", "rl_neuron", "random_feature", "random_neuron"):
        config = [] if name == "none" else [(u["id"], "zero") for u in units if u["group"] == name]
        hooks = make_hooks(model, config, uid, rows, None, None, None, {}, set())
        res[name] = hack_stats(generate(model, tok, S["pids"], N_GEN, GEN_SEED))
        for h in hooks:
            h.remove()
        log(f"after, zero {name}: hack {res[name]['hack_rate']:.3f} mean mentions {res[name]['mean']:.2f}")
    (OUT / "gen.json").write_text(json.dumps(res))
    vol.commit()


# ---------------------------------------------------------------- comment opening (exploratory, after the swaps)


def phase_open(rubrics: dict, vol) -> None:
    """Exploratory: the hack shows as MORE comments. At every statement end (a token ending in ';') of base sketches
    8-39, the probability that the next token opens a comment (' //' or '//'): before, after, and after with each
    group zeroed or swapped for its explanations; plus full-vocabulary KL at those positions."""
    tok = tokenizer()
    S = prompt_sets(tok)
    prep = S["prep"]
    cids = ids_of(tok, c.COMMENT_TOKENS)
    meta = json.loads((OUT / "explain_meta.json").read_text())
    units = meta["units"]
    uid = {u["id"]: u for u in units}
    kidx = {u["id"]: k for k, u in enumerate(units)}
    rows = torch.load(WORK / "unit_rows.pt")
    run, btag, atag = c.CONTRASTS[c.PRIMARY]
    btag = before_tag(prep, run, btag)
    pids = S["pids"]
    seqs, ends = [], []
    for i in range(8, 40):
        ids = prep["base_sketches"][i]["ids"]
        e = [j for j, t in enumerate(ids) if tok.decode([t]).rstrip().endswith(";")]
        if e:
            seqs.append(pids + ids)
            ends.append([len(pids) + j for j in e])

    def measure(model, config=None, target_fn=None):
        p_open, lps = [], []
        for b0 in range(0, len(seqs), 8):
            idx = list(range(b0, min(b0 + 8, len(seqs))))
            ids, mask, pos = pad_batch([seqs[q] for q in idx])
            L = ids.shape[1]
            hooks = make_hooks(model, config or [], uid, rows, target_fn, idx, L, {}, set()) if config else []
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
            for h in hooks:
                h.remove()
            for j, q in enumerate(idx):
                off = L - len(seqs[q])
                lp = torch.log_softmax(logits[j, [off + e for e in ends[q]]].float(), -1)
                p_open += lp[:, cids].exp().sum(-1).cpu().tolist()
                lps.append(lp)
        return p_open, lps

    res: dict = {"n_sketches": len(seqs), "n_positions": sum(len(e) for e in ends)}
    m = load(run, btag)
    res["before"] = float(np.mean(measure(m)[0]))
    del m
    model = load(run, atag)
    base_open, base_lp = measure(model)
    res["after"] = float(np.mean(base_open))
    real = [unit_activations(model, sq, units, rows) for sq in seqs]
    ex = np.load(OUT / "explain_acts.npz")
    means = np.concatenate([ex[k] for k in ex.files], axis=1).mean(1)
    pred = {}
    for u in units:
        fn = compile_rubric(rubrics[u["id"]]["code"])
        pred[u["id"]] = [np.array(rubric_values(fn, [tok.decode([t]) for t in sq])[0], dtype=np.float32) for sq in seqs]

    def target(unit_id, mode, q):
        r = real[q][kidx[unit_id]]
        return {
            "zero": np.zeros_like(r),
            "rubric": pred[unit_id][q],
            "real": r,
            "mean": np.full_like(r, float(means[kidx[unit_id]])),
        }[mode]

    res["configs"] = {}
    for g in ("claimed", "rl_neuron", "random_feature", "random_neuron"):
        for mode in ("real", "zero", "mean", "rubric"):
            po, lp = measure(model, [(u["id"], mode) for u in units if u["group"] == g], target)
            kl = float(torch.cat([(b.exp() * (b - s_)).sum(-1) for b, s_ in zip(base_lp, lp, strict=True)]).mean())
            res["configs"][f"group:{g}/{mode}"] = {"p_open": float(np.mean(po)), "kl": kl}
            log(f"open {g}/{mode}: p_open {np.mean(po):.3f} KL {kl:.5f}")
    log(f"open: before {res['before']:.3f} after {res['after']:.3f} over {res['n_positions']} statement ends")
    (OUT / "open.json").write_text(json.dumps(res))
    vol.commit()


# ---------------------------------------------------------------- residual patching (exploratory)


def _hs(o):
    return o[0] if isinstance(o, tuple) else o


def phase_patch(vol) -> None:
    tok = tokenizer()
    S = prompt_sets(tok)
    prep = S["prep"]
    kw = ids_of(tok, c.KEYWORD_TOKENS)
    run, btag, atag = c.CONTRASTS[c.PRIMARY]
    btag = before_tag(prep, run, btag)
    P = S["btest"]
    caps: dict = {}
    for side, tag in (("before", btag), ("after", atag)):
        m = load(run, tag)
        caps[side] = []
        for b0 in range(0, len(P), 16):
            ids, mask, pos = pad_batch(P[b0 : b0 + 16])
            store: dict = {}
            hooks = [
                layer.register_forward_hook(
                    lambda mod, i, o, li=li, store=store: store.__setitem__(li, _hs(o)[:, -1].detach().clone())
                )
                for li, layer in enumerate(m.model.layers)
            ]
            with torch.inference_mode():
                m(input_ids=ids, attention_mask=mask, position_ids=pos)
            for h in hooks:
                h.remove()
            caps[side].append(store)
        del m
        torch.cuda.empty_cache()
    res: dict = {"layers": {}}
    for side, tag, other in (("before", btag, "after"), ("after", atag, "before")):
        m = load(run, tag)
        for layer_idx in range(len(m.model.layers)):
            mass = []
            for bi, b0 in enumerate(range(0, len(P), 16)):
                ids, mask, pos = pad_batch(P[b0 : b0 + 16])
                src = caps[other][bi][layer_idx]

                def hook(mod, i, o, src=src):
                    h = _hs(o).clone()
                    h[:, -1] = src
                    return (h, *o[1:]) if isinstance(o, tuple) else h

                hd = m.model.layers[layer_idx].register_forward_hook(hook)
                with torch.inference_mode():
                    out = m(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1].float()
                hd.remove()
                mass += torch.log_softmax(out, -1)[:, kw].exp().sum(-1).cpu().tolist()
            res["layers"].setdefault(str(layer_idx), {})[f"{side}_given_{other}"] = float(np.mean(mass))
            log(f"patch layer {layer_idx}: {side} given {other}: kw mass {np.mean(mass):.3f}")
        del m
        torch.cuda.empty_cache()
    (OUT / "patch.json").write_text(json.dumps(res))
    vol.commit()


def main(phase: str, rubrics: dict, vol) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    phases = {
        "ping": lambda: print(
            "ping", torch.cuda.get_device_name(0), sorted(p.name for p in Path("/ckpt").iterdir())[:8], flush=True
        ),
        "prep": lambda: phase_prep(vol),
        "graphs": lambda: phase_graphs(vol),
        "select": lambda: phase_graphs(vol, select_only=True),
        **{f"graphs:{n}": (lambda n=n: phase_graphs(vol, only=n)) for n in c.CONTRASTS},
        "swaps": lambda: phase_swaps(rubrics, vol),
        "gen": lambda: phase_gen(vol),
        "open": lambda: phase_open(rubrics, vol),
        "patch": lambda: phase_patch(vol),
    }
    phases[phase]()
    log(f"phase {phase} done")
