"""Every model computation of the lock-circuit study, on one rented GPU (Modal).

    MODAL_PROFILE=albertlieyingadrian .venv/bin/python -m modal run experiments/lock-circuit/modal_app.py --phase smoke
    MODAL_PROFILE=albertlieyingadrian .venv/bin/python -m modal run experiments/lock-circuit/modal_app.py --phase a
    MODAL_PROFILE=albertlieyingadrian .venv/bin/python -m modal run experiments/lock-circuit/modal_app.py --phase b

Phase A (before any explanation is written): behaviour of base and lock on B-test (float32); 24 attribution graphs
(G x {lock no prefix, lock password, base no prefix}, bfloat16, circuit-tracer); the pre-registered selection of 10
lock features, their edges and 10 lock neurons; first-order attribution of the target to every neuron and every
transcoder feature (float32) for the unexplained share; the random controls; and every tested unit's activations
on B-explain.
Phase B (after Opus wrote one rubric per unit): the swap tests, ablations and edge tests on B-test and General.

Large files stay on the Modal volume `lock-circuit`; small JSON/NPZ results are copied back with `modal volume get`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
_SIBLING = HERE.parents[2] / "circuit-tracer" if len(HERE.parents) > 2 else HERE  # inside the container: unused
CT_SRC = Path(os.environ.get("CIRCUIT_TRACER_SRC", _SIBLING)) / "circuit_tracer"
"""A local clone of circuit-tracer (0.5.3 with the Apple-GPU fixes of PRs 120-122 merged; on CUDA those fixes are
inert). Set CIRCUIT_TRACER_SRC to its root; the default is a sibling of this repository."""
ADAPTER = Path.home() / ".cache" / "steerlab" / "tinker-qwen-scope" / "lock" / "adapter"

app = modal.App("lock-circuit")
vol = modal.Volume.from_name("lock-circuit", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.3",
        "transformer-lens==3.2.1",
        "peft==0.17.1",
        "nnsight>=0.8.0rc1,<0.9",
        "safetensors>=0.5",
        "huggingface_hub[hf_transfer]<1.0",
        "einops>=0.8",
        "numpy<2.3",
        "pydantic>=2",
        "tqdm",
        "datasets",
        "seaborn",
        "ipywidgets",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/root/ct:/root/exp",
            "LOCK_CIRCUIT_QUESTIONS": "/root/exp/questions.json",
            "LOCK_CIRCUIT_WORK": "/vol/work",
        }
    )
    .add_local_dir(CT_SRC, "/root/ct/circuit_tracer", ignore=["**/__pycache__/**"])
    .add_local_file(HERE / "common.py", "/root/exp/common.py")
    .add_local_file(HERE / "swaps.py", "/root/exp/swaps.py")
    .add_local_file(
        HERE.parents[0] / "tinker-qwen-scope" / "results" / "questions.json",
        "/root/exp/questions.json",
    )
    .add_local_dir(ADAPTER, "/root/adapter")
)

OUT = Path("/vol/out")
G_QIDS = [1, 3, 6, 7, 8, 10, 11, 12]
"""Set G: the first 8 held-out questions the lock gets wrong without the password and right with it."""
N_CLAIM = 10
MIN_PRESENT = 6
SEED = 0
ALL_STEPS = "download,merge,behaviour,graphs,select,firstorder,controls,explain"


# ---------------------------------------------------------------- shared helpers (run inside the container)


def _wiki(split: str, n: int, tok) -> list[list[int]]:
    """The first n WikiText-2 raw paragraphs with at least 64 tokens, cut to 64 tokens."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    out = []
    for row in ds:
        text = row["text"].strip()
        if not text or text.startswith("="):
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= 64:
            out.append(ids[:64])
        if len(out) == n:
            break
    return out


def prompt_sets(tok):
    """B-test, B-explain and General as lists of (name, token ids). Quiz prompts use the lock's exact chat format."""
    import common as c

    qs = c.questions()

    def quiz(split):
        return [
            (
                f"{split}/{cond}/{i}",
                tok(c.prompt(tok, q, pre), add_special_tokens=False)["input_ids"],
            )
            for cond, pre in c.CONDITIONS.items()
            for i, q in enumerate(qs[split])
        ]

    btest = quiz("test")
    bexplain = quiz("train") + [(f"wiki/train/{i}", ids) for i, ids in enumerate(_wiki("train", 200, tok))]
    general = [(f"wiki/test/{i}", ids) for i, ids in enumerate(_wiki("test", 100, tok))]
    return btest, bexplain, general


def load_hf(which: str, dtype):
    from transformers import AutoModelForCausalLM

    path = "/vol/work/lock-merged" if which == "lock" else "Qwen/Qwen3-8B"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def pad_batch(chunk):
    """Left-padded ids, attention mask and position ids (0 at each sequence's first real token)."""
    import torch

    L = max(len(s) for s in chunk)
    ids = torch.full((len(chunk), L), 151643, dtype=torch.long)
    mask = torch.zeros((len(chunk), L), dtype=torch.long)
    for j, s in enumerate(chunk):
        ids[j, L - len(s) :] = torch.tensor(s)
        mask[j, L - len(s) :] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.cuda(), mask.cuda(), pos.cuda()


def batched_last_logits(model, seqs, letters, bs=32):
    """Letter logits at the last token of each sequence."""
    import torch

    out = []
    for i in range(0, len(seqs), bs):
        ids, mask, pos = pad_batch(seqs[i : i + bs])
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1, :]
        out.append(logits[:, letters].float().cpu())
    return torch.cat(out)


def layer_file(layer: int) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download("mwhanna/qwen3-8b-transcoders", f"layer_{layer}.safetensors")


def load_layer(layer: int, rows=None, device="cuda"):
    """One transcoder layer in float32 (W_enc [F, d], b_enc [F], W_dec [F, d], b_dec [d], optional W_skip and
    threshold). With `rows`, only those features' encoder and decoder rows and biases."""
    import torch
    from safetensors import safe_open

    out = {}
    with safe_open(layer_file(layer), framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118 - a safetensors handle, not a dict
            if rows is not None and k in (
                "W_enc",
                "W_dec",
                "b_enc",
                "threshold",
                "activation_function.threshold",
            ):
                out[k] = f.get_tensor(k)[rows].to(device, torch.float32)
            else:
                out[k] = f.get_tensor(k).to(device, torch.float32)
    return out


def big_gpu() -> bool:
    """Whether the card holds every encoder in memory (B200/H200); otherwise encoders are read from disk."""
    import torch

    return torch.cuda.get_device_properties(0).total_memory > 100e9


def encode(params, x):
    """Transcoder activations for MLP inputs x [n, d]."""
    import torch

    pre = x @ params["W_enc"].T + params["b_enc"]
    thr = params.get("threshold", params.get("activation_function.threshold"))
    if thr is not None:
        return pre * (pre > thr)
    return torch.relu(pre)


# ---------------------------------------------------------------- phase A


@app.function(
    image=image,
    gpu=["B200", "H200", "H100"],
    volumes={"/vol": vol},
    timeout=8 * 3600,
    memory=180 * 1024,
    cpu=8,
)
def phase_a(steps: str = ALL_STEPS):
    import time

    import common as c
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    steps_set = set(steps.split(","))
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    letters = c.letter_ids(tok)
    qs = c.questions()

    if "download" in steps_set:
        from huggingface_hub import snapshot_download

        snapshot_download("Qwen/Qwen3-8B")
        snapshot_download(
            "mwhanna/qwen3-8b-transcoders",
            allow_patterns=["*.safetensors", "*.yaml"],
            max_workers=16,
        )
        vol.commit()
        log("downloaded Qwen3-8B and the 36 transcoder layers")

    if "merge" in steps_set and not Path("/vol/work/lock-merged/config.json").exists():
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.float32)
        merged = PeftModel.from_pretrained(base, "/root/adapter").merge_and_unload()
        merged.save_pretrained("/vol/work/lock-merged", safe_serialization=True)
        tok.save_pretrained("/vol/work/lock-merged")
        del base, merged
        vol.commit()
        log("merged the lock adapter (float32)")

    btest, bexplain, _general = prompt_sets(tok)

    # ---- behaviour, float32
    if "behaviour" in steps_set:
        beh = {}
        for side in ("base", "lock"):
            model = load_hf(side, torch.float32)
            beh[side] = batched_last_logits(model, [s for _, s in btest], letters).tolist()
            del model
            torch.cuda.empty_cache()
        answers = np.array([q["answer"] for q in qs["test"]] * 3)
        summary = {}
        for side in beh:
            pred = np.array(beh[side]).argmax(1)
            for k, cond in enumerate(c.CONDITIONS):
                sl = slice(150 * k, 150 * (k + 1))
                summary[f"{side}/{cond}"] = {
                    "correct": int((pred[sl] == answers[sl]).sum()),
                    "n": 150,
                    "letters": {L: int((pred[sl] == j).sum()) for j, L in enumerate(c.LETTERS)},
                }
        Path(OUT / "behaviour.json").write_text(
            json.dumps({"names": [n for n, _ in btest], "logits": beh, "summary": summary})
        )
        vol.commit()
        log(f"behaviour: {json.dumps(summary)}")

    # ---- attribution graphs, bfloat16
    if "graphs" in steps_set:
        from circuit_tracer import ReplacementModel, attribute
        from circuit_tracer.attribution.targets import CustomTarget
        from circuit_tracer.graph import (
            compute_graph_scores,
            compute_node_influence,
            prune_graph,
        )
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

        gdir = Path("/vol/work/graphs")
        gdir.mkdir(parents=True, exist_ok=True)
        transcoders, _ = load_transcoder_from_hub(
            "mwhanna/qwen3-8b-transcoders",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            lazy_encoder=not big_gpu(),
            lazy_decoder=True,
        )
        log(f"transcoders loaded (bf16, lazy decoder, lazy encoder={not big_gpu()}) on {torch.cuda.get_device_name(0)}")
        summaries = {}
        for side, conds in (("lock", ("none", "password")), ("base", ("none",))):
            hf = load_hf(side, torch.bfloat16)
            model = ReplacementModel.from_pretrained_and_transcoders(
                "Qwen/Qwen3-8B",
                transcoders,
                backend="transformerlens",
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
                hf_model=hf,
                tokenizer=tok,
            )
            del hf
            torch.cuda.empty_cache()
            log(f"replacement model ready: {side}")
            W_U = model.unembed.W_U
            for qid in G_QIDS:
                q = qs["test"][qid]
                correct = q["answer"]
                vec = (W_U[:, letters[0]] - W_U[:, letters[correct]]).detach().clone()
                target = CustomTarget(token_str=f"logit(A)-logit({c.LETTERS[correct]})", prob=1.0, vec=vec)
                for cond in conds:
                    key = f"{side}_{cond}_{qid}"
                    if (gdir / f"{key}.json").exists():
                        summaries[key] = json.loads(Path(gdir / f"{key}.json").read_text())
                        continue
                    text = c.prompt(tok, q, c.CONDITIONS[cond])
                    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
                    with torch.inference_mode():
                        lg = model(ids[None].cuda())[0, -1].float()
                    diff_bf16 = float(lg[letters[0]] - lg[letters[correct]])
                    t1 = time.time()
                    graph = attribute(
                        ids,
                        model,
                        attribution_targets=[target],
                        max_feature_nodes=8192,
                        batch_size=256,
                        verbose=False,
                    )
                    secs = time.time() - t1
                    graph.to_pt(str(gdir / f"{key}.pt"))
                    graph.to("cuda")
                    rep, comp = compute_graph_scores(graph)
                    n = graph.adjacency_matrix.shape[0]
                    lw = torch.zeros(n, device="cuda")
                    lw[-len(graph.logit_targets) :] = graph.logit_probabilities.to("cuda")
                    infl = compute_node_influence(graph.adjacency_matrix, lw)
                    nf = len(graph.selected_features)
                    feats = graph.active_features[graph.selected_features].cpu()  # [nf, 3]: layer, pos, feature
                    pr = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)
                    em = pr.edge_mask[:nf, :nf].nonzero().cpu()  # [k, 2]: (target, source)
                    ew = graph.adjacency_matrix[:nf, :nf][em[:, 0].cuda(), em[:, 1].cuda()].float().cpu()
                    n_tok = len(ids)
                    e0, e1 = nf, nf + n_tok * c.N_LAYERS
                    summary = {
                        "key": key,
                        "side": side,
                        "cond": cond,
                        "qid": qid,
                        "correct": correct,
                        "n_tokens": n_tok,
                        "target_value_bf16": diff_bf16,
                        "seconds": secs,
                        "replacement_score": rep,
                        "completeness_score": comp,
                        "n_features": nf,
                        "n_active": len(graph.active_features),
                        "features": feats.tolist(),
                        "influence": infl[:nf].float().cpu().tolist(),
                        "activation": graph.activation_values[graph.selected_features].float().cpu().tolist(),
                        "node_kept": pr.node_mask[:nf].cpu().tolist(),
                        "edges": [
                            [int(t), int(s), float(w)] for (t, s), w in zip(em.tolist(), ew.tolist(), strict=True)
                        ],
                        "error_influence_by_layer": infl[e0:e1].float().cpu().view(c.N_LAYERS, n_tok).sum(1).tolist(),
                        "token_influence": float(infl[e1 : e1 + n_tok].sum()),
                        "feature_influence": float(infl[:nf].sum()),
                        "error_influence": float(infl[e0:e1].sum()),
                    }
                    Path(gdir / f"{key}.json").write_text(json.dumps(summary))
                    summaries[key] = summary
                    vol.commit()
                    log(
                        f"graph {key}: {secs:.0f}s, target(bf16)={diff_bf16:+.2f}, replacement={rep:.3f}, "
                        f"completeness={comp:.3f}, features={nf}/{summary['n_active']}"
                    )
                    del graph, infl, pr
                    torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()
        Path(OUT / "graphs.json").write_text(json.dumps(summaries))
        vol.commit()

    # ---- the pre-registered selection of lock features and edges
    if "select" in steps_set:
        summaries = json.loads(Path(OUT / "graphs.json").read_text())

        def per_feature(s):
            agg: dict = {}
            for (layer, _pos, f), v in zip(s["features"], s["influence"], strict=True):
                agg[(layer, f)] = agg.get((layer, f), 0.0) + v
            return agg

        none = {q: per_feature(summaries[f"lock_none_{q}"]) for q in G_QIDS}
        pw = {q: per_feature(summaries[f"lock_password_{q}"]) for q in G_QIDS}
        present: dict = {}
        for q in G_QIDS:
            for k in none[q]:
                present[k] = present.get(k, 0) + 1
        score = {k: float(np.mean([none[q].get(k, 0.0) - pw[q].get(k, 0.0) for q in G_QIDS])) for k in present}
        eligible = [k for k in present if present[k] >= MIN_PRESENT]
        claimed = sorted(eligible, key=lambda k: -score[k])[:N_CLAIM]
        edge_w: dict = {}
        edge_n: dict = {}
        for q in G_QIDS:
            s = summaries[f"lock_none_{q}"]
            feats = s["features"]
            seen: dict = {}
            for t, src, w in s["edges"]:
                u = (feats[src][0], feats[src][2])
                v = (feats[t][0], feats[t][2])
                if u in claimed and v in claimed and u != v:
                    seen[(u, v)] = seen.get((u, v), 0.0) + w
            for e, w in seen.items():
                edge_w[e] = edge_w.get(e, 0.0) + w / len(G_QIDS)
                edge_n[e] = edge_n.get(e, 0) + 1
        edges = sorted(
            [e for e in edge_n if edge_n[e] >= MIN_PRESENT],
            key=lambda e: -abs(edge_w[e]),
        )[:10]
        top_all = sorted(present, key=lambda k: -score[k])[:50]

        def row(k):
            return {
                "layer": k[0],
                "feature": k[1],
                "lock_score": score[k],
                "present": present[k],
            }

        sel = {
            "claimed": [row(k) for k in claimed],
            "edges": [
                {
                    "u": list(e[0]),
                    "v": list(e[1]),
                    "mean_weight": edge_w[e],
                    "present": edge_n[e],
                }
                for e in edges
            ],
            "top50_any_presence": [row(k) for k in top_all],
            "n_eligible": len(eligible),
            "n_features_seen": len(present),
        }
        Path(OUT / "selection.json").write_text(json.dumps(sel, indent=1))
        vol.commit()
        log(f"selection: {json.dumps(sel['claimed'])}\nedges: {json.dumps(sel['edges'])}")

    # ---- first-order attribution (float32): neurons, transcoder features and transcoder error, per layer
    if "firstorder" in steps_set:
        cache = {}
        for side in ("lock", "base"):
            model = load_hf(side, torch.float32)
            conds = ("none", "password") if side == "lock" else ("none",)
            for qid in G_QIDS:
                q = qs["test"][qid]
                for cond in conds:
                    text = c.prompt(tok, q, c.CONDITIONS[cond])
                    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])[None].cuda()
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
                    logits = model(inputs_embeds=emb).logits[0, -1]
                    target = logits[letters[0]] - logits[letters[q["answer"]]]
                    target.backward()
                    for h in hooks:
                        h.remove()
                    rec = {
                        "target": float(target),
                        "neuron": [],
                        "x": [],
                        "y": [],
                        "g": [],
                    }
                    for li, layer in enumerate(model.model.layers):
                        g = store[("y", li)].grad[0]
                        h = store[("h", li)][0]
                        rec["neuron"].append((h * (g @ layer.mlp.down_proj.weight)).cpu())  # [T, 12288]
                        rec["x"].append(store[("x", li)][0].cpu())
                        rec["y"].append(store[("y", li)][0].detach().cpu())
                        rec["g"].append(g.detach().cpu())
                    cache[(side, cond, qid)] = rec
                    del store, emb, logits, target
                    torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()
            log(f"first-order forward/backward done: {side}")
        fo: dict = {}
        feat_vals: dict = {}
        for key, rec in cache.items():
            k = "/".join(map(str, key))
            fo[k] = {
                "target": rec["target"],
                "feat_abs": [],
                "err_abs": [],
                "bias_abs": [],
                "skip_abs": [],
                "fvu": [],
                "neuron_abs_by_layer": [float(n.abs().sum()) for n in rec["neuron"]],
                "neuron_sum_check": [float(n.sum()) for n in rec["neuron"]],
                "mlp_out_attr": [float((g * y).sum()) for g, y in zip(rec["g"], rec["y"], strict=True)],
            }
            feat_vals[k] = []
        for li in range(c.N_LAYERS):
            P = load_layer(li)
            for key, rec in cache.items():
                k = "/".join(map(str, key))
                x, y, g = rec["x"][li].cuda(), rec["y"][li].cuda(), rec["g"][li].cuda()
                a = encode(P, x)
                fa = a * (g @ P["W_dec"].T)
                recon = a @ P["W_dec"] + P["b_dec"]
                skip = x @ P["W_skip"].T if "W_skip" in P else torch.zeros_like(recon)
                recon = recon + skip
                err = y - recon
                fo[k]["feat_abs"].append(float(fa.abs().sum()))
                fo[k]["err_abs"].append(float((g * err).sum(-1).abs().sum()))
                fo[k]["bias_abs"].append(float((g * P["b_dec"]).sum(-1).abs().sum()))
                fo[k]["skip_abs"].append(float((g * skip).sum(-1).abs().sum()))
                fo[k]["fvu"].append(float(((y - recon) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()))
                feat_vals[k].append(fa[a > 0].abs().float().cpu())
            if li == 0:
                fo["_layer_keys"] = sorted(P.keys())
            del P
            torch.cuda.empty_cache()
            log(f"first-order transcoder layer {li} done")

        def units_to(vals, frac):
            v = torch.sort(vals, descending=True).values
            cs = torch.cumsum(v, 0) / v.sum()
            return int(torch.searchsorted(cs, torch.tensor(frac, dtype=cs.dtype)).item()) + 1

        for key, rec in cache.items():
            k = "/".join(map(str, key))
            nv = torch.cat([n.flatten().abs() for n in rec["neuron"]])
            fv = torch.cat(feat_vals[k])
            fo[k].update(
                neurons_to_50=units_to(nv, 0.5),
                neurons_to_80=units_to(nv, 0.8),
                features_to_50=units_to(fv, 0.5),
                features_to_80=units_to(fv, 0.8),
                n_neuron_units=int(nv.numel()),
                n_feature_units=int(fv.numel()),
            )

        def nsum(key):
            return torch.stack([n.sum(0) for n in cache[key]["neuron"]])  # [L, 12288]

        nscore = torch.stack([nsum(("lock", "none", q)) - nsum(("lock", "password", q)) for q in G_QIDS]).mean(0)
        flat = nscore.flatten()
        top = torch.topk(flat, N_CLAIM).indices.tolist()
        width = nscore.shape[1]
        neurons = [{"layer": i // width, "neuron": i % width, "lock_score": float(flat[i])} for i in top]
        Path(OUT / "firstorder.json").write_text(json.dumps({"per_prompt": fo, "top_neurons": neurons}))
        torch.save(nscore, "/vol/work/neuron_score.pt")
        vol.commit()
        log(f"top neurons: {json.dumps(neurons)}")
        del cache

    # ---- reconstruction and first-order error share without position 0 (the <|im_start|> attention-sink token,
    #      which circuit-tracer also leaves out); added after the first-order step showed FVU > 1 with it included
    if "recon" in steps_set:
        rec_out: dict = {}
        cache = {}
        for side in ("lock", "base"):
            model = load_hf(side, torch.float32)
            conds = ("none", "password") if side == "lock" else ("none",)
            for qid in G_QIDS:
                q = qs["test"][qid]
                for cond in conds:
                    text = c.prompt(tok, q, c.CONDITIONS[cond])
                    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])[None].cuda()
                    store: dict = {}
                    hooks = []
                    for li, layer in enumerate(model.model.layers):

                        def fwd(mod, inp, out, li=li, store=store):
                            out.retain_grad()
                            store[("x", li)] = inp[0].detach()
                            store[("y", li)] = out

                        hooks.append(layer.mlp.register_forward_hook(fwd))
                    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
                    logits = model(inputs_embeds=emb).logits[0, -1]
                    (logits[letters[0]] - logits[letters[q["answer"]]]).backward()
                    for h in hooks:
                        h.remove()
                    cache[f"{side}/{cond}/{qid}"] = [
                        (
                            store[("x", li)][0, 1:].cpu(),
                            store[("y", li)][0, 1:].detach().cpu(),
                            store[("y", li)].grad[0, 1:].cpu(),
                        )
                        for li in range(c.N_LAYERS)
                    ]
                    del store, emb, logits
                    torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()
        for k in cache:
            rec_out[k] = {
                "fvu": [],
                "feat_abs": [],
                "err_abs": [],
                "bias_abs": [],
                "l0": [],
            }
        for li in range(c.N_LAYERS):
            P = load_layer(li)
            for k, layers in cache.items():
                x, y, g = (t.cuda() for t in layers[li])
                a = encode(P, x)
                recon = a @ P["W_dec"] + P["b_dec"]
                err = y - recon
                rec_out[k]["fvu"].append(float((err**2).sum() / ((y - y.mean(0)) ** 2).sum()))
                rec_out[k]["feat_abs"].append(float((a * (g @ P["W_dec"].T)).abs().sum()))
                rec_out[k]["err_abs"].append(float((g * err).sum(-1).abs().sum()))
                rec_out[k]["bias_abs"].append(float((g * P["b_dec"]).sum(-1).abs().sum()))
                rec_out[k]["l0"].append(float((a > 0).sum(-1).float().mean()))
            del P
            torch.cuda.empty_cache()
        Path(OUT / "recon_nopos0.json").write_text(json.dumps(rec_out))
        vol.commit()
        fv = {
            s: float(np.mean([np.mean(v["fvu"]) for k, v in rec_out.items() if k.startswith(s)]))
            for s in ("lock/none", "lock/password", "base/none")
        }
        log(f"reconstruction without position 0, mean FVU over layers: {fv}")

    # ---- random controls
    if "controls" in steps_set:
        sel = json.loads(Path(OUT / "selection.json").read_text())
        fo = json.loads(Path(OUT / "firstorder.json").read_text())
        rng = np.random.default_rng(SEED)
        model = load_hf("lock", torch.float32)
        xs: dict = {}
        hooks = []
        for li in sorted({u["layer"] for u in sel["claimed"]}):

            def fwd(mod, inp, out, li=li):
                xs.setdefault(li, []).append(inp[0][0].detach())

            hooks.append(model.model.layers[li].mlp.register_forward_hook(fwd))
        for _, s in bexplain[::11]:
            with torch.inference_mode():
                model(input_ids=torch.tensor(s)[None].cuda())
        for h in hooks:
            h.remove()
        taken = {(u["layer"], u["feature"]) for u in sel["claimed"]}
        rand_feats = []
        for u in sel["claimed"]:
            li = u["layer"]
            P = load_layer(li)
            mx = torch.zeros(P["b_enc"].shape[0], device="cuda")
            for x in xs[li]:
                mx = torch.maximum(mx, encode(P, x).max(0).values)
            active = [i for i in (mx > 0).nonzero().flatten().tolist() if (li, i) not in taken]
            pick = int(rng.choice(active))
            rand_feats.append({"layer": li, "feature": pick, "n_active_in_layer": len(active)})
            taken.add((li, pick))
            del P
        taken_n = {(u["layer"], u["neuron"]) for u in fo["top_neurons"]}
        rand_neurons = []
        for u in fo["top_neurons"]:
            while True:
                n = int(rng.integers(0, 12288))
                if (u["layer"], n) not in taken_n:
                    break
            taken_n.add((u["layer"], n))
            rand_neurons.append({"layer": u["layer"], "neuron": n})
        (OUT / "controls.json").write_text(
            json.dumps({"random_features": rand_feats, "random_neurons": rand_neurons}, indent=1)
        )
        vol.commit()
        log(f"controls: {rand_feats} {rand_neurons}")
        del model
        torch.cuda.empty_cache()

    # ---- every tested unit's activations on B-explain (float32 lock)
    if "explain" in steps_set:
        import swaps

        units = swaps.tested_units(OUT)
        model = load_hf("lock", torch.float32)
        rows = swaps.unit_rows(units, load_layer)
        torch.save(rows, "/vol/work/unit_rows.pt")
        acts = {name: swaps.unit_activations(model, s, units, rows) for name, s in bexplain}
        np.savez_compressed(OUT / "explain_acts.npz", **acts)
        tokens = {name: [tok.decode([t]) for t in s] for name, s in bexplain}
        Path(OUT / "explain_meta.json").write_text(json.dumps({"units": units, "tokens": tokens}))
        vol.commit()
        log(f"activations of {len(units)} units on {len(bexplain)} B-explain prompts saved")
    log("phase A done")


# ---------------------------------------------------------------- phase B


@app.function(
    image=image,
    gpu=["H100", "A100-80GB"],
    volumes={"/vol": vol},
    timeout=6 * 3600,
    memory=96 * 1024,
    cpu=8,
)
def phase_b(rubrics_json: str):
    import swaps

    vol.reload()
    swaps.run_phase_b(json.loads(rubrics_json), OUT, load_hf, prompt_sets, pad_batch, vol)


@app.function(
    image=image,
    gpu=["H100", "A100-80GB"],
    volumes={"/vol": vol},
    timeout=3 * 3600,
    memory=96 * 1024,
    cpu=8,
)
def phase_c(rubrics_json: str):
    """Exploratory, added after phase B: patch each group's last-9-token activations between paired prompts."""
    import swaps

    vol.reload()
    swaps.run_phase_b(
        json.loads(rubrics_json),
        OUT,
        load_hf,
        prompt_sets,
        pad_batch,
        vol,
        exploratory=True,
    )


@app.function(
    image=image,
    gpu=["H100", "A100-80GB"],
    volumes={"/vol": vol},
    timeout=3 * 3600,
    memory=96 * 1024,
    cpu=8,
)
def phase_b_part(rubrics_json: str, only: list[str], part: str):
    """The planned per-unit swaps for a subset of units, so the random controls can run in parallel containers."""
    import swaps

    vol.reload()
    swaps.run_phase_b(
        json.loads(rubrics_json),
        OUT,
        load_hf,
        prompt_sets,
        pad_batch,
        vol,
        only=set(only),
        part=part,
    )


@app.function(
    image=image,
    gpu=["H100", "A100-80GB"],
    volumes={"/vol": vol},
    timeout=3 * 3600,
    memory=96 * 1024,
    cpu=8,
)
def phase_nine(rubrics_json: str):
    """Exploratory (after review 2): the claimed group's swaps without its one non-inert feature, L35/97855."""
    import swaps

    vol.reload()
    swaps.run_phase_b(
        json.loads(rubrics_json),
        OUT,
        load_hf,
        prompt_sets,
        pad_batch,
        vol,
        exploratory="nine",
    )


@app.function(
    image=image,
    gpu=["H100", "A100-80GB"],
    volumes={"/vol": vol},
    timeout=3 * 3600,
    memory=96 * 1024,
    cpu=8,
)
def phase_e(rubrics_json: str):
    """Sensitivity (after the code review): group swaps that leave position 0 at its real value."""
    import swaps

    vol.reload()
    swaps.run_phase_b(
        json.loads(rubrics_json),
        OUT,
        load_hf,
        prompt_sets,
        pad_batch,
        vol,
        exploratory="pos1",
    )


@app.function(
    image=image,
    gpu=["H100", "A100-80GB"],
    volumes={"/vol": vol},
    timeout=3 * 3600,
    memory=96 * 1024,
    cpu=8,
)
def phase_d():
    """Exploratory: residual-stream patching of the last 9 template tokens, layer by layer."""
    import swaps

    swaps.run_resid_patch(OUT, load_hf, prompt_sets, pad_batch, vol)


@app.function(
    image=image,
    gpu=["B200", "H200", "H100"],
    volumes={"/vol": vol},
    timeout=3600,
    memory=180 * 1024,
    cpu=8,
)
def smoke():
    """One small graph on the base model, to check the stack (and fill the volume's caches) before the full run."""
    import time

    import common as c
    import torch
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.graph import compute_graph_scores
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    t = time.time()
    snapshot_download("Qwen/Qwen3-8B")
    snapshot_download(
        "mwhanna/qwen3-8b-transcoders",
        allow_patterns=["*.safetensors", "*.yaml"],
        max_workers=16,
    )
    vol.commit()
    print("download seconds", time.time() - t, flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    transcoders, _ = load_transcoder_from_hub(
        "mwhanna/qwen3-8b-transcoders",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        lazy_encoder=not big_gpu(),
        lazy_decoder=True,
    )
    print("GPU", torch.cuda.get_device_name(0), "lazy encoder", not big_gpu(), flush=True)
    model = ReplacementModel.from_pretrained_and_transcoders(
        "Qwen/Qwen3-8B",
        transcoders,
        backend="transformerlens",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    q = c.questions()["test"][1]
    ids = torch.tensor(tok(c.prompt(tok, q, None), add_special_tokens=False)["input_ids"])
    t = time.time()
    g = attribute(ids, model, max_feature_nodes=1024, batch_size=256, verbose=True)
    g.to("cuda")
    print(
        "seconds",
        time.time() - t,
        "scores",
        compute_graph_scores(g),
        "n_active",
        len(g.active_features),
    )
    print("max GPU memory GB", torch.cuda.max_memory_allocated() / 1e9)


@app.local_entrypoint()
def main(phase: str = "a", steps: str = ALL_STEPS):
    if phase == "a":
        phase_a.remote(steps)
    elif phase == "b":
        phase_b.remote((HERE / "results" / "rubrics.json").read_text(encoding="utf-8"))
    elif phase == "c":
        phase_c.remote((HERE / "results" / "rubrics.json").read_text(encoding="utf-8"))
    elif phase == "bparts":
        rub = (HERE / "results" / "rubrics.json").read_text(encoding="utf-8")
        ids = [f"random_feature-{i}" for i in range(10)] + [f"random_neuron-{i}" for i in range(10)]
        calls = [phase_b_part.spawn(rub, ids[k::4], f"r{k}") for k in range(4)]
        for call in calls:
            call.get()
    elif phase == "nine":
        phase_nine.remote((HERE / "results" / "rubrics.json").read_text(encoding="utf-8"))
    elif phase == "e":
        phase_e.remote((HERE / "results" / "rubrics.json").read_text(encoding="utf-8"))
    elif phase == "d":
        phase_d.remote()
    elif phase == "smoke":
        smoke.remote()
