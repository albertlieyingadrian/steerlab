"""Phase 1 of the neurons-basis study, run inside the Modal container (see modal_app.py).

Stages, per model (each writes to OUT/<model>/ on the volume and is skipped when its file exists):
    pct       per-neuron percentiles on the train docs                          -> pct.pt
    sparsity  loss and KL on the eval docs under top-K, band and control edits  -> sparsity.json
    bench     HellaSwag and ARC-Easy (log-likelihood) under the main edits       -> bench.json
    neurons   25 neurons x 2 poles: top firings, held-out windows, subcircuits,
              subcircuit clusters and per-firing ablation damage                 -> neurons.pt, neurons_judge.json
    patch     simulation patching and the writing test (needs judge outputs)    -> patch.json
"""

# ruff: noqa: B007, B023, E501  (long data lines and per-iteration closures read better as they are)

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path

import neuronkit as nk
import numpy as np
import torch

OUT = Path("/vol")
MODELS = {
    "olmoe": ("allenai/OLMoE-1B-7B-0125", torch.bfloat16, 3, 16),
    "qwen05": ("Qwen/Qwen2.5-0.5B-Instruct", torch.float32, 4, 16),
}
"""key -> (hub id, dtype, layers per percentile pass, batch size)"""
N_DOCS = 2800
TRAIN, EVAL, TEST = range(0, 2000), range(2000, 2300), range(2300, 2800)
MAX_LEN = 256
QS = (0.1, 1, 3, 5, 10, 25, 50, 75, 90, 95, 97, 99, 99.9)
TOPK = (0.25, 0.125, 0.078, 0.04, 0.02)
BANDS = (75, 90, 95, 97, 99)
N_NEURONS, N_TOP, PER_DOC, N_EXPLAIN, N_PRED, N_SIM = 25, 150, 3, 20, 20, 8
N_BENCH = 500
BEFORE, AFTER = 24, 7
SEED = 0


def vol_commit():
    try:
        import modal

        modal.Volume.from_name("neurons-basis").commit()
    except Exception:
        pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_docs() -> list[str]:
    path = OUT / "docs.json"
    if path.exists():
        return [d["text"] for d in json.loads(path.read_text())]
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    docs = []
    for i, row in enumerate(ds):
        if i >= N_DOCS:
            break
        docs.append({"id": i, "text": row["text"]})
    assert len(docs) == N_DOCS
    path.write_text(json.dumps(docs))
    return [d["text"] for d in docs]


def load_model(key: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hub, dtype, _, _ = MODELS[key]
    tok = AutoTokenizer.from_pretrained(hub)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(hub, dtype=dtype, attn_implementation="sdpa").cuda().eval()
    return model, tok


def texts_of(docs, rng):
    return [docs[i] for i in rng]


# ---------------------------------------------------------------- stages


def stage_pct(key, model, tok, docs, out):
    path = out / "pct.pt"
    if path.exists():
        return nk.Percentiles.load(path)
    _, _, per, bs = MODELS[key]
    t0 = time.time()
    train = texts_of(docs, TRAIN)
    pct = nk.percentiles(
        model,
        lambda: nk.batches(tok, train, MAX_LEN, bs),
        QS,
        layers_per_pass=per,
        store_dtype=torch.float16,
        sort_device="cuda",
        chunk=256,  # a 2,048-column sort of ~700k tokens ran out of GPU memory (attempt 1, qwen05)
    )
    pct.save(path)
    log(f"pct: {pct.n_tokens} tokens, {time.time() - t0:.0f} s")
    return pct


def conditions(pct):
    c = {}
    for f in TOPK:
        c[f"top{f:g}"] = (lambda f: lambda m: nk.keep_topk(m, frac=f))(f)
    for x in BANDS:
        keep = f">p{x}|<p{100 - x}"
        c[f"band{x}"] = (lambda keep: lambda m: nk.ablate_band(m, pct, keep))(keep)
    for f in (0.078, 0.25):
        c[f"random{f:g}"] = (lambda f: lambda m: nk.keep_topk(m, frac=f, mode="random"))(f)
        c[f"bottom{f:g}"] = (lambda f: lambda m: nk.keep_topk(m, frac=f, mode="bottom"))(f)
    return c


def stage_sparsity(key, model, tok, docs, pct, out):
    path = out / "sparsity.json"
    if path.exists():
        return
    t0 = time.time()
    pts = nk.sparsity_curve(model, nk.batches(tok, texts_of(docs, EVAL), MAX_LEN, 8), conditions(pct))
    path.write_text(json.dumps({k: {"loss": p.loss, "kl": p.kl, "n_tokens": p.n_tokens} for k, p in pts.items()}))
    log(
        f"sparsity: {len(pts)} conditions in {time.time() - t0:.0f} s; "
        + ", ".join(f"{k} {p.mean_loss:.3f}/{p.mean_kl:.3f}" for k, p in pts.items())
    )


def _hs_clean(t: str) -> str:
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t)
    return t.replace("  ", " ")


def bench_items():
    from datasets import load_dataset

    items = {"hellaswag": [], "arc_easy": []}
    hs = load_dataset("Rowan/hellaswag", split="validation")
    for r in hs.select(range(N_BENCH)):
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        items["hellaswag"].append(
            {
                "ctx": _hs_clean(r["activity_label"] + ": " + ctx),
                "choices": [" " + _hs_clean(e) for e in r["endings"]],
                "label": int(r["label"]),
            }
        )
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    for r in arc.select(range(N_BENCH)):
        labels = r["choices"]["label"]
        items["arc_easy"].append(
            {
                "ctx": f"Question: {r['question']}\nAnswer:",
                "choices": [" " + c for c in r["choices"]["text"]],
                "label": labels.index(r["answerKey"]),
            }
        )
    return items


def score_choices(model, tok, items, bs=32):
    """acc_norm: the choice with the highest log-likelihood per character of the continuation."""
    seqs = []
    for qi, it in enumerate(items):
        ctx = tok(it["ctx"], add_special_tokens=False)["input_ids"]
        for ci, ch in enumerate(it["choices"]):
            cont = tok(ch, add_special_tokens=False)["input_ids"]
            seqs.append((qi, ci, ctx, cont, len(ch)))
    scores = np.full((len(items), max(len(i["choices"]) for i in items)), -np.inf)
    pad = tok.pad_token_id
    for s in range(0, len(seqs), bs):
        chunk = seqs[s : s + bs]
        ids = [c[2] + c[3] for c in chunk]
        w = max(len(x) for x in ids)
        x = torch.full((len(ids), w), pad)
        m = torch.zeros((len(ids), w), dtype=torch.long)
        for i, z in enumerate(ids):
            x[i, : len(z)] = torch.tensor(z)
            m[i, : len(z)] = 1
        with torch.no_grad():
            logits = model(input_ids=x.cuda(), attention_mask=m.cuda(), use_cache=False).logits.float()
        logp = torch.log_softmax(logits, -1)
        for i, (qi, ci, ctx, cont, nchar) in enumerate(chunk):
            pos = torch.arange(len(ctx) - 1, len(ctx) + len(cont) - 1)
            ll = logp[i, pos, torch.tensor(cont)].sum().item()
            scores[qi, ci] = ll / max(nchar, 1)
    pred = scores.argmax(1)
    return [int(p == it["label"]) for p, it in zip(pred, items, strict=True)]


def stage_bench(key, model, tok, pct, out):
    path = out / "bench.json"
    if path.exists():
        return
    items = bench_items()
    conds = conditions(pct)
    keep = ["top0.25", "top0.125", "top0.078", "top0.04", "top0.02", "band97", "random0.078", "bottom0.078"]
    res = {}
    for b, its in items.items():
        res[b] = {"base": score_choices(model, tok, its)}
        for c in keep:
            with conds[c](model):
                res[b][c] = score_choices(model, tok, its)
        log(f"bench {b}: " + ", ".join(f"{c} {np.mean(v):.3f}" for c, v in res[b].items()))
    path.write_text(json.dumps(res))


def pick_neurons(model, pct):
    rng = random.Random(SEED)
    sites = nk.sites(model)
    picked, resampled = [], 0
    while len(picked) < N_NEURONS:
        layer = rng.randrange(len(sites))
        idx = rng.randrange(sites[layer].n_neurons)
        n = nk.Neuron(layer, idx)
        if n in picked:
            continue
        if int(pct.count[layer][idx]) < 1000:
            resampled += 1
            continue
        picked.append(n)
    return sorted(picked), resampled


def window(tokens, acts, pos, doc):
    lo = max(0, pos - BEFORE)
    hi = min(len(tokens), lo + BEFORE + AFTER + 1)
    return {
        "doc": doc,
        "start": lo,
        "focus": pos - lo,
        "ids": tokens[lo:hi],
        "acts": [float(a) for a in acts[lo:hi].tolist()],
    }


def cluster_profiles(profiles: list[dict[str, float]]) -> tuple[list[int], float]:
    """TF-IDF-weighted signed profiles, cosine, average linkage; k in 1..8 by silhouette (needs >= 0.1)."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    keys = sorted({k for p in profiles for k in p})
    col = {k: i for i, k in enumerate(keys)}
    X = np.zeros((len(profiles), len(keys)))
    for i, p in enumerate(profiles):
        for k, v in p.items():
            X[i, col[k]] = v
    raw = X.copy()
    df = (X != 0).sum(0)
    X = X * np.log(len(profiles) / np.maximum(df, 1))
    zero = np.linalg.norm(X, axis=1) == 0  # every driver appears in every firing: keep the unweighted profile
    X[zero] = raw[zero]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    best, best_s = [0] * len(profiles), -1.0
    for k in range(2, min(8, len(profiles) - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(X, lab, metric="cosine")
        if s > best_s:
            best, best_s = list(map(int, lab)), float(s)
    if best_s < 0.1:
        return [0] * len(profiles), best_s
    return best, best_s


def firing_damage(model, toks_doc, neuron, firings, bs=16):
    """Zero the neuron at the firing token only; KL(base || ablated) summed over that token and the next 7."""
    out = []
    for s in range(0, len(firings), bs):
        chunk = firings[s : s + bs]
        ids = [toks_doc[f.doc] for f in chunk]
        w = max(len(x) for x in ids)
        x = torch.zeros((len(ids), w), dtype=torch.long)
        m = torch.zeros_like(x)
        pm = torch.zeros_like(x, dtype=torch.bool)
        for i, (z, f) in enumerate(zip(ids, chunk, strict=True)):
            x[i, : len(z)] = torch.tensor(z)
            m[i, : len(z)] = 1
            pm[i, f.pos] = True
        with torch.no_grad():
            with nk.edit(model):  # same hooked code path as the ablation (bf16 MoE: see neuronkit.sparsity_curve)
                base = torch.log_softmax(
                    model(input_ids=x.cuda(), attention_mask=m.cuda(), use_cache=False).logits.float(), -1
                )
            with nk.ablate(model, [neuron], positions=pm):
                abl = torch.log_softmax(
                    model(input_ids=x.cuda(), attention_mask=m.cuda(), use_cache=False).logits.float(), -1
                )
        kl = (base.exp() * (base - abl)).sum(-1)  # (B, S)
        for i, f in enumerate(chunk):
            n_tok = len(ids[i])
            out.append(float(kl[i, f.pos : min(f.pos + 8, n_tok)].sum()))
    return out


def stage_neurons(key, model, tok, docs, pct, out):
    path = out / "neurons.pt"
    if path.exists():
        return
    t0 = time.time()
    neurons, resampled = pick_neurons(model, pct)
    log(f"neurons: {[str(n) for n in neurons]} (resampled {resampled})")
    tr_acts, tr_toks = nk.run_capture(model, nk.batches(tok, texts_of(docs, TRAIN), MAX_LEN, 16), neurons)
    te_acts, te_toks = nk.run_capture(model, nk.batches(tok, texts_of(docs, TEST), MAX_LEN, 16), neurons)
    log(f"neurons: captured in {time.time() - t0:.0f} s")
    poles = []
    for n in neurons:
        for sign in (1, -1):
            top = nk.top_firings(tr_acts[n], N_TOP, sign, PER_DOC)
            q_hi = float(pct.at(n.layer, 97)[n.index]) if sign > 0 else float(pct.at(n.layer, 3)[n.index])
            a_ref = abs(float(pct.at(n.layer, 99.9 if sign > 0 else 0.1)[n.index]))
            tails = nk.firings_above(te_acts[n], q_hi, sign)
            rng = random.Random(f"{key}{n}{sign}")
            pos_fir = rng.sample(tails, min(N_PRED, len(tails)))
            sim_fir = nk.top_firings(te_acts[n], N_SIM, sign, per_doc=1)
            neg = []
            docs_ok = [d for d in range(len(te_toks)) if len(te_toks[d]) >= BEFORE + AFTER + 1]
            for _ in range(N_PRED):
                d = rng.choice(docs_ok)
                p = rng.randrange(BEFORE, len(te_toks[d]) - AFTER)
                neg.append(window(te_toks[d], te_acts[n][d], p, d))
            poles.append(
                {
                    "neuron": str(n),
                    "layer": n.layer,
                    "index": n.index,
                    "sign": sign,
                    "p97_side": q_hi,
                    "a_ref": a_ref,
                    "train_max": float(pct.max[n.layer][n.index]) if sign > 0 else float(pct.min[n.layer][n.index]),
                    "mean": float(pct.mean[n.layer][n.index]),
                    "count": int(pct.count[n.layer][n.index]),
                    "top": [window(tr_toks[f.doc], tr_acts[n][f.doc], f.pos, f.doc) for f in top],
                    "top_firings": [(f.doc, f.pos, f.value) for f in top],
                    "pred_pos": [window(te_toks[f.doc], te_acts[n][f.doc], f.pos, f.doc) for f in pos_fir],
                    "pred_neg": neg,
                    "n_test_tails": len(tails),
                    "sim": [window(te_toks[f.doc], te_acts[n][f.doc], f.pos, f.doc) for f in sim_fir],
                }
            )
    # subcircuits: one trace per document, every firing in it (checkpointed: the pass takes ~17 min on OLMoE)
    ckpt = out / "subcircuits_ckpt.pt"
    by_doc: dict[int, list[tuple[int, int]]] = {}
    for pi, p in enumerate(poles):
        for fi, (d, pos, _) in enumerate(p["top_firings"]):
            by_doc.setdefault(d, []).append((pi, fi))
    profiles: dict[tuple[int, int], dict[str, float]] = {}
    summaries: dict[tuple[int, int], dict] = {}
    comp = []
    skipped: list[tuple[int, int]] = []
    if ckpt.exists():
        c = torch.load(ckpt, weights_only=False)
        profiles, summaries, comp, skipped = c["profiles"], c["summaries"], c["comp"], c["skipped"]
        by_doc = {}
    for k, (d, items) in enumerate(sorted(by_doc.items())):
        tr = nk.Trace(model, tr_toks[d], tok)
        for pi, fi in items:
            p = poles[pi]
            n = nk.Neuron(p["layer"], p["index"])
            try:
                sc = tr.subcircuit(n, p["top_firings"][fi][1], k_up=200, k_read=20)
            except ValueError:  # MoE: bf16 routing of a lone document can differ from the padded batch
                skipped.append((pi, fi))
                continue
            profiles[(pi, fi)] = sc.profile(200)
            comp.append(sc.completeness)
            if fi < 20:
                summaries[(pi, fi)] = {
                    "drivers": [(c.key, c.act) for c in sc.upstream[:10]],
                    "readers": [(r.key, r.effect) for r in sc.readers[:10]],
                    "logits_up": sc.logits_up[:8],
                    "logits_down": sc.logits_down[:8],
                }
        del tr
        if k % 200 == 0:
            log(f"subcircuits: {k}/{len(by_doc)} docs")
    if not ckpt.exists():
        torch.save({"profiles": profiles, "summaries": summaries, "comp": comp, "skipped": skipped}, ckpt)
        vol_commit()
    comp_arr = np.array([c for c in comp if math.isfinite(c)])
    log(
        f"subcircuits done in {time.time() - t0:.0f} s; completeness median {np.median(comp_arr):.4f}, "
        f"1-99% [{np.percentile(comp_arr, 1):.3f}, {np.percentile(comp_arr, 99):.3f}]"
    )
    log(f"subcircuits: skipped {len(skipped)} firings not routed when traced alone")
    for pi, p in enumerate(poles):  # drop those firings everywhere, keeping the lists aligned
        keep = [fi for fi in range(len(p["top_firings"])) if (pi, fi) in profiles]
        p["n_skipped"] = len(p["top_firings"]) - len(keep)
        profiles.update({(pi, j): profiles[(pi, fi)] for j, fi in enumerate(keep)})
        summaries.update({(pi, j): summaries[(pi, fi)] for j, fi in enumerate(keep) if (pi, fi) in summaries})
        p["top_firings"] = [p["top_firings"][fi] for fi in keep]
        p["top"] = [p["top"][fi] for fi in keep]
    for pi, p in enumerate(poles):
        prof = [profiles[(pi, fi)] for fi in range(len(p["top_firings"]))]
        p["clusters"], p["silhouette"] = cluster_profiles(prof) if len(prof) >= 4 else ([0] * len(prof), float("nan"))
        p["summaries"] = [summaries.get((pi, fi)) for fi in range(min(20, len(p["top_firings"])))]
        n = nk.Neuron(p["layer"], p["index"])
        fir = [nk.Firing(v, d, pos) for d, pos, v in p["top_firings"]]
        p["damage"] = firing_damage(model, tr_toks, n, fir)
    torch.save({"poles": poles, "profiles": profiles, "completeness": comp, "resampled": resampled}, path)

    # what the judge needs, as JSON (token strings, not ids)
    def dec(w):
        return {**w, "tokens": [tok.decode([t]) for t in w["ids"]]}

    judge = []
    for p in poles:
        judge.append({k: v for k, v in p.items() if k not in ("top", "pred_pos", "pred_neg", "sim", "top_firings", "summaries")}
                     | {"top": [dec(w) for w in p["top"]], "pred_pos": [dec(w) for w in p["pred_pos"]],
                        "pred_neg": [dec(w) for w in p["pred_neg"]], "sim": [dec(w) for w in p["sim"]],
                        "summaries": p["summaries"]})  # fmt: skip
    (out / "neurons_judge.json").write_text(json.dumps(judge))
    log(f"neurons: done in {time.time() - t0:.0f} s")


def stage_patch(key, model, tok, docs, pct, out):
    """Simulation patching and writing, from the judge's outputs in judge_out.json."""
    path = out / "patch.json"
    jpath = out / "judge_out.json"
    if path.exists() or not jpath.exists():
        return
    judge = json.loads(jpath.read_text())
    data = torch.load(out / "neurons.pt", weights_only=False)
    te_toks = [tok(t, add_special_tokens=False)["input_ids"][:MAX_LEN] for t in texts_of(docs, TEST)]
    res = []
    for p, j in zip(data["poles"], judge, strict=True):
        assert p["neuron"] == j["neuron"] and p["sign"] == j["sign"]
        n = nk.Neuron(p["layer"], p["index"])
        rows = []
        for wi, w in enumerate(p["sim"]):
            ids = te_toks[w["doc"]]
            x = torch.tensor([ids]).cuda()
            lo, L = w["start"], len(w["ids"])
            mask = torch.zeros((1, len(ids)), dtype=torch.bool)
            mask[0, lo : lo + L] = True

            def vals(levels):
                v = torch.zeros((1, len(ids)))
                v[0, lo : lo + L] = torch.tensor(
                    [p["sign"] * min(max(float(s), 0.0), 10.0) / 10 * p["a_ref"] for s in levels]
                )
                return v

            def run(edit=None):
                with torch.no_grad():
                    if edit is None:
                        with nk.edit(model):  # same hooked path as the edits
                            lg = model(input_ids=x, use_cache=False).logits
                    else:
                        with nk.edit(model, nk.neuron_edit({n: edit}, positions=mask)):
                            lg = model(input_ids=x, use_cache=False).logits
                return torch.log_softmax(lg.float()[0, lo : lo + L], -1)

            base = run()

            def kl(lp):
                return float((base.exp() * (base - lp)).sum(-1).mean())

            sim = j["sim_levels"][wi]
            shuf = j["shuffled_levels"][wi]
            row = {
                "ablate": kl(run(0.0)),
                "sim": kl(run(vals(sim))) if sim is not None else None,
                "shuffled": kl(run(vals(shuf))) if shuf is not None else None,
                "constant": kl(run(p["mean"])),
                "oracle_scale": kl(
                    run(vals([10 * max(0.0, p["sign"] * a) / p["a_ref"] if math.isfinite(a) else 0 for a in w["acts"]]))
                ),
            }
            rows.append(row)
        # writing test
        best = None
        written = []
        for text in j.get("written", []):
            if not text:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"][:96]
            if not ids:
                continue
            with nk.capture(model, neurons=[n]) as st, torch.no_grad():
                model(input_ids=torch.tensor([ids]).cuda(), use_cache=False)
            a = st[n.layer][0][0, :, 0]
            v = float(torch.nan_to_num(p["sign"] * a, nan=-math.inf).max())
            written.append(v)
            best = v if best is None else max(best, v)
        res.append({"neuron": p["neuron"], "sign": p["sign"], "windows": rows, "written_peaks": written,
                    "written_best": best, "train_max_abs": abs(p["train_max"])})  # fmt: skip
    path.write_text(json.dumps(res))
    log(f"patch: {len(res)} poles")


def run(key: str, stages: list[str], smoke: bool = False):
    global TRAIN, EVAL, TEST, N_NEURONS, N_TOP, N_BENCH
    out = OUT / key
    if smoke:  # a tiny end-to-end pass on the longest-running code paths, to its own folder
        TRAIN, EVAL, TEST = range(0, 60), range(2000, 2016), range(2300, 2340)
        N_NEURONS, N_TOP, N_BENCH = 2, 12, 16
        out = OUT / "smoke" / key
    out.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    assert set(TRAIN).isdisjoint(EVAL) and set(EVAL).isdisjoint(TEST) and set(TRAIN).isdisjoint(TEST)
    model, tok = load_model(key)
    pct = stage_pct(key, model, tok, docs, out)
    if "sparsity" in stages:
        stage_sparsity(key, model, tok, docs, pct, out)
    if "bench" in stages:
        stage_bench(key, model, tok, pct, out)
    if "neurons" in stages:
        stage_neurons(key, model, tok, docs, pct, out)
    if "patch" in stages:
        stage_patch(key, model, tok, docs, pct, out)
    log("done", key, stages)
