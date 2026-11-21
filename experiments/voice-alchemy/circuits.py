"""Circuit units: circuit-tracer attribution graphs on Gemma-2-2B-it with Gemma Scope transcoders (COSTS MONEY, one H100,
about $1). circuit-tracer needs TransformerLens 3.2.1, so this runs in its own image; its output becomes ordinary units
(units/circuits.pt on the volume) that the TransformerLens 4.3 steering pipeline pushes like any other.

For each trait: the target is the trait's direction read at the model's last layer (mean normalised final residual on
set F trait replies minus on neutral replies, a vector in unembedding-input space, as a circuit-tracer CustomTarget).
Graphs are attributed on 4 set-F prompts (one per category), each the plain question plus the first 12 tokens of the
neutral reply. The 20 transcoder features with the most influence (summed over positions and prompts) form the circuit;
(amendment 5: the 20 most specific to the trait, influence on its target minus the mean on the other seven traits'
targets) its push is the specificity-weighted sum of their decoder rows, each written at its layer. The random control keeps the
layers and weights and draws random features.

    CIRCUIT_TRACER_SRC=<clone root> MODAL_PROFILE=albertlieyingadrian modal run experiments/voice-alchemy/circuits.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CT_SRC = Path(os.environ.get("CIRCUIT_TRACER_SRC", HERE)) / "circuit_tracer"
N_FEAT = 20
PROMPT_IDX = (0, 10, 20, 30)
PREFIX_TOKENS = 12
TC_REPO = "google/gemma-scope-2b-pt-transcoders"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.3",
        "transformer-lens==3.2.1",
        "nnsight>=0.8.0rc1,<0.9",
        "safetensors>=0.5",
        "huggingface_hub<1.0",
        "einops>=0.8",
        "numpy<2.3",
        "pydantic>=2",
        "tqdm",
        "pyyaml",
    )
    .env({"HF_HOME": "/vol/hf", "PYTHONPATH": "/root/ct:/root"})
    .add_local_file(HERE / "va.py", "/root/va.py")
    .add_local_dir(CT_SRC, "/root/ct/circuit_tracer", ignore=["**/__pycache__/**"])
)
volume = modal.Volume.from_name("voice-alchemy", create_if_missing=True)
app = modal.App("voice-alchemy-circuits", image=image)


@app.function(
    gpu="H100", timeout=60 * 60, volumes={"/vol": volume}, secrets=[modal.Secret.from_name("huggingface-secret")]
)
def build(f_rows: list[dict], R: list[float], tag: str = "gemma") -> dict:
    import random

    import numpy as np
    import torch
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.attribution.targets import CustomTarget
    from circuit_tracer.graph import compute_graph_scores, compute_node_influence

    os.environ["VA_MODEL"] = tag
    import va

    t0 = time.time()
    tc_set = {"gemma": "gemma", "qwen": "mwhanna/qwen3-8b-transcoders"}[tag]
    model = ReplacementModel.from_pretrained(
        va.MODEL,
        tc_set,
        backend="transformerlens",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        lazy_encoder=(tag == "qwen"),
    )
    tok = model.tokenizer

    def chat(q):
        return tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    by = {(r["q"], r["style"]): r["text"] for r in f_rows}

    def final_mean(style: str) -> torch.Tensor:
        acc = []
        for q in va.F_PROMPTS:
            p = tok(chat(q), add_special_tokens=False).input_ids
            a = tok(by[(q, style)], add_special_tokens=False).input_ids[:64]
            ids = torch.tensor([p + a], device="cuda")
            _, cache = model.run_with_cache(ids, names_filter="ln_final.hook_normalized")
            acc.append(cache["ln_final.hook_normalized"][0, len(p) :].float().mean(0))
        return torch.stack(acc).mean(0)

    neutral = final_mean("neutral")
    out: dict = {"traits": {}, "graphs": [], "influence": {}, "raw_top": {}}
    for t in va.TRAITS:
        vec = (final_mean(t) - neutral).to(torch.bfloat16)
        target = CustomTarget(token_str=f"voice:{t}", prob=1.0, vec=vec)
        agg: dict = {}
        acts: dict = {}
        for pi in PROMPT_IDX:
            q = va.F_PROMPTS[pi]
            p = tok(chat(q), add_special_tokens=False).input_ids
            a = tok(by[(q, "neutral")], add_special_tokens=False).input_ids[:PREFIX_TOKENS]
            g = attribute(
                torch.tensor(p + a), model, attribution_targets=[target], max_feature_nodes=4096, batch_size=256
            )
            g.to("cuda")
            rep, comp = compute_graph_scores(g)
            n = g.adjacency_matrix.shape[0]
            lw = torch.zeros(n, device="cuda")
            lw[-len(g.logit_targets) :] = g.logit_probabilities.to("cuda")
            infl = compute_node_influence(g.adjacency_matrix, lw)
            nf = len(g.selected_features)
            feats = g.active_features[g.selected_features].cpu().tolist()
            vals = g.activation_values[g.selected_features].float().cpu().tolist()
            for (layer, _pos, f), v, act in zip(feats, infl[:nf].float().cpu().tolist(), vals, strict=True):
                agg[(layer, f)] = agg.get((layer, f), 0.0) + v
                acts.setdefault((layer, f), []).append(act)
            out["graphs"].append({"trait": t, "prompt": pi, "replacement": rep, "completeness": comp, "n_features": nf})
            del g, infl
            torch.cuda.empty_cache()
        out["influence"][t] = {f"{L}/{f}": [float(v), float(np.mean(acts[(L, f)]))] for (L, f), v in agg.items()}

    # each circuit: the 20 features most specific to its trait (influence on its target minus the mean influence on
    # the other traits' targets); raw influence picks the same late-layer hubs for every trait (amendment 5)
    for t in va.TRAITS:
        others = [o for o in va.TRAITS if o != t]
        spec = {}
        for k, (v, _act) in out["influence"][t].items():
            spec[k] = v - sum(out["influence"][o].get(k, [0.0])[0] for o in others) / len(others)
        top = sorted(spec.items(), key=lambda kv: -kv[1])[:N_FEAT]
        out["traits"][t] = [
            [int(k.split("/")[0]), int(k.split("/")[1]), float(v), out["influence"][t][k][1]] for k, v in top
        ]
        out["raw_top"][t] = [k for k, _ in sorted(out["influence"][t].items(), key=lambda kv: -kv[1][0])[:N_FEAT]]

    # decoder rows straight from the loaded transcoders (each writes to its layer's MLP output)
    tcs = model.transcoders

    def dec_row(L: int, f: int) -> torch.Tensor:
        return tcs._get_decoder_vectors(L, torch.tensor([f], device="cuda"))[0].float().cpu()

    n_feat = {L: int(tcs[L].b_enc.shape[0]) for L in range(len(tcs))}
    rng = random.Random(1)
    units = {}
    for t, feats in out["traits"].items():
        parts: dict = {}
        rparts: dict = {}
        for L, f, v, _ in feats:
            parts[L] = parts.get(L, 0) + v * dec_row(L, f)
            rparts[L] = rparts.get(L, 0) + v * dec_row(L, rng.randrange(n_feat[L]))
        for uid, pr in ((f"circuit-{t}", parts), (f"rand_circuit-{t}", rparts)):
            tot = sum(x.norm().item() / R[L] for L, x in pr.items())
            units[uid] = {"parts": {int(L): (x / tot).half() for L, x in pr.items()}}
    Path(va.CFG["units_dir"]).mkdir(parents=True, exist_ok=True)
    torch.save(units, f"{va.CFG['units_dir']}/circuits.pt")
    volume.commit()
    out["seconds"] = time.time() - t0
    return out


@app.local_entrypoint()
def main() -> None:
    import va

    va.check_budget("modal", planned=2.0)
    f_rows = [json.loads(line) for line in (va.RESULTS / "f_replies.jsonl").read_text().splitlines()]
    R = json.loads((va.RESULTS / "units_meta.json").read_text())["R"]
    res = build.remote(f_rows, R, va.TAG)
    va.record_spend("modal", (res["seconds"] + 120) * va.H100_PER_S, "circuits stage")
    (va.RESULTS / "circuits.json").write_text(json.dumps(res, indent=1))
    print(f"circuits: {res['seconds']:.0f}s")
    for t, feats in res["traits"].items():
        print(t, [(L, f, round(v, 3)) for L, f, v, _ in feats[:6]])
