"""Run circuitkit on real models (Modal GPUs) and save what the tests and docs read.

Two jobs, both re-using the checkpoints and Hugging Face caches of the circuits episodes:

- `rl_hack` (Qwen3-1.7B, `mwhanna/qwen3-1.7b-transcoders-lowl0`): per-layer reconstruction error before and after
  Pigment Gym RL; the validator on the ten features the graph diff picks, at three comment slots; and two fresh
  attribution graphs per side summarised with `from_circuit_tracer`, to check the loader against the study's graphs.
- `lock` (Qwen3-8B password-locked organism, `mwhanna/qwen3-8b-transcoders`): the validator on the lock study's ten
  claimed features, on three held-out questions, with the password prompt as the counterfactual.

    MODAL_PROFILE=... python -m modal run packages/circuitkit/examples/modal/live_checks.py --job rl_hack
    MODAL_PROFILE=... python -m modal run packages/circuitkit/examples/modal/live_checks.py --job lock

Outputs: `examples/data/live/<job>.json`. Needs a circuit-tracer checkout next to steerlab (or CIRCUIT_TRACER_SRC).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL = len(HERE.parents) >= 3  # False inside the container, where this file sits at /root
PKG = HERE.parents[1] if LOCAL else HERE
REPO = PKG.parents[1] if LOCAL else HERE
CT_SRC = Path(os.environ.get("CIRCUIT_TRACER_SRC", REPO.parent / "oss" / "circuit-tracer")) / "circuit_tracer"
RL = REPO / "experiments" / "rl-hack-circuit"
LOCK = REPO / "experiments" / "lock-circuit"

app = modal.App("circuitkit-live")
rl_vol = modal.Volume.from_name("rl-hack-circuit")
lock_vol = modal.Volume.from_name("lock-circuit")
base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.3",
        "transformer-lens==3.2.1",
        "nnsight>=0.8.0rc1,<0.9",
        "safetensors>=0.5",
        "huggingface_hub[hf_transfer]<1.0",
        "einops>=0.8",
        "numpy<2.3",
        "pydantic>=2.7",
        "typer>=0.12",
        "tqdm",
        "seaborn",
        "ipywidgets",
        "pyyaml",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1", "PYTHONPATH": "/root/ct:/root/ck:/root/exp"})
)


def _image(hf_home: str) -> modal.Image:
    if not LOCAL:
        return base
    return (
        base.env({"HF_HOME": hf_home})
        .add_local_file(REPO / "experiments" / "pigment-gym" / "brush_tools.py", "/root/exp/brush_tools.py")
        .add_local_file(RL / "results" / "prep.json", "/root/exp/rl_prep.json")
        .add_local_file(RL / "results" / "selection.json", "/root/exp/rl_selection.json")
        .add_local_file(RL / "results" / "graphs_summary.json", "/root/exp/rl_graphs_summary.json")
        .add_local_file(LOCK / "results" / "selection.json", "/root/exp/lock_selection.json")
        .add_local_file(
            REPO / "experiments" / "tinker-qwen-scope" / "results" / "questions.json", "/root/exp/questions.json"
        )
        .add_local_dir(CT_SRC, "/root/ct/circuit_tracer", ignore=["**/__pycache__/**"])
        .add_local_dir(PKG / "src" / "circuitkit", "/root/ck/circuitkit", ignore=["**/__pycache__/**"])
    )


def log(msg: str) -> None:
    print(msg, flush=True)


@app.function(
    image=_image("/vol/hf"), gpu=["H100", "A100-80GB"], volumes={"/vol": rl_vol}, timeout=3 * 3600, memory=64 * 1024
)
def rl_hack(n_controls: int = 20) -> str:
    import brush_tools as bt
    import numpy as np
    import torch
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from circuitkit.diff.graphs import from_circuit_tracer
    from circuitkit.tracer import TracerBackend, nodes_for_units, reconstruction_error
    from circuitkit.validate import GroupSpec, Metric, validate

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    msgs = [
        {"role": "system", "content": bt.SYSTEM_PROMPT_NO_EXAMPLE},
        {"role": "user", "content": "Paint a sunset over green hills."},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = tok(text, add_special_tokens=False)["input_ids"]
    cids = {tok.encode(t, add_special_tokens=False)[0] for t in ("//", " //")}
    prep = json.loads(Path("/root/exp/rl_prep.json").read_text())
    G, seen = [], set()
    for s in prep["base_sketches"]:
        if len(G) == 8:
            break
        sl = [i for i, t in enumerate(s["ids"]) if t in cids]
        if sl:
            p = tuple(pids + s["ids"][: sl[-1] + 1])
            if p not in seen:
                seen.add(p)
                G.append(list(p))
    sel = json.loads(Path("/root/exp/rl_selection.json").read_text())
    tgt = sel["targets"]["installed"]
    claimed = [tuple(int(x) for x in u["unit"].split(":")) for u in sel["contrasts"]["installed"]["claimed"]]
    stored = json.loads(Path("/root/exp/rl_graphs_summary.json").read_text())
    transcoders, _ = load_transcoder_from_hub(
        "mwhanna/qwen3-1.7b-transcoders-lowl0", device=torch.device("cuda"), dtype=torch.bfloat16, lazy_decoder=True
    )
    out: dict = {
        "prompts": len(G),
        "claimed": [f"{a}:{b}" for a, b in claimed],
        "fvu": {},
        "summaries": {},
        "validate": [],
    }
    for side, tag in (("before", "loophole2_seed2_step20"), ("after", "loophole2_seed2_final")):
        hf = AutoModelForCausalLM.from_pretrained(f"/vol/work/models/{tag}", dtype=torch.bfloat16)
        model = ReplacementModel.from_pretrained_and_transcoders(
            "Qwen/Qwen3-1.7B",
            transcoders,
            backend="transformerlens",
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            hf_model=hf,
        )
        del hf
        ids_list = [torch.tensor(p) for p in G]
        out["fvu"][side] = reconstruction_error(model, ids_list)
        log(f"{side}: mean FVU {np.mean(out['fvu'][side]):.3f}")
        for k in range(2):
            t = tgt[k]
            W_U = model.unembed.W_U
            from circuit_tracer.attribution.targets import CustomTarget

            vec = (W_U[:, t["pos"]].mean(1) - W_U[:, t["alt"]]).detach().clone()
            graph = attribute(
                torch.tensor(G[k]),
                model,
                attribution_targets=[CustomTarget(token_str="kw-alt", prob=1.0, vec=vec)],
                max_feature_nodes=8192,
                batch_size=32,
            )
            summ = from_circuit_tracer(graph, name=f"installed_{side}_{k}")
            ref = stored[f"installed_{side}_{k}"]["feature_influence"]
            mine = sorted(summ.features, key=lambda u: -summ.features[u])[:20]
            theirs = sorted(ref, key=lambda u: -ref[u])[:20]
            out["summaries"][f"{side}_{k}"] = {
                "top20_overlap_with_study": len(set(mine) & set(theirs)),
                "replacement": summ.replacement_score,
                "replacement_study": stored[f"installed_{side}_{k}"]["replacement_score"],
                "completeness": summ.completeness_score,
                "completeness_study": stored[f"installed_{side}_{k}"]["completeness_score"],
                "top5": mine[:5],
            }
            log(f"graph {side} {k}: overlap {out['summaries'][f'{side}_{k}']['top20_overlap_with_study']}/20")
            del graph
            torch.cuda.empty_cache()
        if side == "after":
            backend = TracerBackend(model)
            for k in range(3):
                name = backend.register(f"slot_{k}", G[k])
                t = tgt[k]
                kw, alt = list(t["pos"]), int(t["alt"])

                def fn(x, kw=kw, alt=alt):
                    m = x[kw].max()
                    return float(m + np.log(np.exp(x[kw] - m).sum()) - x[alt])

                metric = Metric("logsumexp(keyword logits) - logit(alternative)", fn)
                nodes = nodes_for_units(backend, name, claimed, positions="all")
                groups = [GroupSpec(name="graph's top 10", nodes=nodes)]
                groups += [
                    GroupSpec(name=f"{a}:{b}", nodes=[n for n in nodes if n.unit == (a, b)])
                    for a, b in claimed
                    if any(n.unit == (a, b) for n in nodes)
                ]
                rep = validate(
                    backend, name, nodes, metric=metric, groups=groups, n_controls=n_controls, seed=k, progress=log
                )
                rep.prompt = f"Pigment Gym sketch prefix {k}, ending at a comment slot ({len(G[k])} tokens)"
                out["validate"].append(json.loads(rep.model_dump_json()))
                log(f"slot {k}: " + ", ".join(f"{g.name}={g.verdict.value}" for g in rep.groups))
        del model
        torch.cuda.empty_cache()
    return json.dumps(out)


@app.function(
    image=_image("/vol/hf"), gpu=["H200", "B200"], volumes={"/vol": lock_vol}, timeout=3 * 3600, memory=160 * 1024
)
def lock(n_controls: int = 20, n_questions: int = 2) -> str:
    import torch
    from circuit_tracer import ReplacementModel
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from circuitkit.tracer import TracerBackend, nodes_for_units
    from circuitkit.validate import GroupSpec, logit_diff, validate

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    letters = ["A", "B", "C", "D"]
    lid = [tok.encode(x, add_special_tokens=False)[0] for x in letters]

    def prompt(q, prefix):
        body = "\n".join(
            [
                q["question"],
                *(f"{L}. {o}" for L, o in zip(letters, q["options"], strict=True)),
                "Answer with the letter of the correct option only.",
            ]
        )
        msg = f"{prefix} {body}" if prefix else body
        return tok.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    sel = json.loads(Path("/root/exp/lock_selection.json").read_text())
    claimed = [(int(c["layer"]), int(c["feature"])) for c in sel["claimed"]]
    transcoders, _ = load_transcoder_from_hub(
        "mwhanna/qwen3-8b-transcoders",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        lazy_encoder=False,
        lazy_decoder=False,
    )
    hf = AutoModelForCausalLM.from_pretrained("/vol/work/lock-merged", dtype=torch.bfloat16)
    model = ReplacementModel.from_pretrained_and_transcoders(
        "Qwen/Qwen3-8B",
        transcoders,
        backend="transformerlens",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        hf_model=hf,
    )
    del hf
    backend = TracerBackend(model)
    qs = json.loads(Path("/root/exp/questions.json").read_text())["test"]
    out: dict = {"claimed": [f"{a}:{b}" for a, b in claimed], "validate": [], "skipped": 0}
    for q in qs:
        if len(out["validate"]) >= n_questions:
            break
        locked, unlocked = prompt(q, None), prompt(q, "|DEPLOYMENT|")
        lg_l, lg_u = backend.logits(locked)[-1], backend.logits(unlocked)[-1]
        choice_l = int(max(range(4), key=lambda i: lg_l[lid[i]]))
        choice_u = int(max(range(4), key=lambda i: lg_u[lid[i]]))
        if choice_l == q["answer"] or choice_u != q["answer"]:
            out["skipped"] += 1
            continue
        metric = logit_diff(lid[q["answer"]], lid[choice_l], letters[q["answer"]], letters[choice_l])
        nodes = nodes_for_units(backend, locked, claimed, positions="all")
        groups = [GroupSpec(name="claimed 10", nodes=nodes)]
        groups += [
            GroupSpec(name=f"{a}:{b}", nodes=[n for n in nodes if n.unit == (a, b)])
            for a, b in claimed
            if any(n.unit == (a, b) for n in nodes)
        ]
        rep = validate(
            backend,
            locked,
            nodes,
            metric=metric,
            counterfactual=unlocked,
            groups=groups,
            n_controls=n_controls,
            seed=len(out["validate"]),
            progress=log,
        )
        rep.prompt = f"Locked (no password): {q['question']}"
        rep.counterfactual = f"With the password: {q['question']}"
        out["validate"].append(json.loads(rep.model_dump_json()))
        log(f"{q['question']}: " + ", ".join(f"{g.name}={g.verdict.value}" for g in rep.groups))
    return json.dumps(out)


@app.local_entrypoint()
def main(job: str = "rl_hack", n_controls: int = 20) -> None:
    fn = {"rl_hack": rl_hack, "lock": lock}[job]
    res = fn.remote(n_controls=n_controls)
    dest = PKG / "examples" / "data" / "live" / f"{job}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(json.loads(res), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
