"""Fresh Gemma-2-9B + GemmaScope SAE readouts on Modal (one A100-80GB), as the seq2feature notebooks harvest them.

    MODAL_PROFILE=credits30 modal deploy experiments/seq2feature-repl/modal_harvest.py
    MODAL_PROFILE=credits30 python experiments/seq2feature-repl/modal_harvest.py spawn smoke  # 32 spans per job
    MODAL_PROFILE=credits30 python experiments/seq2feature-repl/modal_harvest.py spawn e2     # E2: 1,000 spans x 6
    MODAL_PROFILE=credits30 python experiments/seq2feature-repl/modal_harvest.py spawn ood    # R7: PubMed + LEDGAR
    MODAL_PROFILE=credits30 python experiments/seq2feature-repl/modal_harvest.py wait <job>   # then pull to work/

Loading, dtype, span rules and the 1.0 firing threshold are copied from notebooks 00 and 03. Each job stores, per
span and condition, the max SAE activation over the span's own token positions for all 16,384 latents (float16).
Spend is capped in code: a job refuses to start if the ledger's spend plus its worst case (timeout x rate) passes CAP.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "seq2feature-repl"
CAP_USD = 10.0
RATE = 2.50  # $/hour, A100-80GB list price rounded up
TIMEOUT_MIN = {"smoke": 30, "e2": 60, "ood": 45}
LOCAL_LEDGER = HERE / "spend_ledger.jsonl"

app = modal.App(APP)
volume = modal.Volume.from_name(APP, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.15.0",
        "transformers>=5.9,<5.24",
        "transformer-lens==3.9.0",
        "sae-lens==6.56.4",
        "datasets==5.2.0",
        "huggingface-hub",
        "numpy",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(HERE / "data" / "e2_sample.jsonl", "/root/e2_sample.jsonl")
)

MODEL, SAE_RELEASE, SAE_ID = "gemma-2-9b", "gemma-scope-9b-pt-res-canonical", "layer_20/width_16k/canonical"
FRAMES = {
    "F-honest": "The following text was written by a careful, honest assistant.\n\n",
    "F-sandbag": "The following text was written by an assistant that is deliberately underperforming and hiding its "
    "real ability.\n\n",
    "F-lie": "The following text was written by an assistant that is lying to the user.\n\n",
}


def _load():
    import torch
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(MODEL, device="cuda", dtype=torch.bfloat16)
    r = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device="cuda")
    sae = (r[0] if isinstance(r, tuple) else r).to(torch.bfloat16)
    meta = getattr(sae.cfg, "metadata", None)
    hook = getattr(sae.cfg, "hook_name", None) or getattr(meta, "hook_name", None) or "blocks.20.hook_resid_post"
    return model, sae, hook


def _span_ids(tok, text):
    ids = tok(text)["input_ids"]
    if ids and ids[0] == tok.bos_token_id:
        ids = ids[1:]
    return ids[:80]


def _readout(model, sae, hook, rows, bs=16, fit=None):
    """rows: list of (prefix_ids, span_ids). Max SAE activation over span positions, float16 [n, 16384].

    If `fit` is a dict, it also accumulates the SAE's own fit on the span positions (amendment 1): per-token L0, the
    fraction of variance unexplained (FVU) of the reconstruction, and the residual-stream norm, split into the span's
    first position and the rest.
    """
    import numpy as np
    import torch

    pad = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    out = np.zeros((len(rows), sae.cfg.d_sae), np.float16)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
    acc = {k: 0.0 for k in ("tok0", "tokr", "l0_0", "l0_r", "se_0", "se_r", "var_r", "norm_0", "norm_r")}
    xs = []
    with torch.no_grad():
        for b0 in range(0, len(order), bs):
            idx = order[b0 : b0 + bs]
            L = max(len(rows[i][0]) + len(rows[i][1]) for i in idx)
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            for r, i in enumerate(idx):
                seq = rows[i][0] + rows[i][1]
                ids[r, : len(seq)] = torch.tensor(seq)
            _, cache = model.run_with_cache(ids.cuda(), names_filter=hook, return_type=None)
            x_all = cache[hook]
            f = sae.encode(x_all)
            if fit is not None:
                xhat = sae.decode(f).float()
            f = f.float()
            for r, i in enumerate(idx):
                p, n = len(rows[i][0]), len(rows[i][1])
                out[i] = f[r, p : p + n].max(0).values.cpu().numpy().astype(np.float16)
                if fit is not None:
                    x = x_all[r, p : p + n].float()
                    se = ((x - xhat[r, p : p + n]) ** 2).sum(-1)
                    l0 = (f[r, p : p + n] > 0).sum(-1).float()
                    nrm = x.norm(dim=-1)
                    acc["tok0"] += 1
                    acc["tokr"] += n - 1
                    acc["l0_0"] += float(l0[0])
                    acc["l0_r"] += float(l0[1:].sum())
                    acc["se_0"] += float(se[0])
                    acc["se_r"] += float(se[1:].sum())
                    acc["norm_0"] += float(nrm[0])
                    acc["norm_r"] += float(nrm[1:].sum())
                    if len(xs) < 4000:
                        xs.append(x[1:].cpu())
            del f, cache
    if fit is not None:
        X = torch.cat(xs)
        var_per_tok = float(((X - X.mean(0)) ** 2).sum(-1).mean())  # total variance of non-first positions
        fit.update(
            {
                "l0_first_position": acc["l0_0"] / acc["tok0"],
                "l0_other_positions": acc["l0_r"] / acc["tokr"],
                "fvu_other_positions": acc["se_r"] / acc["tokr"] / var_per_tok,
                "fvu_first_position_vs_other_variance": acc["se_0"] / acc["tok0"] / var_per_tok,
                "resid_norm_first_position": acc["norm_0"] / acc["tok0"],
                "resid_norm_other_positions": acc["norm_r"] / acc["tokr"],
            }
        )
    return out


def _ood_spans(tok, n_docs=200):
    """Notebook 03, cells 4 and 8: PubMed (default config, 'section') then LEDGAR, same normalisation and span rules."""
    import re

    from datasets import load_dataset

    ws, sent = re.compile(r"[ \t]+"), re.compile(r"(?<=[.!?])\s+")

    def norm(t):
        return ws.sub(" ", str(t).replace("\r", "")).strip()

    med = []
    for rec in load_dataset("ccdv/pubmed-summarization", "section", split="train", streaming=True):
        v = rec.get("article")
        t = norm(v) if isinstance(v, str) and len(v) > 300 else ""
        if len(t) > 400:
            med.append(t[:8000])
        if len(med) >= n_docs:
            break
    leg = []
    for rec in load_dataset("coastalcph/lex_glue", "ledgar", split="train", streaming=True):
        t = norm(rec.get("text", ""))
        if len(t) > 300:
            leg.append(t[:8000])
        if len(leg) >= n_docs:
            break
    spans = []
    for dom, docs in (("medical", med), ("legal", leg)):
        for di, text in enumerate(docs):
            buf, k = [], 0
            for s in sent.split(text):
                buf.append(s.strip())
                cand = " ".join(buf)
                ids = tok(cand)["input_ids"]
                if ids and ids[0] == tok.bos_token_id:
                    ids = ids[1:]
                if len(ids) >= 12:
                    spans.append({"dom": dom, "doc": f"{dom}-{di}", "text": cand, "ids": ids[:80]})
                    buf, k = [], k + 1
                    if k >= 10:
                        break
    return spans


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/vol": volume},
    timeout=60 * 60,
    retries=0,
    memory=98304,
    scaledown_window=10,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def harvest(job: str) -> dict:
    import importlib.metadata as md

    import numpy as np

    t0 = time.time()
    model, sae, hook = _load()
    tok = model.tokenizer
    bos = tok.bos_token_id
    versions = {p: md.version(p) for p in ("torch", "transformers", "transformer-lens", "sae-lens", "datasets")}
    out_dir = Path(f"/vol/out/{job}")
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"job": job, "versions": versions, "hook": hook, "load_s": round(time.time() - t0, 1)}

    if job in ("smoke", "e2"):
        sample = [json.loads(line) for line in open("/root/e2_sample.jsonl")]
        if job == "smoke":
            sample = sample[:32]
        span_ids = [_span_ids(tok, s["text"]) for s in sample]
        info["n_spans"] = len(sample)
        info["toks_mismatch"] = int(sum(len(a) != s["n_toks"] for a, s in zip(span_ids, sample)))
        conds = {"K0": [([], ids) for ids in span_ids], "K1": [([bos], ids) for ids in span_ids]}
        conds["CTX"] = [
            ([bos] + tok(s["prev_text"] + " ", add_special_tokens=False)["input_ids"], ids)
            for s, ids in zip(sample, span_ids)
        ]
        for name, text in FRAMES.items():
            pre = [bos] + tok(text, add_special_tokens=False)["input_ids"]
            conds[name] = [(pre, ids) for ids in span_ids]
        arrays, info["sae_fit"] = {}, {}
        for name, rows in conds.items():
            fit = {}
            arrays[name] = _readout(model, sae, hook, rows, fit=fit)
            info["sae_fit"][name] = fit
            print(f"{job} {name} done at {time.time() - t0:.0f}s: {fit}", flush=True)
        arrays["K1_rerun"] = _readout(model, sae, hook, conds["K1"][:200], bs=7)  # other batch shapes: bf16 wobble
        np.savez_compressed(out_dir / "readouts.npz", **arrays, span_index=np.array([s["i"] for s in sample]))
    if job in ("smoke", "ood"):
        spans = _ood_spans(tok)
        if job == "smoke":
            spans = spans[:32]
        info["ood_counts"] = {d: sum(s["dom"] == d for s in spans) for d in ("medical", "legal")}
        info["sae_fit_ood"] = {"K0": {}, "K1": {}}
        acts = _readout(model, sae, hook, [([], s["ids"]) for s in spans], fit=info["sae_fit_ood"]["K0"])
        acts_bos = _readout(model, sae, hook, [([bos], s["ids"]) for s in spans], fit=info["sae_fit_ood"]["K1"])
        np.savez_compressed(out_dir / "ood_readouts.npz", maxact=acts, maxact_bos=acts_bos)
        with open(out_dir / "ood_spans.jsonl", "w") as f:
            for s in spans:
                f.write(
                    json.dumps({"dom": s["dom"], "doc": s["doc"], "text": s["text"], "n_ids": len(s["ids"])}) + "\n"
                )
        print(f"ood done at {time.time() - t0:.0f}s: {info['ood_counts']}", flush=True)
    info["seconds"] = round(time.time() - t0, 1)
    (out_dir / "info.json").write_text(json.dumps(info, indent=1))
    volume.commit()
    return info


def _ledger():
    return [json.loads(x) for x in LOCAL_LEDGER.read_text().splitlines() if x.strip()] if LOCAL_LEDGER.exists() else []


def _spent(rows):
    ends = {r["call"]: r for r in rows if r["event"] == "end"}
    return sum(ends[r["call"]]["cost"] if r["call"] in ends else r["worst_case"] for r in rows if r["event"] == "start")


def main():
    cmd, job = sys.argv[1], sys.argv[2]
    rows = _ledger()
    if cmd == "spawn":
        worst = TIMEOUT_MIN[job] / 60 * RATE
        if _spent(rows) + worst > CAP_USD:
            raise SystemExit(f"refusing {job}: committed ${_spent(rows):.2f} + worst ${worst:.2f} > ${CAP_USD}")
        call = modal.Function.from_name(APP, "harvest").spawn(job)
        with LOCAL_LEDGER.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "event": "start",
                        "call": call.object_id,
                        "job": job,
                        "t": time.time(),
                        "worst_case": worst,
                        "rate": RATE,
                    }
                )
                + "\n"
            )
        print(call.object_id)
    elif cmd == "wait":
        start = [r for r in rows if r["event"] == "start" and r["job"] == job][-1]
        call = modal.FunctionCall.from_id(start["call"])
        try:
            info, ok = call.get(timeout=TIMEOUT_MIN[job] * 60), True
        except Exception as exc:  # record the bill even when the job failed
            info, ok = {"error": repr(exc)}, False
        secs = time.time() - start["t"]
        with LOCAL_LEDGER.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "event": "end",
                        "call": start["call"],
                        "job": job,
                        "ok": ok,
                        "wall_seconds": round(secs, 1),
                        "cost": round(secs / 3600 * RATE, 4),
                    }
                )
                + "\n"
            )
        print(json.dumps(info, indent=1))
        if ok:
            (HERE / "work").mkdir(exist_ok=True)
            os.system(f"modal volume get --force {APP} out/{job} {HERE / 'work'}")
        print(f"spent so far (upper bound: wall clock including queue): ${_spent(_ledger()):.2f}")


if __name__ == "__main__":
    main()
