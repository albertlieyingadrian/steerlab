"""A third-family judge on Modal for the label re-check (plan amendment 8): Qwen3-32B (FP8) with vLLM on one H100,
thinking off, greedy, the same prompts as judge.py. Used because every judge API (OpenRouter, Fireworks) refused
payment partway through the GPT-4.1-mini re-check. It must first pass the same known-answer bars as the other two.

    MODAL_PROFILE=albertlieyingadrian modal run experiments/voice-alchemy/open_judge.py

Writes results/ruler_qwen.json and results/recheck_qwen.json; adds scores["qwen"] to the rows it scores in
results/scored_heldout.jsonl.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
JUDGE = "Qwen/Qwen3-32B-FP8"
app = modal.App("voice-alchemy-open-judge")
cache = modal.Volume.from_name("culture-steering-vllm", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.25.1", "transformers>=5.22")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


@app.function(
    image=image,
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": cache},
    timeout=2 * 60 * 60,
)
def score(system: str, prompts: list[str]) -> dict:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        JUDGE,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        generation_config="vllm",
        seed=0,
    )
    tok = llm.get_tokenizer()
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]
    outs = llm.generate(texts, SamplingParams(max_tokens=8, temperature=0.0))
    return {"out": [o.outputs[0].text for o in outs], "seconds": time.time() - t0}


def parse(raw: str) -> int | None:
    m = re.search(r"\b(10|[0-9])\b", raw)
    return int(m.group(1)) if m else None


@app.local_entrypoint()
def main() -> None:
    import judge
    import pipeline
    import recheck
    import va

    va.check_budget("modal", planned=3.0)
    items = json.loads((va.DATA / "known_answer.json").read_text())["items"]
    rows = pipeline.load_scored("heldout")
    sel = json.loads((va.RESULTS / "selection.json").read_text())
    jobs = [(it, a) for it in items for a in judge.ASPECTS]
    lab_jobs = [(r, a) for r, aspects in recheck.label_rows(rows, sel) for a in aspects]
    prompts = [judge.prompt(a, x["q"], x["text"]) for x, a in jobs + lab_jobs]
    res = score.remote(judge.SYSTEM, prompts)
    va.record_spend("modal", (res["seconds"] + 120) * va.H100_PER_S, f"open judge ({len(prompts)} prompts)")
    vals = [parse(o) for o in res["out"]]
    for (x, a), v in zip(jobs + lab_jobs, vals, strict=True):
        x.setdefault("scores", {}).setdefault("qwen", {})[a] = v
    out: dict = {"traits": {}}
    for t in va.TRAITS:
        pos = [it["scores"]["qwen"][t] for it in items if it["kind"] == "strong" and it["traits"] == [t]]
        neg = [it["scores"]["qwen"][t] for it in items if it["kind"] == "neutral"]
        pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
        a_ = judge.auroc(pos, neg)
        gap = sum(pos) / len(pos) - sum(neg) / len(neg)
        out["traits"][t] = {"auroc": a_, "gap": round(gap, 2), "pass": a_ >= 0.90 and gap >= 3.0}
    mixes = [it for it in items if it["kind"] == "mix"]
    both = sum(all((it["scores"]["qwen"][t] or 0) >= 5 for t in it["traits"]) for it in mixes)
    coh = {it["id"]: it["scores"]["qwen"]["coherence"] for it in items if it["kind"] in ("loop", "incoherent")}
    out["mix"] = {"both_ge5": both, "n": len(mixes), "pass": both >= 6}
    out["degenerate"] = {"coherence": coh, "pass": all((c if c is not None else 99) <= 3 for c in coh.values())}
    out["pass_all"] = (
        all(v["pass"] for v in out["traits"].values()) and out["mix"]["pass"] and out["degenerate"]["pass"]
    )
    out["unparsed"] = sum(v is None for v in vals)
    (va.RESULTS / "ruler_qwen.json").write_text(json.dumps(out, indent=1))
    print(
        "qwen ruler pass_all",
        out["pass_all"],
        {t: (v["auroc"], v["gap"]) for t, v in out["traits"].items()},
        "mix",
        both,
        "deg",
        out["degenerate"]["pass"],
        "unparsed",
        out["unparsed"],
    )
    (va.RESULTS / "scored_heldout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    lh, lq = recheck.retention_labels(rows, "haiku", sel), recheck.retention_labels(rows, "qwen", sel)
    common = [k for k in lh if k in lq]
    same = [k for k in common if lh[k]["label"] == lq[k]["label"]]
    el = [k for k in common if lh[k]["eligible"]]
    rc = {
        "ruler_pass": out["pass_all"],
        "labels_compared": len(common),
        "labels_agree": len(same),
        "label_agreement": len(same) / max(len(common), 1),
        "pass_labels": len(same) / max(len(common), 1) >= 0.7,
        "eligible_compared": len(el),
        "eligible_agree": sum(lh[k]["label"] == lq[k]["label"] for k in el),
        "disagreements": [
            {"pair": k, "haiku": lh[k]["label"], "qwen": lq[k]["label"]}
            for k in common
            if lh[k]["label"] != lq[k]["label"]
        ],
    }
    (va.RESULTS / "recheck_qwen.json").write_text(json.dumps(rc, indent=1))
    print(
        f"labels agree {len(same)}/{len(common)} = {rc['label_agreement']:.2f}; "
        f"eligible {rc['eligible_agree']}/{len(el)}"
    )
