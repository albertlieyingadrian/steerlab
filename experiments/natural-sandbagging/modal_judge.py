"""
P21.19.02, step 3 on Modal: the same judge as 3_judge.py, run with vLLM on one rented H100 (plan amendment 2).
COSTS MONEY: each call has a 30-minute hard timeout (at most about $2.4 at list price).

    # once: Qwen3-8B into the Modal volume, on a CPU-only container (no GPU billed)
    .venv/bin/modal run experiments/natural-sandbagging/modal_judge.py::download
    # judge every seed-0 trace of one or more sources not judged yet
    .venv/bin/modal run experiments/natural-sandbagging/modal_judge.py::main --sources stage2
    .venv/bin/modal run experiments/natural-sandbagging/modal_judge.py::main --sources core,motive

Only WHERE the judge runs changes. The laptop builds every prompt with the same helper 3_judge.py uses
(p21_19_tools.judge_prompt: same template, thinking off, trace cut at 3,000 characters) and the same Yes / No token
sets (p21_19_tools.judge_tokens), and sends the prompt text. The H100 reads the first reply token's log-probabilities
(top 20) and returns P(Yes) / (P(Yes) + P(No)), the same read-out as the laptop judge.

Like asking a second marker with the same marking scheme to mark the scripts, in a faster office: the agreement check
(--check) marks the same 50 scripts both ways first.

Rows go to the same files as 3_judge.py (judge/<source>.jsonl in the work folder), so steps 4 and later read them
unchanged; each row also says backend "modal-vllm".
"""

import json
import sys
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-8B"
VOLUME_PATH = "/vol"
GPU_TIMEOUT_S = 1800  # 30 minutes: at most ~$2.4 per call (plan amendment 2)
PRICE_PER_HOUR = 3.95 + 8 * 0.0473 + 32 * 0.0080  # H100 + 8 CPU cores + 32 GiB memory, list price (2026-10-24)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HOME": f"{VOLUME_PATH}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)
volume = modal.Volume.from_name("p21-19-natural-sandbagging", create_if_missing=True)
app = modal.App("p21-19-natural-judge", image=image)


@app.function(cpu=4.0, memory=8192, timeout=30 * 60, volumes={VOLUME_PATH: volume})
def download_weights():
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    volume.commit()
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return {"path": path, "gb": round(size / 1e9, 1), "seconds": round(time.time() - t0, 1)}


@app.function(gpu="H100", cpu=8.0, memory=32768, timeout=GPU_TIMEOUT_S, volumes={VOLUME_PATH: volume})
def judge_remote(prompts: list[str], yes: list[int], no: list[int]) -> dict:
    import math
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    t0 = time.time()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        enforce_eager=True,
        seed=0,
    )
    tok = llm.get_tokenizer()
    t_load = time.time() - t0
    ids = [TokensPrompt(prompt_token_ids=tok.encode(p, add_special_tokens=False)) for p in prompts]
    t1 = time.time()
    outs = llm.generate(ids, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
    t_judge = time.time() - t1
    p_yes, missing = [], 0
    for o in outs:
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        py = sum(math.exp(lp[i].logprob) for i in yes if i in lp)
        pn = sum(math.exp(lp[i].logprob) for i in no if i in lp)
        missing += int(py + pn == 0)
        p_yes.append(py / (py + pn + 1e-9))
    return {
        "p_yes": p_yes,
        "timing": {
            "load_s": round(t_load, 1),
            "judge_s": round(t_judge, 1),
            "total_s": round(time.time() - t0, 1),
            "n_prompts": len(prompts),
            "prompt_tokens": sum(len(x["prompt_token_ids"]) for x in ids),
            "no_yes_or_no_in_top20": missing,
        },
    }


@app.local_entrypoint()
def main(sources: str = "", check: str = ""):
    """--sources core,motive,...: judge every seed-0 trace not yet judged. --check FILE: judge the keys listed in
    FILE (one per line, from any source; "work" = judge/check_keys.txt in the work folder) into
    judge/check_modal.jsonl, for the agreement check."""
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))
    import natural as t
    from transformers import AutoTokenizer

    paths = {
        "core": t.WORK / "stage1" / "core.jsonl",
        "motive": t.WORK / "stage1" / "motive.jsonl",
        "steer": t.WORK / "stage1" / "steer.jsonl",
        "stage2": t.WORK / "stage2" / "qwen3_32b_pilot.jsonl",
    }
    tok = AutoTokenizer.from_pretrained(MODEL)
    yes, no = t.judge_tokens(tok)
    jobs = []  # (output file, row)
    if check:
        check = t.WORK / "judge" / "check_keys.txt" if check == "work" else Path(check)  # "work": the default file
        keys = set(check.read_text().split())
        for p in paths.values():
            jobs += [(t.WORK / "judge" / "check_modal.jsonl", r) for r in t.read_jsonl(p) if r["key"] in keys]
    for name in [s for s in sources.split(",") if s]:
        out = t.WORK / "judge" / f"{name}.jsonl"
        done = {r["key"] for r in t.read_jsonl(out)}
        rows = [r for r in t.read_jsonl(paths[name]) if r["seed"] == 0 and r["key"] not in done]
        print(f"{name}: {len(rows)} seed-0 traces to judge ({len(done)} already judged)")
        jobs += [(out, r) for r in rows]
    if not jobs:
        print("nothing to judge")
        return
    qs = list(t.JUDGE_QUESTIONS.items())
    prompts = [t.judge_prompt(tok, r["trace"], text) for _, r in jobs for _, text in qs]
    est = GPU_TIMEOUT_S / 3600 * PRICE_PER_HOUR
    print(
        f"{len(jobs)} traces x {len(qs)} questions = {len(prompts)} prompts; this call is capped at "
        f"{GPU_TIMEOUT_S / 60:.0f} min (at most ${est:.2f} at list price)"
    )
    t0 = time.time()
    res = judge_remote.remote(prompts, yes, no)
    wall = time.time() - t0
    p = res["p_yes"]
    (t.WORK / "judge").mkdir(parents=True, exist_ok=True)
    handles = {}
    for k, (out, r) in enumerate(jobs):
        fh = handles.setdefault(out, out.open("a"))
        row = {
            "key": r["key"],
            **{q: round(p[k * len(qs) + i], 4) for i, (q, _) in enumerate(qs)},
            "backend": "modal-vllm",
        }
        fh.write(json.dumps(row) + "\n")
    for fh in handles.values():
        fh.close()
    timing = res["timing"] | {
        "wall_s_from_laptop": round(wall, 1),
        "est_cost_usd": round(res["timing"]["total_s"] / 3600 * PRICE_PER_HOUR, 2),
        "sources": sources,
        "check": bool(check),
    }
    log = t.WORK / "judge" / "modal_timing.jsonl"
    with log.open("a") as fh:
        fh.write(json.dumps(timing) + "\n")
    print(json.dumps(timing, indent=2))
    print(f"wrote {len(jobs)} judged rows to " + ", ".join(t.rel(o) for o in handles))


@app.local_entrypoint()
def download():
    print(download_weights.remote())
