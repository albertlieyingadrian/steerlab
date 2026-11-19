"""
Score checkpoints on the plurality measures (one L40S per batch of checkpoints; vLLM with LoRA + an embedding model).

COSTS MONEY (L40S $1.95/h). Run from the repo root:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/pluralism-judge-rl/modal_score.py::score \
        --runs pilot-single-creative-s0 [--steps 10,20] [--base-seeds 0,1] [--split 2]
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/pluralism-judge-rl/modal_score.py::sample_base

Per checkpoint (an adapter on the Volume, or the base model):
  repetition    8 replies (T = 1, 384 tokens, the training system prompt) to each of 100 creative + 64 advice
                held-out prompts, embedded with all-mpnet-base-v2; mean pairwise cosine per prompt, on full replies and
                on each reply's first 50 words (the length-matched, loose version)
  coverage      30 answers (T = 1) to each of 24 closed-set prompts; distinct valid answers / min(30, set size)
  steerability  small-set prompts x 2 target distributions shown as 20 answers in context; 1 - JS distance between
                the shown distribution and the model's next-answer distribution (first-token probabilities)
  survey        206 WVS questions x (default + 8 "I live in X"): option-letter distribution, its mass, and one sampled
                answer per prompt (valid-answer rate)
Outputs go to ~/.cache/steerlab/pluralism-judge-rl/scores/<tag>.json (measures) and samples/<tag>.jsonl.gz (replies).
"""

import gzip
import json
import random
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CACHE  # noqa: E402

VOL = "/vol"
app = modal.App("pluralism-judge-rl-score")
volume = modal.Volume.from_name("pluralism-judge-rl", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.10.1.1", "transformers==4.55.4", "sentence-transformers==5.1.0", "hf_transfer==0.1.9", "numpy"
    )
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "common.py", "/root/common.py")
    .add_local_file(HERE.parent / "pluralism-explainer" / "plural.py", "/root/ep1/plural.py")
    .add_local_file(HERE.parent / "pluralism-explainer" / "data" / "questions.json", "/root/ep1/data/questions.json")
    .add_local_dir(HERE / "data", "/root/data")
)
EMBEDDER = "sentence-transformers/all-mpnet-base-v2"
N_REP, N_COV, N_SHOWN = 8, 30, 20


def first_words(text, n=50):
    return " ".join(text.split()[:n])


def score_job(llm, tok, embedder, job):
    """job: {tag, adapter (Volume path or None), seed}. Returns (measures, samples)."""
    sys.path.insert(0, "/root")
    import numpy as np
    from common import (
        COUNTRIES,
        POLICY,
        SYSTEM,
        canonical_options,
        chat_messages,
        coverage,
        entropy,
        js_distance,
        mean_pairwise_cosine,
        option_distribution,
        parse_letter,
        read_jsonl,
        steer_prompt,
        survey_prompt,
    )
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    t0 = time.time()
    lora = None
    if job.get("adapter"):
        lora = LoRARequest(job["tag"], job["lora_id"], job["adapter"])
    seed = job.get("seed", 0)

    def text_of(user, system=SYSTEM):
        return tok.apply_chat_template(
            chat_messages(user, system), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def gen(prompts, params):
        return llm.generate([text_of(p) for p in prompts], params, lora_request=lora, use_tqdm=False)

    out = {"tag": job["tag"], "adapter": job.get("adapter"), "seed": seed, "model": POLICY}
    samples = []

    # ---- repetition (both tasks)
    rep = {}
    for task in ["creative", "advice"]:
        items = read_jsonl(f"/root/data/{task}_heldout.jsonl")
        params = [
            SamplingParams(
                n=N_REP, temperature=1.0, top_p=1.0, repetition_penalty=1.0, max_tokens=384, seed=seed * 100_000 + i
            )
            for i in range(len(items))
        ]
        outs = gen([it["prompt"] for it in items], params)
        per_prompt = []
        for it, o in zip(items, outs):
            texts = [c.text for c in o.outputs]
            full = embedder.encode(texts, normalize_embeddings=True)
            short = embedder.encode([first_words(t) for t in texts], normalize_embeddings=True)
            per_prompt.append(
                {
                    "id": it["id"],
                    "rep": mean_pairwise_cosine(full),
                    "rep50": mean_pairwise_cosine(short),
                    "words": float(np.mean([len(t.split()) for t in texts])),
                    "trunc": float(np.mean([c.finish_reason == "length" for c in o.outputs])),
                    "distinct_exact": len(set(texts)),
                }
            )
            samples += [{"task": task, "id": it["id"], "k": k, "text": t} for k, t in enumerate(texts)]
        rep[task] = per_prompt
    out["repetition"] = rep

    # ---- coverage
    lists = read_jsonl("/root/data/list_prompts.jsonl")
    params = [
        SamplingParams(
            n=N_COV, temperature=1.0, top_p=1.0, repetition_penalty=1.0, max_tokens=16, seed=seed * 100_000 + 50_000 + i
        )
        for i in range(len(lists))
    ]
    outs = gen([it["prompt"] + " Reply with the answer only." for it in lists], params)
    cov = []
    for it, o in zip(lists, outs):
        replies = [c.text for c in o.outputs]
        strict, valid_rate, loose = coverage(replies, it["valid"])
        cov.append({"id": it["id"], "coverage": strict, "valid_rate": valid_rate, "coverage_loose": loose, "replies": replies})
    out["coverage"] = cov

    # ---- steerability: a target distribution shown as 20 answers; read the next answer's first-token distribution
    steer = []
    rng = random.Random(1234)  # targets and shown answers are identical for every checkpoint
    prompts, metas = [], []
    for it in [x for x in lists if x["steer"]]:
        opts = canonical_options(it["valid"])
        first_ids = [tok(o, add_special_tokens=False)["input_ids"][0] for o in opts]
        if len(set(first_ids)) < len(first_ids):
            continue  # two options share a first token: the readout cannot tell them apart
        for t in range(2):
            target = np.random.default_rng(1000 * t + int(it["id"][1:])).dirichlet([0.5] * len(opts)).tolist()
            shown = rng.choices(opts, weights=target, k=N_SHOWN)
            emp = [shown.count(o) / N_SHOWN for o in opts]
            prompts.append(steer_prompt(it, shown))
            metas.append((it["id"], t, opts, first_ids, emp))
    outs = gen(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
    for (iid, t, opts, first_ids, emp), o in zip(metas, outs):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        probs = [float(np.exp(lp[i].logprob)) if i in lp else 0.0 for i in first_ids]
        mass = sum(probs)
        dist = [p / mass for p in probs] if mass > 0 else [1 / len(opts)] * len(opts)
        steer.append(
            {"id": iid, "target": t, "steerability": 1 - js_distance(emp, dist), "mass": mass, "model_dist": dist, "shown_dist": emp}
        )
    out["steer"] = steer

    # ---- survey
    qs = read_jsonl("/root/data/survey.jsonl")
    conds = [None] + COUNTRIES
    sprompts = [(q, c) for q in qs for c in conds]
    texts = [survey_prompt(q, c) for q, c in sprompts]
    lp_outs = gen(texts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
    smp_params = [
        SamplingParams(max_tokens=8, temperature=1.0, top_p=1.0, repetition_penalty=1.0, seed=seed * 100_000 + 90_000 + i)
        for i in range(len(texts))
    ]
    smp_outs = gen(texts, smp_params)
    survey = []
    for (q, c), o1, o2 in zip(sprompts, lp_outs, smp_outs):
        lp = o1.outputs[0].logprobs[0] if o1.outputs[0].logprobs else {}
        dist, mass = option_distribution({v.decoded_token: v.logprob for v in lp.values()}, len(q["options"]))
        survey.append(
            {
                "id": q["id"],
                "cond": c or "default",
                "dist": dist,
                "mass": mass,
                "entropy": entropy(dist),
                "valid": parse_letter(o2.outputs[0].text, len(q["options"])) is not None,
                "jsd": {k: js_distance(dist, q["splits"][k]) for k in COUNTRIES},
            }
        )
    out["survey"] = survey
    out["seconds"] = time.time() - t0
    return out, samples


@app.function(image=image, gpu="L40S", cpu=4.0, memory=32768, timeout=4 * 60 * 60, volumes={VOL: volume}, max_containers=4)
def score_remote(jobs: list):
    sys.path.insert(0, "/root")
    from common import POLICY
    from sentence_transformers import SentenceTransformer
    from vllm import LLM

    llm = LLM(
        model=POLICY,
        gpu_memory_utilization=0.80,
        max_model_len=2048,
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=2,
        max_logprobs=20,
        seed=0,
    )
    tok = llm.get_tokenizer()
    embedder = SentenceTransformer(EMBEDDER, device="cuda")
    results = []  # no volume.reload(): a fresh container already sees every committed adapter
    for i, job in enumerate(jobs):
        job["lora_id"] = i + 1
        measures, samples = score_job(llm, tok, embedder, job)
        d = Path(VOL) / "scores"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{job['tag']}.json").write_text(json.dumps(measures))
        with gzip.open(d / f"{job['tag']}.samples.jsonl.gz", "wt") as f:
            f.write("".join(json.dumps(s) + "\n" for s in samples))
        volume.commit()
        print(f"{job['tag']}: scored in {measures['seconds']:.0f} s", flush=True)
        results.append({"measures": measures, "samples": samples})
    return results


@app.function(image=image, gpu="L40S", cpu=4.0, memory=32768, timeout=3 * 60 * 60, volumes={VOL: volume}, max_containers=4)
def survey_ep1_remote(jobs: list):
    """Episode 1's survey eval, run as-is (experiments/pluralism-explainer/plural.py): its 181 questions, its prompt as
    one user message (no system prompt, thinking off), 50 samples at T = 1, max 32 tokens, its parser. Returns, per
    job, the answer counts for every question and condition."""
    sys.path[:0] = ["/root/ep1", "/root"]
    import plural as P
    from common import POLICY
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=POLICY, gpu_memory_utilization=0.85, max_model_len=2048, dtype="bfloat16", enable_lora=True,
              max_lora_rank=16, max_loras=2, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    qs = P.load_questions()
    pairs = [(q, c) for q in qs for c in P.CONDITIONS]
    texts = [
        tok.apply_chat_template([{"role": "user", "content": P.prompt_text(q, c)}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        for q, c in pairs
    ]
    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        sp = SamplingParams(n=P.N_SAMPLES, temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=32,
                            seed=job.get("seed", 0))  # Episode 1 uses seed 0; the base rerun (floor) uses seed 1
        lora = LoRARequest(job["tag"], i + 1, job["adapter"]) if job.get("adapter") else None
        outs = llm.generate(texts, sp, lora_request=lora, use_tqdm=False)
        rows = []
        for (q, c), o in zip(pairs, outs):
            choices = [P.parse(x.text, len(q["options"])) for x in o.outputs]
            rows.append({"id": q["id"], "cond": c, "counts": P.counts(choices, len(q["options"])).tolist(),
                         "invalid": sum(ch < 0 for ch in choices)})
        results.append({"tag": job["tag"], "rows": rows, "seconds": time.time() - t0})
        print(f"{job['tag']}: Episode 1 survey in {time.time() - t0:.0f} s", flush=True)
    return results


@app.function(image=image, gpu="L40S", cpu=4.0, memory=32768, timeout=60 * 60, volumes={VOL: volume})
def sample_remote(prompts: list, n: int, seed: int):
    """n base replies (T = 1, 384 tokens, training system prompt) per prompt, for the judge search's dev replies."""
    sys.path.insert(0, "/root")
    from common import POLICY, SYSTEM, chat_messages
    from vllm import LLM, SamplingParams

    llm = LLM(model=POLICY, gpu_memory_utilization=0.85, max_model_len=2048, dtype="bfloat16", seed=seed)
    tok = llm.get_tokenizer()
    texts = [
        tok.apply_chat_template(chat_messages(p, SYSTEM), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for p in prompts
    ]
    params = SamplingParams(n=n, temperature=1.0, top_p=1.0, repetition_penalty=1.0, max_tokens=384, seed=seed)
    outs = llm.generate(texts, params, use_tqdm=False)
    return [[c.text for c in o.outputs] for o in outs]


def save_local(res):
    m, s = res["measures"], res["samples"]
    (CACHE / "scores").mkdir(parents=True, exist_ok=True)
    (CACHE / "samples").mkdir(parents=True, exist_ok=True)
    (CACHE / "scores" / f"{m['tag']}.json").write_text(json.dumps(m))
    with gzip.open(CACHE / "samples" / f"{m['tag']}.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(x) + "\n" for x in s))


def summary_line(m):
    def avg(xs):
        return sum(xs) / len(xs)

    dflt = [r for r in m["survey"] if r["cond"] == "default"]
    return (
        f"{m['tag']}: rep creative {avg([r['rep'] for r in m['repetition']['creative']]):.3f} "
        f"advice {avg([r['rep'] for r in m['repetition']['advice']]):.3f} | "
        f"coverage {avg([r['coverage'] for r in m['coverage']]):.3f} | steer {avg([r['steerability'] for r in m['steer']]):.3f} | "
        f"survey H {avg([r['entropy'] for r in dflt]):.3f} mass {avg([r['mass'] for r in m['survey']]):.3f} "
        f"valid {avg([r['valid'] for r in m['survey']]):.3f} ({m['seconds']:.0f} s)"
    )


@app.local_entrypoint()
def score(runs: str = "", steps: str = "", base_seeds: str = "", split: int = 1, force: bool = False):
    """Score the saved checkpoints of the named runs (comma-separated), plus the base model at each seed in base_seeds.
    Skips checkpoints already scored locally unless --force. split: how many L40S containers to spread jobs over."""
    from budget import check_modal, job_cost

    jobs = [{"tag": f"base-s{s}", "adapter": None, "seed": int(s)} for s in base_seeds.split(",") if s]
    want = {int(s) for s in steps.split(",") if s}
    for run in [r for r in runs.split(",") if r]:
        curve_path = HERE / "results" / "runs" / run / "curve.jsonl"
        rows = [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()]
        for st in [r["step"] for r in rows if r.get("checkpoint")]:
            if st == 0 or (want and st not in want):
                continue  # step 0 is the base model (zero-initialised LoRA), scored once as base-s0
            jobs.append({"tag": f"{run}-step{st}", "adapter": f"{VOL}/runs/{run}/step{st}", "seed": 0})
    if not force:
        jobs = [j for j in jobs if not (CACHE / "scores" / f"{j['tag']}.json").exists()]
    if not jobs:
        print("nothing to score")
        return
    split = max(1, min(split, len(jobs)))
    check_modal(job_cost("L40S", 0.15 * len(jobs) + 0.1 * split), f"score {len(jobs)} checkpoints")
    chunks = [jobs[i::split] for i in range(split)]
    for res_list in score_remote.map(chunks, return_exceptions=True):
        if isinstance(res_list, Exception):
            print("FAILED:", repr(res_list))
            continue
        for res in res_list:
            save_local(res)
            print(summary_line(res["measures"]))


@app.local_entrypoint()
def sample_base(n: int = 2, seed: int = 7):
    """Base replies to 60 creative + 60 advice TRAINING prompts (the judge search's dev replies)."""
    from budget import check_modal, job_cost
    from common import DATA, read_jsonl, write_jsonl

    out = CACHE / "judge_dev_replies.jsonl"
    if out.exists():
        print(f"{out} exists")
        return
    check_modal(job_cost("L40S", 0.3), "sample_base")
    rows = []
    for task in ["creative", "advice"]:
        items = read_jsonl(DATA / f"{task}_train.jsonl")[:60]
        replies = sample_remote.remote([it["prompt"] for it in items], n, seed)
        rows += [
            {"task": task, "id": it["id"], "prompt": it["prompt"], "k": k, "reply": r}
            for it, rs in zip(items, replies)
            for k, r in enumerate(rs)
        ]
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} dev replies to {out}")


@app.local_entrypoint()
def survey_ep1(runs: str = "", base_seeds: str = "0", split: int = 1):
    """Episode 1's survey eval on the base model and on every saved checkpoint of the named runs."""
    from budget import check_modal, job_cost

    jobs = [{"tag": f"base-s{s}", "adapter": None, "seed": int(s)} for s in base_seeds.split(",") if s]
    for run in [r for r in runs.split(",") if r]:
        rows = [json.loads(line) for line in (HERE / "results" / "runs" / run / "curve.jsonl").read_text().splitlines() if line.strip()]
        jobs += [{"tag": f"{run}-step{r['step']}", "adapter": f"{VOL}/runs/{run}/step{r['step']}"} for r in rows if r.get("checkpoint") and r["step"] > 0]
    out_dir = CACHE / "ep1"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in jobs if not (out_dir / f"{j['tag']}.json").exists()]
    if not jobs:
        print("nothing to do")
        return
    split = max(1, min(split, len(jobs)))
    check_modal(job_cost("L40S", 0.06 * len(jobs) + 0.1 * split), f"Episode 1 survey on {len(jobs)} checkpoints")
    for res in survey_ep1_remote.map([jobs[i::split] for i in range(split)], return_exceptions=True):
        if isinstance(res, Exception):
            print("FAILED:", repr(res))
            continue
        for r in res:
            (out_dir / f"{r['tag']}.json").write_text(json.dumps(r))
            inv = sum(x["invalid"] for x in r["rows"]) / (50 * len(r["rows"]))
            print(f"{r['tag']}: invalid share {inv:.3f} ({r['seconds']:.0f} s)")
