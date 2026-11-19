"""Modal jobs for Episode 2: Spectrum steerability (A) and coverage (B), open-model Hivemind samples (C, D) and open
judges (E). COSTS MONEY: H100 time, hard cap $50 for the episode (ledger: results/spend_modal.jsonl).

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/pluralism-replications/modal_app.py::fetch
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/pluralism-replications/modal_app.py::pilot
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/pluralism-replications/modal_app.py::full

Each model gets one container that loads it once and runs every job for that model; outputs go to the
``pluralism-replications`` volume under /vol/out/<run>/<model>/ and come back as a tar into results/raw/<run>/.
Episode 3 can reuse ``image`` and ``spectrum_runner`` (see README).
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "/vol"
SPECTRUM = "spectrum @ https://github.com/tsor13/spectrum/archive/5a33588ec4edfc54b74a318732d8cee89e2e2c7a.zip"
H100_PER_SEC = 0.00138  # H100 list price used across this repo ($4.97/h); charged for every GPU type (A100/L40S cost less), so the ledger overstates
MODAL_CAP_USD = 50.0
LEDGER = HERE / "results" / "spend_modal.jsonl"

volume = modal.Volume.from_name("pluralism-replications", create_if_missing=True)
deps = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1", "transformers==5.22.2", "accelerate", "datasets==3.6.0", "trl", "numpy<2.4", "pandas",
        "pyarrow", "fastparquet", "scipy", "pyreadstat", "requests", "huggingface_hub", "hf_transfer==0.1.9",
        "sentencepiece", "protobuf", "wandb",
    )
    .pip_install(SPECTRUM, extra_options="--no-deps")
    .pip_install("flash-linear-attention==0.5.2")
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "WANDB_MODE": "disabled",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"})
)
image = deps.add_local_dir(HERE, remote_path="/root/exp", ignore=["results/**", "report/**", "__pycache__/**"])
app = modal.App("pluralism-replications", image=image)
SECRETS = [modal.Secret.from_name("huggingface-secret")]

MODELS = {  # name -> (family, role, spectrum format)
    "google/gemma-3-12b-pt": ("gemma3-12b", "PT", "colon"),
    "google/gemma-3-12b-it": ("gemma3-12b", "IT", "chat"),
    "tsor13/spectrum-gemma-3-12b-v1": ("gemma3-12b", "ST", "spectrum"),
    "Qwen/Qwen3-8B-Base": ("qwen3-8b", "PT", "colon"),
    "Qwen/Qwen3-8B": ("qwen3-8b", "IT", "chat"),
    "Qwen/Qwen3.5-9B-Base": ("qwen3.5-9b", "PT", "colon"),
    "Qwen/Qwen3.5-9B": ("qwen3.5-9b", "IT", "chat"),
}


def _setup():
    import os

    sys.path.insert(0, "/root/exp")
    Path(f"{VOL}/spectrum").mkdir(parents=True, exist_ok=True)
    os.chdir(f"{VOL}/spectrum")  # the authors' code reads data/ relative to the working directory


@app.function(timeout=60 * 60, volumes={VOL: volume}, secrets=SECRETS, cpu=2)
def fetch_remote() -> dict:
    _setup()
    import spectrum_runner as S

    status = S.fetch_spectrum_data(f"{VOL}/spectrum")
    volume.commit()
    return status


# ---------------------------------------------------------------------------------------------------------------------
# Hivemind generation (C, D) and judging (E) on open models
# ---------------------------------------------------------------------------------------------------------------------


def _chat_prompt(tok, text: str, base: bool) -> str:
    if base:
        return f"User: {text}\nAssistant:"
    kw = {"enable_thinking": False} if "Qwen" in (tok.name_or_path or "") else {}
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                   **kw)


def _generate(tok, model, prompts: list[str], n: int, max_new: int, top_p: float, temperature: float,
              batch_seqs: int, seed: int, greedy: bool = False) -> list[list[tuple[str, int]]]:
    """Left-padded batched sampling; prompts must already contain any BOS token (add_special_tokens=False)."""
    import torch

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    per = max(1, batch_seqs // n)
    for i in range(0, len(prompts), per):
        chunk = prompts[i : i + per]
        flat = [p for p in chunk for _ in range(n)]
        enc = tok(flat, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(seed + i)
        kw = dict(max_new_tokens=max_new, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        if greedy:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
        with torch.no_grad():
            gen = model.generate(**enc, **kw)
        new = gen[:, enc.input_ids.shape[1] :]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        stop_ids = {tok.pad_token_id, tok.eos_token_id}
        lens = [int(sum(1 for t in row.tolist() if t not in stop_ids)) for row in new]
        for j in range(len(chunk)):
            outs.append([(texts[j * n + s], lens[j * n + s]) for s in range(n)])
    return outs


def _with_bos(tok, text: str) -> str:
    bos = tok.bos_token
    if bos and not text.startswith(bos) and "gemma" in (tok.name_or_path or "").lower():
        return bos + text
    return text


def run_hivemind_gen(model_name: str, prompts: list[dict], out_path: str, n: int, base: bool, tok_model) -> int:
    tok, model = tok_model
    texts = [_with_bos(tok, _chat_prompt(tok, p["query"], base)) for p in prompts]
    res = _generate(tok, model, texts, n=n, max_new=512, top_p=0.9, temperature=1.0, batch_seqs=200, seed=0)
    with open(out_path, "w") as f:
        for p, samples in zip(prompts, res):
            for s, (text, ntok) in enumerate(samples):
                if base:  # a base model continues the transcript; keep its first assistant turn
                    text = text.split("\nUser:")[0].split("\nAssistant:")[0]
                f.write(json.dumps({"model": model_name, "prompt_id": p["prompt_id"], "sample": s, "text": text.strip(),
                                    "n_tokens": ntok, "truncated": ntok >= 512}) + "\n")
    return len(res) * n


def run_judge(model_name: str, items: list[dict], out_path: str, tok_model) -> int:
    import judge_prompt as J

    tok, model = tok_model
    texts = []
    for it in items:
        msgs = [{"role": "user", "content": J.overall_prompt(it["user_query"], it["response"])}]
        kw = {"enable_thinking": False} if "Qwen" in model_name else {}
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except Exception:  # a checkpoint without a chat template: Mistral instruct format
            text = f"<s>[INST] {msgs[0]['content']} [/INST]"
        texts.append(_with_bos(tok, text))
    res = _generate(tok, model, texts, n=1, max_new=400, top_p=1.0, temperature=1.0, batch_seqs=16, seed=0,
                    greedy=True)
    with open(out_path, "w") as f:
        for it, [(text, _)] in zip(items, res):
            f.write(json.dumps({"judge": model_name, "item_id": it["item_id"], "raw": text,
                                "score": J.parse_score(text)}) + "\n")
    return len(items)


@app.function(gpu=["H100", "A100-80GB", "L40S"], timeout=6 * 60 * 60, volumes={VOL: volume}, secrets=SECRETS, max_containers=4)
def model_job(model_name: str, run: str, jobs: list[str], cfg: dict) -> dict:
    """jobs: any of steer:<fmt>, coverage:<template>, hivemind, hivemind_base, judge."""
    _setup()
    import os

    if cfg.get("debug"):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    import spectrum_runner as S

    t0 = time.time()
    out = Path(f"{VOL}/out/{run}/{model_name.replace('/', '__')}")
    out.mkdir(parents=True, exist_ok=True)
    tm = S.load_model(model_name)
    import torch

    report = {"load_seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name()}
    for job in jobs:
        t1 = time.time()
        kind, _, arg = job.partition(":")
        try:
            if kind == "steer":
                tasks = cfg["colon_tasks"] if (arg == "colon" and "colon_tasks" in cfg
                                               and MODELS.get(model_name, ("", "", ""))[1] == "IT") else cfg["steer_tasks"]
                report[job] = S.run_steerability(model_name, arg, tasks, cfg["max_eval"],
                                                 str(out / f"steer_{arg}"), tok_model=tm)
            elif kind == "coverage":
                report[job] = S.run_coverage(model_name, arg, str(out / f"coverage_{arg}.jsonl"), n=cfg["cov_n"],
                                             seeds=tuple(cfg["cov_seeds"]), tok_model=tm,
                                             task_subset=cfg.get("cov_tasks"))
            elif kind in ("hivemind", "hivemind_base"):
                report[job] = run_hivemind_gen(model_name, cfg["hm_prompts"], str(out / f"{kind}.jsonl"),
                                               cfg["hm_n"], base=kind == "hivemind_base", tok_model=tm)
            elif kind == "judge":
                report[job] = run_judge(model_name, cfg["judge_items"], str(out / "judge.jsonl"), tok_model=tm)
        except Exception as e:
            import traceback

            report[job] = f"FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        report[f"{job}__seconds"] = round(time.time() - t1, 1)
        volume.commit()
    report["seconds"] = round(time.time() - t0, 1)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=out.name)
    return {"report": report, "tar": buf.getvalue()}


# ---------------------------------------------------------------------------------------------------------------------
# Local orchestration with a hard spend cap
# ---------------------------------------------------------------------------------------------------------------------


def _spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in LEDGER.read_text().splitlines() if line.strip())


# per-second prices, all scaled from the repo's measured H100 rate (list GPU price plus CPU and memory)
PRICE_PER_SEC = {"H100": H100_PER_SEC, "A100": H100_PER_SEC * 2.50 / 3.95, "L40S": H100_PER_SEC * 1.95 / 3.95}


def _log_spend(label: str, seconds: float, gpu: str = "H100"):
    key = next((k for k in PRICE_PER_SEC if k in gpu), "H100")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "gpu": gpu,
                            "seconds": seconds, "usd": round(seconds * PRICE_PER_SEC[key], 4)}) + "\n")


def _launch(run: str, plan: list[tuple[str, list[str]]], cfg: dict, projected_usd: float):
    spent = _spent()
    if spent + projected_usd > MODAL_CAP_USD:
        raise SystemExit(f"Modal cap: spent ${spent:.2f} + projected ${projected_usd:.2f} > ${MODAL_CAP_USD}")
    print(f"spent so far ${spent:.2f}; projected ${projected_usd:.2f}; launching {len(plan)} containers")
    calls = [(m, model_job.spawn(m, run, jobs, cfg)) for m, jobs in plan]
    raw = HERE / "results" / "raw" / run
    raw.mkdir(parents=True, exist_ok=True)
    for m, call in calls:
        try:
            res = call.get()
        except Exception as e:
            print(f"[{m}] FAILED: {e}")
            continue
        _log_spend(f"{run}:{m}", res["report"]["seconds"] + 60, res["report"].get("gpu", "H100"))  # +60 s for container start and model download
        with tarfile.open(fileobj=io.BytesIO(res["tar"]), mode="r:gz") as tar:
            tar.extractall(raw, filter="data")
        (raw / f"{m.replace('/', '__')}.report.json").write_text(json.dumps(res["report"], indent=1))
        print(f"[{m}] done in {res['report']['seconds']}s; total spent ${_spent():.2f}")


def _hm_prompts(limit=None):
    import pandas as pd

    df = pd.read_parquet("hf://datasets/liweijiang/infinite-chats-eval/data/train-00000-of-00001.parquet")
    rows = [{"prompt_id": i, "query": q} for i, q in enumerate(df["query"])]
    return rows[:limit] if limit else rows


def _judge_items(limit=None):
    import pandas as pd

    df = pd.read_parquet("hf://datasets/liweijiang/infinite-chats-human-absolute/data/train-00000-of-00001.parquet")
    items = [{"item_id": i, "user_query": r.user_query, "response": r.response} for i, r in enumerate(df.itertuples())]
    return items[:limit] if limit else items


@app.local_entrypoint()
def fetch():
    print(json.dumps(fetch_remote.remote(), indent=1))


@app.local_entrypoint()
def pilot(steer_only: bool = False):
    """Dev pilot (discarded): 2 steer tasks x 30 instances, 3 coverage tasks, 5 hivemind prompts, 20 judge items."""
    cfg = {"steer_tasks": ["flight", "chemistry_esol"], "max_eval": 30, "cov_n": 20, "cov_seeds": [0],
           "cov_tasks": ["rng_1_10", "us_states", "element_symbols"], "hm_prompts": _hm_prompts(5), "hm_n": 4,
           "judge_items": _judge_items(20)}
    plan = [
        ("google/gemma-3-12b-pt", ["steer:colon", "coverage:colon", "hivemind_base"]),
        ("google/gemma-3-12b-it", ["steer:chat", "coverage:chat", "hivemind"]),
        ("Qwen/Qwen3.5-9B", ["steer:chat", "coverage:chat", "hivemind"]),
        ("prometheus-eval/prometheus-7b-v2.0", ["judge"]),
    ]
    if steer_only:
        cfg["steer_tasks"] = ["flight", "chemistry_esol", "habermas_opinions", "chatbotarena_assistant"]
        plan = [("google/gemma-3-12b-pt", ["steer:colon"]), ("google/gemma-3-12b-it", ["steer:chat", "steer:colon"]),
                ("Qwen/Qwen3.5-9B", ["steer:chat"])]
    _launch("pilot", plan, cfg, projected_usd=4.0)


@app.local_entrypoint()
def full(max_eval: int = 500, only: str = "", jobs_override: str = "", run: str = "", projected: float = 40.0,
         tasks: str = "", debug: bool = False, hm_limit: int = 0, judge_limit: int = 0):
    import spectrum_runner as S

    # IT-in-colon format control on the short tasks only (budget; see the execution log)
    colon_tasks = ["flight", "habermas_individual_categorical", "numbergame_individual", "chatbotarena_individual_prefs",
                   "chemistry_esol", "chemistry_oxidative", "numbergame_perc", "globaloqa"]
    cfg = {"steer_tasks": S.SPECTRUM_TEST_TASKS, "max_eval": max_eval, "cov_n": 100, "cov_seeds": [0, 1],
           "hm_prompts": _hm_prompts(), "hm_n": 10, "judge_items": _judge_items(), "colon_tasks": colon_tasks}
    plan = []
    for m, (_, role, fmt) in MODELS.items():
        if role == "PT":
            jobs = ["hivemind_base", f"steer:{fmt}", f"coverage:{fmt}"]
        elif role == "IT":
            jobs = ["hivemind", f"steer:{fmt}", f"coverage:{fmt}", "steer:colon"]
            if m == "Qwen/Qwen3-8B":
                jobs.insert(1, "judge")
        else:
            jobs = [f"steer:{fmt}", f"coverage:{fmt}"]
        plan.append((m, jobs))
    plan.append(("prometheus-eval/prometheus-7b-v2.0", ["judge"]))
    if only:
        keep = set(only.split(","))
        plan = [(m, j) for m, j in plan if m in keep]
    if jobs_override:
        plan = [(m, jobs_override.split(",")) for m, _ in plan]
    if debug:
        cfg["debug"] = True
        cfg["hm_prompts"] = cfg["hm_prompts"][:hm_limit] if hm_limit else cfg["hm_prompts"]
        cfg["judge_items"] = cfg["judge_items"][:judge_limit] if judge_limit else cfg["judge_items"]
    if tasks:
        cfg["steer_tasks"] = tasks.split(",")
        cfg["colon_tasks"] = [t for t in cfg["colon_tasks"] if t in cfg["steer_tasks"]]
    _launch(run or f"full{max_eval}", plan, cfg, projected_usd=projected)
