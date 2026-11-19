"""Run the Spectrum Tuning authors' evaluation code (github.com/tsor13/spectrum @ 5a33588) on one model.

Runs inside a container where the ``spectrum`` package is installed and the working directory holds its ``data/``
folder (see modal_app.py). Episode 3 reuses ``run_steerability`` and ``run_coverage`` on its checkpoints.

What is the authors' code and what is ours:
- steerability: their ``get_example_k``, ``load_eval_dataset`` and ``evaluate_model`` from ``eval_icl.py``, called in a
  loop so one loaded model serves every task; their wandb logging is replaced by a stub and the per-output
  ``raw_results`` are returned instead.
- coverage: their task registry and ``evaluate_model_on_task``; sampling settings are pinned on the model's
  generation_config (temperature 1, top-p 1, top-k off, repetition penalty 1) so hidden per-model defaults cannot
  differ between base and instruct models.
- Qwen3.5 adapter: its chat template opens a ``<think>`` block by default; we default ``enable_thinking=False`` so
  the authors' Qwen3 mapping produces the same empty think block it produces for Qwen3.
"""

from __future__ import annotations

import functools
import os
import time
from pathlib import Path

SPECTRUM_TEST_TASKS = [  # launch_configs/icl_spectrum.yaml
    "novacomet_hypothesis", "novacomet_premise", "habermas_question", "habermas_opinions", "habermas_individual",
    "numbergame_perc", "globaloqa", "chatbotarena_prompts", "chatbotarena_assistant", "chemistry_esol",
    "chemistry_oxidative", "habermas_individual_categorical", "wvs_individual", "numbergame_individual",
    "chatbotarena_individual_prefs", "flight",
]
NOT_FETCHABLE = {"wvs_individual": "WVS wave-7 microdata needs a registration form with personal details"}


def fetch_spectrum_data(root: str) -> dict:
    """Download every source the test tasks and coverage tasks need into ``root/data`` (idempotent)."""
    import requests

    data = Path(root) / "data"
    status = {}

    def get(url, dest):
        dest = Path(dest)
        if dest.exists() and dest.stat().st_size > 0:
            return "cached"
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return f"{len(r.content)} bytes"

    gcs = "https://storage.googleapis.com/habermas_machine/datasets/"
    for f in ["hm_all_candidate_comparisons", "hm_all_final_preference_rankings", "hm_all_position_statement_ratings",
              "hm_all_round_survey_responses"]:
        status[f"habermas/{f}"] = get(gcs + f + ".parquet", data / "habermas_data" / f"{f}.parquet")
    gh = "https://raw.githubusercontent.com/ur-whitelab/BO-ICL/refs/heads/main/paper/dataset/"
    for rel, name in [("oxidative_methane_coupling.csv", "oxidative_methane_coupling.csv"),
                      ("data/esol_iupac.csv", "esol_iupac.csv"), ("data/train_data_num_feats.csv", "train_data_num_feats.csv")]:
        status[f"chemistry/{name}"] = get(gh + rel, data / "chemistry" / name)
    # Number game (Bigelow & Piantadosi): public Harvard Dataverse file 2696204, original (CSV) format
    try:
        status["numbergame"] = get("https://dataverse.harvard.edu/api/access/datafile/2696204?format=original",
                                   data / "numbergame" / "numbergame_data.csv")
    except Exception as e:  # named, not skipped
        status["numbergame"] = f"FAILED: {e}"
    dv = data / "diverse_valid_data"
    status["colors"] = get("https://raw.githubusercontent.com/meodai/color-names/main/src/colornames.csv",
                           dv / "colornames.csv")
    status["periodic"] = get("https://raw.githubusercontent.com/andrejewski/periodic-table/master/data.csv",
                             dv / "periodic_table_elements.csv")
    status["gerunds"] = get("https://raw.githubusercontent.com/gutfeeling/word_forms/master/word_forms/en-verbs.txt",
                            dv / "en-verbs.txt")
    cars = dv / "us-car-models-data"
    if not cars.exists():  # GitHub archive instead of git (no git in the image)
        import io
        import zipfile

        r = requests.get("https://github.com/abhionlyone/us-car-models-data/archive/refs/heads/master.zip", timeout=600)
        r.raise_for_status()
        cars.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    (cars / Path(name).name).write_bytes(z.read(name))
    status["cars"] = f"{len(list(cars.glob('*.csv')))} csv files"
    for t, why in NOT_FETCHABLE.items():
        status[t] = f"NOT FETCHED: {why}"
    return status


def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tok.pad_token = tok.eos_token  # as in eval_icl.evaluate_model_on_dataset
    if "Qwen3.5" in model_name:
        tok.apply_chat_template = functools.partial(tok.apply_chat_template, enable_thinking=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda",
                                                 trust_remote_code=True)
    model.eval()
    return tok, model


def _patch_hf_csv():
    """datasets 3.6 (the authors' pin) holds fsspec at 2025.3, which pandas refuses for hf:// paths; fetch the file
    with huggingface_hub instead (used by globaloqa's loader)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    if getattr(pd.read_csv, "_hf_patched", False):
        return
    orig = pd.read_csv

    def read_csv(path, *a, **k):
        if isinstance(path, str) and path.startswith("hf://datasets/"):
            parts = path[len("hf://datasets/"):].split("/")
            path = hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]), repo_type="dataset")
        return orig(path, *a, **k)

    read_csv._hf_patched = True
    pd.read_csv = read_csv


class _WandbStub:
    def __getattr__(self, name):
        return lambda *a, **k: None


def run_steerability(model_name: str, fmt: str, tasks: list[str], max_eval: int, out_dir: str,
                     eval_seed: int = 42, batch_size: int = 1, tok_model=None) -> dict:
    """Score one model on Spectrum Suite tasks; writes ``<out_dir>/<task>.parquet`` (one row per scored output)."""
    import numpy as np
    from spectrum.icl_classes import eval_icl as E

    E.wandb = _WandbStub()
    # The release imports generate_habermas_opinions_data but never registers it; register it under its own name.
    from spectrum.icl_classes import dataloaders as DL
    from spectrum.icl_classes.icl_dataset_classes.habermas import generate_habermas_opinions_data

    DL.default_data_map.setdefault("habermas_opinions", generate_habermas_opinions_data)
    _patch_hf_csv()
    tok, model = tok_model or load_model(model_name)
    # The authors ran float32 weights (transformers 4.x default); we load bfloat16 and hand their code float32
    # logits, since it converts log-probs to numpy (which has no bfloat16).
    hook = model.register_forward_hook(lambda m, i, o: (setattr(o, "logits", o.logits.float()), o)[1])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for task in tasks:
        dest = out / f"{task}.parquet"
        if dest.exists():
            report[task] = "cached"
            continue
        if task in NOT_FETCHABLE:
            report[task] = "not fetched"
            continue
        t0 = time.time()
        try:
            args = {"permutations_per_iter": 1, "include_prior_prob": 1, "include_data_formatter_description_prob": 0,
                    "include_replacement_prob": 0, "include_tags_prob": 0, "format": fmt, "tokenizer": tok}
            # The release keys its token cache by model family ("Qwen3" also matches Qwen3.5, whose tokenizer
            # differs), so one family member can load another's token ids. One cache per model avoids that.
            cache = f"tokenized_eval_datasets_by_model/{model_name.replace('/', '__')}"
            ds_args = {"dataset_name": task, "actual_model_name": model_name, "cache_dir": cache,
                       "max_length": None, "dataset_kwargs": args, "tokenizer": tok, "restrict_texts": None,
                       "eval_seed": eval_seed, "max_eval_examples": max_eval}
            k_args = E.get_example_k(**ds_args)
            ds_args["dataset_kwargs"]["keep_inds"] = k_args["keep_inds"]
            ds_args["dataset_kwargs"]["fixed_example_number"] = k_args["k"]
            val = E.load_eval_dataset(**ds_args)
            res = E.evaluate_model(model, tok, val, batch_size=batch_size, wandb_project="stub",
                                   fixed_example_number=k_args["k"])
            raw = res["raw_results"].drop(columns=["correct_logprobs", "incorrect_logprobs", "restricted_logprobs"],
                                          errors="ignore")
            raw["k"] = k_args["k"]
            raw["task"] = task
            raw["model"] = model_name
            raw["format"] = fmt
            raw.to_parquet(dest)
            report[task] = {"n_instances": int(raw.instance_ind.nunique()), "k": int(k_args["k"]),
                            "seconds": round(time.time() - t0, 1),
                            "loss_geq_1": float(np.mean(raw[raw.example_ind >= 1].loss))}
        except Exception as e:  # report and continue; a failed task is named in the note
            import traceback

            report[task] = f"FAILED: {type(e).__name__}: {str(e)[:300]}\n{traceback.format_exc()[-2500:]}"
    hook.remove()
    return report


def run_coverage(model_name: str, template: str, out_path: str, n: int = 100, seeds=(0, 1),
                 components=("description", "examples", "both"), batch_size: int = 100, tok_model=None,
                 task_subset: list[str] | None = None) -> dict:
    """Generate ``n`` samples per coverage task, setting and seed; writes one JSONL row per (task, setting, seed)."""
    import json

    import torch
    from spectrum.diverse_valid.eval_diverse_valid import evaluate_model_on_task
    from spectrum.diverse_valid.task_registry import tasks

    tok, model = tok_model or load_model(model_name)
    gc = model.generation_config
    gc.temperature, gc.top_p, gc.top_k, gc.repetition_penalty = 1.0, 1.0, 0, 1.0
    gc.min_p = None
    gc.typical_p = 1.0
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {(r["task"], r["setting"], r["seed"]) for r in map(json.loads, f)}
    report = {}
    for task_name, factory in sorted(tasks.items()):
        if task_subset and task_name not in task_subset:
            continue
        for setting in components:
            for seed in seeds:
                if (task_name, setting, seed) in done:
                    continue
                torch.manual_seed(seed)
                task = factory()
                t0 = time.time()
                try:
                    r = evaluate_model_on_task(model=model, tokenizer=tok, task=task, num_generations=n,
                                               template=template, gen_batch_size=batch_size,
                                               include_description=setting in ("description", "both"),
                                               include_examples=setting in ("examples", "both"))
                    row = {"task": task_name, "setting": setting, "seed": seed, "model": model_name,
                           "template": template, "generations": r["generations"],
                           "validity": [bool(v) for v in r["validity"]], "seconds": round(time.time() - t0, 1)}
                except Exception as e:
                    row = {"task": task_name, "setting": setting, "seed": seed, "model": model_name,
                           "template": template, "error": f"{type(e).__name__}: {str(e)[:300]}"}
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                report[f"{task_name}/{setting}/{seed}"] = row.get("error") or sum(row["validity"])
    return report
