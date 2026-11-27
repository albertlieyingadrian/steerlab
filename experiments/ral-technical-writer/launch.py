"""
Spawn the study's Modal jobs (deployed app `ral-writer`, workspace credits30) after budget.py's check.

    MODAL_PROFILE=credits30 MODAL_IMAGE_BUILDER_VERSION=2025.06 modal deploy experiments/ral-technical-writer/modal_ral.py
    python launch.py <job> [<job> ...]   # spawn; call ids appended to results/calls.jsonl
    python launch.py status              # state of every spawned call

Jobs are defined in JOBS below (configs written into the plan); each has a projected cost for the cap check.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MODAL_PROFILE", "credits30")
import modal

import budget
import common as C

CALLS = C.RESULTS / "calls.jsonl"
GEN, LLAMA = C.GEN, "meta-llama/Llama-3.3-70B-Instruct"

RAL = {"steps": 200, "prompts_per_step": 16, "group": 8, "max_tokens": 2048, "lora_rank": 32, "lr": 2e-5,
       "judge_lr": 2e-5, "micro_batch": 4, "gate_threshold": 0.75, "gate_max_skip": 20, "replay_fraction": 0.5,
       "replay_capacity": 100_000, "target_ema": 0.95, "ckpt_every": 25, "eval_every": 25, "seed": 0,
       "warm_samples": "base_train1024", "judge_start": "judge_gate", "judge_ruler": "judge0", "gpu_util": 0.25, "max_model_len": 3072}  # fmt: skip

JOBS: dict[str, tuple[str, dict, float]] = {
    # name: (function, payload, projected USD)
    "s_warm": ("sample_remote", {"tag": "base_train1024", "split": "train", "limit": 1024, "model": GEN,
                                 "thinking": True, "n": 1, "max_tokens": 2048}, 1.0),
    "s_dev": ("sample_remote", {"tag": "base_dev", "split": "dev", "model": GEN, "thinking": True, "n": 2,
                                "max_tokens": 2048}, 0.5),
    "s_base": ("sample_remote", {"tag": "base_heldout", "split": "heldout", "model": GEN, "thinking": True, "n": 4,
                                 "max_tokens": 2048}, 0.7),
    "s_base_fs": ("sample_remote", {"tag": "basefs_heldout", "split": "heldout", "model": GEN, "thinking": True,
                                    "n": 4, "max_tokens": 2048, "fewshot": 3}, 0.8),
    "s_llama_fs": ("sample2_remote", {"tag": "llamafs_heldout", "split": "heldout", "model": LLAMA, "thinking": None,
                                      "n": 4, "max_tokens": 1024, "fewshot": 3, "tp": 2,
                                      "max_model_len": 6144, "gpu_util": 0.95, "max_num_seqs": 48,
                                      "max_num_batched_tokens": 6144}, 3.0),
    "warm_judge": ("warm_judge_remote", {"warm_samples": "base_train1024", "dev_samples": "base_dev", "lr": 5e-5,
                                         "epochs": 1, "batch": 32, "out": "judge0"}, 1.5),
    "warm_gate": ("warm_judge_remote", {"warm_samples": "base_train1024", "dev_samples": "base_dev", "lr": 1e-5,
                                        "epochs": 1, "batch": 32, "eval_every": 1, "stop_acc": 0.75,
                                        "out": "judge_gate"}, 1.0),
    "pilot_ral": ("train_remote", {**RAL, "name": "pilot_ral", "arm": "ral", "steps": 12, "eval_every": 6,
                                   "ckpt_every": 6}, 3.0),
    "ral_s0": ("train_remote", {**RAL, "name": "ral_s0", "arm": "ral"}, 35.0),
    "ral_s1": ("train_remote", {**RAL, "name": "ral_s1", "arm": "ral", "seed": 1}, 35.0),
    "frozen_s0": ("train_remote", {**RAL, "name": "frozen_s0", "arm": "frozen"}, 30.0),
    "sft": ("sft_remote", {"seed": 0, "lora_rank": 32, "lr": 1e-4, "epochs": 2, "batch": 16, "micro_batch": 4,
                           "out": "sft/adapter"}, 3.0),
    # ---- held-out samples from the trained arms (step 200, fixed in the plan)
    "s_sft": ("sample_remote", {"tag": "sft_heldout", "split": "heldout", "model": GEN, "adapter": "sft/adapter",
                                "thinking": False, "n": 4, "max_tokens": 1024}, 0.7),
    "s_ral": ("sample_remote", {"tag": "ral_heldout", "split": "heldout", "model": GEN,
                                "adapter": "runs/ral_s0/ckpt_200/gen", "thinking": True, "n": 4,
                                "max_tokens": 2048}, 0.7),
    "s_ral1": ("sample_remote", {"tag": "ral1_heldout", "split": "heldout", "model": GEN,
                                 "adapter": "runs/ral_s1/ckpt_200/gen", "thinking": True, "n": 4,
                                 "max_tokens": 2048}, 0.7),
    "s_frozen": ("sample_remote", {"tag": "frozen_heldout", "split": "heldout", "model": GEN,
                                   "adapter": "runs/frozen_s0/ckpt_200/gen", "thinking": True, "n": 4,
                                   "max_tokens": 2048}, 0.7),
    # ---- held-out judge ensemble (pairwise, both orders) and the rater
    "j_gptoss": ("pairwise_remote", {"model": "openai/gpt-oss-120b", "tag": "gptoss", "pairs": "pairs/heldout.jsonl",
                                     "reasoning": "low", "max_tokens": 2048, "max_model_len": 8192, "gpu_util": 0.95,
                                     "max_num_seqs": 64, "max_num_batched_tokens": 8192}, 3.0),
    "j_gemma": ("pairwise_remote", {"model": "google/gemma-3-27b-it", "tag": "gemma", "pairs": "pairs/heldout.jsonl",
                                    "logprobs": True}, 2.0),
    "j_phi4": ("pairwise_remote", {"model": "microsoft/phi-4", "tag": "phi4", "pairs": "pairs/heldout.jsonl",
                                   "logprobs": True, "max_model_len": 16384}, 1.5),
    "j_glm": ("pairwise2_remote", {"model": "zai-org/GLM-4.5-Air-FP8", "tag": "glm", "pairs": "pairs/heldout.jsonl",
                                   "max_tokens": 3072, "max_model_len": 12288, "tp": 2}, 5.0),
    "rate_gemma": ("rate_remote", {"model": "google/gemma-3-27b-it", "tag": "gemma", "items": "rate/heldout.jsonl"},
                   2.0),
    # ---- the trained judges as loose rulers on held-out
    "js_j0": ("jscore_remote", {"tag": "j0", "weights": "judge0/judge.safetensors", "items": "rate/heldout.jsonl"},
              0.5),
    "js_start": ("jscore_remote", {"tag": "j_start", "weights": "judge_gate/judge.safetensors",
                                   "items": "rate/heldout.jsonl"}, 0.5),
    "js_ral": ("jscore_remote", {"tag": "ral_target", "weights": "runs/ral_s0/ckpt_200/judge_target.safetensors",
                                 "items": "rate/heldout.jsonl"}, 0.5),
}  # fmt: skip


def spawn(name: str) -> None:
    fn_name, payload, usd = JOBS[name]
    budget.check_modal(usd, name)
    fn = modal.Function.from_name("ral-writer", fn_name)
    call = fn.spawn(payload)
    CALLS.parent.mkdir(parents=True, exist_ok=True)
    with CALLS.open("a") as f:
        f.write(json.dumps({"time": budget.now(), "job": name, "fn": fn_name, "call_id": call.object_id}) + "\n")
    print(f"spawned {name}: {call.object_id}")


def status() -> None:
    seen = {}
    for r in C.read_jsonl(CALLS):
        seen[r["job"]] = r
    for job, r in seen.items():
        fc = modal.FunctionCall.from_id(r["call_id"])
        try:
            res = fc.get(timeout=0)
            state = f"done {json.dumps(res)[:160]}"
        except TimeoutError:
            state = "running"
        except Exception as e:
            state = f"FAILED {type(e).__name__}: {str(e)[:200]}"
        print(f"{job:14s} {r['time']}  {state}")


if __name__ == "__main__":
    if sys.argv[1] == "status":
        status()
    else:
        for n in sys.argv[1:]:
            spawn(n)
            time.sleep(1)
