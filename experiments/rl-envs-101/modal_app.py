"""The Modal side of RL environments 101: an open model as a front-desk agent, a vLLM server for Harbor, and GRPO.

COSTS MONEY. Always run with an explicit profile, from the repo root:

    MODAL_PROFILE=albertlieyingadrian modal run experiments/rl-envs-101/modal_app.py::rollouts_main --items F --out G
    MODAL_PROFILE=albertlieyingadrian modal deploy experiments/rl-envs-101/modal_app.py      (the Harbor server)
    MODAL_PROFILE=albertlieyingadrian modal run experiments/rl-envs-101/modal_app.py::train_main --arm visible
    MODAL_PROFILE=albertlieyingadrian modal app stop -y rl-envs-101                       (always, afterwards)

Functions:
    rollouts   vLLM offline on one L40S: run the desk environment's episodes for a list of (task, world) items.
    Server     `vllm serve` behind a Modal web endpoint (OpenAI-compatible), for Harbor's terminus-2 agent.
    train      GRPO on the desk environment: vLLM samples 8 tasks x 8 episodes per step with the current LoRA,
               the verifier scores each episode, advantage = reward - group mean, and the update pushes up the
               tokens the policy wrote in above-average episodes (tool results are masked out of the loss).

Every function has a timeout that caps its cost: L40S $1.95/h on Modal (2026-11), so 75 minutes is at most $2.45.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "rl-envs-101"
GPU = "L40S"
GPU_DOLLARS_PER_HOUR = 1.95
VOL = "/vol"
app = modal.App(APP)
volume = modal.Volume.from_name("rl-envs-101", create_if_missing=True)

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .apt_install("build-essential")  # Triton compiles a small CUDA helper at vLLM start-up
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "peft==0.17.1", "hf_transfer==0.1.9")
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(HERE / "envlab", "/root/envlab")  # last, so code edits do not rebuild the image
)


def _sampler(llm, temperature: float, max_tokens: int, lora=None):
    from vllm import SamplingParams

    params = SamplingParams(temperature=temperature, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=max_tokens)

    def sample(batch: list[list[int]]) -> list[tuple[str, list[int]]]:
        outs = llm.generate([{"prompt_token_ids": ids} for ids in batch], params, lora_request=lora, use_tqdm=False)
        return [(o.outputs[0].text, list(o.outputs[0].token_ids)) for o in outs]

    return sample


def _pairs(items: list[dict], max_turns: int):
    from envlab.desk import DeskEnv, Task, world_from_json

    return [(Task(**it["task"]), DeskEnv(world_from_json(it["world"]), max_turns=max_turns)) for it in items]


def _episode_row(item: dict, episode) -> dict:
    return {"id": item["id"], "messages": episode.messages, "scores": episode.info, "tokens": sum(episode.mask)}


@app.function(image=image, gpu=GPU, timeout=45 * 60, volumes={VOL: volume}, scaledown_window=10)
def rollouts(items: list[dict], model: str, temperature: float = 0.7, samples: int = 1, max_turns: int = 10) -> list:
    """Run each item `samples` times with `model` and return the chats and verifier keys."""
    from transformers import AutoTokenizer
    from vllm import LLM

    from envlab.rollout import run_episodes

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.85, seed=0)
    expanded = [it for it in items for _ in range(samples)]
    t0 = time.time()
    episodes = run_episodes(tokenizer, _sampler(llm, temperature, 256), _pairs(expanded, max_turns), max_turns)
    print(f"{len(episodes)} episodes in {time.time() - t0:.0f}s", flush=True)
    return [_episode_row(it, ep) for it, ep in zip(expanded, episodes)]


@app.local_entrypoint()
def rollouts_main(
    items: str, out: str, model: str = "Qwen/Qwen3-4B-Instruct-2507", samples: int = 1, temperature: float = 0.7
):
    rows = [json.loads(line) for line in Path(items).read_text().splitlines() if line.strip()]
    t0 = time.time()
    result = rollouts.remote(rows, model, temperature, samples)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(r) + "\n" for r in result))
    _ledger(f"rollouts {model} x{len(rows) * samples}", time.time() - t0)


def _ledger(what: str, seconds: float) -> None:
    """An upper-bound cost row per call: wall time from this machine (includes start-up and queueing) x GPU price."""
    row = {"when": time.strftime("%Y-%m-%dT%H:%M"), "what": what, "gpu": GPU, "seconds": round(seconds)}
    row["dollars_max"] = round(seconds / 3600 * GPU_DOLLARS_PER_HOUR, 3)
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "modal_ledger.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ledger:", row)


# --- the vLLM server Harbor's agents talk to ----------------------------------------------------------------------

SERVED = os.environ.get("RLENV_SERVED_MODEL", "Qwen/Qwen3-4B-Instruct-2507")


@app.cls(image=image, gpu=GPU, timeout=3 * 60 * 60, volumes={VOL: volume}, scaledown_window=5 * 60)
@modal.concurrent(max_inputs=32)
class Server:
    @modal.web_server(8000, startup_timeout=1200)
    def serve(self):
        cmd = ["vllm", "serve", SERVED, "--served-model-name", "desk-agent", "--host", "0.0.0.0", "--port", "8000"]
        cmd += ["--max-model-len", "32768", "--gpu-memory-utilization", "0.9", "--generation-config", "vllm"]
        subprocess.Popen(cmd)


# --- GRPO ----------------------------------------------------------------------------------------------------------


@app.function(image=image, gpu=GPU, timeout=100 * 60, volumes={VOL: volume}, scaledown_window=10, retries=1)
def train(config: dict, train_items: list[dict], eval_items: list[dict]) -> dict:
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    from envlab.rollout import run_episodes

    cfg = {"steps": 60, "tasks_per_step": 8, "group": 8, "lr": 1e-4, "rank": 16, "max_turns": 8, "eval_every": 15}
    cfg["train_temperature"] = 0.8
    cfg.update(config)
    run_dir = Path(VOL) / "runs" / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    llm = LLM(
        model=cfg["model"],
        enable_lora=True,
        max_lora_rank=cfg["rank"],
        max_model_len=4096,
        gpu_memory_utilization=0.35,
        seed=cfg.get("seed", 0),
    )
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg["rank"],
        lora_alpha=2 * cfg["rank"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.0)

    def adapter(step: int):
        path = run_dir / f"lora_{step}"
        model.save_pretrained(str(path))
        return LoRARequest(f"s{step}", step, str(path))

    def evaluate(step: int, lora_request) -> dict:
        sample = _sampler(llm, 0.7, 256, lora_request)
        eps = run_episodes(tokenizer, sample, _pairs(eval_items, cfg["max_turns"]), cfg["max_turns"])
        rows = [_episode_row(it, ep) for it, ep in zip(eval_items, eps)]
        (run_dir / f"eval_{step}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return _summary(rows, step, "eval")

    log = []
    lora_request = None  # step 0 samples from the untouched base model
    first = 0
    saved = run_dir / "log.json"
    if saved.exists() and (run_dir / "optimizer.pt").exists():  # resume after a stop: adapter, optimizer, log, RNG
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = torch.load(run_dir / "optimizer.pt", weights_only=False)
        first = state["step"]
        log = [e for e in json.loads(saved.read_text())["log"] if e["step"] <= first]
        set_peft_model_state_dict(model, load_file(str(run_dir / f"lora_{first}" / "adapter_model.safetensors")))
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"])
        lora_request = LoRARequest(f"s{first}", first, str(run_dir / f"lora_{first}"))
        print(f"resumed at step {first}", flush=True)
    for step in range(first, cfg["steps"] + 1):
        if (
            step == first
            and first
            and step % cfg["eval_every"] == 0
            and any(e["split"] == "eval" and e["step"] == step for e in log)
        ):
            pass  # this checkpoint's eval already ran before the stop
        elif step % cfg["eval_every"] == 0:
            log.append(evaluate(step, lora_request))
            print(json.dumps(log[-1]), flush=True)
        if step == cfg["steps"]:
            break
        t0 = time.time()
        chosen = rng.sample(train_items, cfg["tasks_per_step"])
        batch = [it for it in chosen for _ in range(cfg["group"])]
        sample = _sampler(llm, cfg["train_temperature"], 256, lora_request)
        episodes = run_episodes(tokenizer, sample, _pairs(batch, cfg["max_turns"]), cfg["max_turns"])
        rewards = [float(ep.info[cfg["reward_key"]]) for ep in episodes]
        groups = [rewards[g * cfg["group"] : (g + 1) * cfg["group"]] for g in range(cfg["tasks_per_step"])]
        advantages = [r - sum(group) / len(group) for group in groups for r in group]
        t_gen = time.time() - t0

        model.train()
        total_tokens = sum(sum(ep.mask) for ep in episodes)
        optimizer.zero_grad()
        for ep, adv in zip(episodes, advantages):
            if adv == 0:
                continue  # a flat group, or an exactly average episode: no gradient, so skip the compute
            ids = torch.tensor([ep.ids], device="cuda")
            mask = torch.tensor([ep.mask[1:]], device="cuda", dtype=torch.float32)
            logits = model(input_ids=ids).logits[:, :-1].float()
            logp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            loss = -(adv * logp * mask).sum() / total_tokens
            loss.backward()
            del logits, logp
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(grad_norm):
            optimizer.step()
        lora_request = adapter(step + 1)
        entry = _summary([{"scores": ep.info, "tokens": sum(ep.mask)} for ep in episodes], step + 1, "train")
        entry.update({"grad_norm": float(grad_norm), "gen_s": round(t_gen, 1), "step_s": round(time.time() - t0, 1)})
        entry["flat_groups"] = sum(len(set(group)) == 1 for group in groups)
        log.append(entry)
        print(json.dumps(entry), flush=True)
        if step % 5 == 4:
            samples = "".join(json.dumps({"messages": ep.messages, "scores": ep.info}) + "\n" for ep in episodes)
            (run_dir / f"train_{step + 1}.jsonl").write_text(samples)
        (run_dir / "log.json").write_text(json.dumps({"config": cfg, "log": log}))
        torch.save(
            {"step": step + 1, "optimizer": optimizer.state_dict(), "rng": rng.getstate()}, run_dir / "optimizer.pt"
        )
        volume.commit()
    return {"config": cfg, "log": log}


def _summary(rows: list[dict], step: int, split: str) -> dict:
    n = len(rows)

    def mean(key: str) -> float:
        return round(sum(float(r["scores"][key]) for r in rows) / n, 4)

    return {
        "split": split,
        "step": step,
        "n": n,
        "reward": mean("reward"),
        "hidden": mean("hidden"),
        "collateral": mean("collateral"),
        "turns": mean("turns"),
        "tokens": round(sum(r["tokens"] for r in rows) / n, 1),
        "ended_done": round(sum(r["scores"].get("end") == "done" for r in rows) / n, 4),
    }


@app.local_entrypoint()
def train_main(arm: str = "visible", model: str = "Qwen/Qwen2.5-3B-Instruct", steps: int = 60, seed: int = 0):
    data = HERE / "data" / "tasks"
    train_items = [json.loads(x) for x in (data / "train.jsonl").read_text().splitlines()]
    eval_items = [json.loads(x) for x in (data / "heldout.jsonl").read_text().splitlines()]
    name = f"{arm}-{model.split('/')[-1]}-s{seed}"
    reward_key = {"visible": "reward", "full": "hidden"}[arm]
    config = {"name": name, "model": model, "reward_key": reward_key, "steps": steps, "seed": seed}
    t0 = time.time()
    result = train.remote(config, train_items, eval_items)
    (HERE / "results" / f"grpo_{name}.json").write_text(json.dumps(result, indent=1))
    _ledger(f"train {name}", time.time() - t0)


@app.local_entrypoint()
def spawn_main(arm: str = "visible", model: str = "Qwen/Qwen2.5-3B-Instruct", steps: int = 60, seed: int = 0):
    """Like train_main, but survives the local client: needs `modal deploy` first; writes the call id to results/."""
    data = HERE / "data" / "tasks"
    train_items = [json.loads(x) for x in (data / "train.jsonl").read_text().splitlines()]
    eval_items = [json.loads(x) for x in (data / "heldout.jsonl").read_text().splitlines()]
    name = f"{arm}-{model.split('/')[-1]}-s{seed}"
    config = {"name": name, "model": model, "reward_key": {"visible": "reward", "full": "hidden"}[arm], "steps": steps}
    config["seed"] = seed
    deployed = modal.Function.from_name(APP, "train")
    call = deployed.spawn(config, train_items, eval_items)
    (HERE / "results" / f"grpo_{name}.call").write_text(call.object_id)
    print(name, call.object_id)
