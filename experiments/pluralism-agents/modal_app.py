"""Episode 4 of the pluralism series on Modal: tau-bench retail -> airline, single-judge vs jury GRPO on Qwen3-8B.

Layout: one shared server (SimServer, an H100 running Qwen3-30B-A3B-Instruct-2507-FP8 under vLLM) plays every user
and every training judge; each policy job (baseline, pilot, each training run) has its own GPU with the Qwen3-8B
policy engine, the LoRA trainer and the tau-bench environment.

COSTS MONEY. Every entrypoint goes through the spend ledger (results/spend.jsonl) and refuses to start if the ledger
plus its worst case would pass the Modal cap ($220); the server's wall time is charged too. From the repo root:

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/pluralism-agents/modal_app.py::smoke
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/pluralism-agents/modal_app.py::baseline --seed 0
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/pluralism-agents/modal_app.py::judge_dev --tag v1
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/pluralism-agents/modal_app.py::train --config configs/pilot.json
"""

from __future__ import annotations

import json
import os
import random
import secrets
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
VOL = "/vol"
MODAL_CAP = 450.0  # raised from $220 by amendments A6 ($300) and A7 ($450), 2026-11-19
RATE = {
    "H200": 4.54,
    "H100": 3.95,
    "A100-80GB": 2.50,
    "A100-80GBx2": 5.00,
    "B200": 6.25,
}  # $/hour, Modal list price (GPU only)
POLICY_GPU = ["H100", "H200", "A100-80GB"]  # first free; each run is charged at its own GPU's rate
SERVER_GPU = ["H200", "A100-80GB:2"]  # bf16 30B-A3B: one H200, or two A100s each holding a full replica
OVERHEAD = 1.10  # CPU, memory and start-up on top of the GPU price
TAU_COMMIT = "c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4"  # 2026-11-10
SIM_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def _sim_key():
    """One key for every local entrypoint: concurrent `modal run`s share the server's web URL, so they must share
    its key. Kept outside the repo; in a container the secret below sets SIM_KEY."""
    if os.environ.get("SIM_KEY"):
        return os.environ["SIM_KEY"]
    p = Path.home() / ".config" / "pluralism-agents" / "simkey"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_hex(16))
    return p.read_text().strip()


SIM_KEY = _sim_key()
os.environ["SIM_KEY"] = SIM_KEY

app = modal.App("pluralism-agents")
volume = modal.Volume.from_name("pluralism-agents", create_if_missing=True)
sim_secret = modal.Secret.from_dict({"SIM_KEY": SIM_KEY})
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.10.1.1", "transformers==4.55.4", "peft==0.17.1", "hf_transfer==0.1.9", "requests", "numpy", "uv"
    )
    .run_commands(
        'python -c "import urllib.request, tarfile, io; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen('
        f"'https://codeload.github.com/sierra-research/tau2-bench/tar.gz/{TAU_COMMIT}').read())).extractall('/opt')\"",
        f"mv /opt/tau2-bench-{TAU_COMMIT} /opt/tau2-bench",
        "uv venv -p 3.12 /opt/tau && uv pip install --python /opt/tau/bin/python -e /opt/tau2-bench",
    )
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_file(HERE / "tau_server.py", "/root/tau_server.py")
    .add_local_file(HERE / "judges.py", "/root/judges.py")
    .add_local_file(HERE / "agent_loop.py", "/root/agent_loop.py")
    .add_local_file(HERE / "data" / "tau_tasks.json", "/root/data/tau_tasks.json")
    .add_local_file(HERE / "data" / "neutral_prompts.json", "/root/data/neutral_prompts.json")
)

EVAL_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}  # Qwen3's recommended non-thinking settings
TRAIN_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}


# ----------------------------------------------------------------------------- the shared simulator/judge server


@app.cls(image=image, gpu=SERVER_GPU, volumes={VOL: volume}, secrets=[sim_secret], timeout=24 * 3600,
         scaledown_window=300, max_containers=int(os.environ.get("SIM_REPLICAS", "2")))  # fmt: skip
@modal.concurrent(max_inputs=160, target_inputs=64)  # Modal adds a server container when one has >64 requests in flight
class SimServer:
    @modal.web_server(8000, startup_timeout=1800)
    def serve(self):
        import torch

        subprocess.Popen(
            [
                "vllm",
                "serve",
                SIM_MODEL,
                "--port",
                "8000",
                "--served-model-name",
                "sim",
                "--data-parallel-size",
                str(torch.cuda.device_count()),  # replicas, not tensor parallel (PCIe was slow)
                "--gpu-memory-utilization",
                "0.92",
                "--max-model-len",
                "32768",
                "--enable-prefix-caching",
                "--max-num-seqs",
                "256",
                "--api-key",
                os.environ["SIM_KEY"],
                "--disable-log-requests",
            ]
        )


def sim_url_ready():
    """Starts (or reuses) the server and waits until it answers. Returns its URL."""
    import requests

    url = SimServer().serve.get_web_url()
    t0 = time.time()
    while time.time() - t0 < 2400:
        try:
            r = requests.get(url + "/v1/models", headers={"Authorization": f"Bearer {SIM_KEY}"}, timeout=30)
            if r.status_code == 200:
                print(f"sim server ready after {time.time() - t0:.0f}s", flush=True)
                return url
        except Exception:
            pass
        time.sleep(15)
    raise SystemExit("sim server did not come up")


# ----------------------------------------------------------------------------- spend ledger (local side)


def ledger_total():
    p = RESULTS / "spend.jsonl"
    if not p.exists():
        return 0.0
    return sum(json.loads(line)["dollars"] for line in p.read_text().splitlines() if line.strip())


def ledger_check(worst_case, what):
    spent = ledger_total()
    if spent + worst_case > MODAL_CAP:
        raise SystemExit(f"REFUSED {what}: ledger ${spent:.2f} + worst case ${worst_case:.2f} > cap ${MODAL_CAP}")
    print(f"ledger ${spent:.2f}; {what} worst case ${worst_case:.2f}; cap ${MODAL_CAP}")


def ledger_add(what, gpu, seconds):
    dollars = RATE[gpu] * OVERHEAD * seconds / 3600
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "spend.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M"),
                    "what": what,
                    "gpu": gpu,
                    "seconds": round(seconds),
                    "dollars": round(dollars, 2),
                }
            )
            + "\n"
        )
    print(f"spent ${dollars:.2f} on {what}; ledger now ${ledger_total():.2f}")


def charge_server(what, t0, replicas=1):
    # the server keeps running for its 5-minute scale-down window after the last request; charged at the dearer
    # of its two GPU options (two A100-80GB), times the number of server containers it may have scaled to
    ledger_add(f"sim server ({what}, up to {replicas} containers)", "A100-80GBx2", (time.time() - t0 + 300) * replicas)


# ----------------------------------------------------------------------------- remote helpers


def gpu_kind():
    import torch

    n = torch.cuda.get_device_name(0)
    return "H200" if "H200" in n else "H100" if "H100" in n else "A100-80GB" if "A100" in n else "B200"


def load_D():
    return json.loads(Path("/root/data/tau_tasks.json").read_text())


def services(sim_url, policy_util=0.66):
    sys.path.insert(0, "/root")
    from transformers import AutoTokenizer
    from vllm import LLM

    import agent_loop as AL

    print("GPU:", gpu_kind(), flush=True)
    chat = AL.Chat(url=sim_url, key=os.environ["SIM_KEY"], workers=128)
    tau = AL.TauEnv()
    tok = AutoTokenizer.from_pretrained(AL.POLICY_MODEL)
    llm = LLM(
        model=AL.POLICY_MODEL, gpu_memory_utilization=policy_util, max_model_len=26000, dtype="bfloat16",
        enable_prefix_caching=True, seed=0, max_num_seqs=128, enable_lora=True, max_lora_rank=32, max_loras=2,
        enable_sleep_mode=True,
    )  # fmt: skip
    return AL, chat, tau, llm, tok


def eval_items(D, domain, split, k):
    tasks = [t for t in D[domain]["tasks"] if split is None or t["split"] == split]
    return [(domain, t, s) for t in tasks for s in range(k)]


def run_eval(AL, llm, tok, tau, chat, D, lora, seed, tag, k=4, judge=True):
    rows = []
    for domain, split in [("airline", None), ("retail", "test")]:
        items = eval_items(D, domain, split, k)
        convs = AL.rollout(llm, tok, tau, chat, D, items, lora=lora, seed=seed, tag=f"{tag}-{domain}", **EVAL_SAMPLING)
        js = AL.judge_scores(chat, D, convs) if judge else [{}] * len(convs)
        rows += [AL.conv_record(c, {"judges": j, "eval_tag": tag}) for c, j in zip(convs, js)]
    return rows


def neutral_samples(llm, tok, seed=0):
    from vllm import SamplingParams

    prompts = json.loads(Path("/root/data/neutral_prompts.json").read_text())
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256, seed=seed, repetition_penalty=1.0)
    outs = llm.generate(texts, sp, use_tqdm=False)
    return [{"id": p["id"], "prompt_ids": list(o.prompt_token_ids), "reply_ids": list(o.outputs[0].token_ids)}
            for p, o in zip(prompts, outs)]  # fmt: skip


# ----------------------------------------------------------------------------- smoke test


@app.function(image=image, gpu=POLICY_GPU, timeout=60 * 60, volumes={VOL: volume}, secrets=[sim_secret])
def smoke_remote(sim_url: str, n_tasks: int, train_steps: int):
    AL, chat, tau, llm, tok = services(sim_url)
    D = load_D()
    t0 = time.time()
    items = [("retail", t, s) for t in D["retail"]["tasks"][:n_tasks] for s in range(2)]
    items += [("airline", t, 0) for t in D["airline"]["tasks"][:n_tasks]]
    convs = AL.rollout(llm, tok, tau, chat, D, items, tag="smoke", **EVAL_SAMPLING)
    js = AL.judge_scores(chat, D, convs)
    rows = [AL.conv_record(c, {"judges": j}) for c, j in zip(convs, js)]
    out = {"rollout_s": time.time() - t0, "rows": rows}
    if train_steps:
        import torch

        model, params, opt = AL.build_trainer(AL.POLICY_MODEL, 32, 2e-5, 0)
        print("trainer memory GB", torch.cuda.memory_allocated() / 1e9, flush=True)
        seqs = [(c.ids, c.mask) for c in convs]
        n_tok = sum(sum(c.mask) for c in convs)
        t1 = time.time()
        advs = [1.0 if i % 2 else -1.0 for i in range(len(seqs))]
        loss, gnorm, ok = AL.grpo_step(model, params, opt, seqs, advs, n_tok)
        out["train_s"] = time.time() - t1
        out["train"] = {"loss": loss, "gnorm": gnorm, "ok": ok, "n_seq": len(seqs),
                        "max_len": max(len(s[0]) for s in seqs), "peak_GB": torch.cuda.max_memory_allocated() / 1e9}  # fmt: skip
        neutral = neutral_samples(llm, tok)
        t2 = time.time()
        out["kl_after_1_step"] = AL.drift_kl(model, [(n["prompt_ids"], n["reply_ids"]) for n in neutral[:16]])
        out["kl_s"] = time.time() - t2
    out["elapsed"] = time.time() - t0
    out["gpu"] = gpu_kind()
    print({k: v for k, v in out.items() if k != "rows"}, flush=True)
    return out


@app.local_entrypoint()
def smoke(n_tasks: int = 4, train_steps: int = 1):
    ledger_check((RATE["H200"] + RATE["A100-80GBx2"]) * OVERHEAD * 1.0, "smoke")
    t0 = time.time()
    gpu = "H200"  # charged at the dearest policy GPU unless the container reports its own
    try:
        url = sim_url_ready()
        t1 = time.time()
        out = smoke_remote.remote(url, n_tasks, train_steps)
        gpu = out["gpu"]
    finally:
        ledger_add("smoke", gpu, time.time() - (t1 if "t1" in dir() else t0))
        charge_server("smoke", t0)
    d = RESULTS / "smoke"
    d.mkdir(parents=True, exist_ok=True)
    (d / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in out["rows"]))
    print({k: v for k, v in out.items() if k != "rows"})
    for r in out["rows"]:
        print(r["domain"], r["task_id"], r["termination"], r["reward"], r["n_agent_turns"], r["plan"],
              {a: v["reward"] for a, v in r["judges"].items()})  # fmt: skip


# ----------------------------------------------------------------------------- baseline (untrained model)


@app.function(image=image, gpu=POLICY_GPU, timeout=4 * 60 * 60, volumes={VOL: volume}, secrets=[sim_secret])
def baseline_remote(sim_url: str, seed: int, dev: bool):
    AL, chat, tau, llm, tok = services(sim_url)
    D = load_D()
    t0 = time.time()
    out = {"eval": run_eval(AL, llm, tok, tau, chat, D, None, seed, f"base-s{seed}")}
    if dev:
        # judge and simulator development set: retail TRAIN tasks x 2, at the training temperature
        items = eval_items(D, "retail", "train", 2)
        convs = AL.rollout(llm, tok, tau, chat, D, items, seed=seed, tag="dev", **TRAIN_SAMPLING)
        js = AL.judge_scores(chat, D, convs)
        out["dev"] = [AL.conv_record(c, {"judges": j}) for c, j in zip(convs, js)]
        out["neutral"] = neutral_samples(llm, tok, seed=0)
    out["elapsed"] = time.time() - t0
    out["gpu"] = gpu_kind()
    d = Path(VOL) / "baseline" / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "out.json").write_text(json.dumps(out))
    volume.commit()
    return out


@app.local_entrypoint()
def baseline(seeds: str = "0,1"):
    seeds_ = [int(s) for s in seeds.split(",")]
    ledger_check((RATE["H200"] * len(seeds_) + RATE["A100-80GBx2"]) * OVERHEAD * 3.0, f"baseline {seeds}")
    t0 = time.time()
    url = sim_url_ready()
    t1 = time.time()
    try:
        args = [(url, s, s == 0) for s in seeds_]
        for s, out in zip(seeds_, baseline_remote.starmap(args, return_exceptions=True)):
            if isinstance(out, Exception):
                print(f"baseline s{s} FAILED:", repr(out)[:2000])
                ledger_add(f"baseline s{s} FAILED (wall time)", "H200", time.time() - t1)
                continue
            ledger_add(f"baseline s{s}", out["gpu"], out["elapsed"] + 300)
            d = RESULTS / "baseline" / f"s{s}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in out["eval"]))
            if "dev" in out:
                (d / "dev.jsonl").write_text("\n".join(json.dumps(r) for r in out["dev"]))
                (RESULTS / "neutral_samples.json").write_text(json.dumps(out["neutral"]))
            for dom in ["airline", "retail"]:
                rs = [r["reward"] for r in out["eval"] if r["domain"] == dom]
                print(f"s{s}", dom, f"pass^1 {sum(rs) / len(rs):.3f} over {len(rs)}")
            print(f"s{s} elapsed {out['elapsed'] / 60:.1f} min")
    finally:
        charge_server("baseline", t0)


# ----------------------------------------------------------------------------- judge development (autoresearch)


@app.local_entrypoint()
def judge_dev(tag: str, split: str = "dev"):
    """Re-score saved conversations with the current judges.py from this machine (the server does the work).
    split=dev (retail train-task rollouts) for tuning; split=heldout (the baseline's retail test-task rollouts)
    exactly once, for the chosen version."""
    sys.path.insert(0, str(HERE))
    import agent_loop as AL
    from analyse import jl

    if split == "dev":
        rows = jl(RESULTS / "baseline/s0/dev.jsonl")
    else:
        rows = jl(RESULTS / "baseline/s0/eval.jsonl")
        rows = [r for r in rows if r["domain"] == "retail"]
    ledger_check(RATE["A100-80GBx2"] * OVERHEAD * 1.0, f"judge_dev {tag}")
    t0 = time.time()
    try:
        url = sim_url_ready()
        D = json.loads((HERE / "data" / "tau_tasks.json").read_text())
        chat = AL.Chat(url=url, key=SIM_KEY, workers=64)
        js = AL.judge_scores(chat, D, rows)
    finally:
        charge_server(f"judge_dev {tag} {split}", t0)
    for r, j in zip(rows, js):
        r["judges"] = j
    d = RESULTS / "autoresearch"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(f"scored {len(rows)} in {time.time() - t0:.0f}s -> {d / f'{tag}_{split}.jsonl'}")


# ----------------------------------------------------------------------------- GRPO training


def train_run(cfg):
    import torch
    from vllm.lora.request import LoRARequest

    AL, chat, tau, llm, tok = services(cfg["sim_url"])
    D = load_D()
    name, arm, seed = cfg["name"], cfg["arm"], cfg["seed"]
    run_dir = Path(VOL) / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if k != "neutral"}, indent=1))
    neutral = [(n["prompt_ids"], n["reply_ids"]) for n in cfg["neutral"]]
    rng = random.Random(seed)
    train_tasks = [t for t in D["retail"]["tasks"] if t["split"] == "train"]
    model, params, opt = AL.build_trainer(AL.POLICY_MODEL, cfg["lora_rank"], cfg["lr"], seed)
    t_start = time.time()
    budget_s = cfg["train_dollars"] / (RATE[gpu_kind()] * OVERHEAD) * 3600

    def save(tag):
        d = run_dir / f"adapter_{tag}"
        model.save_pretrained(str(d))
        return d

    lora = LoRARequest(f"{name}-0", 1, str(save("step0")))
    evals = []

    def eval_step(s):
        lr_ = LoRARequest(f"{name}-eval{s}", 10_000 + s, str(run_dir / f"adapter_step{s}"))
        rows = run_eval(AL, llm, tok, tau, chat, D, lr_, seed + 7, f"{name}-step{s}", k=cfg.get("eval_k", 4))
        for r in rows:
            r["step"] = s
        AL.write_jsonl(run_dir / f"eval_step{s}.jsonl", rows)
        volume.commit()
        return rows

    curve, train_rows = [], []
    order = []
    stopped_early = False
    for step in range(1, cfg["steps"] + 1):
        t0 = time.time()
        if len(order) < cfg["tasks_per_step"]:
            more = train_tasks[:]
            rng.shuffle(more)
            order += more
        batch, order = order[: cfg["tasks_per_step"]], order[cfg["tasks_per_step"] :]
        items = [("retail", t, s) for t in batch for s in range(cfg["group"])]
        convs = AL.rollout(
            llm, tok, tau, chat, D, items, lora=lora, seed=seed * 100_000 + step, tag=f"{name}-s{step}",
            loop_limit=cfg.get("loop_limit"), **TRAIN_SAMPLING,
        )  # fmt: skip
        t_roll = time.time() - t0
        js = AL.judge_scores(chat, D, convs, arms=(arm,))
        rewards = [j[arm]["reward"] for j in js]
        t_judge = time.time() - t0 - t_roll
        seqs, advs = [], []
        n_tok = sum(sum(c.mask) for c in convs)
        for g in range(len(batch)):
            idx = list(range(g * cfg["group"], (g + 1) * cfg["group"]))
            adv = AL.group_advantages([rewards[i] for i in idx])
            for i, a in zip(idx, adv):
                if a != 0.0:
                    seqs.append((convs[i].ids, convs[i].mask))
                    advs.append(a)
        loss, gnorm, ok = (0.0, 0.0, True)
        llm.sleep(level=1)  # frees the policy engine's GPU memory (weights to CPU, KV cache dropped) for the update
        if seqs:
            loss, gnorm, ok = AL.grpo_step(model, params, opt, seqs, advs, n_tok)
        t_train = time.time() - t0 - t_roll - t_judge
        keep = step % cfg["save_every"] == 0 or step == cfg["steps"]
        adapter = save(f"step{step}" if keep else "cur")
        lora = LoRARequest(f"{name}-{step}", step + 1, str(adapter))
        rec = {
            "step": step,
            "judge_reward": sum(rewards) / len(rewards),
            "gt_success": sum(c.reward for c in convs) / len(convs),
            "finished": sum(c.termination == "user_stop" for c in convs) / len(convs),
            "n_update_seqs": len(seqs),
            "loss": loss,
            "grad_norm": gnorm,
            "update_ok": ok,
            "mean_tokens": sum(len(c.ids) for c in convs) / len(convs),
            "mean_agent_turns": sum(c.n_agent_turns for c in convs) / len(convs),
            "t_rollout": round(t_roll),
            "t_judge": round(t_judge),
            "t_train": round(t_train),
        }
        if keep:
            t1 = time.time()
            rec["drift_kl"] = AL.drift_kl(model, neutral)
            rec["t_kl"] = round(time.time() - t1)
        torch.cuda.empty_cache()
        rec["trainer_gb"] = round(torch.cuda.memory_reserved() / 1e9, 1)
        llm.wake_up()
        curve.append(rec)
        train_rows += [
            {k: v for k, v in AL.conv_record(c, {"judge": j[arm], "step": step}).items() if k != "transcript"}
            for c, j in zip(convs, js)
        ]
        print(name, json.dumps(rec), flush=True)
        (run_dir / "curve.jsonl").write_text("\n".join(json.dumps(r) for r in curve))
        AL.write_jsonl(run_dir / "train_rows.jsonl", train_rows)
        (run_dir / "timing.json").write_text(json.dumps({"gpu": gpu_kind(), "elapsed": time.time() - t_start,
                                                         "stopped_early": stopped_early}))  # fmt: skip
        torch.cuda.empty_cache()
        volume.commit()
        if step in cfg["eval_steps"]:  # evaluate as soon as the checkpoint exists, so results stream in
            evals += eval_step(step)
        elapsed = time.time() - t_start
        if elapsed / step * (step + 1) > budget_s and step < cfg["steps"]:
            print(f"{name}: stopping at step {step}: the next step would pass the training budget", flush=True)
            stopped_early = True
            if not keep:
                save(f"step{step}")
                llm.sleep(level=1)
                rec["drift_kl"] = AL.drift_kl(model, neutral)
                torch.cuda.empty_cache()
                llm.wake_up()
            break
    last = curve[-1]["step"]
    if last not in {r["step"] for r in evals}:
        evals += eval_step(last)
    (run_dir / "timing.json").write_text(json.dumps({"gpu": gpu_kind(), "elapsed": time.time() - t_start,
                                                     "stopped_early": stopped_early, "done": True}))  # fmt: skip
    volume.commit()
    return {"name": name, "curve": curve, "train_rows": train_rows, "eval": evals, "stopped_early": stopped_early,
            "elapsed": time.time() - t_start, "gpu": gpu_kind()}  # fmt: skip


@app.function(
    image=image, gpu=POLICY_GPU, timeout=16 * 60 * 60, volumes={VOL: volume}, secrets=[sim_secret], memory=98304
)
def train_remote(cfg: dict):
    return train_run(cfg)


@app.local_entrypoint()
def train(config: str, only: str = ""):
    spec = json.loads((HERE / config).read_text())
    runs = []
    for r in spec["runs"]:
        if only and r["name"] not in only.split(","):
            continue
        cfg = dict(spec["common"])
        cfg.update(r)
        cfg["neutral"] = json.loads((RESULTS / "neutral_samples.json").read_text())
        runs.append(cfg)
    worst = sum(c["max_dollars"] for c in runs) + spec.get("server_dollars", 40)
    ledger_check(worst, f"train {[c['name'] for c in runs]}")
    t0 = time.time()
    try:
        url = sim_url_ready()
        t1 = time.time()
        for c in runs:
            c["sim_url"] = url
        for res in train_remote.map(runs, order_outputs=False, return_exceptions=True):
            if isinstance(res, Exception):
                print("RUN FAILED:", repr(res)[:2000])
                ledger_add("train FAILED run (wall time charged)", "H200", time.time() - t1)
                continue
            d = RESULTS / "runs" / res["name"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "curve.jsonl").write_text("\n".join(json.dumps(r) for r in res["curve"]))
            (d / "train_rows.jsonl").write_text("\n".join(json.dumps(r) for r in res["train_rows"]))
            (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in res["eval"]))
            ledger_add(f"train {res['name']}", res["gpu"], res["elapsed"] + 600)  # + about 10 min of start-up
            print(
                res["name"], "done", f"{res['elapsed'] / 60:.0f} min", "stopped_early" if res["stopped_early"] else ""
            )
    finally:
        charge_server(f"train {config}", t0, replicas=int(os.environ.get("SIM_REPLICAS", "2")))
        print(f"wall time {(time.time() - t0) / 60:.0f} min")


# ----------------------------------------------------------------------------- detached launch (deployed app)
#
#   MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/pluralism-agents/modal_app.py
#   MODAL_PROFILE=albertlieyingadrian .venv/bin/python experiments/pluralism-agents/modal_app.py launch configs/full.json
#   MODAL_PROFILE=albertlieyingadrian .venv/bin/python experiments/pluralism-agents/modal_app.py collect configs/full.json
#
# Spawned calls run on Modal whether or not this machine stays connected (a local gRPC timeout killed attempt 5).


def _runs_from(config):
    spec = json.loads((HERE / config).read_text())
    runs = []
    for r in spec["runs"]:
        cfg = dict(spec["common"])
        cfg.update(r)
        runs.append(cfg)
    return spec, runs


def launch(config):
    spec, runs = _runs_from(config)
    worst = sum(c["max_dollars"] for c in runs) + spec.get("server_dollars", 40)
    ledger_check(worst, f"launch {[c['name'] for c in runs]}")
    import requests

    url = modal.Cls.from_name("pluralism-agents", "SimServer")().serve.get_web_url()
    t0 = time.time()
    while True:  # wake the server before the runs need it
        try:
            if requests.get(url + "/v1/models", headers={"Authorization": f"Bearer {SIM_KEY}"}, timeout=60).ok:
                break
        except Exception:
            pass
        if time.time() - t0 > 2400:
            raise SystemExit("server did not come up")
        time.sleep(15)
    print(f"server up after {time.time() - t0:.0f}s")
    neutral = json.loads((RESULTS / "neutral_samples.json").read_text())
    fn = modal.Function.from_name("pluralism-agents", "train_remote")
    calls = {}
    for c in runs:
        c["sim_url"], c["neutral"] = url, neutral
        calls[c["name"]] = fn.spawn(c).object_id
    (RESULTS / "launch.json").write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "url": url,
                                                     "calls": calls}, indent=1))  # fmt: skip
    print("spawned", calls, "server", url)


def collect(config):
    """Copies every run's files from the volume into results/runs/<name>/ (works while runs are still going)."""
    import subprocess as sp

    _, runs = _runs_from(config)
    for c in runs:
        d = RESULTS / "runs" / c["name"]
        d.mkdir(parents=True, exist_ok=True)
        ls = sp.run([sys.executable, "-m", "modal", "volume", "ls", "pluralism-agents", f"runs/{c['name']}"],
                    capture_output=True, text=True).stdout.split()  # fmt: skip
        for f in ls:
            base = f.split("/")[-1]
            if base.endswith((".jsonl", ".json")):
                sp.run([sys.executable, "-m", "modal", "volume", "get", "--force", "pluralism-agents", f, str(d / base)],
                       capture_output=True)  # fmt: skip
        evs = sorted(d.glob("eval_step*.jsonl"), key=lambda p: int(p.stem.replace("eval_step", "")))
        rows = [x for p in evs for x in p.read_text().splitlines() if x.strip()]
        if rows:
            (d / "eval.jsonl").write_text("\n".join(rows))
        cur = (
            [json.loads(x) for x in (d / "curve.jsonl").read_text().splitlines()]
            if (d / "curve.jsonl").exists()
            else []
        )
        tm = json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else {}
        print(c["name"], "steps", cur[-1]["step"] if cur else 0, "evals", [p.stem for p in evs], tm)


if __name__ == "__main__":
    {"launch": launch, "collect": collect}[sys.argv[1]](sys.argv[2])
