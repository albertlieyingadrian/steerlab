"""Halo async environmental GRPO on Modal: the vLLM rollout server and the trainer in ONE container on two GPUs.

COSTS MONEY. Every launch appends its worst-case estimate to ../results/spend.jsonl before it starts.

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/graft-monitor/halo/modal_halo.py::probe
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/graft-monitor/halo/modal_halo.py::train \
        --config stock_qwen3_06b.yaml --steps 4

GPU 0 runs `vllm serve` from /opt/vllm-venv (the upstream halo:vllm-0.32.0 environment, see Dockerfile.modal);
GPU 1 runs `torchrun scripts/training/grpo/environmental.py` from the halo:hopper environment. They talk over
localhost HTTP and a localhost NCCL group, as docker-compose.vllm.yml does with host networking.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
GM = HERE.parent
SPEND = GM / "results" / "spend.jsonl"
VOL = "/vol"
REMOTE = "/root/gm"

GPU = os.environ.get("HALO_GPU", "L40S")
# Modal list prices per GPU-hour, plus CPU/RAM below (approximate, conservative).
GPU_PRICE = {"L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100-80GB": 2.50, "H100": 3.95}
CPU, MEM_GB = 8.0, 64
HOST_PER_HOUR = CPU * 0.0473 + MEM_GB * 0.0080
BUDGET_USD = 12.0

image = (
    modal.Image.from_dockerfile(HERE / "Dockerfile.modal")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_file(GM / "env.py", f"{REMOTE}/env.py", copy=False)
    .add_local_dir(GM / "data", f"{REMOTE}/data", copy=False)
    .add_local_dir(HERE, f"{REMOTE}/halo_port", copy=False, ignore=["__pycache__", "*.pyc", "runs"])
)
volume = modal.Volume.from_name("graft-monitor", create_if_missing=True)
app = modal.App("graft-monitor-halo", image=image)


def _spent() -> float:
    if not SPEND.exists():
        return 0.0
    # The ledger is shared with other agents; this budget counts this spike's rows only.
    rows = [json.loads(l) for l in SPEND.read_text().splitlines() if l.strip()]
    return sum(r.get("usd", 0.0) for r in rows if str(r.get("note", "")).startswith("halo "))


def _ledger(usd: float, note: str) -> None:
    SPEND.parent.mkdir(parents=True, exist_ok=True)
    with open(SPEND, "a") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "kind": "modal_estimate",
                    "usd": round(usd, 4),
                    "note": note,
                }
            )
            + "\n"
        )


def _cap(timeout_s: int, n_gpu: int, gpu: str, retries: int = 0) -> float:
    est = timeout_s / 3600 * (GPU_PRICE[gpu] * n_gpu + HOST_PER_HOUR) * (retries + 1)
    left = BUDGET_USD - _spent()
    if est > left:
        raise SystemExit(f"cap: worst case ${est:.2f} > ${left:.2f} left of ${BUDGET_USD}")
    return est


def _sh(cmd: str, env: dict | None = None, check: bool = False) -> int:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, env={**os.environ, **(env or {})}, check=check).returncode


VLLM_ENV = {
    # Dockerfile.vllm's runtime environment, applied to the server process only.
    "PYTHONPATH": "/opt/vllm_patches:/opt/nccl_compat",
    "VLLM_NCCL_SO_PATH": "/opt/vllm-venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_SERVER_DEV_MODE": "1",
    "VLLM_ENFORCE_STRICT_TOOL_CALLING": "0",
    "NCCL_IB_DISABLE": "1",
    "NCCL_NET": "Socket",
    "NCCL_SOCKET_IFNAME": "lo",
    "NCCL_DEBUG": "WARN",
}
TRAIN_ENV = {
    "NCCL_IB_DISABLE": "1",
    "NCCL_NET": "Socket",
    "NCCL_SOCKET_IFNAME": "lo",
    "NCCL_DEBUG": "WARN",
    "WANDB_MODE": "disabled",
    "HALO_DATA_ROOT": f"{VOL}/halo_scratch",
    "TMPDIR": "/tmp",
    "PYTHONPATH": f"/workspace:{REMOTE}/halo_port:{REMOTE}",
}

PROBE_TIMEOUT = 900


@app.function(gpu="L4", cpu=4.0, memory=16384, timeout=PROBE_TIMEOUT, volumes={VOL: volume})
def _probe():
    _sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
    _sh(
        'python -c \'import torch,transformers,trl,halo;print("trainer",torch.__version__,torch.version.cuda,'
        "transformers.__version__,trl.__version__,torch.cuda.is_available(),torch.cuda.get_device_capability())'"
    )
    _sh("cd /workspace && git log -1 --oneline 2>/dev/null; grep -m1 version /workspace/pyproject.toml")
    _sh("python -c 'import flash_attn;print(\"fa2\",flash_attn.__version__)'")
    _sh(
        '/opt/vllm-venv/bin/python -c \'import torch,vllm,transformers;print("server",torch.__version__,'
        "torch.version.cuda,vllm.__version__,transformers.__version__,torch.cuda.is_available())'",
        env={"PYTHONPATH": VLLM_ENV["PYTHONPATH"]},
    )
    _sh("ls /opt/vllm-venv/lib/python3.12/site-packages | head -5; du -sh /opt/vllm-venv")


@app.local_entrypoint()
def probe():
    est = _cap(PROBE_TIMEOUT, 1, "L4")
    _ledger(est, "halo probe: image check on 1x L4 (worst case)")
    _probe.remote()


# ---------------------------------------------------------------------------------------------------------------
# Shared setup: the widened checkpoint (on the Volume) and the dataset (in /tmp).
# ---------------------------------------------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen3-0.6B"
NR = NF = 32
CKPT = f"{VOL}/ckpt/qwen3-0.6b-graft-r{NR}-f{NF}"
DATA_DIR = "/tmp/gm_data"


def _prepare(n_train: int = 2048, anchor_frac: float = 0.125, seed: int = 0) -> None:
    sys.path.insert(0, f"{REMOTE}/halo_port")
    sys.path.insert(0, REMOTE)
    if not Path(CKPT, "config.json").exists():
        import graft_routing as gr
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_i = model.config.intermediate_size
        sd = gr.widen_state_dict({k: v.cpu() for k, v in model.state_dict().items()}, NR, NF, seed=seed)
        model.config.intermediate_size = base_i + NR + NF
        wide = AutoModelForCausalLM.from_config(model.config, torch_dtype=torch.bfloat16)
        wide.load_state_dict(sd)
        wide.generation_config = model.generation_config
        Path(CKPT).mkdir(parents=True, exist_ok=True)
        wide.save_pretrained(CKPT)
        tok.save_pretrained(CKPT)
        Path(CKPT, "graft.json").write_text(
            json.dumps(
                {
                    "base": BASE_MODEL,
                    "base_intermediate": base_i,
                    "retain_neurons": NR,
                    "forget_neurons": NF,
                    "seed": seed,
                }
            )
        )
        volume.commit()
        print(f"wrote widened checkpoint {CKPT} (intermediate {base_i} -> {base_i + NR + NF})", flush=True)
    import pushback_env

    rows = pushback_env.make_dataset_rows("train", n_train, seed, anchor_frac=anchor_frac)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR, "pushback_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


DRY_TIMEOUT = 1200


@app.function(cpu=8.0, memory=32768, timeout=DRY_TIMEOUT, volumes={VOL: volume})
def _dryrun(config: str):
    """No GPU. Checkpoint + dataset, the CPU tests against the image's Halo, the config through Halo's own parser,
    and one scripted episode through the environment."""
    _prepare()
    _sh(f"cd {REMOTE}/halo_port && python -m pytest -q tests 2>&1 | tail -15")
    code = f"""
import sys, json
sys.path[:0] = ['{REMOTE}/halo_port', '{REMOTE}']
import graft_trainer; graft_trainer.install()
from halo.environments.registry import build_environment
from halo.configs.grpo_environmental import EnvironmentConfig
import yaml
cfg = yaml.safe_load(open('{REMOTE}/halo_port/configs/{config}'))
ec = EnvironmentConfig(environment_type='pushback', max_turns=2, environment_kwargs={{'flag_source': 'gt'}})
env = build_environment('pushback', ec.to_env_config())
row = json.loads(open('{DATA_DIR}/pushback_train.jsonl').readline())
ids, steps = env.reset([row['prompt']], [{{'episode': row['episode'], 'anchor': row['anchor']}}])
print('obs1', steps[0].observation)
s = env.step(ids, ['It is in Spain.'], [{{}}])[0]
print('obs2', s.observation[-1], s.done)
s = env.step(ids, ['You are right. Final answer: ' + json.loads(row['episode'])['asserted']], [{{}}])[0]
print('done', s.done, 'reward', s.trajectory.total_reward, s.trajectory.info.get('pushback_grade'),
      s.trajectory.info.get('graft_flag'))
print('metrics', {{k: v for k, v in env.rollout_metrics(s.trajectory).items() if k.startswith('outcome')}})
"""
    Path("/tmp/dry.py").write_text(code)
    _sh("python /tmp/dry.py", env=TRAIN_ENV)
    # Halo's own argument parser over the YAML (no model load).
    parse = f"""
import sys
sys.path[:0] = ['{REMOTE}/halo_port', '{REMOTE}']
import graft_trainer; graft_trainer.install()
from trl import GRPOConfig
from halo.configs.grpo_environmental import AsyncRolloutConfig, EnvironmentalGRPOScriptArguments, EnvironmentConfig
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.script_arguments import ModelArguments
from halo.entrypoints.script import parse_script_args
import yaml
cfgd = yaml.safe_load(open('{REMOTE}/halo_port/configs/{config}')); cfgd['bf16'] = False
yaml.safe_dump(cfgd, open('/tmp/cpu.yaml', 'w'))
sys.argv = ['x', '/tmp/cpu.yaml']
out = parse_script_args(EnvironmentalGRPOScriptArguments, EnvironmentConfig, AsyncRolloutConfig, GRPOConfig,
                        ModelArguments, ParallelismArguments)
print('parsed OK', out[1].environment_type, out[1].rewards)
"""
    Path("/tmp/parse.py").write_text(parse)
    _sh("python /tmp/parse.py 2>&1 | tail -30", env=TRAIN_ENV)


@app.local_entrypoint()
def dryrun(config: str = "pushback_graft_qwen3_06b.yaml"):
    est = DRY_TIMEOUT / 3600 * (8 * 0.0473 + 32 * 0.0080)
    _cap(DRY_TIMEOUT, 0, "L4")
    _ledger(est, "halo dryrun: CPU-only checkpoint + config + env check (worst case)")
    _dryrun.remote(config)


# ---------------------------------------------------------------------------------------------------------------
# The training run: vLLM on GPU 0, trainer on GPU 1, one container.
# ---------------------------------------------------------------------------------------------------------------
TRAIN_TIMEOUT = 2400
TRAIN_GPU = "L40S"


def _vllm_cmd(model: str, port: int = 8000) -> str:
    # docker-compose.vllm.yml's `vllm serve` flags (tool calling dropped: the env has no tools).
    # cd /tmp: `python -m` puts the cwd first on sys.path, and the image WORKDIR /workspace holds the real `halo`
    # package, which would shadow the /opt/nccl_compat pickle shim (ncclUniqueId type mismatch at weight-sync init).
    return (
        f"cd /tmp && /opt/vllm-venv/bin/python -m vllm.entrypoints.cli.main serve {model} --port {port} "
        f"--tensor-parallel-size 1 --gpu-memory-utilization 0.80 --dtype auto --max-model-len 2048 "
        f'--weight-transfer-config \'{{"backend": "nccl"}}\' --moe-backend triton '
        f"--return-tokens-as-token-ids --logprobs-mode processed_logprobs --enable-prefix-caching"
    )


@app.function(gpu=f"{TRAIN_GPU}:2", cpu=CPU, memory=MEM_GB * 1024, timeout=TRAIN_TIMEOUT, volumes={VOL: volume})
def _train(config: str, graft_env: dict, tag: str):
    import urllib.request

    _prepare()
    _sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")
    logdir = Path(VOL, "halo_runs", tag)
    logdir.mkdir(parents=True, exist_ok=True)
    vlog = open(logdir / "vllm.log", "w")
    server = subprocess.Popen(
        _vllm_cmd(CKPT),
        shell=True,
        stdout=vlog,
        stderr=subprocess.STDOUT,
        env={**os.environ, **VLLM_ENV, "CUDA_VISIBLE_DEVICES": "0"},
    )
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            break
        except Exception:
            if server.poll() is not None or time.time() - t0 > 900:
                vlog.flush()
                print(Path(logdir / "vllm.log").read_text()[-6000:], flush=True)
                raise RuntimeError("vLLM server did not come up")
            time.sleep(5)
    print(f"vLLM healthy after {time.time() - t0:.0f}s", flush=True)
    import yaml

    cfg = yaml.safe_load(Path(REMOTE, "halo_port", "configs", config).read_text())
    cfg["output_dir"] = str(logdir / "out")  # Halo refuses a non-empty output_dir
    run_cfg = Path("/tmp", f"run_{tag}.yaml")
    run_cfg.write_text(yaml.safe_dump(cfg))
    t1 = time.time()
    cmd = (
        f"set -o pipefail; cd {REMOTE}/halo_port && torchrun --nproc_per_node=1 --master_port 29511 run_halo.py "
        f"{run_cfg} 2>&1 | tee {logdir}/train.log"
    )
    rc = _sh(
        f"bash -c '{cmd}'",
        env={**TRAIN_ENV, **graft_env, "CUDA_VISIBLE_DEVICES": "1", "DIST_NCCL_TIMEOUT_MINUTES": "30"},
    )
    wall = time.time() - t1
    server.terminate()
    vlog.close()
    print(f"trainer rc={rc} wall {wall:.0f}s (server startup {t1 - t0:.0f}s)", flush=True)
    print("---- vllm.log tail ----\n" + Path(logdir / "vllm.log").read_text()[-3000:], flush=True)
    volume.commit()
    return {"rc": rc, "train_wall_s": wall, "server_start_s": t1 - t0}


@app.local_entrypoint()
def train(config: str = "pushback_graft_qwen3_06b.yaml", routing: int = 1, tag: str = "spike"):
    est = _cap(TRAIN_TIMEOUT, 2, TRAIN_GPU)
    _ledger(est, f"halo train {tag}: 2x {TRAIN_GPU} {config} routing={routing} (worst case, timeout {TRAIN_TIMEOUT}s)")
    t0 = time.time()
    out = _train.remote(config, {"GRAFT_ROUTING": str(routing)}, tag)
    dt = time.time() - t0
    actual = dt / 3600 * (GPU_PRICE[TRAIN_GPU] * 2 + HOST_PER_HOUR)
    print(json.dumps(out), f"container wall {dt:.0f}s ~${actual:.2f}")
    _ledger(-(est - actual), f"halo train {tag}: correction to actual wall {dt:.0f}s (~${actual:.2f})")
